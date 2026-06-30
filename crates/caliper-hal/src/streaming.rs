//! Live-streaming control runner — the engine half of live scopes.
//!
//! [`ControlLoop::run_record`] collects every tick into a `Vec<Frame>` and only
//! hands it back once the whole rollout is done. A live UI (the Studio scope) or a
//! CLI live mode instead needs each [`Frame`] AS IT IS PRODUCED, plus a way to
//! stop early. [`ControlLoop::run_stream`] drives the loop tick-by-tick exactly as
//! `run_record` does, but pushes each frame to a caller-supplied sink the instant
//! it exists. The sink returns `bool`: `false` is a cooperative cancel that stops
//! the rollout cleanly at the current tick.
//!
//! This is **additive** over [`ControlLoop::step`] / [`ControlLoop::run_record`]:
//! it reuses `step(sp, None)` verbatim, so a frame seen here is bit-for-bit the
//! frame `run_record` would have collected (the streaming path and the batch path
//! are the same computation — proven by the cross-validation tests below).
//!
//! Decimation (`emit_every`) thins WHAT the sink sees, never HOW the loop runs:
//! every tick is still stepped (the backend must integrate each `dt` for the
//! physics to stay deterministic) — only every `N`-th frame is emitted. So a live
//! scope can run a 1 kHz loop while only painting, say, every 20th sample.

use crate::setpoint::Setpoint;
use crate::{ControlLoop, Error, Frame, RobotBackend};

impl<B: RobotBackend> ControlLoop<B> {
    /// Drive the loop for `ticks` ticks, handing each [`Frame`] to `sink` as it is
    /// produced. `sink` returns `false` to cancel cooperatively — the loop stops
    /// at that tick and returns. Returns the number of ticks actually executed
    /// (`== ticks` on a full run, fewer on an early cancel).
    ///
    /// Determinism is identical to [`run_record`](Self::run_record): clock-free
    /// (`t == tick·dt`), no wall-clock. Collecting the streamed frames into a `Vec`
    /// yields the byte-identical sequence `run_record` would return for the same
    /// input.
    pub fn run_stream(
        &mut self,
        sp: &mut dyn Setpoint,
        ticks: usize,
        sink: &mut dyn FnMut(&Frame) -> bool,
    ) -> Result<usize, Error> {
        self.run_stream_every(sp, ticks, 1, sink)
    }

    /// Like [`run_stream`](Self::run_stream), but only emit every `emit_every`-th
    /// frame to `sink` (decimation). The loop still STEPS every tick — decimation
    /// only thins emission, so the integrated dynamics are unchanged and the
    /// emitted frames are a strict subsample of the full `run_record` sequence
    /// (indices `0, emit_every, 2·emit_every, …`).
    ///
    /// The cooperative-cancel `bool` is only observed on emitted frames (the sink
    /// is the only thing that can signal `false`). Returns the number of ticks
    /// actually executed. `emit_every == 0` is rejected.
    pub fn run_stream_every(
        &mut self,
        sp: &mut dyn Setpoint,
        ticks: usize,
        emit_every: usize,
        sink: &mut dyn FnMut(&Frame) -> bool,
    ) -> Result<usize, Error> {
        if emit_every == 0 {
            return Err(Error::Backend(
                "run_stream emit_every must be >= 1, got 0".into(),
            ));
        }
        for i in 0..ticks {
            // Reuse the exact batch-path step; `None` sink means the frame lives
            // only here, so the streamed frame is identical to the recorded one.
            let frame = self.step(sp, None)?;
            if i % emit_every == 0 && !sink(&frame) {
                // Cooperative cancel: the tick HAS executed, so it counts.
                return Ok(i + 1);
            }
        }
        Ok(ticks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PhysicsSimBackend;
    use crate::setpoint::HoldSetpoint;
    use caliper_model::Model;
    use std::path::Path;
    use std::sync::Arc;

    fn model(name: &str) -> Arc<Model> {
        Arc::new(
            Model::from_urdf(Path::new(&format!(
                "{}/../../oracle/fixtures/robots/{}",
                env!("CARGO_MANIFEST_DIR"),
                name
            )))
            .unwrap(),
        )
    }

    /// Build a fresh, fully-deterministic regulation loop (same setup each call) so
    /// two independent rollouts can be compared bit-for-bit.
    fn fresh_loop() -> (ControlLoop<PhysicsSimBackend>, HoldSetpoint) {
        let m = model("dyn_pendulum2.urdf");
        let mut b = PhysicsSimBackend::new(m.clone()).unwrap();
        b.set_state(&[0.5, -0.2], &[0.0, 0.0]).unwrap();
        let loopy = ControlLoop::new(b, m, 1e-3).unwrap();
        let sp = HoldSetpoint::new(vec![0.1, 0.0]);
        (loopy, sp)
    }

    fn bits_eq(a: &Frame, b: &Frame) -> bool {
        if a.tick != b.tick
            || a.t.to_bits() != b.t.to_bits()
            || a.warn != b.warn
            || a.measured.len() != b.measured.len()
            || a.measured_qd.len() != b.measured_qd.len()
            || a.command.len() != b.command.len()
        {
            return false;
        }
        let eq = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        eq(&a.measured, &b.measured)
            && eq(&a.measured_qd, &b.measured_qd)
            && eq(&a.command, &b.command)
    }

    #[test]
    fn stream_matches_record_bitwise() {
        // CROSS-VALIDATION: the streaming path and the batch path are the SAME
        // computation — collecting run_stream's frames reproduces run_record's
        // Vec exactly, field-by-field, bit-for-bit. A divergent streaming impl
        // (skipped step, reordered emit, dropped feed-forward) fails here.
        const N: usize = 500;
        let (mut l_rec, mut sp_rec) = fresh_loop();
        let recorded = l_rec.run_record(&mut sp_rec, N).unwrap();

        let (mut l_str, mut sp_str) = fresh_loop();
        let mut streamed: Vec<Frame> = Vec::new();
        let executed = l_str
            .run_stream(&mut sp_str, N, &mut |f| {
                streamed.push(f.clone());
                true
            })
            .unwrap();

        assert_eq!(executed, N, "full run executes every tick");
        assert_eq!(streamed.len(), recorded.len(), "same frame count");
        for (i, (a, b)) in streamed.iter().zip(recorded.iter()).enumerate() {
            assert!(bits_eq(a, b), "frame {i} differs between stream and record");
        }
        // Frames must arrive in tick order with no gaps.
        for (i, f) in streamed.iter().enumerate() {
            assert_eq!(f.tick, i as u64);
        }
    }

    #[test]
    fn early_stop_returns_right_prefix() {
        // Cancelling after K emitted frames stops the loop at tick K and the
        // collected frames are EXACTLY run_record's first K — same frames, no
        // extra step executed past the cancel.
        const N: usize = 300;
        const K: usize = 37;
        let (mut l_rec, mut sp_rec) = fresh_loop();
        let recorded = l_rec.run_record(&mut sp_rec, N).unwrap();

        let (mut l_str, mut sp_str) = fresh_loop();
        let mut got: Vec<Frame> = Vec::new();
        let executed = l_str
            .run_stream(&mut sp_str, N, &mut |f| {
                got.push(f.clone());
                got.len() < K // keep going until we have K, then cancel
            })
            .unwrap();

        assert_eq!(executed, K, "executed exactly K ticks before cancel");
        assert_eq!(got.len(), K, "collected exactly the K-frame prefix");
        for (i, (a, b)) in got.iter().zip(recorded.iter()).enumerate() {
            assert!(bits_eq(a, b), "prefix frame {i} differs from record");
        }
    }

    #[test]
    fn decimation_emits_right_count_and_subsample() {
        // emit_every = M emits indices 0, M, 2M, … : count = ⌈N/M⌉, and each
        // emitted frame equals the corresponding run_record frame (the loop still
        // steps every tick, so the dynamics are untouched by decimation).
        const N: usize = 200;
        const M: usize = 7;
        let (mut l_rec, mut sp_rec) = fresh_loop();
        let recorded = l_rec.run_record(&mut sp_rec, N).unwrap();

        let (mut l_str, mut sp_str) = fresh_loop();
        let mut got: Vec<Frame> = Vec::new();
        l_str
            .run_stream_every(&mut sp_str, N, M, &mut |f| {
                got.push(f.clone());
                true
            })
            .unwrap();

        let expected = N.div_ceil(M); // ⌈N/M⌉
        assert_eq!(got.len(), expected, "decimated count = ceil(N/M)");
        for (k, f) in got.iter().enumerate() {
            let src = k * M;
            assert_eq!(f.tick, src as u64, "emitted frame {k} is tick {src}");
            assert!(
                bits_eq(f, &recorded[src]),
                "decimated frame {k} != record[{src}]"
            );
        }
    }

    #[test]
    fn emit_every_zero_is_rejected() {
        let (mut l, mut sp) = fresh_loop();
        let r = l.run_stream_every(&mut sp, 10, 0, &mut |_| true);
        assert!(
            matches!(r, Err(Error::Backend(_))),
            "emit_every=0 must error"
        );
    }

    #[test]
    fn zero_ticks_emits_nothing() {
        let (mut l, mut sp) = fresh_loop();
        let mut n = 0usize;
        let executed = l
            .run_stream(&mut sp, 0, &mut |_| {
                n += 1;
                true
            })
            .unwrap();
        assert_eq!(executed, 0);
        assert_eq!(n, 0, "no ticks → no emissions");
    }
}
