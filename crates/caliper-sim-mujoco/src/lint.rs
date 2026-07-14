//! Contact STABILITY LINTER — detects the three classic MuJoCo contact
//! pathologies and says, in plain English, WHICH solver knob to turn:
//!
//! | code | pathology | typical cause |
//! |---|---|---|
//! | C001 | explosion — state diverges or goes non-finite ("spins uncontrollably") | `solref` timeconst below `2·timestep`, under-damped stiff contact, oversized timestep |
//! | C002 | penetration — bodies rest INSIDE each other after settling | soft `solimp` (low `dmin`/`dmax`, wide `width`), timeconst too long |
//! | C003 | jitter — contact forces keep oscillating after settling | under-damped `solref` (dampratio < 1), marginal timeconst ≈ `2·timestep` |
//!
//! Two layers, mirroring `mjcf` vs `sim`:
//! - The CLASSIFIER ([`classify_stability`] over a [`StabilityTrace`]) is pure
//!   math over recorded samples — always compiled, tested without MuJoCo.
//! - The ROLLOUT (`lint_contact_stability`, feature `mujoco`) steps a live
//!   [`MujocoSim`](crate::MujocoSim) through a settle window plus an observe
//!   window, records the trace, and classifies it.
//!
//! Determinism: no randomness anywhere — the rollout inherits MuJoCo's
//! bit-determinism (same binary + same initial state ⇒ same trace ⇒ same
//! findings), and the classifier is a pure function.

use crate::MujocoError;

/// Thresholds and windows for the linter. `Default` is tuned for
/// tabletop-manipulation scales (meter-sized robots, kilogram-sized props);
/// scale the thresholds for very large or very small scenes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LintOptions {
    /// Rollout time (s) given to the scene to settle BEFORE judging anything
    /// — falling props, sagging arms and first impacts all live here.
    pub settle_duration: f64,
    /// Observation window (s) AFTER settling; every metric is computed over
    /// this window only.
    pub observe_duration: f64,
    /// C001: any dof speed (rad/s or m/s) above this after settling is an
    /// explosion — no sane settled tabletop scene moves this fast.
    pub explosion_speed: f64,
    /// C001: energy-proxy growth factor — mean of the observe window's second
    /// half vs its first half. Growth this large while "settled" means the
    /// contact is PUMPING energy in (the numeric-resonance failure mode).
    pub explosion_energy_growth: f64,
    /// C001: absolute energy-proxy floor (`0.5·Σ qd²`) the second half must
    /// also exceed — keeps ratio noise on an almost-at-rest scene (1e-9 vs
    /// 1e-8) from reading as an explosion.
    pub explosion_energy_floor: f64,
    /// C002: contact depth (m) that counts as real penetration (default
    /// 5 mm — visible at tabletop scale).
    pub penetration_depth: f64,
    /// C002: fraction of observed steps that must exceed
    /// [`penetration_depth`](Self::penetration_depth) for it to count as
    /// PERSISTENT (a brief impact spike is fine; resting inside the floor
    /// is not).
    pub penetration_fraction: f64,
    /// C003: jitter fires when the total-normal-force peak-to-peak amplitude
    /// exceeds this fraction of its mean over the observe window.
    pub jitter_force_ratio: f64,
    /// C003: mean total normal force (N) below which jitter is not judged —
    /// no meaningful contact, nothing to oscillate.
    pub min_normal_force: f64,
}

impl Default for LintOptions {
    fn default() -> Self {
        Self {
            settle_duration: 1.0,
            observe_duration: 0.5,
            explosion_speed: 100.0,
            explosion_energy_growth: 25.0,
            explosion_energy_floor: 0.1,
            penetration_depth: 0.005,
            penetration_fraction: 0.8,
            jitter_force_ratio: 0.5,
            min_normal_force: 0.5,
        }
    }
}

impl LintOptions {
    /// Reject non-finite / non-positive knobs loudly before a rollout burns
    /// simulation time on them.
    pub fn validate(&self) -> Result<(), MujocoError> {
        let pos = [
            ("settle_duration", self.settle_duration),
            ("observe_duration", self.observe_duration),
            ("explosion_speed", self.explosion_speed),
            ("explosion_energy_growth", self.explosion_energy_growth),
            ("explosion_energy_floor", self.explosion_energy_floor),
            ("penetration_depth", self.penetration_depth),
            ("jitter_force_ratio", self.jitter_force_ratio),
            ("min_normal_force", self.min_normal_force),
        ];
        for (name, v) in pos {
            if !(v.is_finite() && v > 0.0) {
                return Err(MujocoError::Mjcf(format!(
                    "lint option {name} must be finite and > 0, got {v}"
                )));
            }
        }
        if !(self.penetration_fraction.is_finite()
            && self.penetration_fraction > 0.0
            && self.penetration_fraction <= 1.0)
        {
            return Err(MujocoError::Mjcf(format!(
                "lint option penetration_fraction must be in (0, 1], got {}",
                self.penetration_fraction
            )));
        }
        Ok(())
    }
}

/// Which pathology a finding reports. `Display` prints the stable lint code
/// (`C001`…) that messages and tooling key on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LintCode {
    /// Explosion / divergence.
    C001Explosion,
    /// Persistent penetration.
    C002Penetration,
    /// Contact-force jitter.
    C003Jitter,
}

impl std::fmt::Display for LintCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::C001Explosion => "C001",
            Self::C002Penetration => "C002",
            Self::C003Jitter => "C003",
        })
    }
}

/// One linter finding: the code, what was measured (plain English), and the
/// suggested fix (which knob to turn, in order of likelihood).
#[derive(Clone, Debug)]
pub struct LintFinding {
    pub code: LintCode,
    /// Plain-English statement of what was observed, with the measured
    /// numbers and the thresholds they crossed.
    pub message: String,
    /// The concrete fix to try, most likely first.
    pub suggestion: String,
}

/// Per-step samples recorded over the OBSERVE window of a rollout (the settle
/// window contributes only the non-finite check). All vectors run in step
/// order and share one length; a [`nonfinite_at`](Self::nonfinite_at) rollout
/// may stop short.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StabilityTrace {
    /// Simulation timestep `h` (s) — used to phrase suggestions in concrete
    /// numbers (`2·h`).
    pub h: f64,
    /// Per step: max |qvel| over ALL dofs (robot joints AND prop freejoints).
    pub speeds: Vec<f64>,
    /// Per step: the energy proxy `0.5·Σ qd²` over all dofs. NOT true kinetic
    /// energy (no mass matrix) — only its GROWTH is judged, so the units
    /// cancel out.
    pub energies: Vec<f64>,
    /// Per step: deepest contact penetration (m), `0.0` when contact-free.
    pub depths: Vec<f64>,
    /// Per step: total normal force (N) summed over all contacts.
    pub normal_forces: Vec<f64>,
    /// `Some(step)` when `qpos`/`qvel` went NaN/inf at that rollout step
    /// (counted from the start of the settle window); recording stops there.
    pub nonfinite_at: Option<usize>,
}

/// Classify a recorded trace into findings. Pure and deterministic; empty
/// vector = the scene looks stable. An explosion (C001) is reported ALONE —
/// depth/force statistics sampled during a blow-up are meaningless, so
/// C002/C003 are only judged on non-exploding traces.
pub fn classify_stability(trace: &StabilityTrace, opts: &LintOptions) -> Vec<LintFinding> {
    let two_h = 2.0 * trace.h;
    let c001_fix = format!(
        "increase solref timeconst to >= {two_h} (2×timestep), reduce the timestep, switch to a \
         softer material preset (ContactMaterial::Rubber / ContactMaterial::Foam), or add joint \
         damping"
    );

    // Non-finite state is the unambiguous explosion — report it alone.
    if let Some(step) = trace.nonfinite_at {
        return vec![LintFinding {
            code: LintCode::C001Explosion,
            message: format!(
                "simulation state went non-finite (NaN/inf) at rollout step {step} (t ≈ {:.4} s) \
                 — the contact solver diverged",
                step as f64 * trace.h
            ),
            suggestion: c001_fix,
        }];
    }
    if trace.speeds.is_empty() {
        return Vec::new();
    }

    let mut findings = Vec::new();

    // C001 (a): runaway speed after the scene was given time to settle.
    let max_speed = trace.speeds.iter().copied().fold(0.0f64, f64::max);
    // C001 (b): the energy proxy still GROWING across the observe window.
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let (first, second) = trace.energies.split_at(trace.energies.len() / 2);
    let (e_first, e_second) = (mean(first), mean(second));
    let energy_growing = !first.is_empty()
        && e_second > opts.explosion_energy_growth * e_first
        && e_second > opts.explosion_energy_floor;
    if max_speed > opts.explosion_speed || energy_growing {
        let observed = if max_speed > opts.explosion_speed {
            format!(
                "a dof reached {max_speed:.1} rad/s (or m/s) after settling — threshold \
                 {} — the 'spins uncontrollably' failure",
                opts.explosion_speed
            )
        } else {
            format!(
                "the velocity-energy proxy GREW from {e_first:.3} to {e_second:.3} across the \
                 observe window (>{}×) — the contact is pumping energy into the system",
                opts.explosion_energy_growth
            )
        };
        return vec![LintFinding {
            code: LintCode::C001Explosion,
            message: format!("contact explosion: {observed}"),
            suggestion: c001_fix,
        }];
    }

    // C002: persistent penetration after settling.
    let deep = trace
        .depths
        .iter()
        .filter(|&&d| d > opts.penetration_depth)
        .count();
    let frac = deep as f64 / trace.depths.len() as f64;
    if frac >= opts.penetration_fraction {
        let max_depth = trace.depths.iter().copied().fold(0.0f64, f64::max);
        findings.push(LintFinding {
            code: LintCode::C002Penetration,
            message: format!(
                "persistent penetration: contact depth exceeded {} m in {:.0}% of observed steps \
                 after settling (max {max_depth:.4} m) — bodies are resting inside each other",
                opts.penetration_depth,
                100.0 * frac
            ),
            suggestion: "harden the contact: raise solimp (dmin, dmax) toward 1 and shrink its \
                         width, shorten solref timeconst (keeping it >= 2×timestep), or switch \
                         to a stiffer material preset (ContactMaterial::Rigid / \
                         ContactMaterial::Steel)"
                .into(),
        });
    }

    // C003: contact force still oscillating after settling.
    let f_mean = mean(&trace.normal_forces);
    if f_mean > opts.min_normal_force {
        let f_max = trace.normal_forces.iter().copied().fold(f64::MIN, f64::max);
        let f_min = trace.normal_forces.iter().copied().fold(f64::MAX, f64::min);
        let amp = f_max - f_min;
        if amp > opts.jitter_force_ratio * f_mean {
            findings.push(LintFinding {
                code: LintCode::C003Jitter,
                message: format!(
                    "contact jitter: total normal force oscillates with peak-to-peak amplitude \
                     {amp:.2} N around a mean of {f_mean:.2} N after settling (> {:.0}% of the \
                     mean) — the contact keeps ringing instead of resting",
                    100.0 * opts.jitter_force_ratio
                ),
                suggestion: format!(
                    "set solref dampratio to 1.0 (critical damping), increase solref timeconst \
                     (well above {two_h} = 2×timestep), reduce the timestep, or switch material \
                     preset (ContactMaterial::Rubber damps impacts)"
                ),
            });
        }
    }

    findings
}

#[cfg(feature = "mujoco")]
mod rollout {
    use super::*;
    use crate::MujocoSim;

    /// Roll the sim forward (settle window, then observe window), record a
    /// [`StabilityTrace`], and classify it. ADVANCES `sim` — hand it a
    /// freshly seeded sim and `reset` afterwards if you need the state back.
    ///
    /// Metrics cover ALL dofs and contacts (robot joints, prop freejoints,
    /// ground plane), read through the full MuJoCo state — not just the
    /// mapped robot joints. Deterministic: same sim state + options ⇒ same
    /// findings, bit-for-bit.
    pub fn lint_contact_stability(
        sim: &mut MujocoSim,
        opts: &LintOptions,
    ) -> Result<Vec<LintFinding>, MujocoError> {
        opts.validate()?;
        let h = sim.timestep();
        let settle_steps = (opts.settle_duration / h).round() as usize;
        let observe_steps = ((opts.observe_duration / h).round() as usize).max(2);

        let mut trace = StabilityTrace {
            h,
            ..Default::default()
        };
        let finite = |sim: &MujocoSim| {
            let d = sim.mj_data();
            d.qpos().iter().all(|x| x.is_finite()) && d.qvel().iter().all(|x| x.is_finite())
        };
        for k in 0..settle_steps {
            sim.step_once();
            if !finite(sim) {
                trace.nonfinite_at = Some(k);
                return Ok(classify_stability(&trace, opts));
            }
        }
        for k in 0..observe_steps {
            sim.step_once();
            if !finite(sim) {
                trace.nonfinite_at = Some(settle_steps + k);
                break;
            }
            let qv = sim.mj_data().qvel();
            trace
                .speeds
                .push(qv.iter().fold(0.0f64, |a, &v| a.max(v.abs())));
            trace
                .energies
                .push(0.5 * qv.iter().map(|v| v * v).sum::<f64>());
            trace.depths.push(
                sim.contacts()
                    .iter()
                    .fold(0.0f64, |a, c| a.max(c.depth.max(0.0))),
            );
            trace
                .normal_forces
                .push((0..sim.ncon()).map(|i| sim.contact_force(i)[0]).sum());
        }
        Ok(classify_stability(&trace, opts))
    }
}

#[cfg(feature = "mujoco")]
pub use rollout::lint_contact_stability;

#[cfg(test)]
mod tests {
    use super::*;

    /// A settled, resting trace: slow, constant energy, shallow depth,
    /// steady force.
    fn clean_trace(n: usize) -> StabilityTrace {
        StabilityTrace {
            h: 1e-3,
            speeds: vec![0.01; n],
            energies: vec![5e-5; n],
            depths: vec![0.0004; n],
            normal_forces: vec![9.81; n],
            nonfinite_at: None,
        }
    }

    #[test]
    fn clean_trace_is_finding_free() {
        let findings = classify_stability(&clean_trace(100), &LintOptions::default());
        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn nonfinite_state_is_c001_alone() {
        let mut t = clean_trace(10);
        t.nonfinite_at = Some(37);
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, LintCode::C001Explosion);
        assert_eq!(f[0].code.to_string(), "C001");
        assert!(f[0].message.contains("non-finite"), "{}", f[0].message);
        assert!(
            f[0].suggestion.contains("solref timeconst"),
            "{}",
            f[0].suggestion
        );
        // the concrete 2×timestep number is spelled out
        assert!(f[0].suggestion.contains("0.002"), "{}", f[0].suggestion);
    }

    #[test]
    fn runaway_speed_is_c001() {
        let mut t = clean_trace(100);
        t.speeds[80] = 350.0; // one dof takes off
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, LintCode::C001Explosion);
        assert!(f[0].message.contains("350.0"), "{}", f[0].message);
    }

    #[test]
    fn energy_growth_is_c001() {
        let mut t = clean_trace(100);
        // first half quiet, second half 1000× hotter and above the floor
        for e in t.energies[50..].iter_mut() {
            *e = 5e-2 * 1000.0;
        }
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, LintCode::C001Explosion);
        assert!(f[0].message.contains("pumping energy"), "{}", f[0].message);
    }

    /// Ratio alone must not fire under the absolute floor (1e-9 → 1e-7 is
    /// still a scene at rest).
    #[test]
    fn tiny_energy_ratio_is_not_an_explosion() {
        let mut t = clean_trace(100);
        for e in t.energies[..50].iter_mut() {
            *e = 1e-9;
        }
        for e in t.energies[50..].iter_mut() {
            *e = 1e-7;
        }
        assert!(classify_stability(&t, &LintOptions::default()).is_empty());
    }

    #[test]
    fn persistent_penetration_is_c002() {
        let mut t = clean_trace(100);
        t.depths = vec![0.02; 100]; // resting 2 cm inside the floor
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, LintCode::C002Penetration);
        assert!(f[0].message.contains("0.02"), "{}", f[0].message);
        assert!(f[0].suggestion.contains("solimp"), "{}", f[0].suggestion);
    }

    /// A brief impact spike (depth over threshold for a few steps only) is
    /// NOT persistent penetration.
    #[test]
    fn brief_impact_spike_is_not_c002() {
        let mut t = clean_trace(100);
        for d in t.depths[10..15].iter_mut() {
            *d = 0.02;
        }
        assert!(classify_stability(&t, &LintOptions::default()).is_empty());
    }

    #[test]
    fn oscillating_force_is_c003() {
        let mut t = clean_trace(100);
        // force slams between 0 and 2× the load every step — classic ringing
        t.normal_forces = (0..100)
            .map(|k| if k % 2 == 0 { 0.0 } else { 19.62 })
            .collect();
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, LintCode::C003Jitter);
        assert!(f[0].suggestion.contains("dampratio"), "{}", f[0].suggestion);
    }

    /// Steady contact force (tiny wiggle) is not jitter; neither is a big
    /// RELATIVE wiggle around a negligible force.
    #[test]
    fn steady_or_negligible_force_is_not_c003() {
        let mut t = clean_trace(100);
        t.normal_forces = (0..100).map(|k| 9.81 + 0.01 * (k % 2) as f64).collect();
        assert!(classify_stability(&t, &LintOptions::default()).is_empty());
        t.normal_forces = (0..100)
            .map(|k| 0.02 * (k % 2) as f64) // mean 0.01 N — no real contact
            .collect();
        assert!(classify_stability(&t, &LintOptions::default()).is_empty());
    }

    /// C002 and C003 can co-fire (soft AND ringing), reported in code order.
    #[test]
    fn penetration_and_jitter_co_report() {
        let mut t = clean_trace(100);
        t.depths = vec![0.02; 100];
        t.normal_forces = (0..100)
            .map(|k| if k % 2 == 0 { 0.0 } else { 19.62 })
            .collect();
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 2, "{f:?}");
        assert_eq!(f[0].code, LintCode::C002Penetration);
        assert_eq!(f[1].code, LintCode::C003Jitter);
    }

    /// An exploding trace reports C001 ALONE even when depth/force stats
    /// would also trip — they are garbage during a blow-up.
    #[test]
    fn explosion_suppresses_other_findings() {
        let mut t = clean_trace(100);
        t.speeds[99] = 1e6;
        t.depths = vec![0.5; 100];
        t.normal_forces = (0..100).map(|k| 1e4 * (k % 2) as f64).collect();
        let f = classify_stability(&t, &LintOptions::default());
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].code, LintCode::C001Explosion);
    }

    #[test]
    fn empty_trace_is_finding_free() {
        let t = StabilityTrace {
            h: 1e-3,
            ..Default::default()
        };
        assert!(classify_stability(&t, &LintOptions::default()).is_empty());
    }

    #[test]
    fn bad_lint_options_rejected() {
        let ok = LintOptions::default();
        assert!(ok.validate().is_ok());
        for bad in [
            LintOptions {
                settle_duration: 0.0,
                ..ok
            },
            LintOptions {
                observe_duration: f64::NAN,
                ..ok
            },
            LintOptions {
                explosion_speed: -1.0,
                ..ok
            },
            LintOptions {
                penetration_fraction: 0.0,
                ..ok
            },
            LintOptions {
                penetration_fraction: 1.5,
                ..ok
            },
            LintOptions {
                min_normal_force: f64::INFINITY,
                ..ok
            },
        ] {
            assert!(bad.validate().is_err(), "accepted {bad:?}");
        }
    }

    /// Determinism: the classifier is a pure function — identical traces give
    /// identical findings (message strings included).
    #[test]
    fn classifier_is_deterministic() {
        let mut t = clean_trace(100);
        t.depths = vec![0.02; 100];
        let a = classify_stability(&t, &LintOptions::default());
        let b = classify_stability(&t, &LintOptions::default());
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.code, y.code);
            assert_eq!(x.message, y.message);
            assert_eq!(x.suggestion, y.suggestion);
        }
    }
}
