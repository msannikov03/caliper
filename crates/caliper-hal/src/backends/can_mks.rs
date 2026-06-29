//! MKS SERVO42D/57D backend over CAN (feature `can`).
//!
//! The frame CODEC is real and unit-tested (MKS V1 CAN protocol: 11-bit id +
//! ≤8 data bytes, last byte = additive checksum). The bus binding itself
//! (socketcan) is DEFERRED — every transmit/read returns
//! [`Error::HardwareRequired`](crate::Error::HardwareRequired). This compiles and
//! emits correct frames; it is NOT claimed to drive a motor.

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite, check_len};
use std::f64::consts::PI;

/// MKS magnetic encoder resolution (pulses per revolution, `0x4000`).
pub const MKS_PULSES_PER_REV: i64 = 16384;
const POS_24_MAX: i32 = 0x7F_FF_FF;

/// A raw MKS CAN frame: standard 11-bit id + data (last byte is the checksum).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MksFrame {
    pub id: u16,
    pub data: Vec<u8>,
}
impl MksFrame {
    fn with_checksum(id: u16, mut body: Vec<u8>) -> Self {
        let crc = mks_checksum(id, &body);
        body.push(crc);
        MksFrame { id, data: body }
    }
    /// The command byte (first data byte), if any.
    pub fn command(&self) -> Option<u8> {
        self.data.first().copied()
    }
    /// Verify the trailing checksum.
    pub fn checksum_ok(&self) -> bool {
        match self.data.split_last() {
            Some((&crc, body)) => mks_checksum(self.id, body) == crc,
            None => false,
        }
    }
}

/// MKS additive checksum: `(id + Σ body) & 0xFF`.
pub fn mks_checksum(id: u16, body: &[u8]) -> u8 {
    let mut s = (id & 0xFF) as u32;
    for &b in body {
        s = s.wrapping_add(b as u32);
    }
    (s & 0xFF) as u8
}

/// Radians → encoder pulses (round to nearest).
pub fn rad_to_pulses(rad: f64, ppr: i64) -> i64 {
    (rad / (2.0 * PI) * ppr as f64).round() as i64
}
/// Encoder pulses → radians.
pub fn pulses_to_rad(p: i64, ppr: i64) -> f64 {
    p as f64 / ppr as f64 * 2.0 * PI
}

/// Enable/disable a motor (command `0xF3`).
pub fn encode_enable(id: u16, enable: bool) -> MksFrame {
    MksFrame::with_checksum(id, vec![0xF3, u8::from(enable)])
}

/// Absolute position move by pulses (command `0xF5`): 16-bit speed, 8-bit accel,
/// 24-bit signed absolute position.
pub fn encode_set_position(id: u16, pulses: i32, speed: u16, accel: u8) -> MksFrame {
    let p = pulses.clamp(-POS_24_MAX, POS_24_MAX);
    let body = vec![
        0xF5,
        (speed >> 8) as u8,
        (speed & 0xFF) as u8,
        accel,
        ((p >> 16) & 0xFF) as u8,
        ((p >> 8) & 0xFF) as u8,
        (p & 0xFF) as u8,
    ];
    MksFrame::with_checksum(id, body)
}

/// Request the encoder carry+value (command `0x30`).
pub fn encode_read_encoder(id: u16) -> MksFrame {
    MksFrame::with_checksum(id, vec![0x30])
}

/// Decode a `0x30` reply `[0x30, carry(i32 BE), value(u16 BE), crc]` → absolute pulses.
///
/// `id` is the CAN arbitration id the reply arrived on; it is required to verify
/// the additive checksum (`(id + Σbody) & 0xFF`). The frame is rejected unless the
/// command byte is `0x30`, the length is exactly 8, and the trailing checksum
/// matches — guarding against corrupt or short frames.
pub fn decode_encoder_reply(id: u16, data: &[u8]) -> Result<i64, Error> {
    const EXPECTED_LEN: usize = 8; // [0x30, carry(4 BE), value(2 BE), crc]
    let frame = MksFrame {
        id,
        data: data.to_vec(),
    };
    if data.len() != EXPECTED_LEN || frame.command() != Some(0x30) || !frame.checksum_ok() {
        return Err(Error::Backend("malformed 0x30 encoder reply".into()));
    }
    let carry = i32::from_be_bytes([data[1], data[2], data[3], data[4]]) as i64;
    let value = u16::from_be_bytes([data[5], data[6]]) as i64;
    Ok(carry * MKS_PULSES_PER_REV + value)
}

/// One MKS motor: a CAN id and its encoder resolution.
#[derive(Clone, Debug)]
pub struct MksMotor {
    pub id: u16,
    pub ppr: i64,
}

/// A CAN/MKS robot backend skeleton. Holds per-joint motor config and encodes
/// correct frames; the bus transport is not bound (every op is HardwareRequired).
pub struct CanMksBackend {
    motors: Vec<MksMotor>,
    speed: u16,
    accel: u8,
    q: Vec<f64>,
    enabled: bool,
    estopped: bool,
}

impl CanMksBackend {
    /// Build from `(can_id, pulses_per_rev)` per joint.
    pub fn new(motors: Vec<MksMotor>) -> Self {
        let n = motors.len();
        Self {
            motors,
            speed: 600,
            accel: 2,
            q: vec![0.0; n],
            enabled: false,
            estopped: false,
        }
    }
    /// The CAN frames a position command WOULD send (for inspection / a future bus).
    pub fn encode_command(&self, q: &[f64]) -> Result<Vec<MksFrame>, Error> {
        check_len(q.len(), self.motors.len())?;
        check_finite("q", q)?;
        Ok(self
            .motors
            .iter()
            .zip(q)
            .map(|(m, &qi)| {
                encode_set_position(
                    m.id,
                    rad_to_pulses(qi, m.ppr) as i32,
                    self.speed,
                    self.accel,
                )
            })
            .collect())
    }
}

impl RobotBackend for CanMksBackend {
    fn dof(&self) -> usize {
        self.motors.len()
    }
    fn joint_positions(&self) -> Vec<f64> {
        self.q.clone()
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn is_estopped(&self) -> bool {
        self.estopped
    }
    fn estop(&mut self) -> Result<(), Error> {
        self.estopped = true;
        self.enabled = false;
        Ok(())
    }
    fn clear_estop(&mut self) -> Result<(), Error> {
        self.estopped = false;
        Ok(())
    }
    fn enable(&mut self) -> Result<(), Error> {
        // would transmit encode_enable(id, true) on the bus
        Err(Error::HardwareRequired("CAN/MKS bus not connected"))
    }
    fn disable(&mut self) -> Result<(), Error> {
        Err(Error::HardwareRequired("CAN/MKS bus not connected"))
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        let _frames = self.encode_command(q)?; // codec runs (validated); bus does not
        Err(Error::HardwareRequired("CAN/MKS bus not connected"))
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        Err(Error::HardwareRequired("CAN/MKS bus not connected"))
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_roundtrip() {
        let f = encode_set_position(1, 12345, 600, 2);
        assert!(f.checksum_ok());
        assert_eq!(f.command(), Some(0xF5));
        // tamper → checksum fails
        let mut bad = f.clone();
        bad.data[1] ^= 0xFF;
        assert!(!bad.checksum_ok());
    }

    #[test]
    fn position_encoding_is_24bit_be() {
        let f = encode_set_position(3, 0x01_23_45, 0x02_58, 5);
        // [0xF5, spd_hi, spd_lo, acc, p2, p1, p0, crc]  (position 0x012345, 24-bit BE)
        assert_eq!(&f.data[..7], &[0xF5, 0x02, 0x58, 0x05, 0x01, 0x23, 0x45]);
        assert_eq!(f.data.len(), 8);
    }

    #[test]
    fn pulses_rad_invert() {
        for &deg in &[0.0, 90.0, -123.4, 359.9] {
            let rad = deg * PI / 180.0;
            let p = rad_to_pulses(rad, MKS_PULSES_PER_REV);
            let back = pulses_to_rad(p, MKS_PULSES_PER_REV);
            assert!((back - rad).abs() < 1e-3, "deg={deg}");
        }
    }

    #[test]
    fn encoder_reply_decodes() {
        // carry=2 rev, value=8192 (half rev) → 2*16384 + 8192 = 40960 pulses.
        // crc = (id + Σbody) & 0xFF = (1 + 0x30 + 2 + 0x20) & 0xFF = 0x53.
        let data = [0x30, 0, 0, 0, 2, 0x20, 0x00, 0x53];
        assert_eq!(decode_encoder_reply(1, &data).unwrap(), 2 * 16384 + 8192);
    }

    #[test]
    fn encoder_reply_rejects_bad_frames() {
        // valid frame for id=1
        let good = [0x30, 0, 0, 0, 2, 0x20, 0x00, 0x53];
        assert!(decode_encoder_reply(1, &good).is_ok());
        // short frame (truncated)
        assert!(matches!(
            decode_encoder_reply(1, &good[..6]),
            Err(Error::Backend(_))
        ));
        // overlong frame
        let long = [0x30, 0, 0, 0, 2, 0x20, 0x00, 0x53, 0x00];
        assert!(matches!(
            decode_encoder_reply(1, &long),
            Err(Error::Backend(_))
        ));
        // wrong command/identity byte
        let mut wrong_cmd = good;
        wrong_cmd[0] = 0x31;
        assert!(matches!(
            decode_encoder_reply(1, &wrong_cmd),
            Err(Error::Backend(_))
        ));
        // corrupt payload → checksum mismatch
        let mut corrupt = good;
        corrupt[5] ^= 0xFF;
        assert!(matches!(
            decode_encoder_reply(1, &corrupt),
            Err(Error::Backend(_))
        ));
        // valid bytes but wrong id → checksum mismatch
        assert!(matches!(
            decode_encoder_reply(2, &good),
            Err(Error::Backend(_))
        ));
    }

    #[test]
    fn enable_frame() {
        let f = encode_enable(7, true);
        assert_eq!(&f.data[..2], &[0xF3, 0x01]);
        assert!(f.checksum_ok());
    }

    #[test]
    fn bus_ops_require_hardware() {
        let mut b = CanMksBackend::new(vec![MksMotor {
            id: 1,
            ppr: MKS_PULSES_PER_REV,
        }]);
        // codec works
        assert_eq!(b.encode_command(&[0.5]).unwrap().len(), 1);
        // but the bus does not
        assert!(matches!(b.enable(), Err(Error::HardwareRequired(_))));
        assert!(matches!(
            b.command_joint_positions(&[0.5]),
            Err(Error::HardwareRequired(_))
        ));
        assert!(matches!(b.read_state(), Err(Error::HardwareRequired(_))));
    }
}
