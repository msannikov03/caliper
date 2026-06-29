//! Dynamixel Protocol 2.0 backend (feature `dynamixel`).
//!
//! The packet CODEC is real and unit-tested — including the CRC-16 (poly `0x8005`)
//! against the published Robotis Ping vector (`0x4E19`). The serial bus binding is
//! DEFERRED: every transmit/read returns
//! [`Error::HardwareRequired`](crate::Error::HardwareRequired). Compiles and emits
//! correct packets; NOT claimed to drive a servo.

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite, check_len};
use std::f64::consts::PI;

/// Dynamixel X-series resolution (ticks per revolution).
pub const DXL_TICKS_PER_REV: i64 = 4096;
pub const ADDR_TORQUE_ENABLE: u16 = 64;
pub const ADDR_GOAL_POSITION: u16 = 116;
pub const ADDR_PRESENT_POSITION: u16 = 132;
const INST_READ: u8 = 0x02;
const INST_WRITE: u8 = 0x03;
const INST_SYNC_WRITE: u8 = 0x83;
const BROADCAST_ID: u8 = 0xFE;

/// Protocol-2.0 CRC-16 (polynomial `0x8005`, MSB-first, init `0x0000`).
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Build a Protocol-2.0 instruction packet (header, id, length, instruction,
/// params, CRC-16). Length = params + 3 (instruction + 2 CRC bytes).
pub fn build_packet(id: u8, instruction: u8, params: &[u8]) -> Vec<u8> {
    // The length field is a u16 = params + 3 (instruction + 2 CRC). An oversized
    // payload would silently truncate the `as u16` cast and desync the length
    // field from the bytes actually written. build_packet returns `Vec<u8>` (no
    // Result to surface an error through), so guard by clamping params to the
    // largest payload the length field can describe; real Dynamixel packets are
    // far below this bound, so this never triggers in practice.
    const MAX_PARAMS: usize = (u16::MAX as usize) - 3;
    debug_assert!(
        params.len() <= MAX_PARAMS,
        "Dynamixel packet params exceed the u16 length field"
    );
    let params = &params[..params.len().min(MAX_PARAMS)];
    let len = (params.len() + 3) as u16;
    let mut pkt = vec![
        0xFF,
        0xFF,
        0xFD,
        0x00,
        id,
        (len & 0xFF) as u8,
        (len >> 8) as u8,
        instruction,
    ];
    pkt.extend_from_slice(params);
    let crc = crc16(&pkt);
    pkt.push((crc & 0xFF) as u8);
    pkt.push((crc >> 8) as u8);
    pkt
}

/// Radians → goal ticks, centered at half-range (`0 rad` → `2048`).
///
/// Clamps the float intermediate to the representable `i64` range before the cast
/// and uses `saturating_add` for the half-range offset, so extreme-but-finite
/// input saturates instead of overflowing (mirrors the CAN `rad_to_pulses`
/// clamp). For ordinary inputs the result is unchanged.
pub fn rad_to_ticks(rad: f64, tpr: i64) -> i64 {
    let raw = (rad / (2.0 * PI) * tpr as f64).round();
    let ticks = raw.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    ticks.saturating_add(tpr / 2)
}
/// Goal ticks → radians (inverse of [`rad_to_ticks`]).
pub fn ticks_to_rad(ticks: i64, tpr: i64) -> f64 {
    (ticks - tpr / 2) as f64 / tpr as f64 * 2.0 * PI
}

/// Ping a servo (instruction `0x01`).
pub fn encode_ping(id: u8) -> Vec<u8> {
    build_packet(id, 0x01, &[])
}
/// Torque-enable write (control-table addr 64).
pub fn encode_torque_enable(id: u8, on: bool) -> Vec<u8> {
    build_packet(
        id,
        INST_WRITE,
        &[
            (ADDR_TORQUE_ENABLE & 0xFF) as u8,
            (ADDR_TORQUE_ENABLE >> 8) as u8,
            u8::from(on),
        ],
    )
}
/// Single goal-position write (control-table addr 116, 4-byte LE ticks).
pub fn encode_goal_position(id: u8, ticks: i32) -> Vec<u8> {
    let t = ticks.to_le_bytes();
    build_packet(
        id,
        INST_WRITE,
        &[
            (ADDR_GOAL_POSITION & 0xFF) as u8,
            (ADDR_GOAL_POSITION >> 8) as u8,
            t[0],
            t[1],
            t[2],
            t[3],
        ],
    )
}
/// Read present position (instruction `0x02`, addr 132, len 4).
pub fn encode_read_position(id: u8) -> Vec<u8> {
    build_packet(
        id,
        INST_READ,
        &[
            (ADDR_PRESENT_POSITION & 0xFF) as u8,
            (ADDR_PRESENT_POSITION >> 8) as u8,
            4,
            0,
        ],
    )
}
/// Sync-write goal positions to many servos at once (instruction `0x83`).
pub fn encode_sync_write_goal(servos: &[(u8, i32)]) -> Vec<u8> {
    let mut params = vec![
        (ADDR_GOAL_POSITION & 0xFF) as u8,
        (ADDR_GOAL_POSITION >> 8) as u8,
        4,
        0,
    ];
    for &(id, ticks) in servos {
        params.push(id);
        params.extend_from_slice(&ticks.to_le_bytes());
    }
    build_packet(BROADCAST_ID, INST_SYNC_WRITE, &params)
}

/// One Dynamixel servo: a bus id and its resolution.
#[derive(Clone, Debug)]
pub struct DxlServo {
    pub id: u8,
    pub tpr: i64,
}

/// A Dynamixel robot backend skeleton. Holds per-joint servo config and encodes
/// correct packets; the serial transport is not bound (every op HardwareRequired).
pub struct DynamixelBackend {
    servos: Vec<DxlServo>,
    q: Vec<f64>,
    enabled: bool,
    estopped: bool,
}

impl DynamixelBackend {
    pub fn new(servos: Vec<DxlServo>) -> Self {
        let n = servos.len();
        Self {
            servos,
            q: vec![0.0; n],
            enabled: false,
            estopped: false,
        }
    }
    /// The sync-write packet a position command WOULD send.
    pub fn encode_command(&self, q: &[f64]) -> Result<Vec<u8>, Error> {
        check_len(q.len(), self.servos.len())?;
        check_finite("q", q)?;
        let servos: Vec<(u8, i32)> = self
            .servos
            .iter()
            .zip(q)
            .map(|(s, &qi)| (s.id, rad_to_ticks(qi, s.tpr) as i32))
            .collect();
        Ok(encode_sync_write_goal(&servos))
    }
}

impl RobotBackend for DynamixelBackend {
    fn dof(&self) -> usize {
        self.servos.len()
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
        Err(Error::HardwareRequired(
            "Dynamixel serial bus not connected",
        ))
    }
    fn disable(&mut self) -> Result<(), Error> {
        Err(Error::HardwareRequired(
            "Dynamixel serial bus not connected",
        ))
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        let _packet = self.encode_command(q)?; // codec runs (validated); bus does not
        Err(Error::HardwareRequired(
            "Dynamixel serial bus not connected",
        ))
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        Err(Error::HardwareRequired(
            "Dynamixel serial bus not connected",
        ))
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc16_matches_robotis_ping_vector() {
        // Published Protocol-2.0 Ping(ID=1): FF FF FD 00 01 03 00 01 -> CRC 0x4E19.
        let head = [0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x03, 0x00, 0x01];
        assert_eq!(crc16(&head), 0x4E19);
        // and the full built packet carries it little-endian.
        let pkt = encode_ping(1);
        assert_eq!(
            &pkt,
            &[0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x03, 0x00, 0x01, 0x19, 0x4E]
        );
    }

    #[test]
    fn packet_length_field() {
        let pkt = encode_goal_position(2, 2048);
        // params = addr(2) + ticks(4) = 6; length = 6 + 3 = 9
        assert_eq!(pkt[5], 9);
        assert_eq!(pkt[6], 0);
        assert_eq!(pkt[7], INST_WRITE);
        // crc verifies
        let n = pkt.len();
        let crc = u16::from_le_bytes([pkt[n - 2], pkt[n - 1]]);
        assert_eq!(crc16(&pkt[..n - 2]), crc);
    }

    #[test]
    fn ticks_rad_invert() {
        for &deg in &[0.0, 90.0, -45.0, 179.0] {
            let rad = deg * PI / 180.0;
            let t = rad_to_ticks(rad, DXL_TICKS_PER_REV);
            let back = ticks_to_rad(t, DXL_TICKS_PER_REV);
            assert!((back - rad).abs() < 2e-3, "deg={deg}");
        }
        assert_eq!(rad_to_ticks(0.0, DXL_TICKS_PER_REV), 2048); // centered
    }

    #[test]
    fn rad_to_ticks_saturates_on_extreme_input() {
        // Without the clamp + saturating_add, `huge as i64` then `+ tpr/2` would
        // overflow (panic in debug) or wrap (release). It must saturate instead.
        assert_eq!(rad_to_ticks(f64::MAX, DXL_TICKS_PER_REV), i64::MAX);
        assert_eq!(
            rad_to_ticks(f64::MIN, DXL_TICKS_PER_REV),
            i64::MIN.saturating_add(DXL_TICKS_PER_REV / 2)
        );
        // ordinary inputs are unaffected by the guard
        assert_eq!(rad_to_ticks(0.0, DXL_TICKS_PER_REV), 2048);
    }

    #[test]
    fn sync_write_layout() {
        let pkt = encode_sync_write_goal(&[(1, 100), (2, 200)]);
        assert_eq!(pkt[4], BROADCAST_ID);
        assert_eq!(pkt[7], INST_SYNC_WRITE);
        // params: addr(2)=116,0 len(2)=4,0 then 2*(id + 4 ticks) = 10 → total 14
        assert_eq!(pkt[8], 116);
        assert_eq!(pkt[10], 4);
        let crc = u16::from_le_bytes([pkt[pkt.len() - 2], pkt[pkt.len() - 1]]);
        assert_eq!(crc16(&pkt[..pkt.len() - 2]), crc);
    }

    #[test]
    fn bus_ops_require_hardware() {
        let mut b = DynamixelBackend::new(vec![DxlServo {
            id: 1,
            tpr: DXL_TICKS_PER_REV,
        }]);
        assert!(!b.encode_command(&[0.3]).unwrap().is_empty());
        assert!(matches!(b.enable(), Err(Error::HardwareRequired(_))));
        assert!(matches!(
            b.command_joint_positions(&[0.3]),
            Err(Error::HardwareRequired(_))
        ));
        assert!(matches!(b.read_state(), Err(Error::HardwareRequired(_))));
    }
}
