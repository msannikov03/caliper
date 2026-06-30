//! Remote-control backend (feature `remote`).
//!
//! The wire CODEC is real and unit-tested: a length-prefixed binary frame
//! carrying a command id + `ndof` big-endian `f64` payload + a CRC-16. The
//! transport (the actual TCP/UDP/websocket socket) is DEFERRED — every bus op
//! returns [`Error::NotConnected`](crate::Error::NotConnected) until a socket is
//! wired, mirroring the CAN/Dynamixel skeletons. This compiles and emits correct
//! frames; it is NOT claimed to drive a remote robot.
//!
//! ## Frame layout
//! ```text
//! ┌─────────┬───────┬──────────┬──────────────────┬──────────┐
//! │ MAGIC   │ cmd   │ len(BE)  │ payload          │ crc(BE)  │
//! │ 0xCA 11 │ u8    │ u16      │ ndof × f64 BE    │ u16      │
//! └─────────┴───────┴──────────┴──────────────────┴──────────┘
//! ```
//! `len` is the payload byte count (`8 × ndof`). The CRC-16 (CCITT, poly
//! `0x1021`, init `0xFFFF`) covers everything between the magic and the CRC:
//! `[cmd, len_hi, len_lo, payload…]`.

use crate::{ControlMode, Error, JointState, RobotBackend, check_finite, check_len};

/// Two-byte frame sync marker (`'C','a'` → robot-arm "caliper").
pub const MAGIC: [u8; 2] = [0xCA, 0x11];
/// Bytes that precede the payload: `MAGIC(2) + cmd(1) + len(2)`.
const HEADER_LEN: usize = 5;
/// Trailing CRC width.
const CRC_LEN: usize = 2;
/// Smallest legal frame: header + empty payload + crc.
const MIN_FRAME: usize = HEADER_LEN + CRC_LEN;

/// Frame command id. The remote peer commands joint targets and reports state
/// over the same framing, distinguished by this byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RemoteCmd {
    /// Host → robot: commanded joint positions (rad).
    Command = 0x01,
    /// Robot → host: measured joint positions (rad).
    State = 0x02,
}

impl RemoteCmd {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(RemoteCmd::Command),
            0x02 => Some(RemoteCmd::State),
            _ => None,
        }
    }
}

/// CRC-16/CCITT-FALSE (poly `0x1021`, init `0xFFFF`, no reflection). Self-contained
/// (no external crate) so the lean core stays dependency-free.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Encode a frame: `cmd` + `ndof` big-endian `f64` values, framed and checksummed.
///
/// Returns [`Error::NonFinite`] if any value is not finite (a NaN/inf must never
/// be silently put on the wire as a joint target). The payload byte length must
/// fit the `u16` length field — far beyond any real robot's DoF — guarded below.
pub fn encode_frame(cmd: RemoteCmd, values: &[f64]) -> Result<Vec<u8>, Error> {
    check_finite("frame payload", values)?;
    // 8 bytes per f64; the length field is a u16, so cap ndof accordingly. Real
    // robots are far below this; the guard just keeps the cast lossless.
    const MAX_NDOF: usize = (u16::MAX as usize) / 8;
    if values.len() > MAX_NDOF {
        return Err(Error::Backend(format!(
            "remote frame ndof {} exceeds u16 length field",
            values.len()
        )));
    }
    let payload_len = (values.len() * 8) as u16;
    let mut frame = Vec::with_capacity(MIN_FRAME + payload_len as usize);
    frame.extend_from_slice(&MAGIC);
    frame.push(cmd as u8);
    frame.extend_from_slice(&payload_len.to_be_bytes());
    for &v in values {
        frame.extend_from_slice(&v.to_be_bytes());
    }
    let crc = crc16_ccitt(&frame[MAGIC.len()..]);
    frame.extend_from_slice(&crc.to_be_bytes());
    Ok(frame)
}

/// Decode one whole frame back into `(cmd, values)`.
///
/// Rejects (with [`Error::Backend`]) any frame that is short, lacks the magic,
/// carries an unknown command id, has a length field inconsistent with the buffer
/// or not a multiple of 8, or fails the CRC — so a corrupt frame is never decoded
/// into bogus joint values.
pub fn decode_frame(buf: &[u8]) -> Result<(RemoteCmd, Vec<f64>), Error> {
    if buf.len() < MIN_FRAME {
        return Err(Error::Backend("remote frame too short".into()));
    }
    if buf[..2] != MAGIC {
        return Err(Error::Backend("remote frame bad magic".into()));
    }
    let cmd = RemoteCmd::from_u8(buf[2])
        .ok_or_else(|| Error::Backend(format!("remote frame unknown cmd 0x{:02X}", buf[2])))?;
    let payload_len = u16::from_be_bytes([buf[3], buf[4]]) as usize;
    if payload_len % 8 != 0 {
        return Err(Error::Backend(
            "remote frame payload not f64-aligned".into(),
        ));
    }
    if buf.len() != HEADER_LEN + payload_len + CRC_LEN {
        return Err(Error::Backend("remote frame length mismatch".into()));
    }
    let crc_at = HEADER_LEN + payload_len;
    let want = u16::from_be_bytes([buf[crc_at], buf[crc_at + 1]]);
    // CRC covers [cmd, len_hi, len_lo, payload…] — everything after the magic.
    let got = crc16_ccitt(&buf[MAGIC.len()..crc_at]);
    if got != want {
        return Err(Error::Backend("remote frame crc mismatch".into()));
    }
    let values = buf[HEADER_LEN..crc_at]
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();
    Ok((cmd, values))
}

/// A remote-control robot backend skeleton. Holds the DoF + last-known joint
/// vector and encodes correct frames; the network transport is not bound (every
/// bus op returns [`Error::NotConnected`]).
pub struct RemoteBackend {
    dof: usize,
    addr: String,
    q: Vec<f64>,
    connected: bool,
    enabled: bool,
    estopped: bool,
}

impl RemoteBackend {
    /// Build for `dof` joints targeting `addr` (e.g. `"tcp://10.0.0.2:9000"`),
    /// kept for when a transport is wired. No socket is opened.
    pub fn new(dof: usize, addr: impl Into<String>) -> Self {
        Self {
            dof,
            addr: addr.into(),
            q: vec![0.0; dof],
            connected: false,
            enabled: false,
            estopped: false,
        }
    }

    /// The configured peer address.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Whether a transport is live. Always `false` until a socket is wired.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Open the transport. DEFERRED: no socket layer is linked, so this always
    /// fails [`Error::NotConnected`] rather than pretending to connect.
    pub fn connect(&mut self) -> Result<(), Error> {
        Err(Error::NotConnected)
    }

    /// The bytes a position command WOULD send (for inspection / a future socket).
    pub fn encode_command(&self, q: &[f64]) -> Result<Vec<u8>, Error> {
        check_len(q.len(), self.dof)?;
        encode_frame(RemoteCmd::Command, q)
    }
}

impl RobotBackend for RemoteBackend {
    fn dof(&self) -> usize {
        self.dof
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
        // would require a live socket to the remote peer
        Err(Error::NotConnected)
    }
    fn disable(&mut self) -> Result<(), Error> {
        Err(Error::NotConnected)
    }
    fn command_joint_positions(&mut self, q: &[f64]) -> Result<(), Error> {
        let _frame = self.encode_command(q)?; // codec runs (validated); socket does not
        Err(Error::NotConnected)
    }
    fn read_state(&mut self) -> Result<JointState, Error> {
        Err(Error::NotConnected)
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_frame_roundtrips_exactly() {
        let q = [
            0.0,
            -1.5707963267948966,
            3.141592653589793,
            0.25,
            -0.75,
            2.5,
        ];
        let bytes = encode_frame(RemoteCmd::Command, &q).unwrap();
        // header(5) + 6*8 payload + 2 crc
        assert_eq!(bytes.len(), HEADER_LEN + q.len() * 8 + CRC_LEN);
        assert_eq!(&bytes[..2], &MAGIC);
        assert_eq!(bytes[2], RemoteCmd::Command as u8);
        let (cmd, back) = decode_frame(&bytes).unwrap();
        assert_eq!(cmd, RemoteCmd::Command);
        // exact bit-for-bit recovery (no rounding in the codec)
        assert_eq!(back, q.to_vec());
    }

    #[test]
    fn state_frame_roundtrips_and_is_distinguished() {
        let q = [1.0, 2.0, 3.0];
        let bytes = encode_frame(RemoteCmd::State, &q).unwrap();
        let (cmd, back) = decode_frame(&bytes).unwrap();
        assert_eq!(cmd, RemoteCmd::State);
        assert_eq!(back, q.to_vec());
        // a state frame is not mistaken for a command frame
        assert_ne!(RemoteCmd::State as u8, RemoteCmd::Command as u8);
    }

    #[test]
    fn empty_payload_roundtrips() {
        let bytes = encode_frame(RemoteCmd::State, &[]).unwrap();
        assert_eq!(bytes.len(), MIN_FRAME);
        let (cmd, back) = decode_frame(&bytes).unwrap();
        assert_eq!(cmd, RemoteCmd::State);
        assert!(back.is_empty());
    }

    #[test]
    fn corrupt_payload_fails_crc() {
        let mut bytes = encode_frame(RemoteCmd::Command, &[0.5, 1.5]).unwrap();
        // flip one payload bit → CRC must catch it
        bytes[HEADER_LEN] ^= 0x01;
        assert!(matches!(decode_frame(&bytes), Err(Error::Backend(_))));
    }

    #[test]
    fn flipped_crc_is_rejected() {
        let mut bytes = encode_frame(RemoteCmd::Command, &[0.5]).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        assert!(matches!(decode_frame(&bytes), Err(Error::Backend(_))));
    }

    #[test]
    fn short_frame_is_rejected() {
        let bytes = encode_frame(RemoteCmd::Command, &[0.5, 1.0]).unwrap();
        // truncated below a full frame
        assert!(matches!(decode_frame(&bytes[..6]), Err(Error::Backend(_))));
        assert!(matches!(decode_frame(&[]), Err(Error::Backend(_))));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = encode_frame(RemoteCmd::Command, &[0.5]).unwrap();
        bytes[0] ^= 0xFF;
        assert!(matches!(decode_frame(&bytes), Err(Error::Backend(_))));
    }

    #[test]
    fn unknown_command_is_rejected() {
        let mut bytes = encode_frame(RemoteCmd::Command, &[0.5]).unwrap();
        bytes[2] = 0x7F; // not a known RemoteCmd
        assert!(matches!(decode_frame(&bytes), Err(Error::Backend(_))));
    }

    #[test]
    fn length_field_inconsistent_is_rejected() {
        let bytes = encode_frame(RemoteCmd::Command, &[0.5, 1.0]).unwrap();
        // claim a longer payload than the buffer holds
        let mut tampered = bytes.clone();
        tampered[3] = 0xFF;
        tampered[4] = 0xFF;
        assert!(matches!(decode_frame(&tampered), Err(Error::Backend(_))));
        // claim a non-f64-aligned payload length
        let mut misaligned = bytes;
        misaligned[4] = misaligned[4].wrapping_sub(1);
        assert!(matches!(decode_frame(&misaligned), Err(Error::Backend(_))));
    }

    #[test]
    fn nonfinite_payload_is_refused() {
        assert!(matches!(
            encode_frame(RemoteCmd::Command, &[0.0, f64::NAN]),
            Err(Error::NonFinite { .. })
        ));
        assert!(matches!(
            encode_frame(RemoteCmd::Command, &[f64::INFINITY]),
            Err(Error::NonFinite { .. })
        ));
    }

    #[test]
    fn crc16_distinguishes_single_bit_flips() {
        // a real CRC must change when any input byte changes
        let a = crc16_ccitt(&[
            0x01, 0x00, 0x08, 0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        let mut data = [
            0x01, 0x00, 0x08, 0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        data[5] ^= 0x01;
        assert_ne!(a, crc16_ccitt(&data));
    }

    #[test]
    fn bus_ops_require_connection() {
        let mut b = RemoteBackend::new(3, "tcp://127.0.0.1:9000");
        assert_eq!(b.dof(), 3);
        assert_eq!(b.addr(), "tcp://127.0.0.1:9000");
        // codec works without a socket
        let frame = b.encode_command(&[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(decode_frame(&frame).unwrap().1, vec![0.1, 0.2, 0.3]);
        // but every transport op refuses — never a fake success
        assert!(matches!(b.connect(), Err(Error::NotConnected)));
        assert!(matches!(b.enable(), Err(Error::NotConnected)));
        assert!(matches!(b.disable(), Err(Error::NotConnected)));
        assert!(matches!(
            b.command_joint_positions(&[0.1, 0.2, 0.3]),
            Err(Error::NotConnected)
        ));
        assert!(matches!(b.read_state(), Err(Error::NotConnected)));
        assert!(!b.connected);
    }

    #[test]
    fn command_validates_dof_before_encoding() {
        let b = RemoteBackend::new(3, "tcp://x");
        assert!(matches!(
            b.encode_command(&[0.1, 0.2]),
            Err(Error::DofMismatch { .. })
        ));
    }

    #[test]
    fn estop_latches_locally() {
        let mut b = RemoteBackend::new(2, "tcp://x");
        b.estop().unwrap();
        assert!(b.is_estopped());
        assert!(!b.is_enabled());
        b.clear_estop().unwrap();
        assert!(!b.is_estopped());
    }
}
