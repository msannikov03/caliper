//! Real-hardware backend skeletons. Each is behind its own cargo feature and
//! OFF by default, so the lean core never pulls a transport dependency. The
//! wire-protocol codecs are real and unit-tested; every bus operation returns
//! [`Error::HardwareRequired`](crate::Error::HardwareRequired) (or
//! [`Error::NotConnected`](crate::Error::NotConnected) for the remote transport)
//! — these compile and encode correct frames, but cannot be claimed working
//! without the robot.

#[cfg(feature = "can")]
pub mod can_mks;

#[cfg(feature = "dynamixel")]
pub mod dynamixel;

#[cfg(feature = "remote")]
pub mod remote;
