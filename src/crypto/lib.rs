//! Minimal ABY3-like MPC primitives (Rust sketch)
//!
//! This module will implement:
//! - Shared arithmetic type (field-ish) for add/mul
//! - Conversion to shared bits
//! - Shared bit type and bitwise ops (and/or/not), comparisons, mod
//!
//! The implementation will target a local, single-process simulation of 3 parties
//! using additive secret-sharing across three shares. Later we will extract
//! networking and true MPC communication.

pub mod types;
pub mod arith;
pub mod bits;
pub mod party;

pub use types::*;
pub use arith::*;
pub use bits::*;
pub use party::*;

#[cfg(test)]
pub use arith::{share_secret, reconstruct, add_shares, mul_insecure, mul_ab3};
#[cfg(test)]
pub use bits::{share_bool, reconstruct_bit, and, or, not};
