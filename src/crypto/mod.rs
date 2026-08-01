//! crypto module root for crate::crypto

pub mod aby3_protocol;
pub mod aby3_verify;
pub mod attestations;
pub mod garbled;
pub mod gc_protocol;
pub mod arith;
pub mod bits;
pub mod field;
pub mod identity;
pub mod net;
pub mod protocol;
pub mod security;
pub mod sim;
#[cfg(feature = "pq-tls")]
pub mod tls;
pub mod types;

pub use arith::*;
pub use net::*;
pub use protocol::{
    a2b_party, add_party, mul_party, reveal, scalar_mul_party,
    share_input, share_public, sub_party, PartyShare,
};
pub use security::{
    require_security, Abort, AbortReason, AttestedSecurity, Checked, Security, SecurityGateError,
};
pub use sim::*;
pub use types::*;

#[cfg(test)]
pub use arith::{
    add_shares, eq_simulate, le_simulate, lt_simulate, mul_ab3, reconstruct, share_secret,
};
#[cfg(test)]
pub use bits::{
    a2b_simulate, and, eq_simulate as bits_eq, le_simulate as bits_le, lt_simulate as bits_lt, not,
    or, reconstruct_bit, share_bool,
};
