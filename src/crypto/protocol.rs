//! Generic MPC protocol helpers and utilities.
//!
//! This module contains generic helpers and types that are not specific to any
//! particular protocol. Protocol-specific implementations (ABY3, Garbled Circuits,
//! FHE, etc.) are in their own modules (e.g., aby3_protocol.rs).

// Re-export ABY3-specific protocol from its module for backward compatibility
pub use crate::crypto::aby3_protocol::{
    PartyShare, share_input, share_public, add_party, sub_party, scalar_mul_party,
    reveal, deal_triple, receive_triple, mul_party, a2b_party, reveal_bit_to_all,
};
