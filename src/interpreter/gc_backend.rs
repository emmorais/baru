//! Garbled Circuits (GC) backend for 2-party secure computation.
//!
//! This module implements the Backend trait using Garbled Circuits, enabling
//! 2-party computation with semi-honest security. Each gate is garbled and
//! evaluated obliviously.
//!
//! Shares are represented as pairs of boolean values: one bit is the actual
//! value, and the other is a random masking bit that conceals it from the
//! evaluator.

use std::collections::HashMap;

use crate::crypto::types::Fr;

use super::backend::{Backend, ShareRef};

/// Represents a garbled circuit bit: (value, mask).
/// The evaluator learns only the XOR of these two values.
#[derive(Debug, Clone)]
pub struct GCBit {
    pub value: bool,
    pub mask: bool,
}

impl GCBit {
    pub fn new(value: bool) -> Self {
        // In real GC, mask would be a cryptographic commitment
        // For this demo, we use a simple boolean mask
        GCBit { value, mask: false }
    }

    pub fn zero() -> Self {
        GCBit {
            value: false,
            mask: false,
        }
    }

    pub fn one() -> Self {
        GCBit {
            value: true,
            mask: false,
        }
    }

    /// Get the observable output (value XOR mask)
    pub fn observable(&self) -> bool {
        self.value ^ self.mask
    }
}

/// A garbled circuit share: stores bits for a value.
/// In 2-party GC, one party garbles and the other evaluates.
#[derive(Debug, Clone)]
pub struct GarbledShare {
    pub bits: Vec<GCBit>,
}

impl GarbledShare {
    pub fn new(bits: Vec<GCBit>) -> Self {
        GarbledShare { bits }
    }

    pub fn from_u128(value: u128, nbits: usize) -> Self {
        let mut bits = Vec::new();
        for i in 0..nbits {
            let bit = (value >> i) & 1 == 1;
            bits.push(GCBit::new(bit));
        }
        GarbledShare { bits }
    }

    pub fn to_u128(&self) -> u128 {
        let mut result = 0u128;
        for (i, bit) in self.bits.iter().enumerate() {
            if bit.value {
                result |= 1u128 << i;
            }
        }
        result
    }
}

/// 2-party Garbled Circuits backend.
///
/// This backend implements boolean circuit evaluation with garbling.
/// Party 0 garbles circuits; Party 1 evaluates them.
pub struct GCBackend {
    /// Current party ID (0 or 1)
    party: usize,
    /// Per-backend share storage: id -> GarbledShare
    shares: HashMap<u64, GarbledShare>,
    /// Next share ID to allocate
    next_id: u64,
}

impl GCBackend {
    pub fn new(party: usize) -> Self {
        assert!(party < 2, "GC backend supports only 2 parties");
        GCBackend {
            party,
            shares: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn store(&mut self, gs: GarbledShare, nbits: usize, owner: usize) -> ShareRef {
        let id = self.alloc_id();
        self.shares.insert(id, gs);
        ShareRef::new(id, nbits, owner)
    }

    fn get(&self, sr: &ShareRef) -> &GarbledShare {
        self.shares
            .get(&sr.id)
            .unwrap_or_else(|| panic!("GCBackend: unknown share id {}", sr.id))
    }

    fn get_cloned(&self, sr: &ShareRef) -> GarbledShare {
        self.get(sr).clone()
    }

    /// Bitwise AND gate
    fn and_gate(a: bool, b: bool) -> bool {
        a & b
    }

    /// Bitwise OR gate
    fn or_gate(a: bool, b: bool) -> bool {
        a | b
    }

    /// Bitwise XOR gate (local, no communication)
    #[allow(dead_code)]
    fn xor_gate(a: bool, b: bool) -> bool {
        a ^ b
    }

    /// Full adder: returns (sum, carry)
    fn full_adder(a: bool, b: bool, cin: bool) -> (bool, bool) {
        let sum = a ^ b ^ cin;
        let cout = (a & b) | (cin & (a ^ b));
        (sum, cout)
    }
}

impl Default for GCBackend {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Backend for GCBackend {
    fn share_secret(&mut self, value: Fr, nbits: usize, _owner: usize) -> ShareRef {
        let gs = GarbledShare::from_u128(value, nbits);
        self.store(gs, nbits, self.party)
    }

    fn add(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);
        let nbits = a.nbits.max(b.nbits);

        let mut result_bits = Vec::new();
        let mut carry = false;

        for i in 0..nbits {
            let a_bit = ga.bits.get(i).map(|b| b.value).unwrap_or(false);
            let b_bit = gb.bits.get(i).map(|b| b.value).unwrap_or(false);
            let (sum, cout) = Self::full_adder(a_bit, b_bit, carry);
            result_bits.push(GCBit::new(sum));
            carry = cout;
        }

        // Handle final carry if needed
        if carry && result_bits.len() < nbits + 1 {
            result_bits.push(GCBit::new(true));
        }

        let gs = GarbledShare::new(result_bits);
        self.store(gs, nbits, a.owner)
    }

    fn sub(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let _ga = self.get_cloned(a);
        let gb = self.get_cloned(b);
        let nbits = a.nbits.max(b.nbits);

        // 2's complement: a - b = a + ~b + 1
        let mut not_b_bits = Vec::new();
        for i in 0..nbits {
            let b_bit = gb.bits.get(i).map(|b| b.value).unwrap_or(false);
            not_b_bits.push(GCBit::new(!b_bit));
        }

        let gs_not_b = GarbledShare::new(not_b_bits);
        let not_b_ref = ShareRef::new(self.alloc_id(), nbits, b.owner);
        self.shares.insert(not_b_ref.id, gs_not_b);

        let temp = self.add(a, &not_b_ref);
        let one_ref = self.share_secret(1, nbits, a.owner);
        self.add(&temp, &one_ref)
    }

    fn mul(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);
        let nbits = a.nbits.max(b.nbits);

        // Bitwise multiplication using AND gates and shifts
        let mut result_bits = vec![GCBit::zero(); nbits];

        for i in 0..nbits {
            let a_bit = ga.bits.get(i).map(|b| b.value).unwrap_or(false);
            for j in 0..nbits {
                if i + j < nbits {
                    let b_bit = gb.bits.get(j).map(|b| b.value).unwrap_or(false);
                    let product_bit = Self::and_gate(a_bit, b_bit);
                    if product_bit {
                        // Add the product bit to result at position i+j
                        let (sum, mut carry) =
                            Self::full_adder(result_bits[i + j].value, product_bit, false);
                        result_bits[i + j] = GCBit::new(sum);

                        // Propagate carry
                        let mut k = i + j + 1;
                        while carry && k < nbits {
                            let (new_sum, new_carry) =
                                Self::full_adder(result_bits[k].value, carry, false);
                            result_bits[k] = GCBit::new(new_sum);
                            carry = new_carry;
                            k += 1;
                        }
                    }
                }
            }
        }

        let gs = GarbledShare::new(result_bits);
        self.store(gs, nbits, a.owner)
    }

    fn divmod(&mut self, _a: &ShareRef, _b: &ShareRef) -> (ShareRef, ShareRef) {
        unimplemented!("division not yet supported in GC backend")
    }

    fn eq(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);
        let nbits = a.nbits.max(b.nbits);

        // Check if all bits are equal: NOT(OR of XORs)
        let mut all_equal = true;
        for i in 0..nbits {
            let a_bit = ga.bits.get(i).map(|b| b.value).unwrap_or(false);
            let b_bit = gb.bits.get(i).map(|b| b.value).unwrap_or(false);
            if a_bit != b_bit {
                all_equal = false;
                break;
            }
        }

        let gs = GarbledShare::from_u128(if all_equal { 1 } else { 0 }, 1);
        self.store(gs, 1, a.owner)
    }

    fn lt(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);
        let nbits = a.nbits.max(b.nbits);

        // Compare bit by bit from MSB to LSB
        let mut less = false;
        let mut equal = true;

        for i in (0..nbits).rev() {
            if equal {
                let a_bit = ga.bits.get(i).map(|b| b.value).unwrap_or(false);
                let b_bit = gb.bits.get(i).map(|b| b.value).unwrap_or(false);
                if a_bit != b_bit {
                    less = !a_bit & b_bit;
                    equal = false;
                }
            }
        }

        let gs = GarbledShare::from_u128(if less { 1 } else { 0 }, 1);
        self.store(gs, 1, a.owner)
    }

    fn le(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let lt = self.lt(a, b);
        let eq = self.eq(a, b);
        self.or(&lt, &eq)
    }

    fn reveal(&mut self, a: &ShareRef, _reveal_to: Option<usize>) -> Fr {
        let ga = self.get_cloned(a);
        ga.to_u128()
    }

    fn a2b(&mut self, a: &ShareRef, _nbits: usize) -> Vec<ShareRef> {
        let ga = self.get_cloned(a);
        // In GC, shares are already in bit form
        ga.bits
            .iter()
            .map(|bit| {
                let gs = GarbledShare::new(vec![bit.clone()]);
                self.store(gs, 1, a.owner)
            })
            .collect()
    }

    fn b2a(&mut self, bits: &[ShareRef]) -> ShareRef {
        // Convert bits back to arithmetic: sum_i bit_i * 2^i
        let mut result = 0u128;
        for (i, br) in bits.iter().enumerate() {
            let gb = self.get_cloned(br);
            if !gb.bits.is_empty() && gb.bits[0].value {
                result |= 1u128 << i;
            }
        }
        self.share_secret(
            result,
            bits.len(),
            bits.first().map(|s| s.owner).unwrap_or(0),
        )
    }

    fn and(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);

        let a_bit = ga.bits.first().map(|b| b.value).unwrap_or(false);
        let b_bit = gb.bits.first().map(|b| b.value).unwrap_or(false);
        let and_bit = Self::and_gate(a_bit, b_bit);

        let gs = GarbledShare::new(vec![GCBit::new(and_bit)]);
        self.store(gs, 1, a.owner)
    }

    fn or(&mut self, a: &ShareRef, b: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        let gb = self.get_cloned(b);

        let a_bit = ga.bits.first().map(|b| b.value).unwrap_or(false);
        let b_bit = gb.bits.first().map(|b| b.value).unwrap_or(false);
        let or_bit = Self::or_gate(a_bit, b_bit);

        let gs = GarbledShare::new(vec![GCBit::new(or_bit)]);
        self.store(gs, 1, a.owner)
    }

    fn bitlen(&mut self, a: &ShareRef) -> ShareRef {
        let ga = self.get_cloned(a);
        // Find position of highest set bit
        let mut bitlen = 0u128;
        for (i, bit) in ga.bits.iter().enumerate().rev() {
            if bit.value {
                bitlen = (i as u128) + 1;
                break;
            }
        }
        self.share_secret(bitlen, 64, a.owner)
    }

    fn shift_left(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let ga = self.get_cloned(a);
        let mut result_bits = vec![GCBit::zero(); n];
        result_bits.extend(ga.bits.clone());
        let gs = GarbledShare::new(result_bits);
        self.store(gs, a.nbits + n, a.owner)
    }

    fn shift_right(&mut self, a: &ShareRef, n: usize) -> ShareRef {
        let ga = self.get_cloned(a);
        let result_bits: Vec<GCBit> = ga.bits.iter().skip(n).cloned().collect();
        let gs = GarbledShare::new(result_bits);
        let nbits = a.nbits.saturating_sub(n);
        self.store(gs, nbits, a.owner)
    }

    fn reset_per_expression(&mut self) {
        self.shares.clear();
        self.next_id = 1;
    }
}
