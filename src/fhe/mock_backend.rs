//! Mock FHE backend — plaintext arithmetic behind the `FheBackend` interface.
//!
//! This backend performs all operations on plain `u64` values, making it
//! suitable for:
//!   - Unit tests
//!   - REPL development before a real FHE library is integrated
//!   - Documenting the expected semantics of each gate
//!
//! Security guarantee: NONE.  Do not use in production.

use super::backend::FheBackend;

/// Mock backend identifier.
const BACKEND_ID: &str = "mock-plaintext";

/// Key bytes for the mock backend: a single 0x01 byte as a sentinel.
const MOCK_KEY: &[u8] = &[0x01];

/// Mock FHE backend.  Ciphertext = plaintext `u64`.
#[derive(Debug, Default, Clone)]
pub struct MockFheBackend;

impl FheBackend for MockFheBackend {
    type Ct = u64;

    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    // ── Key management ───────────────────────────────────────────────────────

    fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        // Mock: both client and server key are the sentinel byte
        (MOCK_KEY.to_vec(), MOCK_KEY.to_vec())
    }

    fn encrypt(&self, _client_key_bytes: &[u8], value: u64) -> Self::Ct {
        value // plaintext "encryption"
    }

    fn decrypt(&self, _client_key_bytes: &[u8], ct: &Self::Ct) -> u64 {
        *ct
    }

    fn serialize_ct(&self, ct: &Self::Ct) -> Vec<u8> {
        ct.to_le_bytes().to_vec()
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> Self::Ct {
        let arr: [u8; 8] = bytes.try_into().expect("MockFheBackend: expected 8 bytes");
        u64::from_le_bytes(arr)
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a.wrapping_add(*b)
    }

    fn sub(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a.wrapping_sub(*b)
    }

    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        // Use u128 to avoid truncation before the scale-back in fixed_mul
        ((*a as u128).wrapping_mul(*b as u128)) as u64
    }

    fn add_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        a.wrapping_add(b as u64)
    }

    fn mul_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        ((*a as i64).wrapping_mul(b)) as u64
    }

    // ── Comparisons ──────────────────────────────────────────────────────────

    fn cmpeq(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        if a == b {
            1
        } else {
            0
        }
    }

    fn cmplt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        if (*a as i64) < (*b as i64) {
            1
        } else {
            0
        }
    }

    // ── Bitwise ──────────────────────────────────────────────────────────────

    fn band(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a & b
    }

    fn bor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a | b
    }

    fn bxor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a ^ b
    }

    fn bnot(&self, a: &Self::Ct) -> Self::Ct {
        !a
    }

    // ── Shifts ───────────────────────────────────────────────────────────────

    fn shl(&self, a: &Self::Ct, amount: u32) -> Self::Ct {
        a.wrapping_shl(amount)
    }

    fn shr(&self, a: &Self::Ct, amount: u32) -> Self::Ct {
        a.wrapping_shr(amount)
    }

    // ── Mux ──────────────────────────────────────────────────────────────────

    fn mux(&self, cond: &Self::Ct, t: &Self::Ct, e: &Self::Ct) -> Self::Ct {
        if *cond != 0 {
            *t
        } else {
            *e
        }
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn divmod(&self, a: &Self::Ct, b: &Self::Ct) -> (Self::Ct, Self::Ct) {
        assert!(*b != 0, "MockFheBackend: division by zero");
        (a / b, a % b)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_encrypt_decrypt() {
        let be = MockFheBackend;
        let (ck, _sk) = be.keygen();
        let ct = be.encrypt(&ck, 42);
        assert_eq!(be.decrypt(&ck, &ct), 42);
    }

    #[test]
    fn test_mock_arithmetic() {
        let be = MockFheBackend;
        let (ck, _sk) = be.keygen();
        let a = be.encrypt(&ck, 10);
        let b = be.encrypt(&ck, 3);
        assert_eq!(be.decrypt(&ck, &be.add(&a, &b)), 13);
        assert_eq!(be.decrypt(&ck, &be.sub(&a, &b)), 7);
        assert_eq!(be.decrypt(&ck, &be.mul(&a, &b)), 30);
    }

    #[test]
    fn test_mock_mux() {
        let be = MockFheBackend;
        let (ck, _sk) = be.keygen();
        let one = be.encrypt(&ck, 1);
        let zero = be.encrypt(&ck, 0);
        let t = be.encrypt(&ck, 100);
        let e = be.encrypt(&ck, 200);
        assert_eq!(be.decrypt(&ck, &be.mux(&one, &t, &e)), 100);
        assert_eq!(be.decrypt(&ck, &be.mux(&zero, &t, &e)), 200);
    }

    #[test]
    fn test_mock_fixed_mul() {
        use crate::interpreter::ast::FIXED_SCALE;
        let be = MockFheBackend;
        let (ck, _sk) = be.keygen();
        // 2.0 * 3.0 = 6.0 in Q16.16
        let a = be.encrypt(&ck, (2 * FIXED_SCALE) as u64);
        let b = be.encrypt(&ck, (3 * FIXED_SCALE) as u64);
        let result = be.fixed_mul(&a, &b);
        assert_eq!(be.decrypt(&ck, &result), (6 * FIXED_SCALE) as u64);
    }

    #[test]
    fn test_mock_serialize_roundtrip() {
        let be = MockFheBackend;
        let ct: u64 = 0xDEAD_BEEF_1234_5678;
        let bytes = be.serialize_ct(&ct);
        assert_eq!(be.deserialize_ct(&bytes), ct);
    }
}
