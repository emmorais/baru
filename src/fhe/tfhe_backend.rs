use tfhe::prelude::*;
use tfhe::{
    generate_keys, set_server_key, ClientKey, ConfigBuilder, FheBool, FheUint64, ServerKey,
};

use super::backend::FheBackend;

const BACKEND_ID: &str = "tfhe-rs-v1.6";

fn encode_client_key(ck: &ClientKey) -> Vec<u8> {
    bincode::serialize(ck).expect("TfheBackend: serialize ClientKey")
}
fn decode_client_key(bytes: &[u8]) -> ClientKey {
    bincode::deserialize(bytes).expect("TfheBackend: deserialize ClientKey")
}
fn encode_server_key(sk: &ServerKey) -> Vec<u8> {
    bincode::serialize(sk).expect("TfheBackend: serialize ServerKey")
}
fn decode_server_key(bytes: &[u8]) -> ServerKey {
    bincode::deserialize(bytes).expect("TfheBackend: deserialize ServerKey")
}
fn encode_ct(ct: &FheUint64) -> Vec<u8> {
    bincode::serialize(ct).expect("TfheBackend: serialize FheUint64")
}
fn decode_ct(bytes: &[u8]) -> FheUint64 {
    bincode::deserialize(bytes).expect("TfheBackend: deserialize FheUint64")
}
fn bool_to_uint64(b: FheBool) -> FheUint64 {
    b.cast_into()
}

/// Serialisable wrapper around an `FheUint64` ciphertext.
#[derive(Debug, Clone)]
pub struct TfheCt {
    pub bytes: Vec<u8>,
}

/// Concrete TFHE-rs v1.6 FHE backend.
#[derive(Debug, Default)]
pub struct TfheBackend;

impl TfheBackend {
    pub fn new() -> Self {
        TfheBackend
    }

    /// Install a server key (raw bytes) into TFHE-rs global thread-local.
    /// Call once before any homomorphic operation.
    pub fn install_server_key(sk_bytes: &[u8]) {
        set_server_key(decode_server_key(sk_bytes));
    }

    pub fn encrypt_with_ck(ck_bytes: &[u8], value: u64) -> TfheCt {
        let ck = decode_client_key(ck_bytes);
        TfheCt {
            bytes: encode_ct(&FheUint64::encrypt(value, &ck)),
        }
    }

    pub fn decrypt_with_ck(ck_bytes: &[u8], ct: &TfheCt) -> u64 {
        decode_ct(&ct.bytes).decrypt(&decode_client_key(ck_bytes))
    }
}

impl FheBackend for TfheBackend {
    type Ct = TfheCt;

    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let (ck, sk) = generate_keys(ConfigBuilder::default().build());
        (encode_client_key(&ck), encode_server_key(&sk))
    }

    fn encrypt(&self, ck: &[u8], v: u64) -> Self::Ct {
        Self::encrypt_with_ck(ck, v)
    }

    fn encrypt_trivial(&self, value: i64) -> Self::Ct {
        let ct = FheUint64::try_encrypt_trivial(value as u64)
            .expect("TfheBackend: trivial encryption failed");
        TfheCt {
            bytes: encode_ct(&ct),
        }
    }

    fn decrypt(&self, ck: &[u8], ct: &Self::Ct) -> u64 {
        Self::decrypt_with_ck(ck, ct)
    }

    fn serialize_ct(&self, ct: &Self::Ct) -> Vec<u8> {
        ct.bytes.clone()
    }

    fn deserialize_ct(&self, b: &[u8]) -> Self::Ct {
        TfheCt { bytes: b.to_vec() }
    }

    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) + decode_ct(&b.bytes))),
        }
    }

    fn sub(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) - decode_ct(&b.bytes))),
        }
    }

    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) * decode_ct(&b.bytes))),
        }
    }

    fn add_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) + b as u64)),
        }
    }

    fn mul_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) * b as u64)),
        }
    }

    fn cmpeq(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        let eq: FheBool = decode_ct(&a.bytes).eq(&decode_ct(&b.bytes));
        TfheCt {
            bytes: encode_ct(&bool_to_uint64(eq)),
        }
    }

    fn cmplt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        let lt: FheBool = decode_ct(&a.bytes).lt(&decode_ct(&b.bytes));
        TfheCt {
            bytes: encode_ct(&bool_to_uint64(lt)),
        }
    }

    fn cmpslt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        // Signed compare over FheUint64 in two's complement: flipping the sign
        // bit of both operands turns signed order into unsigned order.
        const SIGN_BIT: u64 = 1u64 << 63;
        let fa = decode_ct(&a.bytes) ^ SIGN_BIT;
        let fb = decode_ct(&b.bytes) ^ SIGN_BIT;
        let lt: FheBool = fa.lt(&fb);
        TfheCt {
            bytes: encode_ct(&bool_to_uint64(lt)),
        }
    }

    fn band(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) & decode_ct(&b.bytes))),
        }
    }

    fn bor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) | decode_ct(&b.bytes))),
        }
    }

    fn bxor(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) ^ decode_ct(&b.bytes))),
        }
    }

    fn bnot(&self, a: &Self::Ct) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(!decode_ct(&a.bytes))),
        }
    }

    fn shl(&self, a: &Self::Ct, n: u32) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) << n)),
        }
    }

    fn shr(&self, a: &Self::Ct, n: u32) -> Self::Ct {
        TfheCt {
            bytes: encode_ct(&(decode_ct(&a.bytes) >> n)),
        }
    }

    fn mux(&self, cond: &Self::Ct, t: &Self::Ct, e: &Self::Ct) -> Self::Ct {
        let fc = decode_ct(&cond.bytes);
        let ft = decode_ct(&t.bytes);
        let fe = decode_ct(&e.bytes);
        // Treat any nonzero condition as true (same semantics as the mock backend).
        let zero: FheUint64 = FheUint64::try_encrypt_trivial(0u64).expect("TfheBackend: trivial 0");
        let result = fc.ne(&zero).if_then_else(&ft, &fe);
        TfheCt {
            bytes: encode_ct(&result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TfheBackend, Vec<u8>) {
        let b = TfheBackend::new();
        let (ck, sk) = b.keygen();
        TfheBackend::install_server_key(&sk);
        (b, ck)
    }

    #[test]
    fn test_tfhe_encrypt_decrypt() {
        let (b, ck) = setup();
        assert_eq!(b.decrypt(&ck, &b.encrypt(&ck, 42)), 42);
    }

    #[test]
    fn test_tfhe_add() {
        let (b, ck) = setup();
        assert_eq!(
            b.decrypt(&ck, &b.add(&b.encrypt(&ck, 10), &b.encrypt(&ck, 32))),
            42
        );
    }

    #[test]
    fn test_tfhe_sub() {
        let (b, ck) = setup();
        assert_eq!(
            b.decrypt(&ck, &b.sub(&b.encrypt(&ck, 100), &b.encrypt(&ck, 58))),
            42
        );
    }

    #[test]
    fn test_tfhe_mul_plain() {
        let (b, ck) = setup();
        assert_eq!(b.decrypt(&ck, &b.mul_plain(&b.encrypt(&ck, 6), 7)), 42);
    }

    #[test]
    fn test_tfhe_shl_shr() {
        let (b, ck) = setup();
        let s = b.shl(&b.encrypt(&ck, 1u64), 5);
        assert_eq!(b.decrypt(&ck, &s), 32);
        assert_eq!(b.decrypt(&ck, &b.shr(&s, 5)), 1);
    }

    #[test]
    fn test_tfhe_fixed_mul() {
        let (b, ck) = setup();
        let scale = 65536u64;
        let r = b.fixed_mul(
            &b.encrypt(&ck, 2 * scale),
            &b.encrypt(&ck, scale + scale / 2),
        );
        assert_eq!(b.decrypt(&ck, &r), 3 * scale);
    }

    #[test]
    fn test_tfhe_mux_true() {
        let (b, ck) = setup();
        let r = b.mux(
            &b.encrypt(&ck, 1u64),
            &b.encrypt(&ck, 100u64),
            &b.encrypt(&ck, 200u64),
        );
        assert_eq!(b.decrypt(&ck, &r), 100);
    }

    #[test]
    fn test_tfhe_mux_false() {
        let (b, ck) = setup();
        let r = b.mux(
            &b.encrypt(&ck, 0u64),
            &b.encrypt(&ck, 100u64),
            &b.encrypt(&ck, 200u64),
        );
        assert_eq!(b.decrypt(&ck, &r), 200);
    }

    #[test]
    fn test_tfhe_cmpeq_true() {
        let (b, ck) = setup();
        assert_eq!(
            b.decrypt(&ck, &b.cmpeq(&b.encrypt(&ck, 7u64), &b.encrypt(&ck, 7u64))),
            1
        );
    }

    #[test]
    fn test_tfhe_cmpeq_false() {
        let (b, ck) = setup();
        assert_eq!(
            b.decrypt(&ck, &b.cmpeq(&b.encrypt(&ck, 7u64), &b.encrypt(&ck, 8u64))),
            0
        );
    }

    #[test]
    fn test_tfhe_cmplt() {
        let (b, ck) = setup();
        assert_eq!(
            b.decrypt(&ck, &b.cmplt(&b.encrypt(&ck, 3u64), &b.encrypt(&ck, 7u64))),
            1
        );
        assert_eq!(
            b.decrypt(&ck, &b.cmplt(&b.encrypt(&ck, 7u64), &b.encrypt(&ck, 3u64))),
            0
        );
    }

    #[test]
    fn test_tfhe_serialize_roundtrip() {
        let (b, ck) = setup();
        let ct = b.encrypt(&ck, 99u64);
        assert_eq!(b.decrypt(&ck, &b.deserialize_ct(&b.serialize_ct(&ct))), 99);
    }

    #[test]
    fn test_tfhe_encrypt_trivial_mixes_with_real_ct() {
        let (b, ck) = setup();
        // A trivial ciphertext is a public constant that must combine
        // correctly with ciphertexts under any client key.
        let zero = b.encrypt_trivial(0);
        let forty_two = b.encrypt(&ck, 42u64);
        let result = b.add(&zero, &forty_two);
        assert_eq!(b.decrypt(&ck, &result), 42);
    }

    #[test]
    fn test_tfhe_cmpslt_signed() {
        let (b, ck) = setup();
        let neg_four = b.encrypt(&ck, (-4i64) as u64);
        let zero = b.encrypt_trivial(0);
        // signed: -4 < 0  →  1
        assert_eq!(b.decrypt(&ck, &b.cmpslt(&neg_four, &zero)), 1);
        // signed: 0 < -4  →  0
        assert_eq!(b.decrypt(&ck, &b.cmpslt(&zero, &neg_four)), 0);
        // unsigned: -4 as u64 is huge, so 0 < -4  →  1
        assert_eq!(b.decrypt(&ck, &b.cmplt(&zero, &neg_four)), 1);
    }
}
