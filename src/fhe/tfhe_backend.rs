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
fn bool_to_uint64(b: FheBool) -> FheUint64 {
    b.cast_into()
}

/// Concrete TFHE-rs v1.6 FHE backend.
///
/// `Ct` is the native `FheUint64` — ciphertexts stay in memory between gates
/// and are only (de)serialised at the runtime boundary (`serialize_ct` /
/// `deserialize_ct`), not once per operation.
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

    pub fn encrypt_with_ck(ck_bytes: &[u8], value: u64) -> FheUint64 {
        let ck = decode_client_key(ck_bytes);
        FheUint64::encrypt(value, &ck)
    }

    pub fn decrypt_with_ck(ck_bytes: &[u8], ct: &FheUint64) -> u64 {
        ct.decrypt(&decode_client_key(ck_bytes))
    }
}

impl FheBackend for TfheBackend {
    type Ct = FheUint64;

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
        FheUint64::try_encrypt_trivial(value as u64)
            .expect("TfheBackend: trivial encryption failed")
    }

    fn decrypt(&self, ck: &[u8], ct: &Self::Ct) -> u64 {
        Self::decrypt_with_ck(ck, ct)
    }

    fn serialize_ct(&self, ct: &Self::Ct) -> Vec<u8> {
        bincode::serialize(ct).expect("TfheBackend: serialize FheUint64")
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> Self::Ct {
        bincode::deserialize(bytes).expect("TfheBackend: deserialize FheUint64")
    }

    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a + b
    }

    fn sub(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a - b
    }

    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        a * b
    }

    fn add_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        a + b as u64
    }

    fn mul_plain(&self, a: &Self::Ct, b: i64) -> Self::Ct {
        a * b as u64
    }

    fn cmpeq(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        bool_to_uint64(a.eq(b))
    }

    fn cmplt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        bool_to_uint64(a.lt(b))
    }

    fn cmpslt(&self, a: &Self::Ct, b: &Self::Ct) -> Self::Ct {
        // Signed compare over FheUint64 in two's complement: flipping the sign
        // bit of both operands turns signed order into unsigned order.
        const SIGN_BIT: u64 = 1u64 << 63;
        let fa = a ^ SIGN_BIT;
        let fb = b ^ SIGN_BIT;
        bool_to_uint64(fa.lt(&fb))
    }

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
        !a.clone()
    }

    fn shl(&self, a: &Self::Ct, n: u32) -> Self::Ct {
        a << n
    }

    fn shr(&self, a: &Self::Ct, n: u32) -> Self::Ct {
        a >> n
    }

    fn mux(&self, cond: &Self::Ct, t: &Self::Ct, e: &Self::Ct) -> Self::Ct {
        // Treat any nonzero condition as true (same semantics as the mock backend).
        let zero: FheUint64 = self.encrypt_trivial(0);
        cond.ne(&zero).if_then_else(t, e)
    }

    fn divmod(&self, a: &Self::Ct, b: &Self::Ct) -> (Self::Ct, Self::Ct) {
        a.div_rem(b)
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

    #[test]
    fn test_tfhe_divmod() {
        let (b, ck) = setup();
        let (q, r) = b.divmod(&b.encrypt(&ck, 17u64), &b.encrypt(&ck, 5u64));
        assert_eq!(b.decrypt(&ck, &q), 3);
        assert_eq!(b.decrypt(&ck, &r), 2);
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
}
