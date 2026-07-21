//! CKKS backends for real-arithmetic circuits.
//!
//! `CkksBackend` wraps Microsoft SEAL (via the vendored `sealy` bindings,
//! see `third_party/sealy/BARU_PATCHES.md`).  `MockCkksBackend` computes the
//! same operations exactly on plain `f64` vectors — the development /
//! differential-testing twin, mirroring `MockFheBackend` for TFHE.
//!
//! ── Scale discipline ─────────────────────────────────────────────────────
//! CKKS ciphertexts carry an encoding scale that *multiplies* under
//! homomorphic multiplication, and SEAL requires exact scale equality for
//! addition.  Instead of rescaling (which drops modulus levels and forces
//! level alignment everywhere), this backend keeps every ciphertext at the
//! top modulus level and tracks a small `depth` per ciphertext:
//!
//! `scale(depth) = 2^(40 · (depth + 1))`
//!
//! Every multiplication (cipher or plain) increments depth; additions lift
//! the shallower operand by multiplying with an encoded 1.0.  Powers of two
//! are exact in `f64`, so equal depths mean bit-identical scales and SEAL
//! adds never mismatch.  With the default modulus chain (60,40,40,60 bits;
//! 140 data bits) this supports multiplicative depth ≤ 2, plenty for the
//! linear/NN workloads Baru targets; deeper circuits fail loudly inside SEAL.

use std::cell::RefCell;

use sealy::{
    CKKSEncoder, CKKSEncryptionParametersBuilder, CKKSEvaluator, Ciphertext,
    CoefficientModulusFactory, Context, Decryptor, DegreeType, Encryptor, Evaluator, FromBytes,
    GaloisKey, KeyGenerator, PublicKey, RelinearizationKey, SecretKey, SecurityLevel, ToBytes,
};

use crate::fhe::backend::RealFheBackend;

const SCALE_BITS: i32 = 40;

/// A SEAL CKKS ciphertext plus its multiplicative depth (scale exponent).
#[derive(Clone)]
pub struct CkksCt {
    pub ct: Ciphertext,
    pub depth: u8,
}

/// Serialised form of `CkksCt` (bincode).
#[derive(serde::Serialize, serde::Deserialize)]
struct CkksCtWire {
    depth: u8,
    bytes: Vec<u8>,
}

/// Client key bundle: secret key (decrypt) + public key (encrypt).
#[derive(serde::Serialize, serde::Deserialize)]
struct ClientKeys {
    secret: Vec<u8>,
    public: Vec<u8>,
}

/// Server key bundle: public (constant promotion), galois (rotations),
/// relinearization (cipher×cipher products).
#[derive(serde::Serialize, serde::Deserialize)]
struct ServerKeysWire {
    public: Vec<u8>,
    galois: Vec<u8>,
    relin: Vec<u8>,
}

struct ServerKeys {
    public: PublicKey,
    galois: GaloisKey,
    relin: RelinearizationKey,
}

pub struct CkksBackend {
    ctx: Context,
    evaluator: CKKSEvaluator,
    slots: usize,
    /// Evaluation keys, present only when constructed via `with_server_keys`.
    server: Option<ServerKeys>,
    /// Lazily constructed encoders, one per depth (scale 2^(40·(d+1))).
    encoders: RefCell<Vec<Option<CKKSEncoder>>>,
}

impl CkksBackend {
    /// Fixed parameters: N=8192 (4096 slots), moduli (60,40,40,60) bits,
    /// 128-bit security.  ~140 bits of data modulus → depth ≤ 2.
    fn context() -> Context {
        let degree = DegreeType::D8192;
        let params = CKKSEncryptionParametersBuilder::new()
            .set_poly_modulus_degree(degree)
            .set_coefficient_modulus(
                CoefficientModulusFactory::build(degree, &[60, SCALE_BITS, SCALE_BITS, 60])
                    .expect("CKKS modulus chain"),
            )
            .build()
            .expect("CKKS parameters");
        Context::new(&params, true, SecurityLevel::TC128).expect("CKKS context")
    }

    /// A backend able to keygen / encrypt / decrypt but not evaluate.
    pub fn new() -> Self {
        let ctx = Self::context();
        let evaluator = CKKSEvaluator::new(&ctx).expect("CKKS evaluator");
        let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(SCALE_BITS)).expect("CKKS encoder");
        let slots = encoder.get_slot_count();
        CkksBackend {
            ctx,
            evaluator,
            slots,
            server: None,
            encoders: RefCell::new(vec![Some(encoder)]),
        }
    }

    /// A backend with evaluation keys installed (required by `fhe-eval`).
    pub fn with_server_keys(server_key_bytes: &[u8]) -> Self {
        let mut be = Self::new();
        let wire: ServerKeysWire =
            bincode::deserialize(server_key_bytes).expect("CKKS server key bundle");
        be.server = Some(ServerKeys {
            public: PublicKey::from_bytes(&be.ctx, &wire.public).expect("CKKS public key"),
            galois: GaloisKey::from_bytes(&be.ctx, &wire.galois).expect("CKKS galois keys"),
            relin: RelinearizationKey::from_bytes(&be.ctx, &wire.relin).expect("CKKS relin keys"),
        });
        be
    }

    pub fn slot_count(&self) -> usize {
        self.slots
    }

    fn server(&self) -> &ServerKeys {
        self.server.as_ref().expect(
            "CKKS backend has no evaluation keys — construct it with with_server_keys \
             (fhe-eval passes the server key bundle from fhe-keygen-ckks)",
        )
    }

    /// Encoder for the scale at `depth` (created on first use).
    fn with_encoder<R>(&self, depth: u8, f: impl FnOnce(&CKKSEncoder) -> R) -> R {
        let mut cache = self.encoders.borrow_mut();
        let d = depth as usize;
        if cache.len() <= d {
            cache.resize_with(d + 1, || None);
        }
        if cache[d].is_none() {
            let scale = 2.0f64.powi(SCALE_BITS * (depth as i32 + 1));
            cache[d] = Some(CKKSEncoder::new(&self.ctx, scale).expect("CKKS encoder"));
        }
        f(cache[d].as_ref().unwrap())
    }

    fn encode_at(&self, values: &[f64], depth: u8) -> sealy::Plaintext {
        let mut padded = values.to_vec();
        padded.resize(self.slots, 0.0);
        self.with_encoder(depth, |enc| enc.encode_f64(&padded).expect("CKKS encode"))
    }

    /// Raise `a` to `target` depth by multiplying with an encoded 1.0
    /// (exact scale bump of 2^40 per step; stays at the top modulus level).
    fn lift(&self, a: &CkksCt, target: u8) -> CkksCt {
        let mut cur = a.clone();
        while cur.depth < target {
            let one = self.encode_at(&vec![1.0; self.slots], 0);
            cur = CkksCt {
                ct: self
                    .evaluator
                    .multiply_plain(&cur.ct, &one)
                    .expect("CKKS lift"),
                depth: cur.depth + 1,
            };
        }
        cur
    }

    /// Align two ciphertexts to a common depth for addition/subtraction.
    fn aligned(&self, a: &CkksCt, b: &CkksCt) -> (CkksCt, CkksCt, u8) {
        let d = a.depth.max(b.depth);
        (self.lift(a, d), self.lift(b, d), d)
    }

    fn encrypt_with(&self, public: &PublicKey, values: &[f64]) -> CkksCt {
        let pt = self.encode_at(values, 0);
        let encryptor = Encryptor::with_public_key(&self.ctx, public).expect("CKKS encryptor");
        CkksCt {
            ct: encryptor.encrypt(&pt).expect("CKKS encrypt"),
            depth: 0,
        }
    }

    fn client_keys(&self, client_key_bytes: &[u8]) -> ClientKeys {
        bincode::deserialize(client_key_bytes).expect("CKKS client key bundle")
    }
}

impl Default for CkksBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RealFheBackend for CkksBackend {
    type Ct = CkksCt;

    fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        let keygen = KeyGenerator::new(&self.ctx).expect("CKKS keygen");
        let secret = keygen.secret_key();
        let public = keygen.create_public_key();
        let galois = keygen.create_galois_keys().expect("CKKS galois keygen");
        let relin = keygen
            .create_relinearization_keys()
            .expect("CKKS relin keygen");
        let client = bincode::serialize(&ClientKeys {
            secret: secret.as_bytes().expect("sk ser"),
            public: public.as_bytes().expect("pk ser"),
        })
        .unwrap();
        let server = bincode::serialize(&ServerKeysWire {
            public: public.as_bytes().expect("pk ser"),
            galois: galois.as_bytes().expect("galois ser"),
            relin: relin.as_bytes().expect("relin ser"),
        })
        .unwrap();
        (client, server)
    }

    fn encrypt(&self, client_key_bytes: &[u8], value: f64) -> CkksCt {
        self.encrypt_vec(client_key_bytes, &[value])
    }

    fn encrypt_vec(&self, client_key_bytes: &[u8], values: &[f64]) -> CkksCt {
        assert!(
            values.len() <= self.slots,
            "CKKS encrypt: {} values exceed the {} available slots",
            values.len(),
            self.slots
        );
        let keys = self.client_keys(client_key_bytes);
        let public = PublicKey::from_bytes(&self.ctx, &keys.public).expect("CKKS public key");
        self.encrypt_with(&public, values)
    }

    fn decrypt(&self, client_key_bytes: &[u8], ct: &CkksCt) -> f64 {
        self.decrypt_vec(client_key_bytes, ct, 1)[0]
    }

    fn decrypt_vec(&self, client_key_bytes: &[u8], ct: &CkksCt, n: usize) -> Vec<f64> {
        let keys = self.client_keys(client_key_bytes);
        let secret = SecretKey::from_bytes(&self.ctx, &keys.secret).expect("CKKS secret key");
        let decryptor = Decryptor::new(&self.ctx, &secret).expect("CKKS decryptor");
        let pt = decryptor.decrypt(&ct.ct).expect("CKKS decrypt");
        // decode_f64 reads the scale stored in the plaintext, so any encoder
        // instance decodes correctly regardless of depth.
        let vals = self.with_encoder(0, |enc| enc.decode_f64(&pt).expect("CKKS decode"));
        vals.into_iter().take(n).collect()
    }

    fn encrypt_public(&self, value: f64) -> CkksCt {
        self.encrypt_with(&self.server().public, &[value])
    }

    fn serialize_ct(&self, ct: &CkksCt) -> Vec<u8> {
        bincode::serialize(&CkksCtWire {
            depth: ct.depth,
            bytes: ct.ct.as_bytes().expect("CKKS ct ser"),
        })
        .unwrap()
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> CkksCt {
        let wire: CkksCtWire = bincode::deserialize(bytes).expect("CKKS ct wire");
        CkksCt {
            ct: Ciphertext::from_bytes(&self.ctx, &wire.bytes).expect("CKKS ct deser"),
            depth: wire.depth,
        }
    }

    fn backend_id(&self) -> &'static str {
        "seal-ckks"
    }

    fn add(&self, a: &CkksCt, b: &CkksCt) -> CkksCt {
        let (a, b, d) = self.aligned(a, b);
        CkksCt {
            ct: self.evaluator.add(&a.ct, &b.ct).expect("CKKS add"),
            depth: d,
        }
    }

    fn sub(&self, a: &CkksCt, b: &CkksCt) -> CkksCt {
        let (a, b, d) = self.aligned(a, b);
        CkksCt {
            ct: self.evaluator.sub(&a.ct, &b.ct).expect("CKKS sub"),
            depth: d,
        }
    }

    fn mul(&self, a: &CkksCt, b: &CkksCt) -> CkksCt {
        // Scales multiply, so no alignment is needed; relinearize to keep
        // the ciphertext at size 2 for later rotations/multiplications.
        let prod = self.evaluator.multiply(&a.ct, &b.ct).expect("CKKS mul");
        let prod = self
            .evaluator
            .relinearize(&prod, &self.server().relin)
            .expect("CKKS relinearize");
        CkksCt {
            ct: prod,
            depth: a.depth + b.depth + 1,
        }
    }

    fn add_plain(&self, a: &CkksCt, b: f64) -> CkksCt {
        let pt = self.encode_at(&vec![b; self.slots], a.depth);
        CkksCt {
            ct: self
                .evaluator
                .add_plain(&a.ct, &pt)
                .expect("CKKS add_plain"),
            depth: a.depth,
        }
    }

    fn mul_plain(&self, a: &CkksCt, b: f64) -> CkksCt {
        if b == 0.0 {
            // SEAL refuses multiply_plain by an all-zero plaintext (the
            // result would be a transparent ciphertext); return a genuine
            // encryption of zero at the right depth instead.
            return self.lift(&self.encrypt_public(0.0), a.depth + 1);
        }
        let pt = self.encode_at(&vec![b; self.slots], 0);
        CkksCt {
            ct: self
                .evaluator
                .multiply_plain(&a.ct, &pt)
                .expect("CKKS mul_plain"),
            depth: a.depth + 1,
        }
    }

    fn dot_plain(&self, a: &CkksCt, weights: &[f64]) -> CkksCt {
        assert!(
            weights.len() <= self.slots,
            "CKKS dot: {} weights exceed the {} available slots",
            weights.len(),
            self.slots
        );
        if weights.iter().all(|w| *w == 0.0) {
            return self.lift(&self.encrypt_public(0.0), a.depth + 1);
        }
        let wt = self.encode_at(weights, 0);
        let mut acc = self
            .evaluator
            .multiply_plain(&a.ct, &wt)
            .expect("CKKS dot mul_plain");
        // Rotate-and-add tree over all slots: after log2(slots) rounds,
        // slot 0 holds the sum of every product.
        let galois = &self.server().galois;
        let mut step = 1usize;
        while step < self.slots {
            let rotated = self
                .evaluator
                .rotate_vector(&acc, step as i32, galois)
                .expect("CKKS rotate");
            acc = self.evaluator.add(&acc, &rotated).expect("CKKS dot add");
            step *= 2;
        }
        CkksCt {
            ct: acc,
            depth: a.depth + 1,
        }
    }
}

// ─── Mock backend ─────────────────────────────────────────────────────────────

/// Plain-`f64` twin of `CkksBackend`: identical semantics, exact arithmetic,
/// no keys, no noise.  A ciphertext is just the slot vector.
pub struct MockCkksBackend;

impl RealFheBackend for MockCkksBackend {
    type Ct = Vec<f64>;

    fn keygen(&self) -> (Vec<u8>, Vec<u8>) {
        (vec![], vec![])
    }

    fn encrypt(&self, _ck: &[u8], value: f64) -> Vec<f64> {
        vec![value]
    }

    fn encrypt_vec(&self, _ck: &[u8], values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    fn decrypt(&self, _ck: &[u8], ct: &Vec<f64>) -> f64 {
        ct.first().copied().unwrap_or(0.0)
    }

    fn decrypt_vec(&self, _ck: &[u8], ct: &Vec<f64>, n: usize) -> Vec<f64> {
        let mut out = ct.clone();
        out.resize(n, 0.0);
        out.truncate(n);
        out
    }

    fn encrypt_public(&self, value: f64) -> Vec<f64> {
        vec![value]
    }

    fn serialize_ct(&self, ct: &Vec<f64>) -> Vec<u8> {
        bincode::serialize(ct).unwrap()
    }

    fn deserialize_ct(&self, bytes: &[u8]) -> Vec<f64> {
        bincode::deserialize(bytes).expect("mock CKKS ct")
    }

    fn backend_id(&self) -> &'static str {
        "ckks-mock"
    }

    fn add(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) + b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn sub(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) - b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn mul(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| a.get(i).unwrap_or(&0.0) * b.get(i).unwrap_or(&0.0))
            .collect()
    }

    fn add_plain(&self, a: &Vec<f64>, b: f64) -> Vec<f64> {
        // SEAL's add_plain adds the constant to every slot; mirror that.
        a.iter().map(|x| x + b).collect()
    }

    fn mul_plain(&self, a: &Vec<f64>, b: f64) -> Vec<f64> {
        a.iter().map(|x| x * b).collect()
    }

    fn dot_plain(&self, a: &Vec<f64>, weights: &[f64]) -> Vec<f64> {
        vec![a
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()]
    }
}
