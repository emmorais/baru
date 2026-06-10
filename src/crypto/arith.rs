use rand::Rng;
use crate::crypto::types::{Share3, Fr};

/// Arithmetic helpers for Share3 values.
///
/// NOTE: these functions are intentionally simple, local-process simulations.
/// Real ABY3 multiplication is interactive and requires communication. These
/// implementations reconstruct-and-reshare (insecure) to provide a usable API
/// while the real protocol is implemented later.

/// Secret-share a plain value into a random Share3.
pub fn share_secret(x: Fr) -> Share3 {
    let mut rng = rand::thread_rng();
    let s0: Fr = rng.gen();
    let s1: Fr = rng.gen();
    let s2 = x.wrapping_sub(s0).wrapping_sub(s1);
    Share3(s0, s1, s2)
}

/// Reconstruct a shared value to plain.
pub fn reconstruct(s: &Share3) -> Fr {
    s.to_plain()
}

/// Add two shares (local, non-interactive).
pub fn add_shares(a: &Share3, b: &Share3) -> Share3 {
    a.clone() + b.clone()
}

/// Subtract two shares (local, non-interactive).
pub fn sub_shares(a: &Share3, b: &Share3) -> Share3 {
    a.clone() - b.clone()
}

/// Multiply two shares using a reconstruct-and-reshare strategy (INSECURE).
///
/// This is a placeholder until a proper ABY3 interactive multiplication is
/// implemented. It reconstructs the product and generates a fresh sharing.
pub fn mul_insecure(a: &Share3, b: &Share3) -> Share3 {
    let pa = reconstruct(a);
    let pb = reconstruct(b);
    let prod = pa.wrapping_mul(pb);
    share_secret(prod)
}

/// Multiply a share by a public scalar (local).
pub fn scalar_mul(a: &Share3, k: Fr) -> Share3 {
    // multiply each local share by the scalar
    Share3(a.0.wrapping_mul(k), a.1.wrapping_mul(k), a.2.wrapping_mul(k))
}

/// Generate a Beaver triple (x, y, z) with z = x * y, all returned as Share3 shares.
/// In a real ABY3 setting these triples would be preprocessed securely. Here we
/// locally sample x,y and produce their sharings.
pub fn gen_beaver_triple() -> (Share3, Share3, Share3) {
    let mut rng = rand::thread_rng();
    let x: Fr = rng.gen();
    let y: Fr = rng.gen();
    let x_sh = share_secret(x);
    let y_sh = share_secret(y);
    let z_sh = share_secret(x.wrapping_mul(y));
    (x_sh, y_sh, z_sh)
}

/// Simulated ABY3 multiplication using Beaver triples.
///
/// Protocol (high-level, single-process simulation):
/// 1. Generate a Beaver triple (x,y,z) with z = x*y.
/// 2. Open d = a - x and e = b - y (we reconstruct differences here to
///    simulate the broadcast/open step).
/// 3. Compute the product shares as: z + d*y + e*x + share(d*e).
///
/// This follows the standard Beaver triple online phase. The implementation
/// reconstructs intermediate public values (d,e) because we operate in a
/// single-process simulator; the structure however mirrors the real protocol.
pub fn mul_ab3(a: &Share3, b: &Share3) -> Share3 {
    // obtain a triple
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    // open differences (d = a - x, e = b - y)
    let da = reconstruct(a).wrapping_sub(reconstruct(&x_sh));
    let eb = reconstruct(b).wrapping_sub(reconstruct(&y_sh));

    // reshare the small public product da*eb
    let cross = share_secret(da.wrapping_mul(eb));

    // compute z + da * y_sh + eb * x_sh + cross
    let term_dy = scalar_mul(&y_sh, da);
    let term_ex = scalar_mul(&x_sh, eb);

    let sum1 = add_shares(&z_sh, &term_dy);
    let sum2 = add_shares(&term_ex, &cross);
    add_shares(&sum1, &sum2)
}

/// Comparison helpers (simulated): reconstruct and reshare boolean result.
pub fn lt_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = reconstruct(a);
    let vb = reconstruct(b);
    share_secret((va < vb) as Fr)
}

pub fn le_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = reconstruct(a);
    let vb = reconstruct(b);
    share_secret((va <= vb) as Fr)
}

pub fn eq_simulate(a: &Share3, b: &Share3) -> Share3 {
    let va = reconstruct(a);
    let vb = reconstruct(b);
    share_secret((va == vb) as Fr)
}

pub use crate::crypto::party::mul_ab3_simulate; // expose simulator variant
