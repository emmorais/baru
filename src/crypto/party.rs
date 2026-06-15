use crate::crypto::arith::{add_shares, gen_beaver_triple, scalar_mul, share_secret};
use crate::crypto::types::{Fr, Share3};

/// Simple per-party view simulator for ABY3-style multiplication.
///
/// This module simulates three parties (0,1,2). For a Share3 value (s0,s1,s2)
/// party i is considered to hold the pair (s_i, s_{i+1 mod 3}). To open a
/// public value (like d = a - x) we simulate each party revealing its first
/// component (s_i) and reconstructing the public value by summing those.

/// Split a Share3 into per-party pairs: party i gets (s_i, s_{i+1}).
fn split_to_party_pairs(s: &Share3) -> Vec<(Fr, Fr)> {
    vec![(s.0, s.1), (s.1, s.2), (s.2, s.0)]
}

/// Simulate ABY3 multiplication across three parties using Beaver triples.
///
/// Input and outputs are global Share3 values. The function internally
/// simulates per-party local computations and the opening of d and e, and
/// then returns a fresh Share3 that secret-shares the product.
pub fn mul_ab3_simulate(a: &Share3, b: &Share3) -> Share3 {
    // Preprocessing: generate Beaver triple shares
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    // split inputs and triples into per-party pairs
    let a_pairs = split_to_party_pairs(a);
    let b_pairs = split_to_party_pairs(b);
    let x_pairs = split_to_party_pairs(&x_sh);
    let y_pairs = split_to_party_pairs(&y_sh);
    let _z_pairs = split_to_party_pairs(&z_sh);

    // Each party computes its local difference share (we reveal the first
    // component of each pair as part of the opening): d_i = a_i - x_i
    let mut d_shares: [Fr; 3] = [0, 0, 0];
    let mut e_shares: [Fr; 3] = [0, 0, 0];
    for i in 0..3 {
        let a_i = a_pairs[i].0; // first component
        let x_i = x_pairs[i].0;
        d_shares[i] = a_i.wrapping_sub(x_i);

        let b_i = b_pairs[i].0;
        let y_i = y_pairs[i].0;
        e_shares[i] = b_i.wrapping_sub(y_i);
    }

    // Opening: parties reveal d_i and e_i; reconstruct public d and e
    let mut d: Fr = 0;
    let mut e: Fr = 0;
    for i in 0..3 {
        d = d.wrapping_add(d_shares[i]);
        e = e.wrapping_add(e_shares[i]);
    }

    // Reshare public cross term and compute per-share terms
    let cross = share_secret(d.wrapping_mul(e));

    // term_dy = d * y_sh (scalar multiply of entire Share3)
    let term_dy = scalar_mul(&y_sh, d);
    let term_ex = scalar_mul(&x_sh, e);

    // output share = z + d*y + e*x + cross
    let s1 = add_shares(&z_sh, &term_dy);
    let s2 = add_shares(&term_ex, &cross);
    add_shares(&s1, &s2)
}
