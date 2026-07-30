use baru::crypto::types::Fr;
use baru::crypto::{arith, bits, sim};

#[test]
fn test_share_add() {
    let a: Fr = 42u128;
    let b: Fr = 7u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let sum = arith::add_shares(&sa, &sb);
    assert_eq!(arith::reconstruct(&sum), a.wrapping_add(b));
}

#[test]
fn test_mul_ab3_matches_the_inproc_protocol() {
    let a: Fr = 5u128;
    let b: Fr = 11u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let prod = arith::mul_ab3(&sa, &sb);
    assert_eq!(arith::reconstruct(&prod), a.wrapping_mul(b));
    // the same product through the inproc transport-orchestrated protocol
    let prod2 = sim::mul_ab3_with_inproc(&sa, &sb);
    assert_eq!(arith::reconstruct(&prod2), a.wrapping_mul(b));
}

#[test]
fn test_bits_ops() {
    let t = bits::share_bool(true);
    let f = bits::share_bool(false);
    let andv = bits::and(&t, &f);
    assert_eq!(bits::reconstruct_bit(&andv), true & false);
    let orv = bits::or(&t, &f);
    assert_eq!(bits::reconstruct_bit(&orv), true | false);
    let notv = bits::not(&t);
    assert!(!bits::reconstruct_bit(&notv));
}

#[test]
fn test_a2b_and_comparisons() {
    // test decomposition and comparison helpers
    let a: Fr = 37u128; // 0b100101
    let b: Fr = 45u128; // 0b101101
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);

    // a2b decomposition
    let bits = bits::a2b_simulate(&sa, 8);
    let mut recovered: u128 = 0;
    for (i, bit_sh) in bits.iter().enumerate() {
        if bits::reconstruct_bit(bit_sh) {
            recovered |= 1u128 << i;
        }
    }
    assert_eq!(recovered, a);

    // new: test inproc a2b wrapper
    let bits_inproc = sim::a2b_with_inproc(&sa, 8);
    let mut recovered_inproc: u128 = 0;
    for (i, bit_sh) in bits_inproc.iter().enumerate() {
        if bits::reconstruct_bit(bit_sh) {
            recovered_inproc |= 1u128 << i;
        }
    }
    assert_eq!(
        recovered_inproc & ((1u128 << 8) - 1),
        a & ((1u128 << 8) - 1)
    );

    // comparisons
    let lt_sh = bits::lt_ab3(&sa, &sb, 8);
    assert_eq!(bits::reconstruct_bit(&lt_sh), a < b);
    let eq_sh = bits::eq_ab3(&sa, &sb, 8);
    assert_eq!(bits::reconstruct_bit(&eq_sh), a == b);
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn prop_mul_randomized() {
    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    for _ in 0..200 {
        let a: Fr = rng.gen();
        let b: Fr = rng.gen();
        let sa = arith::share_secret(a);
        let sb = arith::share_secret(b);
        let p1 = arith::mul_ab3(&sa, &sb);
        assert_eq!(arith::reconstruct(&p1), a.wrapping_mul(b));
        // exercise the inproc transport wrapper as well
        let p2 = sim::mul_ab3_with_inproc(&sa, &sb);
        assert_eq!(arith::reconstruct(&p2), a.wrapping_mul(b));
    }
}

#[test]
fn prop_a2b_and_comparisons_randomized() {
    let mut rng = StdRng::seed_from_u64(0x8765_4321);
    for _ in 0..200 {
        let a: u128 = rng.gen::<u128>() & ((1u128 << 32) - 1);
        let b: u128 = rng.gen::<u128>() & ((1u128 << 32) - 1);
        let sa = arith::share_secret(a);
        let sb = arith::share_secret(b);

        // test a2b decomposition (protocol and simulate)
        let bits_sim = bits::a2b_simulate(&sa, 32);
        let mut rec_sim: u128 = 0;
        for (i, bit_sh) in bits_sim.iter().enumerate() {
            if bits::reconstruct_bit(bit_sh) {
                rec_sim |= 1u128 << i;
            }
        }
        assert_eq!(rec_sim, a);

        let (bits_proto, _carry) = bits::a2b_ab3(&sa, 8);
        let mut rec_proto: u128 = 0;
        for (i, bit_sh) in bits_proto.iter().enumerate() {
            if bits::reconstruct_bit(bit_sh) {
                rec_proto |= 1u128 << i;
            }
        }
        // protocol may operate mod field; assert lower 8 bits match
        assert_eq!(rec_proto & ((1u128 << 8) - 1), a & ((1u128 << 8) - 1));

        // new: check inproc a2b
        let bits_inproc = sim::a2b_with_inproc(&sa, 8);
        let mut rec_inproc: u128 = 0;
        for (i, bit_sh) in bits_inproc.iter().enumerate() {
            if bits::reconstruct_bit(bit_sh) {
                rec_inproc |= 1u128 << i;
            }
        }
        assert_eq!(rec_inproc & ((1u128 << 8) - 1), a & ((1u128 << 8) - 1));

        // comparisons: compare protocol to simulated (reconstruct-based) reference
        let lt_sh = bits::lt_ab3(&sa, &sb, 32);
        let lt_ref = arith::lt_simulate(&sa, &sb);
        let lt_res = bits::reconstruct_bit(&lt_sh);
        let lt_ref_val = arith::reconstruct(&lt_ref) != 0;
        if lt_res != lt_ref_val {
            eprintln!(
                "Random mismatch lt: a={} b={} proto={} ref={}",
                a, b, lt_res, lt_ref_val
            );
            // Dump a2b protocol internals for the exact diff used by bits::lt_ab3
            let diff = arith::sub_shares(&sa, &sb);
            // call the a2b implementation used by bits::lt_ab3 to inspect its actual output
            let (bits_exact, carry_exact) = bits::a2b_ab3(&diff, 32);
            let carry_plain = arith::reconstruct(&carry_exact).is_multiple_of(2);
            eprintln!("a2b_ab3 carry (reconstructed) = {}", carry_plain);
            let mut bits_plain_vec: Vec<bool> = Vec::new();
            for bit in &bits_exact {
                bits_plain_vec.push(bits::reconstruct_bit(bit));
            }
            eprintln!("a2b_ab3 bits (reconstructed) = {:?}", bits_plain_vec);
            // reconstruct numeric diff from protocol bits
            let mut proto_diff: u128 = 0;
            for (i, &b) in bits_plain_vec.iter().enumerate() {
                if b {
                    proto_diff |= 1u128 << i;
                }
            }
            let diff_plain = arith::reconstruct(&diff);
            eprintln!(
                "diff_plain = {} proto_diff(lower 32) = {}",
                diff_plain,
                proto_diff & ((1u128 << 32) - 1)
            );
        }
        assert_eq!(lt_res, lt_ref_val);
        let eq_sh = bits::eq_ab3(&sa, &sb, 32);
        let eq_ref = arith::eq_simulate(&sa, &sb);
        let eq_res = bits::reconstruct_bit(&eq_sh);
        let eq_ref_val = arith::reconstruct(&eq_ref) != 0;
        if eq_res != eq_ref_val {
            eprintln!(
                "Random mismatch eq: a={} b={} proto={} ref={}",
                a, b, eq_res, eq_ref_val
            );
        }
        assert_eq!(eq_res, eq_ref_val);
    }
}

#[test]
fn test_mul_ab3_concurrent() {
    let a: Fr = 19u128;
    let b: Fr = 23u128;
    let sa = arith::share_secret(a);
    let sb = arith::share_secret(b);
    let out = sim::mul_ab3_concurrent(&sa, &sb);
    assert_eq!(arith::reconstruct(&out), a.wrapping_mul(b));
}

#[test]
fn test_a2b_concurrent() {
    let a: Fr = 37u128; // small test value
    let sa = arith::share_secret(a);
    let bits = sim::a2b_concurrent(&sa, 8);
    let mut rec: u128 = 0;
    for (i, bit_sh) in bits.iter().enumerate() {
        if bits::reconstruct_bit(bit_sh) {
            rec |= 1u128 << i;
        }
    }
    assert_eq!(rec & ((1u128 << 8) - 1), a & ((1u128 << 8) - 1));
}
