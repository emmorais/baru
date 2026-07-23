// End-to-end tests for MPC <-> FHE scheme switching (`src/hybrid/`), at the
// Baru-language level (the `share->fhe` / `fhe->share` builtins) and against
// the real TFHE / SEAL-CKKS backends (ignored by default: slow).

use baru::interpreter::eval::{eval_plain as eval, eval_shared};

// ─── REPL-level: mock integer FHE ─────────────────────────────────────────────

#[test]
fn test_share_to_fhe_roundtrip_negative() {
    // A share never reconstructed in MPC crosses into FHE and decrypts to
    // the same signed value (the handle's nbits triggers reduce+sign-extend).
    let src = "(let ((keys (fhe-keygen)))
                 (fhe-dec keys (share->fhe keys (share -5 32))))";
    assert_eq!(eval(src), -5);
}

#[test]
fn test_fhe_to_share_roundtrip() {
    let src = "(let ((keys (fhe-keygen)))
                 (reconstruct (fhe->share keys (fhe-enc keys 7) 32)))";
    assert_eq!(eval_shared(src), 7);
    let neg = "(let ((keys (fhe-keygen)))
                 (reconstruct (fhe->share keys (fhe-enc keys -9) 32)))";
    assert_eq!(eval_shared(neg), -9);
}

#[test]
fn test_mpc_compute_then_switch_to_fhe() {
    // Multiply under MPC (interactive, cheap), then hand the *still-secret*
    // product to the FHE domain.
    let src = "(let ((keys (fhe-keygen)) (x (share 6 32)) (y (share -7 32)))
                 (fhe-dec keys (share->fhe keys (* x y))))";
    assert_eq!(eval(src), -42);
}

#[test]
fn test_fhe_value_then_switch_to_mpc_for_relu() {
    // The Gazelle moment, in the REPL: a value living under FHE needs a
    // comparison (which FHE does badly); switch schemes and do the ReLU
    // under MPC in mux style.
    let src = "(let ((keys (fhe-keygen)))
                 (let ((z (fhe->share keys (fhe-enc keys -4) 32)))
                   (reconstruct (* z (> z 0)))))";
    assert_eq!(eval_shared(src), 0);
    let pos = "(let ((keys (fhe-keygen)))
                 (let ((z (fhe->share keys (fhe-enc keys 11) 32)))
                   (reconstruct (* z (> z 0)))))";
    assert_eq!(eval_shared(pos), 11);
}

#[test]
fn test_fixed_point_share_survives_the_switch() {
    // A Q16.16 share converts with its `fixed` flag intact; fhe-dec returns
    // the Fixed value (the raw scaled residue, reduced and sign-extended).
    let src = "(let ((keys (fhe-keygen)))
                 (fixed->int (fhe-dec keys (share->fhe keys (share 2.5 32)))))";
    assert_eq!(eval(src), 2);
    let neg = "(let ((keys (fhe-keygen)))
                 (fixed->int (* 2 (fhe-dec keys (share->fhe keys (share -1.5 32))))))";
    assert_eq!(eval(neg), -3);
}

#[test]
fn test_round_trip_share_fhe_share() {
    // MPC -> FHE -> MPC: the value survives two scheme switches and further
    // MPC arithmetic afterwards.
    let src = "(let ((keys (fhe-keygen)) (x (share 20 32)))
                 (let ((back (fhe->share keys (share->fhe keys x) 32)))
                   (reconstruct (+ back 1))))";
    assert_eq!(eval_shared(src), 21);
}

// ─── REPL-level: CKKS mock (approximate-real adapter) ─────────────────────────

#[test]
fn test_share_to_fhe_ckks_mock() {
    let src = "(let ((keys (fhe-keygen-ckks-mock)))
                 (fhe-dec keys (share->fhe keys (share 42 32))))";
    assert_eq!(eval(src), 42);
    let neg = "(let ((keys (fhe-keygen-ckks-mock)))
                 (fhe-dec keys (share->fhe keys (share -3 32))))";
    assert_eq!(eval(neg), -3);
}

#[test]
fn test_fhe_to_share_ckks_mock() {
    let src = "(let ((keys (fhe-keygen-ckks-mock)))
                 (reconstruct (fhe->share keys (fhe-enc keys 13) 32)))";
    assert_eq!(eval_shared(src), 13);
}

// ─── Real backends (slow — run with `--ignored --release`) ────────────────────

#[test]
#[ignore] // slow: real TFHE keygen (~seconds) + per-op PBS
fn test_real_tfhe_share_conversion_and_reencrypt() {
    use baru::fhe::{FheBackend, TfheBackend};
    use baru::hybrid::{fhe_to_shares, mpc_reencrypt, shares_to_fhe, FheAdapter, IntFhe};

    let f = TfheBackend::new();
    let (ck, sk) = f.keygen();
    TfheBackend::install_server_key(&sk);
    let a = IntFhe(&f);
    let nbits = 32usize;

    // shares -> fhe -> decrypt
    let m: i64 = -123;
    let residue = (m as i128).rem_euclid(1i128 << nbits) as u128;
    let components = [residue.wrapping_sub(999), 500, 499]; // sums to residue
    let ct = shares_to_fhe(&a, &ck, &components, nbits);
    let got = a.decrypt(&ck, &ct) & 0xFFFF_FFFF;
    assert_eq!(got as u128, residue);

    // fhe -> shares -> sum
    let back = fhe_to_shares(&a, &ck, &ct, 3, nbits);
    let sum = back.iter().fold(0u128, |acc, &c| acc.wrapping_add(c));
    assert_eq!(sum & 0xFFFF_FFFF, residue);

    // interactive re-encryption (the bootstrapping alternative): fresh
    // ciphertext, same residue class.
    let fresh = mpc_reencrypt(&a, &ck, &ct, 3, nbits);
    assert_eq!((a.decrypt(&ck, &fresh) as u128) & 0xFFFF_FFFF, residue);
}

#[test]
#[ignore] // slow: real SEAL keygen
fn test_real_ckks_share_conversion_resets_depth() {
    use baru::fhe::{CkksBackend, RealFheBackend};
    use baru::hybrid::{mpc_reencrypt, RealFhe};

    let client = CkksBackend::new();
    let (ck, sk) = client.keygen();
    let be = CkksBackend::with_server_keys(&sk);
    let a = RealFhe(&be);

    // Burn depth with a real ciphertext-ciphertext multiply (small positive
    // value so the squared residue stays exactly representable in f64).
    let x = be.encrypt(&ck, 21.0);
    let squared = be.mul(&x, &x); // depth 1, value 441
    assert!(squared.depth > 0, "multiply must consume depth");

    // Interactive re-encryption restores a fresh, depth-0 ciphertext of the
    // same value — the levelled-CKKS payoff of scheme switching: no
    // bootstrapping circuit, one mask-and-open round instead.
    let fresh = mpc_reencrypt(&a, &ck, &squared, 3, 32);
    assert_eq!(fresh.depth, 0, "re-encrypted ciphertext must be fresh");
    // Residue-class semantics (see hybrid module docs): the representative
    // may carry k·2^32 from the share sum — reduce before comparing.
    let got = (be.decrypt(&ck, &fresh).round() as i64) & 0xFFFF_FFFF;
    assert_eq!(got, 441);
}
