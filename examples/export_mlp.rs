//! Write the MLP fixture `tools/ckks_cnn.cpp` reads, so Baru's own CKKS arm can
//! evaluate the same network HEIR and the two-party protocol evaluate.
//!
//! `cargo run --release --example export_mlp -- out.txt 64x32x10 [input_index]`
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: export_mlp <path> <shape> [k]");
    let shape: Vec<usize> = args
        .next()
        .expect("usage: export_mlp <path> <shape> [k]")
        .split('x')
        .map(|d| d.parse().expect("shape dims are integers"))
        .collect();
    let k: usize = args.next().map(|s| s.parse().expect("k")).unwrap_or(0);
    let text = baru::bench::two_party_nn::export_mlp_fixture(&shape, 8, k);
    std::fs::write(&path, text).expect("write");
    eprintln!("wrote {path} for shape {shape:?}, input {k}");
}
