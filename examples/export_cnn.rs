// Write the fixture the homomorphic arm reads, so both arms evaluate one definition.
use baru::bench::conv_nn::{export, Cnn};
use baru::ml::Activation;

fn main() {
    let path = std::env::args().nth(1).expect("usage: export_cnn <path>");
    let k: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("input index"))
        .unwrap_or(0);
    let extra: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("extra depth"))
        .unwrap_or(0);
    let net = Cnn::lenet5_deep(8, Activation::Square, extra);
    std::fs::write(&path, export(&net, k)).expect("write");
    println!("wrote {path}");
}
