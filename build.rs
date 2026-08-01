//! Builds the optional NTL+GMP polynomial-multiplication shim
//! (`cpp/ntl_polymul.cpp`, wrapped by `src/fhe/ntl.rs`).
//!
//! Only runs under `--features ntl`, so the default build stays pure Rust and
//! needs no C++ toolchain, no NTL and no GMP.
//!
//! Header/library discovery, first match wins:
//!   1. `NTL_PREFIX` / `GMP_PREFIX` environment variables
//!   2. `pkg-config` (GMP only; NTL ships no `.pc` file)
//!   3. the usual install prefixes (Homebrew on both architectures, /usr/local, /usr)

use std::path::{Path, PathBuf};
use std::process::Command;

const CANDIDATE_PREFIXES: &[&str] = &[
    "/opt/homebrew/opt", // Homebrew, Apple Silicon
    "/usr/local/opt",    // Homebrew, Intel
    "/opt/homebrew",
    "/usr/local",
    "/usr",
];

/// Locate a package prefix containing `include/<probe_header>`.
fn find_prefix(pkg: &str, env_var: &str, probe_header: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        return Some(PathBuf::from(p));
    }
    if let Some(p) = pkg_config_prefix(pkg) {
        if p.join("include").join(probe_header).exists() {
            return Some(p);
        }
    }
    for base in CANDIDATE_PREFIXES {
        // Both `<base>/<pkg>` (Homebrew's per-formula opt dir) and `<base>`.
        for cand in [Path::new(base).join(pkg), PathBuf::from(base)] {
            if cand.join("include").join(probe_header).exists() {
                return Some(cand);
            }
        }
    }
    None
}

fn pkg_config_prefix(pkg: &str) -> Option<PathBuf> {
    let out = Command::new("pkg-config")
        .args(["--variable=prefix", pkg])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

fn main() {
    println!("cargo:rerun-if-changed=cpp/ntl_polymul.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NTL_PREFIX");
    println!("cargo:rerun-if-env-changed=GMP_PREFIX");

    if std::env::var("CARGO_FEATURE_NTL").is_err() {
        return;
    }

    let ntl = find_prefix("ntl", "NTL_PREFIX", "NTL/ZZX.h").unwrap_or_else(|| {
        panic!(
            "feature `ntl` is enabled but NTL/ZZX.h was not found.\n\
             Install NTL (macOS: `brew install ntl`, Debian: `apt install libntl-dev`)\n\
             or point NTL_PREFIX at its install prefix."
        )
    });
    let gmp = find_prefix("gmp", "GMP_PREFIX", "gmp.h").unwrap_or_else(|| {
        panic!(
            "feature `ntl` is enabled but gmp.h was not found.\n\
             Install GMP (macOS: `brew install gmp`, Debian: `apt install libgmp-dev`)\n\
             or point GMP_PREFIX at its install prefix."
        )
    });

    cc::Build::new()
        .cpp(true)
        .std("c++14") // NTL headers require C++11 or later
        .file("cpp/ntl_polymul.cpp")
        .include(ntl.join("include"))
        .include(gmp.join("include"))
        .opt_level(2)
        .warnings(false) // NTL's headers are not warning-clean under -Wall
        .compile("baru_ntl_polymul");

    println!(
        "cargo:rustc-link-search=native={}",
        ntl.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        gmp.join("lib").display()
    );
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
}
