// Library root: expose the interpreter module for tests and the binary.
pub mod interpreter;

pub mod crypto;

pub mod zkp;

pub mod fhe;

pub mod bench;

pub mod repl;

pub mod precompile;

pub mod ml;

pub mod hybrid;

pub mod gadgets;

pub mod demo;

/// MPC compiler: `FlatOp` → protocol-neutral bytecode → round schedule →
/// analytic cost comparison across the protocol matrix.
pub mod mpcc;

/// Encrypted experiment store (BaruDB client + local JSONL fallback).
pub mod db;

/// Private information retrieval — fetch a record without revealing which.
pub mod pir;

/// Oblivious transfer — base OT and IKNP extension.
pub mod ot;

/// Private set intersection.
pub mod psi;
