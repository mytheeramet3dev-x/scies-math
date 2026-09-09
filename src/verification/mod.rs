//! Mathematical verification, derivation provenance, and Lean 4 proof bridge.

pub mod derivation;
pub mod lean_export;

pub use derivation::{DerivationStep, DerivationTree, VerificationStatus};
pub use lean_export::{export_lean4_theorem, expr_to_lean};
