//! Moment-SOS (Sums of Squares) relaxation and polynomial optimization layer.
//!
//! Provides canonical monomial indexing, moment matrix building, Boolean quotient
//! reductions, CNF/3-SAT clause encodings, and SDPA format export/import.

pub mod builder;
pub mod log;
pub mod monomial;
pub mod sat_encoding;
pub mod sdpa_io;

pub use builder::MomentMatrixBuilder;
pub use log::CalculationLog;
pub use monomial::{BooleanDomain, Monomial, MonomialBasis};
pub use sat_encoding::{Clause, CnfFormula, Literal, MomentRelaxationReport};
pub use sdpa_io::{export_sdpa_sparse, import_sdpa_sparse};
