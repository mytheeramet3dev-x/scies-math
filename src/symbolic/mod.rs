//! Symbolic mathematics engine: AST expression trees, canonical simplification, derivatives, and polynomials.

pub mod derivative;
pub mod expr;
pub mod polynomial;
pub mod simplify;

pub use derivative::diff;
pub use expr::Expr;
pub use polynomial::Polynomial;
pub use simplify::simplify;
