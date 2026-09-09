use core::fmt;

/// Unified mathematical and computational error taxonomy for `scies-math-th`.
#[derive(Debug, Clone, PartialEq)]
pub enum SciError {
    /// Input collection or matrix is empty when non-empty was expected.
    EmptyInput,
    /// Division by zero in scalar, linear algebra, or exact arithmetic.
    DivisionByZero,
    /// Operation attempted outside mathematical domain (e.g. sqrt of negative, log of zero, out-of-range probability).
    DomainError(&'static str),
    /// Iterative numerical algorithm did not converge within specified iteration or tolerance limits.
    NonConvergence(&'static str),
    /// Alias for NonConvergence for backward compatibility.
    NonConvergent(&'static str),
    /// Invalid parameter or configuration supplied to a function.
    InvalidParameter(&'static str),
    /// Dimension mismatch between incompatible matrix, vector, or tensor dimensions.
    DimensionMismatch { expected: String, found: String },
    /// Dimensional analysis mismatch between incompatible physical unit quantities.
    DimensionalMismatch { expected: String, found: String },
    /// Algorithm breakdown, catastrophic cancellation, or floating-point breakdown.
    NumericalFailure(&'static str),
    /// Optimization, SDP, or linear program has no feasible solution.
    Infeasible(&'static str),
    /// Backward-compatible alias for Infeasible.
    InfeasibleProblem(&'static str),
    /// Optimization or SDP objective is unbounded along a recession direction.
    Unbounded(&'static str),
    /// Unsupported file or problem format (e.g. unsupported multi-block SDPA).
    UnsupportedFormat(&'static str),
    /// Parse error in external file format (SDPA, DIMACS, CSV).
    ParseError(String),
    /// Exact integer or rational arithmetic overflow (e.g. 128-bit integer bound exceeded).
    ExactArithmeticOverflow(&'static str),
    /// Error during evaluation or parsing of a symbolic expression.
    SymbolicEvaluationError(String),
    /// Matrix is singular or not invertible where invertibility is required.
    SingularMatrix(&'static str),
    /// Numerical condition number exceeded safe tolerance threshold.
    IllConditioned {
        condition_number: f64,
        threshold: f64,
    },
}

pub type SciResult<T> = Result<T, SciError>;

impl fmt::Display for SciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input collection is empty"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::DomainError(msg) => write!(f, "domain error: {msg}"),
            Self::NonConvergence(msg) | Self::NonConvergent(msg) => {
                write!(f, "numerical method did not converge: {msg}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "dimension mismatch: expected '{expected}', found '{found}'"
                )
            }
            Self::DimensionalMismatch { expected, found } => {
                write!(
                    f,
                    "unit dimensional mismatch: expected '{expected}', found '{found}'"
                )
            }
            Self::NumericalFailure(msg) => write!(f, "numerical algorithm failure: {msg}"),
            Self::Infeasible(msg) | Self::InfeasibleProblem(msg) => {
                write!(f, "infeasible problem: {msg}")
            }
            Self::Unbounded(msg) => write!(f, "unbounded problem: {msg}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::ExactArithmeticOverflow(msg) => write!(f, "exact arithmetic overflow: {msg}"),
            Self::SymbolicEvaluationError(msg) => write!(f, "symbolic evaluation error: {msg}"),
            Self::SingularMatrix(msg) => write!(f, "singular matrix: {msg}"),
            Self::IllConditioned {
                condition_number,
                threshold,
            } => {
                write!(
                    f,
                    "ill-conditioned system: condition number {condition_number:.2e} exceeds threshold {threshold:.2e}"
                )
            }
        }
    }
}

impl std::error::Error for SciError {}
