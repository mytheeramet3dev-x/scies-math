use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SciError {
    EmptyInput,
    DivisionByZero,
    DomainError(&'static str),
    NonConvergent(&'static str),
    InvalidParameter(&'static str),
}

pub type SciResult<T> = Result<T, SciError>;

impl fmt::Display for SciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input collection is empty"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::DomainError(msg) => write!(f, "domain error: {msg}"),
            Self::NonConvergent(msg) => write!(f, "method did not converge: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for SciError {}
