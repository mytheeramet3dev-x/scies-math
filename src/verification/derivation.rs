//! Derivation tree and mathematical provenance tracking.

use core::fmt;

/// Formal verification and derivation status of a mathematical claim or computation.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    /// Result has not been independently verified.
    Unchecked,
    /// Result verified through floating-point numerical simulations/diagnostics.
    NumericallyTested { tolerance: f64, max_residual: f64 },
    /// Result rigorously bounded by interval arithmetic.
    IntervalBounded { lower: f64, upper: f64 },
    /// Result derived algebraically via symbolic transformations.
    SymbolicallyDerived { rule: &'static str },
    /// Result formally verified by an interactive theorem prover (e.g. Lean 4 / Coq).
    FormallyVerified {
        prover: &'static str,
        certificate_id: String,
    },
    /// Claim refuted with a concrete counterexample.
    Refuted { counterexample: String },
}

impl fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unchecked => write!(f, "Unchecked"),
            Self::NumericallyTested {
                tolerance,
                max_residual,
            } => {
                write!(
                    f,
                    "NumericallyTested(tol={:e}, res={:e})",
                    tolerance, max_residual
                )
            }
            Self::IntervalBounded { lower, upper } => {
                write!(f, "IntervalBounded([{lower:.4}, {upper:.4}])")
            }
            Self::SymbolicallyDerived { rule } => write!(f, "SymbolicallyDerived({rule})"),
            Self::FormallyVerified {
                prover,
                certificate_id,
            } => {
                write!(f, "FormallyVerified({prover}: #{certificate_id})")
            }
            Self::Refuted { counterexample } => {
                write!(f, "Refuted(counterexample: {counterexample})")
            }
        }
    }
}

/// A single step in an algebraic or mathematical derivation.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivationStep {
    pub step_number: usize,
    pub rule: String,
    pub expression: String,
    pub justification: String,
}

/// A derivation tree recording the exact lineage, assumptions, and steps of a mathematical proof or computation.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivationTree {
    pub goal: String,
    pub assumptions: Vec<String>,
    pub steps: Vec<DerivationStep>,
    pub status: VerificationStatus,
}

impl DerivationTree {
    /// Creates a new derivation tree for a given goal.
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            assumptions: Vec::new(),
            steps: Vec::new(),
            status: VerificationStatus::Unchecked,
        }
    }

    /// Adds a prerequisite mathematical assumption.
    pub fn add_assumption(&mut self, assumption: impl Into<String>) {
        self.assumptions.push(assumption.into());
    }

    /// Adds a sequential step to the derivation.
    pub fn add_step(
        &mut self,
        rule: impl Into<String>,
        expression: impl Into<String>,
        justification: impl Into<String>,
    ) {
        let step_number = self.steps.len() + 1;
        self.steps.push(DerivationStep {
            step_number,
            rule: rule.into(),
            expression: expression.into(),
            justification: justification.into(),
        });
    }

    /// Sets the final verification status.
    pub fn set_status(&mut self, status: VerificationStatus) {
        self.status = status;
    }

    /// Formats a complete human-readable proof trace.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Goal: {}\n", self.goal));
        out.push_str(&format!("Status: {}\n", self.status));
        if !self.assumptions.is_empty() {
            out.push_str("Assumptions:\n");
            for a in &self.assumptions {
                out.push_str(&format!("  - {}\n", a));
            }
        }
        out.push_str("Derivation Steps:\n");
        for step in &self.steps {
            out.push_str(&format!(
                "  {}. [{}] {} (by {})\n",
                step.step_number, step.rule, step.expression, step.justification
            ));
        }
        out
    }
}
