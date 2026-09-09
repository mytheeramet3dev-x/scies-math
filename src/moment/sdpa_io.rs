//! Export and Import utilities for the universal SDPA sparse format (`.dat-s`).
//!
//! SDPA format is the standard benchmark exchange format accepted by high-performance
//! SDP solvers including CSDP, SDPA, Mosek, Clarabel, and SeDuMi.

use crate::errors::{SciError, SciResult};
use crate::sdp::SdpProblem;
use crate::symmetric::SymmetricMatrix;
use std::fmt::Write as FmtWrite;

/// Exports an `SdpProblem` into a valid SDPA sparse format string (`.dat-s`).
pub fn export_sdpa_sparse(problem: &SdpProblem, comment: &str) -> SciResult<String> {
    let mut out = String::new();

    // Line 1: Comment header
    writeln!(&mut out, "\"* {}\"", comment.replace('"', "'"))
        .map_err(|_| SciError::InvalidParameter("formatting error in SDPA export"))?;

    let m = problem.num_constraints();
    let n = problem.matrix_dim();

    // Line 2: Number of constraints m
    writeln!(&mut out, "{m}")
        .map_err(|_| SciError::InvalidParameter("formatting error in SDPA export"))?;

    // Line 3: Number of blocks (1 single semidefinite block)
    writeln!(&mut out, "1")
        .map_err(|_| SciError::InvalidParameter("formatting error in SDPA export"))?;

    // Line 4: Block structure
    writeln!(&mut out, "{n}")
        .map_err(|_| SciError::InvalidParameter("formatting error in SDPA export"))?;

    // Line 5: b vector
    let b_str = problem
        .b
        .iter()
        .map(|v| format!("{:.8e}", v))
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(&mut out, "{b_str}")
        .map_err(|_| SciError::InvalidParameter("formatting error in SDPA export"))?;

    // Matrix 0: Cost matrix C (upper/lower triangular non-zeros)
    write_sdpa_matrix_entries(&mut out, 0, &problem.c)?;

    // Matrices 1..=m: Constraint matrices A_1..A_m
    for (k, ak) in problem.a_constraints.iter().enumerate() {
        write_sdpa_matrix_entries(&mut out, k + 1, ak)?;
    }

    Ok(out)
}

fn write_sdpa_matrix_entries(
    out: &mut String,
    mat_index: usize,
    sym: &SymmetricMatrix,
) -> SciResult<()> {
    let n = sym.size();
    for r in 0..n {
        for c in 0..=r {
            let val = sym.get(r, c)?;
            if val.abs() > 1e-14 {
                // 1-indexed in SDPA
                writeln!(out, "{mat_index} 1 {} {} {:.8e}", r + 1, c + 1, val).map_err(|_| {
                    SciError::InvalidParameter("formatting error in SDPA matrix write")
                })?;
            }
        }
    }
    Ok(())
}

/// Parses an SDPA sparse format string into an `SdpProblem`.
///
/// # Errors
/// Returns [`SciError::InvalidParameter`] if the input is malformed, has invalid token counts,
/// attempts unsupported multi-block partitions ($nblocks \ne 1$), or contains out-of-bounds indices.
pub fn import_sdpa_sparse(input: &str) -> SciResult<SdpProblem> {
    let mut lines = input
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('"') && !l.starts_with('*'));

    let m: usize = lines
        .next()
        .ok_or(SciError::InvalidParameter(
            "missing constraints count m line",
        ))?
        .parse()
        .map_err(|_| SciError::InvalidParameter("invalid constraints count m integer"))?;

    let nblocks: usize = lines
        .next()
        .ok_or(SciError::InvalidParameter("missing nblocks line"))?
        .parse()
        .map_err(|_| SciError::InvalidParameter("invalid nblocks integer"))?;

    if nblocks != 1 {
        return Err(SciError::InvalidParameter(
            "unsupported multi-block structure in SDPA (only single-block nblocks=1 is currently supported)",
        ));
    }

    let block_size_tokens: Vec<&str> = lines
        .next()
        .ok_or(SciError::InvalidParameter("missing block sizes line"))?
        .split_whitespace()
        .collect();

    if block_size_tokens.len() != 1 {
        return Err(SciError::InvalidParameter(
            "block sizes count does not match nblocks=1",
        ));
    }

    let n: usize = block_size_tokens[0]
        .parse()
        .map_err(|_| SciError::InvalidParameter("invalid block size integer"))?;

    if n == 0 {
        return Err(SciError::InvalidParameter("block size must be > 0"));
    }

    let b_line = lines
        .next()
        .ok_or(SciError::InvalidParameter("missing b vector line"))?;
    let b: Vec<f64> = b_line
        .split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .map_err(|_| SciError::InvalidParameter("invalid float in b vector"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if b.len() != m {
        return Err(SciError::InvalidParameter(
            "b vector element count does not match m",
        ));
    }

    let mut c = SymmetricMatrix::zeros(n)?;
    let mut a_constraints = vec![SymmetricMatrix::zeros(n)?; m];

    for (line_no, line) in lines.enumerate() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 5 {
            return Err(SciError::InvalidParameter(
                "malformed SDPA data entry: expected exactly 5 tokens (mat_idx blk_idx row col val)",
            ));
        }

        let mat_idx: usize = tokens[0]
            .parse()
            .map_err(|_| SciError::InvalidParameter("invalid mat_idx integer"))?;
        let blk_idx: usize = tokens[1]
            .parse()
            .map_err(|_| SciError::InvalidParameter("invalid blk_idx integer"))?;
        let r: usize = tokens[2]
            .parse()
            .map_err(|_| SciError::InvalidParameter("invalid row index"))?;
        let c_idx: usize = tokens[3]
            .parse()
            .map_err(|_| SciError::InvalidParameter("invalid col index"))?;
        let val: f64 = tokens[4]
            .parse()
            .map_err(|_| SciError::InvalidParameter("invalid matrix float value"))?;

        if val.is_nan() || val.is_infinite() {
            return Err(SciError::DomainError(
                "SDPA matrix entries must be finite (not NaN or Inf)",
            ));
        }

        if blk_idx != 1 {
            return Err(SciError::InvalidParameter(
                "invalid blk_idx: expected 1 for single-block problem",
            ));
        }

        if r == 0 || c_idx == 0 || r > n || c_idx > n {
            return Err(SciError::InvalidParameter(
                "matrix row or column index out of bounds",
            ));
        }

        if mat_idx > m {
            return Err(SciError::InvalidParameter(
                "mat_idx exceeds total number of constraints m",
            ));
        }

        if mat_idx == 0 {
            let cur = c.get(r - 1, c_idx - 1)?;
            c.set(r - 1, c_idx - 1, cur + val)?;
        } else {
            let cur = a_constraints[mat_idx - 1].get(r - 1, c_idx - 1)?;
            a_constraints[mat_idx - 1].set(r - 1, c_idx - 1, cur + val)?;
        }
        let _ = line_no;
    }

    SdpProblem::new(c, a_constraints, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdpa_sparse_roundtrip() {
        let c = SymmetricMatrix::new(2, vec![1.0, 0.5, 2.0]).unwrap();
        let a1 = SymmetricMatrix::new(2, vec![1.0, 0.0, 1.0]).unwrap();
        let b = vec![3.0];

        let original = SdpProblem::new(c, vec![a1], b).unwrap();
        let exported = export_sdpa_sparse(&original, "Test Problem").unwrap();
        let imported = import_sdpa_sparse(&exported).unwrap();

        assert_eq!(imported.matrix_dim(), original.matrix_dim());
        assert_eq!(imported.num_constraints(), original.num_constraints());
        assert_eq!(imported.b, original.b);
        assert_eq!(imported.c.raw_data(), original.c.raw_data());
    }

    #[test]
    fn test_sdpa_malformed_input_rejection() {
        // Reject invalid multi-block
        let multi_block = "* comment\n1\n2\n2 2\n1.0\n0 1 1 1 1.0\n";
        assert!(import_sdpa_sparse(multi_block).is_err());

        // Reject bad tokens
        let bad_token = "* comment\n1\n1\n2\n1.0\n0 1 1\n";
        assert!(import_sdpa_sparse(bad_token).is_err());

        // Reject out-of-bounds mat_idx
        let oob_mat = "* comment\n1\n1\n2\n1.0\n5 1 1 1 1.0\n";
        assert!(import_sdpa_sparse(oob_mat).is_err());
    }
}
