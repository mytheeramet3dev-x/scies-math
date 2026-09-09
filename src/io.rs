//! Scientific I/O Operations
//!
//! Provides utilities to read and write matrices and datasets from/to standard
//! scientific file formats without external dependencies.
//!
//! # Supported Formats
//! - **CSV (Comma-Separated Values)**: Simple flat data tables.
//! - **Matrix Market (.mtx)**: Academic standard for dense and sparse matrices.

use crate::errors::{SciError, SciResult};
use crate::linear_algebra::DynamicMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// ══════════════════════════════════════════════════════════════════════════════
// CSV (Comma-Separated Values)
// ══════════════════════════════════════════════════════════════════════════════

/// I/O operations for CSV files.
pub struct CsvMatrix;

impl CsvMatrix {
    /// Reads a CSV file into a `DynamicMatrix`.
    ///
    /// If `has_header` is true, the first line will be skipped.
    pub fn read<P: AsRef<Path>>(path: P, has_header: bool) -> SciResult<DynamicMatrix> {
        let file =
            File::open(path).map_err(|_| SciError::InvalidParameter("Could not open file"))?;
        let reader = BufReader::new(file);

        let mut data = Vec::new();
        let mut cols = 0;
        let mut rows = 0;

        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|_| SciError::InvalidParameter("Error reading line"))?;

            if i == 0 && has_header {
                continue;
            }

            let row_data: Vec<f64> = line
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();

            if cols == 0 {
                cols = row_data.len();
            } else if row_data.len() != cols && !row_data.is_empty() {
                return Err(SciError::InvalidParameter(
                    "Inconsistent number of columns in CSV",
                ));
            }

            if !row_data.is_empty() {
                data.extend(row_data);
                rows += 1;
            }
        }

        DynamicMatrix::new(rows, cols, data)
    }

    /// Writes a `DynamicMatrix` to a CSV file.
    pub fn write<P: AsRef<Path>>(path: P, matrix: &DynamicMatrix) -> SciResult<()> {
        let file =
            File::create(path).map_err(|_| SciError::InvalidParameter("Could not create file"))?;
        let mut writer = BufWriter::new(file);

        for r in 0..matrix.rows() {
            for c in 0..matrix.cols() {
                let val = matrix.get(r, c)?;
                write!(writer, "{}", val)
                    .map_err(|_| SciError::InvalidParameter("Error writing to file"))?;
                if c < matrix.cols() - 1 {
                    write!(writer, ",")
                        .map_err(|_| SciError::InvalidParameter("Error writing to file"))?;
                }
            }
            writeln!(writer).map_err(|_| SciError::InvalidParameter("Error writing to file"))?;
        }

        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Matrix Market Format (.mtx)
// ══════════════════════════════════════════════════════════════════════════════

/// I/O operations for Matrix Market format files.
/// Currently supports dense arrays.
pub struct MatrixMarket;

impl MatrixMarket {
    /// Reads a Matrix Market coordinate or array file into a `DynamicMatrix`.
    pub fn read<P: AsRef<Path>>(path: P) -> SciResult<DynamicMatrix> {
        let file =
            File::open(path).map_err(|_| SciError::InvalidParameter("Could not open file"))?;
        let reader = BufReader::new(file);

        let mut lines = reader.lines();

        // Check header
        if let Some(Ok(header)) = lines.next() {
            if !header.starts_with("%%MatrixMarket") {
                return Err(SciError::InvalidParameter("Invalid MatrixMarket header"));
            }
        } else {
            return Err(SciError::InvalidParameter("Empty MatrixMarket file"));
        }

        // Skip comments
        let mut dimensions_line = String::new();
        for line_res in lines.by_ref() {
            let line = line_res.map_err(|_| SciError::InvalidParameter("Error reading line"))?;
            if !line.starts_with('%') {
                dimensions_line = line;
                break;
            }
        }

        // Parse dimensions: M N [L] (L is for coordinate format, ignored in array)
        let dims: Vec<&str> = dimensions_line.split_whitespace().collect();
        if dims.len() < 2 {
            return Err(SciError::InvalidParameter(
                "Invalid dimensions in MatrixMarket file",
            ));
        }

        let rows: usize = dims[0]
            .parse()
            .map_err(|_| SciError::InvalidParameter("Invalid row count"))?;
        let cols: usize = dims[1]
            .parse()
            .map_err(|_| SciError::InvalidParameter("Invalid col count"))?;

        let mut data = vec![0.0; rows * cols];

        if dims.len() == 3 {
            // Coordinate format
            let _entries: usize = dims[2].parse().unwrap_or(0);
            for line_res in lines {
                let line = line_res.unwrap_or_default();
                if line.trim().is_empty() {
                    continue;
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    // Matrix Market is 1-indexed
                    let r: usize = parts[0].parse().unwrap_or(1) - 1;
                    let c: usize = parts[1].parse().unwrap_or(1) - 1;
                    let val: f64 = parts[2].parse().unwrap_or(0.0);
                    if r < rows && c < cols {
                        data[r * cols + c] = val; // Assuming row-major for DynamicMatrix
                    }
                }
            }
        } else {
            // Array format (Column-major by default in MatrixMarket)
            let mut r = 0;
            let mut c = 0;
            for line_res in lines {
                let line = line_res.unwrap_or_default();
                if line.trim().is_empty() {
                    continue;
                }
                let val: f64 = line.trim().parse().unwrap_or(0.0);

                data[r * cols + c] = val;

                r += 1;
                if r == rows {
                    r = 0;
                    c += 1;
                }
            }
        }

        DynamicMatrix::new(rows, cols, data)
    }

    /// Writes a `DynamicMatrix` to a Matrix Market array file.
    pub fn write<P: AsRef<Path>>(path: P, matrix: &DynamicMatrix) -> SciResult<()> {
        let file =
            File::create(path).map_err(|_| SciError::InvalidParameter("Could not create file"))?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "%%MatrixMarket matrix array real general").unwrap();
        writeln!(writer, "% Generated by scies-math-th").unwrap();
        writeln!(writer, "{} {}", matrix.rows(), matrix.cols()).unwrap();

        // Matrix Market expects column-major order for array formats
        for c in 0..matrix.cols() {
            for r in 0..matrix.rows() {
                let val = matrix.get(r, c)?;
                writeln!(writer, "{}", val).unwrap();
            }
        }

        Ok(())
    }
}
