# `linear_operator` Module Documentation

Abstract linear operator interface for matrix-free iterative methods.

## Overview

The [`LinearOperator`] trait defines linear maps $y = A x$ and adjoint maps $y = A^T x$ without requiring concrete full dense storage.

Implementations:
- `DynamicMatrix`
- `SymmetricMatrix`
- `SparseMatrixCsr`
- `SumOperator`
