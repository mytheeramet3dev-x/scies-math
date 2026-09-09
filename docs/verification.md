# `verification` Module Documentation

Mathematical derivation provenance, verification statuses, and Lean 4 / Mathlib4 theorem export bridge.

## Overview

Tracks mathematical provenance and links computational results with formal theorem provers.

| Type / Function | Description |
|---|---|
| [`VerificationStatus`] | Verification state (Unchecked, NumericallyTested, IntervalBounded, SymbolicallyDerived, FormallyVerified, Refuted) |
| [`DerivationTree`] | Step-by-step mathematical proof trace with assumptions and justifications |
| [`export_lean4_theorem`] | Code generator exporting declarations and theorem statements to Lean 4 + Mathlib4 |
