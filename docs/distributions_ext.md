# `distributions_ext` Module Documentation

Extended distributions: Pareto, Laplace, Logistic, Gumbel, Rayleigh,
Truncated Normal, Triangular, Inverse Gaussian, Von Mises, Maxwell-Boltzmann,
Chi, Multinomial, Beta-Binomial, plus missing inverse CDFs.

## Overview

This module expands the distribution catalog with additional continuous and discrete families, plus supporting inverse-CDF routines.

## Topics

- heavy-tailed distributions
- bounded distributions
- circular distributions
- discrete and compound-style helpers
- inverse CDF coverage for distributions that are missing it in the core module

## Notes

- Prefer the core `distributions` module for the most common families.
- Use this extension module when a model requires a less common distribution family.
