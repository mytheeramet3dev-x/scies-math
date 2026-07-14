# `transform` Module Documentation

Geometric transforms for rotation, rigid motion, and spatial composition.

## Overview

This module provides the transform primitives used by geometry, simulation, and 3D workflow code.

## Core types

- `Quaternion`
- `Rotation3`
- `Isometry3`
- `Similarity3`
- `DualQuaternion`

## Notes

- Use quaternions for stable rotation composition.
- Use isometries for rigid transforms that preserve distances.
