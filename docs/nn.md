# `nn` Module Documentation

Neural network primitives.

## Overview

This module provides structures to build, train, and evaluate feed-forward neural networks.
It uses high-performance matrix operations from `linear_algebra` and implements backpropagation with the Adam optimizer.

## Features
- **Layers**: Dense (Fully Connected)
- **Activations**: Linear, ReLU, Sigmoid, Tanh
- **Loss Functions**: Mean Squared Error (MSE)
- **Optimizers**: Adam, SGD

## Notes

- Keep this module lightweight and predictable rather than attempting to replace a full ML framework.
- The core design aims to support small-to-medium scientific models cleanly.
