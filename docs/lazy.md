# `lazy` Module Documentation

Phase 5B — Lazy matrix expression tree.

Chains operations **without computing intermediate results**.
Only one heap allocation when `.eval()` is called.

# Comparison

```text
// EAGER: 3 temporary Vec allocations
let r = a.scale(2.0).add(&b).unwrap().scale(0.5);

// LAZY: 0 temporaries — single pass at eval()
let r = lazy(&a).scale(2.0).add(lazy(&b)).scale(0.5).eval();
```

# Supported ops

| Op | Method |
|---|---|
| A + B | `.add(other)` |
| A − B | `.sub(other)` |
| A × s | `.scale(s)` |
| −A | `.neg()` |
| Aᵀ | `.transpose()` |
| A ∘ B (hadamard) | `.hadamard(other)` |
| map f(x) | `.map(f)` |
| A · B (matmul) | `.matmul(other)` |
| Fuse A·B + C | `.matmul_add(a, b, c)` |