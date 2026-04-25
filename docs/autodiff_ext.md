# `autodiff_ext` Module Documentation

Advanced autodiff utilities built on top of `autodiff` (forward) and `reverse_ad`.

| Feature | Functions |
|---|---|
| Forward-mode extras | `directional_deriv`, `jvp`, full Jacobian matrix |
| Reverse-mode extras | `vjp`, full Jacobian (reverse), gradient norm |
| Vector calculus | `divergence`, `curl_2d`, `curl_3d`, `laplacian` |
| Verification | `gradient_check` (finite-diff vs AD) |
| Higher-order | `second_deriv`, `third_deriv` (forward stacking) |