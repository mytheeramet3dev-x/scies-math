#![allow(
    clippy::should_implement_trait,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::manual_contains,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::type_complexity,
    clippy::redundant_closure,
    clippy::chunks_exact_to_as_chunks,
    clippy::excessive_precision,
    clippy::assign_op_pattern,
    clippy::approx_constant,
    clippy::new_without_default,
    clippy::unwrap_or_default,
    clippy::let_and_return,
    clippy::ptr_arg,
    clippy::double_ended_iterator_last,
    clippy::too_many_arguments,
    clippy::needless_borrows_for_generic_args
)]

pub mod algebra;
pub mod autodiff;
pub mod autodiff_ext;
pub mod calculus;
pub mod complex;
pub mod complex_ext;
pub mod distributions;
pub mod distributions_ext;
pub mod eigensystem;
pub mod eigensystem_ext;
pub mod fft_advanced;
pub mod fitting;
pub mod geometry;
pub mod inference;
pub mod inference_ext;
pub mod interpolation;
pub mod linear_algebra;
pub mod monte_carlo;
pub mod monte_carlo_ext;
pub mod multivariate;
pub mod numerical;
pub mod ode;
pub mod ode_ext;
pub mod opt_multivar;
pub mod opt_multivar_ext;
pub mod optimization;
pub mod pde;
pub mod pde_advanced;
pub mod pde_ext;
pub mod probability;
pub mod probability_ext;
pub mod regression;
pub mod regression_ext;
pub mod reverse_ad;
pub mod rng;
pub mod signal;
pub mod signal_ext;
pub mod sparse;
pub mod statistics;
pub mod tensor;
pub mod timeseries;

pub mod errors;

pub mod perf;

pub mod generic;

pub mod transform;

pub mod rng_ext;

pub mod lazy;

pub mod exact;
pub mod geometry_ext;
pub mod graph;
pub mod io;
pub mod linear_operator;
pub mod moment;
pub mod nn;
pub mod policy;
pub mod sdp;
pub mod sdp_ipm;
pub mod sdp_primitives;
pub mod serde_support;
pub mod special_functions;
pub mod symbolic;
pub mod symmetric;
pub mod uncertainty;
pub mod units;
pub mod verification;
