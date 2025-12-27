//! Shared utilities for vectorized environments.

pub mod rng;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;
