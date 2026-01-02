//! SIMD RF Primitives
//!
//! This module provides SIMD-optimized primitives for RF signal processing:
//!
//! - [`math`]: Transcendental functions (exp, log, sin, cos, etc.)
//! - [`complex`]: Complex number arithmetic for baseband signals
//! - [`random`]: High-performance vectorized random number generation
//! - [`helpers`]: Memory access and geometric utilities

pub mod complex;
pub mod helpers;
pub mod math;
pub mod random;

// Common type aliases for convenience
use std::simd::{f32x8, u32x8, u64x8};

/// SIMD lane width - 8 f32 values processed in parallel
pub const SIMD_LANES: usize = 8;

/// Type alias for 8-wide f32 SIMD vector
pub type F32x8 = f32x8;

/// Type alias for 8-wide u32 SIMD vector
pub type U32x8 = u32x8;

/// Type alias for 8-wide u64 SIMD vector
pub type U64x8 = u64x8;
