//! Operant Core - Traits for high-performance parallel RL environments.
//!
//! This crate provides core abstractions for vectorized RL environments:
//!
//! - [`Environment`] - Trait for implementing vectorized environments
//! - [`StepResult`] - Zero-copy step result for efficient data access
//! - [`ResetMask`] - Packed bitmask for selective environment reset
//! - [`LogData`] - Trait for environment metrics tracking

pub mod env;
pub mod error;

pub use env::{Environment, LogData, ResetMask, StepResult};
pub use error::{OperantError, Result};
