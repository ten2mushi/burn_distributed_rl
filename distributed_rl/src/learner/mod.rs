//! Learner system for Distributed.
//!
//! - `Learner`: Generic learner that performs gradient updates
//!
//! # WGPU Thread Safety
//!
//! With WGPU backend, learners can perform tensor operations directly
//! without synchronization primitives. CubeCL streams provide
//! automatic cross-thread tensor synchronization.

pub mod learner;

#[cfg(test)]
mod tests;

pub use learner::{Learner, LearnerConfig, LearnerHandle};

// Re-export from messages for convenience
pub use crate::messages::{LearnerMsg, LearnerStats};
