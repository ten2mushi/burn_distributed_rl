//! SAC configuration re-exports for runner convenience.
//!
//! This module provides convenient access to SAC configuration
//! from the runners module, similar to how PPO configuration
//! is accessible via `runners::PPOConfig`.

pub use crate::algorithms::sac::{SACConfig, SACStats};
