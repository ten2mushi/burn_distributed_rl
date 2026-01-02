//! Normalization utilities for PPO training.
//!
//! This module provides normalization techniques that improve PPO training stability:
//!
//! - [`ObservationNormalizer`]: Running mean/std normalization for observations
//! - [`RewardNormalizer`]: Return-based reward normalization
//! - [`PopArt`]: Preserving Outputs Precisely while Adaptively Rescaling Targets
//!
//! # Usage
//!
//! Normalization is enabled via config:
//!
//! ```ignore
//! let config = PPOAlgorithmConfig::new()
//!     .with_obs_normalization(Some((-10.0, 10.0)))  // Clip after normalizing
//!     .with_reward_normalization(true);
//! ```

mod observation_normalizer;
mod reward_normalizer;
mod popart;

pub use observation_normalizer::{
    ObservationNormalizer, SharedObservationNormalizer, ObsNormalizationConfig,
};
pub use reward_normalizer::{RewardNormalizer, SharedRewardNormalizer};
pub use popart::{PopArt, PopArtConfig};
