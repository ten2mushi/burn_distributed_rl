//! PPO algorithm components for training.
//!
//! This module provides the PPO-specific implementation:
//!
//! # Core Components
//!
//! - [`PPORolloutBuffer`]: On-policy buffer with rollout semantics
//! - [`PPO`]: Algorithm implementation with GAE and clipped objective
//!
//! # Types
//!
//! - [`PPOTransition<H>`]: Generic transition type for both feed-forward and recurrent
//! - [`PPOBuffer<Trans>`]: Generic buffer for any PPO transition type
//! - [`PPOTransitionTrait`]: Trait for buffer compatibility
//!
//! # Normalization (Advanced)
//!
//! - [`normalization::ObservationNormalizer`]: Running mean/std observation normalization
//! - [`normalization::RewardNormalizer`]: Return-based reward normalization
//! - [`normalization::PopArt`]: Adaptive value target normalization
//!
//! # Type Aliases
//!
//! ## Feed-Forward
//! - [`PPOTransitionFF`]: Transition with no hidden state
//! - [`PPOBufferFF`]: Buffer for feed-forward PPO
//!
//! ## Recurrent
//! - [`PPOTransitionRecurrent`]: Transition with hidden state
//! - [`PPOBufferRecurrent`]: Buffer for recurrent PPO

mod ppo_batch_buffer;
mod ppo;
pub mod ppo_transition;
pub mod ppo_buffer;
pub mod normalization;
pub mod prioritization;

#[cfg(test)]
mod tests;

// Core PPO components
pub use ppo_batch_buffer::{PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch};
pub use ppo::{PPO, PPOProcessedBatch};

// PPO types
pub use ppo_transition::{
    HiddenData, RecurrentHiddenData, PPOTransition,
    PPOTransitionFF, PPOTransitionRecurrent, PPOTransitionTrait,
};
pub use ppo_buffer::{
    PPOBuffer, PPOBufferFF, PPOBufferRecurrent, PPORollouts,
};
