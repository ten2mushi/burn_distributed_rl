//! Distributed PPO algorithm components.
//!
//! This module provides the PPO-specific implementation for distributed training:
//! - `PPORolloutBuffer`: On-policy buffer with rollout semantics
//! - `DistributedPPOBuffer`: Per-environment buffer for multi-actor training
//! - `RecurrentPPOBuffer`: Per-environment buffer for recurrent training with TBPTT
//! - `DistributedPPO`: Algorithm implementation with GAE and clipped objective

mod ppo_buffer;
mod distributed_ppo;
pub mod distributed_ppo_buffer;
pub mod recurrent_ppo_buffer;

#[cfg(test)]
mod tests;

pub use ppo_buffer::{PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch};
pub use distributed_ppo::{DistributedPPO, PPOProcessedBatch};
pub use distributed_ppo_buffer::{DistributedPPOBuffer, DistributedPPORollouts};
pub use recurrent_ppo_buffer::RecurrentPPOBuffer;
