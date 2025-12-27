//! Experience buffer implementations for Distributed.
//!
//! Different algorithms require different buffer semantics:
//! - `RolloutBuffer`: On-policy (PPO) - consumed after each training iteration
//! - `TrajectoryStore`: IMPALA - trajectory-based with FIFO consumption
//! - `SequenceBuffer`: Recurrent PPO - sequence-based for TBPTT training

pub mod buffer_traits;
pub mod rollout_buffer;
pub mod sequence_buffer;
pub mod trajectory_store;

pub use buffer_traits::{ExperienceBuffer, OnPolicyBuffer, TrajectoryBuffer};
pub use rollout_buffer::{RolloutBatch, RolloutBuffer, RolloutBufferConfig};
pub use sequence_buffer::{Sequence, SequenceBatch, SequenceBuffer, SequenceBufferConfig};
pub use trajectory_store::{TrajectoryBatch, TrajectoryStore, TrajectoryStoreConfig};

#[cfg(test)]
mod tests;
