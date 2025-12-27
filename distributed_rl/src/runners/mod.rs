//! Runner implementations for Distributed RL training.
//!
//! This module provides high-performance multi-actor runner implementations:
//!
//! # Distributed Runners
//!
//! - [`DistributedPPORunner`]: Multi-actor PPO with N actors + 1 learner
//! - [`DistributedRecurrentPPORunner`]: Multi-actor recurrent PPO with TBPTT
//! - [`DistributedIMPALARunner`]: Multi-actor IMPALA with V-trace correction
//!
//! # Type Aliases
//!
//! ## PPO (Feed-Forward)
//! - [`DistributedPPODiscrete`]: PPO with discrete actions
//! - [`DistributedPPOContinuous`]: PPO with continuous actions
//!
//! ## Recurrent PPO
//! - [`DistributedRecurrentPPODiscrete`]: Recurrent PPO with discrete actions
//! - [`DistributedRecurrentPPOContinuous`]: Recurrent PPO with continuous actions
//!
//! ## IMPALA
//! - [`DistributedIMPALADiscrete`]: IMPALA with discrete actions
//! - [`DistributedIMPALAContinuous`]: IMPALA with continuous actions
//! - [`DistributedRecurrentIMPALADiscrete`]: Recurrent IMPALA with discrete actions
//! - [`DistributedRecurrentIMPALAContinuous`]: Recurrent IMPALA with continuous actions
//!
//! # Architecture
//!
//! All distributed runners coordinate:
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Shared buffer for experience transfer
//! - BytesSlot for lock-free model synchronization
//!
//! # Recurrent vs Feed-Forward
//!
//! Use `DistributedRecurrentPPORunner` for recurrent policies (LSTM/GRU).
//! The recurrent runner properly handles:
//! - Hidden state persistence across timesteps
//! - Hidden state reset on episode termination
//! - Sequence-based TBPTT training

pub mod distributed_ppo_config;
pub mod distributed_ppo_runner;
pub mod distributed_recurrent_ppo_runner;
pub mod distributed_impala_runner;
pub mod distributed_runner;
pub mod learner;
pub mod rollout_storage;

#[cfg(test)]
pub mod tests;

// Distributed PPO - Feed-Forward (new high-level API)
pub use distributed_ppo_config::{DistributedPPOConfig, DistributedPPOStats};
pub use distributed_ppo_runner::{
    DistributedPPORunner,
    DistributedPPODiscrete,
    DistributedPPOContinuous,
};

// Distributed PPO - Recurrent (new high-level API with proper TBPTT)
pub use distributed_recurrent_ppo_runner::{
    DistributedRecurrentPPORunner,
    DistributedRecurrentPPODiscreteRunner as DistributedRecurrentPPODiscrete,
    DistributedRecurrentPPOContinuousRunner as DistributedRecurrentPPOContinuous,
};

// Distributed IMPALA (new high-level API) - uses unified config from algorithms/impala
pub use crate::algorithms::impala::{IMPALAConfig as DistributedIMPALAConfig, IMPALAStats as DistributedIMPALAStats};
pub use distributed_impala_runner::{
    DistributedIMPALARunner,
    DistributedIMPALADiscrete,
    DistributedIMPALAContinuous,
    DistributedRecurrentIMPALADiscrete,
    DistributedRecurrentIMPALAContinuous,
};

// Generic distributed runner (for custom implementations)
pub use distributed_runner::{DistributedRunner, DistributedRunnerConfig, DistributedRunnerBuilder, TrainingStats};

// Single-threaded learner
pub use learner::{
    Learner, LearnerConfig, LearnerStats, PPOContinuous, PPODiscrete, RecurrentPPOContinuous,
    RecurrentPPODiscrete, StepResult, VectorizedEnv,
};
pub use rollout_storage::{
    extract_minibatch, generate_minibatches, ComputedValues, MinibatchData, MinibatchIndices,
    PreparedSequences, RolloutStorage, Sequence, SequenceBatch,
};
