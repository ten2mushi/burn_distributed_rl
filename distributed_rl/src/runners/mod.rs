//! Runner implementations for RL training.
//!
//! This module provides high-performance multi-actor runner implementations:
//!
//! # PPO Runners
//!
//! - [`PPORunner`]: Runner supporting both feed-forward and recurrent policies
//! - [`PPODiscrete`], [`PPOContinuous`]: Feed-forward PPO type aliases
//! - [`RecurrentPPODiscrete`], [`RecurrentPPOContinuous`]: Recurrent PPO type aliases
//!
//! # IMPALA Runners
//!
//! - [`IMPALARunner`]: Multi-actor IMPALA with V-trace correction
//! - [`IMPALADiscrete`]: IMPALA with discrete actions
//! - [`IMPALAContinuous`]: IMPALA with continuous actions
//! - [`RecurrentIMPALADiscrete`]: Recurrent IMPALA with discrete actions
//! - [`RecurrentIMPALAContinuous`]: Recurrent IMPALA with continuous actions
//!
//! # Architecture
//!
//! All runners coordinate:
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Shared buffer for experience transfer
//! - BytesSlot for lock-free model synchronization
//!
//! # PPO Training Strategies
//!
//! The PPO runner uses compile-time strategy dispatch:
//! - [`FeedForwardStrategy`]: Standard PPO with shuffled minibatches
//! - [`RecurrentStrategy`]: TBPTT-based training for LSTM/GRU policies
//!
//! Hidden state management is handled automatically based on the temporal policy type.

pub mod ppo_config;
pub mod sac_config;
pub mod impala_runner;
pub mod runner;
pub mod learner;
pub mod rollout_storage;
pub mod ppo_strategies;
pub mod sac_strategies;
pub mod ppo_runner;
pub mod sac_runner;

#[cfg(test)]
pub mod tests;

// PPO Configuration (shared by all PPO runners)
pub use ppo_config::{
    PPOConfig, PPOStats,
    EntropySchedule, ObsNormalizationConfig, PopArtConfig,
};

// IMPALA (high-level API) - uses config from algorithms/impala
pub use crate::algorithms::impala::{IMPALAConfig, IMPALAStats};
pub use impala_runner::{
    IMPALARunner,
    IMPALADiscrete,
    IMPALAContinuous,
    RecurrentIMPALADiscrete,
    RecurrentIMPALAContinuous,
};

// Generic runner (for custom implementations)
pub use runner::{Runner, RunnerConfig, RunnerBuilder, TrainingStats};

// Single-threaded learner
pub use learner::{
    Learner, LearnerConfig, LearnerStats, StepResult, VectorizedEnv,
};
pub use rollout_storage::{
    extract_minibatch, generate_minibatches, ComputedValues, MinibatchData, MinibatchIndices,
    PreparedSequences, RolloutStorage, Sequence, SequenceBatch,
};

// PPO Runner (primary API with strategy pattern)
pub use ppo_runner::{
    PPORunner,
    PPODiscrete,
    PPOContinuous,
    RecurrentPPODiscrete,
    RecurrentPPOContinuous,
};

// Training strategies
pub use ppo_strategies::{
    PPOTrainingStrategy,
    FeedForwardStrategy,
    RecurrentStrategy,
    DefaultStrategy,
    TemporalPolicyStrategy,
    PreparedTrainingData,
    SequenceInfo,
};

// SAC Configuration (shared by all SAC runners)
pub use sac_config::{SACConfig, SACStats};

// SAC Runner (primary API with strategy pattern)
pub use sac_runner::{
    SACRunner,
    SACDiscrete,
    SACContinuous,
    RecurrentSACDiscrete,
    RecurrentSACContinuous,
};

// SAC Training strategies
pub use sac_strategies::{
    SACTrainingStrategy,
    FeedForwardSACStrategy,
    RecurrentSACStrategy,
    DefaultSACStrategy,
    TemporalPolicySACStrategy,
    SACLossInfo,
};
