//! # Distributed: True Multi-Actor Distributed RL
//!
//! Production-grade distributed training framework for on-policy (PPO) and
//! off-policy corrected (IMPALA) reinforcement learning algorithms.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Distributed PPO/IMPALA                            │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Thread 1          Thread 2          Thread N                       │
//! │  ┌────────┐        ┌────────┐        ┌────────┐                    │
//! │  │Actor 0 │        │Actor 1 │        │Actor N │                    │
//! │  │ envs   │        │ envs   │        │ envs   │                    │
//! │  │CubeCL  │        │CubeCL  │        │CubeCL  │                    │
//! │  │Stream0 │        │Stream1 │        │StreamN │                    │
//! │  └───┬────┘        └───┬────┘        └───┬────┘                    │
//! │      │                 │                 │                          │
//! │      └─────────────────┼─────────────────┘                          │
//! │                        ▼                                            │
//! │              ┌─────────────────┐      ┌──────────────┐             │
//! │              │ ExperienceBuffer│      │  ModelSlot   │             │
//! │              │ (Lock-free)     │      │ (Swap sync)  │             │
//! │              └────────┬────────┘      └──────┬───────┘             │
//! │                       ▼                      │                      │
//! │              ┌─────────────────┐             │                      │
//! │              │ Learner Thread  │◄────────────┘                      │
//! │              │ (GPU training)  │                                    │
//! │              │ CubeCL StreamL  │                                    │
//! │              └─────────────────┘                                    │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Thread Safety (WGPU)
//!
//! This framework uses the WGPU backend with CubeCL streams for automatic
//! cross-thread tensor synchronization. Each thread gets its own CubeCL
//! stream, and Burn handles all synchronization automatically.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use distributed_rl::{IMPALAConfig, DistributedIMPALADiscrete};
//!
//! let config = IMPALAConfig::new()
//!     .with_n_actors(4)
//!     .with_n_envs_per_actor(64)
//!     .with_trajectory_length(20)
//!     .with_gamma(0.99);
//!
//! let runner: DistributedIMPALADiscrete<B> = DistributedIMPALARunner::new(config);
//! let optimizer = runner.create_optimizer::<MyModel>();
//! runner.run(model_factory, initial_model, env_factory, optimizer, callback);
//! ```

pub mod core;
pub mod messages;
pub mod actors;
pub mod buffers;
pub mod algorithms;
pub mod learner;
pub mod runners;
pub mod metrics;
pub mod environment;
pub mod scheduling;
pub mod checkpoint;

// Re-export commonly used types
pub use core::transition::{Action, Transition, PPOTransition, IMPALATransition, Trajectory};
pub use core::model_version::{VersionCounter, VersionedModel, SharedVersionCounter, version_counter};
pub use core::model_slot::{ModelSlot, SharedModelSlot, model_slot, model_slot_with};
pub use core::record_slot::{RecordSlot, SharedRecordSlot, record_slot, record_slot_with};
pub use core::shared_buffer::{SharedBuffer, SharedReplayBuffer, shared_buffer};
pub use core::experience_buffer::{
    ExperienceBuffer, OnPolicyBuffer, OffPolicyBuffer,
    SharedExperienceBuffer, BufferConfig,
};

// Message types for distributed communication
pub use messages::{ActorMsg, ActorStats as MsgActorStats, LearnerMsg, LearnerStats as MsgLearnerStats};
pub use messages::{CoordinatorMsg, FinishReason, EvalMsg, EvalResult};



pub use buffers::rollout_buffer::{RolloutBuffer, RolloutBufferConfig, RolloutBatch};
pub use buffers::trajectory_store::{TrajectoryStore, TrajectoryStoreConfig, TrajectoryBatch};

pub use actors::actor::{Actor, ActorConfig, ActorHandle};
pub use actors::actor_pool::{ActorPool, ActorPoolConfig};

pub use learner::learner::{Learner, LearnerConfig, LearnerHandle};

// Generic distributed runner (for custom implementations)
pub use runners::distributed_runner::{DistributedRunner, DistributedRunnerConfig, DistributedRunnerBuilder, TrainingStats};

// New generic Learner exports
pub use runners::learner::{
    Learner as GenericLearner, LearnerConfig as GenericLearnerConfig, LearnerStats,
    VectorizedEnv as GenericVectorizedEnv, StepResult as LearnerStepResult,
    PPODiscrete, PPOContinuous, RecurrentPPODiscrete, RecurrentPPOContinuous,
};

// Distributed algorithms
pub use algorithms::{
    DistributedAlgorithm, DistributedPPOConfig, IMPALAConfig, IMPALAStats,
    OnPolicyDistributed, OffPolicyDistributed,
    DistributedPPO, PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch, PPOProcessedBatch,
    DistributedIMPALA, IMPALABuffer, IMPALABufferConfig, IMPALABatch, IMPALAProcessedBatch,
};

// Distributed runners
pub use runners::{
    DistributedIMPALARunner, DistributedIMPALADiscrete, DistributedIMPALAContinuous,
    DistributedRecurrentIMPALADiscrete, DistributedRecurrentIMPALAContinuous,
    DistributedPPORunner, DistributedPPODiscrete, DistributedPPOContinuous,
    DistributedRecurrentPPORunner, DistributedRecurrentPPODiscrete, DistributedRecurrentPPOContinuous,
};

pub use metrics::training_metrics::{TrainingMetrics, SharedTrainingMetrics, training_metrics};
pub use metrics::logger::{TrainingSnapshot, MetricsLogger, ConsoleLogger, CSVLogger, MultiLogger, ProgressLogger};

// Environment abstraction
pub use environment::{
    // IMPALA environment trait
    VectorizedEnv, StepResult, ResetMask, WrapperStepResult,
    // Low-level wrappers
    OperantAdapter, CartPoleEnvWrapper, PendulumEnvWrapper, wrap_cartpole, wrap_pendulum,
    // Ready-to-use environment adapters for generic Learner
    CartPoleEnv, PendulumEnv,
};

// Learning rate scheduling
pub use scheduling::{LRScheduler, ConstantLR, LinearDecay, CosineAnnealing, PolynomialDecay, Warmup};

// Model checkpointing
pub use checkpoint::{Checkpointer, CheckpointerConfig, CheckpointInfo, CheckpointError};
