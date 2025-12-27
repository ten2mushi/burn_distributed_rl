//! Algorithm components for Distributed.
//!
//! - `vtrace`: V-trace off-policy correction for IMPALA
//! - `gae`: Generalized Advantage Estimation for PPO
//! - `policy_loss`: Policy gradient loss functions (discrete)
//! - `continuous_policy`: Squashed Gaussian utilities for continuous actions
//! - `action_policy`: Unified action policy abstraction (discrete/continuous)
//! - `temporal_policy`: Feed-forward vs recurrent abstraction
//! - `actor_critic`: Unified actor-critic model trait
//! - `algorithm`: Algorithm trait for loss computation (PPO, A2C)
//! - `distributed_algorithm`: Distributed algorithm trait with buffer management
//! - `ppo`: Distributed PPO implementation
//! - `impala`: Distributed IMPALA implementation

pub mod action_policy;
pub mod actor_critic;
pub mod algorithm;
pub mod continuous_policy;
pub mod distributed_algorithm;
pub mod gae;
pub mod impala;
pub mod ppo;
pub mod policy_loss;
pub mod temporal_policy;
pub mod vtrace;

#[cfg(test)]
mod tests;

pub use action_policy::{
    ActionPolicy, ActionValue, ContinuousAction, ContinuousPolicy, ContinuousPolicyOutput,
    DiscreteAction, DiscretePolicy, DiscretePolicyOutput, PolicyOutput,
};
pub use temporal_policy::{
    FeedForward, HiddenConfig, HiddenStateType, Recurrent, RecurrentHidden, TemporalPolicy,
};
pub use continuous_policy::{
    entropy_gaussian, entropy_loss_gaussian, log_prob_squashed_gaussian, ppo_combined_loss_continuous,
    ppo_policy_loss_continuous, sample_gaussian, sample_squashed_gaussian, scale_action, unscale_action,
};
pub use gae::{compute_gae, compute_gae_vectorized, normalize_advantages};
pub use policy_loss::{entropy_loss, ppo_clip_loss, value_loss};
pub use vtrace::{compute_vtrace, compute_vtrace_batch, VTraceInput, VTraceResult};
pub use actor_critic::{
    ActorCritic, ActorCriticInference, ForwardOutput, forward_output_discrete_ff, forward_output_continuous_ff,
    forward_output_discrete_recurrent, forward_output_continuous_recurrent,
};
pub use algorithm::{
    Algorithm, LossOutput, PPOAlgorithm, PPOConfig, A2CAlgorithm, A2CConfig,
};
pub use distributed_algorithm::{
    DistributedAlgorithm, DistributedPPOConfig,
    OnPolicyDistributed, OffPolicyDistributed,
};
pub use ppo::{DistributedPPO, PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch, PPOProcessedBatch};
pub use impala::{
    DistributedIMPALA, IMPALABuffer, IMPALABufferConfig, IMPALABatch, IMPALAProcessedBatch,
    IMPALAConfig, IMPALAStats,
};
