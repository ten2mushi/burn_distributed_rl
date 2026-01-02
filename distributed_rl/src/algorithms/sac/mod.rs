//! SAC (Soft Actor-Critic) algorithm implementation.
//!
//! SAC is an off-policy maximum entropy RL algorithm that:
//! - Maximizes expected return plus entropy bonus
//! - Uses twin Q-networks for stable value estimation
//! - Supports automatic entropy coefficient tuning
//!
//! # Architecture
//!
//! Unlike PPO/IMPALA which use a combined ActorCritic model, SAC uses
//! separate actor and critic networks:
//!
//! ```text
//! Actor Network (separate encoder)
//! ├── Encoder (MLP/RNN)
//! └── Policy head → actions + log_probs
//!
//! Critic Networks (separate encoders, twin Q)
//! ├── Q1: Encoder1 → Q-value(s)
//! ├── Q2: Encoder2 → Q-value(s)
//! ├── Q1_target: frozen copy of Q1
//! └── Q2_target: frozen copy of Q2
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::algorithms::sac::{SACConfig, SAC};
//! use distributed_rl::runners::{SACDiscrete, SACContinuous};
//!
//! // Discrete action space (e.g., Atari)
//! let config = SACConfig::discrete();
//! let runner = SACDiscrete::<MyBackend>::feed_forward(config);
//!
//! // Continuous action space (e.g., MuJoCo)
//! let config = SACConfig::continuous();
//! let runner = SACContinuous::<MyBackend>::feed_forward(config);
//! ```

mod config;
mod entropy_tuning;
mod sac;
mod sac_actor;
mod sac_buffer;
mod sac_critic;
mod sac_transition;

// Re-exports
pub use config::{SACConfig, SACStats};
pub use entropy_tuning::{
    categorical_entropy, gaussian_entropy, target_entropy_continuous, target_entropy_discrete,
    EntropyTuner,
};
pub use sac::{
    sac_actor_loss, sac_critic_loss, sac_td_targets, SACLossOutput, SAC,
};
pub use sac_actor::{
    clamp_log_std, SACActorDeterministic, SACActorOutput, SACActorTraining, SACActor,
    LOG_STD_MAX, LOG_STD_MIN,
};
pub use sac_buffer::{SACBuffer, SACBufferConfig};
pub use sac_critic::{
    compute_td_target, expected_q_discrete, gather_q_values, target_value_discrete,
    SACCriticOutput, SACCriticTraining, SACCritic,
};
pub use sac_transition::{SACBatch, SACDataMarker, SACRecurrentData, SACRecurrentTransition, SACTransition, SACTransitionTrait};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pipeline_debug_tests;

#[cfg(test)]
mod buffer_diagnosis;

#[cfg(test)]
mod strategy_diagnosis;
