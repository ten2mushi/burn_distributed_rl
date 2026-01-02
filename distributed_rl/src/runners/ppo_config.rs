//! Configuration for PPO training.
//!
//! This module provides configuration types for multi-actor PPO training
//! with WGPU backend and CubeCL streams.

use std::fmt;

// Re-export normalization configs
pub use crate::algorithms::ppo::normalization::{
    ObsNormalizationConfig, PopArtConfig,
};

/// Entropy coefficient scheduling strategy.
///
/// Controls how the entropy coefficient changes over training.
#[derive(Debug, Clone, PartialEq)]
pub enum EntropySchedule {
    /// Constant entropy coefficient throughout training.
    Constant,
    /// Linear decay from initial value to floor over training.
    Linear { floor: f32 },
    /// Exponential decay: coef = initial * decay_rate^progress + floor.
    Exponential { decay_rate: f32, floor: f32 },
    /// Adaptive entropy based on target entropy (SAC-style, future).
    Adaptive { target_entropy: f32 },
}

impl Default for EntropySchedule {
    fn default() -> Self {
        EntropySchedule::Constant
    }
}

impl EntropySchedule {
    /// Get the entropy coefficient at the given progress (0.0 to 1.0).
    pub fn get_coef(&self, initial_coef: f32, progress: f32) -> f32 {
        let progress = progress.clamp(0.0, 1.0);
        match self {
            EntropySchedule::Constant => initial_coef,
            EntropySchedule::Linear { floor } => {
                // Linear interpolation from initial_coef to floor
                initial_coef + (floor - initial_coef) * progress
            }
            EntropySchedule::Exponential { decay_rate, floor } => {
                // Exponential decay: initial * decay_rate^progress + floor
                // decay_rate < 1 for decay, e.g., 0.1 means fast decay
                floor + (initial_coef - floor) * decay_rate.powf(progress * 10.0)
            }
            EntropySchedule::Adaptive { target_entropy: _ } => {
                // Adaptive entropy is handled separately by the learner
                // Return initial for now
                initial_coef
            }
        }
    }
}

/// Configuration validation error.
///
/// Returned when configuration parameters are invalid or inconsistent.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// A count parameter (n_actors, rollout_length, etc.) must be positive.
    InvalidCount {
        field: &'static str,
        value: usize,
    },
    /// A parameter is outside its valid range.
    OutOfRange {
        field: &'static str,
        value: f32,
        min: f32,
        max: f32,
    },
    /// Minibatch configuration is invalid (transitions < n_minibatches).
    InvalidMinibatch {
        transitions: usize,
        minibatches: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidCount { field, value } => {
                write!(f, "{} must be > 0, got {}", field, value)
            }
            ConfigError::OutOfRange { field, value, min, max } => {
                write!(f, "{} must be in [{}, {}], got {}", field, min, max, value)
            }
            ConfigError::InvalidMinibatch { transitions, minibatches } => {
                write!(
                    f,
                    "transitions_per_rollout ({}) must be >= n_minibatches ({})",
                    transitions, minibatches
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Advantage normalization strategy.
///
/// Controls how advantages are normalized during PPO training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvantageNormalization {
    /// No normalization applied.
    None,
    /// Normalize per minibatch (current behavior, higher variance).
    #[default]
    PerMinibatch,
    /// Normalize globally across entire rollout (lower variance).
    /// Computes mean/std once before epoch training begins.
    Global,
}

/// Configuration for Distributed PPO training.
///
/// This configuration supports N actor threads + 1 learner thread
/// architecture with per-environment rollout collection and GAE.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    // Multi-actor settings
    /// Number of actor threads
    pub n_actors: usize,
    /// Environments per actor (vectorized)
    pub n_envs_per_actor: usize,

    // PPO settings
    /// Steps per rollout per environment
    pub rollout_length: usize,
    /// Training epochs per rollout
    pub n_epochs: usize,
    /// Number of minibatches to split rollout into
    pub n_minibatches: usize,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda parameter
    pub gae_lambda: f32,
    /// PPO clipping ratio
    pub clip_ratio: f32,
    /// Value function loss coefficient
    pub vf_coef: f32,
    /// Entropy bonus coefficient
    pub entropy_coef: f32,
    /// Whether to normalize advantages
    pub normalize_advantages: bool,
    /// Advantage normalization strategy (only used when normalize_advantages is true).
    /// PerMinibatch normalizes each minibatch independently (higher variance).
    /// Global pre-computes stats across entire rollout (lower variance).
    pub advantage_normalization: AdvantageNormalization,
    /// Whether to clip value function loss (default: true).
    /// Prevents large value function updates that can destabilize training.
    pub clip_vloss: bool,
    /// Whether to anneal learning rate linearly to 0 over training.
    /// default: true. Helps with fine-tuning at end of training.
    pub anneal_lr: bool,
    /// Target KL divergence for early stopping epochs.
    /// If approx KL exceeds this threshold, remaining minibatches in the epoch are skipped.
    /// default: None (disabled), but commonly set to 0.01-0.02 for stability.
    /// This prevents over-updating the policy on a single rollout.
    pub target_kl: Option<f64>,

    // Recurrent-specific settings
    /// Truncated backpropagation through time sequence length.
    /// Only used by DistributedRecurrentPPORunner.
    /// Shorter = more stable gradients, less temporal learning.
    /// Longer = better long-term dependencies, risk of gradient explosion.
    /// Default: 32.
    pub tbptt_length: usize,

    // Training settings
    /// Learning rate (used for both actor and critic if actor_lr/critic_lr not set)
    pub learning_rate: f64,
    /// Actor-specific learning rate (overrides learning_rate for policy network).
    /// Default: None (uses learning_rate).
    pub actor_lr: Option<f64>,
    /// Critic-specific learning rate (overrides learning_rate for value network).
    /// Default: None (uses learning_rate).
    pub critic_lr: Option<f64>,
    /// Number of critic updates per actor update.
    /// Higher values strengthen the value function relative to the policy.
    /// Default: 1 (equal updates).
    pub critic_update_ratio: usize,
    /// Maximum gradient norm (None = no clipping)
    pub max_grad_norm: Option<f32>,
    /// Maximum environment steps (stopping condition)
    pub max_env_steps: usize,
    /// Target reward for early stopping (None = no early stopping)
    pub target_reward: Option<f32>,
    /// Logging interval in seconds
    pub log_interval_secs: f32,
    /// Model publish frequency (learner steps between weight updates to actors)
    pub model_publish_freq: usize,

    // Advanced normalization settings

    /// Observation normalization configuration.
    /// When enabled, observations are normalized using running mean/std statistics.
    /// Default: None (disabled).
    pub obs_normalization: Option<ObsNormalizationConfig>,
    /// PopArt value normalization configuration.
    /// Normalizes value targets while preserving output consistency.
    /// Default: None (disabled).
    pub popart: Option<PopArtConfig>,
    /// Reward normalization based on return standard deviation.
    /// Default: false.
    pub reward_normalization: bool,
    /// Entropy coefficient scheduling strategy.
    /// Default: Constant (no scheduling).
    pub entropy_schedule: EntropySchedule,
    /// Target network soft update coefficient (tau).
    /// When set, uses a target value network for more stable bootstrapping.
    /// Target weights: target = tau * online + (1 - tau) * target.
    /// Default: None (disabled).
    pub target_network_tau: Option<f32>,
    /// Minimum log standard deviation for continuous policies.
    /// Prevents exploration collapse by ensuring minimum std = exp(floor).
    /// exp(-2.0) = 0.135, so std is at least 13.5% of mean.
    /// Default: -2.0.
    pub log_std_floor: f32,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            // Multi-actor defaults
            n_actors: 4,
            n_envs_per_actor: 32,

            // PPO defaults
            rollout_length: 64,
            n_epochs: 4,
            n_minibatches: 8,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_ratio: 0.2,
            vf_coef: 0.5,
            entropy_coef: 0.01,
            normalize_advantages: true,
            advantage_normalization: AdvantageNormalization::Global,
            clip_vloss: true,
            anneal_lr: true,
            target_kl: None,   // default (disabled), set to ~0.01-0.02 for stability

            // Recurrent-specific defaults
            tbptt_length: 128,

            // Training defaults
            learning_rate: 3e-4,
            actor_lr: None,  // Use learning_rate
            critic_lr: None, // Use learning_rate
            critic_update_ratio: 1,
            max_grad_norm: Some(0.5),
            max_env_steps: 1_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_publish_freq: 1,

            // Advanced normalization defaults (all disabled)
            obs_normalization: None,
            popart: None,
            reward_normalization: false,
            entropy_schedule: EntropySchedule::Constant,
            target_network_tau: None,
            // Continuous policy defaults
            log_std_floor: -2.0, // exp(-2) = 0.135, prevents exploration collapse
        }
    }
}

impl PPOConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for continuous control environments.
    ///
    /// This preset enables settings that are typically beneficial for continuous
    /// action spaces (e.g., Pendulum, HalfCheetah, Humanoid):
    ///
    /// - **Reward normalization**: Enabled (critical for environments with
    ///   large-magnitude rewards like Pendulum's -2000 range)
    /// - **Observation normalization**: Enabled with sensible defaults
    /// - **Entropy coefficient**: 0.0 (continuous policies use std for exploration)
    /// - **Learning rate**: 3e-4 (standard for continuous control)
    ///
    /// # Example
    ///
    /// ```rust
    /// use distributed_rl::runners::PPOConfig;
    ///
    /// let config = PPOConfig::continuous()
    ///     .with_max_env_steps(1_000_000)
    ///     .with_target_reward(-200.0);  // Pendulum target
    /// ```
    pub fn continuous() -> Self {
        Self {
            reward_normalization: true,
            obs_normalization: Some(ObsNormalizationConfig::default()),
            entropy_coef: 0.0, // Continuous policies explore via std, not entropy bonus
            ..Self::default()
        }
    }

    /// Total environments across all actors.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    /// Transitions per rollout (all envs × rollout length).
    pub fn transitions_per_rollout(&self) -> usize {
        self.total_envs() * self.rollout_length
    }

    /// Minibatch size based on config.
    ///
    /// # Panics
    /// Panics in debug builds if n_minibatches is 0.
    pub fn minibatch_size(&self) -> usize {
        debug_assert!(
            self.n_minibatches > 0,
            "n_minibatches must be > 0 to compute minibatch_size"
        );
        self.transitions_per_rollout() / self.n_minibatches
    }

    /// Validate all configuration parameters.
    ///
    /// Returns `Ok(())` if the configuration is valid, or `Err(ConfigError)`
    /// with details about what's invalid.
    ///
    /// # Validation Rules
    /// - Count parameters (n_actors, n_envs_per_actor, etc.) must be > 0
    /// - gamma and gae_lambda must be in [0.0, 1.0]
    /// - clip_ratio must be in (0.0, 1.0]
    /// - transitions_per_rollout must be >= n_minibatches
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Count validations - must be positive
        if self.n_actors == 0 {
            return Err(ConfigError::InvalidCount {
                field: "n_actors",
                value: 0,
            });
        }
        if self.n_envs_per_actor == 0 {
            return Err(ConfigError::InvalidCount {
                field: "n_envs_per_actor",
                value: 0,
            });
        }
        if self.rollout_length == 0 {
            return Err(ConfigError::InvalidCount {
                field: "rollout_length",
                value: 0,
            });
        }
        if self.n_epochs == 0 {
            return Err(ConfigError::InvalidCount {
                field: "n_epochs",
                value: 0,
            });
        }
        if self.n_minibatches == 0 {
            return Err(ConfigError::InvalidCount {
                field: "n_minibatches",
                value: 0,
            });
        }
        if self.tbptt_length == 0 {
            return Err(ConfigError::InvalidCount {
                field: "tbptt_length",
                value: 0,
            });
        }

        // Range validations
        if self.gamma < 0.0 || self.gamma > 1.0 {
            return Err(ConfigError::OutOfRange {
                field: "gamma",
                value: self.gamma,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.gae_lambda < 0.0 || self.gae_lambda > 1.0 {
            return Err(ConfigError::OutOfRange {
                field: "gae_lambda",
                value: self.gae_lambda,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.clip_ratio <= 0.0 || self.clip_ratio > 1.0 {
            return Err(ConfigError::OutOfRange {
                field: "clip_ratio",
                value: self.clip_ratio,
                min: 0.0,
                max: 1.0,
            });
        }

        // Minibatch size check - must have at least one transition per minibatch
        let transitions = self.transitions_per_rollout();
        if transitions < self.n_minibatches {
            return Err(ConfigError::InvalidMinibatch {
                transitions,
                minibatches: self.n_minibatches,
            });
        }

        Ok(())
    }

    /// Build and validate the configuration.
    ///
    /// This is the recommended way to finalize a configuration after using
    /// builder methods. Returns `Err` if validation fails.
    pub fn build(self) -> Result<Self, ConfigError> {
        self.validate()?;
        Ok(self)
    }

    // Builder methods for multi-actor settings

    /// Set number of actor threads.
    pub fn with_n_actors(mut self, n_actors: usize) -> Self {
        self.n_actors = n_actors;
        self
    }

    /// Set environments per actor.
    pub fn with_n_envs_per_actor(mut self, n_envs: usize) -> Self {
        self.n_envs_per_actor = n_envs;
        self
    }

    // Builder methods for PPO settings

    /// Set rollout length (steps per environment per rollout).
    pub fn with_rollout_length(mut self, length: usize) -> Self {
        self.rollout_length = length;
        self
    }

    /// Set number of training epochs per rollout.
    pub fn with_n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    /// Set number of minibatches.
    pub fn with_n_minibatches(mut self, n: usize) -> Self {
        self.n_minibatches = n;
        self
    }

    /// Set discount factor (gamma).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set GAE lambda.
    pub fn with_gae_lambda(mut self, lambda: f32) -> Self {
        self.gae_lambda = lambda;
        self
    }

    /// Set PPO clipping ratio.
    pub fn with_clip_ratio(mut self, ratio: f32) -> Self {
        self.clip_ratio = ratio;
        self
    }

    /// Set value function coefficient.
    pub fn with_vf_coef(mut self, coef: f32) -> Self {
        self.vf_coef = coef;
        self
    }

    /// Set entropy coefficient.
    pub fn with_entropy_coef(mut self, coef: f32) -> Self {
        self.entropy_coef = coef;
        self
    }

    /// Set whether to normalize advantages.
    pub fn with_normalize_advantages(mut self, normalize: bool) -> Self {
        self.normalize_advantages = normalize;
        self
    }

    /// Set advantage normalization strategy.
    /// Only effective when `normalize_advantages` is true.
    pub fn with_advantage_normalization(mut self, strategy: AdvantageNormalization) -> Self {
        self.advantage_normalization = strategy;
        self
    }

    /// Set whether to clip value function loss (default: true).
    /// When enabled, prevents large value function updates by clipping
    /// the value predictions relative to old values.
    pub fn with_clip_vloss(mut self, clip: bool) -> Self {
        self.clip_vloss = clip;
        self
    }

    /// Set whether to anneal learning rate linearly to 0 (default: true).
    /// The learning rate decreases linearly from initial value to 0
    /// over the course of training (based on max_env_steps).
    pub fn with_anneal_lr(mut self, anneal: bool) -> Self {
        self.anneal_lr = anneal;
        self
    }

    /// Set target KL divergence for early stopping epochs.
    /// If approximate KL exceeds this threshold during an epoch, remaining
    /// minibatches are skipped to prevent over-updating the policy.
    /// Common values: 0.01-0.02 for stability. None = disabled (default).
    pub fn with_target_kl(mut self, target_kl: Option<f64>) -> Self {
        self.target_kl = target_kl;
        self
    }

    // Builder methods for recurrent-specific settings

    /// Set TBPTT (Truncated Backpropagation Through Time) sequence length.
    /// Only used by DistributedRecurrentPPORunner.
    pub fn with_tbptt_length(mut self, length: usize) -> Self {
        self.tbptt_length = length;
        self
    }

    // Builder methods for training settings

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum gradient norm (None = no clipping).
    pub fn with_max_grad_norm(mut self, norm: Option<f32>) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set maximum environment steps.
    pub fn with_max_env_steps(mut self, steps: usize) -> Self {
        self.max_env_steps = steps;
        self
    }

    /// Set target reward for early stopping.
    pub fn with_target_reward(mut self, reward: f32) -> Self {
        self.target_reward = Some(reward);
        self
    }

    /// Set logging interval in seconds.
    pub fn with_log_interval_secs(mut self, secs: f32) -> Self {
        self.log_interval_secs = secs;
        self
    }

    /// Set model publish frequency.
    pub fn with_model_publish_freq(mut self, freq: usize) -> Self {
        self.model_publish_freq = freq;
        self
    }

    // Builder methods for separate learning rates

    /// Set actor-specific learning rate.
    /// When set, the policy network uses this learning rate instead of `learning_rate`.
    pub fn with_actor_lr(mut self, lr: f64) -> Self {
        self.actor_lr = Some(lr);
        self
    }

    /// Set critic-specific learning rate.
    /// When set, the value network uses this learning rate instead of `learning_rate`.
    pub fn with_critic_lr(mut self, lr: f64) -> Self {
        self.critic_lr = Some(lr);
        self
    }

    /// Set the critic update ratio.
    /// This determines how many times the critic is updated per actor update.
    /// Higher values strengthen the value function. Default: 1.
    pub fn with_critic_update_ratio(mut self, ratio: usize) -> Self {
        self.critic_update_ratio = ratio;
        self
    }

    // Builder methods for advanced normalization

    /// Enable observation normalization with the given clip range.
    /// Observations are normalized using running mean/std statistics.
    pub fn with_obs_normalization(mut self, clip_range: Option<(f32, f32)>) -> Self {
        self.obs_normalization = Some(ObsNormalizationConfig::new().with_clip_range(clip_range));
        self
    }

    /// Enable PopArt value normalization.
    /// Normalizes value targets while preserving output consistency.
    pub fn with_popart(mut self, beta: f32) -> Self {
        self.popart = Some(PopArtConfig::new().with_beta(beta));
        self
    }

    /// Enable reward normalization.
    /// Normalizes rewards by the standard deviation of returns.
    pub fn with_reward_normalization(mut self, enabled: bool) -> Self {
        self.reward_normalization = enabled;
        self
    }

    /// Set entropy coefficient scheduling strategy.
    pub fn with_entropy_schedule(mut self, schedule: EntropySchedule) -> Self {
        self.entropy_schedule = schedule;
        self
    }

    /// Enable target network for value function.
    /// Uses soft updates with the given tau coefficient.
    /// Typical values: 0.005 - 0.01.
    pub fn with_target_network(mut self, tau: f32) -> Self {
        self.target_network_tau = Some(tau);
        self
    }

    /// Set minimum log standard deviation for continuous policies.
    /// Prevents exploration collapse by ensuring std >= exp(floor).
    /// Default: -2.0 (exp(-2) = 0.135, 13.5% minimum std).
    pub fn with_log_std_floor(mut self, floor: f32) -> Self {
        self.log_std_floor = floor;
        self
    }

    // Convenience methods

    /// Get the effective actor learning rate.
    pub fn effective_actor_lr(&self) -> f64 {
        self.actor_lr.unwrap_or(self.learning_rate)
    }

    /// Get the effective critic learning rate.
    pub fn effective_critic_lr(&self) -> f64 {
        self.critic_lr.unwrap_or(self.learning_rate)
    }

    /// Get the current entropy coefficient based on training progress.
    /// Progress should be between 0.0 (start) and 1.0 (end).
    pub fn current_entropy_coef(&self, progress: f32) -> f32 {
        self.entropy_schedule.get_coef(self.entropy_coef, progress)
    }
}

/// Training statistics for distributed PPO.
#[derive(Debug, Clone, Default)]
pub struct PPOStats {
    /// Total environment steps across all actors
    pub env_steps: usize,
    /// Total episodes completed
    pub episodes: usize,
    /// Average episode reward (recent 100 episodes)
    pub avg_reward: f32,
    /// Current policy version
    pub policy_version: u64,
    /// Steps per second
    pub steps_per_second: f32,
    /// Policy loss (last training batch)
    pub policy_loss: f32,
    /// Value loss (last training batch)
    pub value_loss: f32,
    /// Entropy (last training batch)
    pub entropy: f32,
    /// Current buffer utilization
    pub buffer_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PPOConfig::new();
        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 32);
        assert_eq!(config.total_envs(), 128);
        assert_eq!(config.rollout_length, 64);
        assert_eq!(config.transitions_per_rollout(), 128 * 64);
    }

    #[test]
    fn test_builder_pattern() {
        let config = PPOConfig::new()
            .with_n_actors(8)
            .with_n_envs_per_actor(16)
            .with_rollout_length(32)
            .with_learning_rate(1e-3)
            .with_target_reward(500.0);

        assert_eq!(config.n_actors, 8);
        assert_eq!(config.n_envs_per_actor, 16);
        assert_eq!(config.total_envs(), 128);
        assert_eq!(config.rollout_length, 32);
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.target_reward, Some(500.0));
    }

    #[test]
    fn test_minibatch_size() {
        let config = PPOConfig::new()
            .with_n_actors(4)
            .with_n_envs_per_actor(32)
            .with_rollout_length(64)
            .with_n_minibatches(8);

        // 4 * 32 * 64 = 8192 transitions
        // 8192 / 8 = 1024 per minibatch
        assert_eq!(config.minibatch_size(), 1024);
    }

    // Validation tests

    #[test]
    fn test_default_config_is_valid() {
        let config = PPOConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_build_validates() {
        let result = PPOConfig::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_n_actors_zero() {
        let config = PPOConfig::new().with_n_actors(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_actors", .. })
        ));
    }

    #[test]
    fn test_validation_n_envs_per_actor_zero() {
        let config = PPOConfig::new().with_n_envs_per_actor(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_envs_per_actor", .. })
        ));
    }

    #[test]
    fn test_validation_rollout_length_zero() {
        let config = PPOConfig::new().with_rollout_length(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "rollout_length", .. })
        ));
    }

    #[test]
    fn test_validation_n_minibatches_zero() {
        let config = PPOConfig::new().with_n_minibatches(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_minibatches", .. })
        ));
    }

    #[test]
    fn test_validation_tbptt_length_zero() {
        let config = PPOConfig::new().with_tbptt_length(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "tbptt_length", .. })
        ));
    }

    #[test]
    fn test_tbptt_length_configurable() {
        let config = PPOConfig::new().with_tbptt_length(16);
        assert_eq!(config.tbptt_length, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_advantage_normalization_default() {
        let config = PPOConfig::new();
        assert_eq!(config.advantage_normalization, AdvantageNormalization::Global);
    }

    #[test]
    fn test_advantage_normalization_global() {
        let config = PPOConfig::new()
            .with_advantage_normalization(AdvantageNormalization::Global);
        assert_eq!(config.advantage_normalization, AdvantageNormalization::Global);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_advantage_normalization_none() {
        let config = PPOConfig::new()
            .with_advantage_normalization(AdvantageNormalization::None);
        assert_eq!(config.advantage_normalization, AdvantageNormalization::None);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_gamma_negative() {
        let config = PPOConfig::new().with_gamma(-0.1);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gamma", .. })
        ));
    }

    #[test]
    fn test_validation_gamma_over_one() {
        let config = PPOConfig::new().with_gamma(1.5);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gamma", .. })
        ));
    }

    #[test]
    fn test_validation_gae_lambda_out_of_range() {
        let config = PPOConfig::new().with_gae_lambda(2.0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gae_lambda", .. })
        ));
    }

    #[test]
    fn test_validation_clip_ratio_zero() {
        let config = PPOConfig::new().with_clip_ratio(0.0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "clip_ratio", .. })
        ));
    }

    #[test]
    fn test_validation_minibatch_exceeds_transitions() {
        // 1 actor * 1 env * 2 steps = 2 transitions, but 10 minibatches
        let config = PPOConfig::new()
            .with_n_actors(1)
            .with_n_envs_per_actor(1)
            .with_rollout_length(2)
            .with_n_minibatches(10);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMinibatch { .. })
        ));
    }

    #[test]
    fn test_validation_edge_values_valid() {
        // Edge case: gamma=0 (myopic), lambda=0 (one-step TD)
        let config = PPOConfig::new()
            .with_gamma(0.0)
            .with_gae_lambda(0.0);
        assert!(config.validate().is_ok());

        // Edge case: gamma=1 (undiscounted), lambda=1 (Monte Carlo)
        let config = PPOConfig::new()
            .with_gamma(1.0)
            .with_gae_lambda(1.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::InvalidCount { field: "n_actors", value: 0 };
        assert_eq!(err.to_string(), "n_actors must be > 0, got 0");

        let err = ConfigError::OutOfRange { field: "gamma", value: 1.5, min: 0.0, max: 1.0 };
        assert_eq!(err.to_string(), "gamma must be in [0, 1], got 1.5");

        let err = ConfigError::InvalidMinibatch { transitions: 2, minibatches: 10 };
        assert_eq!(err.to_string(), "transitions_per_rollout (2) must be >= n_minibatches (10)");
    }

    // Tests for new config options

    #[test]
    fn test_separate_learning_rates() {
        let config = PPOConfig::new()
            .with_learning_rate(1e-3)
            .with_actor_lr(1e-4)
            .with_critic_lr(5e-4);

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.actor_lr, Some(1e-4));
        assert_eq!(config.critic_lr, Some(5e-4));
        assert_eq!(config.effective_actor_lr(), 1e-4);
        assert_eq!(config.effective_critic_lr(), 5e-4);
    }

    #[test]
    fn test_effective_lr_fallback() {
        let config = PPOConfig::new()
            .with_learning_rate(1e-3);

        // When actor_lr/critic_lr not set, fall back to learning_rate
        assert_eq!(config.effective_actor_lr(), 1e-3);
        assert_eq!(config.effective_critic_lr(), 1e-3);
    }

    #[test]
    fn test_critic_update_ratio() {
        let config = PPOConfig::new()
            .with_critic_update_ratio(4);

        assert_eq!(config.critic_update_ratio, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_entropy_schedule_constant() {
        let config = PPOConfig::new()
            .with_entropy_coef(0.01);

        // Constant schedule should return same value at all progress levels
        assert!((config.current_entropy_coef(0.0) - 0.01).abs() < 1e-6);
        assert!((config.current_entropy_coef(0.5) - 0.01).abs() < 1e-6);
        assert!((config.current_entropy_coef(1.0) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_schedule_linear() {
        let config = PPOConfig::new()
            .with_entropy_coef(0.01)
            .with_entropy_schedule(EntropySchedule::Linear { floor: 0.001 });

        // Linear decay from 0.01 to 0.001
        assert!((config.current_entropy_coef(0.0) - 0.01).abs() < 1e-6);
        assert!((config.current_entropy_coef(1.0) - 0.001).abs() < 1e-6);
        // Midpoint should be approximately 0.0055
        let midpoint = config.current_entropy_coef(0.5);
        assert!(midpoint > 0.004 && midpoint < 0.007);
    }

    #[test]
    fn test_entropy_schedule_exponential() {
        let config = PPOConfig::new()
            .with_entropy_coef(0.01)
            .with_entropy_schedule(EntropySchedule::Exponential {
                decay_rate: 0.1,
                floor: 0.001
            });

        // Should start high and decay
        let start = config.current_entropy_coef(0.0);
        let end = config.current_entropy_coef(1.0);
        assert!(start > end);
        assert!(end >= 0.001); // Should be at or above floor
    }

    #[test]
    fn test_obs_normalization_config() {
        let config = PPOConfig::new()
            .with_obs_normalization(Some((-10.0, 10.0)));

        assert!(config.obs_normalization.is_some());
        let obs_norm = config.obs_normalization.unwrap();
        assert_eq!(obs_norm.clip_range, Some((-10.0, 10.0)));
    }

    #[test]
    fn test_popart_config() {
        let config = PPOConfig::new()
            .with_popart(0.001);

        assert!(config.popart.is_some());
        let popart = config.popart.unwrap();
        assert_eq!(popart.beta, 0.001);
    }

    #[test]
    fn test_reward_normalization() {
        let config = PPOConfig::new()
            .with_reward_normalization(true);

        assert!(config.reward_normalization);
    }

    #[test]
    fn test_target_network() {
        let config = PPOConfig::new()
            .with_target_network(0.005);

        assert_eq!(config.target_network_tau, Some(0.005));
    }

    #[test]
    fn test_all_advanced_options() {
        let config = PPOConfig::new()
            .with_actor_lr(1e-4)
            .with_critic_lr(5e-4)
            .with_critic_update_ratio(2)
            .with_obs_normalization(Some((-5.0, 5.0)))
            .with_popart(0.0003)
            .with_reward_normalization(true)
            .with_entropy_schedule(EntropySchedule::Linear { floor: 0.001 })
            .with_target_network(0.01);

        assert!(config.validate().is_ok());
        assert_eq!(config.effective_actor_lr(), 1e-4);
        assert_eq!(config.effective_critic_lr(), 5e-4);
        assert_eq!(config.critic_update_ratio, 2);
        assert!(config.obs_normalization.is_some());
        assert!(config.popart.is_some());
        assert!(config.reward_normalization);
        assert_eq!(config.target_network_tau, Some(0.01));
    }
}
