//! Configuration for Distributed PPO training.
//!
//! This module provides configuration types for multi-actor PPO training
//! with WGPU backend and CubeCL streams.

use std::fmt;

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

/// Configuration for Distributed PPO training.
///
/// This configuration supports N actor threads + 1 learner thread
/// architecture with per-environment rollout collection and GAE.
#[derive(Debug, Clone)]
pub struct DistributedPPOConfig {
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

    // Training settings
    /// Learning rate
    pub learning_rate: f64,
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
}

impl Default for DistributedPPOConfig {
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

            // Training defaults
            learning_rate: 3e-4,
            max_grad_norm: Some(0.5),
            max_env_steps: 1_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_publish_freq: 1,
        }
    }
}

impl DistributedPPOConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
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
}

/// Training statistics for distributed PPO.
#[derive(Debug, Clone, Default)]
pub struct DistributedPPOStats {
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
        let config = DistributedPPOConfig::new();
        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 32);
        assert_eq!(config.total_envs(), 128);
        assert_eq!(config.rollout_length, 64);
        assert_eq!(config.transitions_per_rollout(), 128 * 64);
    }

    #[test]
    fn test_builder_pattern() {
        let config = DistributedPPOConfig::new()
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
        let config = DistributedPPOConfig::new()
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
        let config = DistributedPPOConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_build_validates() {
        let result = DistributedPPOConfig::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_n_actors_zero() {
        let config = DistributedPPOConfig::new().with_n_actors(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_actors", .. })
        ));
    }

    #[test]
    fn test_validation_n_envs_per_actor_zero() {
        let config = DistributedPPOConfig::new().with_n_envs_per_actor(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_envs_per_actor", .. })
        ));
    }

    #[test]
    fn test_validation_rollout_length_zero() {
        let config = DistributedPPOConfig::new().with_rollout_length(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "rollout_length", .. })
        ));
    }

    #[test]
    fn test_validation_n_minibatches_zero() {
        let config = DistributedPPOConfig::new().with_n_minibatches(0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidCount { field: "n_minibatches", .. })
        ));
    }

    #[test]
    fn test_validation_gamma_negative() {
        let config = DistributedPPOConfig::new().with_gamma(-0.1);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gamma", .. })
        ));
    }

    #[test]
    fn test_validation_gamma_over_one() {
        let config = DistributedPPOConfig::new().with_gamma(1.5);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gamma", .. })
        ));
    }

    #[test]
    fn test_validation_gae_lambda_out_of_range() {
        let config = DistributedPPOConfig::new().with_gae_lambda(2.0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "gae_lambda", .. })
        ));
    }

    #[test]
    fn test_validation_clip_ratio_zero() {
        let config = DistributedPPOConfig::new().with_clip_ratio(0.0);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange { field: "clip_ratio", .. })
        ));
    }

    #[test]
    fn test_validation_minibatch_exceeds_transitions() {
        // 1 actor * 1 env * 2 steps = 2 transitions, but 10 minibatches
        let config = DistributedPPOConfig::new()
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
        let config = DistributedPPOConfig::new()
            .with_gamma(0.0)
            .with_gae_lambda(0.0);
        assert!(config.validate().is_ok());

        // Edge case: gamma=1 (undiscounted), lambda=1 (Monte Carlo)
        let config = DistributedPPOConfig::new()
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
}
