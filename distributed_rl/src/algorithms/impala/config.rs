//! Unified IMPALA configuration.
//!
//! This module provides a single, comprehensive configuration type for
//! distributed IMPALA training, consolidating all algorithm and runner settings.

/// Unified configuration for distributed IMPALA training.
///
/// This configuration combines all settings needed for IMPALA training:
/// - Multi-actor settings (parallelism)
/// - Algorithm settings (V-trace, coefficients)
/// - Training settings (optimizer, stopping conditions)
///
/// # Example
///
/// ```ignore
/// let config = IMPALAConfig::new()
///     .with_n_actors(8)
///     .with_n_envs_per_actor(32)
///     .with_trajectory_length(20)
///     .with_learning_rate(1e-4)
///     .with_target_reward(475.0);
/// ```
#[derive(Debug, Clone)]
pub struct IMPALAConfig {
    // === Multi-actor settings ===
    /// Number of actor threads
    pub n_actors: usize,
    /// Environments per actor (vectorized)
    pub n_envs_per_actor: usize,

    // === Algorithm settings ===
    /// Trajectory length for V-trace computation
    pub trajectory_length: usize,
    /// Maximum buffer capacity (trajectories)
    pub buffer_capacity: usize,
    /// Training batch size (trajectories per training step)
    pub batch_size: usize,
    /// Discount factor
    pub gamma: f32,
    /// V-trace rho clipping (importance sampling truncation)
    pub rho_clip: f32,
    /// V-trace c clipping (trace decay)
    pub c_clip: f32,
    /// Value function loss coefficient
    pub vf_coef: f32,
    /// Entropy bonus coefficient
    pub entropy_coef: f32,

    // === Training settings ===
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
    /// How often actors check for new weights (steps)
    pub model_update_freq: usize,
}

impl Default for IMPALAConfig {
    fn default() -> Self {
        Self {
            // Multi-actor defaults
            n_actors: 4,
            n_envs_per_actor: 32,

            // Algorithm defaults (tuned for small-scale async training)
            trajectory_length: 20,
            buffer_capacity: 256,
            batch_size: 32,
            gamma: 0.99,
            rho_clip: 1.5,  // Slightly higher than paper for small-actor async setting
            c_clip: 1.0,    // Standard V-trace trace cutting
            vf_coef: 0.25,  // Reduced to prevent value loss from dominating policy gradient
            entropy_coef: 0.02, // Slightly higher for better exploration

            // Training defaults
            learning_rate: 1e-4,
            max_grad_norm: Some(40.0),
            max_env_steps: 1_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_update_freq: 100,
        }
    }
}

impl IMPALAConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total environments across all actors.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    // === Builder methods for multi-actor settings ===

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

    // === Builder methods for algorithm settings ===

    /// Set trajectory length for V-trace.
    pub fn with_trajectory_length(mut self, length: usize) -> Self {
        self.trajectory_length = length;
        self
    }

    /// Set buffer capacity (max trajectories).
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set training batch size (trajectories).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set discount factor (gamma).
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set V-trace rho clipping.
    pub fn with_rho_clip(mut self, clip: f32) -> Self {
        self.rho_clip = clip;
        self
    }

    /// Set V-trace c clipping.
    pub fn with_c_clip(mut self, clip: f32) -> Self {
        self.c_clip = clip;
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

    // === Builder methods for training settings ===

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

    /// Set model update frequency for actors.
    pub fn with_model_update_freq(mut self, freq: usize) -> Self {
        self.model_update_freq = freq;
        self
    }
}

/// Training statistics for distributed IMPALA.
#[derive(Debug, Clone, Default)]
pub struct IMPALAStats {
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
    /// Current buffer utilization (trajectories)
    pub buffer_size: usize,
    /// Training steps completed
    pub train_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IMPALAConfig::new();
        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 32);
        assert_eq!(config.total_envs(), 128);
        assert_eq!(config.trajectory_length, 20);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.rho_clip, 1.5);
        assert_eq!(config.c_clip, 1.0);
        assert_eq!(config.vf_coef, 0.25);
        assert_eq!(config.entropy_coef, 0.02);
    }

    #[test]
    fn test_builder_pattern() {
        let config = IMPALAConfig::new()
            .with_n_actors(8)
            .with_n_envs_per_actor(16)
            .with_trajectory_length(40)
            .with_rho_clip(0.5)
            .with_target_reward(500.0);

        assert_eq!(config.n_actors, 8);
        assert_eq!(config.n_envs_per_actor, 16);
        assert_eq!(config.trajectory_length, 40);
        assert_eq!(config.rho_clip, 0.5);
        assert_eq!(config.target_reward, Some(500.0));
    }

    #[test]
    fn test_total_envs() {
        let config = IMPALAConfig::new()
            .with_n_actors(4)
            .with_n_envs_per_actor(8);
        assert_eq!(config.total_envs(), 32);
    }
}
