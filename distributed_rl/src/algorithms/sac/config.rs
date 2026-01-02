//! SAC configuration and statistics.
//!
//! This module provides configuration for the SAC algorithm,
//! with presets for discrete and continuous action spaces.

// ============================================================================
// SAC Configuration
// ============================================================================

/// Configuration for SAC algorithm.
///
/// SAC has different optimal settings for discrete vs continuous action spaces:
/// - **Continuous**: Soft target updates (tau=0.005), delayed actor updates
/// - **Discrete**: Hard target updates (every 8000 steps), same update frequency
///
/// Use `SACConfig::continuous()` or `SACConfig::discrete()` for presets.
#[derive(Debug, Clone)]
pub struct SACConfig {
    // ========================================================================
    // Multi-Actor Settings
    // ========================================================================
    /// Number of actor threads collecting experience.
    pub n_actors: usize,

    /// Number of environments per actor.
    pub n_envs_per_actor: usize,

    // ========================================================================
    // Replay Buffer Settings
    // ========================================================================
    /// Maximum transitions to store in replay buffer.
    pub buffer_capacity: usize,

    /// Batch size for training.
    pub batch_size: usize,

    /// Minimum buffer size before training starts.
    pub min_buffer_size: usize,

    // ========================================================================
    // SAC Algorithm Hyperparameters
    // ========================================================================
    /// Discount factor for future rewards.
    pub gamma: f32,

    /// Soft update coefficient for target networks.
    /// Higher = faster target updates. Typically 0.005 for continuous.
    /// Set to 1.0 for hard updates (discrete).
    pub tau: f32,

    /// Actor network learning rate.
    pub actor_lr: f64,

    /// Critic networks learning rate.
    pub critic_lr: f64,

    /// Entropy coefficient (alpha) learning rate.
    pub alpha_lr: f64,

    // ========================================================================
    // Entropy Settings
    // ========================================================================
    /// Enable automatic entropy tuning (learnable alpha).
    pub auto_entropy_tuning: bool,

    /// Initial entropy coefficient.
    pub initial_alpha: f32,

    /// Target entropy. If None, computed automatically:
    /// - Continuous: -dim(A)
    /// - Discrete: 0.89 * log(|A|)
    pub target_entropy: Option<f32>,

    /// Scale factor for target entropy (for discrete).
    /// Default is 0.89 from the SAC-Discrete paper.
    pub target_entropy_scale: f32,

    // ========================================================================
    // Update Frequencies
    // ========================================================================
    /// Actor update frequency (relative to critic updates).
    /// Default: 2 for continuous (TD3-style delayed), 1 for discrete.
    pub policy_update_freq: usize,

    /// Target network update frequency.
    /// For soft updates: 1 (every gradient step).
    /// For hard updates (discrete): ~8000.
    pub target_update_freq: usize,

    /// Number of gradient steps per environment step.
    pub gradient_steps_per_env_step: usize,

    /// Whether to use hard target updates (discrete) vs soft updates (continuous).
    pub hard_target_update: bool,

    // ========================================================================
    // Training Settings
    // ========================================================================
    /// Maximum gradient norm for clipping. None = no clipping.
    pub max_grad_norm: Option<f32>,

    /// Maximum environment steps.
    pub max_env_steps: usize,

    /// Target reward for early stopping. None = no early stopping.
    pub target_reward: Option<f32>,

    /// Logging interval in seconds.
    pub log_interval_secs: f32,

    /// How often actors check for updated weights (in environment steps).
    pub model_update_freq: usize,

    // ========================================================================
    // Recurrent Settings (for recurrent SAC)
    // ========================================================================
    /// Sequence length for recurrent training (TBPTT).
    pub sequence_length: usize,

    /// Hidden state size (for recurrent networks).
    pub hidden_size: usize,

    /// Whether the recurrent cell has a cell state (LSTM) or not (GRU).
    pub has_cell_state: bool,

    // ========================================================================
    // UTD Pacing Settings
    // ========================================================================
    /// Target update-to-data (UTD) ratio: gradient steps per environment step.
    /// Default is 1.0 (one gradient step per env step).
    pub utd_ratio: f32,

    /// Sleep duration in milliseconds when learner is ahead of target UTD.
    /// Prevents busy-waiting while actors collect more data.
    pub sleep_when_ahead_ms: u64,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self::continuous()
    }
}

impl SACConfig {
    /// Create configuration optimized for continuous action spaces.
    ///
    /// Based on the original SAC paper (Haarnoja et al., 2018):
    /// - Soft target updates with tau=0.005
    /// - Delayed actor updates (every 2 critic updates)
    /// - Automatic entropy tuning enabled
    pub fn continuous() -> Self {
        Self {
            // Multi-actor
            n_actors: 4,
            n_envs_per_actor: 1,

            // Buffer
            buffer_capacity: 1_000_000,
            batch_size: 256,
            min_buffer_size: 5_000,

            // SAC hyperparameters
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,

            // Entropy
            auto_entropy_tuning: true,
            initial_alpha: 0.2,
            target_entropy: None, // Auto-compute as -dim(A)
            target_entropy_scale: 1.0,

            // Update frequencies
            policy_update_freq: 2,
            target_update_freq: 1,
            gradient_steps_per_env_step: 1,
            hard_target_update: false,

            // Training
            max_grad_norm: None,
            max_env_steps: 1_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_update_freq: 100,

            // Recurrent
            sequence_length: 20,
            hidden_size: 256,
            has_cell_state: true, // LSTM by default

            // UTD pacing
            utd_ratio: 1.0,
            sleep_when_ahead_ms: 1,
        }
    }

    /// Create configuration optimized for discrete action spaces (Atari-style).
    ///
    /// Based on SAC-Discrete paper (Christodoulou, 2019):
    /// - Hard target updates every 8000 steps
    /// - Same update frequency for actor and critic
    /// - Target entropy scaled by 0.89
    pub fn discrete() -> Self {
        Self {
            // Multi-actor
            n_actors: 4,
            n_envs_per_actor: 1,

            // Buffer
            buffer_capacity: 1_000_000,
            batch_size: 64,
            min_buffer_size: 20_000,

            // SAC hyperparameters
            gamma: 0.99,
            tau: 1.0, // Hard update
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,

            // Entropy
            auto_entropy_tuning: true,
            initial_alpha: 0.2,
            target_entropy: None, // Auto-compute as 0.89 * log(|A|)
            target_entropy_scale: 0.89,

            // Update frequencies
            policy_update_freq: 1, // No delayed updates
            target_update_freq: 8000,
            gradient_steps_per_env_step: 1,
            hard_target_update: true,

            // Training (Atari typically needs more steps)
            max_grad_norm: None,
            max_env_steps: 5_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_update_freq: 100,

            // Recurrent
            sequence_length: 20,
            hidden_size: 256,
            has_cell_state: true,

            // UTD pacing
            utd_ratio: 1.0,
            sleep_when_ahead_ms: 1,
        }
    }

    /// Create configuration for discrete SAC with soft target updates.
    ///
    /// An alternative to `discrete()` that uses soft Polyak updates (tau=0.005)
    /// instead of hard updates every 8000 steps. May be more stable for some tasks.
    pub fn discrete_soft() -> Self {
        Self {
            // Multi-actor
            n_actors: 4,
            n_envs_per_actor: 1,

            // Buffer
            buffer_capacity: 1_000_000,
            batch_size: 64,
            min_buffer_size: 20_000,

            // SAC hyperparameters
            gamma: 0.99,
            tau: 0.005, // Soft update
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,

            // Entropy
            auto_entropy_tuning: true,
            initial_alpha: 0.2,
            target_entropy: None,
            target_entropy_scale: 0.89,

            // Update frequencies
            policy_update_freq: 1,
            target_update_freq: 1, // Every step (soft updates)
            gradient_steps_per_env_step: 1,
            hard_target_update: false, // Soft updates!

            // Training
            max_grad_norm: None,
            max_env_steps: 5_000_000,
            target_reward: None,
            log_interval_secs: 2.0,
            model_update_freq: 100,

            // Recurrent
            sequence_length: 20,
            hidden_size: 256,
            has_cell_state: true,

            // UTD pacing
            utd_ratio: 1.0,
            sleep_when_ahead_ms: 1,
        }
    }

    /// Compute target entropy for the given action space.
    ///
    /// # Arguments
    /// * `action_dim` - Action dimension (for continuous) or number of actions (for discrete)
    /// * `is_discrete` - Whether the action space is discrete
    pub fn compute_target_entropy(&self, action_dim: usize, is_discrete: bool) -> f32 {
        if let Some(target) = self.target_entropy {
            return target;
        }

        if is_discrete {
            // Discrete: target_entropy_scale * log(|A|)
            // This encourages exploration proportional to action space size
            self.target_entropy_scale * (action_dim as f32).ln()
        } else {
            // Continuous: -dim(A)
            // Heuristic from SAC paper: target entropy per dimension of -1
            -(action_dim as f32)
        }
    }

    /// Total number of environments.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    // ========================================================================
    // Builder Methods
    // ========================================================================

    /// Set number of actors.
    pub fn with_n_actors(mut self, n_actors: usize) -> Self {
        self.n_actors = n_actors;
        self
    }

    /// Set environments per actor.
    pub fn with_n_envs_per_actor(mut self, n_envs: usize) -> Self {
        self.n_envs_per_actor = n_envs;
        self
    }

    /// Set buffer capacity.
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set minimum buffer size before training.
    pub fn with_min_buffer_size(mut self, min_size: usize) -> Self {
        self.min_buffer_size = min_size;
        self
    }

    /// Set discount factor.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set soft update coefficient.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Set learning rates (same for all networks).
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.actor_lr = lr;
        self.critic_lr = lr;
        self.alpha_lr = lr;
        self
    }

    /// Set actor learning rate.
    pub fn with_actor_lr(mut self, lr: f64) -> Self {
        self.actor_lr = lr;
        self
    }

    /// Set critic learning rate.
    pub fn with_critic_lr(mut self, lr: f64) -> Self {
        self.critic_lr = lr;
        self
    }

    /// Set alpha learning rate.
    pub fn with_alpha_lr(mut self, lr: f64) -> Self {
        self.alpha_lr = lr;
        self
    }

    /// Set auto entropy tuning.
    pub fn with_auto_entropy_tuning(mut self, enabled: bool) -> Self {
        self.auto_entropy_tuning = enabled;
        self
    }

    /// Set initial alpha.
    pub fn with_initial_alpha(mut self, alpha: f32) -> Self {
        self.initial_alpha = alpha;
        self
    }

    /// Set target entropy explicitly.
    pub fn with_target_entropy(mut self, target: f32) -> Self {
        self.target_entropy = Some(target);
        self
    }

    /// Set policy update frequency.
    pub fn with_policy_update_freq(mut self, freq: usize) -> Self {
        self.policy_update_freq = freq;
        self
    }

    /// Set target update frequency.
    pub fn with_target_update_freq(mut self, freq: usize) -> Self {
        self.target_update_freq = freq;
        self
    }

    /// Set gradient steps per environment step.
    pub fn with_gradient_steps(mut self, steps: usize) -> Self {
        self.gradient_steps_per_env_step = steps;
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

    /// Set logging interval.
    pub fn with_log_interval(mut self, secs: f32) -> Self {
        self.log_interval_secs = secs;
        self
    }

    /// Set model update frequency for actors.
    pub fn with_model_update_freq(mut self, freq: usize) -> Self {
        self.model_update_freq = freq;
        self
    }

    /// Set sequence length for recurrent training.
    pub fn with_sequence_length(mut self, length: usize) -> Self {
        self.sequence_length = length;
        self
    }

    /// Set hidden size for recurrent networks.
    pub fn with_hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set target UTD ratio (gradient steps per environment step).
    pub fn with_utd_ratio(mut self, ratio: f32) -> Self {
        self.utd_ratio = ratio;
        self
    }

    /// Set sleep duration when learner is ahead of target UTD.
    pub fn with_sleep_when_ahead_ms(mut self, ms: u64) -> Self {
        self.sleep_when_ahead_ms = ms;
        self
    }
}

// ============================================================================
// SAC Statistics
// ============================================================================

/// Training statistics for SAC.
#[derive(Debug, Clone, Default)]
pub struct SACStats {
    /// Total environment steps.
    pub env_steps: usize,

    /// Total training steps (gradient updates).
    pub train_steps: usize,

    /// Total episodes completed.
    pub episodes: usize,

    /// Recent episode returns (for averaging).
    pub recent_returns: Vec<f32>,

    /// Mean episode return (from recent episodes).
    pub mean_return: f32,

    /// Current entropy coefficient (alpha).
    pub alpha: f32,

    /// Current actor loss.
    pub actor_loss: f32,

    /// Current critic loss.
    pub critic_loss: f32,

    /// Current alpha loss (if auto-tuning).
    pub alpha_loss: f32,

    /// Mean Q-value (for monitoring).
    pub mean_q: f32,

    /// Mean entropy (for monitoring).
    pub mean_entropy: f32,

    /// Buffer utilization (0.0 to 1.0).
    pub buffer_utilization: f32,

    /// Steps per second.
    pub sps: f32,

    /// Elapsed time in seconds.
    pub elapsed_secs: f32,
}

impl SACStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update mean return from recent returns.
    pub fn update_mean_return(&mut self) {
        if !self.recent_returns.is_empty() {
            self.mean_return =
                self.recent_returns.iter().sum::<f32>() / self.recent_returns.len() as f32;
        }
    }

    /// Add an episode return.
    pub fn add_episode_return(&mut self, return_val: f32, max_recent: usize) {
        self.recent_returns.push(return_val);
        if self.recent_returns.len() > max_recent {
            self.recent_returns.remove(0);
        }
        self.update_mean_return();
    }

    /// Format stats for logging.
    pub fn format(&self) -> String {
        format!(
            "steps={} | episodes={} | return={:.1} | alpha={:.3} | actor_loss={:.3} | critic_loss={:.3} | sps={:.0}",
            self.env_steps,
            self.episodes,
            self.mean_return,
            self.alpha,
            self.actor_loss,
            self.critic_loss,
            self.sps
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_config() {
        let config = SACConfig::continuous();
        assert_eq!(config.tau, 0.005);
        assert!(!config.hard_target_update);
        assert_eq!(config.policy_update_freq, 2);
        assert_eq!(config.target_update_freq, 1);
    }

    #[test]
    fn test_discrete_config() {
        let config = SACConfig::discrete();
        assert_eq!(config.tau, 1.0);
        assert!(config.hard_target_update);
        assert_eq!(config.policy_update_freq, 1);
        assert_eq!(config.target_update_freq, 8000);
        assert_eq!(config.target_entropy_scale, 0.89);
    }

    #[test]
    fn test_target_entropy_continuous() {
        let config = SACConfig::continuous();
        let target = config.compute_target_entropy(3, false);
        assert!((target - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn test_target_entropy_discrete() {
        let config = SACConfig::discrete();
        let target = config.compute_target_entropy(4, true);
        // 0.89 * ln(4) = 0.89 * 1.386 = 1.234
        assert!((target - 1.234).abs() < 0.1);
    }

    #[test]
    fn test_explicit_target_entropy() {
        let config = SACConfig::continuous().with_target_entropy(-5.0);
        let target = config.compute_target_entropy(3, false);
        assert_eq!(target, -5.0);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SACConfig::continuous()
            .with_n_actors(8)
            .with_batch_size(512)
            .with_gamma(0.95)
            .with_learning_rate(1e-3);

        assert_eq!(config.n_actors, 8);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.actor_lr, 1e-3);
        assert_eq!(config.critic_lr, 1e-3);
    }

    #[test]
    fn test_total_envs() {
        let config = SACConfig::continuous()
            .with_n_actors(4)
            .with_n_envs_per_actor(8);
        assert_eq!(config.total_envs(), 32);
    }

    #[test]
    fn test_stats() {
        let mut stats = SACStats::new();

        stats.add_episode_return(100.0, 10);
        stats.add_episode_return(200.0, 10);
        stats.add_episode_return(150.0, 10);

        assert_eq!(stats.recent_returns.len(), 3);
        assert!((stats.mean_return - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_max_recent() {
        let mut stats = SACStats::new();

        for i in 0..15 {
            stats.add_episode_return(i as f32, 10);
        }

        // Should only keep last 10
        assert_eq!(stats.recent_returns.len(), 10);
        assert_eq!(stats.recent_returns[0], 5.0);
    }
}
