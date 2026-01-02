//! SAC (Soft Actor-Critic) algorithm implementation.
//!
//! SAC is an off-policy, maximum entropy RL algorithm that learns:
//! - A policy that maximizes expected return + entropy
//! - Twin Q-networks to reduce overestimation bias
//! - Optional automatic entropy coefficient tuning
//!
//! Key differences from PPO/IMPALA:
//! - **Separate networks**: Actor and critics are separate (no shared encoder)
//! - **Twin Q-networks**: Q1 and Q2 for double Q-learning
//! - **Target networks**: Soft/hard updates for critics only
//! - **Three losses**: Actor, critic, and alpha (entropy) losses
//! - **Per-transition updates**: Samples individual transitions, not trajectories
//!
//! # Architecture
//!
//! ```text
//! SAC Training Flow:
//!
//! 1. Sample batch from replay buffer (uniform random)
//!
//! 2. CRITIC UPDATE (every step):
//!    - Compute target: y = r + γ(1-d)(min_Q_target - α*log_π)
//!    - Update Q1, Q2 to minimize MSE(Q, y)
//!
//! 3. ACTOR UPDATE (every N steps):
//!    - Maximize: E[min_Q(s,a) - α*log_π(a|s)]
//!
//! 4. ALPHA UPDATE (if auto-tuning):
//!    - Minimize: E[-α*(log_π + H_target)]
//!
//! 5. TARGET UPDATE:
//!    - Soft: θ_target ← τ*θ + (1-τ)*θ_target
//!    - Hard: θ_target ← θ (every N steps)
//! ```

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::sync::Arc;

use super::config::SACConfig;
use super::sac_buffer::{SACBuffer, SACBufferConfig};
use super::sac_transition::{SACTransition, SACTransitionTrait};
use crate::algorithms::algorithm::LossOutput;
use crate::algorithms::core_algorithm::{DistributedAlgorithm, OffPolicy};

// ============================================================================
// SAC Algorithm
// ============================================================================

/// SAC (Soft Actor-Critic) algorithm.
///
/// Off-policy algorithm that maximizes expected return plus entropy bonus.
/// Uses twin Q-networks with target networks for stable value estimation.
///
/// # Key Components
///
/// - **Replay Buffer**: Uniform random sampling (not FIFO like IMPALA)
/// - **Twin Critics**: Q1, Q2 for pessimistic value estimation
/// - **Target Networks**: Slow-moving copies for stable TD targets
/// - **Entropy Tuning**: Learnable α to maintain target entropy
///
/// # Usage
///
/// SAC training is handled by `SACRunner` which orchestrates:
/// 1. Actor threads collecting experience
/// 2. Learner thread with separate actor/critic/alpha optimizers
/// 3. Target network updates
///
/// The `DistributedAlgorithm` trait implementation provides buffer management
/// and basic configuration access. The actual loss computation for SAC is
/// more complex than the trait signature allows (separate networks, three
/// losses), so it's implemented in the runner's training strategy.
#[derive(Debug, Clone)]
pub struct SAC<Trans: SACTransitionTrait = SACTransition> {
    config: SACConfig,
    _marker: std::marker::PhantomData<Trans>,
}

impl<Trans: SACTransitionTrait> SAC<Trans> {
    /// Create a new SAC algorithm instance.
    pub fn new(config: SACConfig) -> Self {
        Self {
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create SAC with continuous action space defaults.
    pub fn continuous() -> Self {
        Self::new(SACConfig::continuous())
    }

    /// Create SAC with discrete action space defaults.
    pub fn discrete() -> Self {
        Self::new(SACConfig::discrete())
    }

    /// Get the configuration.
    pub fn config(&self) -> &SACConfig {
        &self.config
    }

    /// Get mutable reference to configuration.
    pub fn config_mut(&mut self) -> &mut SACConfig {
        &mut self.config
    }

    /// Compute target entropy based on action space.
    ///
    /// # Arguments
    /// - `action_dim`: Action dimension (continuous) or number of actions (discrete)
    /// - `is_discrete`: Whether the action space is discrete
    pub fn compute_target_entropy(&self, action_dim: usize, is_discrete: bool) -> f32 {
        self.config.compute_target_entropy(action_dim, is_discrete)
    }

    /// Whether this SAC uses hard target updates (discrete style).
    pub fn uses_hard_updates(&self) -> bool {
        self.config.hard_target_update
    }

    /// Whether automatic entropy tuning is enabled.
    pub fn uses_auto_entropy(&self) -> bool {
        self.config.auto_entropy_tuning
    }
}

// ============================================================================
// DistributedAlgorithm Implementation
// ============================================================================

impl<B, Trans> DistributedAlgorithm<B> for SAC<Trans>
where
    B: AutodiffBackend,
    Trans: SACTransitionTrait,
{
    type Config = SACConfig;
    type Buffer = SACBuffer<Trans>;
    type Batch = Vec<Trans>;

    fn new(config: Self::Config) -> Self {
        SAC::new(config)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn create_buffer(&self, _n_actors: usize, _n_envs_per_actor: usize) -> Arc<Self::Buffer> {
        let config = SACBufferConfig {
            capacity: self.config.buffer_capacity,
            min_size: self.config.min_buffer_size,
            batch_size: self.config.batch_size,
        };
        Arc::new(SACBuffer::new(config))
    }

    fn is_ready(&self, buffer: &Self::Buffer) -> bool {
        buffer.is_training_ready()
    }

    fn sample_batch(&self, buffer: &Self::Buffer) -> Option<Self::Batch> {
        buffer.sample_batch()
    }

    fn handle_staleness(&self, batch: Self::Batch, _current_version: u64) -> Self::Batch {
        // SAC doesn't need staleness handling because:
        // 1. It's off-policy - can use old data without correction
        // 2. Uses uniform random sampling - all data has equal weight
        // 3. No importance sampling correction needed (unlike IMPALA)
        //
        // High sample diversity from uniform sampling naturally handles
        // the distribution shift that would require correction in on-policy
        // or FIFO-sampled algorithms.
        batch
    }

    fn compute_batch_loss(
        &self,
        _batch: &Self::Batch,
        _log_probs: Tensor<B, 1>,
        _entropy: Tensor<B, 1>,
        _values: Tensor<B, 2>,
        device: &B::Device,
    ) -> LossOutput<B> {
        // NOTE: SAC's actual loss computation doesn't fit the DistributedAlgorithm
        // signature because it requires:
        // 1. Separate actor and critic forward passes
        // 2. Target network evaluations
        // 3. Three different losses with different update frequencies
        //
        // The real SAC training logic is in SACTrainingStrategy implementations:
        // - FeedForwardSACStrategy
        // - RecurrentSACStrategy
        //
        // This method returns a placeholder to satisfy the trait.
        // The actual losses are computed in the runner's training loop.
        let zero_loss = Tensor::<B, 1>::zeros([1], device);
        LossOutput::new(zero_loss, 0.0, 0.0, 0.0)
            .with_extra("note", 1.0) // Marker that this is a placeholder
    }

    fn is_off_policy(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "SAC"
    }

    fn n_epochs(&self) -> usize {
        // SAC does single gradient step per sample (no epochs)
        1
    }

    fn n_minibatches(&self) -> usize {
        // Process batch at once
        1
    }
}

// Mark SAC as off-policy algorithm
impl<B, Trans> OffPolicy<B> for SAC<Trans>
where
    B: AutodiffBackend,
    Trans: SACTransitionTrait,
{
}

// ============================================================================
// SAC Loss Output
// ============================================================================

/// Extended loss output for SAC with all three loss components.
///
/// Unlike `LossOutput` which has a single total loss, SAC has three
/// separate losses that are optimized with different frequencies.
#[derive(Debug, Clone)]
pub struct SACLossOutput<B: AutodiffBackend> {
    /// Critic loss (MSE of Q1 + Q2).
    pub critic_loss: Tensor<B, 1>,
    /// Actor loss (-E[min_Q - α*log_π]).
    pub actor_loss: Option<Tensor<B, 1>>,
    /// Alpha loss (-E[α*(log_π + H_target)]).
    pub alpha_loss: Option<Tensor<B, 1>>,

    // Scalar values for logging
    /// Critic loss value.
    pub critic_loss_val: f32,
    /// Actor loss value (0.0 if not computed this step).
    pub actor_loss_val: f32,
    /// Alpha loss value (0.0 if not computed or not auto-tuning).
    pub alpha_loss_val: f32,
    /// Current alpha value.
    pub alpha: f32,
    /// Mean Q-value (for monitoring).
    pub mean_q: f32,
    /// Mean entropy (for monitoring).
    pub mean_entropy: f32,
}

impl<B: AutodiffBackend> SACLossOutput<B> {
    /// Create critic-only output (for steps without actor update).
    pub fn critic_only(
        critic_loss: Tensor<B, 1>,
        critic_loss_val: f32,
        alpha: f32,
        mean_q: f32,
    ) -> Self {
        Self {
            critic_loss,
            actor_loss: None,
            alpha_loss: None,
            critic_loss_val,
            actor_loss_val: 0.0,
            alpha_loss_val: 0.0,
            alpha,
            mean_q,
            mean_entropy: 0.0,
        }
    }

    /// Create full output with all losses.
    pub fn full(
        critic_loss: Tensor<B, 1>,
        actor_loss: Tensor<B, 1>,
        alpha_loss: Option<Tensor<B, 1>>,
        critic_loss_val: f32,
        actor_loss_val: f32,
        alpha_loss_val: f32,
        alpha: f32,
        mean_q: f32,
        mean_entropy: f32,
    ) -> Self {
        Self {
            critic_loss,
            actor_loss: Some(actor_loss),
            alpha_loss,
            critic_loss_val,
            actor_loss_val,
            alpha_loss_val,
            alpha,
            mean_q,
            mean_entropy,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute critic loss for SAC.
///
/// MSE loss for both Q-networks:
/// L_critic = E[(Q1(s,a) - y)² + (Q2(s,a) - y)²]
///
/// where y = r + γ(1-d)(min_Q_target - α*log_π)
///
/// # Arguments
/// - `q1`: Q1 predictions [batch]
/// - `q2`: Q2 predictions [batch]
/// - `targets`: TD targets [batch]
///
/// # Returns
/// Combined critic loss (sum of MSE for both networks)
pub fn sac_critic_loss<B: AutodiffBackend>(
    q1: Tensor<B, 1>,
    q2: Tensor<B, 1>,
    targets: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let q1_loss = (q1 - targets.clone()).powf_scalar(2.0).mean();
    let q2_loss = (q2 - targets).powf_scalar(2.0).mean();
    q1_loss + q2_loss
}

/// Compute actor loss for SAC.
///
/// Actor maximizes: E[min_Q(s, a) - α*log_π(a|s)]
/// Loss (to minimize): E[α*log_π(a|s) - min_Q(s, a)]
///
/// # Arguments
/// - `min_q`: Minimum of Q1, Q2 for current actions [batch]
/// - `log_probs`: Log probabilities of current actions [batch]
/// - `alpha`: Current entropy coefficient
///
/// # Returns
/// Actor loss
pub fn sac_actor_loss<B: AutodiffBackend>(
    min_q: Tensor<B, 1>,
    log_probs: Tensor<B, 1>,
    alpha: f32,
) -> Tensor<B, 1> {
    // L_actor = E[α*log_π - Q] = -E[Q - α*log_π]
    (log_probs.mul_scalar(alpha) - min_q).mean()
}

/// Compute TD targets for SAC.
///
/// y = r + γ * (1 - done) * (min_Q_target(s', a') - α * log_π(a'|s'))
///
/// # Arguments
/// - `rewards`: Rewards [batch]
/// - `terminals`: Terminal flags (1.0 for terminal) [batch]
/// - `min_q_next`: min(Q1_target, Q2_target) for next state actions [batch]
/// - `next_log_probs`: Log probs of next actions [batch]
/// - `gamma`: Discount factor
/// - `alpha`: Entropy coefficient
///
/// # Returns
/// TD targets [batch] (detached, no gradients)
pub fn sac_td_targets<B: AutodiffBackend>(
    rewards: Tensor<B, 1>,
    terminals: Tensor<B, 1>,
    min_q_next: Tensor<B, 1>,
    next_log_probs: Tensor<B, 1>,
    gamma: f32,
    alpha: f32,
) -> Tensor<B, 1> {
    // V(s') = min_Q(s', a') - α * log_π(a'|s')
    let v_next = min_q_next - next_log_probs.mul_scalar(alpha);

    // y = r + γ * (1 - done) * V(s')
    let not_done = terminals.mul_scalar(-1.0).add_scalar(1.0);
    let discounted_v = not_done.mul_scalar(gamma) * v_next;

    rewards + discounted_v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_sac_new() {
        let sac = SAC::<SACTransition>::continuous();
        assert_eq!(sac.config().tau, 0.005);
        assert!(!sac.uses_hard_updates());
        assert!(sac.uses_auto_entropy());
    }

    #[test]
    fn test_sac_discrete() {
        let sac = SAC::<SACTransition>::discrete();
        assert_eq!(sac.config().tau, 1.0);
        assert!(sac.uses_hard_updates());
    }

    #[test]
    fn test_distributed_algorithm_impl() {
        let sac = SAC::<SACTransition>::continuous();

        assert_eq!(<SAC<SACTransition> as DistributedAlgorithm<B>>::name(&sac), "SAC");
        assert!(<SAC<SACTransition> as DistributedAlgorithm<B>>::is_off_policy(&sac));
        assert_eq!(<SAC<SACTransition> as DistributedAlgorithm<B>>::n_epochs(&sac), 1);
    }

    #[test]
    fn test_create_buffer() {
        let sac = SAC::<SACTransition>::continuous();
        let buffer = <SAC<SACTransition> as DistributedAlgorithm<B>>::create_buffer(&sac, 4, 1);

        // SACBufferConfig only has capacity, min_size, and batch_size
        assert_eq!(buffer.config().capacity, sac.config().buffer_capacity);
        assert_eq!(buffer.config().batch_size, sac.config().batch_size);
    }

    #[test]
    fn test_target_entropy() {
        let sac = SAC::<SACTransition>::continuous();
        let target = sac.compute_target_entropy(4, false);
        assert_eq!(target, -4.0);

        let sac = SAC::<SACTransition>::discrete();
        let target = sac.compute_target_entropy(4, true);
        // 0.89 * ln(4) ≈ 1.234
        assert!((target - 1.234).abs() < 0.01);
    }

    #[test]
    fn test_sac_critic_loss() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let q1: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let q2: Tensor<B, 1> = Tensor::from_floats([1.1, 2.1, 3.1], &device);
        let targets: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

        let loss = sac_critic_loss(q1, q2, targets);
        let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

        // Q1 is exact match (loss = 0)
        // Q2 is off by 0.1 each: MSE = (0.1² + 0.1² + 0.1²) / 3 = 0.01
        // Total = 0.01
        assert!(loss_val > 0.0);
        assert!(loss_val < 0.05);
    }

    #[test]
    fn test_sac_actor_loss() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let min_q: Tensor<B, 1> = Tensor::from_floats([10.0, 10.0], &device);
        let log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);
        let alpha = 0.2;

        let loss = sac_actor_loss(min_q, log_probs, alpha);
        let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

        // L = mean(α*log_π - Q) = mean(0.2*(-1) - 10) = mean(-10.2) = -10.2
        assert!((loss_val - (-10.2)).abs() < 0.01);
    }

    #[test]
    fn test_sac_td_targets() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let rewards: Tensor<B, 1> = Tensor::from_floats([1.0, 1.0], &device);
        let terminals: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0], &device);
        let min_q_next: Tensor<B, 1> = Tensor::from_floats([10.0, 10.0], &device);
        let next_log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);

        let targets = sac_td_targets(rewards, terminals, min_q_next, next_log_probs, 0.99, 0.2);
        let data = targets.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        // Non-terminal: y = 1 + 0.99 * (10 - 0.2*(-1)) = 1 + 0.99 * 10.2 = 11.098
        assert!((slice[0] - 11.098).abs() < 0.01);

        // Terminal: y = 1 (no future reward)
        assert!((slice[1] - 1.0).abs() < 0.01);
    }
}
