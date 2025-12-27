//! Algorithm trait for policy gradient loss computation.
//!
//! This module provides the abstraction for different RL algorithms:
//! - PPO (Proximal Policy Optimization) with clipped surrogate objective
//! - A2C (Advantage Actor-Critic) with vanilla policy gradient (future)
//! - TRPO (Trust Region Policy Optimization) (future)
//!
//! The Algorithm trait decouples loss computation from the training loop,
//! allowing the same Learner to work with different algorithms.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::collections::HashMap;

use super::policy_loss::{ppo_clip_loss, value_loss};

// ============================================================================
// Loss Output
// ============================================================================

/// Output from algorithm loss computation.
///
/// Contains the total loss tensor for backpropagation plus scalar
/// values of individual components for logging.
#[derive(Debug, Clone)]
pub struct LossOutput<B: AutodiffBackend> {
    /// Total loss tensor for backpropagation.
    pub total_loss: Tensor<B, 1>,
    /// Policy loss scalar (for logging).
    pub policy_loss: f32,
    /// Value loss scalar (for logging).
    pub value_loss: f32,
    /// Entropy scalar (for logging).
    pub entropy: f32,
    /// Algorithm-specific extra metrics.
    pub extra: HashMap<String, f32>,
}

impl<B: AutodiffBackend> LossOutput<B> {
    /// Create a new loss output.
    pub fn new(
        total_loss: Tensor<B, 1>,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
    ) -> Self {
        Self {
            total_loss,
            policy_loss,
            value_loss,
            entropy,
            extra: HashMap::new(),
        }
    }

    /// Add an extra metric.
    pub fn with_extra(mut self, key: impl Into<String>, value: f32) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

// ============================================================================
// Algorithm Trait
// ============================================================================

/// Trait for policy gradient algorithms.
///
/// Encapsulates the loss computation strategy, allowing the same
/// training loop (Learner) to work with different algorithms like
/// PPO, A2C, or TRPO.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend for gradient computation
///
/// # Design
///
/// The algorithm receives pre-computed tensors from the training loop:
/// - `log_probs`: New policy log probabilities
/// - `entropy`: New policy entropy
/// - `values`: New value estimates
/// - Plus rollout data (old_log_probs, old_values, advantages, returns)
///
/// This design keeps the ActionPolicy abstraction in the Learner,
/// while the Algorithm only deals with tensors.
pub trait Algorithm<B: AutodiffBackend>: Clone + Send + Sync + 'static {
    /// Compute the combined loss for a minibatch.
    ///
    /// # Arguments
    ///
    /// - `log_probs`: Log probabilities from current policy [batch]
    /// - `entropy`: Entropy from current policy [batch]
    /// - `values`: Value estimates from current model [batch, 1]
    /// - `old_log_probs`: Log probabilities from old policy (detached) [batch]
    /// - `old_values`: Value estimates from old model (detached) [batch]
    /// - `advantages`: GAE advantages (detached, normalized) [batch]
    /// - `returns`: Value targets (detached) [batch]
    ///
    /// # Returns
    ///
    /// `LossOutput` with total loss tensor and component scalars for logging.
    fn compute_loss(
        &self,
        log_probs: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
        values: Tensor<B, 2>,
        old_log_probs: Tensor<B, 1>,
        old_values: Tensor<B, 1>,
        advantages: Tensor<B, 1>,
        returns: Tensor<B, 1>,
    ) -> LossOutput<B>;

    /// Algorithm name for logging.
    fn name(&self) -> &'static str;

    /// Value function loss coefficient.
    fn vf_coef(&self) -> f32;

    /// Entropy coefficient (for exploration bonus).
    fn entropy_coef(&self) -> f32;
}

// ============================================================================
// PPO Algorithm
// ============================================================================

/// PPO algorithm configuration.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Clipping ratio for policy loss (default: 0.2).
    pub clip_ratio: f32,
    /// Value function loss coefficient (default: 0.5).
    pub vf_coef: f32,
    /// Entropy coefficient (default: 0.01).
    pub entropy_coef: f32,
    /// Optional value clipping ratio (default: Some(clip_ratio)).
    pub clip_value: Option<f32>,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_ratio: 0.2,
            vf_coef: 0.5,
            entropy_coef: 0.01,
            clip_value: Some(0.2),
        }
    }
}

impl PPOConfig {
    /// Create a new PPO configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the clipping ratio.
    pub fn with_clip_ratio(mut self, clip_ratio: f32) -> Self {
        self.clip_ratio = clip_ratio;
        self
    }

    /// Set the value function coefficient.
    pub fn with_vf_coef(mut self, vf_coef: f32) -> Self {
        self.vf_coef = vf_coef;
        self
    }

    /// Set the entropy coefficient.
    pub fn with_entropy_coef(mut self, entropy_coef: f32) -> Self {
        self.entropy_coef = entropy_coef;
        self
    }

    /// Set value clipping (None to disable).
    pub fn with_clip_value(mut self, clip_value: Option<f32>) -> Self {
        self.clip_value = clip_value;
        self
    }
}

/// PPO (Proximal Policy Optimization) algorithm.
///
/// Implements the clipped surrogate objective from:
/// "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
///
/// The objective clips the probability ratio to prevent
/// destructively large policy updates.
#[derive(Debug, Clone)]
pub struct PPOAlgorithm {
    config: PPOConfig,
}

impl PPOAlgorithm {
    /// Create a new PPO algorithm with default configuration.
    pub fn new() -> Self {
        Self {
            config: PPOConfig::default(),
        }
    }

    /// Create a PPO algorithm with custom configuration.
    pub fn with_config(config: PPOConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }
}

impl Default for PPOAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: AutodiffBackend> Algorithm<B> for PPOAlgorithm {
    fn compute_loss(
        &self,
        log_probs: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
        values: Tensor<B, 2>,
        old_log_probs: Tensor<B, 1>,
        old_values: Tensor<B, 1>,
        advantages: Tensor<B, 1>,
        returns: Tensor<B, 1>,
    ) -> LossOutput<B> {
        // Flatten values to 1D for loss computation
        let values_flat = values.flatten(0, 1);

        // PPO clipped surrogate loss
        let policy_loss = ppo_clip_loss(
            log_probs,
            old_log_probs,
            advantages,
            self.config.clip_ratio,
        );

        // Value loss (optionally clipped)
        let vf_loss = value_loss(values_flat, old_values, returns, self.config.clip_value);

        // Mean entropy
        let mean_entropy = entropy.mean();

        // Combined loss: policy + vf_coef * value - entropy_coef * entropy
        // Note: entropy_loss is negated mean entropy, so we add it with coef
        let total_loss = policy_loss.clone()
            + vf_loss.clone().mul_scalar(self.config.vf_coef)
            - mean_entropy.clone().mul_scalar(self.config.entropy_coef);

        // Extract scalar values for logging
        let policy_loss_val = Self::tensor_to_scalar(&policy_loss);
        let value_loss_val = Self::tensor_to_scalar(&vf_loss);
        let entropy_val = Self::tensor_to_scalar(&mean_entropy);

        LossOutput::new(total_loss, policy_loss_val, value_loss_val, entropy_val)
    }

    fn name(&self) -> &'static str {
        "PPO"
    }

    fn vf_coef(&self) -> f32 {
        self.config.vf_coef
    }

    fn entropy_coef(&self) -> f32 {
        self.config.entropy_coef
    }
}

impl PPOAlgorithm {
    /// Helper to extract scalar from 1D tensor.
    fn tensor_to_scalar<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
        let data = tensor.clone().into_data();
        data.as_slice::<f32>().unwrap()[0]
    }
}

// ============================================================================
// A2C Algorithm (for future extensibility)
// ============================================================================

/// A2C algorithm configuration.
#[derive(Debug, Clone)]
pub struct A2CConfig {
    /// Value function loss coefficient.
    pub vf_coef: f32,
    /// Entropy coefficient.
    pub entropy_coef: f32,
}

impl Default for A2CConfig {
    fn default() -> Self {
        Self {
            vf_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}

/// A2C (Advantage Actor-Critic) algorithm.
///
/// Simple policy gradient without clipping:
/// L = -E[log π(a|s) * A] + vf_coef * value_loss - entropy_coef * entropy
#[derive(Debug, Clone)]
pub struct A2CAlgorithm {
    config: A2CConfig,
}

impl A2CAlgorithm {
    /// Create a new A2C algorithm with default configuration.
    pub fn new() -> Self {
        Self {
            config: A2CConfig::default(),
        }
    }

    /// Create an A2C algorithm with custom configuration.
    pub fn with_config(config: A2CConfig) -> Self {
        Self { config }
    }
}

impl Default for A2CAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: AutodiffBackend> Algorithm<B> for A2CAlgorithm {
    fn compute_loss(
        &self,
        log_probs: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
        values: Tensor<B, 2>,
        _old_log_probs: Tensor<B, 1>,
        _old_values: Tensor<B, 1>,
        advantages: Tensor<B, 1>,
        returns: Tensor<B, 1>,
    ) -> LossOutput<B> {
        let values_flat = values.flatten(0, 1);

        // Vanilla policy gradient: -E[log π(a|s) * A]
        let policy_loss = -(log_probs * advantages).mean();

        // MSE value loss (no clipping for A2C)
        let vf_loss = (values_flat - returns).powf_scalar(2.0).mean();

        // Mean entropy
        let mean_entropy = entropy.mean();

        // Combined loss
        let total_loss = policy_loss.clone()
            + vf_loss.clone().mul_scalar(self.config.vf_coef)
            - mean_entropy.clone().mul_scalar(self.config.entropy_coef);

        // Extract scalars for logging
        let policy_loss_val = Self::tensor_to_scalar(&policy_loss);
        let value_loss_val = Self::tensor_to_scalar(&vf_loss);
        let entropy_val = Self::tensor_to_scalar(&mean_entropy);

        LossOutput::new(total_loss, policy_loss_val, value_loss_val, entropy_val)
    }

    fn name(&self) -> &'static str {
        "A2C"
    }

    fn vf_coef(&self) -> f32 {
        self.config.vf_coef
    }

    fn entropy_coef(&self) -> f32 {
        self.config.entropy_coef
    }
}

impl A2CAlgorithm {
    fn tensor_to_scalar<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
        let data = tensor.clone().into_data();
        data.as_slice::<f32>().unwrap()[0]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::tensor::backend::Backend;

    type B = Autodiff<NdArray<f32>>;

    fn create_test_tensors(
        batch_size: usize,
        device: &<B as Backend>::Device,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 2>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        let log_probs = Tensor::<B, 1>::zeros([batch_size], device);
        let entropy = Tensor::<B, 1>::full([batch_size], 0.5, device);
        let values = Tensor::<B, 2>::zeros([batch_size, 1], device);
        let old_log_probs = Tensor::<B, 1>::zeros([batch_size], device);
        let old_values = Tensor::<B, 1>::zeros([batch_size], device);
        let advantages = Tensor::<B, 1>::full([batch_size], 1.0, device);
        let returns = Tensor::<B, 1>::full([batch_size], 1.0, device);

        (
            log_probs,
            entropy,
            values,
            old_log_probs,
            old_values,
            advantages,
            returns,
        )
    }

    #[test]
    fn test_ppo_algorithm() {
        let device = <B as Backend>::Device::default();
        let alg = PPOAlgorithm::new();

        let (log_probs, entropy, values, old_log_probs, old_values, advantages, returns) =
            create_test_tensors(16, &device);

        let output: LossOutput<B> =
            alg.compute_loss(log_probs, entropy, values, old_log_probs, old_values, advantages, returns);

        // Check output structure
        assert_eq!(output.total_loss.dims(), [1]);
        assert_eq!(<PPOAlgorithm as Algorithm<B>>::name(&alg), "PPO");
        assert_eq!(<PPOAlgorithm as Algorithm<B>>::vf_coef(&alg), 0.5);
        assert_eq!(<PPOAlgorithm as Algorithm<B>>::entropy_coef(&alg), 0.01);
    }

    #[test]
    fn test_a2c_algorithm() {
        let device = <B as Backend>::Device::default();
        let alg = A2CAlgorithm::new();

        let (log_probs, entropy, values, old_log_probs, old_values, advantages, returns) =
            create_test_tensors(16, &device);

        let output: LossOutput<B> =
            alg.compute_loss(log_probs, entropy, values, old_log_probs, old_values, advantages, returns);

        // Check output structure
        assert_eq!(output.total_loss.dims(), [1]);
        assert_eq!(<A2CAlgorithm as Algorithm<B>>::name(&alg), "A2C");
    }

    #[test]
    fn test_ppo_config_builder() {
        let config = PPOConfig::new()
            .with_clip_ratio(0.3)
            .with_vf_coef(0.25)
            .with_entropy_coef(0.02)
            .with_clip_value(Some(0.4));

        assert_eq!(config.clip_ratio, 0.3);
        assert_eq!(config.vf_coef, 0.25);
        assert_eq!(config.entropy_coef, 0.02);
        assert_eq!(config.clip_value, Some(0.4));
    }

    #[test]
    fn test_loss_output_with_extra() {
        let device = <B as Backend>::Device::default();
        let total_loss = Tensor::<B, 1>::zeros([1], &device);

        let output = LossOutput::new(total_loss, 0.1, 0.2, 0.3)
            .with_extra("clip_fraction", 0.15)
            .with_extra("approx_kl", 0.02);

        assert_eq!(output.extra.get("clip_fraction"), Some(&0.15));
        assert_eq!(output.extra.get("approx_kl"), Some(&0.02));
    }
}
