//! Distributed algorithm trait for multi-actor training.
//!
//! This module provides the `DistributedAlgorithm` trait which extends the
//! base `Algorithm` trait with buffer management for distributed training.
//!
//! # Architecture
//!
//! ```text
//! DistributedAlgorithm
//!       ├── Buffer creation and management
//!       ├── Batch sampling strategy
//!       ├── Staleness handling
//!       └── Underlying loss computation (delegates to Algorithm)
//! ```

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::sync::Arc;

use super::algorithm::LossOutput;
use crate::core::experience_buffer::{ExperienceBuffer, OnPolicyBuffer, OffPolicyBuffer};

/// Trait for distributed RL algorithms.
///
/// Extends the base `Algorithm` trait with buffer management and
/// staleness handling for multi-actor training.
///
/// # Type Parameters
/// - `B`: Autodiff backend for gradient computation
///
/// # Implementations
/// - `PPO`: On-policy with GAE, consumes rollouts entirely
/// - `IMPALA`: Off-policy with V-trace correction
pub trait DistributedAlgorithm<B: AutodiffBackend>: Clone + Send + Sync + 'static {
    /// Configuration type for this algorithm.
    type Config: Clone + Send + Sync;

    /// Buffer type used by this algorithm.
    type Buffer: ExperienceBuffer + Send + Sync + 'static;

    /// Batch type returned by sampling.
    type Batch: Send;

    /// Create a new algorithm instance from configuration.
    fn new(config: Self::Config) -> Self;

    /// Get the algorithm configuration.
    fn config(&self) -> &Self::Config;

    /// Create an appropriate buffer for this algorithm.
    ///
    /// # Arguments
    /// - `n_actors`: Number of actor threads
    /// - `n_envs_per_actor`: Environments per actor
    fn create_buffer(&self, n_actors: usize, n_envs_per_actor: usize) -> Arc<Self::Buffer>;

    /// Check if the buffer is ready for training.
    ///
    /// For PPO: True when rollout is complete.
    /// For IMPALA: True when minimum samples are available.
    fn is_ready(&self, buffer: &Self::Buffer) -> bool;

    /// Sample a batch from the buffer.
    ///
    /// - PPO: Returns entire rollout and clears buffer
    /// - IMPALA: Samples trajectories FIFO
    fn sample_batch(&self, buffer: &Self::Buffer) -> Option<Self::Batch>;

    /// Handle staleness in the sampled batch.
    ///
    /// - PPO: Filter out stale data (version mismatch)
    /// - IMPALA: No filtering needed (V-trace handles staleness)
    ///
    /// Returns the batch with staleness applied.
    fn handle_staleness(&self, batch: Self::Batch, current_version: u64) -> Self::Batch;

    /// Compute loss from a batch.
    ///
    /// This is the main training step that combines advantage estimation
    /// and loss computation.
    ///
    /// # Arguments
    /// - `batch`: Sampled batch from buffer
    /// - `model`: Current actor-critic model
    /// - `device`: Burn device for tensor operations
    ///
    /// # Returns
    /// `LossOutput` with total loss tensor and component scalars.
    fn compute_batch_loss(
        &self,
        batch: &Self::Batch,
        log_probs: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
        values: Tensor<B, 2>,
        device: &B::Device,
    ) -> LossOutput<B>;

    /// Whether this algorithm is off-policy.
    ///
    /// - On-policy (PPO): Requires fresh data from current policy
    /// - Off-policy (IMPALA): Can use stale data with corrections
    fn is_off_policy(&self) -> bool;

    /// Algorithm name for logging.
    fn name(&self) -> &'static str;

    /// Number of optimization epochs per batch.
    ///
    /// PPO typically uses multiple epochs (4-10).
    /// IMPALA uses single pass.
    fn n_epochs(&self) -> usize {
        1
    }

    /// Number of minibatches to split the batch into.
    ///
    /// PPO typically uses multiple minibatches.
    /// IMPALA processes full trajectories.
    fn n_minibatches(&self) -> usize {
        1
    }
}

/// Configuration for distributed PPO.
///
/// # Validation
///
/// Use `validate()` to check configuration before training.
/// Invalid configurations can cause training instability or NaN losses.
#[derive(Debug, Clone)]
pub struct PPOAlgorithmConfig {
    /// PPO clipping ratio (ε). Must be > 0. Typical: 0.1-0.3.
    pub clip_ratio: f32,
    /// Value function loss coefficient. Must be >= 0. Typical: 0.5-1.0.
    pub vf_coef: f32,
    /// Entropy bonus coefficient. Must be >= 0. Typical: 0.0-0.1.
    pub entropy_coef: f32,
    /// Optional value clipping. If Some, must be > 0.
    pub clip_value: Option<f32>,

    /// Rollout length per actor. Must be >= 1.
    pub rollout_length: usize,
    /// Discount factor γ. Must be in [0, 1].
    pub gamma: f32,
    /// GAE λ parameter. Must be in [0, 1].
    pub gae_lambda: f32,
    /// Number of optimization epochs. Must be >= 1.
    pub n_epochs: usize,
    /// Number of minibatches. Must be >= 1.
    pub n_minibatches: usize,
    /// Normalize advantages to zero mean, unit variance.
    pub normalize_advantages: bool,
    /// Maximum policy version lag allowed before discarding data.
    pub max_staleness: u64,
}

/// Validation errors for PPO configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum PPOConfigError {
    /// clip_ratio must be > 0
    InvalidClipRatio(f32),
    /// vf_coef must be >= 0
    InvalidVfCoef(f32),
    /// entropy_coef must be >= 0
    InvalidEntropyCoef(f32),
    /// clip_value must be > 0 if Some
    InvalidClipValue(f32),
    /// gamma must be in [0, 1]
    InvalidGamma(f32),
    /// gae_lambda must be in [0, 1]
    InvalidGaeLambda(f32),
    /// n_epochs must be >= 1
    InvalidNEpochs(usize),
    /// n_minibatches must be >= 1
    InvalidNMinibatches(usize),
    /// rollout_length must be >= 1
    InvalidRolloutLength(usize),
}

impl std::fmt::Display for PPOConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidClipRatio(v) => write!(f, "clip_ratio must be > 0, got {}", v),
            Self::InvalidVfCoef(v) => write!(f, "vf_coef must be >= 0, got {}", v),
            Self::InvalidEntropyCoef(v) => write!(f, "entropy_coef must be >= 0, got {}", v),
            Self::InvalidClipValue(v) => write!(f, "clip_value must be > 0 if Some, got {}", v),
            Self::InvalidGamma(v) => write!(f, "gamma must be in [0, 1], got {}", v),
            Self::InvalidGaeLambda(v) => write!(f, "gae_lambda must be in [0, 1], got {}", v),
            Self::InvalidNEpochs(v) => write!(f, "n_epochs must be >= 1, got {}", v),
            Self::InvalidNMinibatches(v) => write!(f, "n_minibatches must be >= 1, got {}", v),
            Self::InvalidRolloutLength(v) => write!(f, "rollout_length must be >= 1, got {}", v),
        }
    }
}

impl std::error::Error for PPOConfigError {}

impl Default for PPOAlgorithmConfig {
    fn default() -> Self {
        Self {
            clip_ratio: 0.2,
            vf_coef: 0.5,
            entropy_coef: 0.01,
            clip_value: Some(0.2),
            rollout_length: 128,
            gamma: 0.99,
            gae_lambda: 0.95,
            n_epochs: 4,
            n_minibatches: 4,
            normalize_advantages: true,
            max_staleness: 1,
        }
    }
}

impl PPOAlgorithmConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration.
    ///
    /// Returns `Ok(())` if all parameters are valid, or the first error found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use distributed_rl::algorithms::core_algorithm::PPOAlgorithmConfig;
    ///
    /// let config = PPOAlgorithmConfig::new().with_gamma(0.99);
    /// config.validate().expect("Invalid config");
    /// ```
    pub fn validate(&self) -> Result<(), PPOConfigError> {
        // clip_ratio must be > 0
        if self.clip_ratio <= 0.0 || !self.clip_ratio.is_finite() {
            return Err(PPOConfigError::InvalidClipRatio(self.clip_ratio));
        }

        // vf_coef must be >= 0
        if self.vf_coef < 0.0 || !self.vf_coef.is_finite() {
            return Err(PPOConfigError::InvalidVfCoef(self.vf_coef));
        }

        // entropy_coef must be >= 0
        if self.entropy_coef < 0.0 || !self.entropy_coef.is_finite() {
            return Err(PPOConfigError::InvalidEntropyCoef(self.entropy_coef));
        }

        // clip_value must be > 0 if Some
        if let Some(cv) = self.clip_value {
            if cv <= 0.0 || !cv.is_finite() {
                return Err(PPOConfigError::InvalidClipValue(cv));
            }
        }

        // gamma must be in [0, 1]
        if self.gamma < 0.0 || self.gamma > 1.0 || !self.gamma.is_finite() {
            return Err(PPOConfigError::InvalidGamma(self.gamma));
        }

        // gae_lambda must be in [0, 1]
        if self.gae_lambda < 0.0 || self.gae_lambda > 1.0 || !self.gae_lambda.is_finite() {
            return Err(PPOConfigError::InvalidGaeLambda(self.gae_lambda));
        }

        // n_epochs must be >= 1
        if self.n_epochs == 0 {
            return Err(PPOConfigError::InvalidNEpochs(self.n_epochs));
        }

        // n_minibatches must be >= 1
        if self.n_minibatches == 0 {
            return Err(PPOConfigError::InvalidNMinibatches(self.n_minibatches));
        }

        // rollout_length must be >= 1
        if self.rollout_length == 0 {
            return Err(PPOConfigError::InvalidRolloutLength(self.rollout_length));
        }

        Ok(())
    }

    /// Validate and panic if invalid.
    ///
    /// Use this in contexts where invalid config is a programming error.
    pub fn validate_or_panic(&self) {
        if let Err(e) = self.validate() {
            panic!("Invalid PPO configuration: {}", e);
        }
    }

    /// Set rollout length.
    pub fn with_rollout_length(mut self, length: usize) -> Self {
        self.rollout_length = length;
        self
    }

    /// Set discount factor.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set GAE lambda.
    pub fn with_gae_lambda(mut self, gae_lambda: f32) -> Self {
        self.gae_lambda = gae_lambda;
        self
    }

    /// Set number of epochs.
    pub fn with_n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set number of minibatches.
    pub fn with_n_minibatches(mut self, n_minibatches: usize) -> Self {
        self.n_minibatches = n_minibatches;
        self
    }

    /// Set clip ratio.
    pub fn with_clip_ratio(mut self, clip_ratio: f32) -> Self {
        self.clip_ratio = clip_ratio;
        self
    }

    /// Set value function coefficient.
    pub fn with_vf_coef(mut self, vf_coef: f32) -> Self {
        self.vf_coef = vf_coef;
        self
    }

    /// Set entropy coefficient.
    pub fn with_entropy_coef(mut self, entropy_coef: f32) -> Self {
        self.entropy_coef = entropy_coef;
        self
    }

    /// Set value clipping.
    pub fn with_clip_value(mut self, clip_value: Option<f32>) -> Self {
        self.clip_value = clip_value;
        self
    }

    /// Set advantage normalization.
    pub fn with_normalize_advantages(mut self, normalize: bool) -> Self {
        self.normalize_advantages = normalize;
        self
    }

    /// Set maximum staleness.
    pub fn with_max_staleness(mut self, max_staleness: u64) -> Self {
        self.max_staleness = max_staleness;
        self
    }
}

// NOTE: DistributedIMPALAConfig has been consolidated into
// crate::algorithms::impala::IMPALAConfig.
// See impala/config.rs for the unified configuration.

/// Marker trait for on-policy distributed algorithms.
pub trait OnPolicy<B: AutodiffBackend>: DistributedAlgorithm<B>
where
    Self::Buffer: OnPolicyBuffer,
{
}

/// Marker trait for off-policy distributed algorithms.
pub trait OffPolicy<B: AutodiffBackend>: DistributedAlgorithm<B>
where
    Self::Buffer: OffPolicyBuffer,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_ppo_config() {
        let config = PPOAlgorithmConfig::new()
            .with_rollout_length(256)
            .with_gamma(0.995)
            .with_n_epochs(10);

        assert_eq!(config.rollout_length, 256);
        assert_eq!(config.gamma, 0.995);
        assert_eq!(config.n_epochs, 10);
    }

    // NOTE: IMPALA config tests are now in impala/config.rs
}
