//! PPO algorithm implementation.
//!
//! Implements the DistributedAlgorithm trait for PPO with:
//! - On-policy rollout collection
//! - GAE advantage estimation
//! - Clipped surrogate objective
//! - Freshness filtering for stale data

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::sync::Arc;

use super::ppo_batch_buffer::{PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch};
use crate::algorithms::algorithm::LossOutput;
use crate::algorithms::core_algorithm::{
    DistributedAlgorithm, PPOAlgorithmConfig, OnPolicy,
};
use crate::algorithms::gae::compute_gae;
use crate::algorithms::policy_loss::{ppo_clip_loss, value_loss};

/// PPO algorithm.
///
/// On-policy algorithm that:
/// 1. Collects rollouts from multiple actors
/// 2. Computes GAE advantages
/// 3. Performs multiple epochs of minibatch updates
/// 4. Uses clipped surrogate objective
#[derive(Debug, Clone)]
pub struct PPO {
    config: PPOAlgorithmConfig,
}

impl PPO {
    /// Create a new distributed PPO algorithm.
    pub fn new(config: PPOAlgorithmConfig) -> Self {
        Self { config }
    }

    /// Compute GAE advantages and returns for a batch.
    ///
    /// # Panics
    ///
    /// Panics if rewards, values, and dones have different lengths.
    pub fn compute_advantages<B: AutodiffBackend>(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        last_value: f32,
        device: &B::Device,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        // Defensive: handle empty input
        if rewards.is_empty() {
            return (
                Tensor::<B, 1>::from_floats(&[] as &[f32], device),
                Tensor::<B, 1>::from_floats(&[] as &[f32], device),
            );
        }

        let (advantages, returns) = compute_gae(
            rewards,
            values,
            dones,
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        );

        let returns_tensor = Tensor::<B, 1>::from_floats(returns.as_slice(), device);

        // Normalize advantages using CPU computation for numerical stability
        // This avoids issues with Burn's var() which may use Bessel's correction
        let advantages_tensor = if !self.config.normalize_advantages {
            // Normalization disabled - return raw advantages regardless of length
            Tensor::<B, 1>::from_floats(advantages.as_slice(), device)
        } else if advantages.len() >= 2 {
            // Normal case: normalize to zero mean, unit variance
            let n = advantages.len() as f32;
            let mean: f32 = advantages.iter().sum::<f32>() / n;
            // Population variance with epsilon for stability
            let var: f32 = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
            let std = (var + 1e-8).sqrt();
            let normalized: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();
            Tensor::<B, 1>::from_floats(normalized.as_slice(), device)
        } else if advantages.len() == 1 {
            // Single element with normalization enabled: can't normalize meaningfully, zero-center
            Tensor::<B, 1>::from_floats([0.0f32].as_slice(), device)
        } else {
            // Empty
            Tensor::<B, 1>::from_floats(advantages.as_slice(), device)
        };

        (advantages_tensor, returns_tensor)
    }

    /// Helper to extract scalar from tensor.
    fn tensor_to_scalar<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
        let data = tensor.clone().into_data();
        data.as_slice::<f32>().unwrap()[0]
    }
}

/// PPO batch with computed advantages.
#[derive(Debug)]
pub struct PPOProcessedBatch {
    /// Original rollout batch.
    pub rollout: PPORolloutBatch,
    /// Computed advantages (after GAE).
    pub advantages: Vec<f32>,
    /// Computed returns (value targets).
    pub returns: Vec<f32>,
}

impl<B: AutodiffBackend> DistributedAlgorithm<B> for PPO {
    type Config = PPOAlgorithmConfig;
    type Buffer = PPORolloutBuffer;
    type Batch = PPOProcessedBatch;

    fn new(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn create_buffer(&self, n_actors: usize, n_envs_per_actor: usize) -> Arc<Self::Buffer> {
        let config = PPORolloutBufferConfig {
            n_actors,
            n_envs_per_actor,
            rollout_length: self.config.rollout_length,
        };
        Arc::new(PPORolloutBuffer::new(config))
    }

    fn is_ready(&self, buffer: &Self::Buffer) -> bool {
        buffer.is_rollout_ready()
    }

    fn sample_batch(&self, buffer: &Self::Buffer) -> Option<Self::Batch> {
        if !buffer.is_rollout_ready() {
            return None;
        }

        let rollout = buffer.consume();

        // Compute GAE advantages and returns
        let rewards = rollout.rewards();
        let values = rollout.values();
        let dones = rollout.dones();

        // Use last value from the rollout (bootstrap)
        let last_value = values.last().copied().unwrap_or(0.0);

        let (advantages, returns) = compute_gae(
            &rewards,
            &values,
            &dones,
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        );

        Some(PPOProcessedBatch {
            rollout,
            advantages,
            returns,
        })
    }

    fn handle_staleness(&self, batch: Self::Batch, current_version: u64) -> Self::Batch {
        // For PPO (on-policy), stale data is dangerous and must be discarded
        let batch_version = batch.rollout.policy_version;
        let staleness = current_version.saturating_sub(batch_version);

        if staleness > self.config.max_staleness {
            // Data is too stale - return empty batch
            // The learner MUST check is_empty() and skip training on this batch
            eprintln!(
                "[PPO] Discarding stale batch: version {} vs current {}, staleness {} > max {}",
                batch_version, current_version, staleness, self.config.max_staleness
            );

            // Return empty batch - this is critical for on-policy correctness
            return PPOProcessedBatch {
                rollout: PPORolloutBatch {
                    transitions: Vec::new(),
                    policy_version: batch_version,
                    n_envs: batch.rollout.n_envs,
                    rollout_length: batch.rollout.rollout_length,
                },
                advantages: Vec::new(),
                returns: Vec::new(),
            };
        }

        batch
    }

    fn compute_batch_loss(
        &self,
        batch: &Self::Batch,
        log_probs: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
        values: Tensor<B, 2>,
        device: &B::Device,
    ) -> LossOutput<B> {
        // Defensive: handle empty batch (e.g., from staleness filtering)
        if batch.rollout.is_empty() {
            // Return zero loss - caller should check batch.is_empty() and skip
            let zero = Tensor::<B, 1>::from_floats([0.0f32].as_slice(), device);
            return LossOutput::new(zero, 0.0, 0.0, 0.0);
        }

        // Get old log probs and values from batch
        let old_log_probs = Tensor::<B, 1>::from_floats(
            batch.rollout.log_probs().as_slice(),
            device,
        );
        let old_values = Tensor::<B, 1>::from_floats(
            batch.rollout.values().as_slice(),
            device,
        );

        // Get advantages and returns with defensive normalization
        let advantages = if self.config.normalize_advantages && batch.advantages.len() >= 2 {
            // Normalize advantages - requires at least 2 samples for meaningful variance
            let adv_slice = batch.advantages.as_slice();
            let n = adv_slice.len() as f32;
            let mean: f32 = adv_slice.iter().sum::<f32>() / n;
            // Use population variance (divide by n, not n-1) with epsilon for stability
            let var: f32 = adv_slice.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
            // Clamp std to prevent division by very small numbers
            let std = (var + 1e-8).sqrt();
            let normalized: Vec<f32> = adv_slice.iter().map(|a| (a - mean) / std).collect();
            Tensor::<B, 1>::from_floats(normalized.as_slice(), device)
        } else if batch.advantages.len() == 1 {
            // Single sample: can't normalize meaningfully, use zero-centered
            // This prevents NaN from variance calculation
            Tensor::<B, 1>::from_floats([0.0f32].as_slice(), device)
        } else {
            // Empty or normalization disabled
            Tensor::<B, 1>::from_floats(batch.advantages.as_slice(), device)
        };

        let returns = Tensor::<B, 1>::from_floats(batch.returns.as_slice(), device);

        // Flatten values to 1D
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

    fn is_off_policy(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "PPO"
    }

    fn n_epochs(&self) -> usize {
        self.config.n_epochs
    }

    fn n_minibatches(&self) -> usize {
        self.config.n_minibatches
    }
}

// Mark as on-policy
impl<B: AutodiffBackend> OnPolicy<B> for PPO {}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};

    // WGPU-only backend per implementation plan
    type B = Autodiff<Wgpu>;

    #[test]
    fn test_distributed_ppo_new() {
        let config = PPOAlgorithmConfig::default();
        let ppo = PPO::new(config);

        // Use turbofish to specify the backend type
        assert_eq!(<PPO as DistributedAlgorithm<B>>::name(&ppo), "PPO");
        assert!(!<PPO as DistributedAlgorithm<B>>::is_off_policy(&ppo));
        assert_eq!(<PPO as DistributedAlgorithm<B>>::n_epochs(&ppo), 4);
        assert_eq!(<PPO as DistributedAlgorithm<B>>::n_minibatches(&ppo), 4);
    }

    #[test]
    fn test_create_buffer() {
        let config = PPOAlgorithmConfig::default();
        let ppo = PPO::new(config);

        let buffer = <PPO as DistributedAlgorithm<B>>::create_buffer(&ppo, 2, 32);
        assert_eq!(buffer.config().n_actors, 2);
        assert_eq!(buffer.config().n_envs_per_actor, 32);
    }
}
