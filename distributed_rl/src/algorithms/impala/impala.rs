//! IMPALA algorithm implementation.
//!
//! Implements the DistributedAlgorithm trait for IMPALA with:
//! - Off-policy trajectory collection
//! - V-trace importance sampling correction
//! - Automatic staleness handling via importance weighting

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::sync::Arc;

use super::config::IMPALAConfig;
use super::impala_buffer::{IMPALABuffer, IMPALABufferConfig, IMPALABatch};
use crate::algorithms::algorithm::LossOutput;
use crate::algorithms::core_algorithm::{
    DistributedAlgorithm, OffPolicy,
};
use crate::algorithms::vtrace::{compute_vtrace, VTraceResult};

/// IMPALA algorithm.
///
/// Off-policy algorithm that:
/// 1. Collects trajectories from multiple actors with behavior policy info
/// 2. Stores trajectories in FIFO buffer
/// 3. Computes V-trace importance sampling correction
/// 4. Uses corrected advantages for policy gradient
#[derive(Debug, Clone)]
pub struct IMPALA {
    config: IMPALAConfig,
}

impl IMPALA {
    /// Create a new distributed IMPALA algorithm.
    pub fn new(config: IMPALAConfig) -> Self {
        Self { config }
    }

    /// Compute V-trace for a batch, given current policy evaluations.
    ///
    /// # Arguments
    /// - `batch`: The sampled batch of trajectories
    /// - `target_log_probs`: Log probabilities under current policy
    /// - `values`: Value estimates under current policy
    /// - `bootstrap_values`: Bootstrap values for each trajectory
    ///
    /// # Returns
    /// V-trace results for each trajectory
    pub fn compute_vtrace_batch(
        &self,
        batch: &IMPALABatch,
        target_log_probs: &[f32],
        values: &[f32],
        bootstrap_values: &[f32],
    ) -> Vec<VTraceResult> {
        let mut results = Vec::with_capacity(batch.trajectories.len());
        let mut offset = 0;

        for (i, traj) in batch.trajectories.iter().enumerate() {
            let len = traj.len();
            let end = offset + len;

            // Extract behavior log probs from trajectory
            let behavior_log_probs: Vec<f32> = traj
                .iter()
                .map(|tr| tr.behavior_log_prob)
                .collect();

            // Extract rewards and dones
            let rewards: Vec<f32> = traj.iter().map(|tr| tr.base.reward).collect();
            let dones: Vec<bool> = traj.iter().map(|tr| tr.done()).collect();

            // Get corresponding slices from current policy evaluation
            let target_slice = &target_log_probs[offset..end];
            let values_slice = &values[offset..end];
            let bootstrap = bootstrap_values.get(i).copied().unwrap_or(0.0);

            let result = compute_vtrace(
                &behavior_log_probs,
                target_slice,
                &rewards,
                values_slice,
                &dones,
                bootstrap,
                self.config.gamma,
                self.config.rho_clip,
                self.config.c_clip,
            );

            results.push(result);
            offset = end;
        }

        results
    }

    /// Helper to extract scalar from tensor.
    fn tensor_to_scalar<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
        let data = tensor.clone().into_data();
        data.as_slice::<f32>().unwrap()[0]
    }
}

/// IMPALA batch with computed V-trace targets.
#[derive(Debug)]
pub struct IMPALAProcessedBatch {
    /// Original trajectory batch.
    pub batch: IMPALABatch,
    /// V-trace targets (flattened).
    pub vtrace_targets: Vec<f32>,
    /// V-trace advantages (flattened).
    pub advantages: Vec<f32>,
    /// Clipped importance weights (flattened).
    pub rhos: Vec<f32>,
}

impl<B: AutodiffBackend> DistributedAlgorithm<B> for IMPALA {
    type Config = IMPALAConfig;
    type Buffer = IMPALABuffer;
    type Batch = IMPALABatch;

    fn new(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn create_buffer(&self, n_actors: usize, n_envs_per_actor: usize) -> Arc<Self::Buffer> {
        let config = IMPALABufferConfig {
            n_actors,
            n_envs_per_actor,
            trajectory_length: self.config.trajectory_length,
            max_trajectories: self.config.buffer_capacity / self.config.trajectory_length,
            batch_size: self.config.batch_size,
        };
        Arc::new(IMPALABuffer::new(config))
    }

    fn is_ready(&self, buffer: &Self::Buffer) -> bool {
        buffer.is_training_ready()
    }

    fn sample_batch(&self, buffer: &Self::Buffer) -> Option<Self::Batch> {
        buffer.sample_batch()
    }

    fn handle_staleness(&self, batch: Self::Batch, current_version: u64) -> Self::Batch {
        // IMPALA handles staleness via V-trace importance sampling.
        // No filtering needed - V-trace automatically corrects for policy lag.

        let staleness = batch.max_staleness(current_version);
        if staleness > 10 {
            // Log but don't discard - V-trace can handle it
            eprintln!(
                "[IMPALA] High staleness detected: {} policy versions behind",
                staleness
            );
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
        // Get behavior log probs from batch
        let _behavior_log_probs = batch.behavior_log_probs();

        // Get current policy log probs (from the log_probs tensor)
        let log_probs_data = log_probs.clone().into_data();
        let target_log_probs: Vec<f32> = log_probs_data.as_slice::<f32>().unwrap().to_vec();

        // Get current values (flatten)
        let values_flat = values.clone().flatten(0, 1);
        let values_data = values_flat.clone().into_data();
        let values_vec: Vec<f32> = values_data.as_slice::<f32>().unwrap().to_vec();

        // Compute V-trace for each trajectory
        // For bootstrap values, we need the value at the end of EACH trajectory (not global last)
        let mut bootstrap_values: Vec<f32> = Vec::with_capacity(batch.trajectories.len());
        let mut offset = 0;
        for traj in &batch.trajectories {
            let len = traj.len();
            let bootstrap = if len == 0 {
                0.0
            } else if traj.transitions.last().map(|tr| tr.done()).unwrap_or(true) {
                // Terminal state: bootstrap is 0
                0.0
            } else {
                // Non-terminal: use the value estimate at the end of THIS trajectory
                values_vec.get(offset + len - 1).copied().unwrap_or(0.0)
            };
            bootstrap_values.push(bootstrap);
            offset += len;
        }

        let vtrace_results = self.compute_vtrace_batch(
            batch,
            &target_log_probs,
            &values_vec,
            &bootstrap_values,
        );

        // Flatten V-trace results
        let mut all_advantages: Vec<f32> = Vec::new();
        let mut all_targets: Vec<f32> = Vec::new();
        let mut all_rhos: Vec<f32> = Vec::new();

        for result in &vtrace_results {
            all_advantages.extend(&result.advantages);
            all_targets.extend(&result.vs);
            all_rhos.extend(&result.rhos);
        }

        // Create tensors
        let advantages = Tensor::<B, 1>::from_floats(all_advantages.as_slice(), device);
        let vtrace_targets = Tensor::<B, 1>::from_floats(all_targets.as_slice(), device);
        let rhos = Tensor::<B, 1>::from_floats(all_rhos.as_slice(), device);

        // IMPALA policy gradient loss:
        // L_policy = -rho * log_pi(a|s) * A
        // rho is the importance sampling ratio, applied here (not in advantage calculation)
        let policy_loss = -(rhos * log_probs * advantages.clone()).mean();

        // Value loss: MSE between values and V-trace targets
        let vf_loss = (values_flat.clone() - vtrace_targets).powf_scalar(2.0).mean();

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
        true
    }

    fn name(&self) -> &'static str {
        "IMPALA"
    }

    fn n_epochs(&self) -> usize {
        // IMPALA uses single pass (off-policy, no need for multiple epochs)
        1
    }

    fn n_minibatches(&self) -> usize {
        // Process full trajectory batch at once
        1
    }
}

// Mark as off-policy
impl<B: AutodiffBackend> OffPolicy<B> for IMPALA {}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};

    // WGPU-only backend per implementation plan
    type B = Autodiff<Wgpu>;

    #[test]
    fn test_distributed_impala_new() {
        let config = IMPALAConfig::default();
        let impala = IMPALA::new(config);

        // Use turbofish to specify the backend type
        assert_eq!(<IMPALA as DistributedAlgorithm<B>>::name(&impala), "IMPALA");
        assert!(<IMPALA as DistributedAlgorithm<B>>::is_off_policy(&impala));
        assert_eq!(<IMPALA as DistributedAlgorithm<B>>::n_epochs(&impala), 1);
        assert_eq!(<IMPALA as DistributedAlgorithm<B>>::n_minibatches(&impala), 1);
    }

    #[test]
    fn test_create_buffer() {
        let config = IMPALAConfig::default();
        let impala: IMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <IMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 4, 32);
        assert_eq!(buffer.config().n_actors, 4);
        assert_eq!(buffer.config().n_envs_per_actor, 32);
    }

    #[test]
    fn test_staleness_handling() {
        use crate::core::transition::{IMPALATransition, Trajectory, Transition};

        let config = IMPALAConfig::default();
        let impala = IMPALA::new(config);

        // Create a batch with some staleness
        let mut traj = Trajectory::new(0);
        traj.push(IMPALATransition {
            base: Transition::new_discrete(vec![0.0], 0, 1.0, vec![1.0], false, false),
            behavior_log_prob: -0.5,
            policy_version: 5,
        });

        let batch = IMPALABatch {
            trajectories: vec![traj],
            policy_versions: vec![5],
        };

        // Handle staleness - IMPALA should keep the batch
        let result = <IMPALA as DistributedAlgorithm<B>>::handle_staleness(&impala, batch, 10);
        assert_eq!(result.trajectories.len(), 1);
        assert_eq!(result.max_staleness(10), 5);
    }
}
