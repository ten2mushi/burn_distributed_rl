//! Automatic entropy coefficient tuning for SAC.
//!
//! SAC uses entropy regularization to encourage exploration:
//! ```text
//! J(π) = E[r + γV] + α * H(π)
//! ```
//!
//! With automatic tuning, α is learned to maintain a target entropy level:
//! ```text
//! max_α E[-α * (log π + H_target)]
//! ```
//!
//! This module provides `EntropyTuner` which manages the learnable log_alpha
//! parameter and computes the entropy loss for optimization.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Entropy Tuner
// ============================================================================

/// Manages automatic entropy coefficient (alpha) tuning for SAC.
///
/// SAC's objective includes an entropy term: max_π E[Q - α*log π]
/// where α controls the exploration-exploitation trade-off.
///
/// With automatic tuning, we optimize α to maintain target entropy:
/// ```text
/// min_α E[α * (-log π - H_target)]
/// ```
///
/// # Usage
///
/// ```ignore
/// // Create tuner
/// let tuner = EntropyTuner::new(0.2, -3.0, &device);
///
/// // Get current alpha for actor loss
/// let alpha = tuner.alpha();
///
/// // Compute alpha loss after actor update
/// let alpha_loss = tuner.loss(log_probs);
///
/// // Backprop and optimizer step...
///
/// // Update cached value for actors
/// tuner.update_cache();
/// ```
pub struct EntropyTuner<B: AutodiffBackend> {
    /// Log of alpha (learnable parameter).
    /// We optimize log_alpha to ensure α > 0.
    log_alpha: Tensor<B, 1>,

    /// Target entropy level.
    /// - Continuous: typically -dim(A)
    /// - Discrete: typically 0.89 * log(|A|)
    target_entropy: f32,

    /// Cached alpha value (for actors without tensor ops).
    cached_alpha: AtomicU64,
}

impl<B: AutodiffBackend> EntropyTuner<B> {
    /// Create a new entropy tuner.
    ///
    /// # Arguments
    /// - `initial_alpha`: Initial entropy coefficient (e.g., 0.2)
    /// - `target_entropy`: Target entropy level
    /// - `device`: Device for tensor operations
    pub fn new(initial_alpha: f32, target_entropy: f32, device: &B::Device) -> Self {
        // Initialize log_alpha such that exp(log_alpha) = initial_alpha
        let log_alpha_value = initial_alpha.ln();
        let log_alpha = Tensor::from_floats([log_alpha_value], device);

        Self {
            log_alpha,
            target_entropy,
            cached_alpha: AtomicU64::new(initial_alpha.to_bits() as u64),
        }
    }

    /// Get the current entropy coefficient.
    ///
    /// Computes α = exp(log_alpha).
    pub fn alpha(&self) -> f32 {
        let exp_log_alpha = self.log_alpha.clone().exp();
        let data = exp_log_alpha.into_data();
        data.as_slice::<f32>().unwrap()[0]
    }

    /// Get the cached alpha value (for actor threads).
    ///
    /// Avoids tensor operations - just reads the cached atomic.
    /// Call `update_cache()` after each optimizer step.
    pub fn cached_alpha(&self) -> f32 {
        let bits = self.cached_alpha.load(Ordering::Relaxed);
        f32::from_bits(bits as u32)
    }

    /// Compute the alpha loss for optimization.
    ///
    /// The loss is: E[α * (-log π - H_target)]
    ///            = -α * E[log π + H_target]
    ///            = -exp(log_alpha) * (mean_log_prob + H_target)
    ///
    /// Note: log_probs should be detached (no gradient through actor).
    ///
    /// # Arguments
    /// - `log_probs`: Log probabilities of actions [batch] (detached!)
    ///
    /// # Returns
    /// Scalar alpha loss for backprop
    pub fn loss(&self, log_probs: Tensor<B, 1>) -> Tensor<B, 1> {
        let alpha = self.log_alpha.clone().exp();
        let mean_log_prob = log_probs.mean();

        // L(α) = -α * (log π + H_target)
        // Minimizing this encourages:
        // - If entropy is too low (log_prob too high), increase α
        // - If entropy is too high (log_prob too low), decrease α
        -(alpha * (mean_log_prob + self.target_entropy))
    }

    /// Compute alpha loss for discrete actions.
    ///
    /// For discrete policies, we use the full distribution entropy:
    /// L(α) = E[α * (-H(π) - H_target)]
    ///      = E[-α * (sum_a π(a|s) * log π(a|s) + H_target)]
    ///
    /// # Arguments
    /// - `action_probs`: Action probabilities [batch, n_actions]
    /// - `log_probs`: Log probabilities [batch, n_actions] (detached!)
    ///
    /// # Returns
    /// Scalar alpha loss for backprop
    pub fn loss_discrete(
        &self,
        action_probs: Tensor<B, 2>,
        log_probs: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let alpha = self.log_alpha.clone().exp();

        // Compute entropy: H = -sum_a π(a) * log π(a)
        let neg_entropy: Tensor<B, 2> = (action_probs * log_probs).sum_dim(1);
        let entropy: Tensor<B, 1> = -neg_entropy.flatten(0, 1);
        let mean_entropy = entropy.mean();

        // L(α) = α * (H_target - H)
        // We want entropy >= H_target, so minimize this
        alpha * (self.target_entropy - mean_entropy)
    }

    /// Update the cached alpha after optimizer step.
    ///
    /// Call this after each alpha optimizer step to update the
    /// atomic cache used by actor threads.
    pub fn update_cache(&self) {
        let alpha = self.alpha();
        self.cached_alpha.store(alpha.to_bits() as u64, Ordering::Relaxed);
    }

    /// Get the log_alpha tensor for the optimizer.
    ///
    /// The optimizer should be created with this tensor as the parameter.
    pub fn log_alpha_tensor(&self) -> Tensor<B, 1> {
        self.log_alpha.clone()
    }

    /// Get the target entropy.
    pub fn target_entropy(&self) -> f32 {
        self.target_entropy
    }

    /// Set the log_alpha from a new tensor (after optimizer step).
    ///
    /// This is needed because Burn optimizers return a new tensor
    /// rather than modifying in place.
    pub fn set_log_alpha(&mut self, log_alpha: Tensor<B, 1>) {
        self.log_alpha = log_alpha;
    }

    /// Create a cloned tuner with the same configuration.
    ///
    /// Useful for creating target entropy tuners or checkpoints.
    pub fn clone_config(&self, device: &B::Device) -> Self {
        Self::new(self.cached_alpha(), self.target_entropy, device)
    }
}

// ============================================================================
// Entropy Computation Helpers
// ============================================================================

/// Compute target entropy for continuous action space.
///
/// Following SAC paper: H_target = -dim(A)
/// This heuristic assumes each action dimension should have
/// a standard deviation of about 1.
pub fn target_entropy_continuous(action_dim: usize) -> f32 {
    -(action_dim as f32)
}

/// Compute target entropy for discrete action space.
///
/// Following SAC-Discrete paper: H_target = 0.89 * log(|A|)
/// This encourages exploration proportional to action space size.
pub fn target_entropy_discrete(n_actions: usize, scale: f32) -> f32 {
    scale * (n_actions as f32).ln()
}

/// Compute entropy of a categorical distribution.
///
/// H = -sum_a π(a) * log π(a)
///
/// # Arguments
/// - `log_probs`: Log probabilities [batch, n_actions]
///
/// # Returns
/// Entropy per sample [batch]
pub fn categorical_entropy<B: AutodiffBackend>(log_probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let probs = log_probs.clone().exp();
    let neg_entropy: Tensor<B, 2> = (probs * log_probs).sum_dim(1);
    -neg_entropy.flatten(0, 1)
}

/// Compute entropy of a Gaussian distribution.
///
/// H = 0.5 * (1 + log(2π)) + log_std
///   ≈ 0.5 * 1.8379 + log_std
///   = 0.919 + log_std (per dimension)
///
/// For multi-dimensional: H = sum_d (0.919 + log_std_d)
///
/// # Arguments
/// - `log_std`: Log standard deviation [batch, action_dim]
///
/// # Returns
/// Entropy per sample [batch]
pub fn gaussian_entropy<B: AutodiffBackend>(log_std: Tensor<B, 2>) -> Tensor<B, 1> {
    // H per dimension = 0.5 * log(2πe) + log_std ≈ 0.9189 + log_std
    const HALF_LOG_2PI_E: f32 = 0.9189;

    let entropy_per_dim = log_std.add_scalar(HALF_LOG_2PI_E);
    let total_entropy: Tensor<B, 2> = entropy_per_dim.sum_dim(1);
    total_entropy.flatten(0, 1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_entropy_tuner_creation() {
        let device = <<B as AutodiffBackend>::InnerBackend as burn::tensor::backend::Backend>::Device::default();
        let tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

        assert!((tuner.alpha() - 0.2).abs() < 0.01);
        assert!((tuner.cached_alpha() - 0.2).abs() < 0.01);
        assert_eq!(tuner.target_entropy(), -3.0);
    }

    #[test]
    fn test_entropy_tuner_loss() {
        let device = <<B as AutodiffBackend>::InnerBackend as burn::tensor::backend::Backend>::Device::default();
        let tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

        // Log probs with entropy higher than target (more negative log_probs)
        let log_probs: Tensor<B, 1> = Tensor::from_floats([-2.0, -3.0, -4.0], &device);

        let loss = tuner.loss(log_probs);
        let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

        // Loss should be computed correctly
        // L = -α * (mean_log_prob + H_target)
        // mean_log_prob = (-2 - 3 - 4) / 3 = -3.0
        // L = -0.2 * (-3.0 + (-3.0)) = -0.2 * (-6.0) = 1.2
        assert!((loss_val - 1.2).abs() < 0.1);
    }

    #[test]
    fn test_cached_alpha_update() {
        let device = <<B as AutodiffBackend>::InnerBackend as burn::tensor::backend::Backend>::Device::default();
        let mut tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

        // Simulate optimizer step by modifying log_alpha
        let new_log_alpha = Tensor::from_floats([0.0], &device); // exp(0) = 1.0
        tuner.set_log_alpha(new_log_alpha);

        // Cached value still old
        assert!((tuner.cached_alpha() - 0.2).abs() < 0.01);

        // Update cache
        tuner.update_cache();

        // Now cached value is updated
        assert!((tuner.cached_alpha() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_target_entropy_continuous() {
        let target = target_entropy_continuous(3);
        assert_eq!(target, -3.0);
    }

    #[test]
    fn test_target_entropy_discrete() {
        let target = target_entropy_discrete(4, 0.89);
        // 0.89 * ln(4) ≈ 0.89 * 1.386 ≈ 1.234
        assert!((target - 1.234).abs() < 0.01);
    }

    #[test]
    fn test_gaussian_entropy() {
        let device = <<B as AutodiffBackend>::InnerBackend as burn::tensor::backend::Backend>::Device::default();

        // log_std = 0 means std = 1
        let log_std: Tensor<B, 2> = Tensor::from_floats([[0.0, 0.0]], &device);
        let entropy = gaussian_entropy(log_std);

        let val = entropy.into_data().as_slice::<f32>().unwrap()[0];

        // H = 2 * 0.9189 = 1.8378 for 2D Gaussian with std=1
        assert!((val - 1.8378).abs() < 0.01);
    }

    #[test]
    fn test_categorical_entropy() {
        let device = <<B as AutodiffBackend>::InnerBackend as burn::tensor::backend::Backend>::Device::default();

        // Uniform distribution over 4 actions (prob = 0.25 each)
        let log_probs: Tensor<B, 2> = Tensor::from_floats(
            [[(0.25_f32).ln(), (0.25_f32).ln(), (0.25_f32).ln(), (0.25_f32).ln()]],
            &device,
        );
        let entropy = categorical_entropy(log_probs);

        let val = entropy.into_data().as_slice::<f32>().unwrap()[0];

        // H of uniform distribution = log(4) ≈ 1.386
        assert!((val - (4.0_f32).ln()).abs() < 0.01);
    }
}
