//! Continuous policy utilities for Gaussian and Squashed Gaussian distributions.
//!
//! This module provides utilities for continuous action spaces using:
//! - **Diagonal Gaussian**: Independent Gaussian per action dimension
//! - **Squashed Gaussian (SAC-style)**: Gaussian with tanh squashing for bounded actions
//!
//! The squashed Gaussian applies `tanh` to bound actions to [-1, 1], then scales
//! to the actual action bounds. This is the standard approach used in SAC and
//! continuous PPO implementations.
//!
//! # Log Probability Correction
//!
//! When using tanh squashing, the log probability must be corrected:
//! ```text
//! log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
//! ```
//! where `u` is the pre-squashed action and `a = tanh(u)`.

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation::tanh, Distribution, Tensor};

// Constants for numerical stability
const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;
const EPSILON: f32 = 1e-6;

/// Sample from a diagonal Gaussian distribution (no squashing).
///
/// # Arguments
/// * `mean` - Mean of the Gaussian: [batch_size, action_dim]
/// * `log_std` - Log standard deviation: [batch_size, action_dim]
///
/// # Returns
/// * `(samples, log_probs)` - Sampled actions and their log probabilities
///   - samples: [batch_size, action_dim]
///   - log_probs: [batch_size] (summed over action dimensions)
pub fn sample_gaussian<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let device = mean.device();
    let dims = mean.dims();
    let batch_size = dims[0];
    let action_dim = dims[1];

    // Clamp log_std for numerical stability
    let log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX);
    let std = log_std.clone().exp();

    // Sample from standard normal
    let noise: Tensor<B, 2> =
        Tensor::random([batch_size, action_dim], Distribution::Normal(0.0, 1.0), &device);

    // Reparameterization: sample = mean + std * noise
    let samples = mean.clone() + std.clone() * noise.clone();

    // Log probability: log N(x; μ, σ) = -0.5 * ((x - μ)/σ)² - log(σ) - 0.5 * log(2π)
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let normalized = noise; // (samples - mean) / std = noise
    let log_prob_per_dim: Tensor<B, 2> =
        -0.5 * normalized.powf_scalar(2.0) - log_std - 0.5 * log_2pi;

    // Sum over action dimensions
    let log_probs: Tensor<B, 1> = log_prob_per_dim.sum_dim(1).squeeze();

    (samples, log_probs)
}

/// Sample from a squashed Gaussian distribution (SAC-style).
///
/// Applies tanh to bound the output to [-1, 1]. The log probability is
/// corrected for the change of variables.
///
/// # Arguments
/// * `mean` - Mean of the Gaussian (pre-squash): [batch_size, action_dim]
/// * `log_std` - Log standard deviation: [batch_size, action_dim]
///
/// # Returns
/// * `(squashed_samples, log_probs)` - Squashed actions and corrected log probabilities
///   - squashed_samples: [batch_size, action_dim] in range (-1, 1)
///   - log_probs: [batch_size] (summed over action dimensions, with tanh correction)
pub fn sample_squashed_gaussian<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    let (pre_squash, gaussian_log_prob) = sample_gaussian(mean, log_std);

    // Apply tanh squashing
    let squashed = tanh(pre_squash.clone());

    // Correction for tanh: log(1 - tanh²(u)) = log(1 - a²) where a = tanh(u)
    // For numerical stability: 1 - tanh²(u) = sech²(u) = 4 / (e^u + e^-u)²
    // We use: log(1 - tanh²(x)) = 2 * (log(2) - x - softplus(-2x))
    let correction = squash_correction(pre_squash);

    // Corrected log probability
    let log_probs = gaussian_log_prob - correction;

    (squashed, log_probs)
}

/// Compute log probability of a squashed Gaussian given the action.
///
/// Given a squashed action `a = tanh(u)`, compute log π(a|s).
///
/// # Arguments
/// * `squashed_action` - The squashed action in range (-1, 1): [batch_size, action_dim]
/// * `mean` - Mean of the Gaussian: [batch_size, action_dim]
/// * `log_std` - Log standard deviation: [batch_size, action_dim]
///
/// # Returns
/// * `log_probs` - Log probabilities: [batch_size]
pub fn log_prob_squashed_gaussian<B: Backend>(
    squashed_action: Tensor<B, 2>,
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // Clamp log_std for numerical stability
    let log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX);

    // Inverse tanh to get pre-squashed action: u = atanh(a)
    // atanh(x) = 0.5 * log((1 + x) / (1 - x))
    // Clamp to avoid numerical issues at boundaries
    let clamped_action = squashed_action.clone().clamp(-1.0 + EPSILON, 1.0 - EPSILON);
    let pre_squash = atanh(clamped_action);

    // Compute Gaussian log probability
    let std = log_std.clone().exp();
    let normalized = (pre_squash.clone() - mean) / std.clone();
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let gaussian_log_prob_per_dim: Tensor<B, 2> =
        -0.5 * normalized.powf_scalar(2.0) - log_std - 0.5 * log_2pi;
    let gaussian_log_prob: Tensor<B, 1> = gaussian_log_prob_per_dim.sum_dim(1).squeeze();

    // Tanh correction
    let correction = squash_correction(pre_squash);

    gaussian_log_prob - correction
}

/// Compute the entropy of a Gaussian distribution (analytical).
///
/// For a diagonal Gaussian with standard deviation σ:
/// H(π) = 0.5 * D * (1 + log(2π)) + Σ log(σ)
///
/// where D is the action dimension.
///
/// # Arguments
/// * `log_std` - Log standard deviation: [batch_size, action_dim]
///
/// # Returns
/// * `entropy` - Per-sample entropy: [batch_size]
pub fn entropy_gaussian<B: Backend>(log_std: Tensor<B, 2>) -> Tensor<B, 1> {
    let action_dim = log_std.dims()[1] as f32;

    // H = 0.5 * D * (1 + log(2π)) + sum(log_std)
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let constant = 0.5 * action_dim * (1.0 + log_2pi);

    let sum_log_std: Tensor<B, 1> = log_std.sum_dim(1).squeeze();

    sum_log_std.add_scalar(constant)
}

/// Compute the entropy loss (negative entropy for minimization).
///
/// # Arguments
/// * `log_std` - Log standard deviation: [batch_size, action_dim]
///
/// # Returns
/// * `entropy_loss` - Scalar loss (mean negative entropy): [1]
pub fn entropy_loss_gaussian<B: AutodiffBackend>(log_std: Tensor<B, 2>) -> Tensor<B, 1> {
    let entropy = entropy_gaussian(log_std);
    -entropy.mean()
}

/// Scale squashed action from [-1, 1] to [low, high].
///
/// # Arguments
/// * `squashed` - Action in [-1, 1]: [batch_size, action_dim]
/// * `low` - Lower bounds: [action_dim]
/// * `high` - Upper bounds: [action_dim]
///
/// # Returns
/// * `scaled` - Action in [low, high]: [batch_size, action_dim]
pub fn scale_action<B: Backend>(
    squashed: Tensor<B, 2>,
    low: &[f32],
    high: &[f32],
) -> Tensor<B, 2> {
    let device = squashed.device();
    let _batch_size = squashed.dims()[0];
    let action_dim = squashed.dims()[1];

    assert_eq!(low.len(), action_dim);
    assert_eq!(high.len(), action_dim);

    // scale = (high - low) / 2
    // offset = (high + low) / 2
    // action = squashed * scale + offset
    let scale: Vec<f32> = low
        .iter()
        .zip(high.iter())
        .map(|(l, h)| (h - l) / 2.0)
        .collect();
    let offset: Vec<f32> = low
        .iter()
        .zip(high.iter())
        .map(|(l, h)| (h + l) / 2.0)
        .collect();

    let scale_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(scale.as_slice(), &device).unsqueeze_dim(0);
    let offset_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(offset.as_slice(), &device).unsqueeze_dim(0);

    // Broadcast: [1, action_dim] * [batch_size, action_dim]
    squashed * scale_tensor + offset_tensor
}

/// Unscale action from [low, high] to [-1, 1].
///
/// # Arguments
/// * `action` - Action in [low, high]: [batch_size, action_dim]
/// * `low` - Lower bounds: [action_dim]
/// * `high` - Upper bounds: [action_dim]
///
/// # Returns
/// * `squashed` - Action in [-1, 1]: [batch_size, action_dim]
pub fn unscale_action<B: Backend>(action: Tensor<B, 2>, low: &[f32], high: &[f32]) -> Tensor<B, 2> {
    let device = action.device();
    let action_dim = action.dims()[1];

    assert_eq!(low.len(), action_dim);
    assert_eq!(high.len(), action_dim);

    // inverse of scale: (action - offset) / scale
    // scale = (high - low) / 2
    // offset = (high + low) / 2
    let scale: Vec<f32> = low
        .iter()
        .zip(high.iter())
        .map(|(l, h)| (h - l) / 2.0)
        .collect();
    let offset: Vec<f32> = low
        .iter()
        .zip(high.iter())
        .map(|(l, h)| (h + l) / 2.0)
        .collect();

    let scale_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(scale.as_slice(), &device).unsqueeze_dim(0);
    let offset_tensor: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(offset.as_slice(), &device).unsqueeze_dim(0);

    (action - offset_tensor) / scale_tensor
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute tanh squashing correction: Σ log(1 - tanh²(u)).
///
/// Uses numerically stable formulation.
fn squash_correction<B: Backend>(pre_squash: Tensor<B, 2>) -> Tensor<B, 1> {
    // log(1 - tanh²(x)) = 2 * log(2) - 2*x - 2*softplus(-2*x)
    // For numerical stability, we use: log(1 - tanh²(x)) ≈ log(4) - 2*|x| for large |x|
    //
    // Actually, the simplest stable form is:
    // log(1 - tanh²(x)) = 2 * (log(2) - x - softplus(-2*x))
    //
    // But even simpler: we can compute directly using the squashed values
    // log(1 - a²) where a = tanh(u), clamped for stability

    let squashed = tanh(pre_squash);
    let one_minus_sq = (-squashed.clone() * squashed + 1.0).clamp(EPSILON, 1.0);
    let log_det_per_dim: Tensor<B, 2> = one_minus_sq.log();
    log_det_per_dim.sum_dim(1).squeeze()
}

/// Inverse hyperbolic tangent: atanh(x) = 0.5 * log((1 + x) / (1 - x))
///
/// # Numerical Stability
///
/// Input should be clamped to (-1, 1) before calling. At x = ±1, atanh is ±∞.
/// We add defensive clamping to prevent log(0) or log(negative).
fn atanh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // Defensive clamp - input should already be in (-1+ε, 1-ε) but be safe
    let x = x.clamp(-1.0 + EPSILON, 1.0 - EPSILON);
    let one_plus_x = x.clone() + 1.0;
    let one_minus_x = -x + 1.0;
    // one_minus_x is guaranteed > ε after clamping, so division is safe
    (one_plus_x / one_minus_x).clamp(EPSILON, f32::MAX).log() * 0.5
}

// ============================================================================
// PPO-specific utilities
// ============================================================================

/// Maximum log ratio before exp() to prevent overflow.
const MAX_LOG_RATIO_CONTINUOUS: f32 = 20.0;

/// Compute PPO policy loss for continuous actions with squashed Gaussian.
///
/// # Arguments
/// * `new_log_probs` - Log probabilities under current policy: [batch_size]
/// * `old_log_probs` - Log probabilities under old policy: [batch_size]
/// * `advantages` - GAE advantages (normalized): [batch_size]
/// * `clip_ratio` - PPO clipping ratio (typically 0.2)
///
/// # Returns
/// * `policy_loss` - Scalar policy loss (for minimization): [1]
///
/// # Numerical Stability
///
/// Log ratio is clamped to [-20, 20] before exp() to prevent overflow.
/// This limits the ratio to approximately [2e-9, 485 million].
pub fn ppo_policy_loss_continuous<B: AutodiffBackend>(
    new_log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    clip_ratio: f32,
) -> Tensor<B, 1> {
    // Probability ratio with numerical stability
    // Clamp log ratio before exp() to prevent overflow
    let log_ratio = new_log_probs - old_log_probs;
    let clamped_log_ratio = log_ratio.clamp(-MAX_LOG_RATIO_CONTINUOUS, MAX_LOG_RATIO_CONTINUOUS);
    let ratio = clamped_log_ratio.exp();

    // Clipped ratio
    let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);

    // Surrogate objectives
    let surr1 = ratio * advantages.clone();
    let surr2 = clipped_ratio * advantages;

    // Take minimum (pessimistic bound)
    let clipped_surr = surr1.min_pair(surr2);

    // Negate for minimization
    -clipped_surr.mean()
}

/// Combined PPO loss for continuous actions.
///
/// total_loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss
///
/// # Arguments
/// * `new_log_probs` - Log probabilities under current policy: [batch_size]
/// * `old_log_probs` - Log probabilities under old policy: [batch_size]
/// * `values` - Current value predictions: [batch_size]
/// * `old_values` - Old value predictions: [batch_size]
/// * `advantages` - GAE advantages (normalized): [batch_size]
/// * `returns` - Target returns: [batch_size]
/// * `log_std` - Current log standard deviation: [batch_size, action_dim]
/// * `clip_ratio` - PPO clipping ratio
/// * `vf_coef` - Value function coefficient
/// * `entropy_coef` - Entropy coefficient
///
/// # Returns
/// * `(total_loss, policy_loss_val, value_loss_val, entropy_val)`
pub fn ppo_combined_loss_continuous<B: AutodiffBackend>(
    new_log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    values: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    log_std: Tensor<B, 2>,
    clip_ratio: f32,
    vf_coef: f32,
    entropy_coef: f32,
) -> (Tensor<B, 1>, f32, f32, f32) {
    // Policy loss
    let policy_loss = ppo_policy_loss_continuous(new_log_probs, old_log_probs, advantages, clip_ratio);

    // Value loss (clipped)
    let values_clipped =
        old_values.clone() + (values.clone() - old_values).clamp(-clip_ratio, clip_ratio);
    let vf_loss1 = (values - returns.clone()).powf_scalar(2.0);
    let vf_loss2 = (values_clipped - returns).powf_scalar(2.0);
    let value_loss = vf_loss1.max_pair(vf_loss2).mean();

    // Entropy loss (negative entropy for minimization)
    let entropy = entropy_gaussian(log_std.clone());
    let entropy_loss = -entropy.clone().mean();

    // Extract scalar values for logging
    let policy_loss_val: f32 = policy_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    let value_loss_val: f32 = value_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    let entropy_val: f32 = entropy.mean().into_data().as_slice::<f32>().unwrap()[0];

    // Combined loss
    let total_loss =
        policy_loss + value_loss.mul_scalar(vf_coef) + entropy_loss.mul_scalar(entropy_coef);

    (total_loss, policy_loss_val, value_loss_val, entropy_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_sample_gaussian() {
        let device = Default::default();
        let mean: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);
        let log_std: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device); // std = 1

        let (samples, log_probs) = sample_gaussian(mean, log_std);

        assert_eq!(samples.dims(), [32, 4]);
        assert_eq!(log_probs.dims(), [32]);

        // Log probs should be reasonable (not NaN or Inf)
        let lp_data = log_probs.into_data();
        let lp_slice = lp_data.as_slice::<f32>().unwrap();
        for &lp in lp_slice {
            assert!(lp.is_finite(), "Log prob should be finite");
        }
    }

    #[test]
    fn test_sample_squashed_gaussian() {
        let device = Default::default();
        let mean: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);
        let log_std: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);

        let (squashed, log_probs) = sample_squashed_gaussian(mean, log_std);

        assert_eq!(squashed.dims(), [32, 4]);
        assert_eq!(log_probs.dims(), [32]);

        // Squashed samples should be in (-1, 1)
        let s_data = squashed.into_data();
        let s_slice = s_data.as_slice::<f32>().unwrap();
        for &s in s_slice {
            assert!(s > -1.0 && s < 1.0, "Squashed sample should be in (-1, 1)");
        }
    }

    #[test]
    fn test_log_prob_squashed_gaussian() {
        let device = Default::default();
        let mean: Tensor<TestBackend, 2> = Tensor::zeros([8, 2], &device);
        let log_std: Tensor<TestBackend, 2> = Tensor::zeros([8, 2], &device);

        // Sample and then compute log prob of the sample
        let (squashed, original_log_probs) = sample_squashed_gaussian(mean.clone(), log_std.clone());

        let computed_log_probs = log_prob_squashed_gaussian(squashed, mean, log_std);

        // Should match (approximately, due to numerical precision)
        let orig_data = original_log_probs.into_data();
        let comp_data = computed_log_probs.into_data();
        let orig_slice = orig_data.as_slice::<f32>().unwrap();
        let comp_slice = comp_data.as_slice::<f32>().unwrap();

        for (o, c) in orig_slice.iter().zip(comp_slice.iter()) {
            assert!(
                (o - c).abs() < 1e-4,
                "Log probs should match: {} vs {}",
                o,
                c
            );
        }
    }

    #[test]
    fn test_entropy_gaussian() {
        let device = Default::default();
        let log_std: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device); // std = 1

        let entropy = entropy_gaussian(log_std);

        assert_eq!(entropy.dims(), [4]);

        // For std=1, entropy per dim = 0.5 * (1 + log(2π)) ≈ 1.419
        // For 2 dims: 2 * 1.419 ≈ 2.838
        let e_data = entropy.into_data();
        let e_slice = e_data.as_slice::<f32>().unwrap();
        for &e in e_slice {
            assert!((e - 2.838).abs() < 0.01, "Entropy should be ~2.838, got {}", e);
        }
    }

    #[test]
    fn test_scale_unscale_action() {
        let device = Default::default();
        let squashed: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.5, -0.5], [0.0, 1.0]], &device);
        let low = vec![-2.0, 0.0];
        let high = vec![2.0, 10.0];

        let scaled = scale_action(squashed.clone(), &low, &high);
        let unscaled = unscale_action(scaled.clone(), &low, &high);

        // Check scaling
        let scaled_data = scaled.into_data();
        let scaled_slice = scaled_data.as_slice::<f32>().unwrap();
        // squashed=0.5 with low=-2, high=2: scale=2, offset=0 -> 0.5*2 + 0 = 1.0
        assert!((scaled_slice[0] - 1.0).abs() < 1e-5);
        // squashed=-0.5 with low=0, high=10: scale=5, offset=5 -> -0.5*5 + 5 = 2.5
        assert!((scaled_slice[1] - 2.5).abs() < 1e-5);

        // Check unscaling recovers original
        let orig_data = squashed.into_data();
        let unsc_data = unscaled.into_data();
        let orig_slice = orig_data.as_slice::<f32>().unwrap();
        let unsc_slice = unsc_data.as_slice::<f32>().unwrap();
        for (o, u) in orig_slice.iter().zip(unsc_slice.iter()) {
            assert!((o - u).abs() < 1e-5, "Unscale should recover original");
        }
    }
}
