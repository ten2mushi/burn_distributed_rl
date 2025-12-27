//! Policy gradient loss functions for PPO and IMPALA.
//!
//! This module provides loss computation functions that work with raw f32 values
//! for algorithm logic, and tensor-based versions for gradient computation.
//!
//! # Numerical Stability
//!
//! All functions that compute importance ratios via exp(log_ratio) clamp the
//! log ratio to [-20, 20] to prevent overflow. This limits ratios to approximately
//! [2e-9, 485 million], which is more than sufficient for any practical scenario.

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Maximum log ratio before exp() to prevent overflow.
/// exp(20) ≈ 485 million, which is far beyond any meaningful importance ratio.
const MAX_LOG_RATIO: f32 = 20.0;

/// PPO clipped surrogate loss (scalar computation for algorithm logic).
///
/// L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
///
/// where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
///
/// # Arguments
///
/// * `log_probs` - Current policy log probs: log π_θ(a|s)
/// * `old_log_probs` - Old policy log probs (from rollout): log π_θ_old(a|s)
/// * `advantages` - GAE advantages (should be normalized)
/// * `clip_ratio` - Clipping ratio ε (typically 0.2)
///
/// # Returns
///
/// Negative of the clipped surrogate objective (for minimization)
///
/// # Numerical Stability
///
/// Log ratio is clamped to [-20, 20] before exp() to prevent overflow.
pub fn ppo_clip_loss_scalar(
    log_probs: &[f32],
    old_log_probs: &[f32],
    advantages: &[f32],
    clip_ratio: f32,
) -> f32 {
    let n = log_probs.len();
    assert_eq!(old_log_probs.len(), n);
    assert_eq!(advantages.len(), n);

    if n == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0f32;

    for i in 0..n {
        // Clamp log ratio to prevent exp() overflow
        let log_ratio = log_probs[i] - old_log_probs[i];
        let clamped_log_ratio = if log_ratio.is_finite() {
            log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
        } else {
            0.0 // Fallback to ratio = 1.0 for NaN/Inf
        };
        let ratio = clamped_log_ratio.exp();
        let clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio);

        let surr1 = ratio * advantages[i];
        let surr2 = clipped_ratio * advantages[i];

        // Take minimum (pessimistic bound)
        total_loss += surr1.min(surr2);
    }

    // Negate for minimization (we want to maximize the objective)
    -total_loss / n as f32
}

/// PPO clipped surrogate loss (tensor computation for gradient).
///
/// # Arguments
///
/// * `log_probs` - Current policy log probs: [batch_size]
/// * `old_log_probs` - Old policy log probs (detached): [batch_size]
/// * `advantages` - GAE advantages (detached, normalized): [batch_size]
/// * `clip_ratio` - Clipping ratio ε (typically 0.2)
///
/// # Returns
///
/// Scalar loss tensor (for backpropagation) - 1D tensor with single element
///
/// # Numerical Stability
///
/// Log ratio is clamped to [-20, 20] before exp() to prevent overflow.
pub fn ppo_clip_loss<B: AutodiffBackend>(
    log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    clip_ratio: f32,
) -> Tensor<B, 1> {
    // Compute probability ratio with numerical stability
    // Clamp log ratio before exp() to prevent overflow
    let log_ratio = log_probs - old_log_probs;
    let clamped_log_ratio = log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO);
    let ratio = clamped_log_ratio.exp();

    // Clipped ratio
    let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);

    // Two surrogate objectives
    let surr1 = ratio * advantages.clone();
    let surr2 = clipped_ratio * advantages;

    // Take minimum (pessimistic bound)
    let clipped_surr = surr1.min_pair(surr2);

    // Negate mean for minimization
    -clipped_surr.mean()
}

/// Value function loss.
///
/// Optionally clipped (PPO2 style) to prevent large value updates.
///
/// # Arguments
///
/// * `values` - Current value predictions: [batch_size]
/// * `old_values` - Old value predictions (for clipping): [batch_size]
/// * `returns` - Target returns: [batch_size]
/// * `clip_value` - Optional value clipping (e.g., 0.2)
pub fn value_loss<B: AutodiffBackend>(
    values: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    clip_value: Option<f32>,
) -> Tensor<B, 1> {
    match clip_value {
        Some(clip) => {
            // Clipped value loss (PPO2 style)
            let values_clipped = old_values.clone()
                + (values.clone() - old_values).clamp(-clip, clip);

            let loss1 = (values - returns.clone()).powf_scalar(2.0);
            let loss2 = (values_clipped - returns).powf_scalar(2.0);

            // Take maximum (conservative update)
            loss1.max_pair(loss2).mean()
        }
        None => {
            // Simple MSE
            (values - returns).powf_scalar(2.0).mean()
        }
    }
}

/// Value function loss (scalar computation).
pub fn value_loss_scalar(
    values: &[f32],
    returns: &[f32],
) -> f32 {
    let n = values.len();
    assert_eq!(returns.len(), n);

    if n == 0 {
        return 0.0;
    }

    let mse: f32 = values.iter()
        .zip(returns.iter())
        .map(|(v, r)| (v - r).powi(2))
        .sum();

    mse / n as f32
}

/// Entropy loss for exploration.
///
/// H(π) = -Σ π(a) log π(a)
///
/// Returns negative entropy (for minimization; add to loss with positive coefficient).
///
/// # Arguments
///
/// * `action_probs` - Action probabilities: [batch_size, n_actions]
pub fn entropy_loss<B: AutodiffBackend>(
    action_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // Clamp probabilities to avoid log(0)
    let probs_clamped = action_probs.clone().clamp(1e-8, 1.0);

    // H(π) = -Σ π(a) log π(a)
    let log_probs = probs_clamped.log();
    // sum_dim returns a 2D tensor, flatten to get 1D then mean
    let per_sample_entropy: Tensor<B, 1> = -(action_probs * log_probs).sum_dim(1).flatten(0, 1);
    let mean_entropy = per_sample_entropy.mean();

    // Return negative entropy (so adding it to loss encourages entropy)
    -mean_entropy
}

/// Compute entropy from log probabilities (scalar).
pub fn entropy_from_log_probs(log_probs: &[f32], probs: &[f32]) -> f32 {
    // H(π) = -Σ π(a) log π(a)
    log_probs.iter()
        .zip(probs.iter())
        .map(|(log_p, p)| -p * log_p)
        .sum()
}

/// Combined PPO loss components (for logging).
pub struct PPOLossComponents {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
}

/// Compute combined PPO loss with all components.
///
/// total_loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss
pub fn ppo_combined_loss<B: AutodiffBackend>(
    log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    values: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    action_probs: Tensor<B, 2>,
    clip_ratio: f32,
    vf_coef: f32,
    entropy_coef: f32,
) -> (Tensor<B, 1>, PPOLossComponents) {
    let policy_loss = ppo_clip_loss(log_probs, old_log_probs, advantages, clip_ratio);
    let vf_loss = value_loss(values, old_values, returns, Some(clip_ratio));
    let ent_loss = entropy_loss(action_probs);

    // Extract scalar values for logging (1D tensors with single element)
    let policy_loss_data = policy_loss.clone().into_data();
    let policy_loss_val: f32 = policy_loss_data.as_slice::<f32>().unwrap()[0];

    let value_loss_data = vf_loss.clone().into_data();
    let value_loss_val: f32 = value_loss_data.as_slice::<f32>().unwrap()[0];

    let ent_loss_data = ent_loss.clone().into_data();
    let entropy_val: f32 = -ent_loss_data.as_slice::<f32>().unwrap()[0]; // Negate to get actual entropy

    let total_loss = policy_loss + vf_loss.mul_scalar(vf_coef) + ent_loss.mul_scalar(entropy_coef);

    let total_loss_data = total_loss.clone().into_data();
    let total_loss_val: f32 = total_loss_data.as_slice::<f32>().unwrap()[0];

    let components = PPOLossComponents {
        policy_loss: policy_loss_val,
        value_loss: value_loss_val,
        entropy: entropy_val,
        total_loss: total_loss_val,
    };

    (total_loss, components)
}

/// IMPALA policy gradient loss with V-trace advantages.
///
/// Uses V-trace corrected advantages for off-policy gradient estimation.
pub fn impala_policy_loss<B: AutodiffBackend>(
    log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Simple policy gradient: -E[log π(a|s) * A]
    -(log_probs * advantages).mean()
}

/// IMPALA policy gradient loss (scalar).
pub fn impala_policy_loss_scalar(
    log_probs: &[f32],
    advantages: &[f32],
) -> f32 {
    let n = log_probs.len();
    assert_eq!(advantages.len(), n);

    if n == 0 {
        return 0.0;
    }

    let loss: f32 = log_probs.iter()
        .zip(advantages.iter())
        .map(|(lp, a)| -lp * a)
        .sum();

    loss / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_clip_loss_scalar_no_clip() {
        // When ratio is within bounds, clipping doesn't affect result
        let log_probs = vec![-1.0, -1.0];
        let old_log_probs = vec![-1.0, -1.0];  // Same = ratio of 1.0
        let advantages = vec![1.0, 1.0];  // Positive advantages

        let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, 0.2);

        // Loss should be negative (we negate the objective)
        // surr = 1.0 * 1.0 = 1.0, mean = 1.0, negated = -1.0
        assert!((loss - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_clip_loss_scalar_clipped() {
        // When ratio is outside bounds, clipping takes effect
        let log_probs = vec![0.0];  // log(1.0)
        let old_log_probs = vec![-1.0];  // log(exp(-1)) ≈ log(0.368)
        // ratio = exp(0 - (-1)) = e ≈ 2.718

        let advantages = vec![1.0];
        let clip_ratio = 0.2;

        let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

        // ratio ≈ 2.718, clipped to 1.2
        // surr1 = 2.718 * 1.0 = 2.718
        // surr2 = 1.2 * 1.0 = 1.2
        // min = 1.2, negated = -1.2
        assert!((loss - (-1.2)).abs() < 0.01);
    }

    #[test]
    fn test_value_loss_scalar() {
        let values = vec![1.0, 2.0, 3.0];
        let returns = vec![1.0, 2.0, 3.0];  // Perfect predictions

        let loss = value_loss_scalar(&values, &returns);
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_value_loss_scalar_mse() {
        let values = vec![1.0, 2.0];
        let returns = vec![2.0, 4.0];  // Errors: 1, 2

        let loss = value_loss_scalar(&values, &returns);
        // MSE = ((1-2)^2 + (2-4)^2) / 2 = (1 + 4) / 2 = 2.5
        assert!((loss - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_impala_policy_loss_scalar() {
        let log_probs = vec![-0.5, -1.0];
        let advantages = vec![1.0, 2.0];

        let loss = impala_policy_loss_scalar(&log_probs, &advantages);

        // loss = -E[log_p * A] = -((-0.5 * 1.0 + -1.0 * 2.0) / 2)
        //      = -((-0.5 - 2.0) / 2) = -(-1.25) = 1.25
        assert!((loss - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_from_log_probs() {
        // Uniform distribution over 2 actions: p = [0.5, 0.5], log_p = [log(0.5), log(0.5)]
        let log_probs = vec![-0.693, -0.693];  // ln(0.5) ≈ -0.693
        let probs = vec![0.5, 0.5];

        let entropy = entropy_from_log_probs(&log_probs, &probs);

        // H = -Σ p * log(p) = -2 * 0.5 * (-0.693) = 0.693
        assert!((entropy - 0.693).abs() < 0.01);
    }
}
