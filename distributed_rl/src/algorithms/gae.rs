//! Generalized Advantage Estimation for PPO/A2C.
//!
//! GAE provides a family of policy gradient estimators parameterized by λ:
//! - λ = 0: one-step TD (low variance, high bias)
//! - λ = 1: Monte Carlo (high variance, low bias)
//! - λ ∈ (0, 1): interpolation
//!
//! ## Formula
//!
//! A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
//! where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
//!
//! ## References
//!
//! - Schulman et al., "High-Dimensional Continuous Control Using
//!   Generalized Advantage Estimation" (2016)

/// Compute GAE advantages and returns for a single trajectory.
///
/// # Arguments
///
/// * `rewards` - rewards received [T]
/// * `values` - value estimates V(s) [T]
/// * `dones` - episode termination flags [T]
/// * `last_value` - V(s_T) for bootstrap (0 if terminal)
/// * `gamma` - discount factor
/// * `gae_lambda` - GAE λ parameter
///
/// # Returns
///
/// (advantages, returns) - both [T]
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    assert_eq!(values.len(), n);
    assert_eq!(dones.len(), n);

    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];

    let mut gae = 0.0f32;
    let mut next_value = last_value;

    for t in (0..n).rev() {
        let not_done = if dones[t] { 0.0 } else { 1.0 };

        // TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        let delta = rewards[t] + gamma * next_value * not_done - values[t];

        // GAE: A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * gae_lambda * not_done * gae;

        advantages[t] = gae;
        returns[t] = gae + values[t];

        next_value = values[t];
    }

    (advantages, returns)
}

/// Compute GAE for vectorized environments.
///
/// Transitions are stored interleaved: [env0_t0, env1_t0, ..., env0_t1, env1_t1, ...]
///
/// # Arguments
///
/// * `rewards` - rewards [n_envs * rollout_len]
/// * `values` - value estimates [n_envs * rollout_len]
/// * `dones` - done flags [n_envs * rollout_len]
/// * `last_values` - bootstrap values for each env [n_envs]
/// * `n_envs` - number of parallel environments
/// * `gamma` - discount factor
/// * `gae_lambda` - GAE λ parameter
///
/// # Returns
///
/// (advantages, returns) - both [n_envs * rollout_len]
pub fn compute_gae_vectorized(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_values: &[f32],
    n_envs: usize,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let total_len = rewards.len();
    assert_eq!(values.len(), total_len);
    assert_eq!(dones.len(), total_len);
    assert_eq!(last_values.len(), n_envs);

    let rollout_len = total_len / n_envs;
    let mut advantages = vec![0.0f32; total_len];
    let mut returns = vec![0.0f32; total_len];

    for env_idx in 0..n_envs {
        // Extract this env's data
        let env_rewards: Vec<f32> = (0..rollout_len)
            .map(|t| rewards[t * n_envs + env_idx])
            .collect();
        let env_values: Vec<f32> = (0..rollout_len)
            .map(|t| values[t * n_envs + env_idx])
            .collect();
        let env_dones: Vec<bool> = (0..rollout_len)
            .map(|t| dones[t * n_envs + env_idx])
            .collect();

        let (env_advantages, env_returns) = compute_gae(
            &env_rewards,
            &env_values,
            &env_dones,
            last_values[env_idx],
            gamma,
            gae_lambda,
        );

        // Write back to interleaved layout
        for t in 0..rollout_len {
            advantages[t * n_envs + env_idx] = env_advantages[t];
            returns[t * n_envs + env_idx] = env_returns[t];
        }
    }

    (advantages, returns)
}

/// Normalize advantages to zero mean and unit variance.
///
/// This is a common practice in PPO to improve training stability.
///
/// # Edge Cases
///
/// - Empty slice: no-op
/// - Single element: sets to 0.0 (can't compute meaningful variance)
/// - All same values: sets all to 0.0 (variance is 0, epsilon prevents NaN)
pub fn normalize_advantages(advantages: &mut [f32]) {
    if advantages.is_empty() {
        return;
    }

    // Single element: can't compute meaningful variance, zero-center
    if advantages.len() == 1 {
        advantages[0] = 0.0;
        return;
    }

    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    // Population variance with epsilon for stability
    let variance = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
    let std = (variance + 1e-8).sqrt();

    for a in advantages.iter_mut() {
        *a = (*a - mean) / std;
    }
}

/// Normalize advantages per-environment (for vectorized case).
///
/// Normalizes within each environment's trajectory separately.
pub fn normalize_advantages_per_env(advantages: &mut [f32], n_envs: usize) {
    if advantages.is_empty() || n_envs == 0 {
        return;
    }

    let rollout_len = advantages.len() / n_envs;

    for env_idx in 0..n_envs {
        // Extract this env's advantages
        let env_advantages: Vec<f32> = (0..rollout_len)
            .map(|t| advantages[t * n_envs + env_idx])
            .collect();

        // Compute mean and std
        let n = env_advantages.len() as f32;
        let mean = env_advantages.iter().sum::<f32>() / n;
        let variance = env_advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
        let std = (variance + 1e-8).sqrt();

        // Normalize in place
        for t in 0..rollout_len {
            advantages[t * n_envs + env_idx] = (advantages[t * n_envs + env_idx] - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gae_simple() {
        // Simple trajectory: 3 steps, no termination
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, false];
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let last_value = 0.5;

        let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

        // Verify dimensions
        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Verify advantages are positive (rewards > values generally)
        for a in &advantages {
            assert!(*a > 0.0, "Expected positive advantages, got {}", a);
        }

        // Verify returns = advantages + values
        for (i, (&a, &v)) in advantages.iter().zip(values.iter()).enumerate() {
            assert!(
                (returns[i] - (a + v)).abs() < 1e-6,
                "return[{}] != advantage[{}] + value[{}]",
                i, i, i
            );
        }
    }

    #[test]
    fn test_gae_with_terminal() {
        let rewards = vec![1.0, 1.0, 0.0];
        let values = vec![0.5, 0.5, 0.0];
        let dones = vec![false, false, true];
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let last_value = 0.0;

        let (advantages, _returns) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

        // Last advantage: δ = 0 - 0 = 0 (reward=0, value=0, done=true so no bootstrap)
        assert!(advantages[2].abs() < 1e-6, "Expected advantages[2]≈0, got {}", advantages[2]);
    }

    #[test]
    fn test_gae_lambda_extremes() {
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let dones = vec![false, false, false];
        let gamma = 0.99;
        let last_value = 0.0;

        // λ = 0: one-step TD, A_t = r_t + γV(s_{t+1}) - V(s_t)
        let (adv_0, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, 0.0);

        // Each advantage should be just the immediate reward (since values are 0)
        assert!((adv_0[2] - 1.0).abs() < 1e-6);

        // λ = 1: MC-like, should have larger advantages for earlier states
        let (adv_1, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, 1.0);

        // First advantage should be largest (accumulates all future rewards)
        assert!(adv_1[0] > adv_1[1]);
        assert!(adv_1[1] > adv_1[2]);
    }

    #[test]
    fn test_gae_vectorized() {
        // 2 envs, 3 steps each
        // Interleaved: [e0t0, e1t0, e0t1, e1t1, e0t2, e1t2]
        let rewards = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let values = vec![0.5, 1.0, 0.5, 1.0, 0.5, 1.0];
        let dones = vec![false, false, false, false, false, false];
        let last_values = vec![0.5, 1.0];  // env0, env1
        let n_envs = 2;
        let gamma = 0.99;
        let gae_lambda = 0.95;

        let (advantages, returns) = compute_gae_vectorized(
            &rewards, &values, &dones, &last_values, n_envs, gamma, gae_lambda
        );

        assert_eq!(advantages.len(), 6);
        assert_eq!(returns.len(), 6);

        // Env 1 should have larger advantages (higher rewards)
        // Check last step
        assert!(advantages[5] > advantages[4], "env1 last > env0 last");
    }

    #[test]
    fn test_normalize_advantages() {
        let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_advantages(&mut advantages);

        // Check mean is ~0
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        assert!(mean.abs() < 1e-6, "Expected mean≈0, got {}", mean);

        // Check std is ~1
        let variance: f32 = advantages.iter().map(|a| a.powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-6, "Expected std≈1, got {}", std);
    }

    #[test]
    fn test_normalize_advantages_empty() {
        let mut advantages: Vec<f32> = vec![];
        normalize_advantages(&mut advantages);
        assert!(advantages.is_empty());
    }

    #[test]
    fn test_normalize_advantages_single() {
        let mut advantages = vec![5.0];
        normalize_advantages(&mut advantages);
        // Single value: (5 - 5) / sqrt(0 + eps) ≈ 0
        assert!(advantages[0].abs() < 1e-3);
    }
}
