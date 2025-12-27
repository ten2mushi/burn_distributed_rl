//! V-trace off-policy correction for IMPALA.
//!
//! V-trace handles policy lag in distributed training by using importance
//! sampling with clipping to correct for the difference between behavior
//! policy μ (at collection time) and target policy π (at training time).
//!
//! ## Key Equations
//!
//! Importance weights:
//! - ρ_t = min(ρ̄, π(a_t|s_t) / μ(a_t|s_t))  - clipped importance weight
//! - c_t = min(c̄, π(a_t|s_t) / μ(a_t|s_t))  - trace cutting coefficient
//!
//! V-trace target:
//! v_s = V(s) + Σ_{t≥s} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t
//! where δ_t = ρ_t (r_t + γ V(s_{t+1}) - V(s_t))
//!
//! ## References
//!
//! - Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with
//!   Importance Weighted Actor-Learner Architectures" (2018)

/// V-trace computation result.
#[derive(Debug, Clone)]
pub struct VTraceResult {
    /// V-trace targets: v_s for each state
    pub vs: Vec<f32>,
    /// Policy gradient advantages: ρ * (r + γ*v_{next} - V)
    pub advantages: Vec<f32>,
    /// Clipped importance weights
    pub rhos: Vec<f32>,
}

/// Input for V-trace computation.
#[derive(Debug, Clone)]
pub struct VTraceInput {
    /// Log probabilities under behavior policy (at collection time)
    pub behavior_log_probs: Vec<f32>,
    /// Log probabilities under target policy (current policy)
    pub target_log_probs: Vec<f32>,
    /// Rewards received
    pub rewards: Vec<f32>,
    /// Value estimates under current policy
    pub values: Vec<f32>,
    /// Episode termination flags
    pub dones: Vec<bool>,
    /// Bootstrap value V(s_T) for the last state
    pub bootstrap_value: f32,
}

/// Compute V-trace targets and advantages for a single trajectory.
///
/// # Arguments
///
/// * `behavior_log_probs` - log π_μ(a|s) at collection time
/// * `target_log_probs` - log π(a|s) under current policy
/// * `rewards` - rewards received
/// * `values` - V(s) under current policy
/// * `dones` - episode termination flags
/// * `bootstrap_value` - V(s_T) for bootstrap
/// * `gamma` - discount factor
/// * `rho_bar` - importance weight clip (default: 1.0)
/// * `c_bar` - trace cutting clip (default: 1.0)
///
/// # Returns
///
/// VTraceResult containing V-trace targets, advantages, and clipped importance weights.
/// Maximum log ratio before exp() to prevent overflow.
/// exp(20) ≈ 485 million, exp(88) ≈ 1.6e38 (near f32::MAX).
/// We use 20.0 as a conservative limit since importance weights > 1000 are
/// practically useless anyway and indicate severe policy divergence.
const MAX_LOG_RATIO: f32 = 20.0;

pub fn compute_vtrace(
    behavior_log_probs: &[f32],
    target_log_probs: &[f32],
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) -> VTraceResult {
    let n = rewards.len();

    // Defensive: handle empty input
    if n == 0 {
        return VTraceResult {
            vs: vec![],
            advantages: vec![],
            rhos: vec![],
        };
    }

    assert_eq!(behavior_log_probs.len(), n);
    assert_eq!(target_log_probs.len(), n);
    assert_eq!(values.len(), n);
    assert_eq!(dones.len(), n);

    let mut vs = vec![0.0f32; n];
    let mut advantages = vec![0.0f32; n];
    let mut rhos = vec![0.0f32; n];

    // Compute importance weights with numerical stability
    // Pre-clamp log_ratio to prevent exp() overflow
    for i in 0..n {
        let log_ratio = target_log_probs[i] - behavior_log_probs[i];
        // Clamp log ratio to prevent exp() overflow (exp(88) ≈ f32::MAX)
        // Also handle NaN: if either log_prob is NaN, ratio becomes 1.0 (on-policy fallback)
        let clamped_log_ratio = if log_ratio.is_finite() {
            log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
        } else {
            0.0 // Fallback to on-policy (ratio = 1.0)
        };
        let ratio = clamped_log_ratio.exp();
        rhos[i] = ratio.min(rho_bar);
    }

    // Compute c (trace cutting coefficients) with same stability
    let cs: Vec<f32> = (0..n)
        .map(|i| {
            let log_ratio = target_log_probs[i] - behavior_log_probs[i];
            let clamped_log_ratio = if log_ratio.is_finite() {
                log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
            } else {
                0.0
            };
            clamped_log_ratio.exp().min(c_bar)
        })
        .collect();

    // Backward pass for V-trace
    let mut next_vs = bootstrap_value;
    let mut next_value = bootstrap_value;

    for t in (0..n).rev() {
        let not_done = if dones[t] { 0.0 } else { 1.0 };

        // TD error: δ_t = ρ_t * (r_t + γ * V(s_{t+1}) - V(s_t))
        let delta = rhos[t] * (rewards[t] + gamma * next_value * not_done - values[t]);

        // V-trace target: v_s = V(s) + δ_s + γ * c_s * (v_{s+1} - V(s+1))
        vs[t] = values[t] + delta + gamma * not_done * cs[t] * (next_vs - next_value);

        // Advantage for policy gradient using V-trace targets
        // NOTE: rho is NOT included here - it's applied externally in the policy loss
        // Per IMPALA paper eq(3): A_t = r_t + γ*v_{t+1} - V(s_t)
        // Using V-trace target v_{t+1} (not raw value V) provides multi-step credit assignment
        advantages[t] = rewards[t] + gamma * next_vs * not_done - values[t];

        next_vs = vs[t];
        next_value = values[t];
    }

    VTraceResult { vs, advantages, rhos }
}

/// Compute V-trace for a batch of trajectories.
pub fn compute_vtrace_batch(
    trajectories: &[VTraceInput],
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) -> Vec<VTraceResult> {
    trajectories
        .iter()
        .map(|traj| {
            compute_vtrace(
                &traj.behavior_log_probs,
                &traj.target_log_probs,
                &traj.rewards,
                &traj.values,
                &traj.dones,
                traj.bootstrap_value,
                gamma,
                rho_bar,
                c_bar,
            )
        })
        .collect()
}

/// Compute V-trace from IMPALATransition trajectories.
///
/// This is a convenience function that takes transitions and current policy
/// evaluations, computing V-trace targets.
pub fn compute_vtrace_from_transitions(
    behavior_log_probs: &[f32],
    target_log_probs: &[f32],
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    bootstrap_value: f32,
    gamma: f32,
) -> VTraceResult {
    // Use default clipping values from IMPALA paper
    compute_vtrace(
        behavior_log_probs,
        target_log_probs,
        rewards,
        values,
        dones,
        bootstrap_value,
        gamma,
        1.0,  // rho_bar
        1.0,  // c_bar
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vtrace_on_policy() {
        // When behavior = target, V-trace should reduce to TD(λ) with λ=1
        let log_probs = vec![-1.0, -1.0, -1.0];
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, false];

        let result = compute_vtrace(
            &log_probs, // behavior
            &log_probs, // target (same = on-policy)
            &rewards,
            &values,
            &dones,
            0.5,  // bootstrap
            0.99, // gamma
            1.0,  // rho_bar
            1.0,  // c_bar
        );

        // All importance weights should be 1.0 (on-policy)
        for rho in &result.rhos {
            assert!((*rho - 1.0).abs() < 1e-6, "Expected rho=1.0, got {}", rho);
        }

        // V-trace targets should be computed
        assert_eq!(result.vs.len(), 3);
        assert_eq!(result.advantages.len(), 3);
    }

    #[test]
    fn test_vtrace_clipping() {
        // Test that importance weights are clipped
        let behavior_log_probs = vec![-2.0, -2.0];  // Low probability actions
        let target_log_probs = vec![-0.1, -0.1];    // High probability now

        let rewards = vec![1.0, 1.0];
        let values = vec![0.5, 0.5];
        let dones = vec![false, false];

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1.0,  // rho_bar
            1.0,  // c_bar
        );

        // Importance weights should be clipped to 1.0
        for rho in &result.rhos {
            assert!(*rho <= 1.0 + 1e-6, "Expected rho<=1.0, got {}", rho);
        }
    }

    #[test]
    fn test_vtrace_off_policy() {
        // Test off-policy correction
        let behavior_log_probs = vec![-1.0, -1.0];  // behavior policy
        let target_log_probs = vec![-0.5, -0.5];    // target policy (higher prob)

        let rewards = vec![1.0, 1.0];
        let values = vec![0.5, 0.5];
        let dones = vec![false, false];

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1.0,
            1.0,
        );

        // Importance weights should be exp(-0.5 - (-1.0)) = exp(0.5) ≈ 1.65
        // But clipped to 1.0
        for rho in &result.rhos {
            assert!((*rho - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vtrace_terminal_state() {
        // Test with terminal state
        let log_probs = vec![-1.0, -1.0, -1.0];
        let rewards = vec![1.0, 1.0, 0.0];
        let values = vec![0.5, 0.5, 0.0];
        let dones = vec![false, false, true];

        let result = compute_vtrace(
            &log_probs,
            &log_probs,
            &rewards,
            &values,
            &dones,
            0.0,  // bootstrap (terminal)
            0.99,
            1.0,
            1.0,
        );

        // Last advantage should use terminal bootstrap (0)
        assert!(result.vs[2].abs() < 1e-6, "Expected vs[2]≈0 for terminal");
    }

    #[test]
    fn test_vtrace_batch() {
        let traj1 = VTraceInput {
            behavior_log_probs: vec![-1.0, -1.0],
            target_log_probs: vec![-1.0, -1.0],
            rewards: vec![1.0, 1.0],
            values: vec![0.5, 0.5],
            dones: vec![false, false],
            bootstrap_value: 0.5,
        };

        let traj2 = VTraceInput {
            behavior_log_probs: vec![-0.5],
            target_log_probs: vec![-0.5],
            rewards: vec![2.0],
            values: vec![1.0],
            dones: vec![true],
            bootstrap_value: 0.0,
        };

        let results = compute_vtrace_batch(&[traj1, traj2], 0.99, 1.0, 1.0);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vs.len(), 2);
        assert_eq!(results[1].vs.len(), 1);
    }

    #[test]
    fn test_vtrace_empty() {
        let result = compute_vtrace(
            &[],
            &[],
            &[],
            &[],
            &[],
            0.0,
            0.99,
            1.0,
            1.0,
        );

        assert!(result.vs.is_empty());
        assert!(result.advantages.is_empty());
        assert!(result.rhos.is_empty());
    }

    // ========================================================================
    // Regression tests for bug fixes
    // ========================================================================

    #[test]
    fn test_advantage_uses_vtrace_target() {
        // Per IMPALA paper eq(3), advantage uses V-trace target v_{t+1}:
        // A_t = r_t + γ*v_{t+1} - V(s_t)
        // For single step, v_{t+1} = bootstrap, so result is same as using V.
        // For multi-step, V-trace targets provide multi-step credit assignment.

        let log_probs = vec![-1.0];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];
        let bootstrap = 0.8;  // V-trace target for next state
        let gamma = 0.99;

        let result = compute_vtrace(
            &log_probs,
            &log_probs,  // on-policy: rho=1
            &rewards,
            &values,
            &dones,
            bootstrap,
            gamma,
            1.0,
            1.0,
        );

        // For single step: next_vs = bootstrap = 0.8
        // advantage = r + γ*next_vs - V = 1.0 + 0.99*0.8 - 0.5 = 1.292
        let expected_advantage = 1.0 + gamma * bootstrap - values[0];
        assert!(
            (result.advantages[0] - expected_advantage).abs() < 1e-6,
            "Expected advantage={}, got {}",
            expected_advantage,
            result.advantages[0]
        );
    }

    #[test]
    fn test_rho_not_in_advantages() {
        // IMPALA paper: rho weights the policy gradient externally, not in advantages.
        // Advantage = r + γ*v_{t+1} - V(s_t)  (no rho)
        // Policy loss = -rho * log_prob * advantage  (rho applied here)

        // Create off-policy scenario where rho != 1
        let behavior_log_probs = vec![-2.0];  // Low probability action
        let target_log_probs = vec![-0.5];    // Higher probability now
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];
        let bootstrap = 0.5;
        let gamma = 0.99;

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            bootstrap,
            gamma,
            1.0,  // rho_bar
            1.0,  // c_bar
        );

        // Advantage uses V-trace target (which equals bootstrap for single step):
        // A = r + γ*v_{next} - V = 1.0 + 0.99*0.5 - 0.5 = 0.995
        let expected_advantage = 1.0 + gamma * bootstrap - values[0];
        assert!(
            (result.advantages[0] - expected_advantage).abs() < 1e-6,
            "Advantage should not include rho! Expected {}, got {}",
            expected_advantage,
            result.advantages[0]
        );

        // rho is tracked separately for policy loss
        // rho = exp(target - behavior) = exp(-0.5 - (-2.0)) = exp(1.5) ≈ 4.48
        // But clipped to 1.0
        assert!(
            (result.rhos[0] - 1.0).abs() < 1e-6,
            "rho should be clipped to 1.0"
        );
    }

    #[test]
    fn test_vtrace_numerical_stability() {
        // Test with extreme log probabilities
        let behavior_log_probs = vec![-10.0, -0.01, -5.0];
        let target_log_probs = vec![-0.01, -10.0, -5.0];
        let rewards = vec![1.0, -1.0, 0.5];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, false];

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.0,
            0.99,
            1.0,
            1.0,
        );

        // All results should be finite
        for vs in &result.vs {
            assert!(vs.is_finite(), "vs should be finite, got {}", vs);
        }
        for adv in &result.advantages {
            assert!(adv.is_finite(), "advantage should be finite, got {}", adv);
        }
        for rho in &result.rhos {
            assert!(rho.is_finite(), "rho should be finite, got {}", rho);
            assert!(*rho <= 1.0, "rho should be clipped to <= 1.0");
        }
    }

    #[test]
    fn test_vtrace_single_step() {
        // Test V-trace with single transition
        let log_probs = vec![-1.0];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

        let result = compute_vtrace(
            &log_probs,
            &log_probs,
            &rewards,
            &values,
            &dones,
            0.5,  // bootstrap
            0.99,
            1.0,
            1.0,
        );

        assert_eq!(result.vs.len(), 1);
        assert_eq!(result.advantages.len(), 1);
        assert_eq!(result.rhos.len(), 1);

        // For single step on-policy:
        // advantage = r + γ*v_next - V(s) = 1.0 + 0.99*0.5 - 0.5 = 0.995
        // (for single step, v_next = bootstrap)
        let expected_adv = 1.0 + 0.99 * 0.5 - 0.5;
        assert!((result.advantages[0] - expected_adv).abs() < 1e-6);
    }

    #[test]
    fn test_multistep_advantage_uses_vtrace_targets() {
        // Multi-step test to verify advantages use V-trace targets, not raw values.
        // This is critical for proper multi-step credit assignment.
        let log_probs = vec![-1.0, -1.0, -1.0];
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0]; // Poor value estimates
        let dones = vec![false, false, false];
        let bootstrap = 0.0;
        let gamma = 0.99;

        let result = compute_vtrace(
            &log_probs,
            &log_probs, // on-policy: rho=1, c=1
            &rewards,
            &values,
            &dones,
            bootstrap,
            gamma,
            1.0,
            1.0,
        );

        // With on-policy and zero values, V-trace targets accumulate returns:
        // vs[2] = 0 + 1*(1 + 0.99*0 - 0) + 0.99*1*(0 - 0) = 1.0
        // vs[1] = 0 + 1*(1 + 0.99*0 - 0) + 0.99*1*(1.0 - 0) = 1 + 0.99 = 1.99
        // vs[0] = 0 + 1*(1 + 0.99*0 - 0) + 0.99*1*(1.99 - 0) = 1 + 1.9701 = 2.9701

        // Advantages use V-trace targets:
        // adv[2] = 1 + 0.99*bootstrap - 0 = 1.0
        // adv[1] = 1 + 0.99*vs[2] - 0 = 1 + 0.99*1.0 = 1.99
        // adv[0] = 1 + 0.99*vs[1] - 0 = 1 + 0.99*1.99 = 2.9701

        // All advantages should be positive and increasing toward the start
        assert!(result.advantages[0] > result.advantages[1]);
        assert!(result.advantages[1] > result.advantages[2]);
        assert!(result.advantages[2] > 0.9);

        // Verify specific values
        assert!((result.advantages[2] - 1.0).abs() < 1e-5);
        assert!((result.advantages[1] - 1.99).abs() < 1e-4);
        assert!((result.advantages[0] - 2.9701).abs() < 1e-3);
    }
}
