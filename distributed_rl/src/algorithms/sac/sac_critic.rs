//! SAC Critic trait with twin Q-networks.
//!
//! SAC uses twin Q-networks (Q1, Q2) for stable value estimation:
//! - **Double Q-learning**: min(Q1, Q2) reduces overestimation bias
//! - **Target networks**: Slow-moving copies for stable TD targets
//!
//! Architecture differs by action type:
//! - **Continuous**: Q(s, a) → scalar. Action concatenated with state features.
//! - **Discrete**: Q(s) → [n_actions]. Outputs Q-value for each action.

use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use crate::algorithms::action_policy::ActionPolicy;
use crate::algorithms::temporal_policy::{HiddenStateType, TemporalPolicy};

// ============================================================================
// SAC Critic Output
// ============================================================================

/// Output from SAC critic forward pass.
///
/// Contains Q-values from both Q-networks and updated hidden state.
#[derive(Clone)]
pub struct SACCriticOutput<B: Backend, T: TemporalPolicy<B>> {
    /// Q-values from critic 1.
    /// - Continuous: [batch, 1] (single Q-value for (s,a) pair)
    /// - Discrete: [batch, n_actions] (Q-values for all actions)
    pub q1: Tensor<B, 2>,

    /// Q-values from critic 2 (twin Q for stability).
    /// Same shape as q1.
    pub q2: Tensor<B, 2>,

    /// Updated hidden state (for recurrent policies).
    pub hidden: T::Hidden,
}

impl<B: Backend, T: TemporalPolicy<B>> SACCriticOutput<B, T> {
    /// Create a new critic output.
    pub fn new(q1: Tensor<B, 2>, q2: Tensor<B, 2>, hidden: T::Hidden) -> Self {
        Self { q1, q2, hidden }
    }

    /// Get minimum Q-value across both critics (pessimistic estimate).
    ///
    /// This is the key to SAC's stability - using the minimum prevents
    /// the overestimation bias that affects single Q-network methods.
    pub fn min_q(&self) -> Tensor<B, 2> {
        self.q1.clone().min_pair(self.q2.clone())
    }

    /// Get the shape of Q-values [batch, q_dim].
    pub fn shape(&self) -> [usize; 2] {
        self.q1.dims()
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.q1.dims()[0]
    }

    /// Get Q output dimension (1 for continuous, n_actions for discrete).
    pub fn q_dim(&self) -> usize {
        self.q1.dims()[1]
    }
}

// ============================================================================
// SAC Critic Trait (Inference)
// ============================================================================

/// SAC Critic trait for Q-value estimation.
///
/// SAC uses twin Q-networks to reduce overestimation bias.
/// The critic architecture differs by action space:
///
/// **Continuous Action Space:**
/// - Q(s, a) → scalar
/// - Actions are concatenated with state features
/// - Used for computing Q(s, a) for sampled actions
///
/// **Discrete Action Space:**
/// - Q(s) → [n_actions]
/// - Outputs Q-value for every action
/// - Used with policy probs: E_a[Q(s,a)] = sum(π(a|s) * Q(s,a))
///
/// # Type Parameters
/// - `B`: Backend (could be autodiff or inner backend)
/// - `A`: Action policy (for type information)
/// - `T`: Temporal policy (feed-forward or recurrent)
pub trait SACCritic<B, A, T>: Module<B> + Clone + Send + 'static
where
    B: Backend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Forward pass through both Q-networks.
    ///
    /// # Arguments
    /// - `obs`: Observations tensor [batch, obs_dim]
    /// - `action`: Actions (handling differs by action type):
    ///   - Continuous: Some([batch, action_dim]) - actions to evaluate
    ///   - Discrete: None - outputs Q-values for all actions
    /// - `hidden`: Hidden state (for recurrent), ignored for feed-forward
    ///
    /// # Returns
    /// Critic output containing Q1, Q2 values and updated hidden state.
    fn forward(
        &self,
        obs: Tensor<B, 2>,
        action: Option<Tensor<B, 2>>,
        hidden: T::Hidden,
    ) -> SACCriticOutput<B, T>;

    /// Get the observation dimension.
    fn obs_size(&self) -> usize;

    /// Get the action dimension (for continuous) or number of actions (for discrete).
    fn action_dim(&self) -> usize;

    /// Get the temporal policy configuration.
    fn temporal_policy(&self) -> T;

    /// Create initial hidden state for given number of environments.
    fn initial_hidden(&self, n_envs: usize, device: &B::Device) -> T::Hidden {
        self.temporal_policy().initial_hidden(n_envs, device)
    }

    /// Check if this critic uses a recurrent architecture.
    fn is_recurrent(&self) -> bool {
        T::Hidden::is_stateful()
    }

    /// Check if this critic is for discrete action space.
    ///
    /// For discrete, forward() should be called with action=None
    /// and returns Q-values for all actions.
    fn is_discrete(&self) -> bool;
}

// ============================================================================
// SAC Critic Trait (Training)
// ============================================================================

/// SAC Critic trait for training with autodiff backend.
///
/// Extends `SACCritic` with gradient computation capabilities.
pub trait SACCriticTraining<B, A, T>: SACCritic<B, A, T> + AutodiffModule<B>
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
}

// Blanket implementation for any type that satisfies the requirements
impl<M, B, A, T> SACCriticTraining<B, A, T> for M
where
    M: SACCritic<B, A, T> + AutodiffModule<B>,
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
}

// ============================================================================
// Helper Functions for Discrete SAC
// ============================================================================

/// Compute expected Q-value under policy for discrete actions.
///
/// For discrete SAC, we compute the expectation analytically:
/// E_a[Q(s,a) - α*log π(a|s)] = sum_a π(a|s) * (Q(s,a) - α*log π(a|s))
///
/// # Arguments
/// - `q_values`: Q-values for all actions [batch, n_actions]
/// - `action_probs`: π(a|s) for all actions [batch, n_actions]
/// - `log_probs`: log π(a|s) for all actions [batch, n_actions]
/// - `alpha`: Entropy coefficient
///
/// # Returns
/// Expected Q-value minus entropy term [batch, 1]
pub fn expected_q_discrete<B: Backend>(
    q_values: Tensor<B, 2>,
    action_probs: Tensor<B, 2>,
    log_probs: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 1> {
    // sum_a π(a|s) * (Q(s,a) - α*log π(a|s))
    let q_minus_entropy = q_values - log_probs.mul_scalar(alpha);
    let expected: Tensor<B, 2> = (action_probs * q_minus_entropy).sum_dim(1);
    expected.flatten(0, 1)
}

/// Compute target value for discrete SAC.
///
/// V(s') = sum_a π(a|s') * (min_Q(s',a) - α*log π(a|s'))
///
/// # Arguments
/// - `min_q_next`: min(Q1, Q2) for next states [batch, n_actions]
/// - `next_probs`: π(a|s') for all actions [batch, n_actions]
/// - `next_log_probs`: log π(a|s') for all actions [batch, n_actions]
/// - `alpha`: Entropy coefficient
///
/// # Returns
/// Target value V(s') [batch, 1]
pub fn target_value_discrete<B: Backend>(
    min_q_next: Tensor<B, 2>,
    next_probs: Tensor<B, 2>,
    next_log_probs: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 1> {
    expected_q_discrete(min_q_next, next_probs, next_log_probs, alpha)
}

/// Gather Q-values for specific actions (discrete).
///
/// # Arguments
/// - `q_values`: Q-values for all actions [batch, n_actions]
/// - `actions`: Action indices [batch]
/// - `device`: Device for tensor operations
///
/// # Returns
/// Q-values for taken actions [batch]
pub fn gather_q_values<B: Backend>(
    q_values: Tensor<B, 2>,
    actions: &[u32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let batch_size = actions.len();
    let action_indices: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
    let actions_tensor = Tensor::<B, 1, burn::tensor::Int>::from_ints(
        action_indices.as_slice(),
        device,
    );
    let actions_2d = actions_tensor.reshape([batch_size, 1]);

    // Gather Q-values at action indices
    let gathered = q_values.gather(1, actions_2d);
    gathered.flatten(0, 1)
}

// ============================================================================
// Helper Functions for Continuous SAC
// ============================================================================

/// Compute TD target for continuous SAC.
///
/// y = r + γ * (1 - done) * (min_Q(s', a') - α * log π(a'|s'))
///
/// where a' ~ π(·|s')
///
/// # Arguments
/// - `rewards`: Rewards [batch]
/// - `terminals`: Terminal flags [batch]
/// - `min_q_next`: min(Q1, Q2)(s', a') where a' ~ π [batch]
/// - `next_log_probs`: log π(a'|s') [batch]
/// - `gamma`: Discount factor
/// - `alpha`: Entropy coefficient
///
/// # Returns
/// TD targets [batch]
pub fn compute_td_target<B: Backend>(
    rewards: Tensor<B, 1>,
    terminals: Tensor<B, 1>,
    min_q_next: Tensor<B, 1>,
    next_log_probs: Tensor<B, 1>,
    gamma: f32,
    alpha: f32,
) -> Tensor<B, 1> {
    // V(s') = Q(s', a') - α * log π(a'|s')
    let v_next = min_q_next - next_log_probs.mul_scalar(alpha);

    // y = r + γ * (1 - done) * V(s')
    let not_done = terminals.mul_scalar(-1.0).add_scalar(1.0);
    rewards + not_done.mul_scalar(gamma) * v_next
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_gather_q_values() {
        let device = <B as Backend>::Device::default();

        // Q-values for 3 actions
        let q: Tensor<B, 2> = Tensor::from_floats(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device,
        );

        // Select action 2 for first sample, action 0 for second
        let gathered = gather_q_values(q, &[2, 0], &device);

        let data = gathered.into_data();
        let slice: &[f32] = data.as_slice().unwrap();

        assert!((slice[0] - 3.0).abs() < 0.01);
        assert!((slice[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_expected_q_discrete() {
        let device = <B as Backend>::Device::default();

        // Q-values and policy for 2 actions
        let q: Tensor<B, 2> = Tensor::from_floats([[1.0, 2.0]], &device);
        let probs: Tensor<B, 2> = Tensor::from_floats([[0.5, 0.5]], &device);
        let log_probs = probs.clone().log();

        let expected = expected_q_discrete(q, probs, log_probs, 0.0);

        let data = expected.into_data();
        let slice: &[f32] = data.as_slice().unwrap();

        // Without entropy: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert!((slice[0] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_td_target() {
        let device = <B as Backend>::Device::default();

        let rewards: Tensor<B, 1> = Tensor::from_floats([1.0, 1.0], &device);
        let terminals: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0], &device);
        let min_q_next: Tensor<B, 1> = Tensor::from_floats([10.0, 10.0], &device);
        let next_log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);

        let targets = compute_td_target(
            rewards,
            terminals,
            min_q_next,
            next_log_probs,
            0.99,
            0.2,
        );

        let data = targets.into_data();
        let slice: &[f32] = data.as_slice().unwrap();

        // For non-terminal: y = 1.0 + 0.99 * (10.0 - 0.2 * (-1.0)) = 1.0 + 0.99 * 10.2 = 11.098
        assert!((slice[0] - 11.098).abs() < 0.01);

        // For terminal: y = 1.0 (no future reward)
        assert!((slice[1] - 1.0).abs() < 0.01);
    }
}
