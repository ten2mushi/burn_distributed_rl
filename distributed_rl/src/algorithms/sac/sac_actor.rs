//! SAC Actor trait and output types.
//!
//! The SAC actor network outputs a stochastic policy:
//! - **Continuous**: Squashed Gaussian (mean, log_std) → tanh → scaled to action bounds
//! - **Discrete**: Softmax logits → Categorical distribution
//!
//! Unlike the combined `ActorCritic` trait used by PPO/IMPALA, SAC uses
//! separate actor and critic networks, allowing:
//! - Different update frequencies (delayed actor updates)
//! - Separate optimizers with different learning rates
//! - Target networks only for critics

use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use crate::algorithms::action_policy::{ActionPolicy, PolicyOutput};
use crate::algorithms::temporal_policy::{HiddenStateType, TemporalPolicy};

// ============================================================================
// SAC Actor Output
// ============================================================================

/// Output from SAC actor forward pass.
///
/// Contains the policy output (for action sampling) and updated hidden state
/// (for recurrent policies).
#[derive(Clone)]
pub struct SACActorOutput<B: Backend, A: ActionPolicy<B>, T: TemporalPolicy<B>> {
    /// Policy output (logits for discrete, mean/log_std for continuous).
    pub policy: A::Output,
    /// Updated hidden state (for recurrent policies).
    pub hidden: T::Hidden,
}

impl<B: Backend, A: ActionPolicy<B>, T: TemporalPolicy<B>> SACActorOutput<B, A, T> {
    /// Create a new actor output.
    pub fn new(policy: A::Output, hidden: T::Hidden) -> Self {
        Self { policy, hidden }
    }

    /// Sample actions and compute log probabilities.
    ///
    /// Returns `(actions, log_probs)` where:
    /// - `actions`: Vector of actions (one per batch item)
    /// - `log_probs`: Log probability of each sampled action
    pub fn sample(&self, device: &B::Device) -> (Vec<A::Action>, Vec<f32>) {
        self.policy.sample(device)
    }

    /// Compute log probabilities for given actions.
    ///
    /// Used during training to compute π(a|s) for actions sampled from replay.
    pub fn log_prob(&self, actions: &[A::Action], device: &B::Device) -> Tensor<B, 1> {
        self.policy.log_prob(actions, device)
    }

    /// Compute entropy of the policy.
    ///
    /// Used for entropy regularization in SAC's objective.
    pub fn entropy(&self) -> Tensor<B, 1> {
        self.policy.entropy()
    }
}

// ============================================================================
// SAC Actor Trait (Inference)
// ============================================================================

/// SAC Actor trait for inference.
///
/// This trait defines the interface for SAC actor networks, which can be used
/// on any backend (autodiff or non-autodiff). Actor threads use the non-autodiff
/// version via `model.valid()`.
///
/// # Type Parameters
/// - `B`: Backend (could be autodiff or inner backend)
/// - `A`: Action policy (discrete or continuous)
/// - `T`: Temporal policy (feed-forward or recurrent)
pub trait SACActor<B, A, T>: Module<B> + Clone + Send + 'static
where
    B: Backend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Forward pass through the actor network.
    ///
    /// # Arguments
    /// - `obs`: Observations tensor [batch, obs_dim]
    /// - `hidden`: Hidden state (for recurrent), ignored for feed-forward
    ///
    /// # Returns
    /// Actor output containing policy distribution and updated hidden state.
    fn forward(&self, obs: Tensor<B, 2>, hidden: T::Hidden) -> SACActorOutput<B, A, T>;

    /// Get the observation dimension.
    fn obs_size(&self) -> usize;

    /// Get the action policy configuration.
    fn action_policy(&self) -> A;

    /// Get the temporal policy configuration.
    fn temporal_policy(&self) -> T;

    /// Create initial hidden state for given number of environments.
    fn initial_hidden(&self, n_envs: usize, device: &B::Device) -> T::Hidden {
        self.temporal_policy().initial_hidden(n_envs, device)
    }

    /// Check if this actor uses a recurrent architecture.
    fn is_recurrent(&self) -> bool {
        T::Hidden::is_stateful()
    }

    /// Get the action dimension.
    fn action_dim(&self) -> usize {
        self.action_policy().action_dim()
    }

    /// Get the action space size (number of discrete actions or continuous dim).
    fn action_space_size(&self) -> usize {
        self.action_policy().action_space_size()
    }
}

// ============================================================================
// SAC Actor Trait (Training)
// ============================================================================

/// SAC Actor trait for training with autodiff backend.
///
/// Extends `SACActor` with gradient computation capabilities.
/// The learner thread uses this for training.
pub trait SACActorTraining<B, A, T>: SACActor<B, A, T> + AutodiffModule<B>
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
}

// Blanket implementation for any type that satisfies the requirements
impl<M, B, A, T> SACActorTraining<B, A, T> for M
where
    M: SACActor<B, A, T> + AutodiffModule<B>,
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
}

// ============================================================================
// SAC Actor with Deterministic Action
// ============================================================================

/// Extension trait for getting deterministic (mean) actions.
///
/// Useful for evaluation where we want the most likely action
/// rather than sampling from the distribution.
pub trait SACActorDeterministic<B, A, T>: SACActor<B, A, T>
where
    B: Backend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Get deterministic (mean) action.
    ///
    /// For continuous policies, returns the mean of the Gaussian.
    /// For discrete policies, returns the argmax of the logits.
    fn deterministic_action(&self, obs: Tensor<B, 2>, hidden: T::Hidden, device: &B::Device)
        -> (Vec<A::Action>, T::Hidden);
}

// ============================================================================
// Continuous Actor Helpers
// ============================================================================

/// Log standard deviation bounds for continuous policies.
///
/// These bounds prevent numerical instability in the squashed Gaussian.
pub const LOG_STD_MIN: f32 = -5.0;
pub const LOG_STD_MAX: f32 = 2.0;

/// Clamp log_std to safe bounds.
///
/// Uses a soft clamp via tanh rescaling (from SpinningUp/Denis Yarats):
/// ```text
/// log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (tanh(raw) + 1)
/// ```
pub fn clamp_log_std<B: Backend>(raw_log_std: Tensor<B, 2>) -> Tensor<B, 2> {
    let tanh_out = raw_log_std.tanh();
    let half_range = (LOG_STD_MAX - LOG_STD_MIN) / 2.0;
    // LOG_STD_MIN + half_range * (tanh + 1) = LOG_STD_MIN + half_range + half_range * tanh
    let offset = LOG_STD_MIN + half_range;
    tanh_out.mul_scalar(half_range).add_scalar(offset)
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
    fn test_clamp_log_std() {
        let device = <B as Backend>::Device::default();

        // Test various input values
        let raw: Tensor<B, 2> = Tensor::from_floats([[-10.0], [0.0], [10.0]], &device);
        let clamped = clamp_log_std(raw);

        let data = clamped.into_data();
        let slice: &[f32] = data.as_slice().unwrap();

        // For very negative input, tanh ≈ -1, so result ≈ LOG_STD_MIN
        assert!((slice[0] - LOG_STD_MIN).abs() < 0.01);

        // For zero input, tanh = 0, so result = (LOG_STD_MIN + LOG_STD_MAX) / 2
        let mid = (LOG_STD_MIN + LOG_STD_MAX) / 2.0;
        assert!((slice[1] - mid).abs() < 0.01);

        // For very positive input, tanh ≈ 1, so result ≈ LOG_STD_MAX
        assert!((slice[2] - LOG_STD_MAX).abs() < 0.01);
    }

    #[test]
    fn test_log_std_bounds() {
        // Verify the constants are reasonable
        assert!(LOG_STD_MIN < 0.0);
        assert!(LOG_STD_MAX > 0.0);
        assert!(LOG_STD_MIN < LOG_STD_MAX);

        // exp(LOG_STD_MIN) ≈ 0.0067 (very small variance)
        // exp(LOG_STD_MAX) ≈ 7.39 (moderate variance)
        let std_min = LOG_STD_MIN.exp();
        let std_max = LOG_STD_MAX.exp();

        assert!(std_min > 0.001);
        assert!(std_min < 0.1);
        assert!(std_max > 1.0);
        assert!(std_max < 10.0);
    }
}
