//! Unified Actor-Critic model trait for PPO/A2C/TRPO algorithms.
//!
//! This module provides the core abstraction for actor-critic neural networks
//! that work across all combinations of action policies and temporal structures.
//!
//! # Design
//!
//! The `ActorCritic` trait is parameterized by:
//! - `A: ActionPolicy<B>` - Discrete or continuous action handling
//! - `T: TemporalPolicy<B>` - Feed-forward or recurrent structure
//!
//! This allows the same trait to be implemented for:
//! - Feed-forward discrete models (standard PPO)
//! - Feed-forward continuous models (continuous PPO)
//! - Recurrent discrete models (LSTM-PPO)
//! - Recurrent continuous models (LSTM-PPO continuous)

use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use super::action_policy::{ActionPolicy, PolicyOutput};
use super::temporal_policy::TemporalPolicy;

// ============================================================================
// ForwardOutput
// ============================================================================

/// Output from an actor-critic forward pass.
///
/// Contains:
/// - `policy`: Policy output (logits or mean/log_std) for action sampling
/// - `values`: Value estimates [batch, 1]
/// - `hidden`: Updated hidden state (() for feed-forward, RecurrentHidden for recurrent)
#[derive(Clone)]
pub struct ForwardOutput<B, A, T>
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Policy output for action sampling.
    pub policy: A::Output,
    /// Value estimates [batch, 1].
    pub values: Tensor<B, 2>,
    /// Updated hidden state (for recurrent policies).
    pub hidden: T::Hidden,
}

impl<B, A, T> ForwardOutput<B, A, T>
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Create a new forward output.
    pub fn new(policy: A::Output, values: Tensor<B, 2>, hidden: T::Hidden) -> Self {
        Self {
            policy,
            values,
            hidden,
        }
    }

    /// Sample actions from the policy output.
    ///
    /// Returns (actions, log_probs) for rollout collection.
    pub fn sample_actions(&self, device: &B::Device) -> (Vec<A::Action>, Vec<f32>) {
        self.policy.sample(device)
    }

    /// Get value estimates as a 1D tensor.
    pub fn values_flat(&self) -> Tensor<B, 1> {
        self.values.clone().flatten(0, 1)
    }
}

// ============================================================================
// ActorCritic Trait
// ============================================================================

/// Unified actor-critic model trait for policy gradient algorithms.
///
/// This trait abstracts over the forward pass of actor-critic networks,
/// allowing the same training logic to work with any combination of:
/// - Action policies (discrete, continuous)
/// - Temporal structures (feed-forward, recurrent)
///
/// # Type Parameters
///
/// - `B`: Autodiff backend (e.g., `Autodiff<NdArray>`)
/// - `A`: Action policy type (e.g., `DiscretePolicy`, `ContinuousPolicy`)
/// - `T`: Temporal policy type (e.g., `FeedForward`, `Recurrent`)
///
/// # Implementation Notes
///
/// For **feed-forward** models:
/// - `hidden` parameter is `()` and can be ignored
/// - Return `()` as the hidden state in output
///
/// For **recurrent** models:
/// - `hidden` contains per-environment hidden states
/// - Must update and return the new hidden states
///
/// # Example
///
/// ```ignore
/// // Feed-forward discrete policy
/// impl<B: AutodiffBackend> ActorCritic<B, DiscretePolicy, FeedForward> for MyModel<B> {
///     fn forward(
///         &self,
///         obs: Tensor<B, 2>,
///         _hidden: (),
///     ) -> ForwardOutput<B, DiscretePolicy, FeedForward> {
///         let (logits, values) = self.network.forward(obs);
///         ForwardOutput::new(
///             DiscretePolicyOutput { logits },
///             values,
///             (),
///         )
///     }
/// }
/// ```
pub trait ActorCritic<B, A, T>: Module<B> + AutodiffModule<B> + Clone + Send + 'static
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Forward pass through the actor-critic network.
    ///
    /// # Arguments
    ///
    /// - `obs`: Observation tensor [batch, obs_size]
    /// - `hidden`: Hidden state (from previous step or initial)
    ///
    /// # Returns
    ///
    /// `ForwardOutput` containing:
    /// - Policy output for action sampling/log_prob computation
    /// - Value estimates
    /// - Updated hidden state
    fn forward(&self, obs: Tensor<B, 2>, hidden: T::Hidden) -> ForwardOutput<B, A, T>;

    /// Observation size expected by the model.
    fn obs_size(&self) -> usize;

    /// Get the action policy configuration.
    fn action_policy(&self) -> A;

    /// Get the temporal policy configuration.
    fn temporal_policy(&self) -> T;

    /// Create initial hidden state for the given number of environments.
    ///
    /// For feed-forward models, returns `()`.
    /// For recurrent models, returns zeroed hidden states.
    fn initial_hidden(&self, n_envs: usize, device: &B::Device) -> T::Hidden {
        self.temporal_policy().initial_hidden(n_envs, device)
    }

    /// Whether this model uses recurrent architecture.
    fn is_recurrent(&self) -> bool {
        T::is_recurrent()
    }
}

// ============================================================================
// Helper Functions for Model Construction
// ============================================================================

/// Helper to create ForwardOutput for feed-forward discrete models.
pub fn forward_output_discrete_ff<B>(
    logits: Tensor<B, 2>,
    values: Tensor<B, 2>,
) -> ForwardOutput<B, super::action_policy::DiscretePolicy, super::temporal_policy::FeedForward>
where
    B: AutodiffBackend,
{
    ForwardOutput::new(
        super::action_policy::DiscretePolicyOutput { logits },
        values,
        (),
    )
}

/// Helper to create ForwardOutput for feed-forward continuous models.
pub fn forward_output_continuous_ff<B>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    values: Tensor<B, 2>,
    bounds: (Vec<f32>, Vec<f32>),
) -> ForwardOutput<B, super::action_policy::ContinuousPolicy, super::temporal_policy::FeedForward>
where
    B: AutodiffBackend,
{
    ForwardOutput::new(
        super::action_policy::ContinuousPolicyOutput {
            mean,
            log_std,
            bounds,
        },
        values,
        (),
    )
}

/// Helper to create ForwardOutput for recurrent discrete models.
pub fn forward_output_discrete_recurrent<B>(
    logits: Tensor<B, 2>,
    values: Tensor<B, 2>,
    hidden: super::temporal_policy::RecurrentHidden<B>,
) -> ForwardOutput<B, super::action_policy::DiscretePolicy, super::temporal_policy::Recurrent>
where
    B: AutodiffBackend,
{
    ForwardOutput::new(
        super::action_policy::DiscretePolicyOutput { logits },
        values,
        hidden,
    )
}

/// Helper to create ForwardOutput for recurrent continuous models.
pub fn forward_output_continuous_recurrent<B>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    values: Tensor<B, 2>,
    bounds: (Vec<f32>, Vec<f32>),
    hidden: super::temporal_policy::RecurrentHidden<B>,
) -> ForwardOutput<B, super::action_policy::ContinuousPolicy, super::temporal_policy::Recurrent>
where
    B: AutodiffBackend,
{
    ForwardOutput::new(
        super::action_policy::ContinuousPolicyOutput {
            mean,
            log_std,
            bounds,
        },
        values,
        hidden,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
    use crate::algorithms::temporal_policy::FeedForward;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::backend::Backend;

    type B = Autodiff<NdArray<f32>>;

    /// Simple feed-forward actor-critic for testing.
    #[derive(Module, Debug)]
    struct TestModel<B: Backend> {
        policy_head: Linear<B>,
        value_head: Linear<B>,
        #[module(skip)]
        n_actions: usize,
        #[module(skip)]
        obs_size: usize,
    }

    impl TestModel<B> {
        fn new(obs_size: usize, n_actions: usize, device: &<B as Backend>::Device) -> Self {
            Self {
                policy_head: LinearConfig::new(obs_size, n_actions).init(device),
                value_head: LinearConfig::new(obs_size, 1).init(device),
                n_actions,
                obs_size,
            }
        }
    }

    impl ActorCritic<B, DiscretePolicy, FeedForward> for TestModel<B> {
        fn forward(
            &self,
            obs: Tensor<B, 2>,
            _hidden: (),
        ) -> ForwardOutput<B, DiscretePolicy, FeedForward> {
            let logits = self.policy_head.forward(obs.clone());
            let values = self.value_head.forward(obs);
            ForwardOutput::new(DiscretePolicyOutput { logits }, values, ())
        }

        fn obs_size(&self) -> usize {
            self.obs_size
        }

        fn action_policy(&self) -> DiscretePolicy {
            DiscretePolicy::new(self.n_actions)
        }

        fn temporal_policy(&self) -> FeedForward {
            FeedForward::new()
        }
    }

    #[test]
    fn test_actor_critic_forward() {
        let device = <B as Backend>::Device::default();
        let model = TestModel::new(4, 2, &device);

        // Create batch of observations
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let hidden = <FeedForward as TemporalPolicy<B>>::initial_hidden(
            &model.temporal_policy(),
            8,
            &device,
        );

        // Forward pass
        let output = model.forward(obs, hidden);

        // Check shapes
        assert_eq!(output.policy.logits.dims(), [8, 2]);
        assert_eq!(output.values.dims(), [8, 1]);
        assert_eq!(output.hidden, ());
    }

    #[test]
    fn test_actor_critic_sample() {
        let device = <B as Backend>::Device::default();
        let model = TestModel::new(4, 3, &device);

        let obs = Tensor::<B, 2>::zeros([16, 4], &device);
        let output = model.forward(obs, ());

        // Sample actions
        let (actions, log_probs) = output.sample_actions(&device);

        assert_eq!(actions.len(), 16);
        assert_eq!(log_probs.len(), 16);

        // Check all actions are valid
        for action in &actions {
            assert!(action.0 < 3, "Action should be in range [0, 3)");
        }
    }

    #[test]
    fn test_helper_functions() {
        let device = <B as Backend>::Device::default();

        // Test discrete feed-forward helper
        let logits = Tensor::<B, 2>::zeros([4, 3], &device);
        let values = Tensor::<B, 2>::zeros([4, 1], &device);
        let output = forward_output_discrete_ff(logits, values);
        assert_eq!(output.policy.logits.dims(), [4, 3]);
        assert_eq!(output.hidden, ());

        // Test continuous feed-forward helper
        let mean = Tensor::<B, 2>::zeros([4, 2], &device);
        let log_std = Tensor::<B, 2>::zeros([4, 2], &device);
        let values = Tensor::<B, 2>::zeros([4, 1], &device);
        let output = forward_output_continuous_ff(
            mean,
            log_std,
            values,
            (vec![-1.0, -1.0], vec![1.0, 1.0]),
        );
        assert_eq!(output.policy.mean.dims(), [4, 2]);
        assert_eq!(output.policy.action_dim(), 2);
        assert_eq!(output.hidden, ());
    }

    #[test]
    fn test_is_recurrent() {
        let device = <B as Backend>::Device::default();
        let model = TestModel::new(4, 2, &device);

        assert!(!model.is_recurrent());
    }
}
