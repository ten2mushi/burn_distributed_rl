//! Unified Actor-Critic model trait for PPO/A2C/TRPO algorithms.
//!
//! This module provides the core abstraction for actor-critic neural networks
//! that work across all combinations of action policies and temporal structures.
//!
//! # Design
//!
//! The trait hierarchy supports both inference and training modes:
//!
//! - [`ActorCriticInference`]: Base trait for inference - works with any `B: Backend`
//!   (including inner backends without autodiff). Used by actor threads.
//!
//! - [`ActorCritic`]: Training trait - requires `B: AutodiffBackend` and extends
//!   `ActorCriticInference`. Used by learner threads for gradient computation.
//!
//! This design follows Burn's philosophy where inference should use `model.valid()`
//! to get an inner backend model without gradient tracking overhead.
//!
//! # Type Parameters
//!
//! - `A: ActionPolicy<B>` - Discrete or continuous action handling
//! - `T: TemporalPolicy<B>` - Feed-forward or recurrent structure
//!
//! # Usage Pattern
//!
//! ```ignore
//! // Learner thread (training with gradients)
//! let model: M where M: ActorCritic<Autodiff<Wgpu>, A, T>
//! let output = model.forward(obs, hidden);  // Creates computation graph
//! let loss = compute_loss(&output);
//! let grads = loss.backward();
//!
//! // Actor thread (inference without gradients)
//! let inference_model = model.valid();  // M::InnerModule on Wgpu
//! let output = inference_model.forward(obs, hidden);  // No graph!
//! ```

use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::{AutodiffBackend, Backend};
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
///
/// This struct is generic over `B: Backend`, allowing it to work with both
/// autodiff backends (for training) and inner backends (for inference).
#[derive(Clone)]
pub struct ForwardOutput<B, A, T>
where
    B: Backend,
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
    B: Backend,
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
// ActorCriticInference Trait - For Inference (Any Backend)
// ============================================================================

/// Base trait for actor-critic inference - works with any Backend.
///
/// This trait is designed to work with both autodiff backends (like `Autodiff<Wgpu>`)
/// and their inner backends (like `Wgpu`). Actor threads should use the inner backend
/// via `model.valid()` to avoid computation graph accumulation.
///
/// # Type Parameters
///
/// - `B`: Any backend (NOT restricted to `AutodiffBackend`)
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
/// // Implement for any Backend (not just AutodiffBackend)
/// impl<B: Backend> ActorCriticInference<B, DiscretePolicy, FeedForward> for MyModel<B> {
///     fn forward(
///         &self,
///         obs: Tensor<B, 2>,
///         _hidden: (),
///     ) -> ForwardOutput<B, DiscretePolicy, FeedForward> {
///         let (logits, values) = self.network_forward(obs);
///         ForwardOutput::new(
///             DiscretePolicyOutput { logits },
///             values,
///             (),
///         )
///     }
///     // ... other methods
/// }
/// ```
pub trait ActorCriticInference<B, A, T>: Module<B> + Clone + Send + 'static
where
    B: Backend,
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
// ActorCritic Trait - For Training (AutodiffBackend)
// ============================================================================

/// Training trait for actor-critic models - requires AutodiffBackend.
///
/// This trait extends [`ActorCriticInference`] and adds the [`AutodiffModule`] bound,
/// enabling gradient computation for training. The learner thread uses this trait.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend (e.g., `Autodiff<Wgpu>`, `Autodiff<NdArray>`)
/// - `A`: Action policy type
/// - `T`: Temporal policy type
///
/// # Design
///
/// This trait deliberately has no additional methods. It serves as a marker that:
/// 1. The model can compute gradients (`AutodiffModule<B>`)
/// 2. The model's `InnerModule` can be used for inference on `B::InnerBackend`
///
/// # Example
///
/// ```ignore
/// // After implementing ActorCriticInference<B: Backend>, implement ActorCritic:
/// impl<B: AutodiffBackend> ActorCritic<B, DiscretePolicy, FeedForward> for MyModel<B> {}
/// ```
pub trait ActorCritic<B, A, T>: ActorCriticInference<B, A, T> + AutodiffModule<B>
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    // Training-specific methods can be added here if needed.
    // Currently empty - training methods are inherited from ActorCriticInference.
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
    B: Backend,
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
    B: Backend,
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
    B: Backend,
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
    B: Backend,
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

    type B = Autodiff<NdArray<f32>>;
    type InnerB = NdArray<f32>;

    /// Simple feed-forward actor-critic for testing.
    #[derive(Module, Debug, Clone)]
    struct TestModel<Backend: burn::tensor::backend::Backend> {
        policy_head: Linear<Backend>,
        value_head: Linear<Backend>,
        #[module(skip)]
        n_actions: usize,
        #[module(skip)]
        obs_size: usize,
    }

    impl<Backend: burn::tensor::backend::Backend> TestModel<Backend> {
        fn new(obs_size: usize, n_actions: usize, device: &Backend::Device) -> Self {
            Self {
                policy_head: LinearConfig::new(obs_size, n_actions).init(device),
                value_head: LinearConfig::new(obs_size, 1).init(device),
                n_actions,
                obs_size,
            }
        }
    }

    // Implement ActorCriticInference for any Backend
    impl<Backend: burn::tensor::backend::Backend> ActorCriticInference<Backend, DiscretePolicy, FeedForward>
        for TestModel<Backend>
    {
        fn forward(
            &self,
            obs: Tensor<Backend, 2>,
            _hidden: (),
        ) -> ForwardOutput<Backend, DiscretePolicy, FeedForward> {
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

    // Implement ActorCritic for AutodiffBackend
    impl ActorCritic<B, DiscretePolicy, FeedForward> for TestModel<B> {}

    #[test]
    fn test_actor_critic_forward() {
        let device = <B as Backend>::Device::default();
        let model = TestModel::<B>::new(4, 2, &device);

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
        let model = TestModel::<B>::new(4, 3, &device);

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
        let model = TestModel::<B>::new(4, 2, &device);

        assert!(!model.is_recurrent());
    }

    #[test]
    fn test_inner_backend_inference() {
        // This test validates that the model.valid() pattern works for inner backend inference.
        // The actual inner backend inference is validated in the runners where the generic
        // constraints are properly expressed through the function signatures.
        //
        // Note: Direct type conversion from TestModel<Autodiff<NdArray>> to TestModel<NdArray>
        // requires explicit trait bounds in function signatures, as demonstrated in the runner code:
        //   M::InnerModule: ActorCriticInference<B::InnerBackend, A, T>
        use burn::module::AutodiffModule;

        let device = <B as Backend>::Device::default();
        let model = TestModel::<B>::new(4, 2, &device);

        // Verify that valid() can be called (returns M::InnerModule)
        // The type conversion to inner backend is enforced through trait bounds in runners
        let _inference_model = AutodiffModule::<B>::valid(&model);

        // Forward pass on autodiff backend (training mode) works
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let output = model.forward(obs, ());

        // Verify output
        assert_eq!(output.policy.logits.dims(), [8, 2]);
        assert_eq!(output.values.dims(), [8, 1]);
    }
}
