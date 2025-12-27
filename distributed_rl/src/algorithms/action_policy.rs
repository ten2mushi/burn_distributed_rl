//! Action policy abstractions for discrete and continuous action spaces.
//!
//! This module provides the core traits for abstracting over action types:
//! - [`ActionValue`]: Scalar action representation for environment interaction
//! - [`PolicyOutput`]: Model output that can sample actions and compute log probs
//! - [`ActionPolicy`]: Main abstraction combining action type and policy output
//!
//! # Implementations
//!
//! - [`DiscreteAction`] / [`DiscretePolicy`]: Categorical distribution over N actions
//! - [`ContinuousAction`] / [`ContinuousPolicy`]: Squashed Gaussian for bounded continuous

use burn::tensor::backend::Backend;
use burn::tensor::{activation::softmax, Int, Tensor};
use std::fmt::Debug;

use super::continuous_policy::{
    entropy_gaussian, log_prob_squashed_gaussian, sample_squashed_gaussian, scale_action,
};

// ============================================================================
// ActionValue Trait - Scalar action representation
// ============================================================================

/// Scalar action value for environment interaction.
///
/// This trait represents actions in a form suitable for:
/// - Storage in experience buffers (as floats)
/// - Passing to environments (as floats)
/// - Conversion to/from tensors for training
pub trait ActionValue: Clone + Send + Sync + Debug + 'static {
    /// Number of floats needed to represent this action.
    /// - Discrete: 1 (the action index)
    /// - Continuous: action_dim
    fn size(&self) -> usize;

    /// Convert to float slice for environment stepping and storage.
    fn as_floats(&self) -> Vec<f32>;

    /// Create from raw float slice (for buffer retrieval).
    fn from_floats(data: &[f32]) -> Self;
}

/// Discrete action value (single index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscreteAction(pub u32);

impl ActionValue for DiscreteAction {
    fn size(&self) -> usize {
        1
    }

    fn as_floats(&self) -> Vec<f32> {
        vec![self.0 as f32]
    }

    fn from_floats(data: &[f32]) -> Self {
        Self(data[0] as u32)
    }
}

impl From<u32> for DiscreteAction {
    fn from(val: u32) -> Self {
        Self(val)
    }
}

impl From<DiscreteAction> for u32 {
    fn from(val: DiscreteAction) -> Self {
        val.0
    }
}

/// Continuous action value (vector of floats).
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousAction(pub Vec<f32>);

impl ActionValue for ContinuousAction {
    fn size(&self) -> usize {
        self.0.len()
    }

    fn as_floats(&self) -> Vec<f32> {
        self.0.clone()
    }

    fn from_floats(data: &[f32]) -> Self {
        Self(data.to_vec())
    }
}

impl From<Vec<f32>> for ContinuousAction {
    fn from(val: Vec<f32>) -> Self {
        Self(val)
    }
}

// ============================================================================
// PolicyOutput Trait - Model output for sampling and log prob
// ============================================================================

/// Policy output from model forward pass.
///
/// This trait abstracts over the different policy parameterizations:
/// - Discrete: logits → softmax → categorical
/// - Continuous: (mean, log_std) → squashed Gaussian
///
/// It provides methods for both:
/// - Rollout collection: [`sample`] (detached, returns scalar actions)
/// - Training: [`log_prob`], [`entropy`] (with gradient flow)
pub trait PolicyOutput<B: Backend>: Clone + Send + 'static {
    /// The action value type produced by sampling this policy.
    type Action: ActionValue;

    /// Sample actions and compute log probabilities (for rollout collection).
    ///
    /// Returns `(actions, log_probs)` where:
    /// - actions: One action per batch item
    /// - log_probs: Log probability of each sampled action
    fn sample(&self, device: &B::Device) -> (Vec<Self::Action>, Vec<f32>);

    /// Compute log probabilities for given actions (with gradient flow).
    ///
    /// # Arguments
    /// * `actions` - Action tensor appropriate for this policy type
    ///   - Discrete: [batch] int tensor of action indices
    ///   - Continuous: [batch, action_dim] float tensor
    fn log_prob(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 1>;

    /// Compute entropy for regularization (with gradient flow).
    fn entropy(&self) -> Tensor<B, 1>;
}

// ============================================================================
// Discrete Policy Output
// ============================================================================

/// Discrete policy output (logits for categorical distribution).
#[derive(Clone)]
pub struct DiscretePolicyOutput<B: Backend> {
    /// Unnormalized log probabilities: [batch, n_actions]
    pub logits: Tensor<B, 2>,
}

impl<B: Backend> DiscretePolicyOutput<B> {
    /// Create from logits tensor.
    pub fn new(logits: Tensor<B, 2>) -> Self {
        Self { logits }
    }

    /// Get probabilities (softmax of logits).
    pub fn probs(&self) -> Tensor<B, 2> {
        softmax(self.logits.clone(), 1)
    }

    /// Number of actions.
    pub fn n_actions(&self) -> usize {
        self.logits.dims()[1]
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.logits.dims()[0]
    }
}

impl<B: Backend> PolicyOutput<B> for DiscretePolicyOutput<B> {
    type Action = DiscreteAction;

    fn sample(&self, _device: &B::Device) -> (Vec<Self::Action>, Vec<f32>) {
        let probs = self.probs();
        let probs_data = probs.to_data();
        let probs_slice: &[f32] = probs_data.as_slice().expect("Failed to get probs slice");

        let batch_size = self.batch_size();
        let n_actions = self.n_actions();

        let mut actions = Vec::with_capacity(batch_size);
        let mut log_probs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Categorical sampling via cumulative sum
            let rand_val = fastrand::f32();
            let mut cumsum = 0.0;
            let mut selected = (n_actions - 1) as u32; // Default to last action

            for a in 0..n_actions {
                cumsum += probs_slice[i * n_actions + a];
                // Select this action if rand_val < cumsum, OR if this is the last action
                // The second condition handles floating-point precision issues where
                // probs don't sum to exactly 1.0
                if rand_val < cumsum || a == n_actions - 1 {
                    selected = a as u32;
                    break;
                }
            }

            let prob = probs_slice[i * n_actions + selected as usize];
            actions.push(DiscreteAction(selected));
            log_probs.push((prob + 1e-8).ln());
        }

        (actions, log_probs)
    }

    fn log_prob(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 1> {
        let batch_size = actions.len();
        let probs = self.probs();

        // Convert actions to tensor for gather
        let action_indices: Vec<i32> = actions.iter().map(|a| a.0 as i32).collect();
        let actions_tensor: Tensor<B, 1, Int> =
            Tensor::from_ints(action_indices.as_slice(), device);
        let actions_2d: Tensor<B, 2, Int> = actions_tensor.reshape([batch_size, 1]);

        // Gather probabilities for selected actions
        let selected_probs = probs.gather(1, actions_2d);
        let selected_probs_1d: Tensor<B, 1> = selected_probs.flatten(0, 1);

        // Log probability with numerical stability
        (selected_probs_1d + 1e-8).log()
    }

    fn entropy(&self) -> Tensor<B, 1> {
        let probs = self.probs();
        let log_probs = (probs.clone() + 1e-8).log();
        // H = -sum(p * log(p))
        let neg_entropy: Tensor<B, 2> = (probs * log_probs).sum_dim(1);
        -neg_entropy.flatten(0, 1)
    }
}

// ============================================================================
// Continuous Policy Output
// ============================================================================

/// Continuous policy output (squashed Gaussian parameters).
#[derive(Clone)]
pub struct ContinuousPolicyOutput<B: Backend> {
    /// Mean of the Gaussian (pre-squash): [batch, action_dim]
    pub mean: Tensor<B, 2>,
    /// Log standard deviation: [batch, action_dim]
    pub log_std: Tensor<B, 2>,
    /// Action bounds for scaling: (low, high)
    pub bounds: (Vec<f32>, Vec<f32>),
}

impl<B: Backend> ContinuousPolicyOutput<B> {
    /// Create from mean and log_std tensors.
    pub fn new(mean: Tensor<B, 2>, log_std: Tensor<B, 2>, bounds: (Vec<f32>, Vec<f32>)) -> Self {
        Self {
            mean,
            log_std,
            bounds,
        }
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        self.mean.dims()[1]
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.mean.dims()[0]
    }
}

impl<B: Backend> PolicyOutput<B> for ContinuousPolicyOutput<B> {
    type Action = ContinuousAction;

    fn sample(&self, _device: &B::Device) -> (Vec<Self::Action>, Vec<f32>) {
        // Sample using squashed Gaussian
        let (squashed, log_probs_tensor) =
            sample_squashed_gaussian(self.mean.clone(), self.log_std.clone());

        // Scale to action bounds
        let scaled = scale_action(squashed, &self.bounds.0, &self.bounds.1);

        // Convert to vectors
        let scaled_data = scaled.to_data();
        let scaled_slice: &[f32] = scaled_data.as_slice().expect("Failed to get scaled slice");

        let log_probs_data = log_probs_tensor.to_data();
        let log_probs_slice: &[f32] = log_probs_data
            .as_slice()
            .expect("Failed to get log_probs slice");

        let batch_size = self.batch_size();
        let action_dim = self.action_dim();

        let mut actions = Vec::with_capacity(batch_size);
        let mut log_probs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let action_vec: Vec<f32> = (0..action_dim)
                .map(|j| scaled_slice[i * action_dim + j])
                .collect();
            actions.push(ContinuousAction(action_vec));
            log_probs.push(log_probs_slice[i]);
        }

        (actions, log_probs)
    }

    fn log_prob(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 1> {
        let batch_size = actions.len();
        let action_dim = self.action_dim();

        // Reconstruct action tensor (already in scaled space, need to unscale to squashed)
        let mut action_floats = Vec::with_capacity(batch_size * action_dim);
        for action in actions {
            action_floats.extend_from_slice(&action.0);
        }

        let action_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(action_floats.as_slice(), device)
                .reshape([batch_size, action_dim]);

        // Unscale to [-1, 1] range for log_prob computation
        let unscaled = super::continuous_policy::unscale_action(
            action_tensor,
            &self.bounds.0,
            &self.bounds.1,
        );

        // Compute log prob with squashed Gaussian
        log_prob_squashed_gaussian(unscaled, self.mean.clone(), self.log_std.clone())
    }

    fn entropy(&self) -> Tensor<B, 1> {
        // Analytical Gaussian entropy (ignoring tanh correction for entropy bonus)
        entropy_gaussian(self.log_std.clone())
    }
}

// ============================================================================
// ActionPolicy Trait - Main abstraction
// ============================================================================

/// Main action policy trait combining action type and policy configuration.
///
/// This trait provides:
/// - Action type information ([`Self::Action`])
/// - Policy output type ([`Self::Output`])
/// - Conversion utilities between tensors and actions
pub trait ActionPolicy<B: Backend>: Clone + Send + Sync + 'static {
    /// The action value type for environment interaction.
    type Action: ActionValue;

    /// The policy output type from model forward.
    type Output: PolicyOutput<B, Action = Self::Action>;

    /// Number of action dimensions.
    /// - Discrete: 1
    /// - Continuous: action_dim
    fn action_dim(&self) -> usize;

    /// Size of the action space.
    /// - Discrete: n_actions
    /// - Continuous: same as action_dim
    fn action_space_size(&self) -> usize;

    /// Create policy output from model tensors.
    fn create_output(&self, logits_or_mean: Tensor<B, 2>, extra: Option<Tensor<B, 2>>)
        -> Self::Output;

    /// Convert actions to tensor for training.
    fn actions_to_tensor(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 2>;

    /// Convert actions to flat f32 vector for environment stepping.
    fn actions_to_floats(&self, actions: &[Self::Action]) -> Vec<f32> {
        actions.iter().flat_map(|a| a.as_floats()).collect()
    }
}

// ============================================================================
// DiscretePolicy Implementation
// ============================================================================

/// Discrete action policy (categorical distribution).
#[derive(Debug, Clone)]
pub struct DiscretePolicy {
    /// Number of discrete actions.
    pub n_actions: usize,
}

impl DiscretePolicy {
    /// Create a new discrete policy.
    pub fn new(n_actions: usize) -> Self {
        Self { n_actions }
    }
}

impl<B: Backend> ActionPolicy<B> for DiscretePolicy {
    type Action = DiscreteAction;
    type Output = DiscretePolicyOutput<B>;

    fn action_dim(&self) -> usize {
        1
    }

    fn action_space_size(&self) -> usize {
        self.n_actions
    }

    fn create_output(
        &self,
        logits: Tensor<B, 2>,
        _extra: Option<Tensor<B, 2>>,
    ) -> Self::Output {
        DiscretePolicyOutput::new(logits)
    }

    fn actions_to_tensor(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 2> {
        let batch_size = actions.len();
        let action_indices: Vec<f32> = actions.iter().map(|a| a.0 as f32).collect();
        Tensor::<B, 1>::from_floats(action_indices.as_slice(), device).reshape([batch_size, 1])
    }
}

// ============================================================================
// ContinuousPolicy Implementation
// ============================================================================

/// Continuous action policy (squashed Gaussian distribution).
#[derive(Debug, Clone)]
pub struct ContinuousPolicy {
    /// Action dimension.
    pub action_dim: usize,
    /// Action bounds: (low, high) per dimension.
    pub bounds: (Vec<f32>, Vec<f32>),
}

impl ContinuousPolicy {
    /// Create a new continuous policy with specified bounds.
    pub fn new(action_dim: usize, low: Vec<f32>, high: Vec<f32>) -> Self {
        assert_eq!(low.len(), action_dim);
        assert_eq!(high.len(), action_dim);
        Self {
            action_dim,
            bounds: (low, high),
        }
    }

    /// Create with symmetric bounds [-bound, bound] for all dimensions.
    pub fn symmetric(action_dim: usize, bound: f32) -> Self {
        Self::new(action_dim, vec![-bound; action_dim], vec![bound; action_dim])
    }

    /// Create with unit bounds [-1, 1] for all dimensions.
    pub fn unit(action_dim: usize) -> Self {
        Self::symmetric(action_dim, 1.0)
    }
}

impl<B: Backend> ActionPolicy<B> for ContinuousPolicy {
    type Action = ContinuousAction;
    type Output = ContinuousPolicyOutput<B>;

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn action_space_size(&self) -> usize {
        self.action_dim
    }

    fn create_output(
        &self,
        mean: Tensor<B, 2>,
        log_std: Option<Tensor<B, 2>>,
    ) -> Self::Output {
        let log_std = log_std.expect("ContinuousPolicy requires log_std as extra tensor");
        ContinuousPolicyOutput::new(mean, log_std, self.bounds.clone())
    }

    fn actions_to_tensor(&self, actions: &[Self::Action], device: &B::Device) -> Tensor<B, 2> {
        let batch_size = actions.len();
        let action_floats: Vec<f32> = actions.iter().flat_map(|a| a.0.clone()).collect();
        Tensor::<B, 1>::from_floats(action_floats.as_slice(), device)
            .reshape([batch_size, self.action_dim])
    }
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
    fn test_discrete_action() {
        let action = DiscreteAction(5);
        assert_eq!(action.size(), 1);
        assert_eq!(action.as_floats(), vec![5.0]);
        assert_eq!(DiscreteAction::from_floats(&[5.0]), action);
    }

    #[test]
    fn test_continuous_action() {
        let action = ContinuousAction(vec![0.5, -0.3, 0.1]);
        assert_eq!(action.size(), 3);
        assert_eq!(action.as_floats(), vec![0.5, -0.3, 0.1]);
    }

    #[test]
    fn test_discrete_policy_output_sample() {
        let device = Default::default();
        let logits: Tensor<B, 2> = Tensor::from_floats([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], &device);
        let output = DiscretePolicyOutput::new(logits);

        let (actions, log_probs) = output.sample(&device);
        assert_eq!(actions.len(), 2);
        assert_eq!(log_probs.len(), 2);

        // Actions should be valid indices
        for action in &actions {
            assert!(action.0 < 3);
        }
    }

    #[test]
    fn test_discrete_policy_output_entropy() {
        let device = Default::default();
        // Uniform distribution should have higher entropy
        let uniform: Tensor<B, 2> = Tensor::from_floats([[1.0, 1.0, 1.0]], &device);
        let output_uniform = DiscretePolicyOutput::new(uniform);

        // Peaked distribution should have lower entropy
        let peaked: Tensor<B, 2> = Tensor::from_floats([[10.0, 0.0, 0.0]], &device);
        let output_peaked = DiscretePolicyOutput::new(peaked);

        let entropy_uniform = output_uniform.entropy().into_data().as_slice::<f32>().unwrap()[0];
        let entropy_peaked = output_peaked.entropy().into_data().as_slice::<f32>().unwrap()[0];

        assert!(entropy_uniform > entropy_peaked);
    }

    #[test]
    fn test_discrete_policy() {
        let policy = DiscretePolicy::new(4);
        assert_eq!(policy.n_actions, 4);
        // Test trait methods with explicit type
        assert_eq!(<DiscretePolicy as ActionPolicy<B>>::action_dim(&policy), 1);
        assert_eq!(<DiscretePolicy as ActionPolicy<B>>::action_space_size(&policy), 4);
    }

    #[test]
    fn test_continuous_policy() {
        let policy = ContinuousPolicy::symmetric(3, 2.0);
        assert_eq!(policy.action_dim, 3);
        assert_eq!(policy.bounds.0, vec![-2.0, -2.0, -2.0]);
        assert_eq!(policy.bounds.1, vec![2.0, 2.0, 2.0]);
        // Test trait methods with explicit type
        assert_eq!(<ContinuousPolicy as ActionPolicy<B>>::action_dim(&policy), 3);
        assert_eq!(<ContinuousPolicy as ActionPolicy<B>>::action_space_size(&policy), 3);
    }
}
