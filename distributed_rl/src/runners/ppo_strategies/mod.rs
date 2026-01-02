//! Training strategies for unified PPO runner.
//!
//! This module provides the `PPOTrainingStrategy` trait which encapsulates all
//! differences between feed-forward and recurrent PPO training:
//!
//! - **Transition creation**: How to store hidden state (or not) in transitions
//! - **Hidden state management**: How to update/reset hidden state during rollout
//! - **Training data preparation**: Flatten+shuffle vs TBPTT sequences
//! - **Training loop**: Standard minibatch vs sequence-based training
//!
//! # Design
//!
//! The strategy pattern enables compile-time dispatch via the sealed
//! `TemporalPolicyStrategy` trait, providing zero-cost abstraction.
//!
//! ```text
//! PPORunner<A, T, B, S>
//!     where S = DefaultStrategy<T>
//!           T: TemporalPolicyStrategy
//!
//! FeedForward  → FeedForwardStrategy   (trivial hidden ops, shuffle transitions)
//! Recurrent    → RecurrentStrategy     (persist hidden, TBPTT sequences)
//! ```

pub mod feed_forward;
pub mod recurrent;

pub use feed_forward::FeedForwardStrategy;
pub use recurrent::RecurrentStrategy;

use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;

use crate::algorithms::action_policy::{ActionPolicy, ActionValue};
use crate::algorithms::ppo::normalization::SharedRewardNormalizer;
use crate::algorithms::ppo::ppo_transition::{HiddenData, PPOTransition};
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig, Recurrent, TemporalPolicy};
use crate::core::transition::Transition;
use crate::runners::ppo_config::PPOConfig;

// ============================================================================
// PreparedTrainingData
// ============================================================================

/// Training data prepared from per-environment rollouts.
///
/// Contains all data needed for PPO training, organized in a format
/// suitable for the training strategy (flat batches or sequences).
#[derive(Debug)]
pub struct PreparedTrainingData<Act: ActionValue> {
    /// Flattened state observations: [n_transitions, obs_dim]
    pub states: Vec<f32>,
    /// Actions for each transition
    pub actions: Vec<Act>,
    /// Old log probabilities from behavior policy
    pub old_log_probs: Vec<f32>,
    /// Old value estimates
    pub old_values: Vec<f32>,
    /// GAE advantages (normalized if configured)
    pub advantages: Vec<f32>,
    /// Returns (advantages + values)
    pub returns: Vec<f32>,
    /// Done flags for each transition (terminal OR truncated)
    pub dones: Vec<bool>,
    /// Terminal flags for each transition (TRUE episode termination only)
    /// Used for hidden state reset during recurrent training.
    pub terminals: Vec<bool>,
    /// Observation dimension
    pub obs_dim: usize,
    /// Total number of transitions
    pub n_transitions: usize,
    /// Optional sequence metadata for recurrent training
    pub sequence_info: Option<SequenceInfo>,
}

/// Sequence information for recurrent training (TBPTT).
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Per-sequence hidden states (serialized)
    pub initial_hidden_states: Vec<Vec<f32>>,
    /// Sequence lengths
    pub sequence_lengths: Vec<usize>,
    /// Sequence start indices in the flat data
    pub sequence_starts: Vec<usize>,
    /// Mask for valid positions (1.0 = valid, 0.0 = padding)
    pub mask: Option<Vec<f32>>,
}

// ============================================================================
// PPOTrainingStrategy Trait
// ============================================================================

/// Strategy trait for PPO training customization.
///
/// This trait encapsulates ALL differences between feed-forward and recurrent
/// PPO training, enabling a single unified runner implementation.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend
/// - `A`: Action policy (discrete or continuous)
/// - `T`: Temporal policy (feed-forward or recurrent)
///
/// # Implementation Notes
///
/// Implementors should be zero-sized types (ZSTs) for zero-cost abstraction.
/// The strategy is selected at compile time via `DefaultStrategy<T>`.
pub trait PPOTrainingStrategy<B, A, T>: Send + Sync + Default + 'static
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Hidden data type stored in transitions.
    /// - `()` for feed-forward (zero-cost)
    /// - `RecurrentHiddenData` for recurrent
    type TransitionData: HiddenData;

    /// Create a transition from actor state using serialized hidden.
    ///
    /// # Arguments
    ///
    /// * `base` - Core transition data (state, action, reward, etc.)
    /// * `log_prob` - Log probability of action
    /// * `value` - Value estimate V(s)
    /// * `bootstrap_value` - Optional V(s_{t+1}) for truncated rollouts
    /// * `hidden_data` - Serialized hidden state for this environment (empty for feed-forward)
    /// * `sequence_id` - Unique episode/sequence identifier (0 for feed-forward)
    /// * `step_in_sequence` - Position in sequence (0 for feed-forward)
    /// * `is_sequence_start` - Whether this is first step after reset
    #[allow(clippy::too_many_arguments)]
    fn create_transition(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
        hidden_data: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> PPOTransition<Self::TransitionData>;

    /// Prepare training data from per-environment rollouts.
    ///
    /// This handles:
    /// - Reward normalization (if enabled)
    /// - GAE computation per environment
    /// - Advantage normalization (if configured)
    /// - Data organization (flat for feed-forward, sequences for recurrent)
    ///
    /// # Arguments
    ///
    /// * `rollouts` - Per-environment rollouts: `rollouts[env_id] = transitions`
    /// * `config` - PPO configuration
    /// * `hidden_config` - Hidden state configuration
    /// * `reward_normalizer` - Optional reward normalizer (if enabled in config)
    ///
    /// # Returns
    ///
    /// Prepared training data ready for minibatch training
    fn prepare_training_data(
        rollouts: Vec<Vec<PPOTransition<Self::TransitionData>>>,
        config: &PPOConfig,
        hidden_config: &HiddenConfig,
        reward_normalizer: Option<&SharedRewardNormalizer>,
    ) -> PreparedTrainingData<A::Action>;

    /// Train one epoch on prepared data.
    ///
    /// # Arguments
    ///
    /// * `model` - Model to train
    /// * `optimizer` - Optimizer
    /// * `data` - Prepared training data
    /// * `config` - PPO configuration
    /// * `hidden_config` - Hidden state configuration
    /// * `device` - Device for tensor creation
    /// * `current_env_steps` - Current environment step count (for LR annealing)
    ///
    /// # Returns
    ///
    /// `(updated_model, kl_early_stop)` - The trained model and whether KL early stop triggered
    #[allow(clippy::too_many_arguments)]
    fn train_epoch<M, O>(
        model: M,
        optimizer: &mut O,
        data: &PreparedTrainingData<A::Action>,
        config: &PPOConfig,
        hidden_config: &HiddenConfig,
        device: &B::Device,
        current_env_steps: usize,
        initial_lr: f64,
    ) -> (M, bool)
    where
        M: AutodiffModule<B> + crate::algorithms::actor_critic::ActorCritic<B, A, T>,
        O: Optimizer<M, B>;
}

// ============================================================================
// Sealed Trait for Compile-Time Strategy Selection
// ============================================================================

mod private {
    pub trait Sealed {}
}

/// Sealed trait for compile-time strategy selection.
///
/// Maps temporal policy types to their corresponding training strategies:
/// - `FeedForward` → `FeedForwardStrategy`
/// - `Recurrent` → `RecurrentStrategy`
///
/// This trait is sealed to prevent external implementations.
pub trait TemporalPolicyStrategy: private::Sealed {
    /// The training strategy type for this temporal policy.
    type Strategy;
}

impl private::Sealed for FeedForward {}
impl TemporalPolicyStrategy for FeedForward {
    type Strategy = FeedForwardStrategy;
}

impl private::Sealed for Recurrent {}
impl TemporalPolicyStrategy for Recurrent {
    type Strategy = RecurrentStrategy;
}

/// Type alias for the default strategy based on temporal policy.
///
/// Usage:
/// ```ignore
/// PPORunner<A, T, B, DefaultStrategy<T>>
/// ```
pub type DefaultStrategy<T> = <T as TemporalPolicyStrategy>::Strategy;

// ============================================================================
// Shared Utilities
// ============================================================================

/// Normalize advantages with numerical safety guards.
///
/// Returns normalized advantages, or original if normalization fails.
pub fn normalize_advantages(advantages: Vec<f32>) -> Vec<f32> {
    let n = advantages.len();
    if n == 0 {
        log::warn!("Cannot normalize empty advantages batch");
        return advantages;
    }

    let mean = advantages.iter().sum::<f32>() / n as f32;
    let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + 1e-8).sqrt();

    if !mean.is_finite() || !std.is_finite() {
        log::warn!(
            "Non-finite statistics in advantage normalization: mean={}, std={}. Using raw advantages.",
            mean, std
        );
        return advantages;
    }

    let mut nan_count = 0;
    let normalized: Vec<_> = advantages
        .iter()
        .map(|a| {
            let norm = (a - mean) / std;
            if !norm.is_finite() {
                nan_count += 1;
                0.0
            } else {
                norm
            }
        })
        .collect();

    if nan_count > 0 {
        log::warn!(
            "Found {} non-finite advantages after normalization. Replaced with zeros.",
            nan_count
        );
    }

    normalized
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_advantages() {
        let adv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm = normalize_advantages(adv);

        // Mean should be ~0
        let mean: f32 = norm.iter().sum::<f32>() / norm.len() as f32;
        assert!(mean.abs() < 1e-5);

        // Std should be ~1
        let var: f32 = norm.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm.len() as f32;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_normalize_empty() {
        let adv: Vec<f32> = vec![];
        let norm = normalize_advantages(adv);
        assert!(norm.is_empty());
    }

    #[test]
    fn test_strategy_selection_types() {
        // Compile-time check that strategy selection works
        type FFStrategy = DefaultStrategy<FeedForward>;
        type RecStrategy = DefaultStrategy<Recurrent>;

        // These are just type checks - if they compile, the test passes
        fn _check_ff(_: FFStrategy) {}
        fn _check_rec(_: RecStrategy) {}
    }
}
