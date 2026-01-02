//! Training strategies for unified SAC runner.
//!
//! This module provides the `SACTrainingStrategy` trait which encapsulates
//! differences between feed-forward and recurrent SAC training:
//!
//! - **Transition creation**: How to store hidden state in transitions
//! - **Hidden state management**: How to update/reset hidden state
//! - **Training step**: Single-step vs TBPTT training
//!
//! # Design
//!
//! The strategy pattern enables compile-time dispatch via the sealed
//! `TemporalPolicySACStrategy` trait, providing zero-cost abstraction.
//!
//! ```text
//! SACRunner<A, T, B, S>
//!     where S = DefaultSACStrategy<T>
//!           T: TemporalPolicySACStrategy
//!
//! FeedForward  → FeedForwardSACStrategy   (no hidden state)
//! Recurrent    → RecurrentSACStrategy     (TBPTT sequences)
//! ```

pub mod feed_forward;
pub mod recurrent;

pub use feed_forward::FeedForwardSACStrategy;
pub use recurrent::RecurrentSACStrategy;

use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;

use crate::algorithms::action_policy::ActionPolicy;
use crate::algorithms::sac::{
    EntropyTuner, SACActor, SACCritic, SACDataMarker, SACTransition,
};
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig, Recurrent, TemporalPolicy};
use crate::core::target_network::TargetNetworkManager;
use crate::core::transition::Transition;
use crate::runners::sac_config::SACConfig;

// ============================================================================
// SACLossInfo
// ============================================================================

/// Training loss information for a SAC step.
#[derive(Debug, Clone, Default)]
pub struct SACLossInfo {
    /// Critic loss value.
    pub critic_loss: f32,
    /// Actor loss value (0.0 if not updated this step).
    pub actor_loss: f32,
    /// Alpha loss value (0.0 if not updated or not auto-tuning).
    pub alpha_loss: f32,
    /// Current alpha value.
    pub alpha: f32,
    /// Mean Q-value.
    pub mean_q: f32,
    /// Mean entropy.
    pub mean_entropy: f32,
}

// ============================================================================
// SACTrainingStrategy Trait
// ============================================================================

/// Strategy trait for SAC training customization.
///
/// Encapsulates ALL differences between feed-forward and recurrent SAC.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend
/// - `A`: Action policy (discrete or continuous)
/// - `T`: Temporal policy (feed-forward or recurrent)
pub trait SACTrainingStrategy<B, A, T>: Send + Sync + Default + 'static
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
{
    /// Hidden data type stored in transitions.
    type TransitionData: SACDataMarker;

    /// Create a SAC transition from base transition.
    fn create_transition(base: Transition) -> SACTransition<Self::TransitionData>;

    /// Train one step on a batch.
    ///
    /// Updates critic, optionally actor, optionally alpha, and target networks.
    ///
    /// # Returns
    /// (updated_actor, updated_critic, updated_target_critic, loss_info)
    #[allow(clippy::too_many_arguments)]
    fn train_step<Actor, Critic, ActorOpt, CriticOpt>(
        actor: Actor,
        critic: Critic,
        target_critic: Critic,
        actor_optimizer: &mut ActorOpt,
        critic_optimizer: &mut CriticOpt,
        entropy_tuner: &mut EntropyTuner<B>,
        batch: &[SACTransition<Self::TransitionData>],
        config: &SACConfig,
        hidden_config: &HiddenConfig,
        device: &B::Device,
        gradient_step: usize,
        target_manager: &TargetNetworkManager,
    ) -> (Actor, Critic, Critic, SACLossInfo)
    where
        Actor: SACActor<B, A, T> + AutodiffModule<B> + Clone,
        Critic: SACCritic<B, A, T> + AutodiffModule<B> + Clone,
        ActorOpt: Optimizer<Actor, B>,
        CriticOpt: Optimizer<Critic, B>;
}

// ============================================================================
// Sealed Trait for Compile-Time Strategy Selection
// ============================================================================

mod private {
    pub trait Sealed {}
}

/// Sealed trait for compile-time SAC strategy selection.
pub trait TemporalPolicySACStrategy: private::Sealed {
    /// The training strategy type for this temporal policy.
    type Strategy;
}

impl private::Sealed for FeedForward {}
impl TemporalPolicySACStrategy for FeedForward {
    type Strategy = FeedForwardSACStrategy;
}

impl private::Sealed for Recurrent {}
impl TemporalPolicySACStrategy for Recurrent {
    type Strategy = RecurrentSACStrategy;
}

/// Type alias for the default SAC strategy based on temporal policy.
pub type DefaultSACStrategy<T> = <T as TemporalPolicySACStrategy>::Strategy;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_selection_types() {
        // Compile-time check that strategy selection works
        type FFStrategy = DefaultSACStrategy<FeedForward>;
        type RecStrategy = DefaultSACStrategy<Recurrent>;

        fn _check_ff(_: FFStrategy) {}
        fn _check_rec(_: RecStrategy) {}
    }
}
