//! Modular reward system with compile-time composition.
//!
//! Provides zero-cost generic reward composition via trait implementations
//! on tuples, enabling compile-time optimization without vtable overhead.
//!
//! # Architecture
//!
//! The reward system uses the [`RewardComponent`] trait to define individual
//! reward/penalty terms. Components can be composed via tuples:
//!
//! ```ignore
//! use quadcopter_env::reward::*;
//!
//! // Compose reward at compile-time (zero vtable overhead)
//! let reward = (
//!     PositionError { weight: 1.0 },
//!     AttitudePenalty { weight: 0.5 },
//!     AliveBonus { bonus: 0.1 },
//! );
//! ```
//!
//! # Built-in Components
//!
//! - [`PositionError`] - Penalizes distance from target position
//! - [`VelocityError`] - Penalizes velocity difference from target
//! - [`AttitudePenalty`] - Penalizes roll/pitch deviation
//! - [`AngularVelocityPenalty`] - Penalizes angular velocity magnitude
//! - [`ActionMagnitudePenalty`] - Penalizes deviation from hover thrust
//! - [`ActionRatePenalty`] - Penalizes action changes (smoothness)
//! - [`AliveBonus`] - Constant survival reward

pub mod components;
pub mod presets;

pub use components::*;
pub use presets::*;

#[cfg(feature = "simd")]
use std::simd::f32x8;

use crate::state::QuadcopterState;

/// Trait for reward components that can be composed at compile-time.
///
/// This trait enables zero-cost composition via tuple implementations.
/// Each component computes its contribution to the total reward.
pub trait RewardComponent: Clone + Send + Sync {
    /// Human-readable name for logging/debugging.
    const NAME: &'static str;

    /// Compute reward contribution for a single environment.
    ///
    /// # Arguments
    /// * `state` - Current quadcopter state
    /// * `idx` - Environment index
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32;

    /// Compute reward contribution for 8 environments (SIMD).
    ///
    /// Default implementation calls scalar version 8 times.
    /// Override for SIMD-optimized implementations.
    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        f32x8::from_array([
            self.compute(state, base_idx),
            self.compute(state, base_idx + 1),
            self.compute(state, base_idx + 2),
            self.compute(state, base_idx + 3),
            self.compute(state, base_idx + 4),
            self.compute(state, base_idx + 5),
            self.compute(state, base_idx + 6),
            self.compute(state, base_idx + 7),
        ])
    }
}

// ============================================================================
// Tuple Implementations (HList-style composition)
// ============================================================================

/// Empty tuple - base case for composition.
impl RewardComponent for () {
    const NAME: &'static str = "Empty";

    #[inline(always)]
    fn compute(&self, _state: &QuadcopterState, _idx: usize) -> f32 {
        0.0
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, _state: &QuadcopterState, _base_idx: usize) -> f32x8 {
        f32x8::splat(0.0)
    }
}

/// 2-tuple composition.
impl<A: RewardComponent, B: RewardComponent> RewardComponent for (A, B) {
    const NAME: &'static str = "Composed2";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx) + self.1.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx) + self.1.compute_simd(state, base_idx)
    }
}

/// 3-tuple composition.
impl<A: RewardComponent, B: RewardComponent, C: RewardComponent> RewardComponent for (A, B, C) {
    const NAME: &'static str = "Composed3";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx) + self.1.compute(state, idx) + self.2.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx)
            + self.1.compute_simd(state, base_idx)
            + self.2.compute_simd(state, base_idx)
    }
}

/// 4-tuple composition.
impl<A: RewardComponent, B: RewardComponent, C: RewardComponent, D: RewardComponent>
    RewardComponent for (A, B, C, D)
{
    const NAME: &'static str = "Composed4";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx)
            + self.1.compute(state, idx)
            + self.2.compute(state, idx)
            + self.3.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx)
            + self.1.compute_simd(state, base_idx)
            + self.2.compute_simd(state, base_idx)
            + self.3.compute_simd(state, base_idx)
    }
}

/// 5-tuple composition.
impl<
        A: RewardComponent,
        B: RewardComponent,
        C: RewardComponent,
        D: RewardComponent,
        E: RewardComponent,
    > RewardComponent for (A, B, C, D, E)
{
    const NAME: &'static str = "Composed5";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx)
            + self.1.compute(state, idx)
            + self.2.compute(state, idx)
            + self.3.compute(state, idx)
            + self.4.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx)
            + self.1.compute_simd(state, base_idx)
            + self.2.compute_simd(state, base_idx)
            + self.3.compute_simd(state, base_idx)
            + self.4.compute_simd(state, base_idx)
    }
}

/// 6-tuple composition.
impl<
        A: RewardComponent,
        B: RewardComponent,
        C: RewardComponent,
        D: RewardComponent,
        E: RewardComponent,
        F: RewardComponent,
    > RewardComponent for (A, B, C, D, E, F)
{
    const NAME: &'static str = "Composed6";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx)
            + self.1.compute(state, idx)
            + self.2.compute(state, idx)
            + self.3.compute(state, idx)
            + self.4.compute(state, idx)
            + self.5.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx)
            + self.1.compute_simd(state, base_idx)
            + self.2.compute_simd(state, base_idx)
            + self.3.compute_simd(state, base_idx)
            + self.4.compute_simd(state, base_idx)
            + self.5.compute_simd(state, base_idx)
    }
}

/// 7-tuple composition (covers standard hover reward).
impl<
        A: RewardComponent,
        B: RewardComponent,
        C: RewardComponent,
        D: RewardComponent,
        E: RewardComponent,
        F: RewardComponent,
        G: RewardComponent,
    > RewardComponent for (A, B, C, D, E, F, G)
{
    const NAME: &'static str = "Composed7";

    #[inline(always)]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        self.0.compute(state, idx)
            + self.1.compute(state, idx)
            + self.2.compute(state, idx)
            + self.3.compute(state, idx)
            + self.4.compute(state, idx)
            + self.5.compute(state, idx)
            + self.6.compute(state, idx)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        self.0.compute_simd(state, base_idx)
            + self.1.compute_simd(state, base_idx)
            + self.2.compute_simd(state, base_idx)
            + self.3.compute_simd(state, base_idx)
            + self.4.compute_simd(state, base_idx)
            + self.5.compute_simd(state, base_idx)
            + self.6.compute_simd(state, base_idx)
    }
}

// ============================================================================
// Reward Computation Functions
// ============================================================================

/// Compute reward for a single environment using any RewardComponent.
#[inline]
pub fn compute_reward<R: RewardComponent>(
    reward_fn: &R,
    state: &QuadcopterState,
    idx: usize,
) -> f32 {
    reward_fn.compute(state, idx)
}

/// Compute rewards for all environments.
pub fn compute_rewards_all<R: RewardComponent>(
    reward_fn: &R,
    state: &QuadcopterState,
    output: &mut [f32],
) {
    for idx in 0..state.num_envs {
        output[idx] = reward_fn.compute(state, idx);
    }
}

/// Compute rewards for all environments using SIMD.
#[cfg(feature = "simd")]
pub fn compute_rewards_all_simd<R: RewardComponent>(
    reward_fn: &R,
    state: &QuadcopterState,
    output: &mut [f32],
) {
    let chunks = state.num_envs / 8;
    let remainder = state.num_envs % 8;

    // Process full SIMD chunks
    for chunk in 0..chunks {
        let base_idx = chunk * 8;
        let rewards = reward_fn.compute_simd(state, base_idx);
        rewards.copy_to_slice(&mut output[base_idx..base_idx + 8]);
    }

    // Handle remainder with scalar
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = base + i;
        output[idx] = reward_fn.compute(state, idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_reward() {
        let state = QuadcopterState::new(1, 0);
        let reward = ().compute(&state, 0);
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn test_tuple_composition_adds() {
        let state = QuadcopterState::new(1, 0);

        let bonus1 = AliveBonus { bonus: 0.1 };
        let bonus2 = AliveBonus { bonus: 0.2 };

        let composed = (bonus1, bonus2);
        let reward = composed.compute(&state, 0);

        assert!((reward - 0.3).abs() < 1e-6, "Expected 0.3, got {}", reward);
    }

    #[test]
    fn test_seven_tuple() {
        let state = QuadcopterState::new(1, 0);
        let reward_fn = presets::hover();
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }
}
