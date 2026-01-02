//! PPO transition type for both feed-forward and recurrent policies.
//!
//! This module provides a generic transition type that supports both stateless
//! (feed-forward) and stateful (recurrent) PPO training with zero-cost abstraction
//! for the feed-forward case.
//!
//! # Design
//!
//! The transition is generic over `H: HiddenData`, where:
//! - `H = ()` for feed-forward (zero-sized, no runtime overhead)
//! - `H = RecurrentHiddenData` for recurrent (stores serialized hidden state)

use crate::core::transition::{Action, Transition};

// ============================================================================
// HiddenData Trait
// ============================================================================

/// Trait for hidden data storage in transitions.
///
/// This trait enables zero-cost abstraction: `()` stores nothing (feed-forward),
/// while `RecurrentHiddenData` stores the serialized hidden state.
pub trait HiddenData: Clone + Send + Default + 'static {
    /// Whether this hidden data type is empty (feed-forward).
    fn is_empty() -> bool;

    /// Get the hidden state data as a slice (empty for feed-forward).
    fn as_slice(&self) -> &[f32];

    /// Create from a slice of hidden state data.
    fn from_slice(data: &[f32]) -> Self;
}

/// Feed-forward hidden data - unit type with no storage.
impl HiddenData for () {
    fn is_empty() -> bool {
        true
    }

    fn as_slice(&self) -> &[f32] {
        &[]
    }

    fn from_slice(_data: &[f32]) -> Self {}
}

// ============================================================================
// RecurrentHiddenData
// ============================================================================

/// Hidden state data storage for recurrent policies.
///
/// Stores the serialized hidden state (and cell state for LSTM) as a flat vector.
///
/// # Temporal Causality Invariant (Curry-Howard Correspondence)
///
/// The hidden state stored in a transition MUST be the **INPUT** hidden state
/// (h_{t-1}) that was used to produce the log_prob and value in this transition.
///
/// ## Type-Theoretic Contract
///
/// ```text
/// Transition_t = {
///     state:    o_t,
///     log_prob: log π(a_t | o_t, h_{t-1}),  // Computed with h_{t-1}
///     value:    V(o_t, h_{t-1}),            // Computed with h_{t-1}
///     hidden:   h_{t-1}                      // MUST store h_{t-1}, NOT h_t!
/// }
/// ```
///
/// ## Why This Matters
///
/// During training, we initialize the LSTM with `transition[0].hidden` and replay
/// the sequence. If we store h_t (output) instead of h_{t-1} (input):
///
/// - Collection computed: `log π(a_0 | o_0, h_{-1}=zeros)`
/// - Training computes:   `log π(a_0 | o_0, h_0)` ← WRONG conditioning!
///
/// The PPO ratio `π_new(a|s,h_t) / π_old(a|s,h_{t-1})` becomes meaningless because
/// numerator and denominator condition on different histories.
///
/// ## Enforced By
///
/// The actor must extract hidden states BEFORE the forward pass, not after.
/// See `ppo_runner.rs` for the correct implementation pattern.
#[derive(Debug, Clone, Default)]
pub struct RecurrentHiddenData {
    /// Flattened hidden state data.
    ///
    /// **Invariant**: This is h_{t-1} (the INPUT to the forward pass that
    /// produced this transition), NOT h_t (the OUTPUT).
    pub data: Vec<f32>,
}

impl RecurrentHiddenData {
    /// Create from a vector of hidden state data.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl HiddenData for RecurrentHiddenData {
    fn is_empty() -> bool {
        false
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

// ============================================================================
// PPOTransition
// ============================================================================

/// Unified PPO transition generic over hidden state representation.
///
/// # Type Parameters
///
/// - `H`: Hidden data type (`()` for feed-forward, `RecurrentHiddenData` for recurrent)
///
/// # Zero-Cost Abstraction
///
/// For `H = ()`:
/// - `hidden_data` is zero-sized (no memory overhead)
/// - `sequence_id`, `step_in_sequence`, `is_sequence_start` have negligible impact
///
/// # Fields
///
/// - Core PPO fields: state, action, reward, log_prob, value
/// - Bootstrap value for rollout boundary handling
/// - Recurrent fields: hidden_data, sequence tracking
#[derive(Debug, Clone)]
pub struct PPOTransition<H: HiddenData> {
    /// Base transition data (state, action, reward, next_state, terminal, truncated)
    pub base: Transition,
    /// Log probability of action under behavior policy: log π(a|s)
    pub log_prob: f32,
    /// Value estimate at state: V(s)
    pub value: f32,
    /// Bootstrap value V(s_{t+1}) for truncated rollouts at boundaries.
    /// Some(v) if this transition is at a rollout boundary and the episode
    /// didn't truly terminate (truncated or ongoing).
    /// None for mid-rollout transitions or true terminal states.
    pub bootstrap_value: Option<f32>,
    /// Hidden state data (zero-sized for feed-forward, Vec<f32> for recurrent).
    ///
    /// **Temporal Invariant**: For recurrent policies, this MUST be h_{t-1}
    /// (the INPUT hidden state), not h_t (the OUTPUT). See `RecurrentHiddenData`
    /// for the full type-theoretic specification.
    pub hidden_data: H,
    /// Unique identifier for this episode/sequence (0 for feed-forward)
    pub sequence_id: u64,
    /// Position within the sequence (0-indexed, 0 for feed-forward)
    pub step_in_sequence: u32,
    /// Whether this is the first step after a reset (false for feed-forward)
    pub is_sequence_start: bool,
}

impl<H: HiddenData> PPOTransition<H> {
    /// Create a new unified transition.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
        hidden_data: H,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> Self {
        Self {
            base,
            log_prob,
            value,
            bootstrap_value,
            hidden_data,
            sequence_id,
            step_in_sequence,
            is_sequence_start,
        }
    }

    /// Get the state observation.
    pub fn state(&self) -> &[f32] {
        &self.base.state
    }

    /// Get the action.
    pub fn action(&self) -> &Action {
        &self.base.action
    }

    /// Get the reward.
    pub fn reward(&self) -> f32 {
        self.base.reward
    }

    /// Check if episode ended (terminal or truncated).
    pub fn done(&self) -> bool {
        self.base.done()
    }

    /// Check if episode truly terminated (not just truncated).
    pub fn terminal(&self) -> bool {
        self.base.terminal
    }

    /// Check if episode was truncated (e.g., time limit).
    pub fn truncated(&self) -> bool {
        self.base.truncated
    }
}

// ============================================================================
// Feed-Forward Transition Helpers
// ============================================================================

impl PPOTransition<()> {
    /// Create a feed-forward transition (no hidden state).
    pub fn feed_forward(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
    ) -> Self {
        Self {
            base,
            log_prob,
            value,
            bootstrap_value,
            hidden_data: (),
            sequence_id: 0,
            step_in_sequence: 0,
            is_sequence_start: false,
        }
    }

    /// Create a feed-forward transition with discrete action.
    pub fn feed_forward_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
        log_prob: f32,
        value: f32,
    ) -> Self {
        Self::feed_forward(
            Transition::new_discrete(state, action, reward, next_state, terminal, truncated),
            log_prob,
            value,
            None,
        )
    }
}

// ============================================================================
// Recurrent Transition Helpers
// ============================================================================

impl PPOTransition<RecurrentHiddenData> {
    /// Create a recurrent transition with hidden state.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
        hidden_state: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> Self {
        Self {
            base,
            log_prob,
            value,
            bootstrap_value,
            hidden_data: RecurrentHiddenData::new(hidden_state),
            sequence_id,
            step_in_sequence,
            is_sequence_start,
        }
    }

    /// Get the hidden state data.
    pub fn hidden_state(&self) -> &[f32] {
        &self.hidden_data.data
    }
}

// ============================================================================
// PPOTransitionTrait for Buffer Compatibility
// ============================================================================

/// Trait for PPO transitions enabling generic buffer handling.
///
/// This trait provides a common interface for accessing transition data,
/// allowing the unified buffer to work with any transition type.
pub trait PPOTransitionTrait: Clone + Send + 'static {
    /// Get the state observation.
    fn state(&self) -> &[f32];

    /// Get the action.
    fn action(&self) -> &Action;

    /// Get the reward.
    fn reward(&self) -> f32;

    /// Get the log probability.
    fn log_prob(&self) -> f32;

    /// Get the value estimate.
    fn value(&self) -> f32;

    /// Check if episode ended (terminal or truncated).
    fn done(&self) -> bool;

    /// Check if episode truly terminated.
    fn terminal(&self) -> bool;

    /// Check if episode was truncated.
    fn truncated(&self) -> bool;

    /// Get bootstrap value if at rollout boundary.
    fn bootstrap_value(&self) -> Option<f32>;
}

impl<H: HiddenData> PPOTransitionTrait for PPOTransition<H> {
    fn state(&self) -> &[f32] {
        &self.base.state
    }

    fn action(&self) -> &Action {
        &self.base.action
    }

    fn reward(&self) -> f32 {
        self.base.reward
    }

    fn log_prob(&self) -> f32 {
        self.log_prob
    }

    fn value(&self) -> f32 {
        self.value
    }

    fn done(&self) -> bool {
        self.base.done()
    }

    fn terminal(&self) -> bool {
        self.base.terminal
    }

    fn truncated(&self) -> bool {
        self.base.truncated
    }

    fn bootstrap_value(&self) -> Option<f32> {
        self.bootstrap_value
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Feed-forward PPO transition (no hidden state storage).
pub type PPOTransitionFF = PPOTransition<()>;

/// Recurrent PPO transition (with hidden state storage).
pub type PPOTransitionRecurrent = PPOTransition<RecurrentHiddenData>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_transition() {
        let t = PPOTransitionFF::feed_forward_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            false,
            -0.693,
            1.0,
        );

        assert_eq!(t.state(), &[1.0, 2.0]);
        assert_eq!(t.action().as_discrete(), 1);
        assert_eq!(t.reward(), 0.5);
        assert!((t.log_prob() - (-0.693)).abs() < 0.001);
        assert_eq!(t.value(), 1.0);
        assert!(!t.done());
        assert!(!t.terminal());
        assert!(!t.truncated());
        assert!(t.bootstrap_value().is_none());

        // Feed-forward should have zero-cost hidden data
        assert!(<() as HiddenData>::is_empty());
        assert_eq!(t.hidden_data.as_slice().len(), 0);
    }

    #[test]
    fn test_recurrent_transition() {
        let t = PPOTransitionRecurrent::recurrent(
            Transition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false, false),
            -0.5,
            0.8,
            Some(0.5),
            vec![0.1, 0.2, 0.3, 0.4],
            42,
            5,
            false,
        );

        assert_eq!(t.state(), &[1.0]);
        assert_eq!(t.hidden_state(), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(t.sequence_id, 42);
        assert_eq!(t.step_in_sequence, 5);
        assert!(!t.is_sequence_start);
        assert_eq!(t.bootstrap_value(), Some(0.5));

        // Recurrent should have non-empty hidden data
        assert!(!<RecurrentHiddenData as HiddenData>::is_empty());
    }

    #[test]
    fn test_terminal_vs_truncated() {
        // Terminal transition
        let t1 = PPOTransitionFF::feed_forward(
            Transition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], true, false),
            -0.5,
            1.0,
            None,
        );
        assert!(t1.done());
        assert!(t1.terminal());
        assert!(!t1.truncated());

        // Truncated transition
        let t2 = PPOTransitionFF::feed_forward(
            Transition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false, true),
            -0.5,
            1.0,
            Some(0.9),
        );
        assert!(t2.done());
        assert!(!t2.terminal());
        assert!(t2.truncated());
        assert_eq!(t2.bootstrap_value(), Some(0.9));
    }

    #[test]
    fn test_hidden_data_trait() {
        // Unit type
        let empty: () = <() as HiddenData>::from_slice(&[1.0, 2.0]);
        assert!(<() as HiddenData>::is_empty());
        assert_eq!(empty.as_slice().len(), 0);

        // RecurrentHiddenData
        let data = RecurrentHiddenData::from_slice(&[0.1, 0.2, 0.3]);
        assert!(!<RecurrentHiddenData as HiddenData>::is_empty());
        assert_eq!(data.as_slice(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_ppo_transition_trait() {
        // Test that both types implement the trait
        fn assert_trait<T: PPOTransitionTrait>(_t: &T) {}

        let ff = PPOTransitionFF::feed_forward_discrete(
            vec![1.0],
            0,
            1.0,
            vec![2.0],
            false,
            false,
            -0.5,
            0.5,
        );
        assert_trait(&ff);

        let rec = PPOTransitionRecurrent::recurrent(
            Transition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false, false),
            -0.5,
            0.5,
            None,
            vec![0.1, 0.2],
            0,
            0,
            true,
        );
        assert_trait(&rec);
    }
}
