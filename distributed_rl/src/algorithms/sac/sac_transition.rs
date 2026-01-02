//! SAC-specific transition types for replay buffer storage.
//!
//! SAC transitions are simpler than IMPALA transitions because:
//! - No policy version tracking (uniform replay doesn't need staleness correction)
//! - No V-trace, so no behavior log probability needed
//!
//! # Design
//!
//! `SACTransition<D>` is generic over additional data `D`:
//! - `SACTransition<()>` for feed-forward (zero overhead)
//! - `SACTransition<RecurrentData>` for recurrent with hidden state

use crate::core::transition::{Action, Transition};
use std::fmt::Debug;

// ============================================================================
// SAC Transition (Generic)
// ============================================================================

/// SAC transition for uniform replay buffer.
///
/// Stores the minimal information needed for SAC training:
/// - Current state
/// - Action taken
/// - Reward received
/// - Next state
/// - Terminal flag
/// - Additional data (hidden state for recurrent, () for feed-forward)
///
/// # Type Parameters
///
/// - `D`: Additional data type (defaults to `()` for feed-forward)
#[derive(Debug, Clone)]
pub struct SACTransition<D = ()> {
    /// Current state observation
    pub state: Vec<f32>,
    /// Action taken (discrete index or continuous vector)
    pub action: Action,
    /// Reward received
    pub reward: f32,
    /// Next state observation
    pub next_state: Vec<f32>,
    /// Episode terminated (not truncated)
    pub terminal: bool,
    /// Additional data (hidden state, sequence info, etc.)
    pub data: D,
}

// Implementation for default (feed-forward) case
impl SACTransition<()> {
    /// Create a new SAC transition from a base transition.
    pub fn new(base: Transition) -> Self {
        Self {
            state: base.state,
            action: base.action,
            reward: base.reward,
            next_state: base.next_state,
            terminal: base.terminal,
            data: (),
        }
    }

    /// Create a new SAC transition with discrete action.
    pub fn new_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
    ) -> Self {
        Self {
            state,
            action: Action::Discrete(action),
            reward,
            next_state,
            terminal,
            data: (),
        }
    }

    /// Create a new SAC transition with continuous action.
    pub fn new_continuous(
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
    ) -> Self {
        Self {
            state,
            action: Action::Continuous(action),
            reward,
            next_state,
            terminal,
            data: (),
        }
    }
}

// Implementation for any data type
impl<D> SACTransition<D> {
    /// Create a new SAC transition with custom data.
    pub fn with_data(base: Transition, data: D) -> Self {
        Self {
            state: base.state,
            action: base.action,
            reward: base.reward,
            next_state: base.next_state,
            terminal: base.terminal,
            data,
        }
    }

    /// Create with discrete action and custom data.
    pub fn new_discrete_with_data(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        data: D,
    ) -> Self {
        Self {
            state,
            action: Action::Discrete(action),
            reward,
            next_state,
            terminal,
            data,
        }
    }

    /// Create with continuous action and custom data.
    pub fn new_continuous_with_data(
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        data: D,
    ) -> Self {
        Self {
            state,
            action: Action::Continuous(action),
            reward,
            next_state,
            terminal,
            data,
        }
    }

    /// Get the state observation.
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Get the action.
    pub fn action(&self) -> &Action {
        &self.action
    }

    /// Get the reward.
    pub fn reward(&self) -> f32 {
        self.reward
    }

    /// Get the next state observation.
    pub fn next_state(&self) -> &[f32] {
        &self.next_state
    }

    /// Check if terminal.
    pub fn terminal(&self) -> bool {
        self.terminal
    }

    /// Get the additional data.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Get mutable reference to additional data.
    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// State dimension.
    pub fn state_dim(&self) -> usize {
        self.state.len()
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        match &self.action {
            Action::Discrete(_) => 1,
            Action::Continuous(a) => a.len(),
        }
    }

    /// Check if action is discrete.
    pub fn is_discrete(&self) -> bool {
        matches!(self.action, Action::Discrete(_))
    }
}

// ============================================================================
// Hidden Data Marker Trait
// ============================================================================

/// Marker trait for SAC transition data types.
///
/// This trait allows differentiating between feed-forward (no hidden state)
/// and recurrent (with hidden state) transitions.
pub trait SACDataMarker: Clone + Send + Sync + Default + 'static {
    /// Returns true if this data type contains hidden state.
    fn has_hidden_data() -> bool {
        false
    }
}

/// Feed-forward transitions have no hidden state.
impl SACDataMarker for () {}

// ============================================================================
// Recurrent Hidden Data
// ============================================================================

/// Hidden data stored in recurrent SAC transitions.
#[derive(Debug, Clone, Default)]
pub struct SACRecurrentData {
    /// Serialized hidden state at this transition.
    pub hidden: Vec<f32>,
    /// Sequence ID for grouping.
    pub sequence_id: u64,
    /// Step within the sequence.
    pub step_in_sequence: u32,
    /// Whether this is the first step of a sequence.
    pub is_sequence_start: bool,
}

/// Recurrent transitions have hidden state.
impl SACDataMarker for SACRecurrentData {
    fn has_hidden_data() -> bool {
        true
    }
}

/// Type alias for recurrent SAC transition.
pub type SACRecurrentTransition = SACTransition<SACRecurrentData>;

impl SACRecurrentTransition {
    /// Create a new recurrent SAC transition with discrete action.
    pub fn new_recurrent_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        hidden: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> Self {
        Self::new_discrete_with_data(
            state,
            action,
            reward,
            next_state,
            terminal,
            SACRecurrentData {
                hidden,
                sequence_id,
                step_in_sequence,
                is_sequence_start,
            },
        )
    }

    /// Create a new recurrent SAC transition with continuous action.
    pub fn new_recurrent_continuous(
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        hidden: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> Self {
        Self::new_continuous_with_data(
            state,
            action,
            reward,
            next_state,
            terminal,
            SACRecurrentData {
                hidden,
                sequence_id,
                step_in_sequence,
                is_sequence_start,
            },
        )
    }

    /// Get the hidden state.
    pub fn hidden_state(&self) -> &[f32] {
        &self.data.hidden
    }

    /// Get the sequence ID.
    pub fn sequence_id(&self) -> u64 {
        self.data.sequence_id
    }

    /// Get step in sequence.
    pub fn step_in_sequence(&self) -> u32 {
        self.data.step_in_sequence
    }

    /// Check if sequence start.
    pub fn is_sequence_start(&self) -> bool {
        self.data.is_sequence_start
    }
}

// ============================================================================
// Trait for SAC Transitions
// ============================================================================

/// Trait for SAC transitions (unified interface for FF and recurrent).
pub trait SACTransitionTrait: Clone + Send + Sync + 'static {
    /// Get the state observation.
    fn state(&self) -> &[f32];

    /// Get the action.
    fn action(&self) -> &Action;

    /// Get the reward.
    fn reward(&self) -> f32;

    /// Get the next state observation.
    fn next_state(&self) -> &[f32];

    /// Check if the episode terminated.
    fn terminal(&self) -> bool;

    /// Check if this is a recurrent transition with hidden state.
    fn has_hidden(&self) -> bool {
        false
    }

    /// Get the hidden state (empty for feed-forward).
    fn get_hidden_state(&self) -> &[f32] {
        &[]
    }

    /// Get sequence ID (0 for feed-forward).
    fn get_sequence_id(&self) -> u64 {
        0
    }

    /// Check if this is the start of a new sequence.
    fn get_is_sequence_start(&self) -> bool {
        true
    }
}

// Blanket implementation for any SACTransition<D> where D implements SACDataMarker
impl<D: SACDataMarker> SACTransitionTrait for SACTransition<D> {
    fn state(&self) -> &[f32] {
        &self.state
    }

    fn action(&self) -> &Action {
        &self.action
    }

    fn reward(&self) -> f32 {
        self.reward
    }

    fn next_state(&self) -> &[f32] {
        &self.next_state
    }

    fn terminal(&self) -> bool {
        self.terminal
    }

    fn has_hidden(&self) -> bool {
        D::has_hidden_data()
    }
}

// Extended trait for recurrent transitions with hidden state access
pub trait SACRecurrentTransitionTrait: SACTransitionTrait {
    /// Get the hidden state.
    fn get_hidden_state(&self) -> &[f32];

    /// Get sequence ID.
    fn get_sequence_id(&self) -> u64;

    /// Check if this is the start of a new sequence.
    fn get_is_sequence_start(&self) -> bool;
}

// Implementation for recurrent
impl SACRecurrentTransitionTrait for SACRecurrentTransition {
    fn get_hidden_state(&self) -> &[f32] {
        &self.data.hidden
    }

    fn get_sequence_id(&self) -> u64 {
        self.data.sequence_id
    }

    fn get_is_sequence_start(&self) -> bool {
        self.data.is_sequence_start
    }
}

// ============================================================================
// SAC Batch
// ============================================================================

/// Batch of SAC transitions for training.
#[derive(Debug, Clone)]
pub struct SACBatch<D = ()> {
    /// Batch of transitions.
    pub transitions: Vec<SACTransition<D>>,
}

impl<D: Clone + Send + Sync + 'static> SACBatch<D> {
    /// Create a new batch.
    pub fn new(transitions: Vec<SACTransition<D>>) -> Self {
        Self { transitions }
    }

    /// Number of transitions in the batch.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Get all states (flattened).
    pub fn states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.state.iter().copied())
            .collect()
    }

    /// Get all next states (flattened).
    pub fn next_states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.next_state.iter().copied())
            .collect()
    }

    /// Get all rewards.
    pub fn rewards(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.reward).collect()
    }

    /// Get all terminal flags.
    pub fn terminals(&self) -> Vec<bool> {
        self.transitions.iter().map(|t| t.terminal).collect()
    }

    /// Get all actions (flattened for continuous, indices for discrete).
    pub fn actions_floats(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| match &t.action {
                Action::Discrete(a) => vec![*a as f32],
                Action::Continuous(a) => a.clone(),
            })
            .collect()
    }

    /// Get discrete action indices (panics if continuous).
    pub fn discrete_actions(&self) -> Vec<u32> {
        self.transitions
            .iter()
            .map(|t| t.action.as_discrete())
            .collect()
    }

    /// Get continuous actions (panics if discrete).
    pub fn continuous_actions(&self) -> Vec<Vec<f32>> {
        self.transitions
            .iter()
            .map(|t| t.action.as_continuous().to_vec())
            .collect()
    }

    /// State dimension (from first transition).
    pub fn state_dim(&self) -> usize {
        self.transitions
            .first()
            .map(|t| t.state.len())
            .unwrap_or(0)
    }

    /// Action dimension (from first transition).
    pub fn action_dim(&self) -> usize {
        self.transitions
            .first()
            .map(|t| match &t.action {
                Action::Discrete(_) => 1,
                Action::Continuous(a) => a.len(),
            })
            .unwrap_or(0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sac_transition_discrete() {
        let t = SACTransition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
        );
        assert_eq!(t.action.as_discrete(), 1);
        assert!(!t.terminal);
        assert_eq!(t.state_dim(), 2);
        assert!(t.is_discrete());
    }

    #[test]
    fn test_sac_transition_continuous() {
        let t = SACTransition::new_continuous(
            vec![1.0, 2.0],
            vec![0.5, -0.3],
            0.5,
            vec![2.0, 3.0],
            true,
        );
        assert_eq!(t.action.as_continuous(), &[0.5, -0.3]);
        assert!(t.terminal);
        assert_eq!(t.action_dim(), 2);
        assert!(!t.is_discrete());
    }

    #[test]
    fn test_sac_transition_from_transition() {
        let base = Transition::new_discrete(
            vec![1.0, 2.0],
            0,
            1.0,
            vec![2.0, 3.0],
            false,
            false,
        );
        let t = SACTransition::new(base);
        assert_eq!(t.state(), &[1.0, 2.0]);
        assert_eq!(t.reward(), 1.0);
    }

    #[test]
    fn test_sac_recurrent_transition() {
        let t = SACRecurrentTransition::new_recurrent_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            vec![0.1, 0.2, 0.3, 0.4], // hidden state
            42,                        // sequence_id
            5,                         // step_in_sequence
            false,                     // is_sequence_start
        );
        assert_eq!(t.action.as_discrete(), 1);
        assert_eq!(t.hidden_state().len(), 4);
        assert_eq!(t.sequence_id(), 42);
        assert_eq!(t.step_in_sequence(), 5);
        assert!(!t.is_sequence_start());
    }

    #[test]
    fn test_sac_transition_trait() {
        let ff = SACTransition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false);
        let recurrent = SACRecurrentTransition::new_recurrent_discrete(
            vec![1.0], 0, 1.0, vec![2.0], false,
            vec![0.1, 0.2], 0, 0, true,
        );

        // Test trait methods
        assert!(!ff.has_hidden());
        assert!(ff.get_hidden_state().is_empty());

        assert!(recurrent.has_hidden());
        assert_eq!(SACRecurrentTransitionTrait::get_hidden_state(&recurrent).len(), 2);
        assert!(SACRecurrentTransitionTrait::get_is_sequence_start(&recurrent));
    }

    #[test]
    fn test_sac_batch() {
        let transitions = vec![
            SACTransition::new_discrete(vec![1.0, 2.0], 0, 1.0, vec![2.0, 3.0], false),
            SACTransition::new_discrete(vec![3.0, 4.0], 1, 0.5, vec![4.0, 5.0], true),
        ];
        let batch = SACBatch::new(transitions);

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.state_dim(), 2);
        assert_eq!(batch.rewards(), vec![1.0, 0.5]);
        assert_eq!(batch.terminals(), vec![false, true]);
        assert_eq!(batch.discrete_actions(), vec![0, 1]);
    }

    #[test]
    fn test_sac_batch_continuous() {
        let transitions = vec![
            SACTransition::new_continuous(vec![1.0], vec![0.5, -0.3], 1.0, vec![2.0], false),
            SACTransition::new_continuous(vec![3.0], vec![0.1, 0.2], 0.5, vec![4.0], true),
        ];
        let batch = SACBatch::new(transitions);

        assert_eq!(batch.action_dim(), 2);
        let actions = batch.continuous_actions();
        assert_eq!(actions[0], vec![0.5, -0.3]);
        assert_eq!(actions[1], vec![0.1, 0.2]);
    }

    #[test]
    fn test_sac_transition_with_data() {
        #[derive(Debug, Clone)]
        struct CustomData {
            value: i32,
        }

        let base = Transition::new_discrete(
            vec![1.0],
            0,
            1.0,
            vec![2.0],
            false,
            false,
        );
        let t = SACTransition::with_data(base, CustomData { value: 42 });
        assert_eq!(t.data().value, 42);
    }
}
