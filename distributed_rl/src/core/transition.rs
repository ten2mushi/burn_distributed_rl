//! Generic transition types for distributed RL algorithms.
//!
//! This module provides transition types optimized for different algorithms:
//! - `Transition`: Base type for all algorithms
//! - `PPOTransition`: Includes log_prob and value for on-policy learning
//! - `IMPALATransition`: Includes behavior policy info for V-trace correction
//! - `Trajectory`: Sequence of transitions from one environment

use std::fmt::Debug;

/// Action representation (discrete or continuous).
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Discrete action index
    Discrete(u32),
    /// Continuous action vector
    Continuous(Vec<f32>),
}

impl Action {
    /// Get discrete action index, panics if continuous.
    pub fn as_discrete(&self) -> u32 {
        match self {
            Action::Discrete(a) => *a,
            Action::Continuous(_) => panic!("Expected discrete action"),
        }
    }

    /// Get continuous action vector, panics if discrete.
    pub fn as_continuous(&self) -> &[f32] {
        match self {
            Action::Discrete(_) => panic!("Expected continuous action"),
            Action::Continuous(a) => a,
        }
    }
}

/// Base transition for all algorithms.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current state observation
    pub state: Vec<f32>,
    /// Action taken
    pub action: Action,
    /// Reward received
    pub reward: f32,
    /// Next state observation
    pub next_state: Vec<f32>,
    /// Episode terminated (goal reached, failure, etc.)
    pub terminal: bool,
    /// Episode truncated (time limit, etc.)
    pub truncated: bool,
}

impl Transition {
    /// Create a new transition with discrete action.
    pub fn new_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
    ) -> Self {
        Self {
            state,
            action: Action::Discrete(action),
            reward,
            next_state,
            terminal,
            truncated,
        }
    }

    /// Create a new transition with continuous action.
    pub fn new_continuous(
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
    ) -> Self {
        Self {
            state,
            action: Action::Continuous(action),
            reward,
            next_state,
            terminal,
            truncated,
        }
    }

    /// Check if episode ended (terminal or truncated).
    pub fn done(&self) -> bool {
        self.terminal || self.truncated
    }
}

/// PPO transition with policy information for on-policy learning.
///
/// Stores log probability and value estimate at collection time,
/// needed for computing the PPO clipped surrogate objective.
#[derive(Debug, Clone)]
pub struct PPOTransition {
    /// Base transition data
    pub base: Transition,
    /// Log probability of action under behavior policy: log π(a|s)
    pub log_prob: f32,
    /// Value estimate at state: V(s)
    pub value: f32,
}

impl PPOTransition {
    /// Create a new PPO transition with discrete action.
    pub fn new_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
        log_prob: f32,
        value: f32,
    ) -> Self {
        Self {
            base: Transition::new_discrete(state, action, reward, next_state, terminal, truncated),
            log_prob,
            value,
        }
    }

    /// Check if episode ended.
    pub fn done(&self) -> bool {
        self.base.done()
    }
}

/// Recurrent PPO transition with hidden state for sequence-based training.
///
/// Extends PPOTransition with recurrent state information needed for
/// training recurrent policies using TBPTT (Truncated Backpropagation Through Time).
#[derive(Debug, Clone)]
pub struct RecurrentPPOTransition {
    /// Base PPO transition data (state, action, reward, log_prob, value)
    pub base: PPOTransition,
    /// Hidden state at this timestep (flattened, includes cell state for LSTM)
    pub hidden_state: Vec<f32>,
    /// Unique identifier for this episode/sequence
    pub sequence_id: u64,
    /// Position within the sequence (0-indexed)
    pub step_in_sequence: usize,
    /// Whether this is the first step after a reset (hidden should be zero)
    pub is_sequence_start: bool,
    /// Bootstrap value V(s_T) if this is a truncated rollout boundary.
    pub bootstrap_value: Option<f32>,
}

impl RecurrentPPOTransition {
    /// Create a new recurrent PPO transition with discrete action.
    pub fn new_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
        log_prob: f32,
        value: f32,
        hidden_state: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: usize,
        is_sequence_start: bool,
        bootstrap_value: Option<f32>,
    ) -> Self {
        Self {
            base: PPOTransition::new_discrete(
                state,
                action,
                reward,
                next_state,
                terminal,
                truncated,
                log_prob,
                value,
            ),
            hidden_state,
            sequence_id,
            step_in_sequence,
            is_sequence_start,
            bootstrap_value,
        }
    }

    /// Create from an existing PPO transition plus recurrent info.
    pub fn from_ppo(
        ppo_transition: PPOTransition,
        hidden_state: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: usize,
        is_sequence_start: bool,
    ) -> Self {
        Self {
            base: ppo_transition,
            hidden_state,
            sequence_id,
            step_in_sequence,
            is_sequence_start,
            bootstrap_value: None,
        }
    }

    /// Check if episode ended (terminal or truncated).
    pub fn done(&self) -> bool {
        self.base.done()
    }

    /// Check if episode truly terminated (not just truncated).
    pub fn terminal(&self) -> bool {
        self.base.base.terminal
    }

    /// Check if episode was truncated (e.g., time limit).
    pub fn truncated(&self) -> bool {
        self.base.base.truncated
    }

    /// Get the state observation.
    pub fn state(&self) -> &[f32] {
        &self.base.base.state
    }

    /// Get the action.
    pub fn action(&self) -> &Action {
        &self.base.base.action
    }

    /// Get the reward.
    pub fn reward(&self) -> f32 {
        self.base.base.reward
    }

    /// Get the log probability.
    pub fn log_prob(&self) -> f32 {
        self.base.log_prob
    }

    /// Get the value estimate.
    pub fn value(&self) -> f32 {
        self.base.value
    }
}

/// IMPALA transition with behavior policy info for V-trace correction.
///
/// Stores the behavior policy's log probability and model version,
/// enabling off-policy correction via importance sampling.
#[derive(Debug, Clone)]
pub struct IMPALATransition {
    /// Base transition data
    pub base: Transition,
    /// Log probability under behavior policy (at collection time)
    pub behavior_log_prob: f32,
    /// Policy version that generated this transition
    pub policy_version: u64,
}

impl IMPALATransition {
    /// Create a new IMPALA transition with discrete action.
    pub fn new_discrete(
        state: Vec<f32>,
        action: u32,
        reward: f32,
        next_state: Vec<f32>,
        terminal: bool,
        truncated: bool,
        behavior_log_prob: f32,
        policy_version: u64,
    ) -> Self {
        Self {
            base: Transition::new_discrete(state, action, reward, next_state, terminal, truncated),
            behavior_log_prob,
            policy_version,
        }
    }

    /// Check if episode ended.
    pub fn done(&self) -> bool {
        self.base.done()
    }
}

/// Trajectory: sequence of transitions from one environment.
///
/// Used for algorithms that need episode structure (IMPALA, some PPO variants).
#[derive(Debug, Clone)]
pub struct Trajectory<T> {
    /// Ordered sequence of transitions
    pub transitions: Vec<T>,
    /// Environment ID that generated this trajectory
    pub env_id: usize,
    /// Total undiscounted return (if episode is complete)
    pub episode_return: Option<f32>,
}

impl<T> Trajectory<T> {
    /// Create a new empty trajectory.
    pub fn new(env_id: usize) -> Self {
        Self {
            transitions: Vec::new(),
            env_id,
            episode_return: None,
        }
    }

    /// Create a trajectory with pre-allocated capacity.
    pub fn with_capacity(env_id: usize, capacity: usize) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            env_id,
            episode_return: None,
        }
    }

    /// Add a transition to the trajectory.
    pub fn push(&mut self, transition: T) {
        self.transitions.push(transition);
    }

    /// Get the number of transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Clear all transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.episode_return = None;
    }

    /// Iterate over transitions.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.transitions.iter()
    }
}

impl<T> Default for Trajectory<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_discrete() {
        let t = Transition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            false,
        );
        assert_eq!(t.action.as_discrete(), 1);
        assert!(!t.done());
    }

    #[test]
    fn test_transition_terminal() {
        let t = Transition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            true,
            false,
        );
        assert!(t.done());
    }

    #[test]
    fn test_transition_truncated() {
        let t = Transition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            true,
        );
        assert!(t.done());
    }

    #[test]
    fn test_ppo_transition() {
        let t = PPOTransition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            false,
            -0.693,  // log(0.5)
            1.0,
        );
        assert_eq!(t.base.action.as_discrete(), 1);
        assert!((t.log_prob - (-0.693)).abs() < 0.001);
        assert_eq!(t.value, 1.0);
    }

    #[test]
    fn test_impala_transition() {
        let t = IMPALATransition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            false,
            -0.693,
            42,
        );
        assert_eq!(t.policy_version, 42);
    }

    #[test]
    fn test_trajectory() {
        let mut traj = Trajectory::<PPOTransition>::new(0);
        assert!(traj.is_empty());

        traj.push(PPOTransition::new_discrete(
            vec![1.0], 0, 1.0, vec![2.0], false, false, -0.5, 0.5,
        ));
        traj.push(PPOTransition::new_discrete(
            vec![2.0], 1, 1.0, vec![3.0], true, false, -0.3, 0.8,
        ));

        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());

        traj.clear();
        assert!(traj.is_empty());
    }

    #[test]
    fn test_recurrent_ppo_transition() {
        let t = RecurrentPPOTransition::new_discrete(
            vec![1.0, 2.0],
            1,
            0.5,
            vec![2.0, 3.0],
            false,
            false,
            -0.693,
            1.0,
            vec![0.1, 0.2, 0.3, 0.4], // hidden state
            42,                        // sequence_id
            5,                         // step_in_sequence
            false,                     // is_sequence_start
            None,                      // bootstrap_value
        );
        assert_eq!(t.base.base.action.as_discrete(), 1);
        assert!((t.log_prob() - (-0.693)).abs() < 0.001);
        assert_eq!(t.value(), 1.0);
        assert_eq!(t.hidden_state.len(), 4);
        assert_eq!(t.sequence_id, 42);
        assert_eq!(t.step_in_sequence, 5);
        assert!(!t.is_sequence_start);
    }

    #[test]
    fn test_recurrent_ppo_transition_from_ppo() {
        let ppo = PPOTransition::new_discrete(
            vec![1.0], 0, 1.0, vec![2.0], false, false, -0.5, 0.5,
        );
        let recurrent = RecurrentPPOTransition::from_ppo(
            ppo,
            vec![0.0, 0.0], // hidden state
            0,              // sequence_id
            0,              // step_in_sequence
            true,           // is_sequence_start
        );
        assert!(recurrent.is_sequence_start);
        assert_eq!(recurrent.step_in_sequence, 0);
    }
}
