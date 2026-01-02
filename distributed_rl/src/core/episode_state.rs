//! Type-safe episode state classification.
//!
//! This module provides a type-safe representation of episode termination modes,
//! ensuring correct handling of the terminal/truncated distinction throughout the codebase.
//!
//! # Key Distinction
//!
//! - **Terminal**: Episode truly ended (agent died, goal reached, absorbing state)
//!   - Hidden state: RESET (new episode, fresh memory)
//!   - Bootstrap value: 0.0 (no future rewards possible)
//!   - Sequence ID: INCREMENT (new sequence begins)
//!
//! - **Truncated**: Episode hit external limit (time limit, step limit)
//!   - Hidden state: PRESERVE (same episode semantically continues)
//!   - Bootstrap value: V(s', h') (estimate remaining value)
//!   - Sequence ID: UNCHANGED (same sequence continues)
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::core::EpisodeState;
//!
//! let state = EpisodeState::from_flags(terminal, truncated);
//! if state.should_reset_hidden() {
//!     hidden.reset(env_idx);
//! }
//! ```

/// Episode state classification for correct terminal/truncated handling.
///
/// This enum encodes the three possible states of an episode:
/// - Running: Episode is ongoing
/// - Terminal: Episode ended due to absorbing state (reset hidden)
/// - Truncated: Episode ended due to external limit (preserve hidden)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpisodeState {
    /// Episode is ongoing, no termination.
    Running,
    /// Episode terminated due to reaching an absorbing state.
    /// Hidden state should be reset, bootstrap value should be 0.
    Terminal,
    /// Episode truncated due to external limit (time, steps).
    /// Hidden state should be preserved, bootstrap value should be V(s').
    Truncated,
}

impl EpisodeState {
    /// Create episode state from terminal and truncated flags.
    ///
    /// # Arguments
    ///
    /// * `terminal` - True if episode reached absorbing state
    /// * `truncated` - True if episode hit external limit
    ///
    /// # Returns
    ///
    /// The appropriate episode state. If both are true, Terminal takes precedence.
    #[inline]
    pub fn from_flags(terminal: bool, truncated: bool) -> Self {
        if terminal {
            Self::Terminal
        } else if truncated {
            Self::Truncated
        } else {
            Self::Running
        }
    }

    /// Whether hidden state should be reset for this episode state.
    ///
    /// Returns `true` only for `Terminal` - truncated episodes preserve hidden state
    /// because the episode semantically continues.
    #[inline]
    pub fn should_reset_hidden(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Whether a new sequence should start for this episode state.
    ///
    /// Returns `true` only for `Terminal` - truncated episodes remain in the same
    /// sequence for TBPTT purposes.
    #[inline]
    pub fn should_new_sequence(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Whether bootstrap value is needed for this episode state.
    ///
    /// Returns `true` for `Truncated` - terminal episodes don't need bootstrap
    /// because there are no future rewards.
    #[inline]
    pub fn needs_bootstrap(&self) -> bool {
        matches!(self, Self::Truncated)
    }

    /// Whether the episode is done (either terminal or truncated).
    #[inline]
    pub fn is_done(&self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Whether this is a terminal state.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Whether this is a truncated state.
    #[inline]
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Truncated)
    }

    /// Whether this is a running (ongoing) state.
    #[inline]
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }
}

impl Default for EpisodeState {
    fn default() -> Self {
        Self::Running
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_flags_terminal() {
        let state = EpisodeState::from_flags(true, false);
        assert_eq!(state, EpisodeState::Terminal);
        assert!(state.should_reset_hidden());
        assert!(state.should_new_sequence());
        assert!(!state.needs_bootstrap());
        assert!(state.is_done());
    }

    #[test]
    fn test_from_flags_truncated() {
        let state = EpisodeState::from_flags(false, true);
        assert_eq!(state, EpisodeState::Truncated);
        assert!(!state.should_reset_hidden()); // CRITICAL: Don't reset hidden!
        assert!(!state.should_new_sequence()); // CRITICAL: Same sequence!
        assert!(state.needs_bootstrap());
        assert!(state.is_done());
    }

    #[test]
    fn test_from_flags_running() {
        let state = EpisodeState::from_flags(false, false);
        assert_eq!(state, EpisodeState::Running);
        assert!(!state.should_reset_hidden());
        assert!(!state.should_new_sequence());
        assert!(!state.needs_bootstrap());
        assert!(!state.is_done());
    }

    #[test]
    fn test_from_flags_both_true() {
        // If both are true, Terminal takes precedence
        let state = EpisodeState::from_flags(true, true);
        assert_eq!(state, EpisodeState::Terminal);
    }

    #[test]
    fn test_default() {
        assert_eq!(EpisodeState::default(), EpisodeState::Running);
    }

    #[test]
    fn test_invariant_hidden_reset_only_on_terminal() {
        // This test encodes the critical invariant: INV-H1
        // Hidden state reset should occur if and only if terminal is true
        let test_cases = [
            (false, false, false), // Running: no reset
            (true, false, true),   // Terminal: reset
            (false, true, false),  // Truncated: NO reset (critical!)
            (true, true, true),    // Both: Terminal takes precedence, reset
        ];

        for (terminal, truncated, should_reset) in test_cases {
            let state = EpisodeState::from_flags(terminal, truncated);
            assert_eq!(
                state.should_reset_hidden(),
                should_reset,
                "Failed for terminal={}, truncated={}: expected reset={}",
                terminal,
                truncated,
                should_reset
            );
        }
    }

    #[test]
    fn test_invariant_sequence_only_on_terminal() {
        // This test encodes the critical invariant: INV-S1
        // Sequence ID should increment if and only if terminal is true
        let test_cases = [
            (false, false, false), // Running: same sequence
            (true, false, true),   // Terminal: new sequence
            (false, true, false),  // Truncated: SAME sequence (critical!)
            (true, true, true),    // Both: Terminal takes precedence, new sequence
        ];

        for (terminal, truncated, should_new_seq) in test_cases {
            let state = EpisodeState::from_flags(terminal, truncated);
            assert_eq!(
                state.should_new_sequence(),
                should_new_seq,
                "Failed for terminal={}, truncated={}: expected new_seq={}",
                terminal,
                truncated,
                should_new_seq
            );
        }
    }
}
