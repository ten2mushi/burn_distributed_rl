//! Sequence buffer for recurrent PPO training.
//!
//! Manages episode-structured transitions for TBPTT (Truncated Backpropagation
//! Through Time). Collects transitions with hidden states and chunks them into
//! fixed-length sequences for efficient batch training.

use std::collections::VecDeque;

use crate::core::transition::RecurrentPPOTransition;

/// Configuration for the sequence buffer.
#[derive(Debug, Clone)]
pub struct SequenceBufferConfig {
    /// Number of parallel environments.
    pub n_envs: usize,
    /// TBPTT sequence length (chunk size for training).
    pub sequence_length: usize,
    /// Burn-in length (steps to run without gradient for hidden warm-up).
    pub burn_in_length: usize,
    /// Maximum sequences to store before oldest are discarded.
    pub max_sequences: usize,
}

impl Default for SequenceBufferConfig {
    fn default() -> Self {
        Self {
            n_envs: 64,
            sequence_length: 16,
            burn_in_length: 0,
            max_sequences: 256,
        }
    }
}

impl SequenceBufferConfig {
    /// Create a new sequence buffer config.
    pub fn new(n_envs: usize, sequence_length: usize) -> Self {
        Self {
            n_envs,
            sequence_length,
            ..Default::default()
        }
    }

    /// Set burn-in length for hidden state warm-up.
    pub fn with_burn_in(mut self, burn_in_length: usize) -> Self {
        self.burn_in_length = burn_in_length;
        self
    }

    /// Set maximum stored sequences.
    pub fn with_max_sequences(mut self, max_sequences: usize) -> Self {
        self.max_sequences = max_sequences;
        self
    }
}

/// A fixed-length sequence of transitions for TBPTT training.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Transitions in this sequence (exactly sequence_length items).
    pub transitions: Vec<RecurrentPPOTransition>,
    /// Initial hidden state for this sequence (flattened).
    pub initial_hidden: Vec<f32>,
    /// Environment ID that generated this sequence.
    pub env_id: usize,
    /// Sequence ID (for debugging/tracking).
    pub sequence_id: u64,
    /// Mask indicating which timesteps are valid (for padded sequences).
    pub valid_mask: Vec<bool>,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(
        transitions: Vec<RecurrentPPOTransition>,
        initial_hidden: Vec<f32>,
        env_id: usize,
        sequence_id: u64,
    ) -> Self {
        let valid_mask = vec![true; transitions.len()];
        Self {
            transitions,
            initial_hidden,
            env_id,
            sequence_id,
            valid_mask,
        }
    }

    /// Create a padded sequence (for incomplete episodes at end of rollout).
    pub fn new_padded(
        transitions: Vec<RecurrentPPOTransition>,
        initial_hidden: Vec<f32>,
        env_id: usize,
        sequence_id: u64,
        target_length: usize,
    ) -> Self {
        let n_valid = transitions.len();
        let mut valid_mask = vec![true; n_valid];
        valid_mask.resize(target_length, false);

        // Pad transitions with clones of last transition (will be masked out)
        let mut padded_transitions = transitions;
        while padded_transitions.len() < target_length {
            if let Some(last) = padded_transitions.last() {
                padded_transitions.push(last.clone());
            } else {
                break;
            }
        }

        Self {
            transitions: padded_transitions,
            initial_hidden,
            env_id,
            sequence_id,
            valid_mask,
        }
    }

    /// Get the number of transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Get number of valid (non-padded) timesteps.
    pub fn n_valid(&self) -> usize {
        self.valid_mask.iter().filter(|&&v| v).count()
    }
}

/// Buffer for managing recurrent PPO training data.
///
/// Collects transitions from multiple environments and chunks them into
/// fixed-length sequences suitable for TBPTT training.
pub struct SequenceBuffer {
    config: SequenceBufferConfig,
    /// In-progress episodes for each environment.
    episodes: Vec<Vec<RecurrentPPOTransition>>,
    /// Completed sequences ready for training.
    sequences: VecDeque<Sequence>,
    /// Counter for generating sequence IDs.
    sequence_counter: u64,
    /// Initial hidden states for each in-progress episode.
    episode_initial_hidden: Vec<Vec<f32>>,
}

impl SequenceBuffer {
    /// Create a new sequence buffer.
    pub fn new(config: SequenceBufferConfig) -> Self {
        let episodes = vec![Vec::new(); config.n_envs];
        let episode_initial_hidden = vec![Vec::new(); config.n_envs];

        Self {
            config,
            episodes,
            sequences: VecDeque::new(),
            sequence_counter: 0,
            episode_initial_hidden,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SequenceBufferConfig {
        &self.config
    }

    /// Number of completed sequences available.
    pub fn n_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Check if buffer has enough sequences for a minibatch.
    pub fn has_minibatch(&self, minibatch_size: usize) -> bool {
        self.sequences.len() >= minibatch_size
    }

    /// Set the initial hidden state for an environment's current episode.
    pub fn set_initial_hidden(&mut self, env_id: usize, hidden: Vec<f32>) {
        if env_id < self.episode_initial_hidden.len() {
            self.episode_initial_hidden[env_id] = hidden;
        }
    }

    /// Add a transition to the appropriate environment's episode.
    pub fn push(&mut self, env_id: usize, transition: RecurrentPPOTransition) {
        if env_id >= self.episodes.len() {
            return;
        }

        let is_start = transition.is_sequence_start;
        let is_done = transition.done();

        // If this is a sequence start (new episode), capture the initial hidden state.
        // This OVERRIDES any carried-over hidden from a previous episode's last chunk,
        // since is_sequence_start=true signals a fresh episode has begun.
        if is_start {
            self.episode_initial_hidden[env_id] = transition.hidden_state.clone();
        }

        self.episodes[env_id].push(transition);

        // Check if we should create a sequence (episode done or reached chunk length)
        let episode_len = self.episodes[env_id].len();
        let chunk_len = self.config.sequence_length;

        if is_done || episode_len >= chunk_len {
            self.finalize_episode_chunk(env_id, is_done);
        }
    }

    /// Finalize a chunk from an episode, creating a sequence.
    fn finalize_episode_chunk(&mut self, env_id: usize, episode_done: bool) {
        let episode = &mut self.episodes[env_id];
        if episode.is_empty() {
            return;
        }

        let chunk_len = self.config.sequence_length;
        let initial_hidden = std::mem::take(&mut self.episode_initial_hidden[env_id]);

        if episode.len() >= chunk_len {
            // Take a full chunk
            let chunk: Vec<_> = episode.drain(..chunk_len).collect();

            // Carry over hidden state for next chunk if episode is continuing (not done).
            // This is critical for recurrent training: the hidden state from the last
            // transition of this chunk becomes the initial hidden for the next chunk,
            // even if the episode is temporarily empty (more transitions will come).
            if !episode_done {
                self.episode_initial_hidden[env_id] = chunk.last().unwrap().hidden_state.clone();
            }

            let sequence = Sequence::new(chunk, initial_hidden, env_id, self.sequence_counter);
            self.sequence_counter += 1;
            self.add_sequence(sequence);
        } else if episode_done {
            // Episode ended before reaching chunk length - create padded sequence
            let chunk: Vec<_> = episode.drain(..).collect();
            let sequence =
                Sequence::new_padded(chunk, initial_hidden, env_id, self.sequence_counter, chunk_len);
            self.sequence_counter += 1;
            self.add_sequence(sequence);
        }
    }

    /// Add a sequence to the buffer, respecting max capacity.
    fn add_sequence(&mut self, sequence: Sequence) {
        self.sequences.push_back(sequence);
        while self.sequences.len() > self.config.max_sequences {
            self.sequences.pop_front();
        }
    }

    /// Pop a sequence from the buffer (FIFO).
    pub fn pop(&mut self) -> Option<Sequence> {
        self.sequences.pop_front()
    }

    /// Get sequences for a minibatch without removing them.
    pub fn sample_minibatch(&self, size: usize) -> Vec<&Sequence> {
        self.sequences.iter().take(size).collect()
    }

    /// Take sequences for training (removes them from buffer).
    pub fn take_minibatch(&mut self, size: usize) -> Vec<Sequence> {
        let n = size.min(self.sequences.len());
        self.sequences.drain(..n).collect()
    }

    /// Force-finalize all in-progress episodes (e.g., at end of rollout).
    pub fn finalize_all(&mut self) {
        for env_id in 0..self.config.n_envs {
            if !self.episodes[env_id].is_empty() {
                self.finalize_episode_chunk(env_id, true);
            }
        }
    }

    /// Clear all data from the buffer.
    pub fn clear(&mut self) {
        for episode in &mut self.episodes {
            episode.clear();
        }
        for hidden in &mut self.episode_initial_hidden {
            hidden.clear();
        }
        self.sequences.clear();
    }

    /// Get total number of transitions across all in-progress episodes.
    pub fn n_in_progress(&self) -> usize {
        self.episodes.iter().map(|e| e.len()).sum()
    }

    /// Get total number of transitions in completed sequences.
    pub fn n_completed(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
    }

    /// Iterator over completed sequences.
    pub fn iter(&self) -> impl Iterator<Item = &Sequence> {
        self.sequences.iter()
    }
}

/// Batch of sequences for training.
#[derive(Debug)]
pub struct SequenceBatch {
    /// Batch of sequences.
    pub sequences: Vec<Sequence>,
}

impl SequenceBatch {
    /// Create a new sequence batch.
    pub fn new(sequences: Vec<Sequence>) -> Self {
        Self { sequences }
    }

    /// Number of sequences in the batch.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get batch dimensions: (batch_size, sequence_length).
    pub fn dims(&self) -> (usize, usize) {
        if self.sequences.is_empty() {
            (0, 0)
        } else {
            (self.sequences.len(), self.sequences[0].len())
        }
    }

    /// Extract states as flattened vector.
    ///
    /// Returns shape info and data for reconstruction as tensor.
    pub fn states_flat(&self, obs_size: usize) -> (usize, usize, usize, Vec<f32>) {
        let (batch_size, seq_len) = self.dims();
        let mut data = Vec::with_capacity(batch_size * seq_len * obs_size);

        for seq in &self.sequences {
            for t in &seq.transitions {
                data.extend_from_slice(t.state());
            }
        }

        (batch_size, seq_len, obs_size, data)
    }

    /// Extract actions as vector (discrete).
    pub fn actions_discrete(&self) -> Vec<u32> {
        let mut actions = Vec::new();
        for seq in &self.sequences {
            for t in &seq.transitions {
                actions.push(t.action().as_discrete());
            }
        }
        actions
    }

    /// Extract rewards.
    pub fn rewards(&self) -> Vec<f32> {
        let mut rewards = Vec::new();
        for seq in &self.sequences {
            for t in &seq.transitions {
                rewards.push(t.reward());
            }
        }
        rewards
    }

    /// Extract log probabilities.
    pub fn log_probs(&self) -> Vec<f32> {
        let mut log_probs = Vec::new();
        for seq in &self.sequences {
            for t in &seq.transitions {
                log_probs.push(t.log_prob());
            }
        }
        log_probs
    }

    /// Extract values.
    pub fn values(&self) -> Vec<f32> {
        let mut values = Vec::new();
        for seq in &self.sequences {
            for t in &seq.transitions {
                values.push(t.value());
            }
        }
        values
    }

    /// Extract validity masks.
    pub fn valid_masks(&self) -> Vec<bool> {
        let mut masks = Vec::new();
        for seq in &self.sequences {
            masks.extend_from_slice(&seq.valid_mask);
        }
        masks
    }

    /// Extract initial hidden states (one per sequence).
    pub fn initial_hiddens(&self) -> Vec<&[f32]> {
        self.sequences.iter().map(|s| s.initial_hidden.as_slice()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::PPOTransition;

    fn make_transition(
        env_id: usize,
        step: usize,
        terminal: bool,
        seq_id: u64,
    ) -> RecurrentPPOTransition {
        let ppo = PPOTransition::new_discrete(
            vec![step as f32],
            0,
            1.0,
            vec![(step + 1) as f32],
            terminal,
            false,
            -0.5,
            1.0,
        );
        RecurrentPPOTransition::from_ppo(
            ppo,
            vec![0.1, 0.2], // hidden state
            seq_id,
            step,
            step == 0,
        )
    }

    #[test]
    fn test_sequence_buffer_basic() {
        let config = SequenceBufferConfig::new(2, 4);
        let mut buffer = SequenceBuffer::new(config);

        assert_eq!(buffer.n_sequences(), 0);

        // Add 4 transitions to env 0 - should create a sequence
        for step in 0..4 {
            buffer.push(0, make_transition(0, step, false, 0));
        }

        assert_eq!(buffer.n_sequences(), 1);
    }

    #[test]
    fn test_sequence_buffer_episode_done() {
        let config = SequenceBufferConfig::new(2, 8);
        let mut buffer = SequenceBuffer::new(config);

        // Add 3 transitions then terminal - should create padded sequence
        for step in 0..3 {
            buffer.push(0, make_transition(0, step, step == 2, 0));
        }

        assert_eq!(buffer.n_sequences(), 1);
        let seq = buffer.pop().unwrap();
        assert_eq!(seq.len(), 8); // Padded to sequence_length
        assert_eq!(seq.n_valid(), 3); // Only 3 valid
    }

    #[test]
    fn test_sequence_buffer_multiple_envs() {
        let config = SequenceBufferConfig::new(2, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Add to both environments
        for step in 0..4 {
            buffer.push(0, make_transition(0, step, false, 0));
            buffer.push(1, make_transition(1, step, false, 1));
        }

        assert_eq!(buffer.n_sequences(), 2);
    }

    #[test]
    fn test_sequence_batch() {
        let seq1 = Sequence::new(
            vec![make_transition(0, 0, false, 0), make_transition(0, 1, false, 0)],
            vec![0.0, 0.0],
            0,
            0,
        );
        let seq2 = Sequence::new(
            vec![make_transition(1, 0, false, 1), make_transition(1, 1, false, 1)],
            vec![0.0, 0.0],
            1,
            1,
        );

        let batch = SequenceBatch::new(vec![seq1, seq2]);
        assert_eq!(batch.dims(), (2, 2));
        assert_eq!(batch.len(), 2);

        let rewards = batch.rewards();
        assert_eq!(rewards.len(), 4);
    }

    #[test]
    fn test_finalize_all() {
        let config = SequenceBufferConfig::new(2, 8);
        let mut buffer = SequenceBuffer::new(config);

        // Add some transitions without completing sequences
        buffer.push(0, make_transition(0, 0, false, 0));
        buffer.push(0, make_transition(0, 1, false, 0));
        buffer.push(1, make_transition(1, 0, false, 1));

        assert_eq!(buffer.n_sequences(), 0);
        assert_eq!(buffer.n_in_progress(), 3);

        // Force finalize
        buffer.finalize_all();

        assert_eq!(buffer.n_sequences(), 2);
        assert_eq!(buffer.n_in_progress(), 0);
    }
}
