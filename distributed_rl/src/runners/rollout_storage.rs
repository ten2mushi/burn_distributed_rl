//! Generic rollout storage for on-policy algorithms.
//!
//! This module provides a unified storage structure for collecting
//! transitions during rollout, working with any ActionValue type.
//!
//! The storage is designed to be used by the generic Learner and
//! supports both discrete and continuous action spaces.
//!
//! # Sequence Handling
//!
//! For recurrent training, the storage can prepare sequences that are
//! truncated at terminal states (Option B). This ensures proper BPTT
//! by not allowing gradients to flow across episode boundaries.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::marker::PhantomData;

use crate::algorithms::action_policy::ActionValue;

// ============================================================================
// Sequence Types for Recurrent Training
// ============================================================================

/// A sequence of transitions for recurrent training.
///
/// Sequences are truncated at terminal states to ensure proper BPTT.
/// Each sequence represents a contiguous stretch of timesteps for a
/// single environment.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Environment index this sequence is from
    pub env_idx: usize,
    /// Starting step index (inclusive)
    pub start_step: usize,
    /// Ending step index (exclusive)
    pub end_step: usize,
    /// Index into hidden_states for initial hidden at start of sequence
    pub hidden_state_idx: usize,
    /// Actual length of this sequence
    pub length: usize,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(
        env_idx: usize,
        start_step: usize,
        end_step: usize,
        hidden_state_idx: usize,
    ) -> Self {
        Self {
            env_idx,
            start_step,
            end_step,
            hidden_state_idx,
            length: end_step - start_step,
        }
    }
}

/// Collection of prepared sequences with batch extraction methods.
pub struct PreparedSequences<'a, A: ActionValue, H: Clone> {
    /// Reference to the storage
    storage: &'a RolloutStorage<A, H>,
    /// Reference to computed values
    computed: &'a ComputedValues,
    /// List of sequences
    pub sequences: Vec<Sequence>,
    /// Maximum sequence length across all sequences
    pub max_length: usize,
}

// ============================================================================
// RolloutStorage
// ============================================================================

/// Generic rollout storage for on-policy algorithms.
///
/// Stores transitions during rollout collection, then provides methods
/// to extract data as tensors for training.
///
/// # Type Parameters
///
/// - `A`: Action value type (e.g., `DiscreteAction`, `ContinuousAction`)
/// - `H`: Hidden state type (e.g., `()`, `RecurrentHidden<B>`)
pub struct RolloutStorage<A: ActionValue, H: Clone> {
    /// Flattened observations: [step * n_envs * obs_size]
    pub states: Vec<f32>,
    /// Actions taken at each step
    pub actions: Vec<A>,
    /// Rewards received
    pub rewards: Vec<f32>,
    /// Done flags (terminal or truncated)
    pub dones: Vec<bool>,
    /// Terminal flags (true terminal, not truncation)
    pub terminals: Vec<bool>,
    /// Value estimates from model
    pub values: Vec<f32>,
    /// Log probabilities of actions
    pub log_probs: Vec<f32>,
    /// Hidden states at each step (for recurrent policies)
    pub hidden_states: Vec<H>,

    // Metadata
    /// Number of parallel environments
    pub n_envs: usize,
    /// Observation size
    pub obs_size: usize,
    /// Number of steps collected
    pub step_count: usize,
    /// Configured rollout length
    pub rollout_len: usize,

    _marker: PhantomData<A>,
}

impl<A: ActionValue, H: Clone + Default> RolloutStorage<A, H> {
    /// Create new storage with given capacity.
    pub fn new(n_envs: usize, rollout_len: usize, obs_size: usize) -> Self {
        let capacity = n_envs * rollout_len;
        Self {
            states: Vec::with_capacity(capacity * obs_size),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            terminals: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            hidden_states: Vec::with_capacity(capacity),
            n_envs,
            obs_size,
            step_count: 0,
            rollout_len,
            _marker: PhantomData,
        }
    }

    /// Push a step of transitions (from all environments).
    ///
    /// # Arguments
    ///
    /// - `states`: Flattened observations [n_envs * obs_size]
    /// - `actions`: Actions for each environment [n_envs]
    /// - `rewards`: Rewards for each environment [n_envs]
    /// - `dones`: Done flags for each environment [n_envs]
    /// - `terminals`: Terminal flags for each environment [n_envs]
    /// - `values`: Value estimates for each environment [n_envs]
    /// - `log_probs`: Log probabilities for each environment [n_envs]
    /// - `hidden`: Hidden state at this step (for recurrent)
    pub fn push_step(
        &mut self,
        states: &[f32],
        actions: Vec<A>,
        rewards: &[f32],
        dones: &[bool],
        terminals: &[bool],
        values: &[f32],
        log_probs: &[f32],
        hidden: H,
    ) {
        debug_assert_eq!(states.len(), self.n_envs * self.obs_size);
        debug_assert_eq!(actions.len(), self.n_envs);
        debug_assert_eq!(rewards.len(), self.n_envs);
        debug_assert_eq!(dones.len(), self.n_envs);
        debug_assert_eq!(terminals.len(), self.n_envs);
        debug_assert_eq!(values.len(), self.n_envs);
        debug_assert_eq!(log_probs.len(), self.n_envs);

        self.states.extend_from_slice(states);
        self.actions.extend(actions);
        self.rewards.extend_from_slice(rewards);
        self.dones.extend_from_slice(dones);
        self.terminals.extend_from_slice(terminals);
        self.values.extend_from_slice(values);
        self.log_probs.extend_from_slice(log_probs);
        self.hidden_states.push(hidden);

        self.step_count += 1;
    }

    /// Check if storage is full.
    pub fn is_full(&self) -> bool {
        self.step_count >= self.rollout_len
    }

    /// Get total number of transitions.
    pub fn len(&self) -> usize {
        self.step_count * self.n_envs
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.step_count == 0
    }

    /// Clear storage for next rollout.
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.dones.clear();
        self.terminals.clear();
        self.values.clear();
        self.log_probs.clear();
        self.hidden_states.clear();
        self.step_count = 0;
    }

    /// Get states as tensor [total_transitions, obs_size].
    pub fn states_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        let n = self.len();
        Tensor::<B, 1>::from_floats(&self.states[..], device).reshape([n, self.obs_size])
    }

    /// Get values as 1D tensor [total_transitions].
    pub fn values_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.values[..], device)
    }

    /// Get log probs as 1D tensor [total_transitions].
    pub fn log_probs_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.log_probs[..], device)
    }

    /// Get rewards as 1D tensor [total_transitions].
    pub fn rewards_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.rewards[..], device)
    }

    /// Get done flags as float tensor (0.0 or 1.0) [total_transitions].
    pub fn dones_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let dones_f32: Vec<f32> = self.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
        Tensor::<B, 1>::from_floats(&dones_f32[..], device)
    }

    /// Get rewards as nested Vec [rollout_len][n_envs].
    pub fn rewards_by_step(&self) -> Vec<Vec<f32>> {
        (0..self.step_count)
            .map(|step| {
                let start = step * self.n_envs;
                self.rewards[start..start + self.n_envs].to_vec()
            })
            .collect()
    }

    /// Get values as nested Vec [rollout_len][n_envs].
    pub fn values_by_step(&self) -> Vec<Vec<f32>> {
        (0..self.step_count)
            .map(|step| {
                let start = step * self.n_envs;
                self.values[start..start + self.n_envs].to_vec()
            })
            .collect()
    }

    /// Get dones as nested Vec [rollout_len][n_envs].
    pub fn dones_by_step(&self) -> Vec<Vec<bool>> {
        (0..self.step_count)
            .map(|step| {
                let start = step * self.n_envs;
                self.dones[start..start + self.n_envs].to_vec()
            })
            .collect()
    }

    /// Get iterator over (env_idx, step) pairs for a specific environment.
    pub fn env_indices(&self, env_idx: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.step_count).map(move |step| step * self.n_envs + env_idx)
    }

    /// Prepare sequences for recurrent training with truncation at terminals.
    ///
    /// This method scans through each environment's trajectory and creates
    /// sequences that are truncated at terminal states. This ensures proper
    /// BPTT by preventing gradient flow across episode boundaries.
    ///
    /// # Arguments
    ///
    /// * `computed` - The computed advantages and returns
    /// * `max_seq_len` - Maximum sequence length (for chunking long episodes)
    ///
    /// # Returns
    ///
    /// A `PreparedSequences` object containing all sequences ready for batched training.
    pub fn prepare_sequences<'a>(
        &'a self,
        computed: &'a ComputedValues,
        max_seq_len: usize,
    ) -> PreparedSequences<'a, A, H> {
        let mut sequences = Vec::new();
        let mut max_length = 0usize;

        for env_idx in 0..self.n_envs {
            let mut seq_start = 0usize;

            for step in 0..self.step_count {
                let flat_idx = step * self.n_envs + env_idx;
                let is_terminal = self.terminals[flat_idx];
                let is_done = self.dones[flat_idx];
                let at_max_len = (step - seq_start + 1) >= max_seq_len;
                let at_rollout_end = step == self.step_count - 1;

                // End sequence if: terminal, at max length, or at rollout end
                if is_terminal || is_done || at_max_len || at_rollout_end {
                    let seq_end = step + 1; // exclusive
                    let seq_len = seq_end - seq_start;

                    if seq_len > 0 {
                        // The hidden state index is the step index in hidden_states
                        // For step 0, we use initial hidden (index 0 or default)
                        let hidden_idx = seq_start;

                        let sequence = Sequence::new(env_idx, seq_start, seq_end, hidden_idx);
                        max_length = max_length.max(sequence.length);
                        sequences.push(sequence);
                    }

                    // Next sequence starts after this step
                    seq_start = step + 1;
                }
            }
        }

        PreparedSequences {
            storage: self,
            computed,
            sequences,
            max_length,
        }
    }
}

// ============================================================================
// PreparedSequences Implementation
// ============================================================================

impl<'a, A: ActionValue, H: Clone> PreparedSequences<'a, A, H> {
    /// Get number of sequences.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Generate shuffled sequence indices for minibatching.
    pub fn generate_sequence_batches(&self, minibatch_size: usize) -> Vec<Vec<usize>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut indices: Vec<usize> = (0..self.sequences.len()).collect();
        indices.shuffle(&mut thread_rng());

        indices
            .chunks(minibatch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Get the maximum sequence length in a batch of sequences.
    pub fn batch_max_length(&self, seq_indices: &[usize]) -> usize {
        seq_indices
            .iter()
            .map(|&idx| self.sequences[idx].length)
            .max()
            .unwrap_or(0)
    }

    /// Extract a batch of sequences as tensors with padding.
    ///
    /// Returns tensors where shorter sequences are zero-padded to max_len,
    /// along with a mask tensor indicating valid timesteps.
    ///
    /// # Returns
    ///
    /// `SequenceBatch` containing:
    /// - states: [batch, max_len, obs_size]
    /// - actions: [batch, max_len] (as Vec<Vec<A>>)
    /// - old_log_probs: [batch, max_len]
    /// - old_values: [batch, max_len]
    /// - advantages: [batch, max_len]
    /// - returns: [batch, max_len]
    /// - masks: [batch, max_len] (1.0 for valid, 0.0 for padding)
    /// - lengths: actual length of each sequence
    pub fn extract_sequence_batch(&self, seq_indices: &[usize]) -> SequenceBatch<A> {
        let batch_size = seq_indices.len();
        let max_len = self.batch_max_length(seq_indices);
        let obs_size = self.storage.obs_size;
        let n_envs = self.storage.n_envs;

        // Allocate padded arrays
        let mut states = vec![0.0f32; batch_size * max_len * obs_size];
        let mut actions: Vec<Vec<A>> = vec![Vec::with_capacity(max_len); batch_size];
        let mut old_log_probs = vec![0.0f32; batch_size * max_len];
        let mut old_values = vec![0.0f32; batch_size * max_len];
        let mut advantages = vec![0.0f32; batch_size * max_len];
        let mut returns = vec![0.0f32; batch_size * max_len];
        let mut masks = vec![0.0f32; batch_size * max_len];
        let mut lengths = Vec::with_capacity(batch_size);
        let mut hidden_indices = Vec::with_capacity(batch_size);

        for (batch_idx, &seq_idx) in seq_indices.iter().enumerate() {
            let seq = &self.sequences[seq_idx];
            lengths.push(seq.length);
            hidden_indices.push(seq.hidden_state_idx);

            for t in 0..seq.length {
                let step = seq.start_step + t;
                let flat_idx = step * n_envs + seq.env_idx;

                // States: [batch_idx, t, :] in flattened form
                let state_dst_start = (batch_idx * max_len + t) * obs_size;
                let state_src_start = flat_idx * obs_size;
                states[state_dst_start..state_dst_start + obs_size]
                    .copy_from_slice(&self.storage.states[state_src_start..state_src_start + obs_size]);

                // Actions
                actions[batch_idx].push(self.storage.actions[flat_idx].clone());

                // Scalar values: [batch_idx, t]
                let scalar_idx = batch_idx * max_len + t;
                old_log_probs[scalar_idx] = self.storage.log_probs[flat_idx];
                old_values[scalar_idx] = self.storage.values[flat_idx];
                advantages[scalar_idx] = self.computed.advantages[flat_idx];
                returns[scalar_idx] = self.computed.returns[flat_idx];
                masks[scalar_idx] = 1.0;
            }

            // Pad actions with clones of the last action (won't affect loss due to mask)
            if let Some(last_action) = actions[batch_idx].last().cloned() {
                while actions[batch_idx].len() < max_len {
                    actions[batch_idx].push(last_action.clone());
                }
            }
        }

        SequenceBatch {
            states,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
            masks,
            lengths,
            hidden_indices,
            batch_size,
            max_len,
            obs_size,
        }
    }
}

/// A batch of sequences for recurrent training.
///
/// All sequences are padded to the same length (max_len) with zeros.
/// The masks tensor indicates which timesteps are valid (1.0) vs padding (0.0).
pub struct SequenceBatch<A: ActionValue> {
    /// Flattened states [batch * max_len * obs_size]
    pub states: Vec<f32>,
    /// Actions per sequence [batch][max_len]
    pub actions: Vec<Vec<A>>,
    /// Old log probs [batch * max_len]
    pub old_log_probs: Vec<f32>,
    /// Old values [batch * max_len]
    pub old_values: Vec<f32>,
    /// Advantages [batch * max_len]
    pub advantages: Vec<f32>,
    /// Returns [batch * max_len]
    pub returns: Vec<f32>,
    /// Masks (1.0 = valid, 0.0 = padding) [batch * max_len]
    pub masks: Vec<f32>,
    /// Actual length of each sequence [batch]
    pub lengths: Vec<usize>,
    /// Hidden state indices for each sequence [batch]
    pub hidden_indices: Vec<usize>,
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Observation size
    pub obs_size: usize,
}

impl<A: ActionValue> SequenceBatch<A> {
    /// Get states at timestep t as tensor [batch, obs_size].
    pub fn states_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 2> {
        let mut step_states = Vec::with_capacity(self.batch_size * self.obs_size);
        for batch_idx in 0..self.batch_size {
            let start = (batch_idx * self.max_len + t) * self.obs_size;
            step_states.extend_from_slice(&self.states[start..start + self.obs_size]);
        }
        Tensor::<B, 1>::from_floats(&step_states[..], device)
            .reshape([self.batch_size, self.obs_size])
    }

    /// Get actions at timestep t.
    pub fn actions_at_step(&self, t: usize) -> Vec<A> {
        self.actions
            .iter()
            .map(|seq_actions| seq_actions[t].clone())
            .collect()
    }

    /// Get old log probs at timestep t as tensor [batch].
    pub fn old_log_probs_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let step_lp: Vec<f32> = (0..self.batch_size)
            .map(|batch_idx| self.old_log_probs[batch_idx * self.max_len + t])
            .collect();
        Tensor::<B, 1>::from_floats(&step_lp[..], device)
    }

    /// Get old values at timestep t as tensor [batch].
    pub fn old_values_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let step_v: Vec<f32> = (0..self.batch_size)
            .map(|batch_idx| self.old_values[batch_idx * self.max_len + t])
            .collect();
        Tensor::<B, 1>::from_floats(&step_v[..], device)
    }

    /// Get advantages at timestep t as tensor [batch].
    pub fn advantages_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let step_adv: Vec<f32> = (0..self.batch_size)
            .map(|batch_idx| self.advantages[batch_idx * self.max_len + t])
            .collect();
        Tensor::<B, 1>::from_floats(&step_adv[..], device)
    }

    /// Get returns at timestep t as tensor [batch].
    pub fn returns_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let step_ret: Vec<f32> = (0..self.batch_size)
            .map(|batch_idx| self.returns[batch_idx * self.max_len + t])
            .collect();
        Tensor::<B, 1>::from_floats(&step_ret[..], device)
    }

    /// Get masks at timestep t as tensor [batch].
    pub fn masks_at_step<B: Backend>(&self, t: usize, device: &B::Device) -> Tensor<B, 1> {
        let step_mask: Vec<f32> = (0..self.batch_size)
            .map(|batch_idx| self.masks[batch_idx * self.max_len + t])
            .collect();
        Tensor::<B, 1>::from_floats(&step_mask[..], device)
    }

    /// Get total valid timesteps (sum of all sequence lengths).
    pub fn total_valid_steps(&self) -> usize {
        self.lengths.iter().sum()
    }
}

// ============================================================================
// Computed Advantages/Returns Storage
// ============================================================================

/// Computed advantages and returns for training.
///
/// These are computed after rollout collection using GAE.
pub struct ComputedValues {
    /// Advantages (normalized)
    pub advantages: Vec<f32>,
    /// Value targets (returns)
    pub returns: Vec<f32>,
}

impl ComputedValues {
    /// Create from raw advantages and returns.
    pub fn new(advantages: Vec<f32>, returns: Vec<f32>) -> Self {
        Self { advantages, returns }
    }

    /// Get advantages as tensor.
    pub fn advantages_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.advantages[..], device)
    }

    /// Get returns as tensor.
    pub fn returns_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.returns[..], device)
    }
}

// ============================================================================
// Minibatch Iterator
// ============================================================================

/// Indices for a minibatch.
pub struct MinibatchIndices {
    /// Indices into the flattened storage arrays
    pub indices: Vec<usize>,
}

impl MinibatchIndices {
    /// Create from index list.
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Get number of samples in this minibatch.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Generate shuffled minibatch indices.
pub fn generate_minibatches(total: usize, minibatch_size: usize) -> Vec<MinibatchIndices> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut thread_rng());

    indices
        .chunks(minibatch_size)
        .map(|chunk| MinibatchIndices::new(chunk.to_vec()))
        .collect()
}

/// Extract minibatch data from storage.
pub fn extract_minibatch<A: ActionValue, H: Clone>(
    storage: &RolloutStorage<A, H>,
    computed: &ComputedValues,
    indices: &MinibatchIndices,
) -> MinibatchData<A> {
    let batch_size = indices.len();
    let obs_size = storage.obs_size;

    let mut states = Vec::with_capacity(batch_size * obs_size);
    let mut actions = Vec::with_capacity(batch_size);
    let mut old_values = Vec::with_capacity(batch_size);
    let mut old_log_probs = Vec::with_capacity(batch_size);
    let mut advantages = Vec::with_capacity(batch_size);
    let mut returns = Vec::with_capacity(batch_size);

    for &idx in &indices.indices {
        // Extract observation
        let obs_start = idx * obs_size;
        states.extend_from_slice(&storage.states[obs_start..obs_start + obs_size]);

        actions.push(storage.actions[idx].clone());
        old_values.push(storage.values[idx]);
        old_log_probs.push(storage.log_probs[idx]);
        advantages.push(computed.advantages[idx]);
        returns.push(computed.returns[idx]);
    }

    MinibatchData {
        states,
        actions,
        old_values,
        old_log_probs,
        advantages,
        returns,
        obs_size,
    }
}

/// Data for a single minibatch.
pub struct MinibatchData<A: ActionValue> {
    /// Flattened states [batch * obs_size]
    pub states: Vec<f32>,
    /// Actions [batch]
    pub actions: Vec<A>,
    /// Old value estimates [batch]
    pub old_values: Vec<f32>,
    /// Old log probabilities [batch]
    pub old_log_probs: Vec<f32>,
    /// Advantages [batch]
    pub advantages: Vec<f32>,
    /// Returns (value targets) [batch]
    pub returns: Vec<f32>,
    /// Observation size
    pub obs_size: usize,
}

impl<A: ActionValue> MinibatchData<A> {
    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.actions.len()
    }

    /// Get states as tensor [batch, obs_size].
    pub fn states_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        let n = self.batch_size();
        Tensor::<B, 1>::from_floats(&self.states[..], device).reshape([n, self.obs_size])
    }

    /// Get old values as tensor [batch].
    pub fn old_values_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.old_values[..], device)
    }

    /// Get old log probs as tensor [batch].
    pub fn old_log_probs_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.old_log_probs[..], device)
    }

    /// Get advantages as tensor [batch].
    pub fn advantages_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.advantages[..], device)
    }

    /// Get returns as tensor [batch].
    pub fn returns_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(&self.returns[..], device)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::action_policy::DiscreteAction;

    #[test]
    fn test_rollout_storage_new() {
        let storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(4, 10, 8);

        assert!(storage.is_empty());
        assert!(!storage.is_full());
        assert_eq!(storage.n_envs, 4);
        assert_eq!(storage.rollout_len, 10);
        assert_eq!(storage.obs_size, 8);
    }

    #[test]
    fn test_rollout_storage_push_step() {
        let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

        // Push first step
        storage.push_step(
            &[0.0; 8], // 2 envs * 4 obs_size
            vec![DiscreteAction(0), DiscreteAction(1)],
            &[1.0, 2.0],
            &[false, false],
            &[false, false],
            &[0.5, 0.6],
            &[-0.3, -0.4],
            (),
        );

        assert_eq!(storage.step_count, 1);
        assert_eq!(storage.len(), 2); // 1 step * 2 envs
        assert!(!storage.is_full());

        // Push remaining steps
        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(1), DiscreteAction(0)],
            &[2.0, 3.0],
            &[false, true],
            &[false, true],
            &[0.7, 0.8],
            &[-0.5, -0.6],
            (),
        );
        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(0), DiscreteAction(0)],
            &[3.0, 4.0],
            &[true, false],
            &[true, false],
            &[0.9, 1.0],
            &[-0.7, -0.8],
            (),
        );

        assert_eq!(storage.step_count, 3);
        assert_eq!(storage.len(), 6);
        assert!(storage.is_full());
    }

    #[test]
    fn test_rollout_storage_clear() {
        let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(0), DiscreteAction(1)],
            &[1.0, 2.0],
            &[false, false],
            &[false, false],
            &[0.5, 0.6],
            &[-0.3, -0.4],
            (),
        );

        assert!(!storage.is_empty());

        storage.clear();

        assert!(storage.is_empty());
        assert_eq!(storage.step_count, 0);
    }

    #[test]
    fn test_generate_minibatches() {
        let batches = generate_minibatches(100, 32);

        // Should have 4 batches: 32, 32, 32, 4
        assert_eq!(batches.len(), 4);

        // First three should be full
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 32);
        assert_eq!(batches[2].len(), 32);

        // Last one is remainder
        assert_eq!(batches[3].len(), 4);

        // All indices should be unique
        let mut all_indices: Vec<usize> = batches.iter().flat_map(|b| b.indices.clone()).collect();
        all_indices.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(all_indices, expected);
    }

    #[test]
    fn test_rewards_by_step() {
        let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(0), DiscreteAction(1)],
            &[1.0, 2.0],
            &[false, false],
            &[false, false],
            &[0.5, 0.6],
            &[-0.3, -0.4],
            (),
        );
        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(1), DiscreteAction(0)],
            &[3.0, 4.0],
            &[false, true],
            &[false, true],
            &[0.7, 0.8],
            &[-0.5, -0.6],
            (),
        );

        let rewards_by_step = storage.rewards_by_step();
        assert_eq!(rewards_by_step.len(), 2);
        assert_eq!(rewards_by_step[0], vec![1.0, 2.0]);
        assert_eq!(rewards_by_step[1], vec![3.0, 4.0]);
    }
}
