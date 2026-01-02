//! Temporal policy abstractions for feed-forward and recurrent architectures.
//!
//! This module provides the core traits for abstracting over temporal structure:
//! - [`HiddenStateType`]: Abstraction over hidden state representations
//! - [`TemporalPolicy`]: Main trait for temporal policy configuration
//!
//! # Implementations
//!
//! - [`FeedForward`]: Stateless policies (Hidden = `()`)
//! - [`Recurrent`]: Recurrent policies with LSTM/GRU hidden states

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::fmt::Debug;

use crate::core::recurrent::HiddenState;

// ============================================================================
// HiddenStateType Trait
// ============================================================================

/// Trait for hidden state type abstraction.
///
/// This trait allows abstracting over:
/// - No hidden state (`()` for feed-forward policies)
/// - Recurrent hidden states ([`RecurrentHidden`] for LSTM/GRU)
///
/// The trait provides methods for:
/// - Creating initial states
/// - Resetting states for specific environments
/// - Serialization/deserialization for buffer storage
pub trait HiddenStateType<B: Backend>: Clone + Send + Default + 'static {
    /// Create initial hidden state for given number of environments.
    fn initial(n_envs: usize, device: &B::Device, config: &HiddenConfig) -> Self;

    /// Whether this is a "real" hidden state (vs unit type for feed-forward).
    fn is_stateful() -> bool;

    /// Reset state for a single environment (for terminal states).
    fn reset(&mut self, env_idx: usize, device: &B::Device);

    /// Reset states at specific environment indices (for terminal states).
    fn reset_indices(&mut self, indices: &[usize], device: &B::Device, config: &HiddenConfig);

    /// Get hidden state for a single environment as a vector.
    fn get_env_vec(&self, env_idx: usize) -> Vec<f32>;

    /// Flatten to vector for buffer storage.
    fn to_vec(&self) -> Vec<f32>;

    /// Restore from flattened vector.
    fn from_vec(data: &[f32], n_envs: usize, device: &B::Device, config: &HiddenConfig) -> Self;

    /// Size of flattened state per environment (0 for unit type).
    fn per_env_size(config: &HiddenConfig) -> usize;
}

/// Configuration for hidden state creation.
#[derive(Debug, Clone, Default)]
pub struct HiddenConfig {
    /// Size of hidden state vector per environment.
    pub hidden_size: usize,
    /// Whether this is LSTM (true) or GRU (false).
    pub has_cell: bool,
}

impl HiddenConfig {
    /// Create LSTM configuration.
    pub fn lstm(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            has_cell: true,
        }
    }

    /// Create GRU configuration.
    pub fn gru(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            has_cell: false,
        }
    }

    /// Create feed-forward (no hidden state) configuration.
    pub fn none() -> Self {
        Self::default()
    }
}

// ============================================================================
// Feed-Forward Hidden State (Unit Type)
// ============================================================================

/// Feed-forward "hidden state" - unit type with no actual state.
impl<B: Backend> HiddenStateType<B> for () {
    fn initial(_n_envs: usize, _device: &B::Device, _config: &HiddenConfig) -> Self {}

    fn is_stateful() -> bool {
        false
    }

    fn reset(&mut self, _env_idx: usize, _device: &B::Device) {
        // No-op for stateless policy
    }

    fn reset_indices(&mut self, _indices: &[usize], _device: &B::Device, _config: &HiddenConfig) {
        // No-op for stateless policy
    }

    fn get_env_vec(&self, _env_idx: usize) -> Vec<f32> {
        Vec::new()
    }

    fn to_vec(&self) -> Vec<f32> {
        Vec::new()
    }

    fn from_vec(_data: &[f32], _n_envs: usize, _device: &B::Device, _config: &HiddenConfig) -> Self {
    }

    fn per_env_size(_config: &HiddenConfig) -> usize {
        0
    }
}

// ============================================================================
// Recurrent Hidden State
// ============================================================================

/// Recurrent hidden state container for per-environment states.
#[derive(Clone)]
pub struct RecurrentHidden<B: Backend> {
    /// Per-environment hidden states.
    pub states: Vec<HiddenState<B>>,
    /// Hidden size.
    pub hidden_size: usize,
    /// Whether this is LSTM (has cell state).
    pub has_cell: bool,
}

impl<B: Backend> RecurrentHidden<B> {
    /// Create initial hidden states for all environments.
    pub fn new(n_envs: usize, hidden_size: usize, has_cell: bool, device: &B::Device) -> Self {
        let states = (0..n_envs)
            .map(|_| {
                if has_cell {
                    HiddenState::lstm(
                        Tensor::zeros([1, hidden_size], device),
                        Tensor::zeros([1, hidden_size], device),
                    )
                } else {
                    HiddenState::gru(Tensor::zeros([1, hidden_size], device))
                }
            })
            .collect();

        Self {
            states,
            hidden_size,
            has_cell,
        }
    }

    /// Get hidden state for a specific environment.
    pub fn get(&self, env_idx: usize) -> &HiddenState<B> {
        &self.states[env_idx]
    }

    /// Set hidden state for a specific environment.
    pub fn set(&mut self, env_idx: usize, state: HiddenState<B>) {
        self.states[env_idx] = state;
    }

    /// Reset state for a specific environment to zeros.
    pub fn reset(&mut self, env_idx: usize, device: &B::Device) {
        self.states[env_idx] = if self.has_cell {
            HiddenState::lstm(
                Tensor::zeros([1, self.hidden_size], device),
                Tensor::zeros([1, self.hidden_size], device),
            )
        } else {
            HiddenState::gru(Tensor::zeros([1, self.hidden_size], device))
        };
    }

    /// Stack all hidden states into a batch for forward pass.
    ///
    /// Returns (hidden_batch, cell_batch_option) where tensors are [n_envs, hidden_size].
    pub fn to_batch(&self) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let hidden_batch = Tensor::cat(
            self.states
                .iter()
                .map(|s| s.hidden.clone())
                .collect::<Vec<_>>(),
            0,
        );

        let cell_batch = if self.has_cell {
            Some(Tensor::cat(
                self.states
                    .iter()
                    .map(|s| s.cell.clone().unwrap())
                    .collect::<Vec<_>>(),
                0,
            ))
        } else {
            None
        };

        (hidden_batch, cell_batch)
    }

    /// Create a batched HiddenState from all environments.
    pub fn to_batched_state(&self) -> HiddenState<B> {
        let (h, c) = self.to_batch();
        if let Some(cell) = c {
            HiddenState::lstm(h, cell)
        } else {
            HiddenState::gru(h)
        }
    }

    /// Update from a batched HiddenState result.
    pub fn from_batched_state(&mut self, batched: HiddenState<B>) {
        let n_envs = self.states.len();
        for env_idx in 0..n_envs {
            let h_slice = batched
                .hidden
                .clone()
                .slice([env_idx..(env_idx + 1), 0..self.hidden_size]);

            let state = if self.has_cell {
                let c_slice = batched
                    .cell
                    .clone()
                    .unwrap()
                    .slice([env_idx..(env_idx + 1), 0..self.hidden_size]);
                HiddenState::lstm(h_slice, c_slice)
            } else {
                HiddenState::gru(h_slice)
            };

            self.states[env_idx] = state;
        }
    }

    /// Create a detached copy of the hidden state by recreating tensors from data.
    ///
    /// This breaks the autodiff computation graph, preventing gradients from flowing
    /// through hidden state transitions. Essential for TBPTT (Truncated Backpropagation
    /// Through Time) where we don't want to backpropagate through the entire sequence.
    ///
    /// # Why This Is Needed
    ///
    /// Without detaching, each `model.forward()` call adds to the computation graph.
    /// When processing sequences timestep-by-timestep, the graph grows to O(seq_len * n_sequences)
    /// nodes, causing `backward()` to hang or run extremely slowly.
    ///
    /// This matches how the original log_probs were collected during rollout
    /// (inference mode, no gradient tracking).
    pub fn detached(&self) -> Self {
        let device = self.states[0].hidden.device();

        // Recreate tensors from data to break computation graph
        let states = self.states
            .iter()
            .map(|state| {
                // Extract data and recreate fresh tensors (no gradient history)
                let h_data = state.hidden.clone().into_data();
                let h_fresh = Tensor::from_data(h_data, &device);

                if self.has_cell {
                    let c_data = state.cell.as_ref().unwrap().clone().into_data();
                    let c_fresh = Tensor::from_data(c_data, &device);
                    HiddenState::lstm(h_fresh, c_fresh)
                } else {
                    HiddenState::gru(h_fresh)
                }
            })
            .collect();

        Self {
            states,
            hidden_size: self.hidden_size,
            has_cell: self.has_cell,
        }
    }
}

impl<B: Backend> Default for RecurrentHidden<B> {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            hidden_size: 0,
            has_cell: false,
        }
    }
}

impl<B: Backend> HiddenStateType<B> for RecurrentHidden<B> {
    fn initial(n_envs: usize, device: &B::Device, config: &HiddenConfig) -> Self {
        Self::new(n_envs, config.hidden_size, config.has_cell, device)
    }

    fn is_stateful() -> bool {
        true
    }

    fn reset(&mut self, env_idx: usize, device: &B::Device) {
        // Delegate to the inherent method
        RecurrentHidden::reset(self, env_idx, device);
    }

    fn reset_indices(&mut self, indices: &[usize], device: &B::Device, _config: &HiddenConfig) {
        for &idx in indices {
            self.reset(idx, device);
        }
    }

    fn get_env_vec(&self, env_idx: usize) -> Vec<f32> {
        self.get(env_idx).to_vec()
    }

    fn to_vec(&self) -> Vec<f32> {
        self.states.iter().flat_map(|s| s.to_vec()).collect()
    }

    fn from_vec(data: &[f32], n_envs: usize, device: &B::Device, config: &HiddenConfig) -> Self {
        let per_env_size = Self::per_env_size(config);
        let states = (0..n_envs)
            .map(|i| {
                let start = i * per_env_size;
                let end = start + per_env_size;
                HiddenState::from_vec(
                    &data[start..end],
                    1,
                    config.hidden_size,
                    config.has_cell,
                    device,
                )
            })
            .collect();

        Self {
            states,
            hidden_size: config.hidden_size,
            has_cell: config.has_cell,
        }
    }

    fn per_env_size(config: &HiddenConfig) -> usize {
        if config.has_cell {
            config.hidden_size * 2 // LSTM: h + c
        } else {
            config.hidden_size // GRU: just h
        }
    }
}

// ============================================================================
// TemporalPolicy Trait
// ============================================================================

/// Main trait for temporal policy abstraction.
///
/// This trait abstracts over the temporal structure of policies:
/// - [`FeedForward`]: Stateless, no hidden state management
/// - [`Recurrent`]: Maintains per-environment hidden states
///
/// The trait provides:
/// - Configuration for hidden state creation
/// - Methods for initializing and managing hidden states
pub trait TemporalPolicy<B: Backend>: Clone + Send + Sync + 'static {
    /// The hidden state type for this temporal policy.
    type Hidden: HiddenStateType<B>;

    /// Whether this policy is recurrent (maintains state across timesteps).
    fn is_recurrent() -> bool {
        Self::Hidden::is_stateful()
    }

    /// Get the hidden state configuration.
    fn hidden_config(&self) -> HiddenConfig;

    /// Create initial hidden state for given number of environments.
    fn initial_hidden(&self, n_envs: usize, device: &B::Device) -> Self::Hidden {
        Self::Hidden::initial(n_envs, device, &self.hidden_config())
    }

    /// Handle terminal states by resetting hidden at given indices.
    fn handle_terminals(
        &self,
        hidden: &mut Self::Hidden,
        terminal_indices: &[usize],
        device: &B::Device,
    ) {
        if Self::Hidden::is_stateful() && !terminal_indices.is_empty() {
            hidden.reset_indices(terminal_indices, device, &self.hidden_config());
        }
    }
}

// ============================================================================
// FeedForward Implementation
// ============================================================================

/// Feed-forward temporal policy (no hidden state).
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedForward;

impl FeedForward {
    /// Create a new feed-forward temporal policy.
    pub fn new() -> Self {
        Self
    }
}

impl<B: Backend> TemporalPolicy<B> for FeedForward {
    type Hidden = ();

    fn hidden_config(&self) -> HiddenConfig {
        HiddenConfig::none()
    }
}

// ============================================================================
// Recurrent Implementation
// ============================================================================

/// Recurrent temporal policy (LSTM/GRU hidden state).
#[derive(Debug, Clone)]
pub struct Recurrent {
    /// Size of hidden state.
    pub hidden_size: usize,
    /// Whether to use LSTM (with cell state) or GRU.
    pub has_cell: bool,
}

impl Recurrent {
    /// Create LSTM-based recurrent policy.
    pub fn lstm(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            has_cell: true,
        }
    }

    /// Create GRU-based recurrent policy.
    pub fn gru(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            has_cell: false,
        }
    }
}

impl<B: Backend> TemporalPolicy<B> for Recurrent {
    type Hidden = RecurrentHidden<B>;

    fn hidden_config(&self) -> HiddenConfig {
        HiddenConfig {
            hidden_size: self.hidden_size,
            has_cell: self.has_cell,
        }
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
    fn test_feed_forward() {
        let policy = FeedForward::new();
        let device = <B as Backend>::Device::default();

        assert!(!<FeedForward as TemporalPolicy<B>>::is_recurrent());

        let hidden: () = <FeedForward as TemporalPolicy<B>>::initial_hidden(&policy, 4, &device);
        assert_eq!(hidden, ());
    }

    #[test]
    fn test_recurrent_lstm() {
        let policy = Recurrent::lstm(64);
        let device = <B as Backend>::Device::default();

        assert!(<Recurrent as TemporalPolicy<B>>::is_recurrent());
        assert_eq!(policy.hidden_size, 64);
        assert!(policy.has_cell);

        let hidden: RecurrentHidden<B> =
            <Recurrent as TemporalPolicy<B>>::initial_hidden(&policy, 4, &device);
        assert_eq!(hidden.states.len(), 4);
        assert!(hidden.states[0].cell.is_some());
    }

    #[test]
    fn test_recurrent_gru() {
        let policy = Recurrent::gru(32);
        let device = <B as Backend>::Device::default();

        assert!(<Recurrent as TemporalPolicy<B>>::is_recurrent());
        assert_eq!(policy.hidden_size, 32);
        assert!(!policy.has_cell);

        let hidden: RecurrentHidden<B> =
            <Recurrent as TemporalPolicy<B>>::initial_hidden(&policy, 2, &device);
        assert_eq!(hidden.states.len(), 2);
        assert!(hidden.states[0].cell.is_none());
    }

    #[test]
    fn test_recurrent_hidden_reset() {
        let policy = Recurrent::lstm(16);
        let device = <B as Backend>::Device::default();

        let mut hidden: RecurrentHidden<B> =
            <Recurrent as TemporalPolicy<B>>::initial_hidden(&policy, 4, &device);

        // Modify one state
        hidden.states[1] = HiddenState::lstm(
            Tensor::ones([1, 16], &device),
            Tensor::ones([1, 16], &device),
        );

        // Reset indices 1 and 3
        <Recurrent as TemporalPolicy<B>>::handle_terminals(&policy, &mut hidden, &[1, 3], &device);

        // Check that index 1 is back to zeros
        let h1_data = hidden.states[1].hidden.clone().into_data();
        let h1_sum: f32 = h1_data.as_slice().unwrap().iter().sum();
        assert_eq!(h1_sum, 0.0);
    }

    #[test]
    fn test_recurrent_hidden_to_batch() {
        let policy = Recurrent::lstm(8);
        let device = <B as Backend>::Device::default();

        let hidden: RecurrentHidden<B> =
            <Recurrent as TemporalPolicy<B>>::initial_hidden(&policy, 3, &device);
        let (h_batch, c_batch) = hidden.to_batch();

        assert_eq!(h_batch.dims(), [3, 8]);
        assert!(c_batch.is_some());
        assert_eq!(c_batch.unwrap().dims(), [3, 8]);
    }

    #[test]
    fn test_hidden_state_serialization() {
        let policy = Recurrent::lstm(4);
        let device = <B as Backend>::Device::default();

        let hidden: RecurrentHidden<B> =
            <Recurrent as TemporalPolicy<B>>::initial_hidden(&policy, 2, &device);

        // Serialize
        let vec = hidden.to_vec();
        assert_eq!(vec.len(), 2 * 4 * 2); // 2 envs * 4 hidden * 2 (h + c)

        // Deserialize
        let config = <Recurrent as TemporalPolicy<B>>::hidden_config(&policy);
        let restored = RecurrentHidden::<B>::from_vec(&vec, 2, &device, &config);

        assert_eq!(restored.states.len(), 2);
        assert_eq!(restored.hidden_size, 4);
        assert!(restored.has_cell);
    }
}
