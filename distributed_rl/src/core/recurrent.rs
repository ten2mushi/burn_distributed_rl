//! Recurrent cell abstractions for recurrent PPO.
//!
//! Provides a generic trait for recurrent cells (LSTM, GRU) that can be used
//! with the recurrent PPO runner. Users can implement custom RNN architectures
//! by implementing the `RecurrentCell` trait.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, LstmState};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

// ============================================================================
// Generic Recurrent Cell Trait
// ============================================================================

/// Hidden state for recurrent cells.
///
/// Generic representation that can store LSTM (h, c), GRU (h), or custom states.
#[derive(Debug, Clone)]
pub struct HiddenState<B: Backend> {
    /// Primary hidden state (h for both LSTM and GRU)
    pub hidden: Tensor<B, 2>,
    /// Optional cell state (c for LSTM, None for GRU)
    pub cell: Option<Tensor<B, 2>>,
}

impl<B: Backend> HiddenState<B> {
    /// Create LSTM-style state with both hidden and cell.
    pub fn lstm(hidden: Tensor<B, 2>, cell: Tensor<B, 2>) -> Self {
        Self {
            hidden,
            cell: Some(cell),
        }
    }

    /// Create GRU-style state with only hidden.
    pub fn gru(hidden: Tensor<B, 2>) -> Self {
        Self { hidden, cell: None }
    }

    /// Get the hidden state tensor.
    pub fn h(&self) -> &Tensor<B, 2> {
        &self.hidden
    }

    /// Get the cell state tensor (if present).
    pub fn c(&self) -> Option<&Tensor<B, 2>> {
        self.cell.as_ref()
    }

    /// Total number of floats needed to store this state (for serialization).
    pub fn size(&self) -> usize {
        let h_size = self.hidden.dims()[0] * self.hidden.dims()[1];
        let c_size = self.cell.as_ref().map(|c| c.dims()[0] * c.dims()[1]).unwrap_or(0);
        h_size + c_size
    }

    /// Flatten state to a vector for storage.
    pub fn to_vec(&self) -> Vec<f32> {
        let h_data = self.hidden.clone().into_data();
        let h_slice: &[f32] = h_data.as_slice().unwrap();
        let mut result: Vec<f32> = h_slice.to_vec();

        if let Some(ref c) = self.cell {
            let c_data = c.clone().into_data();
            let c_slice: &[f32] = c_data.as_slice().unwrap();
            result.extend_from_slice(c_slice);
        }

        result
    }

    /// Create state from flattened vector.
    pub fn from_vec(
        data: &[f32],
        batch_size: usize,
        hidden_size: usize,
        has_cell: bool,
        device: &B::Device,
    ) -> Self {
        let h_len = batch_size * hidden_size;
        let h_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(&data[..h_len], device)
            .reshape([batch_size, hidden_size]);

        let cell = if has_cell {
            let c_tensor: Tensor<B, 2> = Tensor::<B, 1>::from_floats(&data[h_len..], device)
                .reshape([batch_size, hidden_size]);
            Some(c_tensor)
        } else {
            None
        };

        Self {
            hidden: h_tensor,
            cell,
        }
    }
}

/// Trait for recurrent cells that can be used with recurrent PPO.
///
/// This trait abstracts over different RNN architectures (LSTM, GRU, custom)
/// allowing users to plug in their own recurrent architectures.
pub trait RecurrentCell<B: Backend>: Module<B> + Clone + Send + 'static {
    /// Process a single timestep.
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, input_size]
    /// * `state` - Current hidden state
    ///
    /// # Returns
    /// * `output` - Output tensor [batch, hidden_size]
    /// * `new_state` - Updated hidden state
    fn step(&self, input: Tensor<B, 2>, state: &HiddenState<B>) -> (Tensor<B, 2>, HiddenState<B>);

    /// Create initial zero hidden state for given batch size.
    fn initial_state(&self, batch_size: usize, device: &B::Device) -> HiddenState<B>;

    /// Hidden size (output dimension).
    fn hidden_size(&self) -> usize;

    /// Input size.
    fn input_size(&self) -> usize;

    /// Whether this cell uses a cell state (LSTM) or just hidden (GRU).
    fn has_cell_state(&self) -> bool;

    /// Process a sequence of inputs.
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor [batch, seq_len, input_size]
    /// * `initial_state` - Initial hidden state
    ///
    /// # Returns
    /// * `outputs` - Output tensor [batch, seq_len, hidden_size]
    /// * `final_state` - Final hidden state
    fn forward_sequence(
        &self,
        input_seq: Tensor<B, 3>,
        initial_state: HiddenState<B>,
    ) -> (Tensor<B, 3>, HiddenState<B>) {
        let [batch_size, seq_len, _] = input_seq.dims();
        let _device = input_seq.device();

        let mut outputs = Vec::with_capacity(seq_len);
        let mut state = initial_state;

        for t in 0..seq_len {
            // Extract timestep [batch, input_size]
            let input_t = input_seq
                .clone()
                .slice([0..batch_size, t..(t + 1), 0..self.input_size()])
                .reshape([batch_size, self.input_size()]);

            let (output_t, new_state) = self.step(input_t, &state);
            outputs.push(output_t.reshape([batch_size, 1, self.hidden_size()]));
            state = new_state;
        }

        // Stack outputs along sequence dimension
        let output_tensor = Tensor::cat(outputs, 1);
        (output_tensor, state)
    }
}

// ============================================================================
// LSTM Wrapper
// ============================================================================

/// Configuration for LSTM cell wrapper.
#[derive(Debug, Clone)]
pub struct LstmCellConfig {
    /// Input feature size.
    pub d_input: usize,
    /// Hidden state size.
    pub d_hidden: usize,
    /// Whether to use bias.
    pub bias: bool,
}

impl LstmCellConfig {
    /// Create new LSTM config.
    pub fn new(d_input: usize, d_hidden: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            bias: true,
        }
    }

    /// Set bias option.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Initialize the LSTM cell.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmCellWrapper<B> {
        let lstm = LstmConfig::new(self.d_input, self.d_hidden, self.bias).init(device);
        LstmCellWrapper {
            lstm,
            d_input: self.d_input,
            d_hidden: self.d_hidden,
        }
    }
}

/// Wrapper around Burn's Lstm for single-step operation.
#[derive(Module, Debug)]
pub struct LstmCellWrapper<B: Backend> {
    lstm: Lstm<B>,
    #[module(skip)]
    d_input: usize,
    #[module(skip)]
    d_hidden: usize,
}

impl<B: Backend> RecurrentCell<B> for LstmCellWrapper<B> {
    fn step(&self, input: Tensor<B, 2>, state: &HiddenState<B>) -> (Tensor<B, 2>, HiddenState<B>) {
        let [batch_size, _] = input.dims();

        // Convert to sequence format [batch, 1, input]
        let input_seq = input.reshape([batch_size, 1, self.d_input]);

        // Convert HiddenState to LstmState
        let lstm_state = LstmState::new(
            state.cell.clone().expect("LSTM requires cell state"),
            state.hidden.clone(),
        );

        // Forward through LSTM
        let (output_seq, new_lstm_state) = self.lstm.forward(input_seq, Some(lstm_state));

        // Extract output [batch, 1, hidden] -> [batch, hidden]
        let output = output_seq.reshape([batch_size, self.d_hidden]);

        // Convert back to HiddenState
        let new_state = HiddenState::lstm(new_lstm_state.hidden, new_lstm_state.cell);

        (output, new_state)
    }

    fn initial_state(&self, batch_size: usize, device: &B::Device) -> HiddenState<B> {
        let hidden = Tensor::zeros([batch_size, self.d_hidden], device);
        let cell = Tensor::zeros([batch_size, self.d_hidden], device);
        HiddenState::lstm(hidden, cell)
    }

    fn hidden_size(&self) -> usize {
        self.d_hidden
    }

    fn input_size(&self) -> usize {
        self.d_input
    }

    fn has_cell_state(&self) -> bool {
        true
    }
}

// ============================================================================
// GRU Implementation (from scratch since Burn doesn't have it)
// ============================================================================

/// Configuration for GRU cell.
#[derive(Debug, Clone)]
pub struct GruCellConfig {
    /// Input feature size.
    pub d_input: usize,
    /// Hidden state size.
    pub d_hidden: usize,
    /// Whether to use bias.
    pub bias: bool,
}

impl GruCellConfig {
    /// Create new GRU config.
    pub fn new(d_input: usize, d_hidden: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            bias: true,
        }
    }

    /// Set bias option.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Initialize the GRU cell.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GruCell<B> {
        // Reset gate: r = σ(W_ir * x + W_hr * h + b_r)
        let reset_input = LinearConfig::new(self.d_input, self.d_hidden)
            .with_bias(self.bias)
            .init(device);
        let reset_hidden = LinearConfig::new(self.d_hidden, self.d_hidden)
            .with_bias(false)
            .init(device);

        // Update gate: z = σ(W_iz * x + W_hz * h + b_z)
        let update_input = LinearConfig::new(self.d_input, self.d_hidden)
            .with_bias(self.bias)
            .init(device);
        let update_hidden = LinearConfig::new(self.d_hidden, self.d_hidden)
            .with_bias(false)
            .init(device);

        // Candidate hidden: n = tanh(W_in * x + r ⊙ (W_hn * h) + b_n)
        let candidate_input = LinearConfig::new(self.d_input, self.d_hidden)
            .with_bias(self.bias)
            .init(device);
        let candidate_hidden = LinearConfig::new(self.d_hidden, self.d_hidden)
            .with_bias(false)
            .init(device);

        GruCell {
            reset_input,
            reset_hidden,
            update_input,
            update_hidden,
            candidate_input,
            candidate_hidden,
            d_input: self.d_input,
            d_hidden: self.d_hidden,
        }
    }
}

/// GRU cell implementation.
///
/// GRU equations:
/// - r = σ(W_ir * x + W_hr * h + b_r)     (reset gate)
/// - z = σ(W_iz * x + W_hz * h + b_z)     (update gate)
/// - n = tanh(W_in * x + r ⊙ (W_hn * h))  (candidate hidden)
/// - h' = (1 - z) ⊙ n + z ⊙ h             (new hidden)
#[derive(Module, Debug)]
pub struct GruCell<B: Backend> {
    reset_input: Linear<B>,
    reset_hidden: Linear<B>,
    update_input: Linear<B>,
    update_hidden: Linear<B>,
    candidate_input: Linear<B>,
    candidate_hidden: Linear<B>,
    #[module(skip)]
    d_input: usize,
    #[module(skip)]
    d_hidden: usize,
}

impl<B: Backend> RecurrentCell<B> for GruCell<B> {
    fn step(&self, input: Tensor<B, 2>, state: &HiddenState<B>) -> (Tensor<B, 2>, HiddenState<B>) {
        let h = state.h();

        // Reset gate: r = σ(W_ir * x + W_hr * h)
        let r = sigmoid(
            self.reset_input.forward(input.clone()) + self.reset_hidden.forward(h.clone()),
        );

        // Update gate: z = σ(W_iz * x + W_hz * h)
        let z = sigmoid(
            self.update_input.forward(input.clone()) + self.update_hidden.forward(h.clone()),
        );

        // Candidate hidden: n = tanh(W_in * x + r ⊙ (W_hn * h))
        let n = (self.candidate_input.forward(input)
            + r * self.candidate_hidden.forward(h.clone()))
        .tanh();

        // New hidden: h' = (1 - z) ⊙ n + z ⊙ h
        let ones = Tensor::ones_like(&z);
        let new_h = (ones - z.clone()) * n + z * h.clone();

        (new_h.clone(), HiddenState::gru(new_h))
    }

    fn initial_state(&self, batch_size: usize, device: &B::Device) -> HiddenState<B> {
        let hidden = Tensor::zeros([batch_size, self.d_hidden], device);
        HiddenState::gru(hidden)
    }

    fn hidden_size(&self) -> usize {
        self.d_hidden
    }

    fn input_size(&self) -> usize {
        self.d_input
    }

    fn has_cell_state(&self) -> bool {
        false
    }
}

// ============================================================================
// Recurrent Cell Enum (for dynamic dispatch)
// ============================================================================

/// Enum for selecting recurrent cell type in configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentCellType {
    /// LSTM cell with hidden + cell state.
    Lstm,
    /// GRU cell with hidden state only.
    Gru,
}

impl Default for RecurrentCellType {
    fn default() -> Self {
        RecurrentCellType::Lstm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_lstm_cell_step() {
        let device = Default::default();
        let config = LstmCellConfig::new(4, 8);
        let cell = config.init::<B>(&device);

        let batch_size = 2;
        let input: Tensor<B, 2> = Tensor::zeros([batch_size, 4], &device);
        let state = cell.initial_state(batch_size, &device);

        let (output, new_state) = cell.step(input, &state);

        assert_eq!(output.dims(), [batch_size, 8]);
        assert_eq!(new_state.hidden.dims(), [batch_size, 8]);
        assert!(new_state.cell.is_some());
    }

    #[test]
    fn test_gru_cell_step() {
        let device = Default::default();
        let config = GruCellConfig::new(4, 8);
        let cell = config.init::<B>(&device);

        let batch_size = 2;
        let input: Tensor<B, 2> = Tensor::zeros([batch_size, 4], &device);
        let state = cell.initial_state(batch_size, &device);

        let (output, new_state) = cell.step(input, &state);

        assert_eq!(output.dims(), [batch_size, 8]);
        assert_eq!(new_state.hidden.dims(), [batch_size, 8]);
        assert!(new_state.cell.is_none());
    }

    #[test]
    fn test_hidden_state_serialization() {
        let device: <B as Backend>::Device = Default::default();
        let batch_size = 2;
        let hidden_size = 4;

        // Test LSTM state
        let h: Tensor<B, 2> = Tensor::ones([batch_size, hidden_size], &device);
        let c: Tensor<B, 2> = Tensor::ones([batch_size, hidden_size], &device) * 2.0;
        let state = HiddenState::lstm(h, c);

        let flat = state.to_vec();
        assert_eq!(flat.len(), batch_size * hidden_size * 2);

        let restored = HiddenState::<B>::from_vec(&flat, batch_size, hidden_size, true, &device);
        assert!(restored.cell.is_some());
    }
}
