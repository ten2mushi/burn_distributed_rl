//! Multi-head attention mechanism for neural networks.
//!
//! Attention mechanisms allow networks to focus on relevant parts of the input,
//! useful for relational reasoning and handling variable-length sequences.
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::nn::{MultiHeadAttention, MultiHeadAttentionConfig};
//!
//! let config = MultiHeadAttentionConfig::new(64, 8);  // d_model=64, n_heads=8
//! let attention: MultiHeadAttention<Backend> = config.init(&device);
//!
//! // Self-attention
//! let output = attention.forward(query, key, value, None);
//!
//! // With attention mask
//! let output = attention.forward(query, key, value, Some(mask));
//! ```
//!
//! # Applications in RL
//!
//! - Variable-length observation sequences
//! - Relational reasoning between entities
//! - Long-range temporal dependencies (alternative to LSTM)
//! - Transformer-based policies

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::Distribution;

/// Configuration for MultiHeadAttention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Model dimension (must be divisible by n_heads)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dropout probability (currently unused, placeholder for future)
    pub dropout: f64,
    /// Whether to use bias in projections
    pub bias: bool,
}

impl MultiHeadAttentionConfig {
    /// Create a new configuration.
    ///
    /// # Arguments
    /// * `d_model` - Total model dimension
    /// * `n_heads` - Number of attention heads (d_model must be divisible by n_heads)
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            d_model,
            n_heads
        );
        Self {
            d_model,
            n_heads,
            dropout: 0.0,
            bias: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set whether to use bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Initialize the attention module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let d_k = self.d_model / self.n_heads;
        let scale = (d_k as f64).sqrt();

        // Xavier initialization for projection matrices
        let init_std = (2.0 / (self.d_model + self.d_model) as f64).sqrt() as f32;

        let w_q = Tensor::<B, 2>::random(
            [self.d_model, self.d_model],
            Distribution::Normal(0.0, init_std as f64),
            device,
        );
        let w_k = Tensor::<B, 2>::random(
            [self.d_model, self.d_model],
            Distribution::Normal(0.0, init_std as f64),
            device,
        );
        let w_v = Tensor::<B, 2>::random(
            [self.d_model, self.d_model],
            Distribution::Normal(0.0, init_std as f64),
            device,
        );
        let w_o = Tensor::<B, 2>::random(
            [self.d_model, self.d_model],
            Distribution::Normal(0.0, init_std as f64),
            device,
        );

        let (b_q, b_k, b_v, b_o) = if self.bias {
            (
                Some(Param::from_tensor(Tensor::zeros([self.d_model], device))),
                Some(Param::from_tensor(Tensor::zeros([self.d_model], device))),
                Some(Param::from_tensor(Tensor::zeros([self.d_model], device))),
                Some(Param::from_tensor(Tensor::zeros([self.d_model], device))),
            )
        } else {
            (None, None, None, None)
        };

        MultiHeadAttention {
            w_q: Param::from_tensor(w_q),
            w_k: Param::from_tensor(w_k),
            w_v: Param::from_tensor(w_v),
            w_o: Param::from_tensor(w_o),
            b_q,
            b_k,
            b_v,
            b_o,
            n_heads: self.n_heads,
            d_model: self.d_model,
            d_k,
            scale: scale as f32,
        }
    }
}

/// Multi-head attention mechanism.
///
/// Computes scaled dot-product attention with multiple heads:
///
/// ```text
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
/// where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
/// ```
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection weight [d_model, d_model]
    w_q: Param<Tensor<B, 2>>,
    /// Key projection weight [d_model, d_model]
    w_k: Param<Tensor<B, 2>>,
    /// Value projection weight [d_model, d_model]
    w_v: Param<Tensor<B, 2>>,
    /// Output projection weight [d_model, d_model]
    w_o: Param<Tensor<B, 2>>,
    /// Query projection bias
    b_q: Option<Param<Tensor<B, 1>>>,
    /// Key projection bias
    b_k: Option<Param<Tensor<B, 1>>>,
    /// Value projection bias
    b_v: Option<Param<Tensor<B, 1>>>,
    /// Output projection bias
    b_o: Option<Param<Tensor<B, 1>>>,
    /// Number of attention heads
    n_heads: usize,
    /// Model dimension
    d_model: usize,
    /// Per-head dimension
    d_k: usize,
    /// Scaling factor (1/sqrt(d_k))
    scale: f32,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass for multi-head attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch, seq_q, d_model]
    /// * `key` - Key tensor of shape [batch, seq_k, d_model]
    /// * `value` - Value tensor of shape [batch, seq_k, d_model]
    /// * `mask` - Optional attention mask of shape [batch, seq_q, seq_k]
    ///           Values of -inf mask out attention
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_q, d_model]
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_q, _] = query.dims();
        let [_, seq_k, _] = key.dims();

        // Linear projections
        let q = self.project(&query, &self.w_q, &self.b_q);
        let k = self.project(&key, &self.w_k, &self.b_k);
        let v = self.project(&value, &self.w_v, &self.b_v);

        // Reshape for multi-head attention: [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
        let q = q.reshape([batch_size, seq_q, self.n_heads, self.d_k])
            .swap_dims(1, 2);
        let k = k.reshape([batch_size, seq_k, self.n_heads, self.d_k])
            .swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_k, self.n_heads, self.d_k])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        // scores: [batch, n_heads, seq_q, seq_k]
        let scores = q.matmul(k.swap_dims(2, 3)) / self.scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            // Expand mask for heads: [batch, seq_q, seq_k] -> [batch, n_heads, seq_q, seq_k]
            let mask = mask.unsqueeze_dim(1);
            scores + mask
        } else {
            scores
        };

        // Softmax over key dimension
        let attention_weights = burn::tensor::activation::softmax(scores, 3);

        // Apply attention to values
        // [batch, n_heads, seq_q, d_k]
        let attended = attention_weights.matmul(v);

        // Reshape back: [batch, n_heads, seq_q, d_k] -> [batch, seq_q, d_model]
        let attended = attended.swap_dims(1, 2)
            .reshape([batch_size, seq_q, self.d_model]);

        // Output projection
        self.project(&attended, &self.w_o, &self.b_o)
    }

    /// Self-attention convenience method.
    ///
    /// Applies attention where query = key = value.
    pub fn self_attention(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        self.forward(x.clone(), x.clone(), x, mask)
    }

    /// Project input through a linear layer.
    fn project(
        &self,
        x: &Tensor<B, 3>,
        weight: &Param<Tensor<B, 2>>,
        bias: &Option<Param<Tensor<B, 1>>>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();

        // Reshape to 2D for matmul: [batch * seq, d_model]
        let x_flat = x.clone().reshape([batch * seq, self.d_model]);

        // Linear: x @ W^T + b
        let projected = x_flat.matmul(weight.val().transpose());

        let projected = match bias {
            Some(b) => projected + b.val().unsqueeze_dim(0),
            None => projected,
        };

        // Reshape back to 3D: [batch, seq, d_model]
        projected.reshape([batch, seq, self.d_model])
    }

    /// Get model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get number of heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn get_device() -> <TestBackend as Backend>::Device {
        Default::default()
    }

    #[test]
    fn test_attention_forward() {
        let device = get_device();
        let config = MultiHeadAttentionConfig::new(64, 8);
        let attention: MultiHeadAttention<TestBackend> = config.init(&device);

        let batch_size = 2;
        let seq_len = 10;

        let query = Tensor::random([batch_size, seq_len, 64], Distribution::Normal(0.0, 1.0), &device);
        let key = Tensor::random([batch_size, seq_len, 64], Distribution::Normal(0.0, 1.0), &device);
        let value = Tensor::random([batch_size, seq_len, 64], Distribution::Normal(0.0, 1.0), &device);

        let output = attention.forward(query, key, value, None);

        assert_eq!(output.dims(), [batch_size, seq_len, 64]);
    }

    #[test]
    fn test_self_attention() {
        let device = get_device();
        let config = MultiHeadAttentionConfig::new(32, 4);
        let attention: MultiHeadAttention<TestBackend> = config.init(&device);

        let x = Tensor::random([1, 5, 32], Distribution::Normal(0.0, 1.0), &device);
        let output = attention.self_attention(x, None);

        assert_eq!(output.dims(), [1, 5, 32]);
    }

    #[test]
    fn test_cross_attention() {
        let device = get_device();
        let config = MultiHeadAttentionConfig::new(64, 8);
        let attention: MultiHeadAttention<TestBackend> = config.init(&device);

        // Cross-attention: query from one sequence, key/value from another
        let query = Tensor::random([2, 4, 64], Distribution::Normal(0.0, 1.0), &device);
        let key = Tensor::random([2, 10, 64], Distribution::Normal(0.0, 1.0), &device);
        let value = Tensor::random([2, 10, 64], Distribution::Normal(0.0, 1.0), &device);

        let output = attention.forward(query, key, value, None);

        // Output should have query's sequence length
        assert_eq!(output.dims(), [2, 4, 64]);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_invalid_config() {
        // Should panic: 65 is not divisible by 8
        let _config = MultiHeadAttentionConfig::new(65, 8);
    }
}
