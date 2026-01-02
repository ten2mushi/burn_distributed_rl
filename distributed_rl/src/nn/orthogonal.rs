//! Orthogonal initialization for neural networks.
//!
//! Orthogonal initialization produces weight matrices with orthogonal columns,
//! which helps preserve gradient magnitudes during backpropagation.
//!
//! # Theory
//!
//! For an orthogonal matrix Q:
//! - All singular values = 1
//! - ||Qx|| = ||x|| (norm preserved)
//! - Gradients neither explode nor vanish
//!
//! Compared to Xavier/Glorot initialization:
//! - Xavier preserves variance on average
//! - Orthogonal preserves norm exactly
//! - Better for deep networks and RNNs
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
//!
//! // Create a linear layer with orthogonal initialization
//! let config = OrthogonalLinearConfig::new(64, 64)
//!     .with_gain(1.41)  // sqrt(2) for ReLU
//!     .with_bias(true);
//! let linear: OrthogonalLinear<Backend> = config.init(&device);
//!
//! let output = linear.forward(input);
//! ```
//!
//! # Gain Values
//!
//! - 1.0: Linear/Identity activations
//! - sqrt(2) ≈ 1.41: ReLU activations
//! - sqrt(2 / (1 + a²)): Leaky ReLU with slope a
//! - 5/3 ≈ 1.67: Tanh activations

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::Distribution;

/// Configuration for OrthogonalLinear layer.
#[derive(Debug, Clone)]
pub struct OrthogonalLinearConfig {
    /// Number of input features.
    pub d_input: usize,
    /// Number of output features.
    pub d_output: usize,
    /// Gain factor for scaling the orthogonal weights.
    /// Default: 1.0 (use sqrt(2) for ReLU, ~1.67 for tanh)
    pub gain: f64,
    /// Whether to include a bias term.
    pub bias: bool,
}

impl OrthogonalLinearConfig {
    /// Create a new configuration.
    pub fn new(d_input: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_output,
            gain: 1.0,
            bias: true,
        }
    }

    /// Set the gain factor.
    ///
    /// Common values:
    /// - 1.0 for linear/sigmoid activations
    /// - sqrt(2) ≈ 1.41 for ReLU
    /// - 5/3 ≈ 1.67 for tanh
    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Set whether to include bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Initialize the layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> OrthogonalLinear<B> {
        let weight = generate_orthogonal_weights::<B>(
            self.d_output,
            self.d_input,
            self.gain,
            device,
        );

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.d_output], device)))
        } else {
            None
        };

        OrthogonalLinear {
            weight: Param::from_tensor(weight),
            bias,
            d_input: self.d_input,
            d_output: self.d_output,
        }
    }
}

/// Linear layer with orthogonal initialization.
///
/// Functionally equivalent to Burn's Linear layer, but initialized with
/// orthogonal weights for better gradient flow in deep networks.
#[derive(Module, Debug)]
pub struct OrthogonalLinear<B: Backend> {
    /// Weight matrix of shape [d_output, d_input]
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Input dimension (for reference)
    d_input: usize,
    /// Output dimension (for reference)
    d_output: usize,
}

impl<B: Backend> OrthogonalLinear<B> {
    /// Forward pass for 2D input: y = xW^T + b
    ///
    /// # Arguments
    /// * `input` - Tensor of shape [batch_size, d_input]
    ///
    /// # Returns
    /// Tensor of shape [batch_size, d_output]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Linear: output = input @ weight^T + bias
        let output = input.matmul(self.weight.val().transpose());

        match &self.bias {
            Some(bias) => output + bias.val().unsqueeze_dim(0),
            None => output,
        }
    }

    /// Forward pass for 3D input (batched sequences): y = xW^T + b
    ///
    /// # Arguments
    /// * `input` - Tensor of shape [batch_size, seq_len, d_input]
    ///
    /// # Returns
    /// Tensor of shape [batch_size, seq_len, d_output]
    pub fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _d_in] = input.dims();

        // Reshape to 2D: [batch * seq, d_input]
        let input_2d = input.reshape([batch * seq, self.d_input]);

        // Forward through linear
        let output_2d = self.forward(input_2d);

        // Reshape back to 3D: [batch, seq, d_output]
        output_2d.reshape([batch, seq, self.d_output])
    }

    /// Get input dimension.
    pub fn d_input(&self) -> usize {
        self.d_input
    }

    /// Get output dimension.
    pub fn d_output(&self) -> usize {
        self.d_output
    }

    /// Reinitialize weights with new orthogonal values.
    pub fn reinit(&mut self, gain: f64, device: &B::Device) {
        let new_weight = generate_orthogonal_weights::<B>(
            self.d_output,
            self.d_input,
            gain,
            device,
        );
        self.weight = Param::from_tensor(new_weight);

        if let Some(ref mut bias) = self.bias {
            *bias = Param::from_tensor(Tensor::zeros([self.d_output], device));
        }
    }
}

/// Generate orthogonal weight matrix using Gram-Schmidt process.
///
/// Since Burn doesn't have built-in QR decomposition, we use an iterative
/// Gram-Schmidt orthogonalization approach.
///
/// # Arguments
/// * `rows` - Number of rows (output features)
/// * `cols` - Number of columns (input features)
/// * `gain` - Scaling factor for the final weights
/// * `device` - Device to create the tensor on
///
/// # Returns
/// Orthogonal weight tensor of shape [rows, cols]
pub fn generate_orthogonal_weights<B: Backend>(
    rows: usize,
    cols: usize,
    gain: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Start with random matrix
    let random = Tensor::<B, 2>::random([rows, cols], Distribution::Normal(0.0, 1.0), device);

    // For square or tall matrices, orthogonalize columns
    // For wide matrices, orthogonalize rows
    let (orthogonal, _transposed) = if rows >= cols {
        (gram_schmidt_columns::<B>(random, device), false)
    } else {
        // Transpose, orthogonalize columns, transpose back
        let transposed = random.transpose();
        let ortho = gram_schmidt_columns::<B>(transposed, device);
        (ortho.transpose(), true)
    };

    // Apply gain
    orthogonal * (gain as f32)
}

/// Gram-Schmidt orthogonalization of columns.
///
/// Makes each column orthogonal to all previous columns while preserving
/// the span of the matrix.
fn gram_schmidt_columns<B: Backend>(
    matrix: Tensor<B, 2>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [rows, cols] = matrix.dims();

    // Extract columns as vectors
    let mut columns: Vec<Tensor<B, 1>> = (0..cols)
        .map(|i| matrix.clone().slice([0..rows, i..i + 1]).squeeze::<1>())
        .collect();

    // Orthogonalize each column against all previous ones
    for i in 0..cols {
        let mut vi = columns[i].clone();

        // Subtract projections onto previous orthogonal vectors
        for j in 0..i {
            let vj = &columns[j];
            // proj = (vi · vj) / (vj · vj) * vj
            let dot_ij = dot_product::<B>(&vi, vj);
            let dot_jj = dot_product::<B>(vj, vj);

            // Avoid division by zero
            let scale = dot_ij / (dot_jj + 1e-10);
            let projection = vj.clone() * scale;
            vi = vi - projection;
        }

        // Normalize to unit length
        let norm = vi.clone().powf_scalar(2.0).sum().sqrt();
        let norm_scalar: f32 = norm.clone().into_scalar().elem();

        if norm_scalar > 1e-10 {
            columns[i] = vi / norm;
        } else {
            // If the vector is too small (linearly dependent), use random
            columns[i] = Tensor::random([rows], Distribution::Normal(0.0, 1.0), device);
            let norm = columns[i].clone().powf_scalar(2.0).sum().sqrt();
            columns[i] = columns[i].clone() / norm;
        }
    }

    // Stack columns back into matrix
    let stacked: Vec<Tensor<B, 2>> = columns
        .into_iter()
        .map(|c| c.unsqueeze_dim(1))
        .collect();

    Tensor::cat(stacked, 1)
}

/// Compute dot product of two 1D tensors.
fn dot_product<B: Backend>(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
    let product = a.clone() * b.clone();
    let sum = product.sum();
    sum.into_scalar().elem()
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
    fn test_orthogonal_linear_forward() {
        let device = get_device();
        let config = OrthogonalLinearConfig::new(4, 3);
        let linear: OrthogonalLinear<TestBackend> = config.init(&device);

        let input = Tensor::random([2, 4], Distribution::Normal(0.0, 1.0), &device);
        let output = linear.forward(input);

        assert_eq!(output.dims(), [2, 3]);
    }

    #[test]
    fn test_orthogonal_weights_square() {
        let device = get_device();
        let weights = generate_orthogonal_weights::<TestBackend>(4, 4, 1.0, &device);

        assert_eq!(weights.dims(), [4, 4]);

        // Check orthogonality: W @ W^T should be approximately identity
        let product = weights.clone().matmul(weights.transpose());
        let identity = Tensor::<TestBackend, 2>::eye(4, &device);

        let diff = (product - identity).abs().mean().into_scalar();
        assert!(diff.elem::<f32>() < 0.1, "Matrix should be approximately orthogonal");
    }

    #[test]
    fn test_orthogonal_weights_tall() {
        let device = get_device();
        let weights = generate_orthogonal_weights::<TestBackend>(8, 4, 1.0, &device);

        assert_eq!(weights.dims(), [8, 4]);

        // For tall matrix, W^T @ W should be approximately identity
        let product = weights.clone().transpose().matmul(weights);
        let identity = Tensor::<TestBackend, 2>::eye(4, &device);

        let diff = (product - identity).abs().mean().into_scalar();
        assert!(diff.elem::<f32>() < 0.1, "Columns should be approximately orthonormal");
    }

    #[test]
    fn test_orthogonal_weights_wide() {
        let device = get_device();
        let weights = generate_orthogonal_weights::<TestBackend>(4, 8, 1.0, &device);

        assert_eq!(weights.dims(), [4, 8]);

        // For wide matrix, W @ W^T should be approximately identity
        let product = weights.clone().matmul(weights.transpose());
        let identity = Tensor::<TestBackend, 2>::eye(4, &device);

        let diff = (product - identity).abs().mean().into_scalar();
        assert!(diff.elem::<f32>() < 0.1, "Rows should be approximately orthonormal");
    }

    #[test]
    fn test_gain_scaling() {
        let device = get_device();

        let weights_g1 = generate_orthogonal_weights::<TestBackend>(4, 4, 1.0, &device);
        let weights_g2 = generate_orthogonal_weights::<TestBackend>(4, 4, 2.0, &device);

        // Weights with gain=2 should have larger values
        let mean_g1: f32 = weights_g1.abs().mean().into_scalar().elem();
        let mean_g2: f32 = weights_g2.abs().mean().into_scalar().elem();

        // g2 should be approximately 2x g1 (with some tolerance for randomness)
        assert!(mean_g2 > mean_g1 * 1.5, "Higher gain should produce larger weights");
    }

    #[test]
    fn test_no_bias() {
        let device = get_device();
        let config = OrthogonalLinearConfig::new(4, 3).with_bias(false);
        let linear: OrthogonalLinear<TestBackend> = config.init(&device);

        assert!(linear.bias.is_none());
    }
}
