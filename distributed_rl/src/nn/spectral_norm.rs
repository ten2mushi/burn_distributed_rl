//! Spectral normalization for neural networks.
//!
//! Spectral normalization constrains the Lipschitz constant of the network
//! by normalizing weights by their largest singular value (spectral norm).
//!
//! # Theory
//!
//! For a matrix W, the spectral norm is the largest singular value σ_max(W).
//! Spectral normalization divides W by σ_max(W), ensuring ||W||_2 = 1.
//!
//! This provides:
//! - Bounded Lipschitz constant for each layer
//! - More stable training dynamics
//! - Prevention of value function explosion in RL
//!
//! # Algorithm
//!
//! Uses power iteration to efficiently estimate σ_max(W):
//! 1. Initialize random vectors u, v
//! 2. Repeat: v = W^T u / ||W^T u||, u = W v / ||W v||
//! 3. σ_max ≈ u^T W v
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::nn::{SpectralNormLinear, SpectralNormLinearConfig};
//!
//! let config = SpectralNormLinearConfig::new(64, 64)
//!     .with_n_power_iterations(1);  // Usually 1 is sufficient
//! let linear: SpectralNormLinear<Backend> = config.init(&device);
//!
//! let output = linear.forward(input);
//! ```
//!
//! # References
//!
//! - "Spectral Normalization for Generative Adversarial Networks" (Miyato et al., 2018)

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::Distribution;

/// Configuration for SpectralNormLinear layer.
#[derive(Debug, Clone)]
pub struct SpectralNormLinearConfig {
    /// Number of input features.
    pub d_input: usize,
    /// Number of output features.
    pub d_output: usize,
    /// Number of power iterations for spectral norm estimation.
    pub n_power_iterations: usize,
    /// Whether to include a bias term.
    pub bias: bool,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
}

impl SpectralNormLinearConfig {
    /// Create a new configuration.
    pub fn new(d_input: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_output,
            n_power_iterations: 1,
            bias: true,
            epsilon: 1e-12,
        }
    }

    /// Set number of power iterations.
    pub fn with_n_power_iterations(mut self, n: usize) -> Self {
        self.n_power_iterations = n;
        self
    }

    /// Set whether to include bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Initialize the layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralNormLinear<B> {
        // Initialize weight with normal distribution (will be normalized)
        let weight = Tensor::<B, 2>::random(
            [self.d_output, self.d_input],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.d_output], device)))
        } else {
            None
        };

        // Initialize u and v vectors for power iteration
        let u = Tensor::<B, 1>::random([self.d_output], Distribution::Normal(0.0, 1.0), device);
        let u = normalize_vector(u);

        let v = Tensor::<B, 1>::random([self.d_input], Distribution::Normal(0.0, 1.0), device);
        let v = normalize_vector(v);

        SpectralNormLinear {
            weight: Param::from_tensor(weight),
            bias,
            u: Param::from_tensor(u),
            v: Param::from_tensor(v),
            n_power_iterations: self.n_power_iterations,
            epsilon: self.epsilon,
        }
    }
}

/// Linear layer with spectral normalization.
///
/// The weights are normalized by their spectral norm (largest singular value)
/// before each forward pass to ensure ||W||_2 = 1.
#[derive(Module, Debug)]
pub struct SpectralNormLinear<B: Backend> {
    /// Weight matrix of shape [d_output, d_input]
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Left singular vector for power iteration
    u: Param<Tensor<B, 1>>,
    /// Right singular vector for power iteration
    v: Param<Tensor<B, 1>>,
    /// Number of power iterations
    n_power_iterations: usize,
    /// Epsilon for numerical stability
    epsilon: f32,
}

impl<B: Backend> SpectralNormLinear<B> {
    /// Forward pass with spectrally normalized weights.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Get normalized weight
        let w_normalized = self.normalized_weight();

        // Linear: output = input @ weight^T + bias
        let output = input.matmul(w_normalized.transpose());

        match &self.bias {
            Some(bias) => output + bias.val().unsqueeze(),
            None => output,
        }
    }

    /// Get the spectrally normalized weight matrix.
    pub fn normalized_weight(&self) -> Tensor<B, 2> {
        let weight = self.weight.val();
        let mut u = self.u.val();
        let mut v = self.v.val();

        // Power iteration to estimate spectral norm
        for _ in 0..self.n_power_iterations {
            // v = W^T u / ||W^T u||
            let v_new: Tensor<B, 1> = weight.clone().transpose().matmul(u.clone().unsqueeze_dim(1)).squeeze();
            v = normalize_vector(v_new);

            // u = W v / ||W v||
            let u_new: Tensor<B, 1> = weight.clone().matmul(v.clone().unsqueeze_dim(1)).squeeze();
            u = normalize_vector(u_new);
        }

        // Compute spectral norm: σ = u^T W v
        // Result is [1, 1], flatten and extract scalar
        let sigma_tensor = u.clone().unsqueeze_dim(0)
            .matmul(weight.clone())
            .matmul(v.unsqueeze_dim(1));
        let sigma_1d: Tensor<B, 1> = sigma_tensor.flatten(0, 1);
        let sigma: f32 = sigma_1d.mean().into_scalar().elem();

        // Add epsilon for stability and normalize
        let sigma_safe = sigma + self.epsilon;
        weight / sigma_safe
    }

    /// Get the current estimated spectral norm.
    pub fn spectral_norm(&self) -> f32 {
        let weight = self.weight.val();
        let u = self.u.val();
        let v = self.v.val();

        // σ = u^T W v
        // Result is [1, 1], flatten and extract scalar
        let sigma_tensor = u.unsqueeze_dim(0)
            .matmul(weight)
            .matmul(v.unsqueeze_dim(1));
        let sigma_1d: Tensor<B, 1> = sigma_tensor.flatten(0, 1);

        sigma_1d.mean().into_scalar().elem()
    }

    /// Get input dimension.
    pub fn d_input(&self) -> usize {
        self.weight.dims()[1]
    }

    /// Get output dimension.
    pub fn d_output(&self) -> usize {
        self.weight.dims()[0]
    }
}

/// Normalize a vector to unit length.
fn normalize_vector<B: Backend>(v: Tensor<B, 1>) -> Tensor<B, 1> {
    let norm = v.clone().powf_scalar(2.0).sum().sqrt();
    v / (norm + 1e-12)
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
    fn test_spectral_norm_forward() {
        let device = get_device();
        let config = SpectralNormLinearConfig::new(4, 3);
        let linear: SpectralNormLinear<TestBackend> = config.init(&device);

        let input = Tensor::random([2, 4], Distribution::Normal(0.0, 1.0), &device);
        let output = linear.forward(input);

        assert_eq!(output.dims(), [2, 3]);
    }

    #[test]
    fn test_spectral_norm_bounded() {
        let device = get_device();
        let config = SpectralNormLinearConfig::new(4, 4).with_n_power_iterations(5);
        let linear: SpectralNormLinear<TestBackend> = config.init(&device);

        let normalized_weight = linear.normalized_weight();

        // Compute actual spectral norm via power iteration on normalized weight
        let mut u = Tensor::<TestBackend, 1>::random([4], Distribution::Normal(0.0, 1.0), &device);
        u = normalize_vector(u);

        for _ in 0..10 {
            let v: Tensor<TestBackend, 1> = normalized_weight.clone().transpose().matmul(u.clone().unsqueeze_dim(1)).squeeze();
            let v = normalize_vector(v);
            let u_new: Tensor<TestBackend, 1> = normalized_weight.clone().matmul(v.unsqueeze_dim(1)).squeeze();
            u = normalize_vector(u_new);
        }

        // Spectral norm should be approximately 1
        // Result is [1, 1], flatten and extract scalar
        let sigma_tensor = u.clone().unsqueeze_dim(0)
            .matmul(normalized_weight.clone())
            .matmul(
                normalized_weight.clone().transpose().matmul(u.unsqueeze_dim(1))
            );
        let sigma_1d: Tensor<TestBackend, 1> = sigma_tensor.flatten(0, 1);

        let sigma_val: f32 = sigma_1d.mean().sqrt().into_scalar().elem();

        // Should be close to 1 (with some tolerance)
        assert!(
            (sigma_val - 1.0).abs() < 0.5,
            "Spectral norm should be approximately 1, got {}",
            sigma_val
        );
    }

    #[test]
    fn test_no_bias() {
        let device = get_device();
        let config = SpectralNormLinearConfig::new(4, 3).with_bias(false);
        let linear: SpectralNormLinear<TestBackend> = config.init(&device);

        assert!(linear.bias.is_none());
    }
}
