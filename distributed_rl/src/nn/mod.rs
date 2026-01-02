//! Neural network utility modules for RL.
//!
//! This module provides specialized neural network components useful for
//! reinforcement learning that aren't available in Burn's standard library.
//!
//! # Modules
//!
//! - [`orthogonal`]: Orthogonal initialization for improved gradient flow
//! - [`spectral_norm`]: Spectral normalization for Lipschitz-constrained networks
//! - [`attention`]: Multi-head attention mechanisms for relational reasoning

pub mod orthogonal;
pub mod spectral_norm;
pub mod attention;

pub use orthogonal::{OrthogonalLinear, OrthogonalLinearConfig, generate_orthogonal_weights};
pub use spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};
pub use attention::{MultiHeadAttention, MultiHeadAttentionConfig};
