//! # Operant
//!
//! High-performance SIMD-optimized reinforcement learning environments.
//!
//! This crate provides vectorized RL environments with automatic SIMD
//! optimization for maximum throughput on modern CPUs.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use operant::{CartPole, VecEnv};
//!
//! // Create 1024 parallel CartPole environments
//! let mut env = CartPole::new(1024);
//!
//! // Reset all environments
//! let obs = env.reset();
//!
//! // Step with actions
//! let actions = vec![0; 1024]; // discrete actions
//! let (obs, rewards, terminals, truncations) = env.step(&actions);
//! ```
//!
//! ## Features
//!
//! - `simd` (default): Enable SIMD optimizations (requires nightly)
//! - `parallel`: Enable multi-threaded parallel execution via rayon
//!
//! ## Crate Structure
//!
//! - [`operant_core`]: Core traits and vectorization primitives
//! - [`operant_envs`]: Environment implementations (CartPole, MountainCar, Pendulum)

// Re-export core traits and types
pub use operant_core::*;

// Re-export environment implementations
pub use operant_envs::*;
