//! Target network utilities for stable value estimation.
//!
//! Target networks are copies of the main network that are updated slowly,
//! providing stable bootstrap targets during training. This technique was
//! introduced in DQN and is used in many RL algorithms.
//!
//! # Theory
//!
//! In temporal difference learning, we bootstrap from our own value estimates:
//! ```text
//! target = r + γ * V(s')
//! ```
//!
//! Using the same network for both prediction and targets creates a moving target
//! problem. Target networks solve this by:
//! 1. Keeping a separate "target" copy of the network
//! 2. Updating target weights slowly via soft updates (Polyak averaging)
//!
//! # Soft Updates (Polyak Averaging)
//!
//! ```text
//! θ_target = τ * θ_online + (1 - τ) * θ_target
//! ```
//!
//! Where τ is typically small (0.005 - 0.01), ensuring smooth target evolution.
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::core::target_network::soft_update;
//!
//! // In training loop:
//! target_model = soft_update(online_model, target_model, tau);
//! ```
//!
//! # Applications
//!
//! - DQN and variants (Double DQN, Dueling DQN)
//! - SAC (Soft Actor-Critic)
//! - TD3 (Twin Delayed DDPG)
//! - PPO with target value network (optional stabilization)

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Soft Update Implementation via ModuleMapper
// ============================================================================

/// Stores a flattened parameter tensor with its original shape.
///
/// Parameters are stored as 1D tensors to sidestep const generic dimension issues
/// when storing tensors of varying dimensions in a collection.
struct FlattenedParam<B: Backend> {
    /// The parameter tensor flattened to 1D
    tensor: Tensor<B, 1>,
}

/// Extracts all parameters from a module into a Vec for soft updates.
///
/// Parameters are collected in traversal order, which is deterministic for
/// modules with the same architecture. This allows matching parameters
/// between two independently created models of the same structure.
struct ParamExtractor<B: Backend> {
    params: Vec<FlattenedParam<B>>,
}

impl<B: Backend> ParamExtractor<B> {
    fn new() -> Self {
        Self { params: Vec::new() }
    }

    fn into_params(self) -> Vec<FlattenedParam<B>> {
        self.params
    }
}

impl<B: Backend> ModuleMapper<B> for ParamExtractor<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let val = param.val();
        let total_size: usize = val.dims().iter().product();

        // Flatten to 1D for storage
        let flattened = val.clone().reshape([total_size]);

        self.params.push(FlattenedParam { tensor: flattened });

        // Return the original param unchanged
        param
    }
}

/// Performs Polyak averaging (soft update) between online and target parameters.
///
/// For each parameter: `θ_target = τ * θ_online + (1 - τ) * θ_target`
///
/// Parameters are matched by traversal order, not by ParamId. This allows
/// soft updates between independently created models of the same architecture.
struct SoftUpdateMapper<B: Backend> {
    online_params: Vec<FlattenedParam<B>>,
    tau: f32,
    index: RefCell<usize>,
}

impl<B: Backend> SoftUpdateMapper<B> {
    fn new(online_params: Vec<FlattenedParam<B>>, tau: f32) -> Self {
        Self {
            online_params,
            tau,
            index: RefCell::new(0),
        }
    }
}

impl<B: Backend> ModuleMapper<B> for SoftUpdateMapper<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let target_val = param.val();
        let shape = target_val.dims();
        let total_size: usize = shape.iter().product();

        // Get current index and increment
        let idx = *self.index.borrow();
        *self.index.borrow_mut() = idx + 1;

        // Flatten target
        let target_flat = target_val.reshape([total_size]);

        // Get corresponding online parameter by index
        if let Some(online) = self.online_params.get(idx) {
            // Polyak averaging: τ * θ_online + (1 - τ) * θ_target
            let interpolated = online.tensor.clone().mul_scalar(self.tau)
                + target_flat.mul_scalar(1.0 - self.tau);

            // Reshape back to original shape
            let reshaped = interpolated.reshape(shape);

            // Create new Param with same ID but updated value
            Param::initialized(param.id.clone(), reshaped)
        } else {
            // No matching online param found - keep target unchanged
            // This shouldn't happen if models have matching architectures
            param
        }
    }
}

/// Trait for models that support soft updates.
///
/// This trait should be implemented by models that need target network functionality.
/// The implementation handles the interpolation between online and target weights.
pub trait SoftUpdatable<B: Backend>: Sized {
    /// Perform soft update from the online model.
    ///
    /// Updates self (target) towards the online model using Polyak averaging:
    /// ```text
    /// θ_target = τ * θ_online + (1 - τ) * θ_target
    /// ```
    fn soft_update_from(&self, online: &Self, tau: f32, device: &B::Device) -> Self;

    /// Perform hard copy from the online model.
    fn hard_copy_from(online: &Self, device: &B::Device) -> Self;
}

/// Perform soft update (Polyak averaging) from online to target model.
///
/// Updates target weights using the formula:
/// ```text
/// θ_target = τ * θ_online + (1 - τ) * θ_target
/// ```
///
/// This is the standard approach for stabilizing temporal difference learning
/// in algorithms like SAC, TD3, and DQN variants.
///
/// # Arguments
/// * `online` - The online (training) model with current weights
/// * `target` - The target model to update
/// * `tau` - Interpolation coefficient (0.0 to 1.0, typically 0.005-0.01)
/// * `_device` - Device for tensor operations (unused, tensors carry their device)
///
/// # Returns
/// Updated target model with interpolated weights
///
/// # Example
/// ```ignore
/// // In SAC training loop:
/// target_critic = soft_update(&critic, target_critic, 0.005, &device);
/// ```
pub fn soft_update<B, M>(online: &M, target: M, tau: f32, _device: &B::Device) -> M
where
    B: Backend,
    M: Module<B>,
{
    // Edge case: tau = 1.0 means hard copy (full replacement)
    if (tau - 1.0).abs() < 1e-6 {
        return online.clone();
    }

    // Edge case: tau = 0.0 means no update
    if tau.abs() < 1e-6 {
        return target;
    }

    // Step 1: Extract all parameters from online model
    let mut extractor = ParamExtractor::new();
    let _ = online.clone().map(&mut extractor);
    let online_params = extractor.into_params();

    // Step 2: Apply soft update to target model parameters
    let mut updater = SoftUpdateMapper::new(online_params, tau);
    target.map(&mut updater)
}

/// Configuration for target network management.
#[derive(Debug, Clone)]
pub struct TargetNetworkConfig {
    /// Soft update coefficient (tau).
    /// Higher values mean faster target updates.
    /// Typical values: 0.005 - 0.01
    pub tau: f32,
    /// Update frequency (in training steps).
    /// When > 1, performs hard update every N steps instead of soft updates.
    pub update_freq: usize,
    /// Whether to use hard updates instead of soft updates.
    pub hard_update: bool,
}

impl Default for TargetNetworkConfig {
    fn default() -> Self {
        Self {
            tau: 0.005,
            update_freq: 1,
            hard_update: false,
        }
    }
}

impl TargetNetworkConfig {
    /// Create a new config with soft updates.
    pub fn soft(tau: f32) -> Self {
        Self {
            tau,
            update_freq: 1,
            hard_update: false,
        }
    }

    /// Create a new config with hard updates.
    pub fn hard(update_freq: usize) -> Self {
        Self {
            tau: 1.0,
            update_freq,
            hard_update: true,
        }
    }

    /// Set the tau value.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Set the update frequency.
    pub fn with_update_freq(mut self, freq: usize) -> Self {
        self.update_freq = freq;
        self
    }
}

/// Manager for target network updates.
///
/// Handles the logic of when and how to update target networks based on configuration.
/// Uses `AtomicUsize` for the step counter to support interior mutability,
/// allowing `&self` in `maybe_update` (no need for `&mut self`).
#[derive(Debug)]
pub struct TargetNetworkManager {
    config: TargetNetworkConfig,
    step_counter: AtomicUsize,
}

impl Clone for TargetNetworkManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            step_counter: AtomicUsize::new(self.step_counter.load(Ordering::Relaxed)),
        }
    }
}

impl TargetNetworkManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: TargetNetworkConfig) -> Self {
        Self {
            config,
            step_counter: AtomicUsize::new(0),
        }
    }

    /// Create a manager for soft updates.
    pub fn soft(tau: f32) -> Self {
        Self::new(TargetNetworkConfig::soft(tau))
    }

    /// Create a manager for hard updates.
    pub fn hard(update_freq: usize) -> Self {
        Self::new(TargetNetworkConfig::hard(update_freq))
    }

    /// Check if an update should be performed and update the target if needed.
    ///
    /// Uses interior mutability (AtomicUsize) so this takes `&self` instead of `&mut self`.
    /// This is critical for SAC where the manager is passed by reference to train_step.
    ///
    /// Returns the (potentially updated) target model.
    pub fn maybe_update<B, M>(
        &self,
        online: &M,
        target: M,
        device: &B::Device,
    ) -> M
    where
        B: Backend,
        M: Module<B>,
    {
        // Atomically increment and get the new step count
        let step = self.step_counter.fetch_add(1, Ordering::Relaxed) + 1;

        if self.config.hard_update {
            // Hard update: copy weights every N steps
            if step % self.config.update_freq == 0 {
                hard_copy(online, device)
            } else {
                target
            }
        } else {
            // Soft update: interpolate weights every step
            if step % self.config.update_freq == 0 {
                soft_update(online, target, self.config.tau, device)
            } else {
                target
            }
        }
    }

    /// Get the current step count.
    pub fn steps(&self) -> usize {
        self.step_counter.load(Ordering::Relaxed)
    }

    /// Reset the step counter.
    pub fn reset(&mut self) {
        self.step_counter.store(0, Ordering::Relaxed);
    }

    /// Get the configuration.
    pub fn config(&self) -> &TargetNetworkConfig {
        &self.config
    }
}

/// Perform a hard copy of model weights (tau = 1.0).
///
/// Creates a clone of the online model.
pub fn hard_copy<B, M>(online: &M, _device: &B::Device) -> M
where
    B: Backend,
    M: Module<B> + Clone,
{
    online.clone()
}

/// Exponential moving average (EMA) for tracking statistics.
///
/// Useful for tracking running estimates that should give more weight to recent values.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    /// Current EMA value
    value: f64,
    /// Decay factor (0 < decay < 1, typically 0.99 or 0.999)
    decay: f64,
    /// Number of updates
    count: usize,
    /// Whether to use bias correction
    bias_correction: bool,
}

impl ExponentialMovingAverage {
    /// Create a new EMA tracker.
    pub fn new(decay: f64) -> Self {
        Self {
            value: 0.0,
            decay,
            count: 0,
            bias_correction: true,
        }
    }

    /// Create an EMA tracker without bias correction.
    pub fn without_bias_correction(decay: f64) -> Self {
        Self {
            value: 0.0,
            decay,
            count: 0,
            bias_correction: false,
        }
    }

    /// Update the EMA with a new value.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.value = self.decay * self.value + (1.0 - self.decay) * value;
    }

    /// Get the current EMA value.
    ///
    /// Applies bias correction if enabled.
    pub fn get(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        if self.bias_correction {
            // Bias correction: divide by (1 - decay^count)
            let correction = 1.0 - self.decay.powi(self.count as i32);
            self.value / correction
        } else {
            self.value
        }
    }

    /// Get the raw (uncorrected) EMA value.
    pub fn raw(&self) -> f64 {
        self.value
    }

    /// Reset the EMA.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.count = 0;
    }

    /// Get the number of updates.
    pub fn count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_network_config_soft() {
        let config = TargetNetworkConfig::soft(0.005);
        assert_eq!(config.tau, 0.005);
        assert_eq!(config.update_freq, 1);
        assert!(!config.hard_update);
    }

    #[test]
    fn test_target_network_config_hard() {
        let config = TargetNetworkConfig::hard(100);
        assert_eq!(config.tau, 1.0);
        assert_eq!(config.update_freq, 100);
        assert!(config.hard_update);
    }

    #[test]
    fn test_target_network_manager_soft() {
        let manager = TargetNetworkManager::soft(0.01);
        assert_eq!(manager.steps(), 0);
        assert!(!manager.config().hard_update);
    }

    #[test]
    fn test_target_network_manager_hard() {
        let manager = TargetNetworkManager::hard(10);
        assert_eq!(manager.steps(), 0);
        assert!(manager.config().hard_update);
    }

    #[test]
    fn test_ema_basic() {
        let mut ema = ExponentialMovingAverage::new(0.9);

        // First update
        ema.update(10.0);
        assert!(ema.get() > 9.0 && ema.get() < 11.0);

        // More updates converge towards new value
        for _ in 0..100 {
            ema.update(20.0);
        }
        assert!((ema.get() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_ema_without_bias_correction() {
        let mut ema = ExponentialMovingAverage::without_bias_correction(0.9);

        ema.update(10.0);
        // Without bias correction, first update gives decay * 0 + (1-decay) * 10 = 1.0
        assert!((ema.get() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = ExponentialMovingAverage::new(0.9);
        ema.update(10.0);
        ema.update(20.0);

        ema.reset();
        assert_eq!(ema.count(), 0);
        assert_eq!(ema.get(), 0.0);
    }

    // ========================================================================
    // Soft Update Tests
    // ========================================================================

    use burn::backend::NdArray;
    use burn::nn::{Linear, LinearConfig};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_soft_update_tau_zero_returns_target() {
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let target = LinearConfig::new(4, 4).init::<TestBackend>(&device);

        // Get original target weights
        let target_weight_before = target.weight.val().into_data();

        // tau = 0 should return target unchanged
        let updated = soft_update::<TestBackend, _>(&online, target, 0.0, &device);
        let updated_weight = updated.weight.val().into_data();

        // Weights should be identical to original target
        let target_slice = target_weight_before.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        for (t, u) in target_slice.iter().zip(updated_slice.iter()) {
            assert!(
                (t - u).abs() < 1e-6,
                "tau=0 should return target unchanged"
            );
        }
    }

    #[test]
    fn test_soft_update_tau_one_returns_online() {
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let target = LinearConfig::new(4, 4).init::<TestBackend>(&device);

        // Get original online weights
        let online_weight = online.weight.val().into_data();

        // tau = 1 should return online copy (hard update)
        let updated = soft_update::<TestBackend, _>(&online, target, 1.0, &device);
        let updated_weight = updated.weight.val().into_data();

        // Weights should be identical to online
        let online_slice = online_weight.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        for (o, u) in online_slice.iter().zip(updated_slice.iter()) {
            assert!(
                (o - u).abs() < 1e-6,
                "tau=1 should return online weights"
            );
        }
    }

    #[test]
    fn test_soft_update_interpolation() {
        let device = <TestBackend as Backend>::Device::default();

        // Create two linear layers with controlled weights
        let online = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let target = LinearConfig::new(4, 4).init::<TestBackend>(&device);

        let online_weight = online.weight.val().into_data();
        let target_weight = target.weight.val().into_data();

        // tau = 0.5 should give average of online and target
        let tau = 0.5f32;
        let updated = soft_update::<TestBackend, _>(&online, target, tau, &device);
        let updated_weight = updated.weight.val().into_data();

        let online_slice = online_weight.as_slice::<f32>().unwrap();
        let target_slice = target_weight.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        for i in 0..online_slice.len() {
            let expected = tau * online_slice[i] + (1.0 - tau) * target_slice[i];
            assert!(
                (updated_slice[i] - expected).abs() < 1e-5,
                "Expected {}, got {} at index {} (tau={})",
                expected,
                updated_slice[i],
                i,
                tau
            );
        }
    }

    #[test]
    fn test_soft_update_typical_tau() {
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(8, 4).init::<TestBackend>(&device);
        let target = LinearConfig::new(8, 4).init::<TestBackend>(&device);

        let online_weight = online.weight.val().into_data();
        let target_weight = target.weight.val().into_data();

        // Typical SAC tau value
        let tau = 0.005f32;
        let updated = soft_update::<TestBackend, _>(&online, target, tau, &device);
        let updated_weight = updated.weight.val().into_data();

        let online_slice = online_weight.as_slice::<f32>().unwrap();
        let target_slice = target_weight.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        for i in 0..online_slice.len() {
            let expected = tau * online_slice[i] + (1.0 - tau) * target_slice[i];
            assert!(
                (updated_slice[i] - expected).abs() < 1e-5,
                "Soft update failed at index {} with tau={}",
                i,
                tau
            );
        }
    }

    #[test]
    fn test_soft_update_bias() {
        // Test that bias parameters are also updated
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(4, 4)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let target = LinearConfig::new(4, 4)
            .with_bias(true)
            .init::<TestBackend>(&device);

        let online_bias = online.bias.as_ref().unwrap().val().into_data();
        let target_bias = target.bias.as_ref().unwrap().val().into_data();

        let tau = 0.3f32;
        let updated = soft_update::<TestBackend, _>(&online, target, tau, &device);
        let updated_bias = updated.bias.as_ref().unwrap().val().into_data();

        let online_slice = online_bias.as_slice::<f32>().unwrap();
        let target_slice = target_bias.as_slice::<f32>().unwrap();
        let updated_slice = updated_bias.as_slice::<f32>().unwrap();

        for i in 0..online_slice.len() {
            let expected = tau * online_slice[i] + (1.0 - tau) * target_slice[i];
            assert!(
                (updated_slice[i] - expected).abs() < 1e-5,
                "Bias soft update failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_target_network_manager_with_soft_update() {
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let target = LinearConfig::new(4, 4).init::<TestBackend>(&device);

        let online_weight = online.weight.val().into_data();
        let target_weight_before = target.weight.val().into_data();

        // Create manager with soft updates
        let tau = 0.1f32;
        let manager = TargetNetworkManager::new(TargetNetworkConfig {
            tau,
            update_freq: 1,
            hard_update: false,
        });

        // Perform update via manager (now takes &self, not &mut self)
        let updated = manager.maybe_update::<TestBackend, _>(&online, target, &device);
        let updated_weight = updated.weight.val().into_data();

        let online_slice = online_weight.as_slice::<f32>().unwrap();
        let target_slice = target_weight_before.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        // Verify interpolation
        for i in 0..online_slice.len() {
            let expected = tau * online_slice[i] + (1.0 - tau) * target_slice[i];
            assert!(
                (updated_slice[i] - expected).abs() < 1e-5,
                "Manager soft update failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_target_network_manager_step_counter_persists() {
        // Verify that the step counter persists across multiple calls
        // (this was the bug - .clone() was resetting the counter)
        let device = <TestBackend as Backend>::Device::default();

        let online = LinearConfig::new(4, 4).init::<TestBackend>(&device);

        // Create manager with hard updates every 3 steps
        let manager = TargetNetworkManager::new(TargetNetworkConfig {
            tau: 1.0,
            update_freq: 3,
            hard_update: true,
        });

        assert_eq!(manager.steps(), 0);

        // First call: step 1, no update
        let target1 = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let _ = manager.maybe_update::<TestBackend, _>(&online, target1, &device);
        assert_eq!(manager.steps(), 1);

        // Second call: step 2, no update
        let target2 = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let _ = manager.maybe_update::<TestBackend, _>(&online, target2, &device);
        assert_eq!(manager.steps(), 2);

        // Third call: step 3, should update (3 % 3 == 0)
        let target3 = LinearConfig::new(4, 4).init::<TestBackend>(&device);
        let target3_weight_before = target3.weight.val().into_data();
        let updated = manager.maybe_update::<TestBackend, _>(&online, target3, &device);
        assert_eq!(manager.steps(), 3);

        // Verify hard update occurred (weights should match online, not target3)
        let online_weight = online.weight.val().into_data();
        let updated_weight = updated.weight.val().into_data();
        let online_slice = online_weight.as_slice::<f32>().unwrap();
        let target3_slice = target3_weight_before.as_slice::<f32>().unwrap();
        let updated_slice = updated_weight.as_slice::<f32>().unwrap();

        // Updated should match online (hard copy), not target3
        for i in 0..online_slice.len() {
            assert!(
                (updated_slice[i] - online_slice[i]).abs() < 1e-6,
                "Hard update should copy online weights at step 3"
            );
            // Ensure target3 was different from online
            if (target3_slice[i] - online_slice[i]).abs() > 1e-6 {
                assert!(
                    (updated_slice[i] - target3_slice[i]).abs() > 1e-6,
                    "Updated should not equal target3"
                );
            }
        }
    }
}
