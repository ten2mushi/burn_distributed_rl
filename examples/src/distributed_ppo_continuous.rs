//! Distributed PPO Example - Continuous Actions (High-Level API)
//!
//! This example demonstrates multi-actor PPO training with continuous actions
//! on Pendulum using the `DistributedPPORunner` with Model Factory pattern.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Squashed Gaussian policy for bounded continuous actions
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//!
//! # Environment
//!
//! Uses operant's SIMD-optimized Pendulum environment for high throughput.
//! Operant environments use f32x8 SIMD vectorization for parallel physics.
//!
//! Run with: `LIBRARY_PATH=/opt/homebrew/lib cargo run --features distributed --release -- distributed-ppo-continuous`

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{ContinuousPolicy, ContinuousPolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::PendulumEnv;
use distributed_rl::runners::{DistributedPPOConfig, DistributedPPOContinuous};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// Log std bounds for numerical stability
const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;

// ============================================================================
// Network Definition
// ============================================================================

/// Actor-Critic network for continuous PPO with squashed Gaussian policy.
#[derive(Module, Debug)]
pub struct ContinuousPPONet<B: burn::tensor::backend::Backend> {
    shared_0: Linear<B>,
    shared_1: Linear<B>,
    policy_mean: Linear<B>,
    policy_log_std: Linear<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> ContinuousPPONet<B> {
    /// Create a new ContinuousPPONet for Pendulum.
    pub fn new(device: &B::Device) -> Self {
        Self {
            shared_0: LinearConfig::new(PendulumEnv::OBS_SIZE, 64).init(device),
            shared_1: LinearConfig::new(64, 64).init(device),
            policy_mean: LinearConfig::new(64, PendulumEnv::ACTION_DIM).init(device),
            policy_log_std: LinearConfig::new(64, PendulumEnv::ACTION_DIM).init(device),
            value_head: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Forward pass returning (mean, log_std, values).
    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let x = relu(self.shared_0.forward(x));
        let x = relu(self.shared_1.forward(x));
        // Policy mean with tanh to bound to reasonable range
        let mean = tanh(self.policy_mean.forward(x.clone()));
        // Log std with clamping for numerical stability
        let log_std = self
            .policy_log_std
            .forward(x.clone())
            .clamp(LOG_STD_MIN, LOG_STD_MAX);
        let values = self.value_head.forward(x);
        (mean, log_std, values)
    }
}

// ============================================================================
// ActorCritic Implementation
// ============================================================================

impl ActorCritic<B, ContinuousPolicy, FeedForward> for ContinuousPPONet<B> {
    fn forward(
        &self,
        obs: Tensor<B, 2>,
        _hidden: (),
    ) -> ForwardOutput<B, ContinuousPolicy, FeedForward> {
        let (mean, log_std, values) = self.forward_net(obs);
        ForwardOutput::new(
            ContinuousPolicyOutput {
                mean,
                log_std,
                bounds: PendulumEnv::action_bounds(),
            },
            values,
            (),
        )
    }

    fn obs_size(&self) -> usize {
        PendulumEnv::OBS_SIZE
    }

    fn action_policy(&self) -> ContinuousPolicy {
        let (low, high) = PendulumEnv::action_bounds();
        ContinuousPolicy::new(PendulumEnv::ACTION_DIM, low, high)
    }

    fn temporal_policy(&self) -> FeedForward {
        FeedForward::new()
    }
}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Distributed PPO Continuous (High-Level API) ===");
    println!("Environment: Pendulum (operant SIMD)");
    println!();

    // Configuration
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_rollout_length(128)
        .with_n_epochs(4)
        .with_n_minibatches(8)
        .with_learning_rate(3e-4)
        .with_max_env_steps(3_000_000);

    println!("Configuration:");
    println!(
        "  Actors: {} x {} envs = {} total",
        config.n_actors,
        config.n_envs_per_actor,
        config.total_envs()
    );
    println!(
        "  Rollout: {} steps x {} envs = {} transitions",
        config.rollout_length,
        config.total_envs(),
        config.transitions_per_rollout()
    );
    println!(
        "  Action bounds: [{}, {}]",
        PendulumEnv::ACTION_LOW,
        PendulumEnv::ACTION_HIGH
    );
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = ContinuousPPONet::<B>::new(&device);
    println!("Model created successfully.");
    println!();

    // Create runner using the config
    let runner: DistributedPPOContinuous<B> =
        distributed_rl::runners::DistributedPPORunner::new(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| ContinuousPPONet::<B>::new(device);

    // Environment factory - creates operant SIMD Pendulum for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
    };

    // Optimizer
    let optimizer = AdamConfig::new().init::<B, ContinuousPPONet<B>>();

    // Run training with progress callback
    println!("Starting training...");
    println!();

    let _trained_model = runner.run(
        model_factory,
        initial_model,
        env_factory,
        optimizer,
        |stats| {
            println!(
                "Steps: {:>8} | Episodes: {:>5} | Reward: {:>7.1} | Version: {:>4} | SPS: {:>6.0}",
                stats.env_steps,
                stats.episodes,
                stats.avg_reward,
                stats.policy_version,
                stats.steps_per_second
            );
            if stats.avg_reward > -200.0 {
                println!("  >> Excellent Performance! (avg > -200)");
            }
        },
    );
}
