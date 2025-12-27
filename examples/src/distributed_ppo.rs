//! Distributed PPO Example using the High-Level API
//!
//! This example demonstrates multi-actor PPO training using the clean
//! `DistributedPPORunner` API with the Model Factory pattern.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//!
//! # Environment
//!
//! Uses operant's SIMD-optimized CartPole environment for high throughput.
//! Operant environments use f32x8 SIMD vectorization for parallel physics.
//!
//! Run with: `LIBRARY_PATH=/opt/homebrew/lib cargo run --features distributed --release -- distributed-ppo`

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::runners::{DistributedPPOConfig, DistributedPPODiscrete};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// Network Definition
// ============================================================================

/// Simple Actor-Critic network for CartPole.
#[derive(Module, Debug)]
pub struct PPONet<B: burn::tensor::backend::Backend> {
    shared_0: Linear<B>,
    shared_1: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> PPONet<B> {
    /// Create a new PPONet for CartPole.
    pub fn new(device: &B::Device) -> Self {
        Self {
            shared_0: LinearConfig::new(CartPoleEnv::OBS_SIZE, 64).init(device),
            shared_1: LinearConfig::new(64, 64).init(device),
            policy_head: LinearConfig::new(64, CartPoleEnv::N_ACTIONS).init(device),
            value_head: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Forward pass returning (logits, values).
    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = relu(self.shared_0.forward(x));
        let x = relu(self.shared_1.forward(x));
        let logits = self.policy_head.forward(x.clone());
        let values = self.value_head.forward(x);
        (logits, values)
    }
}

// ============================================================================
// ActorCriticInference Implementation - For ANY Backend (including Inner)
// ============================================================================

impl<Backend: burn::tensor::backend::Backend> ActorCriticInference<Backend, DiscretePolicy, FeedForward>
    for PPONet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        _hidden: (),
    ) -> ForwardOutput<Backend, DiscretePolicy, FeedForward> {
        let (logits, values) = self.forward_net(obs);
        ForwardOutput::new(DiscretePolicyOutput { logits }, values, ())
    }

    fn obs_size(&self) -> usize {
        CartPoleEnv::OBS_SIZE
    }

    fn action_policy(&self) -> DiscretePolicy {
        DiscretePolicy::new(CartPoleEnv::N_ACTIONS)
    }

    fn temporal_policy(&self) -> FeedForward {
        FeedForward::new()
    }
}

// ============================================================================
// ActorCritic Implementation - For AutodiffBackend (Training)
// ============================================================================

impl ActorCritic<B, DiscretePolicy, FeedForward> for PPONet<B> {}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Distributed PPO (High-Level API) ===");
    println!("Environment: CartPole (operant SIMD)");
    println!();

    // Configuration
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_rollout_length(64)
        .with_n_epochs(4)
        .with_n_minibatches(8)
        .with_learning_rate(3e-4)
        .with_target_reward(475.0)
        .with_max_env_steps(1_000_000);

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
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = PPONet::<B>::new(&device);
    println!("Model created successfully.");
    println!();

    // Create runner using the config
    let runner: DistributedPPODiscrete<B> =
        distributed_rl::runners::DistributedPPORunner::new(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| PPONet::<B>::new(device);

    // Environment factory - creates operant SIMD CartPole for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Optimizer
    let optimizer = AdamConfig::new().init::<B, PPONet<B>>();

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
                "Steps: {:>8} | Episodes: {:>5} | Reward: {:>6.1} | Version: {:>4} | SPS: {:>6.0}",
                stats.env_steps,
                stats.episodes,
                stats.avg_reward,
                stats.policy_version,
                stats.steps_per_second
            );
        },
    );
}
