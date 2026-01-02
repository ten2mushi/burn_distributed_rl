//! IMPALA Example using the High-Level API
//!
//! This example demonstrates multi-actor IMPALA training using the clean
//! `IMPALARunner` API with the Model Factory pattern.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - V-trace importance sampling correction for off-policy learning
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//!
//! # Environment
//!
//! Uses operant's SIMD-optimized CartPole environment for high throughput.
//! Operant environments use f32x8 SIMD vectorization for parallel physics.
//!
//! # IMPALA vs PPO
//!
//! IMPALA uses asynchronous experience collection with V-trace correction,
//! making it more sample-efficient but with higher off-policy lag. PPO uses
//! synchronous collection with on-policy updates for more stable learning.
//!
//! Run with: `LIBRARY_PATH=/opt/homebrew/lib cargo run --features distributed --release -- distributed-impala`

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
// AdamConfig now configured via runner.create_optimizer()
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::runners::{IMPALAConfig, IMPALADiscrete, IMPALARunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// Network Definition
// ============================================================================

/// Simple Actor-Critic network for CartPole.
#[derive(Module, Debug)]
pub struct IMPALANet<B: burn::tensor::backend::Backend> {
    shared_0: Linear<B>,
    shared_1: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> IMPALANet<B> {
    /// Create a new IMPALANet for CartPole.
    pub fn new(device: &B::Device) -> Self {
        Self {
            shared_0: LinearConfig::new(CartPoleEnv::OBS_SIZE, 128).init(device),
            shared_1: LinearConfig::new(128, 128).init(device),
            policy_head: LinearConfig::new(128, CartPoleEnv::N_ACTIONS).init(device),
            value_head: LinearConfig::new(128, 1).init(device),
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

/// Implement inference trait for any Backend.
/// This allows the model to run on both Autodiff<Wgpu> (learner) and Wgpu (actors).
impl<Backend: burn::tensor::backend::Backend> ActorCriticInference<Backend, DiscretePolicy, FeedForward>
    for IMPALANet<Backend>
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

/// Implement training trait for AutodiffBackend.
/// This marker trait enables gradient computation on the learner thread.
impl ActorCritic<B, DiscretePolicy, FeedForward> for IMPALANet<B> {
    // Training methods inherited from ActorCriticInference.
    // Currently empty - this is a marker trait that adds AutodiffModule bound.
}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== IMPALA (High-Level API) ===");
    println!("Environment: CartPole (operant SIMD)");
    println!();

    // Configuration
    // Note: Uses tuned defaults (rho_clip=1.5, vf_coef=0.25, entropy_coef=0.02)
    let config = IMPALAConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_trajectory_length(20)
        .with_buffer_capacity(256)
        .with_batch_size(32)
        .with_learning_rate(3e-4) // Lower than before for stability
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
        "  Trajectory length: {}, Buffer: {}, Batch: {}",
        config.trajectory_length, config.buffer_capacity, config.batch_size
    );
    println!(
        "  V-trace: gamma={}, rho_clip={}, c_clip={}",
        config.gamma, config.rho_clip, config.c_clip
    );
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = IMPALANet::<B>::new(&device);
    println!("Model created successfully.");
    println!();

    // Create runner using the config
    let runner: IMPALADiscrete<B> =
        distributed_rl::runners::IMPALARunner::new(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| IMPALANet::<B>::new(device);

    // Environment factory - creates operant SIMD CartPole for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Optimizer with gradient clipping from config
    let optimizer = runner.create_optimizer::<IMPALANet<B>>();

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
                "Steps: {:>8} | Episodes: {:>5} | Reward: {:>6.1} | Version: {:>4} | SPS: {:>6.0} | Buffer: {:>3}",
                stats.env_steps,
                stats.episodes,
                stats.avg_reward,
                stats.policy_version,
                stats.steps_per_second,
                stats.buffer_size
            );
        },
    );
}
