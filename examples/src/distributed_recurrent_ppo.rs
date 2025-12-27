//! Distributed PPO Example - Discrete Actions with Recurrent/LSTM (High-Level API)
//!
//! This example demonstrates multi-actor PPO training with discrete actions
//! on CartPole using a recurrent (LSTM) actor-critic network.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - LSTM for temporal dependencies (memory across timesteps)
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//!
//! # Environment
//!
//! Uses operant's SIMD-optimized CartPole environment for high throughput.
//! Operant environments use f32x8 SIMD vectorization for parallel physics.
//!
//! # When to Use Recurrent Policies
//!
//! Recurrent policies are useful when:
//! - The environment is partially observable (agent can't see full state)
//! - Temporal patterns matter (e.g., velocity from position history)
//! - Memory is needed to solve the task
//!
//! Run with: `LIBRARY_PATH=/opt/homebrew/lib cargo run --features distributed --release -- distributed-recurrent-ppo`

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::{Recurrent, RecurrentHidden};
use distributed_rl::core::recurrent::{LstmCellConfig, LstmCellWrapper, RecurrentCell};
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::runners::{DistributedPPOConfig, DistributedRecurrentPPODiscrete, DistributedRecurrentPPORunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// LSTM configuration
const HIDDEN_SIZE: usize = 64;
const ENCODED_SIZE: usize = 32;

// ============================================================================
// Network Definition
// ============================================================================

/// Recurrent Actor-Critic network with LSTM for discrete PPO.
///
/// Architecture: Encoder -> LSTM -> Policy/Value heads
#[derive(Module, Debug)]
pub struct RecurrentPPONet<B: burn::tensor::backend::Backend> {
    encoder: Linear<B>,
    lstm: LstmCellWrapper<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> RecurrentPPONet<B> {
    /// Create a new RecurrentPPONet for CartPole.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(CartPoleEnv::OBS_SIZE, ENCODED_SIZE).init(device),
            lstm: LstmCellConfig::new(ENCODED_SIZE, HIDDEN_SIZE).init(device),
            policy_head: LinearConfig::new(HIDDEN_SIZE, CartPoleEnv::N_ACTIONS).init(device),
            value_head: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
        }
    }
}

// ============================================================================
// ActorCritic Implementation
// ============================================================================

impl ActorCritic<B, DiscretePolicy, Recurrent> for RecurrentPPONet<B> {
    fn forward(
        &self,
        obs: Tensor<B, 2>,
        hidden: RecurrentHidden<B>,
    ) -> ForwardOutput<B, DiscretePolicy, Recurrent> {
        let batch_size = obs.dims()[0];
        let device = obs.device();

        // Encode -> LSTM
        let encoded = relu(self.encoder.forward(obs));
        let batched_hidden = hidden.to_batched_state();
        let (_output, new_hidden_state) = self.lstm.step(encoded, &batched_hidden);

        // Policy and value from LSTM hidden state
        let h = new_hidden_state.hidden.clone();
        let logits = self.policy_head.forward(h.clone());
        let value = self.value_head.forward(h);

        // Update hidden states
        let mut new_hidden = RecurrentHidden::new(batch_size, HIDDEN_SIZE, true, &device);
        new_hidden.from_batched_state(new_hidden_state);

        ForwardOutput::new(DiscretePolicyOutput { logits }, value, new_hidden)
    }

    fn obs_size(&self) -> usize {
        CartPoleEnv::OBS_SIZE
    }

    fn action_policy(&self) -> DiscretePolicy {
        DiscretePolicy::new(CartPoleEnv::N_ACTIONS)
    }

    fn temporal_policy(&self) -> Recurrent {
        Recurrent::lstm(HIDDEN_SIZE)
    }
}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Distributed Recurrent PPO (High-Level API) ===");
    println!("Environment: CartPole (operant SIMD, LSTM policy)");
    println!();

    // Configuration - adjusted for recurrent training
    // Key insight: n_epochs > 1 causes hidden state drift/instability
    // Solution: n_epochs=1 with more frequent updates (shorter rollout)
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_rollout_length(32)   // Shorter rollout = more frequent updates
        .with_n_epochs(1)          // Single epoch to avoid hidden state drift
        .with_n_minibatches(2)     // Fewer minibatches
        .with_learning_rate(1e-3)  // Higher LR to compensate for fewer updates
        .with_vf_coef(0.5)
        .with_entropy_coef(0.01)
        .with_target_reward(475.0)
        .with_max_env_steps(500_000);

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
        "  LSTM: hidden_size={}, encoded_size={}",
        HIDDEN_SIZE, ENCODED_SIZE
    );
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = RecurrentPPONet::<B>::new(&device);
    println!("Model created successfully (with LSTM layer).");
    println!();

    // Create runner using the config (uses specialized recurrent runner with TBPTT)
    let runner: DistributedRecurrentPPODiscrete<B> =
        DistributedRecurrentPPORunner::new(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| RecurrentPPONet::<B>::new(device);

    // Environment factory - creates operant SIMD CartPole for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Optimizer
    let optimizer = AdamConfig::new().init::<B, RecurrentPPONet<B>>();

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
            if stats.avg_reward >= 195.0 {
                println!("  >> Solved with Recurrent Policy!");
            }
        },
    );
}
