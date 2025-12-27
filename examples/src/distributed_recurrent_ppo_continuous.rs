//! Distributed PPO Example - Continuous Actions with Recurrent/LSTM (High-Level API)
//!
//! This example demonstrates multi-actor PPO training with continuous actions
//! on Pendulum using a recurrent (LSTM) actor-critic network.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - LSTM for temporal dependencies (memory across timesteps)
//! - Squashed Gaussian policy for bounded continuous actions
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//!
//! # Environment
//!
//! Uses operant's SIMD-optimized Pendulum environment for high throughput.
//! Operant environments use f32x8 SIMD vectorization for parallel physics.
//!
//! # When to Use Recurrent + Continuous
//!
//! This combination is useful for:
//! - Continuous control with partial observability
//! - Tasks requiring memory of past observations
//! - Robotics with noisy sensors
//!
//! Run with: `LIBRARY_PATH=/opt/homebrew/lib cargo run --features distributed --release -- distributed-recurrent-ppo-continuous`

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{ContinuousPolicy, ContinuousPolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::{Recurrent, RecurrentHidden};
use distributed_rl::core::recurrent::{LstmCellConfig, LstmCellWrapper, RecurrentCell};
use distributed_rl::environment::PendulumEnv;
use distributed_rl::runners::{DistributedPPOConfig, DistributedRecurrentPPOContinuous, DistributedRecurrentPPORunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// LSTM configuration
const HIDDEN_SIZE: usize = 64;
const ENCODED_SIZE: usize = 32;

// Log std bounds for numerical stability
const LOG_STD_MIN: f32 = -20.0;
const LOG_STD_MAX: f32 = 2.0;

// ============================================================================
// Network Definition
// ============================================================================

/// Recurrent Actor-Critic network with LSTM for continuous PPO.
///
/// Architecture: Encoder -> LSTM -> Policy(mean, log_std)/Value heads
#[derive(Module, Debug)]
pub struct RecurrentContinuousPPONet<B: burn::tensor::backend::Backend> {
    encoder: Linear<B>,
    lstm: LstmCellWrapper<B>,
    policy_mean: Linear<B>,
    policy_log_std: Linear<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> RecurrentContinuousPPONet<B> {
    /// Create a new RecurrentContinuousPPONet for Pendulum.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(PendulumEnv::OBS_SIZE, ENCODED_SIZE).init(device),
            lstm: LstmCellConfig::new(ENCODED_SIZE, HIDDEN_SIZE).init(device),
            policy_mean: LinearConfig::new(HIDDEN_SIZE, PendulumEnv::ACTION_DIM).init(device),
            policy_log_std: LinearConfig::new(HIDDEN_SIZE, PendulumEnv::ACTION_DIM).init(device),
            value_head: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
        }
    }
}

// ============================================================================
// ActorCriticInference Implementation - For ANY Backend (including Inner)
// ============================================================================

impl<Backend: burn::tensor::backend::Backend> ActorCriticInference<Backend, ContinuousPolicy, Recurrent>
    for RecurrentContinuousPPONet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        hidden: RecurrentHidden<Backend>,
    ) -> ForwardOutput<Backend, ContinuousPolicy, Recurrent> {
        let batch_size = obs.dims()[0];
        let device = obs.device();

        // Encode -> LSTM
        let encoded = relu(self.encoder.forward(obs));
        let batched_hidden = hidden.to_batched_state();
        let (_output, new_hidden_state) = self.lstm.step(encoded, &batched_hidden);

        // Policy and value from LSTM hidden state
        let h = new_hidden_state.hidden.clone();
        let mean = tanh(self.policy_mean.forward(h.clone()));
        let log_std = self
            .policy_log_std
            .forward(h.clone())
            .clamp(LOG_STD_MIN, LOG_STD_MAX);
        let value = self.value_head.forward(h);

        // Update hidden states
        let mut new_hidden = RecurrentHidden::new(batch_size, HIDDEN_SIZE, true, &device);
        new_hidden.from_batched_state(new_hidden_state);

        ForwardOutput::new(
            ContinuousPolicyOutput {
                mean,
                log_std,
                bounds: PendulumEnv::action_bounds(),
            },
            value,
            new_hidden,
        )
    }

    fn obs_size(&self) -> usize {
        PendulumEnv::OBS_SIZE
    }

    fn action_policy(&self) -> ContinuousPolicy {
        let (low, high) = PendulumEnv::action_bounds();
        ContinuousPolicy::new(PendulumEnv::ACTION_DIM, low, high)
    }

    fn temporal_policy(&self) -> Recurrent {
        Recurrent::lstm(HIDDEN_SIZE)
    }
}

// ============================================================================
// ActorCritic Implementation - For AutodiffBackend (Training)
// ============================================================================

impl ActorCritic<B, ContinuousPolicy, Recurrent> for RecurrentContinuousPPONet<B> {}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Distributed Recurrent PPO Continuous (High-Level API) ===");
    println!("Environment: Pendulum (operant SIMD, LSTM policy)");
    println!();

    // Configuration - adjusted for recurrent + continuous training
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(16) // Smaller per-actor for recurrent
        .with_rollout_length(128)
        .with_n_epochs(4)
        .with_n_minibatches(4)
        .with_learning_rate(3e-4)
        .with_max_env_steps(2_000_000);

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
    println!(
        "  Action bounds: [{}, {}]",
        PendulumEnv::ACTION_LOW,
        PendulumEnv::ACTION_HIGH
    );
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = RecurrentContinuousPPONet::<B>::new(&device);
    println!("Model created successfully (LSTM + continuous policy).");
    println!();

    // Create runner using the config (uses specialized recurrent runner with TBPTT)
    let runner: DistributedRecurrentPPOContinuous<B> =
        DistributedRecurrentPPORunner::new(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| RecurrentContinuousPPONet::<B>::new(device);

    // Environment factory - creates operant SIMD Pendulum for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
    };

    // Optimizer
    let optimizer = AdamConfig::new().init::<B, RecurrentContinuousPPONet<B>>();

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
                println!("  >> Excellent Performance with Recurrent Policy!");
            }
        },
    );
}
