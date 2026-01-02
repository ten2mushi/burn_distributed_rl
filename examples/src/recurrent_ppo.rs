//! Recurrent PPO Example using LSTM-Based Policy
//!
//! This example demonstrates multi-actor PPO training with recurrent (LSTM)
//! policies using the unified `PPORunner<A, T, B, S>` API.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - LSTM-based policy for temporal dependencies
//! - TBPTT (Truncated Backpropagation Through Time) for training
//!
//! # Key Differences from Feed-Forward PPO
//!
//! - Network includes LSTM layer for temporal memory
//! - Hidden state maintained per-environment during rollout
//! - Training uses TBPTT chunks with proper hidden state handling
//! - Done flags trigger hidden state reset
//!
//! # Network:
//! - Inline hidden state reset: (1-done) * h
//! - GAE computed over full rollout before TBPTT chunking
//! - Same initial hidden state for all training epochs

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::activation::tanh;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::{HiddenConfig, Recurrent};
use distributed_rl::core::recurrent::{HiddenState, LstmCellConfig, LstmCellWrapper, RecurrentCell};
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{PPOConfig, PPORunner, RecurrentPPODiscrete};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// LSTM-Based Actor-Critic Network
// ============================================================================

/// LSTM-based Actor-Critic network for CartPole.
///
/// Architecture:
/// - Encoder: 2 hidden layers (64 units each, Tanh activation)
/// - LSTM: Single layer with configurable hidden size
/// - Actor head: Linear to action logits
/// - Critic head: Linear to value estimate
///
/// Key features:
/// - Separate actor/critic networks (no shared trunk, except LSTM)
/// - Orthogonal initialization for improved training
/// - Maintains per-environment hidden state
#[derive(Module, Debug)]
pub struct RecurrentPPONet<B: burn::tensor::backend::Backend> {
    // Encoder (shared feature extraction)
    encoder_0: OrthogonalLinear<B>,
    encoder_1: OrthogonalLinear<B>,
    // LSTM for temporal memory
    lstm: LstmCellWrapper<B>,
    // Actor head (policy)
    actor_head: OrthogonalLinear<B>,
    // Critic head (value function)
    critic_head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01;  // Small gain for near-uniform initial policy
const VALUE_GAIN: f64 = 1.0;    // Standard gain for value head

/// LSTM hidden size
const LSTM_HIDDEN_SIZE: usize = 64;

impl<B: burn::tensor::backend::Backend> RecurrentPPONet<B> {
    /// Create a new LSTM-based PPONet for CartPole.
    ///
    /// Initialization strategy:
    /// - Hidden layers: Orthogonal with gain=sqrt(2) for Tanh
    /// - Policy head: Orthogonal with gain=0.01 (keeps initial policy near uniform)
    /// - Value head: Orthogonal with gain=1.0
    /// - LSTM: Burn's default initialization
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Encoder network
            encoder_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            encoder_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            // LSTM for temporal processing
            lstm: LstmCellConfig::new(64, LSTM_HIDDEN_SIZE).init(device),
            // Actor head (policy logits)
            actor_head: OrthogonalLinearConfig::new(LSTM_HIDDEN_SIZE, CartPoleEnv::N_ACTIONS)
                .with_gain(POLICY_GAIN)
                .init(device),
            // Critic head (value estimate)
            critic_head: OrthogonalLinearConfig::new(LSTM_HIDDEN_SIZE, 1)
                .with_gain(VALUE_GAIN)
                .init(device),
        }
    }

    /// Forward pass with hidden state.
    ///
    /// # Arguments
    ///
    /// * `x` - Observation tensor [batch, obs_dim]
    /// * `hidden` - LSTM hidden state (h, c)
    ///
    /// # Returns
    ///
    /// (logits, values, new_hidden)
    pub fn forward_net(
        &self,
        x: Tensor<B, 2>,
        hidden: HiddenState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, HiddenState<B>) {
        // Encode observation
        let encoded = tanh(self.encoder_0.forward(x));
        let encoded = tanh(self.encoder_1.forward(encoded));

        // LSTM step
        let (lstm_out, new_hidden) = self.lstm.step(encoded, &hidden);

        // Actor head (policy logits)
        let logits = self.actor_head.forward(lstm_out.clone());

        // Critic head (value estimate)
        let values = self.critic_head.forward(lstm_out);

        (logits, values, new_hidden)
    }

    /// Get initial hidden state for a batch of environments.
    #[allow(dead_code)]
    pub fn initial_hidden(&self, batch_size: usize, device: &B::Device) -> HiddenState<B> {
        self.lstm.initial_state(batch_size, device)
    }
}

// ============================================================================
// ActorCriticInference Implementation - For ANY Backend (including Inner)
// ============================================================================

impl<Backend: burn::tensor::backend::Backend>
    ActorCriticInference<Backend, DiscretePolicy, Recurrent> for RecurrentPPONet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        hidden: <Recurrent as distributed_rl::algorithms::temporal_policy::TemporalPolicy<Backend>>::Hidden,
    ) -> ForwardOutput<Backend, DiscretePolicy, Recurrent> {
        // Convert per-environment hidden states to batched HiddenState for forward pass
        let batched_hidden = hidden.to_batched_state();

        // Forward through network
        let (logits, values, new_batched_hidden) = self.forward_net(obs, batched_hidden);

        // Update hidden states from batch results
        let mut new_hidden = hidden.clone();
        new_hidden.from_batched_state(new_batched_hidden);

        ForwardOutput::new(DiscretePolicyOutput { logits }, values, new_hidden)
    }

    fn obs_size(&self) -> usize {
        CartPoleEnv::OBS_SIZE
    }

    fn action_policy(&self) -> DiscretePolicy {
        DiscretePolicy::new(CartPoleEnv::N_ACTIONS)
    }

    fn temporal_policy(&self) -> Recurrent {
        Recurrent::lstm(LSTM_HIDDEN_SIZE)
    }
}

// ============================================================================
// ActorCritic Implementation - For AutodiffBackend (Training)
// ============================================================================

impl ActorCritic<B, DiscretePolicy, Recurrent> for RecurrentPPONet<B> {}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Recurrent PPO Runner (LSTM) ===");
    println!("Environment: CartPole");
    println!("Policy: LSTM-based recurrent policy");
    println!("Strategy: RecurrentStrategy (TBPTT training)");
    println!();

    // Configuration for recurrent PPO
    //
    // PERFORMANCE NOTE: TBPTT throughput scales with:
    //   SPS ∝ n_envs / (tbptt_length × overhead)
    //
    // Previous config (low SPS ~75-80):
    //   - 4 envs: tiny batch, poor GPU utilization
    //   - 128 TBPTT: 128 sequential forward passes per batch
    //   - 1 sequence/batch: minimal parallelism
    //
    // Improved config for better throughput:
    //   - 16 envs: better GPU utilization
    //   - 32 TBPTT: 4x shallower graph, still learns temporal
    //   - 4 sequences/batch: more parallelism
    //
    // Note: CartPole episodes are short (~20-200 steps), so 32 TBPTT
    // is sufficient to capture most temporal dependencies.
    let config = PPOConfig::new()
        .with_n_actors(2)
        .with_n_envs_per_actor(8)     // 16 total envs (was 4)
        .with_rollout_length(128)
        .with_n_epochs(4)
        .with_n_minibatches(4)        // 16 sequences / 4 = 4 sequences per batch
        .with_tbptt_length(32)        // Reduced from 128 (4x faster training)
        .with_learning_rate(2.5e-4)
        .with_entropy_coef(0.01)
        .with_clip_vloss(true)
        .with_anneal_lr(true)
        .with_target_kl(Some(0.015))
        .with_target_reward(475.0)
        .with_max_env_steps(500_000);

    // Hidden state configuration for LSTM
    let hidden_config = HiddenConfig::lstm(LSTM_HIDDEN_SIZE);

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
    println!("  TBPTT length: {}", config.tbptt_length);
    println!("  LSTM hidden size: {}", LSTM_HIDDEN_SIZE);
    println!("  LR annealing: {}", config.anneal_lr);
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = RecurrentPPONet::<B>::new(&device);
    println!("Model created (LSTM-based, Orthogonal init).");
    println!();

    // Create recurrent PPO runner
    // Note: Uses PPORunner::new() with hidden_config for recurrent policies
    let runner: RecurrentPPODiscrete<B> = PPORunner::new(config.clone(), hidden_config);

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| RecurrentPPONet::<B>::new(device);

    // Environment factory - creates operant SIMD CartPole for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Optimizer with gradient clipping
    let optimizer = runner.create_optimizer::<RecurrentPPONet<B>>();

    // Run training with progress callback
    println!("Starting recurrent PPO training...");
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
