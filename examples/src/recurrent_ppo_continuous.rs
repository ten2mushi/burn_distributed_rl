//! Recurrent PPO Continuous Example using LSTM-Based Policy
//!
//! This example demonstrates multi-actor PPO training with recurrent (LSTM)
//! policies for continuous action spaces using the unified `PPORunner<A, T, B, S>` API.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - LSTM-based policy for temporal dependencies
//! - TBPTT (Truncated Backpropagation Through Time) for training
//! - Squashed Gaussian policy for bounded continuous actions
//!
//! # Environment: Pendulum-v1
//!
//! Classic inverted pendulum control task:
//! - Observation: [cos(θ), sin(θ), θ_dot] (3D)
//! - Action: torque in [-2.0, 2.0] (1D continuous)
//! - Reward: -θ² - 0.1*θ_dot² - 0.001*torque²
//! - Goal: Balance pendulum upright (θ = 0)
//!
//! # Network Architecture
//! - Inline hidden state reset: (1-done) * h
//! - GAE computed over full rollout before TBPTT chunking
//! - Same initial hidden state for all training epochs

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::tensor::activation::tanh;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{ContinuousPolicy, ContinuousPolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::{HiddenConfig, Recurrent};
use distributed_rl::core::recurrent::{HiddenState, LstmCellConfig, LstmCellWrapper, RecurrentCell};
use distributed_rl::environment::PendulumEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{PPOConfig, PPORunner, RecurrentPPOContinuous};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// LSTM-Based Actor-Critic Network for Continuous Actions
// ============================================================================

/// LSTM-based Actor-Critic network for Pendulum with continuous actions.
///
/// Architecture:
/// - Encoder: 2 hidden layers (64 units each, Tanh activation)
/// - LSTM: Single layer with configurable hidden size
/// - Actor head: Linear to action mean
/// - State-independent log_std parameter
/// - Critic head: Linear to value estimate
///
/// Key features:
/// - Separate actor/critic networks (no shared trunk, except LSTM)
/// - Orthogonal initialization for improved training
/// - Squashed Gaussian policy for bounded actions
/// - Maintains per-environment hidden state
#[derive(Module, Debug)]
pub struct RecurrentPPOContinuousNet<B: burn::tensor::backend::Backend> {
    // Encoder (shared feature extraction)
    encoder_0: OrthogonalLinear<B>,
    encoder_1: OrthogonalLinear<B>,
    // LSTM for temporal memory
    lstm: LstmCellWrapper<B>,
    // Actor head (policy mean)
    actor_mean: OrthogonalLinear<B>,
    // State-independent log_std parameter
    log_std: Param<Tensor<B, 1>>,
    // Critic head (value function)
    critic_head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01;  // Small gain for near-uniform initial policy
const VALUE_GAIN: f64 = 1.0;    // Standard gain for value head

/// Initial log_std value
const INITIAL_LOG_STD: f32 = 0.0; // std = 1.0 initially

/// LSTM hidden size
const LSTM_HIDDEN_SIZE: usize = 64;

impl<B: burn::tensor::backend::Backend> RecurrentPPOContinuousNet<B> {
    /// Create a new LSTM-based PPOContinuousNet for Pendulum.
    ///
    /// Initialization strategy:
    /// - Hidden layers: Orthogonal with gain=sqrt(2) for Tanh
    /// - Mean head: Orthogonal with gain=0.01 (keeps initial actions near 0)
    /// - Value head: Orthogonal with gain=1.0
    /// - log_std: Initialized to 0 (std = 1)
    /// - LSTM: Burn's default initialization
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Encoder network
            encoder_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            encoder_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            // LSTM for temporal processing
            lstm: LstmCellConfig::new(64, LSTM_HIDDEN_SIZE).init(device),
            // Actor head (policy mean)
            actor_mean: OrthogonalLinearConfig::new(LSTM_HIDDEN_SIZE, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // State-independent log_std (learned parameter)
            log_std: Param::from_tensor(
                Tensor::zeros([PendulumEnv::ACTION_DIM], device).add_scalar(INITIAL_LOG_STD),
            ),
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
    /// (mean, log_std, values, new_hidden)
    pub fn forward_net(
        &self,
        x: Tensor<B, 2>,
        hidden: HiddenState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, HiddenState<B>) {
        let batch_size = x.dims()[0];

        // Encode observation
        let encoded = tanh(self.encoder_0.forward(x));
        let encoded = tanh(self.encoder_1.forward(encoded));

        // LSTM step
        let (lstm_out, new_hidden) = self.lstm.step(encoded, &hidden);

        // Actor head (policy mean)
        let mean = self.actor_mean.forward(lstm_out.clone());

        // Broadcast log_std to [batch_size, action_dim]
        let log_std = self
            .log_std
            .val()
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // Critic head (value estimate)
        let values = self.critic_head.forward(lstm_out);

        (mean, log_std, values, new_hidden)
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
    ActorCriticInference<Backend, ContinuousPolicy, Recurrent> for RecurrentPPOContinuousNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        hidden: <Recurrent as distributed_rl::algorithms::temporal_policy::TemporalPolicy<Backend>>::Hidden,
    ) -> ForwardOutput<Backend, ContinuousPolicy, Recurrent> {
        let _batch_size = obs.dims()[0];
        let _device = obs.device();

        // Convert per-environment hidden states to batched HiddenState for forward pass
        let batched_hidden = hidden.to_batched_state();

        // Forward through network
        let (mean, log_std, values, new_batched_hidden) = self.forward_net(obs, batched_hidden);

        // Update hidden states from batch results
        let mut new_hidden = hidden.clone();
        new_hidden.from_batched_state(new_batched_hidden);

        // Create ContinuousPolicyOutput with action bounds
        let policy_output = ContinuousPolicyOutput::new(
            mean,
            log_std,
            (vec![PendulumEnv::ACTION_LOW], vec![PendulumEnv::ACTION_HIGH]),
        );

        ForwardOutput::new(policy_output, values, new_hidden)
    }

    fn obs_size(&self) -> usize {
        PendulumEnv::OBS_SIZE
    }

    fn action_policy(&self) -> ContinuousPolicy {
        ContinuousPolicy::new(
            PendulumEnv::ACTION_DIM,
            vec![PendulumEnv::ACTION_LOW],
            vec![PendulumEnv::ACTION_HIGH],
        )
    }

    fn temporal_policy(&self) -> Recurrent {
        Recurrent::lstm(LSTM_HIDDEN_SIZE)
    }
}

// ============================================================================
// ActorCritic Implementation - For AutodiffBackend (Training)
// ============================================================================

impl ActorCritic<B, ContinuousPolicy, Recurrent> for RecurrentPPOContinuousNet<B> {}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== Recurrent PPO Continuous Runner (LSTM) ===");
    println!("Environment: Pendulum-v1");
    println!("Action Space: Continuous [-2.0, 2.0]");
    println!("Policy: LSTM-based recurrent policy");
    println!("Strategy: RecurrentStrategy (TBPTT training)");
    println!();

    // Configuration tuned for Pendulum with recurrent policy
    //
    // CRITICAL: TBPTT training time scales as:
    //   sequential_passes = tbptt_length × n_epochs × n_minibatches
    //
    // The bottleneck is NOT batch size, but the number of sequential
    // LSTM forward/backward passes. Each minibatch requires tbptt_length
    // sequential steps, so more minibatches = more sequential overhead.
    //
    // Config analysis:
    //   - 512 rollout × 16 envs = 8k transitions
    //   - 8k / 32 tbptt = 256 sequences
    //   - 256 / 4 minibatches = 64 sequences/batch (good GPU parallelism)
    //   - Sequential passes: 32 × 4 × 4 = 512 (matches discrete)
    let config = PPOConfig::new()
        .with_n_actors(2)
        .with_n_envs_per_actor(8)       // 16 total envs
        .with_rollout_length(512)       // Moderate rollout
        .with_n_epochs(4)               // Standard PPO epochs
        .with_n_minibatches(4)          // KEY: Keep low for TBPTT efficiency
        .with_tbptt_length(32)          // Shallow enough for fast backward
        .with_learning_rate(3e-4)       // Standard PPO learning rate
        .with_gamma(0.99)
        .with_gae_lambda(0.95)
        .with_entropy_coef(0.0)         // Low/no entropy for continuous (exploration via std)
        .with_clip_vloss(true)
        .with_anneal_lr(true)
        .with_target_kl(None)           // No early stopping for continuous
        .with_target_reward(-200.0)     // Good performance threshold
        .with_max_env_steps(1_000_000);

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
    println!("  Clipped value loss: {}", config.clip_vloss);
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = RecurrentPPOContinuousNet::<B>::new(&device);
    println!("Model created (LSTM-based, Orthogonal init).");
    println!();

    // Create recurrent PPO runner for continuous actions
    // Note: Uses PPORunner::new() with hidden_config for recurrent policies
    let runner: RecurrentPPOContinuous<B> = PPORunner::new(config.clone(), hidden_config);

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| RecurrentPPOContinuousNet::<B>::new(device);

    // Environment factory - creates Pendulum for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
    };

    // Optimizer with gradient clipping (critical for PPO stability)
    let optimizer = runner.create_optimizer::<RecurrentPPOContinuousNet<B>>();

    // Run training with progress callback
    println!("Starting recurrent PPO continuous training...");
    println!("Target reward: -200.0 (lower is better for Pendulum)");
    println!();

    let _trained_model = runner.run(
        model_factory,
        initial_model,
        env_factory,
        optimizer,
        |stats| {
            println!(
                "Steps: {:>8} | Episodes: {:>5} | Reward: {:>8.1} | Version: {:>4} | SPS: {:>6.0}",
                stats.env_steps,
                stats.episodes,
                stats.avg_reward,
                stats.policy_version,
                stats.steps_per_second
            );
        },
    );
}
