//! PPO Continuous Example with State-Independent Exploration
//!
//! This example demonstrates multi-actor PPO training with continuous actions
//! using the unified `PPORunner<A, T, B, S>` API with **state-independent exploration**.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//! - Squashed Gaussian policy for bounded continuous actions
//! - **State-Independent Exploration**: log_std is a single learned parameter
//!
//! # Environment: Pendulum-v1
//!
//! Classic inverted pendulum control task:
//! - Observation: [cos(θ), sin(θ), θ_dot] (3D)
//! - Action: torque in [-2.0, 2.0] (1D continuous)
//! - Reward: -θ² - 0.1*θ_dot² - 0.001*torque²
//! - Goal: Balance pendulum upright (θ = 0)
//!
//! # State-Independent Exploration
//!
//!
//! ```text
//! log_std = Param([action_dim])  // same exploration for all states
//! ```
//!
//! Use `ppo_continuous.rs` for the SDE variant.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::tensor::activation::tanh;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{ContinuousPolicy, ContinuousPolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::PendulumEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{PPOConfig, PPOContinuous, PPORunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// Network Definition
// ============================================================================

/// Actor-Critic network for Pendulum with continuous actions.
///
/// Key features:
/// - **Separate networks**: Actor and critic have independent weights
/// - **Tanh activations**: Better gradient flow than ReLU for RL
/// - **Orthogonal initialization**
/// - **Squashed Gaussian policy**: tanh-bounded actions for safety
/// - **State-Independent log_std**: Single learned parameter
#[derive(Module, Debug)]
pub struct PPOContinuousStateIndependentNet<B: burn::tensor::backend::Backend> {
    // Actor network (policy)
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    // State-independent log_std parameter
    log_std: Param<Tensor<B, 1>>,
    // Critic network (value function)
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01; // Small gain for near-uniform initial policy
const VALUE_GAIN: f64 = 1.0; // Standard gain for value head

/// Initial log_std value
const INITIAL_LOG_STD: f32 = 0.0; // std = 1.0 initially

impl<B: burn::tensor::backend::Backend> PPOContinuousStateIndependentNet<B> {
    /// Create a new PPOContinuousStateIndependentNet for Pendulum.
    ///
    /// Initialization strategy:
    /// - Hidden layers: Orthogonal with gain=sqrt(2) for Tanh
    /// - Mean head: Orthogonal with gain=0.01 (keeps initial actions near 0)
    /// - Value head: Orthogonal with gain=1.0
    /// - log_std: Initialized to 0 (std = 1)
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Actor network (policy)
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // State-independent log_std (learned parameter)
            log_std: Param::from_tensor(
                Tensor::zeros([PendulumEnv::ACTION_DIM], device).add_scalar(INITIAL_LOG_STD),
            ),
            // Critic network (value function)
            critic_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            critic_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            critic_head: OrthogonalLinearConfig::new(64, 1)
                .with_gain(VALUE_GAIN)
                .init(device),
        }
    }

    /// Forward pass returning (mean, log_std, values).
    ///
    /// Uses Tanh activations
    /// The log_std is state-independent - same exploration level for all states.
    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];

        // Actor forward (policy mean)
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let mean = self.actor_mean.forward(actor);

        // Broadcast log_std to [batch_size, action_dim]
        // State-independent: same exploration for all states
        let log_std = self
            .log_std
            .val()
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // Critic forward (value function)
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// ActorCriticInference Implementation - For ANY Backend (including Inner)
// ============================================================================

impl<Backend: burn::tensor::backend::Backend>
    ActorCriticInference<Backend, ContinuousPolicy, FeedForward>
    for PPOContinuousStateIndependentNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        _hidden: (),
    ) -> ForwardOutput<Backend, ContinuousPolicy, FeedForward> {
        let (mean, log_std, values) = self.forward_net(obs);

        // Create ContinuousPolicyOutput with action bounds
        let policy_output = ContinuousPolicyOutput::new(
            mean,
            log_std,
            (vec![PendulumEnv::ACTION_LOW], vec![PendulumEnv::ACTION_HIGH]),
        );

        ForwardOutput::new(policy_output, values, ())
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

    fn temporal_policy(&self) -> FeedForward {
        FeedForward::new()
    }
}

// ============================================================================
// ActorCritic Implementation - For AutodiffBackend (Training)
// ============================================================================

impl ActorCritic<B, ContinuousPolicy, FeedForward> for PPOContinuousStateIndependentNet<B> {}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== PPO Continuous State-Independent (Pendulum) ===");
    println!("Environment: Pendulum-v1");
    println!("Action Space: Continuous [-2.0, 2.0]");
    println!("Exploration: State-Independent");
    println!();

    // Configuration tuned for Pendulum
    // Key settings to prevent policy collapse in continuous control:
    // - Non-zero entropy_coef for exploration
    // - target_kl for KL early stopping
    // - reward_normalization for stable gradient scales
    // - log_std_floor to prevent exploration collapse
    let config = PPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(64) // 256 total envs for stable statistics
        .with_rollout_length(256) // Shorter = fresher data, less off-policy
        .with_n_epochs(10) // More epochs for better sample efficiency
        .with_n_minibatches(32) // Smaller minibatches
        .with_learning_rate(2.5e-4) // Slightly lower for stability
        .with_gamma(0.99)
        .with_gae_lambda(0.95)
        .with_entropy_coef(0.005) // Small but non-zero for exploration
        .with_clip_vloss(true)
        .with_anneal_lr(true)
        .with_target_kl(Some(0.02)) // KL early stopping for policy stability
        .with_reward_normalization(true)
        .with_log_std_floor(-2.0) // Prevent exploration collapse (min std = 13.5%)
        .with_target_reward(-200.0) // Good performance threshold
        .with_max_env_steps(5_000_000);

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
    println!("  LR annealing: {}", config.anneal_lr);
    println!("  Clipped value loss: {}", config.clip_vloss);
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = PPOContinuousStateIndependentNet::<B>::new(&device);
    println!("Model created (Orthogonal init, state-independent log_std).");
    println!();

    // Create unified runner using PPORunner::feed_forward()
    let runner: PPOContinuous<B> = PPORunner::feed_forward(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| PPOContinuousStateIndependentNet::<B>::new(device);

    // Environment factory - creates Pendulum for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
    };

    // Optimizer with gradient clipping (critical for PPO stability)
    let optimizer = runner.create_optimizer::<PPOContinuousStateIndependentNet<B>>();

    // Run training with progress callback
    println!("Starting training (state-independent exploration)...");
    println!("Target reward: -200.0 (expected ~1.5-2.5M steps)");
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
