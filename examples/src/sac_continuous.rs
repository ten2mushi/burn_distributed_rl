//! SAC Continuous Example for Pendulum
//!
//! This example demonstrates SAC training with continuous actions
//! using the `SACRunner<A, T, B, S>` API with separate actor and critic networks.
//!
//! # Architecture (Different from Discrete SAC!)
//!
//! For continuous actions, the critic takes (state, action) as input:
//!
//! ```text
//! Actor Network:
//! ├── encoder: MLP (obs → hidden)
//! ├── mean_head: Linear → mean [batch, action_dim]
//! └── log_std_head: Linear → log_std [batch, action_dim]
//!
//! Critic Network (Twin Q):
//! ├── Q1: MLP(concat(obs, action)) → [batch, 1]
//! └── Q2: MLP(concat(obs, action)) → [batch, 1]
//! ```
//!
//! # Continuous SAC
//!
//! For continuous actions, SAC uses:
//! - Squashed Gaussian policy: actions = tanh(sample) * scale
//! - Critic evaluates Q(s,a) for sampled actions
//! - Log prob correction for tanh squashing
//!
//! # Environment: Pendulum-v1
//!
//! Classic inverted pendulum control task:
//! - Observation: [cos(θ), sin(θ), θ_dot] (3D)
//! - Action: torque in [-2.0, 2.0] (1D continuous)
//! - Reward: -θ² - 0.1*θ_dot² - 0.001*torque²
//! - Goal: Balance pendulum upright (θ = 0)

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{ContinuousPolicy, ContinuousPolicyOutput};
use distributed_rl::algorithms::sac::{
    clamp_log_std, SACActor, SACActorOutput, SACCritic, SACCriticOutput,
};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::PendulumEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{SACConfig, SACContinuous, SACRunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// SAC Actor Network (Continuous)
// ============================================================================

/// SAC Actor network for Pendulum (continuous actions).
///
/// Outputs mean and log_std for a squashed Gaussian distribution.
/// Actions are sampled as: action = tanh(mean + std * noise) * scale
#[derive(Module, Debug)]
pub struct SACActorContinuousNet<B: Backend> {
    // Encoder
    encoder_0: OrthogonalLinear<B>,
    encoder_1: OrthogonalLinear<B>,
    // Mean and log_std heads
    mean_head: OrthogonalLinear<B>,
    log_std_head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01; // Small gain for near-zero initial actions

impl<B: Backend> SACActorContinuousNet<B> {
    /// Create a new SAC actor for Pendulum.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 256)
                .with_gain(SQRT_2)
                .init(device),
            encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            mean_head: OrthogonalLinearConfig::new(256, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            log_std_head: OrthogonalLinearConfig::new(256, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
        }
    }

    /// Forward pass returning (mean, log_std).
    fn forward_net(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = tanh(self.encoder_0.forward(obs));
        let x = tanh(self.encoder_1.forward(x));

        let mean = self.mean_head.forward(x.clone());
        let raw_log_std = self.log_std_head.forward(x);

        // Clamp log_std to safe bounds for numerical stability
        let log_std = clamp_log_std(raw_log_std);

        (mean, log_std)
    }
}

// Implement SACActor for any Backend
impl<Backend: burn::tensor::backend::Backend> SACActor<Backend, ContinuousPolicy, FeedForward>
    for SACActorContinuousNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        _hidden: (),
    ) -> SACActorOutput<Backend, ContinuousPolicy, FeedForward> {
        let (mean, log_std) = self.forward_net(obs);

        // Create policy output with action bounds
        let policy = ContinuousPolicyOutput::new(
            mean,
            log_std,
            (vec![PendulumEnv::ACTION_LOW], vec![PendulumEnv::ACTION_HIGH]),
        );

        SACActorOutput::new(policy, ())
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
// SAC Critic Network (Twin Q, Continuous)
// ============================================================================

/// SAC Critic network with twin Q-networks for Pendulum.
///
/// For continuous actions, takes (state, action) as input:
/// - Input: concat(obs, action) = [batch, obs_dim + action_dim]
/// - Output: Q(s,a) = [batch, 1]
#[derive(Module, Debug)]
pub struct SACCriticContinuousNet<B: Backend> {
    // Q1 network: takes concat(obs, action)
    q1_encoder_0: OrthogonalLinear<B>,
    q1_encoder_1: OrthogonalLinear<B>,
    q1_head: OrthogonalLinear<B>,
    // Q2 network
    q2_encoder_0: OrthogonalLinear<B>,
    q2_encoder_1: OrthogonalLinear<B>,
    q2_head: OrthogonalLinear<B>,
}

const VALUE_GAIN: f64 = 1.0;

impl<B: Backend> SACCriticContinuousNet<B> {
    /// Create a new SAC critic for Pendulum.
    pub fn new(device: &B::Device) -> Self {
        let input_size = PendulumEnv::OBS_SIZE + PendulumEnv::ACTION_DIM;

        Self {
            // Q1 network
            q1_encoder_0: OrthogonalLinearConfig::new(input_size, 256)
                .with_gain(SQRT_2)
                .init(device),
            q1_encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            q1_head: OrthogonalLinearConfig::new(256, 1)
                .with_gain(VALUE_GAIN)
                .init(device),
            // Q2 network
            q2_encoder_0: OrthogonalLinearConfig::new(input_size, 256)
                .with_gain(SQRT_2)
                .init(device),
            q2_encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            q2_head: OrthogonalLinearConfig::new(256, 1)
                .with_gain(VALUE_GAIN)
                .init(device),
        }
    }

    /// Forward pass for Q1.
    fn forward_q1(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate obs and action along feature dimension
        let x = Tensor::cat(vec![obs, action], 1);
        let x = tanh(self.q1_encoder_0.forward(x));
        let x = tanh(self.q1_encoder_1.forward(x));
        self.q1_head.forward(x)
    }

    /// Forward pass for Q2.
    fn forward_q2(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![obs, action], 1);
        let x = tanh(self.q2_encoder_0.forward(x));
        let x = tanh(self.q2_encoder_1.forward(x));
        self.q2_head.forward(x)
    }
}

// Implement SACCritic for any Backend
impl<Backend: burn::tensor::backend::Backend> SACCritic<Backend, ContinuousPolicy, FeedForward>
    for SACCriticContinuousNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        action: Option<Tensor<Backend, 2>>,
        _hidden: (),
    ) -> SACCriticOutput<Backend, FeedForward> {
        // For continuous, action must be provided
        let action = action.expect("Continuous SAC critic requires action input");

        let q1 = self.forward_q1(obs.clone(), action.clone());
        let q2 = self.forward_q2(obs, action);

        SACCriticOutput::new(q1, q2, ())
    }

    fn obs_size(&self) -> usize {
        PendulumEnv::OBS_SIZE
    }

    fn action_dim(&self) -> usize {
        PendulumEnv::ACTION_DIM
    }

    fn temporal_policy(&self) -> FeedForward {
        FeedForward::new()
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== SAC Continuous (Pendulum) ===");
    println!("Environment: Pendulum-v1");
    println!("Action Space: Continuous [-2.0, 2.0]");
    println!("Architecture: Separate Actor + Twin Q Critics");
    println!();

    // SAC configuration for continuous actions
    // Standard SAC hyperparameters for MuJoCo-like tasks
    let config = SACConfig::continuous()
        .with_n_actors(2)
        .with_n_envs_per_actor(8) // 16 total envs
        .with_buffer_capacity(1_000_000)
        .with_batch_size(256)
        .with_min_buffer_size(5000) // More warmup for continuous
        .with_actor_lr(3e-4)
        .with_critic_lr(3e-4)
        .with_alpha_lr(3e-4)
        .with_gamma(0.99)
        .with_tau(0.005) // Soft target updates
        .with_auto_entropy_tuning(true)
        .with_policy_update_freq(2) // Delayed policy updates
        .with_target_reward(-200.0) // Good performance for Pendulum
        .with_max_env_steps(500_000);

    println!("Configuration:");
    println!(
        "  Actors: {} x {} envs = {} total",
        config.n_actors,
        config.n_envs_per_actor,
        config.total_envs()
    );
    println!(
        "  Buffer: capacity={}, min_size={}, batch={}",
        config.buffer_capacity, config.min_buffer_size, config.batch_size
    );
    println!("  Tau: {} (soft target updates)", config.tau);
    println!("  Policy update freq: {} (delayed updates)", config.policy_update_freq);
    println!("  Auto entropy tuning: {}", config.auto_entropy_tuning);
    println!();

    // Initialize device and models
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_actor = SACActorContinuousNet::<B>::new(&device);
    let initial_critic = SACCriticContinuousNet::<B>::new(&device);
    println!("Models created (Orthogonal init, squashed Gaussian policy).");
    println!();

    // Create SAC runner for continuous actions
    let runner: SACContinuous<B> = SACRunner::feed_forward(config.clone());

    // Model factories
    let actor_factory = |device: &WgpuDevice| SACActorContinuousNet::<B>::new(device);
    let critic_factory = |device: &WgpuDevice| SACCriticContinuousNet::<B>::new(device);

    // Environment factory
    let env_factory = |_actor_id: usize, n_envs: usize| {
        PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
    };

    // Create optimizers
    let (actor_optimizer, critic_optimizer) =
        runner.create_optimizers::<SACActorContinuousNet<B>, SACCriticContinuousNet<B>>();

    // Run training
    println!("Starting SAC training...");
    println!("Target reward: -200.0 (lower is better for Pendulum)");
    println!();

    let (_trained_actor, _trained_critic) = runner.run(
        actor_factory,
        critic_factory,
        initial_actor,
        initial_critic,
        env_factory,
        actor_optimizer,
        critic_optimizer,
        |stats| {
            println!(
                "Steps: {:>8} | Episodes: {:>5} | Return: {:>8.1} | Buffer: {:>5.1}% | SPS: {:>6.0}",
                stats.env_steps,
                stats.episodes,
                stats.mean_return,
                stats.buffer_utilization * 100.0,
                stats.sps
            );
        },
    );
}
