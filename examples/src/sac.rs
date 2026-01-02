//! SAC Discrete Example for CartPole
//!
//! This example demonstrates SAC training with discrete actions
//! using the `SACRunner<A, T, B, S>` API with separate actor and critic networks.
//!
//! # Architecture (Different from PPO!)
//!
//! SAC uses **separate** actor and critic networks, unlike PPO's combined ActorCritic:
//!
//! ```text
//! Actor Network:
//! ├── encoder: MLP (obs → hidden)
//! └── head: Linear → logits [batch, n_actions]
//!
//! Critic Network (Twin Q):
//! ├── Q1: encoder1 + head1 → [batch, n_actions]
//! └── Q2: encoder2 + head2 → [batch, n_actions]
//! ```
//!
//! # Discrete SAC
//!
//! For discrete actions, SAC computes the full expectation over all actions:
//! - Critic outputs Q(s,a) for ALL actions: [batch, n_actions]
//! - No action input to critic (action=None in forward)
//! - Actor loss uses: Σ_a π(a|s) * (α*log π(a|s) - Q(s,a))
//!
//! # Key Differences from PPO
//!
//! - Off-policy: Uses replay buffer instead of on-policy rollouts
//! - Separate optimizers: Actor, Critic, and Alpha have separate LRs
//! - Target networks: Critic has slowly-updated target copies
//! - Entropy tuning: Alpha is learned automatically
//! - Twin Q: Two Q-networks prevent overestimation

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::sac::{SACActor, SACActorOutput, SACCritic, SACCriticOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{SACConfig, SACDiscrete, SACRunner};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// SAC Actor Network
// ============================================================================

/// SAC Actor network for CartPole (discrete actions).
///
/// Outputs logits for a categorical distribution over actions.
/// Uses separate encoder from critic (key SAC design choice).
#[derive(Module, Debug)]
pub struct SACActorNet<B: Backend> {
    // Encoder
    encoder_0: OrthogonalLinear<B>,
    encoder_1: OrthogonalLinear<B>,
    // Policy head
    head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01; // Small gain for near-uniform initial policy

impl<B: Backend> SACActorNet<B> {
    /// Create a new SAC actor for CartPole.
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 256)
                .with_gain(SQRT_2)
                .init(device),
            encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            head: OrthogonalLinearConfig::new(256, CartPoleEnv::N_ACTIONS)
                .with_gain(POLICY_GAIN)
                .init(device),
        }
    }

    /// Forward pass returning logits.
    fn forward_net(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = tanh(self.encoder_0.forward(obs));
        let x = tanh(self.encoder_1.forward(x));
        self.head.forward(x)
    }
}

// Implement SACActor for any Backend
impl<Backend: burn::tensor::backend::Backend> SACActor<Backend, DiscretePolicy, FeedForward>
    for SACActorNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        _hidden: (),
    ) -> SACActorOutput<Backend, DiscretePolicy, FeedForward> {
        let logits = self.forward_net(obs);
        SACActorOutput::new(DiscretePolicyOutput { logits }, ())
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
// SAC Critic Network (Twin Q)
// ============================================================================

/// SAC Critic network with twin Q-networks for CartPole.
///
/// For discrete actions, outputs Q-values for ALL actions: [batch, n_actions].
/// No action input - the full Q-table is computed in one forward pass.
#[derive(Module, Debug)]
pub struct SACCriticNet<B: Backend> {
    // Q1 network
    q1_encoder_0: OrthogonalLinear<B>,
    q1_encoder_1: OrthogonalLinear<B>,
    q1_head: OrthogonalLinear<B>,
    // Q2 network (separate for independence)
    q2_encoder_0: OrthogonalLinear<B>,
    q2_encoder_1: OrthogonalLinear<B>,
    q2_head: OrthogonalLinear<B>,
}

const VALUE_GAIN: f64 = 1.0; // Standard gain for value heads

impl<B: Backend> SACCriticNet<B> {
    /// Create a new SAC critic for CartPole.
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Q1 network
            q1_encoder_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 256)
                .with_gain(SQRT_2)
                .init(device),
            q1_encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            q1_head: OrthogonalLinearConfig::new(256, CartPoleEnv::N_ACTIONS)
                .with_gain(VALUE_GAIN)
                .init(device),
            // Q2 network
            q2_encoder_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 256)
                .with_gain(SQRT_2)
                .init(device),
            q2_encoder_1: OrthogonalLinearConfig::new(256, 256)
                .with_gain(SQRT_2)
                .init(device),
            q2_head: OrthogonalLinearConfig::new(256, CartPoleEnv::N_ACTIONS)
                .with_gain(VALUE_GAIN)
                .init(device),
        }
    }

    /// Forward pass for Q1.
    fn forward_q1(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = tanh(self.q1_encoder_0.forward(obs));
        let x = tanh(self.q1_encoder_1.forward(x));
        self.q1_head.forward(x)
    }

    /// Forward pass for Q2.
    fn forward_q2(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = tanh(self.q2_encoder_0.forward(obs));
        let x = tanh(self.q2_encoder_1.forward(x));
        self.q2_head.forward(x)
    }
}

// Implement SACCritic for any Backend
impl<Backend: burn::tensor::backend::Backend> SACCritic<Backend, DiscretePolicy, FeedForward>
    for SACCriticNet<Backend>
{
    fn forward(
        &self,
        obs: Tensor<Backend, 2>,
        _action: Option<Tensor<Backend, 2>>, // Ignored for discrete
        _hidden: (),
    ) -> SACCriticOutput<Backend, FeedForward> {
        // For discrete actions, output Q-values for ALL actions
        let q1 = self.forward_q1(obs.clone());
        let q2 = self.forward_q2(obs);
        SACCriticOutput::new(q1, q2, ())
    }

    fn obs_size(&self) -> usize {
        CartPoleEnv::OBS_SIZE
    }

    fn action_dim(&self) -> usize {
        CartPoleEnv::N_ACTIONS
    }

    fn temporal_policy(&self) -> FeedForward {
        FeedForward::new()
    }

    fn is_discrete(&self) -> bool {
        true
    }
}

// ============================================================================
// Training Entry Point
// ============================================================================

#[allow(unused)]
pub fn run() {
    println!("=== SAC Discrete (CartPole) ===");
    println!("Environment: CartPole");
    println!("Action Space: Discrete (2 actions)");
    println!("Architecture: Separate Actor + Twin Q Critics");
    println!();

    // SAC configuration for discrete actions with soft target updates.
    // Use `discrete_soft()` for Polyak updates (tau=0.005 every step).
    // Alternatively, use `discrete()` for hard updates (tau=1.0 every 8000 steps).
    let config = SACConfig::discrete_soft()
        .with_n_actors(1) // Single actor for simpler setup
        .with_n_envs_per_actor(4)
        .with_buffer_capacity(100_000)
        .with_batch_size(256)
        .with_min_buffer_size(1000) // Start training after 1k transitions
        .with_actor_lr(3e-4)
        .with_critic_lr(3e-4)
        .with_alpha_lr(3e-4)
        .with_gamma(0.99)
        .with_auto_entropy_tuning(true)
        .with_target_reward(475.0)
        .with_max_env_steps(500_000)
        .with_utd_ratio(1.0); // One gradient step per env step

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
    println!(
        "  Gradient steps per iteration: {}",
        config.gradient_steps_per_env_step
    );
    println!("  Auto entropy tuning: {}", config.auto_entropy_tuning);
    println!();

    // Initialize device and models
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_actor = SACActorNet::<B>::new(&device);
    let initial_critic = SACCriticNet::<B>::new(&device);
    println!("Models created (Orthogonal init).");
    println!();

    // Create SAC runner for discrete actions
    let runner: SACDiscrete<B> = SACRunner::feed_forward(config.clone());

    // Model factories - create fresh models on each device
    let actor_factory = |device: &WgpuDevice| SACActorNet::<B>::new(device);
    let critic_factory = |device: &WgpuDevice| SACCriticNet::<B>::new(device);

    // Environment factory
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Create optimizers
    let (actor_optimizer, critic_optimizer) =
        runner.create_optimizers::<SACActorNet<B>, SACCriticNet<B>>();

    // Run training
    println!("Starting SAC training...");
    println!("Target reward: 475.0");
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
                "Steps: {:>8} | Ep: {:>5} | Return: {:>6.1} | Alpha: {:.3} | Q: {:>6.1} | CrLoss: {:>7.3} | SPS: {:>6.0}",
                stats.env_steps,
                stats.episodes,
                stats.mean_return,
                stats.alpha,
                stats.mean_q,
                stats.critic_loss,
                stats.sps
            );
        },
    );
}
