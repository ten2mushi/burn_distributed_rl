//! PPO Example using the Strategy-Based Runner
//!
//! This example demonstrates multi-actor PPO training using the
//! `PPORunner<A, T, B, S>` API with compile-time strategy selection.
//!
//! # Architecture
//!
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - Model Factory pattern for WGPU thread-safety
//! - BytesSlot for weight synchronization
//! - Strategy pattern for feed-forward vs recurrent dispatch
//!
//! # Runner Benefits
//!
//! - Single `PPORunner` type for both feed-forward and recurrent policies
//! - Zero-cost abstraction: feed-forward has no hidden state overhead
//! - Type-safe compile-time strategy selection
//! - Cleaner API with `PPORunner::feed_forward()` constructor
//!
//! # Network
//! - Separate actor/critic networks (no shared trunk)
//! - Tanh activations (not ReLU)
//! - Orthogonal initialization (via custom OrthogonalLinear)
//! - Small gain (0.01) for policy head keeps initial policy near uniform

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::activation::tanh;
use burn::tensor::Tensor;

use distributed_rl::algorithms::action_policy::{DiscretePolicy, DiscretePolicyOutput};
use distributed_rl::algorithms::actor_critic::{ActorCritic, ActorCriticInference, ForwardOutput};
use distributed_rl::algorithms::temporal_policy::FeedForward;
use distributed_rl::environment::CartPoleEnv;
use distributed_rl::nn::{OrthogonalLinear, OrthogonalLinearConfig};
use distributed_rl::runners::{PPOConfig, PPORunner, PPODiscrete};

// ============================================================================
// Backend Type
// ============================================================================

type B = Autodiff<Wgpu>;

// ============================================================================
// Network Definition
// ============================================================================

/// Actor-Critic network for CartPole.
///
/// Key differences from typical implementations:
/// - **Separate networks**: Actor and critic have independent weights
/// - **Tanh activations**: Better gradient flow than ReLU for RL
/// - **Orthogonal initialization**: using custom OrthogonalLinear
/// - **Small gain for policy head**: gain=0.01 keeps initial policy near uniform
#[derive(Module, Debug)]
pub struct PPONet<B: burn::tensor::backend::Backend> {
    // Actor network (policy)
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_head: OrthogonalLinear<B>,
    // Critic network (value function)
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

/// Gain constants for orthogonal initialization
const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01;  // Small gain for near-uniform initial policy
const VALUE_GAIN: f64 = 1.0;    // Standard gain for value head

impl<B: burn::tensor::backend::Backend> PPONet<B> {
    /// Create a new PPONet for CartPole.
    ///
    /// Initialization strategy:
    /// - Hidden layers: Orthogonal with gain=sqrt(2) for Tanh
    /// - Policy head: Orthogonal with gain=0.01 (keeps initial policy near uniform)
    /// - Value head: Orthogonal with gain=1.0
    /// - All biases: Zero
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Actor network (policy)
            actor_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_head: OrthogonalLinearConfig::new(64, CartPoleEnv::N_ACTIONS)
                .with_gain(POLICY_GAIN)
                .init(device),
            // Critic network (value function)
            critic_0: OrthogonalLinearConfig::new(CartPoleEnv::OBS_SIZE, 64)
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

    /// Forward pass returning (logits, values).
    ///
    /// Uses Tanh activations
    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Actor forward (policy)
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let logits = self.actor_head.forward(actor);

        // Critic forward (value function)
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (logits, values)
    }
}

// ============================================================================
// ActorCriticInference Implementation - For ANY Backend (including Inner)
// ============================================================================

impl<Backend: burn::tensor::backend::Backend>
    ActorCriticInference<Backend, DiscretePolicy, FeedForward> for PPONet<Backend>
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
    println!("=== PPO Runner (Strategy Pattern) ===");
    println!("Environment: CartPole");
    println!("Strategy: FeedForwardStrategy (zero-cost abstraction)");
    println!();

    let config = PPOConfig::new()
        .with_n_actors(1)
        .with_n_envs_per_actor(4) // 4
        .with_rollout_length(128)
        .with_n_epochs(4) // 4
        .with_n_minibatches(4) // 4
        .with_learning_rate(2.5e-4)
        .with_entropy_coef(0.01)
        .with_clip_vloss(true)
        .with_anneal_lr(true) // true
        .with_target_kl(Some(0.015)) // Some(0.015)
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
    println!("  LR annealing: {}", config.anneal_lr);
    println!("  Clipped value loss: {}", config.clip_vloss);
    println!();

    // Initialize device and model
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = PPONet::<B>::new(&device);
    println!();

    // NEW: Create unified runner using PPORunner::feed_forward()
    // This automatically selects FeedForwardStrategy at compile time
    let runner: PPODiscrete<B> = PPORunner::feed_forward(config.clone());

    // Model factory - creates fresh model on each actor's device
    let model_factory = |device: &WgpuDevice| PPONet::<B>::new(device);

    // Environment factory - creates operant SIMD CartPole for each actor
    let env_factory = |_actor_id: usize, n_envs: usize| {
        CartPoleEnv::new(n_envs).expect("Failed to create CartPole")
    };

    // Optimizer with gradient clipping (critical for PPO stability)
    // Uses max_grad_norm from config (default 0.5)
    let optimizer = runner.create_optimizer::<PPONet<B>>();

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
