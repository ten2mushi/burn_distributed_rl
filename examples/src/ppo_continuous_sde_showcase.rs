//! PPO Continuous SDE Showcase - Clean State-Dependent Exploration Architectures
//!
//! This example demonstrates multiple approaches to state-dependent exploration (SDE),
//! comparing their architectures and trade-offs.
//!
//! # Exploration Strategies
//!
//! ## 1. State-Independent (Baseline)
//! ```text
//! log_std = Param([action_dim])  // Single learned parameter
//! ```
//! - Simplest approach
//! - Same exploration for all states
//! - Clean optimization (no gradient interference)
//!
//! ## 2. Shared Features SDE (Naive - Often Unstable)
//! ```text
//! actor_features = MLP(obs)
//! mean = Linear(actor_features)
//! log_std = Linear(actor_features)  // SHARED features - causes interference!
//! ```
//! - Gradient interference between mean and log_std heads
//! - Can cause oscillating training
//! - Included for comparison only
//!
//! ## 3. Separate Network SDE (Clean)
//! ```text
//! actor_features = MLP_actor(obs)
//! mean = Linear(actor_features)
//!
//! std_features = MLP_std(obs)  // SEPARATE network!
//! log_std = Linear(std_features)
//! ```
//! - No gradient interference
//! - Independent representations for action and uncertainty
//! - More parameters but cleaner optimization
//!
//! ## 4. Residual SDE (Stable + Adaptive)
//! ```text
//! base_log_std = Param([action_dim])  // State-independent baseline
//! adjustment = Linear(actor_features) * scale  // Small correction
//! log_std = base_log_std + adjustment
//! ```
//! - Keeps SI stability as baseline
//! - Adds small state-dependent corrections
//! - Best of both worlds
//!
//! ## 5. Generalized SDE (gSDE - SB3 Style)
//! ```text
//! exploration_mat = Param([hidden_dim, action_dim])
//! latent_noise = randn([batch, hidden_dim])
//! exploration_noise = latent_noise @ exploration_mat
//! action = mean + exploration_noise * std
//! ```
//! - Separate exploration pathway
//! - State-correlated noise without coupling to policy features
//! - Used by Stable Baselines 3

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
// Exploration Strategy Selection
// ============================================================================

/// Exploration strategy for continuous PPO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationStrategy {
    /// State-independent: single learned log_std parameter
    StateIndependent,
    /// Shared features: log_std from actor features (often unstable)
    SharedFeatures,
    /// Separate network: independent MLP for log_std (clean SDE)
    SeparateNetwork,
    /// Residual: SI baseline + small state-dependent adjustment
    Residual,
    /// Generalized SDE: separate exploration matrix (SB3 style)
    Generalized,
}

impl ExplorationStrategy {
    pub fn description(&self) -> &'static str {
        match self {
            Self::StateIndependent => "State-Independent (SI) - single learned parameter",
            Self::SharedFeatures => "Shared Features SDE - coupled optimization (unstable)",
            Self::SeparateNetwork => "Separate Network SDE - independent log_std MLP",
            Self::Residual => "Residual SDE - SI baseline + state-dependent adjustment",
            Self::Generalized => "Generalized SDE (gSDE) - separate exploration matrix",
        }
    }
}

// ============================================================================
// Initialization Constants
// ============================================================================

const SQRT_2: f64 = 1.41421356; // sqrt(2) for Tanh hidden layers
const POLICY_GAIN: f64 = 0.01; // Small gain for policy heads
const VALUE_GAIN: f64 = 1.0; // Standard gain for value head
const INITIAL_LOG_STD: f32 = 0.0; // std = 1.0 initially
const RESIDUAL_SCALE: f32 = 0.1; // Scale for residual adjustments

// ============================================================================
// Strategy 1: State-Independent (Baseline)
// ============================================================================

/// State-Independent exploration: single learned log_std parameter.
///
/// Architecture:
/// ```text
/// obs ──► [actor_0] ──► tanh ──► [actor_1] ──► tanh ──► [actor_mean] ──► mean
///
/// log_std = Param([action_dim])  // Broadcast to batch
///
/// obs ──► [critic_0] ──► tanh ──► [critic_1] ──► tanh ──► [critic_head] ──► value
/// ```
#[derive(Module, Debug)]
pub struct StateIndependentNet<B: burn::tensor::backend::Backend> {
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    log_std: Param<Tensor<B, 1>>,
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

impl<B: burn::tensor::backend::Backend> StateIndependentNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            log_std: Param::from_tensor(
                Tensor::zeros([PendulumEnv::ACTION_DIM], device).add_scalar(INITIAL_LOG_STD),
            ),
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

    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];

        // Actor: obs → features → mean
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let mean = self.actor_mean.forward(actor);

        // State-independent log_std: broadcast to batch
        let log_std = self
            .log_std
            .val()
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // Critic: separate network
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// Strategy 2: Shared Features SDE (Naive - For Comparison)
// ============================================================================

/// Shared Features SDE: log_std derived from actor features.
///
/// WARNING: This architecture causes gradient interference and is often unstable.
/// Included for educational comparison only.
///
/// Architecture:
/// ```text
/// obs ──► [actor_0] ──► tanh ──► [actor_1] ──► tanh ──┬──► [actor_mean] ──► mean
///                                                     │
///                                                     └──► [actor_log_std] ──► log_std
///                                                          (SHARED features!)
/// ```
#[derive(Module, Debug)]
pub struct SharedFeaturesNet<B: burn::tensor::backend::Backend> {
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    actor_log_std: OrthogonalLinear<B>, // Shares features with mean!
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

impl<B: burn::tensor::backend::Backend> SharedFeaturesNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // Log_std head shares actor features - causes gradient interference
            actor_log_std: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
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

    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Actor: shared features for both mean and log_std
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));

        let mean = self.actor_mean.forward(actor.clone());
        let log_std = self.actor_log_std.forward(actor); // SHARED!

        // Critic
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// Strategy 3: Separate Network SDE (Clean)
// ============================================================================

/// Separate Network SDE: independent MLP for log_std.
///
/// This is the cleanest SDE architecture - no gradient interference.
///
/// Architecture:
/// ```text
/// obs ──► [actor_0] ──► tanh ──► [actor_1] ──► tanh ──► [actor_mean] ──► mean
///
/// obs ──► [std_0] ──► tanh ──► [std_1] ──► tanh ──► [std_head] ──► log_std
///         (SEPARATE network - no interference!)
///
/// obs ──► [critic_0] ──► tanh ──► [critic_1] ──► tanh ──► [critic_head] ──► value
/// ```
#[derive(Module, Debug)]
pub struct SeparateNetworkNet<B: burn::tensor::backend::Backend> {
    // Actor network (policy mean)
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    // SEPARATE log_std network
    std_0: OrthogonalLinear<B>,
    std_1: OrthogonalLinear<B>,
    std_head: OrthogonalLinear<B>,
    // Critic network
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

impl<B: burn::tensor::backend::Backend> SeparateNetworkNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Actor network
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // Separate log_std network (independent features)
            std_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            std_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            std_head: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // Critic network
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

    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Actor: obs → actor features → mean
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let mean = self.actor_mean.forward(actor);

        // Separate std network: obs → std features → log_std
        let std_features = tanh(self.std_0.forward(x.clone()));
        let std_features = tanh(self.std_1.forward(std_features));
        let log_std = self.std_head.forward(std_features);

        // Critic
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// Strategy 4: Residual SDE (SI + State-Dependent Adjustment)
// ============================================================================

/// Residual SDE: state-independent baseline + small state-dependent correction.
///
/// This keeps the stability of SI while allowing adaptive exploration.
///
/// Architecture:
/// ```text
/// obs ──► [actor_0] ──► tanh ──► [actor_1] ──► tanh ──┬──► [actor_mean] ──► mean
///                                                     │
///                                                     └──► [adjustment_head] ──► adj
///
/// log_std = base_log_std + scale * adj
///           ^^^^^^^^^^^^   ^^^^^^^^^^
///           SI baseline    small correction (scale=0.1)
/// ```
#[derive(Module, Debug)]
pub struct ResidualSDENet<B: burn::tensor::backend::Backend> {
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    // Residual SDE components
    base_log_std: Param<Tensor<B, 1>>,     // SI baseline
    adjustment_head: OrthogonalLinear<B>,  // State-dependent adjustment
    // Critic
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

impl<B: burn::tensor::backend::Backend> ResidualSDENet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            // SI baseline
            base_log_std: Param::from_tensor(
                Tensor::zeros([PendulumEnv::ACTION_DIM], device).add_scalar(INITIAL_LOG_STD),
            ),
            // Small adjustment head
            adjustment_head: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
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

    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];

        // Actor features
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let mean = self.actor_mean.forward(actor.clone());

        // Residual SDE: base + scaled adjustment
        let base = self
            .base_log_std
            .val()
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        let adjustment = self.adjustment_head.forward(actor) * RESIDUAL_SCALE;
        let log_std = base + adjustment;

        // Critic
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// Strategy 5: Generalized SDE (gSDE - SB3 Style)
// ============================================================================

/// Generalized SDE (gSDE): separate exploration matrix.
///
/// This is the approach used by Stable Baselines 3. The exploration noise
/// is state-correlated but uses a separate pathway from policy features.
///
/// Architecture:
/// ```text
/// obs ──► [actor_0] ──► tanh ──► [actor_1] ──► tanh ──► actor_features
///                                                          │
///                                              ┌───────────┼───────────┐
///                                              ▼           ▼           ▼
///                                        [actor_mean]  [base_std]  exploration
///                                              │           │           │
///                                              ▼           ▼           ▼
///                                            mean    base_log_std   noise
///
/// action = mean + exp(log_std) * noise
///
/// The exploration uses actor_features but through a SEPARATE matrix,
/// not the same linear head as mean. This reduces gradient coupling.
/// ```
#[derive(Module, Debug)]
pub struct GeneralizedSDENet<B: burn::tensor::backend::Backend> {
    actor_0: OrthogonalLinear<B>,
    actor_1: OrthogonalLinear<B>,
    actor_mean: OrthogonalLinear<B>,
    // gSDE components
    base_log_std: Param<Tensor<B, 1>>,        // Base exploration level
    exploration_mat: Param<Tensor<B, 2>>,     // [hidden_dim, action_dim]
    // Critic
    critic_0: OrthogonalLinear<B>,
    critic_1: OrthogonalLinear<B>,
    critic_head: OrthogonalLinear<B>,
}

impl<B: burn::tensor::backend::Backend> GeneralizedSDENet<B> {
    pub fn new(device: &B::Device) -> Self {
        // Initialize exploration matrix with small values
        let exploration_mat = Tensor::<B, 2>::random(
            [64, PendulumEnv::ACTION_DIM],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );

        Self {
            actor_0: OrthogonalLinearConfig::new(PendulumEnv::OBS_SIZE, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_1: OrthogonalLinearConfig::new(64, 64)
                .with_gain(SQRT_2)
                .init(device),
            actor_mean: OrthogonalLinearConfig::new(64, PendulumEnv::ACTION_DIM)
                .with_gain(POLICY_GAIN)
                .init(device),
            base_log_std: Param::from_tensor(
                Tensor::zeros([PendulumEnv::ACTION_DIM], device).add_scalar(INITIAL_LOG_STD),
            ),
            exploration_mat: Param::from_tensor(exploration_mat),
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

    pub fn forward_net(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];

        // Actor features
        let actor = tanh(self.actor_0.forward(x.clone()));
        let actor = tanh(self.actor_1.forward(actor));
        let mean = self.actor_mean.forward(actor.clone());

        // gSDE: state-correlated exploration through separate matrix
        // The exploration matrix transforms actor features into exploration noise
        // This is state-dependent but doesn't couple gradients like shared heads
        let exploration_contribution = actor.matmul(self.exploration_mat.val().clone());

        // Combine base log_std with exploration contribution
        let base = self
            .base_log_std
            .val()
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // The exploration contribution modulates the base std
        // Using tanh to bound the contribution
        let log_std = base + tanh(exploration_contribution) * 0.5;

        // Critic
        let critic = tanh(self.critic_0.forward(x));
        let critic = tanh(self.critic_1.forward(critic));
        let values = self.critic_head.forward(critic);

        (mean, log_std, values)
    }
}

// ============================================================================
// Macro for ActorCriticInference Implementation
// ============================================================================

macro_rules! impl_actor_critic {
    ($net_type:ident) => {
        impl<Backend: burn::tensor::backend::Backend>
            ActorCriticInference<Backend, ContinuousPolicy, FeedForward> for $net_type<Backend>
        {
            fn forward(
                &self,
                obs: Tensor<Backend, 2>,
                _hidden: (),
            ) -> ForwardOutput<Backend, ContinuousPolicy, FeedForward> {
                let (mean, log_std, values) = self.forward_net(obs);

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

        impl ActorCritic<B, ContinuousPolicy, FeedForward> for $net_type<B> {}
    };
}

impl_actor_critic!(StateIndependentNet);
impl_actor_critic!(SharedFeaturesNet);
impl_actor_critic!(SeparateNetworkNet);
impl_actor_critic!(ResidualSDENet);
impl_actor_critic!(GeneralizedSDENet);

// ============================================================================
// Training Entry Point
// ============================================================================

/// Common PPO configuration for all strategies.
fn create_config() -> PPOConfig {
    PPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(64)
        .with_rollout_length(256)
        .with_n_epochs(10)
        .with_n_minibatches(32)
        .with_learning_rate(2.5e-4)
        .with_gamma(0.99)
        .with_gae_lambda(0.95)
        .with_entropy_coef(0.005)
        .with_clip_vloss(true)
        .with_anneal_lr(true)
        .with_target_kl(Some(0.02))
        .with_reward_normalization(true)
        .with_log_std_floor(-2.0)
        .with_target_reward(-200.0)
        .with_max_env_steps(5_000_000)
}

/// Print common header.
fn print_header(strategy: ExplorationStrategy) {
    println!("=== PPO Continuous SDE Showcase (Pendulum) ===");
    println!("Environment: Pendulum-v1");
    println!("Action Space: Continuous [-2.0, 2.0]");
    println!("Strategy: {}", strategy.description());
    println!();
}

/// Print config details.
fn print_config(config: &PPOConfig) {
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
}

/// Environment factory.
fn env_factory(_actor_id: usize, n_envs: usize) -> PendulumEnv {
    PendulumEnv::new(n_envs).expect("Failed to create Pendulum")
}

/// Progress callback.
fn progress_callback(stats: &distributed_rl::runners::PPOStats) {
    println!(
        "Steps: {:>8} | Episodes: {:>5} | Reward: {:>8.1} | Version: {:>4} | SPS: {:>6.0}",
        stats.env_steps,
        stats.episodes,
        stats.avg_reward,
        stats.policy_version,
        stats.steps_per_second
    );
}

#[allow(unused)]
pub fn run() {
    run_state_independent()
}

#[allow(unused)]
pub fn run_state_independent() {
    print_header(ExplorationStrategy::StateIndependent);
    let config = create_config();
    print_config(&config);

    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = StateIndependentNet::<B>::new(&device);
    println!("Model created.");
    println!();

    let runner: PPOContinuous<B> = PPORunner::feed_forward(config);
    let model_factory = |device: &WgpuDevice| StateIndependentNet::<B>::new(device);
    let optimizer = runner.create_optimizer::<StateIndependentNet<B>>();

    println!("Starting training...");
    println!("Target reward: -200.0");
    println!();

    let _ = runner.run(model_factory, initial_model, env_factory, optimizer, progress_callback);
}

#[allow(unused)]
pub fn run_shared_features() {
    print_header(ExplorationStrategy::SharedFeatures);
    let config = create_config();
    print_config(&config);

    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = SharedFeaturesNet::<B>::new(&device);
    println!("Model created (WARNING: Shared features - may be unstable!).");
    println!();

    let runner: PPOContinuous<B> = PPORunner::feed_forward(config);
    let model_factory = |device: &WgpuDevice| SharedFeaturesNet::<B>::new(device);
    let optimizer = runner.create_optimizer::<SharedFeaturesNet<B>>();

    println!("Starting training...");
    println!("Target reward: -200.0");
    println!();

    let _ = runner.run(model_factory, initial_model, env_factory, optimizer, progress_callback);
}

#[allow(unused)]
pub fn run_separate_network() {
    print_header(ExplorationStrategy::SeparateNetwork);
    let config = create_config();
    print_config(&config);

    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = SeparateNetworkNet::<B>::new(&device);
    println!("Model created (Separate log_std network - clean SDE).");
    println!();

    let runner: PPOContinuous<B> = PPORunner::feed_forward(config);
    let model_factory = |device: &WgpuDevice| SeparateNetworkNet::<B>::new(device);
    let optimizer = runner.create_optimizer::<SeparateNetworkNet<B>>();

    println!("Starting training...");
    println!("Target reward: -200.0");
    println!();

    let _ = runner.run(model_factory, initial_model, env_factory, optimizer, progress_callback);
}

#[allow(unused)]
pub fn run_residual() {
    print_header(ExplorationStrategy::Residual);
    let config = create_config();
    print_config(&config);

    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = ResidualSDENet::<B>::new(&device);
    println!("Model created (Residual SDE - SI baseline + adjustment).");
    println!();

    let runner: PPOContinuous<B> = PPORunner::feed_forward(config);
    let model_factory = |device: &WgpuDevice| ResidualSDENet::<B>::new(device);
    let optimizer = runner.create_optimizer::<ResidualSDENet<B>>();

    println!("Starting training...");
    println!("Target reward: -200.0");
    println!();

    let _ = runner.run(model_factory, initial_model, env_factory, optimizer, progress_callback);
}

#[allow(unused)]
pub fn run_generalized() {
    print_header(ExplorationStrategy::Generalized);
    let config = create_config();
    print_config(&config);

    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let initial_model = GeneralizedSDENet::<B>::new(&device);
    println!("Model created (gSDE - separate exploration matrix).");
    println!();

    let runner: PPOContinuous<B> = PPORunner::feed_forward(config);
    let model_factory = |device: &WgpuDevice| GeneralizedSDENet::<B>::new(device);
    let optimizer = runner.create_optimizer::<GeneralizedSDENet<B>>();

    println!("Starting training...");
    println!("Target reward: -200.0");
    println!();

    let _ = runner.run(model_factory, initial_model, env_factory, optimizer, progress_callback);
}
