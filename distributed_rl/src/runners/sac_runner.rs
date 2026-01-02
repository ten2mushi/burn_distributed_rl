//! SAC runner with multi-actor architecture.
//!
//! This module provides `SACRunner`, a high-level API for multi-actor
//! SAC training with separate actor and critic networks.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                        Main Thread                              │
//! │  • Validates model factories                                    │
//! │  • Spawns actors and learner                                   │
//! │  • Monitors progress                                           │
//! │  • Handles shutdown                                            │
//! └────────────────────────────────────────────────────────────────┘
//!          ↓                                              ↓
//! ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
//! │   Actor 0        │  │   Actor 1        │  │   Actor N-1      │
//! │  • Own device    │  │  • Own device    │  │  • Own device    │
//! │  • Actor network │  │  • Actor network │  │  • Actor network │
//! │  • M envs        │  │  • M envs        │  │  • M envs        │
//! │  • Push trans    │  │  • Push trans    │  │  • Push trans    │
//! └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
//!          └───────────────┬─────┴─────────────────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │      SACBuffer       │
//!               │  • Ring buffer       │
//!               │  • Uniform sampling  │
//!               │  • Lock-free push    │
//!               └──────────┬───────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │       Learner        │
//!               │  • Sample batches    │
//!               │  • Critic update     │
//!               │  • Actor update      │
//!               │  • Alpha update      │
//!               │  • Target update     │
//!               │  • Publish actor     │
//!               └──────────────────────┘
//! ```
//!
//! # Key Differences from PPO/IMPALA
//!
//! - **Separate networks**: Actor and Critic(s) are independent
//! - **Twin Q-networks**: Two critics for pessimistic value estimation
//! - **Target networks**: Soft/hard updates for stable TD targets
//! - **Three optimizers**: Separate for actor, critic, and alpha
//! - **Per-step training**: Not batched epochs like PPO
//! - **Uniform replay**: Random sampling from buffer
//!
//! # Type Aliases
//!
//! - `SACDiscrete<B>`: Feed-forward discrete actions
//! - `SACContinuous<B>`: Feed-forward continuous actions
//! - `RecurrentSACDiscrete<B>`: Recurrent discrete actions
//! - `RecurrentSACContinuous<B>`: Recurrent continuous actions

use burn::module::AutodiffModule;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algorithms::action_policy::{ActionPolicy, ActionValue};
use crate::algorithms::sac::{
    SACBuffer, SACBufferConfig, SACConfig, SACStats,
    SACTransition,
    SACActor,
    SACCritic,
    EntropyTuner,
};
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig, HiddenStateType, Recurrent, TemporalPolicy};
use crate::core::bytes_slot::{bytes_slot_with, SharedBytesSlot};
use crate::core::experience_buffer::ExperienceBuffer;
use crate::core::transition::Transition;
use crate::core::target_network::{TargetNetworkConfig, TargetNetworkManager};

use super::learner::VectorizedEnv;
use super::sac_strategies::{DefaultSACStrategy, SACLossInfo, SACTrainingStrategy, TemporalPolicySACStrategy};

/// Maximum number of recent episode rewards to keep for statistics.
const MAX_RECENT_REWARDS: usize = 1000;

// ============================================================================
// SACRunner
// ============================================================================

/// Unified SAC runner supporting both feed-forward and recurrent policies.
///
/// Coordinates N actor threads + 1 learner thread for SAC training.
/// Uses bytes-based weight transfer for WGPU compatibility.
///
/// # Type Parameters
///
/// - `A`: Action policy (`DiscretePolicy` or `ContinuousPolicy`)
/// - `T`: Temporal policy (`FeedForward` or `Recurrent`)
/// - `B`: Autodiff backend (e.g., `Autodiff<Wgpu>`)
/// - `S`: Training strategy (auto-selected via `DefaultSACStrategy<T>`)
pub struct SACRunner<A, T, B, S = DefaultSACStrategy<T>>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B> + TemporalPolicySACStrategy,
    B: AutodiffBackend,
    S: SACTrainingStrategy<B, A, T>,
{
    config: SACConfig,
    hidden_config: HiddenConfig,
    _marker: PhantomData<(A, T, B, S)>,
}

impl<A, T, B, S> SACRunner<A, T, B, S>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B> + TemporalPolicySACStrategy,
    B: AutodiffBackend,
    S: SACTrainingStrategy<B, A, T>,
{
    /// Create a new SAC runner.
    ///
    /// # Arguments
    ///
    /// * `config` - SAC configuration
    /// * `hidden_config` - Hidden state configuration (use `HiddenConfig::none()` for feed-forward)
    pub fn new(config: SACConfig, hidden_config: HiddenConfig) -> Self {
        Self {
            config,
            hidden_config,
            _marker: PhantomData,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &SACConfig {
        &self.config
    }

    /// Get hidden configuration.
    pub fn hidden_config(&self) -> &HiddenConfig {
        &self.hidden_config
    }

    /// Create configured optimizers for actor and critic.
    ///
    /// Returns (actor_optimizer, critic_optimizer).
    /// Note: Alpha (entropy coefficient) is tuned internally by the learner.
    pub fn create_optimizers<Actor, Critic>(
        &self,
    ) -> (
        impl Optimizer<Actor, B>,
        impl Optimizer<Critic, B>,
    )
    where
        Actor: AutodiffModule<B>,
        Critic: AutodiffModule<B>,
    {
        let mut actor_config = AdamConfig::new().with_epsilon(1e-5);
        let mut critic_config = AdamConfig::new().with_epsilon(1e-5);

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            actor_config =
                actor_config.with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm)));
            critic_config =
                critic_config.with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm)));
        }

        (actor_config.init(), critic_config.init())
    }

    /// Run distributed SAC training.
    ///
    /// # Arguments
    ///
    /// - `actor_factory`: Creates a fresh actor network on a given device
    /// - `critic_factory`: Creates a fresh critic network on a given device
    /// - `initial_actor`: Actor with initial/pretrained weights
    /// - `initial_critic`: Critic with initial/pretrained weights
    /// - `env_factory`: Creates environment for given (actor_id, n_envs)
    /// - `callback`: Called periodically with training stats
    ///
    /// # Returns
    ///
    /// (trained_actor, trained_critic)
    pub fn run<Actor, Critic, ActorF, CriticF, EF, E, ActorOpt, CriticOpt, F>(
        &self,
        actor_factory: ActorF,
        critic_factory: CriticF,
        initial_actor: Actor,
        initial_critic: Critic,
        env_factory: EF,
        actor_optimizer: ActorOpt,
        critic_optimizer: CriticOpt,
        callback: F,
    ) -> (Actor, Critic)
    where
        ActorF: Fn(&B::Device) -> Actor + Send + Sync + Clone + 'static,
        CriticF: Fn(&B::Device) -> Critic + Send + Sync + Clone + 'static,
        Actor: SACActor<B, A, T> + AutodiffModule<B> + Clone + 'static,
        Critic: SACCritic<B, A, T> + AutodiffModule<B> + Clone + 'static,
        Actor::Record: Send + 'static,
        Critic::Record: Send + 'static,
        // Inner backend bounds for actor inference
        Actor::InnerModule: SACActor<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        <T as TemporalPolicy<B::InnerBackend>>::Hidden: HiddenStateType<B::InnerBackend>,
        EF: Fn(usize, usize) -> E + Send + Sync + Clone + 'static,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action> + 'static,
        ActorOpt: Optimizer<Actor, B> + 'static,
        CriticOpt: Optimizer<Critic, B> + 'static,
        F: Fn(&SACStats),
        S::TransitionData: Clone + Send + Sync + 'static,
    {
        let device = B::Device::default();
        let config = self.config.clone();
        let hidden_config = self.hidden_config.clone();

        let is_recurrent = <T as TemporalPolicy<B>>::is_recurrent();
        let is_discrete = initial_critic.is_discrete();
        let runner_type = if is_recurrent { "Recurrent SAC" } else { "SAC" };
        let action_type = if is_discrete { "Discrete" } else { "Continuous" };

        println!("=== {} Runner ({}) ===", runner_type, action_type);
        println!(
            "Actors: {} x {} envs = {} total",
            config.n_actors,
            config.n_envs_per_actor,
            config.total_envs()
        );
        println!(
            "Buffer capacity: {}, Min size: {}, Batch: {}",
            config.buffer_capacity, config.min_buffer_size, config.batch_size
        );
        println!(
            "Tau: {}, Hard updates: {}",
            config.tau, config.hard_target_update
        );
        if is_recurrent {
            println!(
                "Sequence length: {}, Hidden size: {}",
                config.sequence_length, hidden_config.hidden_size
            );
        }

        // Validate model factories
        println!("Validating model factories...");
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        {
            let test_actor = actor_factory(&device);
            let test_record = initial_actor.clone().into_record();
            let _loaded = test_actor.load_record(test_record);

            let test_critic = critic_factory(&device);
            let test_record = initial_critic.clone().into_record();
            let _loaded = test_critic.load_record(test_record);
            println!("Model factories validated successfully.");
        }

        // Compute target entropy
        // For discrete: use action_space_size (number of discrete actions)
        // For continuous: use action_dim (continuous action vector dimension)
        let action_space = if is_discrete {
            initial_actor.action_space_size()
        } else {
            initial_actor.action_dim()
        };
        let target_entropy = config.compute_target_entropy(action_space, is_discrete);
        println!("Target entropy: {:.3}", target_entropy);

        // Serialize initial actor weights
        let initial_actor_bytes = recorder
            .record(initial_actor.clone().into_record(), ())
            .expect("Failed to serialize initial actor");
        println!("Initial actor serialized: {} bytes", initial_actor_bytes.len());

        // Create shared state
        let actor_bytes_slot: SharedBytesSlot = bytes_slot_with(initial_actor_bytes);
        let buffer = Arc::new(SACBuffer::<SACTransition<S::TransitionData>>::new(
            SACBufferConfig {
                capacity: config.buffer_capacity,
                min_size: config.min_buffer_size,
                batch_size: config.batch_size,
            },
        ));
        let shutdown = Arc::new(AtomicBool::new(false));
        let total_env_steps = Arc::new(AtomicUsize::new(0));
        let total_episodes = Arc::new(AtomicUsize::new(0));
        let train_steps = Arc::new(AtomicUsize::new(0));
        let recent_rewards = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(MAX_RECENT_REWARDS)));

        // Create target critic on main thread BEFORE spawning actors
        // (WGPU requires GPU operations on main thread, and we don't want
        // actors collecting data while GPU is initializing)
        let mut target_critic = critic_factory(&device);
        target_critic = target_critic.load_record(initial_critic.clone().into_record());

        // Spawn actor threads
        let mut actor_handles = Vec::with_capacity(config.n_actors);
        for actor_id in 0..config.n_actors {
            let actor_factory = actor_factory.clone();
            let env_factory = env_factory.clone();
            let buffer = Arc::clone(&buffer);
            let slot = Arc::clone(&actor_bytes_slot);
            let shutdown_flag = Arc::clone(&shutdown);
            let env_steps = Arc::clone(&total_env_steps);
            let episodes = Arc::clone(&total_episodes);
            let rewards = Arc::clone(&recent_rewards);
            let cfg = config.clone();
            let hcfg = hidden_config.clone();

            let handle = std::thread::Builder::new()
                .name(format!("{}-Actor-{}", runner_type, actor_id))
                .spawn(move || {
                    Self::actor_thread::<Actor, ActorF, EF, E>(
                        actor_id,
                        &cfg,
                        &hcfg,
                        actor_factory,
                        env_factory,
                        slot,
                        buffer,
                        shutdown_flag,
                        env_steps,
                        episodes,
                        rewards,
                    );
                })
                .expect("Failed to spawn actor thread");

            actor_handles.push(handle);
        }

        // Spawn learner thread
        let learner_buffer = Arc::clone(&buffer);
        let learner_slot = Arc::clone(&actor_bytes_slot);
        let learner_shutdown = Arc::clone(&shutdown);
        let learner_train_steps = Arc::clone(&train_steps);
        let learner_env_steps = Arc::clone(&total_env_steps);
        let shared_loss_info: Arc<Mutex<SACLossInfo>> = Arc::new(Mutex::new(SACLossInfo::default()));
        let learner_loss_info = Arc::clone(&shared_loss_info);
        let cfg = config.clone();
        let hcfg = hidden_config.clone();

        let learner_handle = std::thread::Builder::new()
            .name(format!("{}-Learner", runner_type))
            .spawn(move || {
                Self::learner_thread::<Actor, Critic, ActorOpt, CriticOpt>(
                    &cfg,
                    &hcfg,
                    initial_actor,
                    initial_critic,
                    target_critic,
                    actor_optimizer,
                    critic_optimizer,
                    target_entropy,
                    learner_slot,
                    learner_buffer,
                    learner_shutdown,
                    learner_train_steps,
                    learner_env_steps,
                    learner_loss_info,
                )
            })
            .expect("Failed to spawn learner thread");

        // Main monitoring loop
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let mut stats = SACStats::default();

        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));

            let env_steps = total_env_steps.load(Ordering::Relaxed);
            let episodes = total_episodes.load(Ordering::Relaxed);
            let t_steps = train_steps.load(Ordering::Relaxed);

            let avg_reward = {
                let rewards_guard = recent_rewards.lock();
                if !rewards_guard.is_empty() {
                    let recent: Vec<f32> = rewards_guard.iter().rev().take(100).copied().collect();
                    recent.iter().sum::<f32>() / recent.len() as f32
                } else {
                    0.0
                }
            };

            // Update stats
            stats.env_steps = env_steps;
            stats.episodes = episodes;
            stats.mean_return = avg_reward;
            stats.train_steps = t_steps;
            stats.sps = env_steps as f32 / start_time.elapsed().as_secs_f32();
            stats.buffer_utilization = buffer.len() as f32 / config.buffer_capacity as f32;
            stats.elapsed_secs = start_time.elapsed().as_secs_f32();

            // Update loss info from learner
            {
                let loss = shared_loss_info.lock();
                stats.actor_loss = loss.actor_loss;
                stats.critic_loss = loss.critic_loss;
                stats.alpha_loss = loss.alpha_loss;
                stats.alpha = loss.alpha;
                stats.mean_q = loss.mean_q;
                stats.mean_entropy = loss.mean_entropy;
            }

            // Check termination
            if env_steps >= config.max_env_steps {
                println!("\nReached max env steps: {}", config.max_env_steps);
                shutdown.store(true, Ordering::Relaxed);
                break;
            }

            if let Some(target) = config.target_reward {
                if avg_reward >= target && episodes > 100 {
                    println!(
                        "\n=== SOLVED! Avg reward: {:.1} >= {} ===",
                        avg_reward, target
                    );
                    shutdown.store(true, Ordering::Relaxed);
                    break;
                }
            }

            // Log progress
            if last_log_time.elapsed().as_secs_f32() >= config.log_interval_secs {
                callback(&stats);
                last_log_time = Instant::now();
            }
        }

        // Shutdown
        shutdown.store(true, Ordering::Relaxed);

        for handle in actor_handles {
            let _ = handle.join();
        }

        let (trained_actor, trained_critic) = learner_handle.join().expect("Learner thread panicked");

        println!("\n=== Training Complete ===");
        println!("Duration: {:.1}s", start_time.elapsed().as_secs_f32());
        println!("Total steps: {}", total_env_steps.load(Ordering::Relaxed));
        println!("Total episodes: {}", total_episodes.load(Ordering::Relaxed));
        println!("Train steps: {}", train_steps.load(Ordering::Relaxed));

        (trained_actor, trained_critic)
    }

    /// Actor thread: collects transitions and pushes to buffer.
    fn actor_thread<Actor, ActorF, EF, E>(
        actor_id: usize,
        config: &SACConfig,
        hidden_config: &HiddenConfig,
        actor_factory: ActorF,
        env_factory: EF,
        actor_bytes_slot: SharedBytesSlot,
        buffer: Arc<SACBuffer<SACTransition<S::TransitionData>>>,
        shutdown: Arc<AtomicBool>,
        env_steps: Arc<AtomicUsize>,
        episodes: Arc<AtomicUsize>,
        rewards: Arc<Mutex<VecDeque<f32>>>,
    ) where
        ActorF: Fn(&B::Device) -> Actor,
        Actor: SACActor<B, A, T> + AutodiffModule<B>,
        Actor::Record: Send + 'static,
        Actor::InnerModule: SACActor<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        <T as TemporalPolicy<B::InnerBackend>>::Hidden: HiddenStateType<B::InnerBackend>,
        EF: Fn(usize, usize) -> E,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action>,
        S::TransitionData: Clone,
    {
        let device = B::Device::default();
        let inner_device = <B::InnerBackend as Backend>::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Create actor using factory
        let mut actor = actor_factory(&device);

        // Load initial weights
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            if let Some(bytes) = actor_bytes_slot.get() {
                if let Ok(record) = recorder.load(bytes, &device) {
                    actor = actor.load_record(record);
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        // Get inference actor (no autodiff)
        let mut inference_actor: Actor::InnerModule = actor.valid();

        // Create environment
        let mut env = env_factory(actor_id, config.n_envs_per_actor);
        let obs_size = env.obs_size();
        let n_envs = config.n_envs_per_actor;

        // Initialize state
        let mut obs_buffer = vec![0.0f32; n_envs * obs_size];
        env.write_observations(&mut obs_buffer);
        let mut episode_rewards = vec![0.0f32; n_envs];
        let mut last_version = actor_bytes_slot.version();
        let mut local_steps = 0usize;

        // Pre-allocated buffers
        let mut current_obs: Vec<Vec<f32>> = (0..n_envs)
            .map(|_| vec![0.0f32; obs_size])
            .collect();

        // Hidden state management
        let mut hidden = <T as TemporalPolicy<B::InnerBackend>>::Hidden::initial(
            n_envs,
            &inner_device,
            hidden_config,
        );

        while !shutdown.load(Ordering::Relaxed) {
            // Check for weight updates periodically
            if local_steps % config.model_update_freq == 0 {
                let current_version = actor_bytes_slot.version();
                if current_version > last_version {
                    if let Some(bytes) = actor_bytes_slot.get() {
                        if let Ok(record) = recorder.load(bytes, &device) {
                            actor = actor.load_record(record);
                            inference_actor = actor.valid();
                            last_version = current_version;
                        }
                    }
                }
            }

            // Save current obs
            for i in 0..n_envs {
                let start = i * obs_size;
                current_obs[i].copy_from_slice(&obs_buffer[start..start + obs_size]);
            }

            // Forward pass on inner backend
            let obs_tensor =
                Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                    .reshape([n_envs, obs_size]);

            let output = inference_actor.forward(obs_tensor, hidden.clone());

            // Sample actions
            let (actions, _log_probs) = output.sample(&inner_device);

            // Update hidden state
            hidden = output.hidden;

            // Step environment
            let step_result = env.step(&actions);
            env.write_observations(&mut obs_buffer);

            // Create transitions
            for i in 0..n_envs {
                let action_floats = actions[i].as_floats();
                // Use policy type to determine action storage format, not dimension.
                // This is critical for 1D continuous action spaces (e.g., Pendulum).
                let action = if <A as ActionPolicy<B>>::is_discrete() {
                    crate::core::transition::Action::Discrete(action_floats[0] as u32)
                } else {
                    crate::core::transition::Action::Continuous(action_floats)
                };

                let next_start = i * obs_size;
                let next_state = obs_buffer[next_start..next_start + obs_size].to_vec();

                let base = Transition {
                    state: current_obs[i].clone(),
                    action,
                    reward: step_result.rewards[i],
                    next_state,
                    terminal: step_result.terminals[i],
                    truncated: step_result.dones[i] && !step_result.terminals[i],
                };

                // Create SAC transition using strategy
                let transition = S::create_transition(base);
                buffer.push(transition);
            }

            // Handle terminal states
            let terminal_indices: Vec<usize> = step_result
                .dones
                .iter()
                .enumerate()
                .filter_map(|(i, &done)| if done { Some(i) } else { None })
                .collect();

            if !terminal_indices.is_empty() {
                // Reset hidden state for terminated environments
                for &i in &terminal_indices {
                    hidden.reset(i, &inner_device);
                }

                env.reset_envs(&terminal_indices);
                env.write_observations(&mut obs_buffer);
            }

            // Track episode stats
            for (i, (&reward, &done)) in step_result
                .rewards
                .iter()
                .zip(step_result.dones.iter())
                .enumerate()
            {
                episode_rewards[i] += reward;
                if done {
                    let ep_reward = episode_rewards[i];
                    episode_rewards[i] = 0.0;
                    episodes.fetch_add(1, Ordering::Relaxed);
                    {
                        let mut rewards_guard = rewards.lock();
                        if rewards_guard.len() >= MAX_RECENT_REWARDS {
                            rewards_guard.pop_front();
                        }
                        rewards_guard.push_back(ep_reward);
                    }
                }
            }

            env_steps.fetch_add(n_envs, Ordering::Relaxed);
            local_steps += 1;
        }
    }

    /// Learner thread: trains actor and critic networks with UTD pacing.
    fn learner_thread<Actor, Critic, ActorOpt, CriticOpt>(
        config: &SACConfig,
        hidden_config: &HiddenConfig,
        mut actor: Actor,
        mut critic: Critic,
        mut target_critic: Critic,  // Now passed in, not created here
        mut actor_optimizer: ActorOpt,
        mut critic_optimizer: CriticOpt,
        target_entropy: f32,
        actor_bytes_slot: SharedBytesSlot,
        buffer: Arc<SACBuffer<SACTransition<S::TransitionData>>>,
        shutdown: Arc<AtomicBool>,
        train_steps: Arc<AtomicUsize>,
        env_steps: Arc<AtomicUsize>,
        shared_loss_info: Arc<Mutex<SACLossInfo>>,
    ) -> (Actor, Critic)
    where
        Actor: SACActor<B, A, T> + AutodiffModule<B> + Clone,
        Critic: SACCritic<B, A, T> + AutodiffModule<B> + Clone,
        Actor::Record: Send + 'static,
        Critic::Record: Send + 'static,
        ActorOpt: Optimizer<Actor, B>,
        CriticOpt: Optimizer<Critic, B>,
        S::TransitionData: Clone,
    {
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Target critic is passed in (created on main thread for WGPU compatibility)

        // Create target network manager
        let target_manager = TargetNetworkManager::new(TargetNetworkConfig {
            tau: config.tau,
            update_freq: config.target_update_freq,
            hard_update: config.hard_target_update,
        });

        // Create entropy tuner
        let mut entropy_tuner = EntropyTuner::new(
            config.initial_alpha,
            target_entropy,
            &device,
        );

        let mut gradient_step = 0usize;
        let sleep_duration = Duration::from_millis(config.sleep_when_ahead_ms);

        while !shutdown.load(Ordering::Relaxed) {
            // Wait for sufficient buffer size
            if !buffer.is_training_ready() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            // UTD pacing: compute target gradient steps based on env steps
            let current_env_steps = env_steps.load(Ordering::Relaxed);
            let target_gradient_steps = (current_env_steps as f32 * config.utd_ratio) as usize;

            // If learner is ahead of target UTD, sleep briefly to let actors catch up
            if gradient_step >= target_gradient_steps && target_gradient_steps > 0 {
                std::thread::sleep(sleep_duration);
                continue;
            }

            // Train until we reach target gradient steps (or shutdown/buffer exhausted)
            while gradient_step < target_gradient_steps {
                // Sample batch
                let batch = match buffer.sample_batch() {
                    Some(b) => b,
                    None => break, // Buffer exhausted, wait for more data
                };

                // Train on batch using strategy
                let (updated_actor, updated_critic, updated_target_critic, loss_info) =
                    S::train_step(
                        actor,
                        critic,
                        target_critic,
                        &mut actor_optimizer,
                        &mut critic_optimizer,
                        &mut entropy_tuner,
                        &batch,
                        config,
                        hidden_config,
                        &device,
                        gradient_step,
                        &target_manager,
                    );

                actor = updated_actor;
                critic = updated_critic;
                target_critic = updated_target_critic;

                // Update shared loss info
                {
                    let mut guard = shared_loss_info.lock();
                    *guard = loss_info;
                }

                gradient_step += 1;

                // Publish updated actor weights periodically
                if gradient_step % config.model_update_freq == 0 {
                    if let Ok(bytes) = recorder.record(actor.clone().into_record(), ()) {
                        actor_bytes_slot.publish(bytes);
                    }
                }

                // Check shutdown in inner loop for responsiveness
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }

            train_steps.store(gradient_step, Ordering::Relaxed);
        }

        // Final publish
        if let Ok(bytes) = recorder.record(actor.clone().into_record(), ()) {
            actor_bytes_slot.publish(bytes);
        }

        (actor, critic)
    }
}

// ============================================================================
// Convenience Constructors
// ============================================================================

impl<A, B, S> SACRunner<A, FeedForward, B, S>
where
    A: ActionPolicy<B>,
    B: AutodiffBackend,
    S: SACTrainingStrategy<B, A, FeedForward>,
{
    /// Create a feed-forward SAC runner.
    pub fn feed_forward(config: SACConfig) -> Self {
        Self::new(config, HiddenConfig::none())
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

use crate::algorithms::action_policy::{ContinuousPolicy, DiscretePolicy};
use super::sac_strategies::{FeedForwardSACStrategy, RecurrentSACStrategy};

/// Feed-forward SAC runner with discrete actions.
pub type SACDiscrete<B> = SACRunner<DiscretePolicy, FeedForward, B, FeedForwardSACStrategy>;

/// Feed-forward SAC runner with continuous actions.
pub type SACContinuous<B> = SACRunner<ContinuousPolicy, FeedForward, B, FeedForwardSACStrategy>;

/// Recurrent SAC runner with discrete actions.
pub type RecurrentSACDiscrete<B> = SACRunner<DiscretePolicy, Recurrent, B, RecurrentSACStrategy>;

/// Recurrent SAC runner with continuous actions.
pub type RecurrentSACContinuous<B> = SACRunner<ContinuousPolicy, Recurrent, B, RecurrentSACStrategy>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_aliases_compile() {
        // Compile-time checks
        fn _check_sac_discrete<B: AutodiffBackend>(_: SACDiscrete<B>) {}
        fn _check_sac_continuous<B: AutodiffBackend>(_: SACContinuous<B>) {}
        fn _check_rec_sac_discrete<B: AutodiffBackend>(_: RecurrentSACDiscrete<B>) {}
        fn _check_rec_sac_continuous<B: AutodiffBackend>(_: RecurrentSACContinuous<B>) {}
    }

    #[test]
    fn test_default_config() {
        use burn::backend::{Autodiff, NdArray};
        type B = Autodiff<NdArray<f32>>;

        let config = SACConfig::continuous();
        let _runner: SACContinuous<B> = SACRunner::feed_forward(config);

        let config = SACConfig::discrete();
        let hidden_config = HiddenConfig::lstm(64);
        let _runner: RecurrentSACDiscrete<B> = SACRunner::new(config, hidden_config);
    }
}
