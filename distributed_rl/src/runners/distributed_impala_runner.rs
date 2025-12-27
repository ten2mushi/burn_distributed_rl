//! Generic distributed IMPALA runner with multi-actor architecture.
//!
//! This module provides `DistributedIMPALARunner`, a high-level API for multi-actor
//! IMPALA training with V-trace off-policy correction. It coordinates N actor threads
//! and 1 learner thread, using `RecordSlot` for thread-safe weight transfer.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                        Main Thread                              │
//! │  • Validates model factory compatibility                       │
//! │  • Spawns actors and learner                                   │
//! │  • Monitors progress                                           │
//! │  • Handles shutdown                                            │
//! └────────────────────────────────────────────────────────────────┘
//!          ↓                                              ↓
//! ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
//! │   Actor 0        │  │   Actor 1        │  │   Actor N-1      │
//! │  • Own WGPU dev  │  │  • Own WGPU dev  │  │  • Own WGPU dev  │
//! │  • Factory→Model │  │  • Factory→Model │  │  • Factory→Model │
//! │  • M envs        │  │  • M envs        │  │  • M envs        │
//! │  • Async collect │  │  • Async collect │  │  • Async collect │
//! │  • Push trajs    │  │  • Push trajs    │  │  • Push trajs    │
//! └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
//!          └───────────────┬─────┴─────────────────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │     IMPALABuffer     │
//!               │  • FIFO trajectory   │
//!               │  • Lock-free push    │
//!               │  • Batch sampling    │
//!               └──────────┬───────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │      Learner         │
//!               │  • Sample batches    │
//!               │  • V-trace correction│
//!               │  • Train & publish   │
//!               └──────────────────────┘
//! ```
//!
//! # Key Differences from PPO
//!
//! - **Off-policy**: Actors don't wait for learner to consume data
//! - **V-trace**: Importance sampling correction for policy lag
//! - **Trajectory FIFO**: Oldest trajectories processed first
//! - **Async collection**: Actors continuously push without synchronization
//!
//! # Type Aliases
//!
//! - `DistributedIMPALADiscrete<B>`: Feed-forward discrete
//! - `DistributedIMPALAContinuous<B>`: Feed-forward continuous

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::actor_critic::{ActorCritic, ActorCriticInference};
use crate::algorithms::impala::{IMPALABuffer, IMPALABufferConfig};
use crate::algorithms::impala::{IMPALAConfig, IMPALAStats};
use crate::algorithms::temporal_policy::TemporalPolicy;
use crate::algorithms::vtrace::compute_vtrace;
use crate::core::bytes_slot::{bytes_slot_with, SharedBytesSlot};
use crate::core::experience_buffer::ExperienceBuffer;
use crate::core::transition::{IMPALATransition, Trajectory, Transition};

use super::learner::VectorizedEnv;

/// Maximum number of recent episode rewards to keep for statistics.
/// Prevents unbounded memory growth during long training runs.
const MAX_RECENT_REWARDS: usize = 1000;

// ============================================================================
// DistributedIMPALARunner
// ============================================================================

/// Generic distributed IMPALA runner.
///
/// Coordinates N actor threads + 1 learner thread for IMPALA training
/// with V-trace off-policy correction.
///
/// # Type Parameters
///
/// - `A`: Action policy (`DiscretePolicy` or `ContinuousPolicy`)
/// - `T`: Temporal policy (`FeedForward` or `Recurrent`)
/// - `B`: Autodiff backend (e.g., `Autodiff<Wgpu>`)
///
/// # Off-Policy Training
///
/// Unlike PPO, IMPALA actors collect asynchronously. The learner samples
/// batches from a FIFO buffer and applies V-trace correction to handle
/// policy lag between collection and training time.
pub struct DistributedIMPALARunner<A, T, B>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
    B: AutodiffBackend,
{
    config: IMPALAConfig,
    _marker: PhantomData<(A, T, B)>,
}

impl<A, T, B> DistributedIMPALARunner<A, T, B>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
    B: AutodiffBackend,
{
    /// Create a new distributed IMPALA runner.
    pub fn new(config: IMPALAConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &IMPALAConfig {
        &self.config
    }

    /// Create a configured Adam optimizer with gradient clipping.
    ///
    /// Uses the `max_grad_norm` from config for gradient norm clipping.
    /// This is a convenience method - you can also create your own optimizer
    /// with custom configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runner = DistributedIMPALADiscrete::new(config);
    /// let optimizer = runner.create_optimizer::<MyModel>();
    /// runner.run(model_factory, initial_model, env_factory, optimizer, callback);
    /// ```
    pub fn create_optimizer<M>(&self) -> impl Optimizer<M, B>
    where
        M: AutodiffModule<B>,
    {
        let mut adam_config = AdamConfig::new().with_epsilon(1e-5);

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            adam_config = adam_config.with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm)));
        }

        adam_config.init()
    }

    /// Run distributed IMPALA training.
    ///
    /// # Arguments
    ///
    /// - `model_factory`: Creates a fresh model architecture on a given device.
    ///   Called once per actor thread.
    /// - `initial_model`: Model with initial/pretrained weights.
    /// - `env_factory`: Creates environment for given (actor_id, n_envs)
    /// - `optimizer`: Optimizer for training
    /// - `callback`: Called periodically with training stats
    ///
    /// # Returns
    ///
    /// Trained model
    pub fn run<M, MF, EF, E, O, F>(
        &self,
        model_factory: MF,
        initial_model: M,
        env_factory: EF,
        optimizer: O,
        callback: F,
    ) -> M
    where
        MF: Fn(&B::Device) -> M + Send + Sync + Clone + 'static,
        M: ActorCritic<B, A, T> + Clone + 'static,
        M::Record: Send + 'static,
        // Inner backend bounds for actor inference (no autodiff graphs)
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        EF: Fn(usize, usize) -> E + Send + Sync + Clone + 'static,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action> + 'static,
        O: Optimizer<M, B> + 'static,
        F: Fn(&IMPALAStats),
    {
        let device = B::Device::default();
        let config = self.config.clone();

        println!("=== Distributed IMPALA Runner ===");
        println!(
            "Actors: {} x {} envs = {} total",
            config.n_actors,
            config.n_envs_per_actor,
            config.total_envs()
        );
        println!(
            "Trajectory length: {}, Buffer capacity: {}",
            config.trajectory_length, config.buffer_capacity
        );

        // Validate model factory
        println!("Validating model factory compatibility...");
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        {
            let test_model = model_factory(&device);
            let test_record = initial_model.clone().into_record();
            let _loaded = test_model.load_record(test_record);
            println!("Model factory validated successfully.");
        }

        // Serialize initial weights to bytes (Vec<u8> is Send + Sync)
        let initial_bytes = recorder
            .record(initial_model.clone().into_record(), ())
            .expect("Failed to serialize initial model");

        // Create shared state using bytes-based transfer for WGPU compatibility
        let bytes_slot: SharedBytesSlot = bytes_slot_with(initial_bytes);
        let buffer = Arc::new(IMPALABuffer::new(IMPALABufferConfig {
            n_actors: config.n_actors,
            n_envs_per_actor: config.n_envs_per_actor,
            trajectory_length: config.trajectory_length,
            max_trajectories: config.buffer_capacity,
            batch_size: config.batch_size,
        }));
        let shutdown = Arc::new(AtomicBool::new(false));
        let policy_version = Arc::new(AtomicU64::new(0));
        let total_env_steps = Arc::new(AtomicUsize::new(0));
        let total_episodes = Arc::new(AtomicUsize::new(0));
        let train_steps = Arc::new(AtomicUsize::new(0));
        let recent_rewards = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(MAX_RECENT_REWARDS)));

        // Spawn actor threads
        let mut actor_handles = Vec::with_capacity(config.n_actors);
        for actor_id in 0..config.n_actors {
            let model_factory = model_factory.clone();
            let env_factory = env_factory.clone();
            let buffer = Arc::clone(&buffer);
            let slot = Arc::clone(&bytes_slot);
            let shutdown_flag = Arc::clone(&shutdown);
            let env_steps = Arc::clone(&total_env_steps);
            let episodes = Arc::clone(&total_episodes);
            let rewards = Arc::clone(&recent_rewards);
            let version = Arc::clone(&policy_version);
            let cfg = config.clone();

            let handle = std::thread::Builder::new()
                .name(format!("IMPALA-Actor-{}", actor_id))
                .spawn(move || {
                    Self::actor_thread::<M, MF, EF, E>(
                        actor_id,
                        &cfg,
                        model_factory,
                        env_factory,
                        slot,
                        buffer,
                        shutdown_flag,
                        env_steps,
                        episodes,
                        rewards,
                        version,
                    );
                })
                .expect("Failed to spawn actor thread");

            actor_handles.push(handle);
        }

        // Spawn learner thread
        let learner_buffer = Arc::clone(&buffer);
        let learner_slot = Arc::clone(&bytes_slot);
        let learner_shutdown = Arc::clone(&shutdown);
        let learner_version = Arc::clone(&policy_version);
        let learner_train_steps = Arc::clone(&train_steps);
        let cfg = config.clone();
        let learner_model = initial_model;
        let learner_optimizer = optimizer;

        let learner_handle = std::thread::Builder::new()
            .name("IMPALA-Learner".to_string())
            .spawn(move || {
                Self::learner_thread::<M, O>(
                    &cfg,
                    learner_model,
                    learner_optimizer,
                    learner_slot,
                    learner_buffer,
                    learner_shutdown,
                    learner_version,
                    learner_train_steps,
                )
            })
            .expect("Failed to spawn learner thread");

        // Main monitoring loop
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let mut stats = IMPALAStats::default();

        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));

            let env_steps = total_env_steps.load(Ordering::Relaxed);
            let episodes = total_episodes.load(Ordering::Relaxed);
            let version = policy_version.load(Ordering::Relaxed);
            let t_steps = train_steps.load(Ordering::Relaxed);

            // Calculate average reward
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
            stats.avg_reward = avg_reward;
            stats.policy_version = version;
            stats.steps_per_second = env_steps as f32 / start_time.elapsed().as_secs_f32();
            stats.buffer_size = buffer.len();
            stats.train_steps = t_steps;

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

        let trained_model = learner_handle.join().expect("Learner thread panicked");

        println!("\n=== Training Complete ===");
        println!("Duration: {:.1}s", start_time.elapsed().as_secs_f32());
        println!(
            "Total steps: {}",
            total_env_steps.load(Ordering::Relaxed)
        );
        println!(
            "Total episodes: {}",
            total_episodes.load(Ordering::Relaxed)
        );

        trained_model
    }

    /// Actor thread: collects trajectories and pushes to buffer asynchronously.
    ///
    /// Uses inner backend (non-autodiff) for inference to avoid computation graph
    /// accumulation. This is critical for maintaining consistent SPS throughout training.
    fn actor_thread<M, MF, EF, E>(
        actor_id: usize,
        config: &IMPALAConfig,
        model_factory: MF,
        env_factory: EF,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<IMPALABuffer>,
        shutdown: Arc<AtomicBool>,
        env_steps: Arc<AtomicUsize>,
        episodes: Arc<AtomicUsize>,
        rewards: Arc<Mutex<VecDeque<f32>>>,
        policy_version: Arc<AtomicU64>,
    ) where
        MF: Fn(&B::Device) -> M,
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        // Inner backend bounds for graph-free inference
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        EF: Fn(usize, usize) -> E,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action>,
    {
        // Create devices: autodiff for weight loading, inner for inference
        let device = B::Device::default();
        let inner_device = <B::InnerBackend as Backend>::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Create model using factory (on autodiff backend for weight loading)
        let mut model = model_factory(&device);

        // Load initial weights from bytes
        // NOTE: Use get() instead of take() to allow multiple actors to access initial weights
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            if let Some(bytes) = bytes_slot.get() {
                if let Ok(record) = recorder.load(bytes, &device) {
                    model = model.load_record(record);
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        // Get inference model on inner backend (no autodiff - no computation graphs!)
        // This is the key optimization: forward passes won't accumulate orphaned graphs.
        let mut inference_model: M::InnerModule = model.valid();

        // Create environment
        let mut env = env_factory(actor_id, config.n_envs_per_actor);
        let obs_size = env.obs_size();
        let n_envs = config.n_envs_per_actor;
        let global_env_offset = actor_id * n_envs;

        // Initialize state
        let mut obs_buffer = vec![0.0f32; n_envs * obs_size];
        env.write_observations(&mut obs_buffer);
        let mut episode_rewards = vec![0.0f32; n_envs];

        // Per-environment trajectory builders
        let mut trajectories: Vec<Trajectory<IMPALATransition>> = (0..n_envs)
            .map(|i| Trajectory::with_capacity(global_env_offset + i, config.trajectory_length))
            .collect();

        let mut last_version = bytes_slot.version();
        let mut local_steps = 0usize;

        while !shutdown.load(Ordering::Relaxed) {
            // Check for weight updates periodically
            // NOTE: Use get() instead of take() to allow multiple actors to access weights
            if local_steps % config.model_update_freq == 0 {
                let current_version = bytes_slot.version();
                if current_version > last_version {
                    if let Some(bytes) = bytes_slot.get() {
                        if let Ok(record) = recorder.load(bytes, &device) {
                            model = model.load_record(record);
                            // Refresh inference model with updated weights
                            inference_model = model.valid();
                            last_version = current_version;
                        }
                    }
                }
            }

            // Use Acquire ordering to ensure we see all writes that happened before the Release
            let current_policy_version = policy_version.load(Ordering::Acquire);

            // Save current obs
            let current_obs: Vec<Vec<f32>> = (0..n_envs)
                .map(|i| {
                    let start = i * obs_size;
                    obs_buffer[start..start + obs_size].to_vec()
                })
                .collect();

            // Forward pass on INNER backend - NO computation graph accumulation!
            let obs_tensor =
                Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                    .reshape([n_envs, obs_size]);

            let hidden = inference_model.initial_hidden(n_envs, &inner_device);
            let output = inference_model.forward(obs_tensor, hidden);

            // Sample actions (also on inner backend - no graph overhead)
            let (actions, log_probs) = output.sample_actions(&inner_device);

            // Step environment
            let step_result = env.step(&actions);

            // Get next observations
            env.write_observations(&mut obs_buffer);
            let next_obs: Vec<Vec<f32>> = (0..n_envs)
                .map(|i| {
                    let start = i * obs_size;
                    obs_buffer[start..start + obs_size].to_vec()
                })
                .collect();

            // Create transitions and add to trajectories
            for i in 0..n_envs {
                let action_floats = actions[i].as_floats();
                let action = if action_floats.len() == 1 {
                    crate::core::transition::Action::Discrete(action_floats[0] as u32)
                } else {
                    crate::core::transition::Action::Continuous(action_floats)
                };

                let transition = IMPALATransition {
                    base: Transition {
                        state: current_obs[i].clone(),
                        action,
                        reward: step_result.rewards[i],
                        next_state: next_obs[i].clone(),
                        terminal: step_result.terminals[i],
                        truncated: step_result.dones[i] && !step_result.terminals[i],
                    },
                    behavior_log_prob: log_probs[i],
                    policy_version: current_policy_version,
                };

                trajectories[i].push(transition);

                // Track episode rewards
                episode_rewards[i] += step_result.rewards[i];

                // Check trajectory completion
                let done = step_result.dones[i];
                if trajectories[i].len() >= config.trajectory_length || done {
                    if done {
                        trajectories[i].episode_return = Some(episode_rewards[i]);
                        episodes.fetch_add(1, Ordering::Relaxed);
                        {
                            let mut rewards_guard = rewards.lock();
                            if rewards_guard.len() >= MAX_RECENT_REWARDS {
                                rewards_guard.pop_front();
                            }
                            rewards_guard.push_back(episode_rewards[i]);
                        }
                        episode_rewards[i] = 0.0;
                    }

                    // Push completed trajectory to buffer
                    let completed = std::mem::replace(
                        &mut trajectories[i],
                        Trajectory::with_capacity(global_env_offset + i, config.trajectory_length),
                    );
                    buffer.push_trajectory(completed);
                }
            }

            // Reset done environments
            let reset_indices: Vec<usize> = step_result
                .dones
                .iter()
                .enumerate()
                .filter_map(|(i, &done)| if done { Some(i) } else { None })
                .collect();

            if !reset_indices.is_empty() {
                env.reset_envs(&reset_indices);
                env.write_observations(&mut obs_buffer);
            }

            env_steps.fetch_add(n_envs, Ordering::Relaxed);
            local_steps += 1;
        }
    }

    /// Learner thread: samples batches, applies V-trace, trains model.
    fn learner_thread<M, O>(
        config: &IMPALAConfig,
        mut model: M,
        mut optimizer: O,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<IMPALABuffer>,
        shutdown: Arc<AtomicBool>,
        policy_version: Arc<AtomicU64>,
        train_steps: Arc<AtomicUsize>,
    ) -> M
    where
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        O: Optimizer<M, B>,
    {
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Pre-allocate reusable buffers outside the loop to reduce allocation overhead.
        // This is a significant optimization: Vec::with_capacity() inside the loop was
        // causing continuous allocations that degraded SPS by ~30% over long runs.
        // estimate: batch_size(32) * trajectory_length(20) = 640 transitions typical
        let estimated_transitions = config.batch_size * config.trajectory_length;
        let estimated_obs_size = 4; // CartPole obs_size, will grow if needed
        let mut all_states: Vec<f32> =
            Vec::with_capacity(estimated_transitions * estimated_obs_size);
        let mut all_actions: Vec<A::Action> = Vec::with_capacity(estimated_transitions);
        let mut all_advantages: Vec<f32> = Vec::with_capacity(estimated_transitions);
        let mut all_vtrace_targets: Vec<f32> = Vec::with_capacity(estimated_transitions);
        let mut all_rhos: Vec<f32> = Vec::with_capacity(estimated_transitions);
        let mut all_values: Vec<f32> = Vec::with_capacity(estimated_transitions);
        let mut all_target_log_probs: Vec<f32> = Vec::with_capacity(estimated_transitions);

        while !shutdown.load(Ordering::Relaxed) {
            // Wait for batch
            if !buffer.is_training_ready() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Sample batch
            let batch = match buffer.sample_batch() {
                Some(b) => b,
                None => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
            };

            // ==== BATCHED PROCESSING ====
            // Process all trajectories in a single forward/backward pass for efficiency

            // Filter out empty trajectories
            let valid_trajs: Vec<_> = batch
                .trajectories
                .iter()
                .filter(|t| !t.is_empty())
                .collect();

            if valid_trajs.is_empty() {
                continue;
            }

            // Get observation size from first valid trajectory
            let obs_size = valid_trajs[0].transitions[0].base.state.len();

            // 1. Collect all states into a single batch (clear and reuse pre-allocated buffers)
            let total_transitions: usize = valid_trajs.iter().map(|t| t.len()).sum();
            all_states.clear();
            all_actions.clear();

            for traj in valid_trajs.iter() {
                for tr in traj.iter() {
                    all_states.extend_from_slice(&tr.base.state);
                    let action = match &tr.base.action {
                        crate::core::transition::Action::Discrete(a) => {
                            A::Action::from_floats(&[*a as f32])
                        }
                        crate::core::transition::Action::Continuous(a) => {
                            A::Action::from_floats(a)
                        }
                    };
                    all_actions.push(action);
                }
            }

            // 2. Single forward pass for training
            // NOTE: We avoid a separate forward pass for bootstrap values because:
            // - Extra forward passes create computation graphs never consumed by backward()
            // - Burn's Autodiff+WGPU doesn't immediately free these orphaned graphs
            // - This causes GPU resource accumulation and eventual freeze
            // Instead, we approximate V(s_n) ≈ V(s_{n-1}) for bootstrap, which is
            // acceptable for short trajectories with smooth value functions.
            let states_tensor = Tensor::<B, 1>::from_floats(all_states.as_slice(), &device)
                .reshape([total_transitions, obs_size]);
            let hidden = model.initial_hidden(total_transitions, &device);
            let output = model.forward(states_tensor, hidden);

            // 3. Extract values and compute log probs (reusing pre-allocated buffers)
            let values_flat = output.values_flat();
            all_values.clear();
            all_values.extend(
                values_flat
                    .clone()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap(),
            );

            let log_probs_tensor = output.policy.log_prob(&all_actions, &device);
            all_target_log_probs.clear();
            all_target_log_probs.extend(
                log_probs_tensor
                    .clone()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap(),
            );

            // 4. Compute V-trace per trajectory (clear and reuse pre-allocated buffers)
            all_advantages.clear();
            all_vtrace_targets.clear();
            all_rhos.clear();
            let mut offset = 0;

            for traj in valid_trajs.iter() {
                let traj_len = traj.len();

                // Extract per-trajectory data
                let traj_rewards: Vec<f32> = traj.iter().map(|t| t.base.reward).collect();
                let traj_dones: Vec<bool> = traj.iter().map(|t| t.done()).collect();
                // Extract TRUE terminals (not truncated) for bootstrap decision
                let traj_terminals: Vec<bool> = traj.iter().map(|t| t.base.terminal).collect();
                let behavior_log_probs: Vec<f32> =
                    traj.iter().map(|t| t.behavior_log_prob).collect();

                // Get target log probs and values for this trajectory
                let target_log_probs = &all_target_log_probs[offset..offset + traj_len];
                let values = &all_values[offset..offset + traj_len];

                // Bootstrap value: 0 for TRUE terminal, V(s_{n-1}) approximation for non-terminal
                // NOTE: Ideally we'd use V(s_n) but that requires an extra forward pass which
                // causes GPU resource leaks in Burn's Autodiff+WGPU. The approximation
                // V(s_n) ≈ V(s_{n-1}) is acceptable when value function is smooth.
                let bootstrap_value = if traj_terminals.last().copied().unwrap_or(true) {
                    0.0 // TRUE terminal - episode ended
                } else {
                    // Approximation: use V(s_{n-1}) instead of V(s_n)
                    values.last().copied().unwrap_or(0.0)
                };

                // Compute V-trace for this trajectory
                let vtrace = compute_vtrace(
                    &behavior_log_probs,
                    target_log_probs,
                    &traj_rewards,
                    values,
                    &traj_dones,
                    bootstrap_value,
                    config.gamma,
                    config.rho_clip,
                    config.c_clip,
                );

                // Collect results
                all_advantages.extend(&vtrace.advantages);
                all_vtrace_targets.extend(&vtrace.vs);
                all_rhos.extend(&vtrace.rhos);

                offset += traj_len;
            }

            // 6. Note: IMPALA does NOT normalize advantages (unlike PPO)
            // For constant-reward environments like CartPole, normalization would
            // zero out gradients since all 1-step TD errors are similar.
            // V-trace targets already provide appropriate scaling.

            // 7. Create tensors for loss computation
            let advantages_tensor =
                Tensor::<B, 1>::from_floats(all_advantages.as_slice(), &device);
            let vtrace_targets_tensor =
                Tensor::<B, 1>::from_floats(all_vtrace_targets.as_slice(), &device);
            let rhos_tensor = Tensor::<B, 1>::from_floats(all_rhos.as_slice(), &device);

            // 8. Compute batched losses
            // Policy loss: -rho * log_prob * advantage (rho applied here, not in advantages)
            let policy_loss =
                -(rhos_tensor * log_probs_tensor.clone() * advantages_tensor).mean();

            // Value loss: MSE to V-trace targets
            let value_loss = (values_flat - vtrace_targets_tensor).powf_scalar(2.0).mean();

            // Entropy bonus
            let entropy = output.policy.entropy();
            let entropy_loss = -entropy.mean();

            // Total loss
            let total_loss = policy_loss
                + value_loss.mul_scalar(config.vf_coef)
                + entropy_loss.mul_scalar(config.entropy_coef);

            // 9. Single backward pass
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // 10. Single optimizer step (for entire batch)
            model = optimizer.step(config.learning_rate, model, grads);

            // Publish updated weights as bytes
            if let Ok(bytes) = recorder.record(model.clone().into_record(), ()) {
                bytes_slot.publish(bytes);
            }
            policy_version.fetch_add(1, Ordering::Release);
            train_steps.fetch_add(1, Ordering::Relaxed);
        }

        model
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

use crate::algorithms::action_policy::{ContinuousPolicy, DiscretePolicy};
use crate::algorithms::temporal_policy::{FeedForward, Recurrent};

/// Distributed IMPALA with discrete actions and feed-forward policy.
pub type DistributedIMPALADiscrete<B> = DistributedIMPALARunner<DiscretePolicy, FeedForward, B>;

/// Distributed IMPALA with continuous actions and feed-forward policy.
pub type DistributedIMPALAContinuous<B> =
    DistributedIMPALARunner<ContinuousPolicy, FeedForward, B>;

/// Distributed IMPALA with discrete actions and recurrent policy.
pub type DistributedRecurrentIMPALADiscrete<B> =
    DistributedIMPALARunner<DiscretePolicy, Recurrent, B>;

/// Distributed IMPALA with continuous actions and recurrent policy.
pub type DistributedRecurrentIMPALAContinuous<B> =
    DistributedIMPALARunner<ContinuousPolicy, Recurrent, B>;
