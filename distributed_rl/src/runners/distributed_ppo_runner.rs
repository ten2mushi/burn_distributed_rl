//! Generic distributed PPO runner with multi-actor architecture.
//!
//! This module provides `DistributedPPORunner`, a high-level API for multi-actor
//! PPO training. It coordinates N actor threads and 1 learner thread, using
//! bytes-based weight transfer for WGPU compatibility.
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
//! │  • Load weights  │  │  • Load weights  │  │  • Load weights  │
//! │  • Collect exp   │  │  • Collect exp   │  │  • Collect exp   │
//! └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
//!          └───────────────┬─────┴─────────────────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │  DistributedPPOBuffer │
//!               │  • Per-env rollouts   │
//!               │  • Sync on epoch      │
//!               └──────────┬───────────┘
//!                          ↓
//!               ┌──────────────────────┐
//!               │      Learner         │
//!               │  • Compute GAE       │
//!               │  • Train epochs      │
//!               │  • Publish weights   │
//!               └──────────────────────┘
//! ```
//!
//! # Weight Transfer
//!
//! Uses bytes-based serialization (`BinBytesRecorder`) for weight transfer.
//! This is required because WGPU model records may not implement `Sync`.
//!
//! # Type Aliases
//!
//! - `DistributedPPODiscrete<B>`: Feed-forward discrete
//! - `DistributedPPOContinuous<B>`: Feed-forward continuous
//! - `DistributedRecurrentPPODiscrete<B>`: Recurrent discrete
//! - `DistributedRecurrentPPOContinuous<B>`: Recurrent continuous

use burn::optim::{GradientsParams, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::actor_critic::ActorCritic;
use crate::algorithms::gae::compute_gae;
use crate::algorithms::temporal_policy::TemporalPolicy;
use crate::core::bytes_slot::{bytes_slot_with, SharedBytesSlot};
use crate::core::transition::PPOTransition;

use super::distributed_ppo_config::{DistributedPPOConfig, DistributedPPOStats};
use super::learner::VectorizedEnv;

use crate::algorithms::ppo::distributed_ppo_buffer::DistributedPPOBuffer;

/// Maximum number of recent episode rewards to keep for statistics.
/// Prevents unbounded memory growth during long training runs.
const MAX_RECENT_REWARDS: usize = 1000;

// ============================================================================
// DistributedPPORunner
// ============================================================================

/// Generic distributed PPO runner.
///
/// Coordinates N actor threads + 1 learner thread for PPO training.
/// Uses bytes-based weight transfer for WGPU compatibility.
///
/// # Type Parameters
///
/// - `A`: Action policy (`DiscretePolicy` or `ContinuousPolicy`)
/// - `T`: Temporal policy (`FeedForward` or `Recurrent`)
/// - `B`: Autodiff backend (e.g., `Autodiff<Wgpu>`)
pub struct DistributedPPORunner<A, T, B>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
    B: AutodiffBackend,
{
    config: DistributedPPOConfig,
    _marker: PhantomData<(A, T, B)>,
}

impl<A, T, B> DistributedPPORunner<A, T, B>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
    B: AutodiffBackend,
{
    /// Create a new distributed PPO runner.
    pub fn new(config: DistributedPPOConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &DistributedPPOConfig {
        &self.config
    }

    /// Run distributed PPO training.
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
        EF: Fn(usize, usize) -> E + Send + Sync + Clone + 'static,
        E: VectorizedEnv<A::Action> + 'static,
        O: Optimizer<M, B> + 'static,
        F: Fn(&DistributedPPOStats),
    {
        let device = B::Device::default();
        let config = self.config.clone();

        println!("=== Distributed PPO Runner ===");
        println!(
            "Actors: {} x {} envs = {} total",
            config.n_actors,
            config.n_envs_per_actor,
            config.total_envs()
        );
        println!(
            "Rollout: {} steps x {} envs = {} transitions",
            config.rollout_length,
            config.total_envs(),
            config.transitions_per_rollout()
        );

        // Validate configuration early to fail fast with clear error
        config.validate().expect("Invalid DistributedPPOConfig");

        // Validate model factory
        println!("Validating model factory compatibility...");
        {
            let test_model = model_factory(&device);
            let test_record = initial_model.clone().into_record();
            let _loaded = test_model.load_record(test_record);
            println!("Model factory validated successfully.");
        }

        // Serialize initial model to bytes
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let initial_bytes = recorder
            .record(initial_model.clone().into_record(), ())
            .expect("Failed to serialize initial model");
        println!("Initial model serialized: {} bytes", initial_bytes.len());

        // Create shared state
        let bytes_slot: SharedBytesSlot = bytes_slot_with(initial_bytes);
        let buffer = Arc::new(DistributedPPOBuffer::new(
            config.rollout_length,
            config.total_envs(),
        ));
        let shutdown = Arc::new(AtomicBool::new(false));
        let policy_version = Arc::new(AtomicU64::new(0));
        let total_env_steps = Arc::new(AtomicUsize::new(0));
        let total_episodes = Arc::new(AtomicUsize::new(0));
        let recent_rewards = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(MAX_RECENT_REWARDS)));

        // Synchronization for on-policy: actors wait for learner to consume
        let consumed_epoch = Arc::new(AtomicU64::new(0));

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
            let consumed = Arc::clone(&consumed_epoch);
            let cfg = config.clone();

            let handle = std::thread::Builder::new()
                .name(format!("PPO-Actor-{}", actor_id))
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
                        consumed,
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
        let learner_consumed = Arc::clone(&consumed_epoch);
        let cfg = config.clone();
        let learner_model = initial_model;
        let learner_optimizer = optimizer;

        let learner_handle = std::thread::Builder::new()
            .name("PPO-Learner".to_string())
            .spawn(move || {
                Self::learner_thread::<M, O>(
                    &cfg,
                    learner_model,
                    learner_optimizer,
                    learner_slot,
                    learner_buffer,
                    learner_shutdown,
                    learner_version,
                    learner_consumed,
                )
            })
            .expect("Failed to spawn learner thread");

        // Main monitoring loop
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let mut stats = DistributedPPOStats::default();

        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));

            let env_steps = total_env_steps.load(Ordering::Relaxed);
            let episodes = total_episodes.load(Ordering::Relaxed);
            let version = policy_version.load(Ordering::Relaxed);

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
            stats.buffer_size = buffer.total_transitions();

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

    /// Actor thread: creates model via factory, collects experience, pushes to buffer.
    fn actor_thread<M, MF, EF, E>(
        actor_id: usize,
        config: &DistributedPPOConfig,
        model_factory: MF,
        env_factory: EF,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<DistributedPPOBuffer>,
        shutdown: Arc<AtomicBool>,
        env_steps: Arc<AtomicUsize>,
        episodes: Arc<AtomicUsize>,
        rewards: Arc<Mutex<VecDeque<f32>>>,
        consumed_epoch: Arc<AtomicU64>,
    ) where
        MF: Fn(&B::Device) -> M,
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        EF: Fn(usize, usize) -> E,
        E: VectorizedEnv<A::Action>,
    {
        // Create actor's own device
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Create model using factory
        let mut model = model_factory(&device);

        // Load initial weights from bytes slot
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

        // Create environment
        let mut env = env_factory(actor_id, config.n_envs_per_actor);
        let obs_size = env.obs_size();
        let n_envs = config.n_envs_per_actor;
        let global_env_offset = actor_id * n_envs;

        // Initialize state
        let mut obs_buffer = vec![0.0f32; n_envs * obs_size];
        env.write_observations(&mut obs_buffer);
        let mut episode_rewards = vec![0.0f32; n_envs];
        let mut last_version = bytes_slot.version();
        let mut local_epoch = 0u64;
        let mut steps_this_epoch = 0usize;

        while !shutdown.load(Ordering::Relaxed) {
            // Synchronization: wait for learner to consume previous epoch
            if steps_this_epoch >= config.rollout_length {
                while consumed_epoch.load(Ordering::Acquire) <= local_epoch {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }

                // Get updated weights if version changed
                let current_version = bytes_slot.version();
                if current_version > last_version {
                    if let Some(bytes) = bytes_slot.get() {
                        if let Ok(record) = recorder.load(bytes, &device) {
                            model = model.load_record(record);
                            last_version = current_version;
                        }
                    }
                }

                local_epoch += 1;
                steps_this_epoch = 0;
            }

            // Save current obs
            let current_obs: Vec<Vec<f32>> = (0..n_envs)
                .map(|i| {
                    let start = i * obs_size;
                    obs_buffer[start..start + obs_size].to_vec()
                })
                .collect();

            // Forward pass
            let obs_tensor = Tensor::<B, 1>::from_floats(obs_buffer.as_slice(), &device)
                .reshape([n_envs, obs_size]);

            let hidden = model.initial_hidden(n_envs, &device);
            let output = model.forward(obs_tensor, hidden);

            // Sample actions
            let (actions, log_probs) = output.sample_actions(&device);
            let values: Vec<f32> = output
                .values_flat()
                .into_data()
                .as_slice()
                .unwrap()
                .to_vec();

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

            // Create transitions
            let mut transitions = Vec::with_capacity(n_envs);
            for i in 0..n_envs {
                let action_floats = actions[i].as_floats();
                let action = if action_floats.len() == 1 {
                    crate::core::transition::Action::Discrete(action_floats[0] as u32)
                } else {
                    crate::core::transition::Action::Continuous(action_floats)
                };

                transitions.push(PPOTransition {
                    base: crate::core::transition::Transition {
                        state: current_obs[i].clone(),
                        action,
                        reward: step_result.rewards[i],
                        next_state: next_obs[i].clone(),
                        terminal: step_result.terminals[i],
                        truncated: step_result.dones[i] && !step_result.terminals[i],
                    },
                    log_prob: log_probs[i],
                    value: values[i],
                });
            }

            // Push to buffer
            buffer.push_batch(transitions, global_env_offset);

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
            steps_this_epoch += 1;
        }
    }

    /// Learner thread: trains on rollouts and publishes weights.
    fn learner_thread<M, O>(
        config: &DistributedPPOConfig,
        mut model: M,
        mut optimizer: O,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<DistributedPPOBuffer>,
        shutdown: Arc<AtomicBool>,
        policy_version: Arc<AtomicU64>,
        consumed_epoch: Arc<AtomicU64>,
    ) -> M
    where
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        O: Optimizer<M, B>,
    {
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        while !shutdown.load(Ordering::Relaxed) {
            // Wait for rollout
            if !buffer.is_ready() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Consume rollouts
            let per_env_rollouts = buffer.consume();

            // Compute GAE for each environment
            let mut all_states: Vec<f32> = Vec::new();
            let mut all_actions: Vec<A::Action> = Vec::new();
            let mut all_old_log_probs: Vec<f32> = Vec::new();
            let mut all_old_values: Vec<f32> = Vec::new();
            let mut all_advantages: Vec<f32> = Vec::new();
            let mut all_returns: Vec<f32> = Vec::new();

            for rollout in per_env_rollouts {
                if rollout.is_empty() {
                    continue;
                }

                let rewards: Vec<f32> = rollout.iter().map(|t| t.base.reward).collect();
                let values: Vec<f32> = rollout.iter().map(|t| t.value).collect();
                let dones: Vec<bool> = rollout.iter().map(|t| t.done()).collect();
                // Extract TRUE terminals (not truncated) for bootstrap decision
                let terminals: Vec<bool> = rollout.iter().map(|t| t.base.terminal).collect();

                // Bootstrap value: 0 only if TRUE terminal, else use V(s')
                // Critical distinction: truncated episodes (time limit) still need bootstrap
                // because the episode didn't truly end - the agent could have continued.
                let bootstrap = if terminals.last().copied().unwrap_or(true) {
                    0.0 // TRUE terminal - episode ended, no future value
                } else {
                    values.last().copied().unwrap_or(0.0) // Truncated or ongoing - use V(s')
                };

                // Compute GAE
                let (advantages, returns) = compute_gae(
                    &rewards,
                    &values,
                    &dones,
                    bootstrap,
                    config.gamma,
                    config.gae_lambda,
                );

                // Collect training data
                for (i, t) in rollout.iter().enumerate() {
                    all_states.extend(&t.base.state);

                    // Convert action
                    let action = match &t.base.action {
                        crate::core::transition::Action::Discrete(a) => {
                            A::Action::from_floats(&[*a as f32])
                        }
                        crate::core::transition::Action::Continuous(a) => {
                            A::Action::from_floats(a)
                        }
                    };
                    all_actions.push(action);

                    all_old_log_probs.push(t.log_prob);
                    all_old_values.push(t.value);
                    all_advantages.push(advantages[i]);
                    all_returns.push(returns[i]);
                }
            }

            if all_states.is_empty() {
                // Signal actors that this epoch is consumed
                consumed_epoch.fetch_add(1, Ordering::Release);
                continue;
            }

            let n_transitions = all_actions.len();
            let obs_size = all_states.len() / n_transitions;

            // Normalize advantages with numerical safety guards
            let advantages = if config.normalize_advantages {
                debug_assert!(n_transitions > 0, "Cannot normalize empty advantages");
                let mean = all_advantages.iter().sum::<f32>() / n_transitions as f32;
                let var = all_advantages
                    .iter()
                    .map(|a| (a - mean).powi(2))
                    .sum::<f32>()
                    / n_transitions as f32;
                // Epsilon guard prevents division by zero when all advantages are equal
                let std = (var + 1e-8).sqrt();
                let normalized: Vec<_> = all_advantages
                    .iter()
                    .map(|a| (a - mean) / std)
                    .collect();

                // Debug check for numerical issues after normalization
                #[cfg(debug_assertions)]
                {
                    for (i, &adv) in normalized.iter().enumerate() {
                        debug_assert!(
                            adv.is_finite(),
                            "Non-finite advantage at index {}: {} (mean={}, std={})",
                            i, adv, mean, std
                        );
                    }
                }
                normalized
            } else {
                all_advantages
            };

            // Training epochs
            for _epoch in 0..config.n_epochs {
                // Create minibatches
                let indices: Vec<usize> = (0..n_transitions).collect();
                let minibatch_size = config.minibatch_size();

                for batch_start in (0..n_transitions).step_by(minibatch_size) {
                    let batch_end = (batch_start + minibatch_size).min(n_transitions);
                    let batch_indices = &indices[batch_start..batch_end];
                    let batch_size = batch_indices.len();

                    // Extract minibatch data
                    let batch_states: Vec<f32> = batch_indices
                        .iter()
                        .flat_map(|&i| &all_states[i * obs_size..(i + 1) * obs_size])
                        .copied()
                        .collect();
                    let batch_actions: Vec<A::Action> = batch_indices
                        .iter()
                        .map(|&i| all_actions[i].clone())
                        .collect();
                    let batch_old_log_probs: Vec<f32> = batch_indices
                        .iter()
                        .map(|&i| all_old_log_probs[i])
                        .collect();
                    let batch_advantages: Vec<f32> =
                        batch_indices.iter().map(|&i| advantages[i]).collect();
                    let batch_returns: Vec<f32> =
                        batch_indices.iter().map(|&i| all_returns[i]).collect();

                    // Forward pass
                    let obs_tensor =
                        Tensor::<B, 1>::from_floats(batch_states.as_slice(), &device)
                            .reshape([batch_size, obs_size]);
                    let hidden = model.initial_hidden(batch_size, &device);
                    let output = model.forward(obs_tensor, hidden);

                    // Compute log probs and entropy
                    let log_probs = output.policy.log_prob(&batch_actions, &device);
                    let entropy = output.policy.entropy();
                    let values = output.values_flat();

                    // PPO loss computation
                    let old_log_probs =
                        Tensor::<B, 1>::from_floats(batch_old_log_probs.as_slice(), &device);
                    let advantages_tensor =
                        Tensor::<B, 1>::from_floats(batch_advantages.as_slice(), &device);
                    let returns_tensor =
                        Tensor::<B, 1>::from_floats(batch_returns.as_slice(), &device);

                    // Ratio
                    let ratio = (log_probs.clone() - old_log_probs).exp();

                    // Clipped surrogate
                    let surr1 = ratio.clone() * advantages_tensor.clone();
                    let surr2 = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                        * advantages_tensor;
                    let policy_loss = -surr1.clone().min_pair(surr2).mean();

                    // Value loss
                    let value_loss = (values - returns_tensor).powf_scalar(2.0).mean();

                    // Entropy
                    let entropy_loss = -entropy.mean();

                    // Total loss
                    let total_loss = policy_loss
                        + value_loss.mul_scalar(config.vf_coef)
                        + entropy_loss.mul_scalar(config.entropy_coef);

                    // Backward pass
                    let grads = total_loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);

                    // Optimizer step
                    model = optimizer.step(config.learning_rate, model, grads);
                }
            }

            // Serialize and publish updated weights
            if let Ok(bytes) = recorder.record(model.clone().into_record(), ()) {
                bytes_slot.publish(bytes);
            }
            policy_version.fetch_add(1, Ordering::Release);

            // Signal actors that this epoch is consumed
            consumed_epoch.fetch_add(1, Ordering::Release);
        }

        model
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

use crate::algorithms::action_policy::{ContinuousPolicy, DiscretePolicy};
use crate::algorithms::temporal_policy::{FeedForward, Recurrent};

/// Distributed PPO with discrete actions and feed-forward policy.
pub type DistributedPPODiscrete<B> = DistributedPPORunner<DiscretePolicy, FeedForward, B>;

/// Distributed PPO with continuous actions and feed-forward policy.
pub type DistributedPPOContinuous<B> = DistributedPPORunner<ContinuousPolicy, FeedForward, B>;

/// Distributed PPO with discrete actions and recurrent policy.
pub type DistributedRecurrentPPODiscrete<B> = DistributedPPORunner<DiscretePolicy, Recurrent, B>;

/// Distributed PPO with continuous actions and recurrent policy.
pub type DistributedRecurrentPPOContinuous<B> =
    DistributedPPORunner<ContinuousPolicy, Recurrent, B>;
