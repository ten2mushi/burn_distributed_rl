//! PPO runner supporting both feed-forward and recurrent policies.
//!
//! This module provides `PPORunner<A, T, B, S>`, a high-level API for
//! multi-actor PPO training that works with both feed-forward and recurrent
//! policies through the strategy pattern.
//!
//! # Architecture
//!
//! ```text
//! PPORunner<A, T, B, S>
//!   │
//!   ├── A: ActionPolicy (DiscretePolicy, ContinuousPolicy)
//!   ├── T: TemporalPolicy (FeedForward, Recurrent)
//!   ├── B: AutodiffBackend (Autodiff<Wgpu>, etc.)
//!   └── S: PPOTrainingStrategy (auto-selected based on T)
//!
//! FeedForward  →  FeedForwardStrategy
//!                 • No hidden state
//!                 • Shuffle all transitions
//!                 • Standard minibatch training
//!
//! Recurrent    →  RecurrentStrategy
//!                 • Persist hidden across timesteps
//!                 • Reset hidden on terminal
//!                 • TBPTT sequence training
//! ```
//!
//! # Type Aliases
//!
//! ```ignore
//! // Feed-forward
//! pub type PPODiscrete<B> = PPORunner<DiscretePolicy, FeedForward, B>;
//! pub type PPOContinuous<B> = PPORunner<ContinuousPolicy, FeedForward, B>;
//!
//! // Recurrent
//! pub type RecurrentPPODiscrete<B> = PPORunner<DiscretePolicy, Recurrent, B>;
//! pub type RecurrentPPOContinuous<B> = PPORunner<ContinuousPolicy, Recurrent, B>;
//! ```
//!
//! # Usage
//!
//! ```ignore
//! // Feed-forward (default strategy auto-selected)
//! let runner = PPODiscrete::<Autodiff<Wgpu>>::new(config);
//! let trained = runner.run(model_factory, initial_model, env_factory, optimizer, callback);
//!
//! // Recurrent (default strategy auto-selected)
//! let runner = RecurrentPPODiscrete::<Autodiff<Wgpu>>::new(config, hidden_config);
//! let trained = runner.run(model_factory, initial_model, env_factory, optimizer, callback);
//! ```

use burn::module::AutodiffModule;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algorithms::action_policy::{ActionPolicy, ActionValue};
use crate::algorithms::actor_critic::{ActorCritic, ActorCriticInference};
use crate::algorithms::ppo::ppo_buffer::PPOBuffer;
use crate::algorithms::ppo::ppo_transition::{PPOTransitionTrait, PPOTransition};
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig, HiddenStateType, Recurrent, TemporalPolicy};
use crate::core::bytes_slot::{bytes_slot_with, SharedBytesSlot};
use crate::core::transition::Transition;

use super::ppo_config::{PPOConfig, PPOStats};
use super::learner::VectorizedEnv;
use super::ppo_strategies::{DefaultStrategy, PPOTrainingStrategy, TemporalPolicyStrategy};
use crate::algorithms::ppo::normalization::SharedRewardNormalizer;

/// Maximum number of recent episode rewards to keep for statistics.
const MAX_RECENT_REWARDS: usize = 1000;

// ============================================================================
// PPORunner
// ============================================================================

/// Unified PPO runner supporting both feed-forward and recurrent policies.
///
/// Coordinates N actor threads + 1 learner thread for PPO training.
/// Uses bytes-based weight transfer for WGPU compatibility.
///
/// # Type Parameters
///
/// - `A`: Action policy (`DiscretePolicy` or `ContinuousPolicy`)
/// - `T`: Temporal policy (`FeedForward` or `Recurrent`)
/// - `B`: Autodiff backend (e.g., `Autodiff<Wgpu>`)
/// - `S`: Training strategy (auto-selected via `DefaultStrategy<T>`)
pub struct PPORunner<A, T, B, S = DefaultStrategy<T>>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B> + TemporalPolicyStrategy,
    B: AutodiffBackend,
    S: PPOTrainingStrategy<B, A, T>,
{
    config: PPOConfig,
    hidden_config: HiddenConfig,
    _marker: PhantomData<(A, T, B, S)>,
}

impl<A, T, B, S> PPORunner<A, T, B, S>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B> + TemporalPolicyStrategy,
    B: AutodiffBackend,
    S: PPOTrainingStrategy<B, A, T>,
{
    /// Create a new unified PPO runner.
    ///
    /// # Arguments
    ///
    /// * `config` - PPO configuration
    /// * `hidden_config` - Hidden state configuration (use `HiddenConfig::none()` for feed-forward)
    pub fn new(config: PPOConfig, hidden_config: HiddenConfig) -> Self {
        Self {
            config,
            hidden_config,
            _marker: PhantomData,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Get hidden configuration.
    pub fn hidden_config(&self) -> &HiddenConfig {
        &self.hidden_config
    }

    /// Create a configured Adam optimizer with gradient clipping.
    pub fn create_optimizer<M>(&self) -> impl Optimizer<M, B>
    where
        M: AutodiffModule<B>,
    {
        let mut adam_config = AdamConfig::new().with_epsilon(1e-5);

        if let Some(max_norm) = self.config.max_grad_norm {
            adam_config =
                adam_config.with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm)));
        }

        adam_config.init()
    }

    /// Run distributed PPO training.
    ///
    /// # Arguments
    ///
    /// - `model_factory`: Creates a fresh model architecture on a given device
    /// - `initial_model`: Model with initial/pretrained weights
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
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        <T as TemporalPolicy<B::InnerBackend>>::Hidden: HiddenStateType<B::InnerBackend>,
        EF: Fn(usize, usize) -> E + Send + Sync + Clone + 'static,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action> + 'static,
        O: Optimizer<M, B> + 'static,
        F: Fn(&PPOStats),
        PPOTransition<S::TransitionData>: PPOTransitionTrait,
    {
        let device = B::Device::default();
        let config = self.config.clone();
        let hidden_config = self.hidden_config.clone();

        let is_recurrent = <T as TemporalPolicy<B>>::is_recurrent();
        let runner_type = if is_recurrent { "Recurrent PPO" } else { "PPO" };

        println!("=== {} Runner ===", runner_type);
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
        if is_recurrent {
            println!(
                "TBPTT length: {}, Hidden size: {}",
                config.tbptt_length, hidden_config.hidden_size
            );
        }

        // Validate configuration
        config.validate().expect("Invalid PPOConfig");

        // Validate model factory
        println!("Validating model factory compatibility...");
        {
            let test_model = model_factory(&device);
            let test_record = initial_model.clone().into_record();
            let _loaded = test_model.load_record(test_record);
            println!("Model factory validated successfully.");
        }

        // Serialize initial model
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let initial_bytes = recorder
            .record(initial_model.clone().into_record(), ())
            .expect("Failed to serialize initial model");
        println!("Initial model serialized: {} bytes", initial_bytes.len());

        // Create shared state
        let bytes_slot: SharedBytesSlot = bytes_slot_with(initial_bytes);
        let buffer = Arc::new(PPOBuffer::<PPOTransition<S::TransitionData>>::new(
            config.rollout_length,
            config.total_envs(),
        ));
        let shutdown = Arc::new(AtomicBool::new(false));
        let policy_version = Arc::new(AtomicU64::new(0));
        let total_env_steps = Arc::new(AtomicUsize::new(0));
        let total_episodes = Arc::new(AtomicUsize::new(0));
        let recent_rewards = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(MAX_RECENT_REWARDS)));
        let consumed_epoch = Arc::new(AtomicU64::new(0));

        // Create reward normalizer if enabled
        let reward_normalizer: Option<Arc<SharedRewardNormalizer>> = if config.reward_normalization {
            Some(Arc::new(SharedRewardNormalizer::new(config.total_envs())))
        } else {
            None
        };

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
            let reward_norm = reward_normalizer.clone();
            let cfg = config.clone();
            let hcfg = hidden_config.clone();

            let handle = std::thread::Builder::new()
                .name(format!("{}-Actor-{}", runner_type, actor_id))
                .spawn(move || {
                    Self::actor_thread::<M, MF, EF, E>(
                        actor_id,
                        &cfg,
                        &hcfg,
                        model_factory,
                        env_factory,
                        slot,
                        buffer,
                        shutdown_flag,
                        env_steps,
                        episodes,
                        rewards,
                        consumed,
                        reward_norm,
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
        let learner_env_steps = Arc::clone(&total_env_steps);
        let learner_reward_normalizer = reward_normalizer.clone();
        let cfg = config.clone();
        let hcfg = hidden_config.clone();
        let learner_model = initial_model;
        let learner_optimizer = optimizer;

        let learner_handle = std::thread::Builder::new()
            .name(format!("{}-Learner", runner_type))
            .spawn(move || {
                Self::learner_thread::<M, O>(
                    &cfg,
                    &hcfg,
                    learner_model,
                    learner_optimizer,
                    learner_slot,
                    learner_buffer,
                    learner_shutdown,
                    learner_version,
                    learner_consumed,
                    learner_env_steps,
                    learner_reward_normalizer,
                )
            })
            .expect("Failed to spawn learner thread");

        // Main monitoring loop
        let start_time = Instant::now();
        let mut last_log_time = Instant::now();
        let mut stats = PPOStats::default();

        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));

            let env_steps = total_env_steps.load(Ordering::Relaxed);
            let episodes = total_episodes.load(Ordering::Relaxed);
            let version = policy_version.load(Ordering::Relaxed);

            let avg_reward = {
                let rewards_guard = recent_rewards.lock();
                if !rewards_guard.is_empty() {
                    let recent: Vec<f32> = rewards_guard.iter().rev().take(100).copied().collect();
                    recent.iter().sum::<f32>() / recent.len() as f32
                } else {
                    0.0
                }
            };

            stats.env_steps = env_steps;
            stats.episodes = episodes;
            stats.avg_reward = avg_reward;
            stats.policy_version = version;
            stats.steps_per_second = env_steps as f32 / start_time.elapsed().as_secs_f32();
            stats.buffer_size = buffer.total_transitions();

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
        println!("Total steps: {}", total_env_steps.load(Ordering::Relaxed));
        println!("Total episodes: {}", total_episodes.load(Ordering::Relaxed));

        trained_model
    }

    /// Actor thread: collects experience using strategy pattern.
    fn actor_thread<M, MF, EF, E>(
        actor_id: usize,
        config: &PPOConfig,
        hidden_config: &HiddenConfig,
        model_factory: MF,
        env_factory: EF,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<PPOBuffer<PPOTransition<S::TransitionData>>>,
        shutdown: Arc<AtomicBool>,
        env_steps: Arc<AtomicUsize>,
        episodes: Arc<AtomicUsize>,
        rewards: Arc<Mutex<VecDeque<f32>>>,
        consumed_epoch: Arc<AtomicU64>,
        reward_normalizer: Option<Arc<SharedRewardNormalizer>>,
    ) where
        MF: Fn(&B::Device) -> M,
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, T>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        T: TemporalPolicy<B::InnerBackend>,
        <T as TemporalPolicy<B::InnerBackend>>::Hidden: HiddenStateType<B::InnerBackend>,
        EF: Fn(usize, usize) -> E,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action>,
        PPOTransition<S::TransitionData>: PPOTransitionTrait,
    {
        let device = B::Device::default();
        let inner_device = <B::InnerBackend as Backend>::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Create model using factory
        let mut model = model_factory(&device);

        // Load initial weights
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

        // Get inference model (no autodiff)
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
        let mut last_version = bytes_slot.version();
        let mut local_epoch = 0u64;
        let mut steps_this_epoch = 0usize;

        // Pre-allocated buffers
        let mut current_obs: Vec<Vec<f32>> = (0..n_envs)
            .map(|_| vec![0.0f32; obs_size])
            .collect();

        // Hidden state management (initialized per strategy's needs)
        let mut hidden = <T as TemporalPolicy<B::InnerBackend>>::Hidden::initial(n_envs, &inner_device, hidden_config);

        // Sequence tracking for recurrent
        let mut sequence_ids: Vec<u64> = (0..n_envs)
            .map(|i| (actor_id * n_envs + i) as u64 * 1_000_000)
            .collect();
        let mut steps_in_sequence: Vec<u32> = vec![0; n_envs];
        let mut is_sequence_start: Vec<bool> = vec![true; n_envs];

        while !shutdown.load(Ordering::Relaxed) {
            // Synchronization: wait for learner to consume previous epoch
            if steps_this_epoch >= config.rollout_length {
                while consumed_epoch.load(Ordering::Acquire) <= local_epoch {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }

                // Get updated weights
                let current_version = bytes_slot.version();
                if current_version > last_version {
                    if let Some(bytes) = bytes_slot.get() {
                        if let Ok(record) = recorder.load(bytes, &device) {
                            model = model.load_record(record);
                            inference_model = model.valid();
                            last_version = current_version;
                        }
                    }
                }

                local_epoch += 1;
                steps_this_epoch = 0;
            }

            // Save current obs
            for i in 0..n_envs {
                let start = i * obs_size;
                current_obs[i].copy_from_slice(&obs_buffer[start..start + obs_size]);
            }

            // CRITICAL: Extract INPUT hidden states BEFORE forward pass
            // The hidden state used to compute (log_prob, value) must be stored in the transition.
            // Type invariant: transition[t].hidden = h_{t-1} (input to forward that produced this transition)
            // NOT h_t (output), which would create a temporal type mismatch during training.
            let input_hidden_states: Vec<Vec<f32>> = (0..n_envs)
                .map(|i| hidden.get_env_vec(i))
                .collect();

            // Forward pass
            let obs_tensor =
                Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                    .reshape([n_envs, obs_size]);

            let output = inference_model.forward(obs_tensor, hidden.clone());

            // Sample actions
            let (actions, log_probs) = output.sample_actions(&inner_device);
            let values: Vec<f32> = output
                .values_flat()
                .into_data()
                .as_slice()
                .unwrap()
                .to_vec();

            // Update hidden state (just use output directly)
            hidden = output.hidden;

            // Step environment
            let step_result = env.step(&actions);
            env.write_observations(&mut obs_buffer);

            // Compute bootstrap values at rollout boundary
            let mut bootstrap_values: Vec<Option<f32>> = vec![None; n_envs];
            if steps_this_epoch == config.rollout_length - 1 {
                let next_obs_tensor =
                    Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                        .reshape([n_envs, obs_size]);

                // Use current hidden for bootstrap (important for recurrent)
                let bootstrap_output = inference_model.forward(next_obs_tensor, hidden.clone());
                let bootstrap_vals: Vec<f32> = bootstrap_output
                    .values_flat()
                    .into_data()
                    .as_slice()
                    .unwrap()
                    .to_vec();

                for i in 0..n_envs {
                    if !step_result.dones[i] {
                        bootstrap_values[i] = Some(bootstrap_vals[i]);
                    }
                }
            }

            // Create transitions using strategy
            let mut transitions = Vec::with_capacity(n_envs);
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

                // Use INPUT hidden state (extracted BEFORE forward pass)
                // This ensures temporal causality: transition stores h_{t-1}, not h_t
                let hidden_data = input_hidden_states[i].clone();

                // Use strategy to create transition
                let transition = S::create_transition(
                    base,
                    log_probs[i],
                    values[i],
                    bootstrap_values[i],
                    hidden_data,
                    sequence_ids[i],
                    steps_in_sequence[i],
                    is_sequence_start[i],
                );

                transitions.push(transition);

                // Update sequence tracking
                steps_in_sequence[i] += 1;
                is_sequence_start[i] = false;
            }

            // Push to buffer
            buffer.push_batch(transitions, global_env_offset);

            // Handle terminal states - CRITICAL: Use terminals for hidden reset, NOT dones!
            // This distinction is essential for recurrent policies:
            // - Terminal: Episode truly ended (goal reached, failure, etc.) - reset hidden
            // - Truncated: Episode hit time limit - preserve hidden (memory continues)
            let terminal_indices: Vec<usize> = step_result
                .terminals
                .iter()
                .enumerate()
                .filter_map(|(i, &terminal)| if terminal { Some(i) } else { None })
                .collect();

            // Reset hidden state and sequence tracking ONLY for true terminals
            for &i in &terminal_indices {
                hidden.reset(i, &inner_device);
                sequence_ids[i] += 1;
                steps_in_sequence[i] = 0;
                is_sequence_start[i] = true;
            }

            // Environment reset for ALL done (terminal OR truncated)
            let done_indices: Vec<usize> = step_result
                .dones
                .iter()
                .enumerate()
                .filter_map(|(i, &done)| if done { Some(i) } else { None })
                .collect();

            if !done_indices.is_empty() {
                env.reset_envs(&done_indices);
                env.write_observations(&mut obs_buffer);
            }

            // Track episode stats and update reward normalizer
            for (i, (&reward, &done)) in step_result
                .rewards
                .iter()
                .zip(step_result.dones.iter())
                .enumerate()
            {
                // Update reward normalizer for return std tracking
                if let Some(ref normalizer) = reward_normalizer {
                    normalizer.update(global_env_offset + i, reward, done);
                }

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
            steps_this_epoch += 1;
        }
    }

    /// Learner thread: trains using strategy pattern.
    fn learner_thread<M, O>(
        config: &PPOConfig,
        hidden_config: &HiddenConfig,
        mut model: M,
        mut optimizer: O,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<PPOBuffer<PPOTransition<S::TransitionData>>>,
        shutdown: Arc<AtomicBool>,
        policy_version: Arc<AtomicU64>,
        consumed_epoch: Arc<AtomicU64>,
        total_env_steps: Arc<AtomicUsize>,
        reward_normalizer: Option<Arc<SharedRewardNormalizer>>,
    ) -> M
    where
        M: ActorCritic<B, A, T>,
        M::Record: Send + 'static,
        O: Optimizer<M, B>,
        PPOTransition<S::TransitionData>: PPOTransitionTrait,
    {
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let initial_lr = config.learning_rate;

        while !shutdown.load(Ordering::Relaxed) {
            if !buffer.is_ready() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Consume rollouts
            let per_env_rollouts = buffer.consume();

            // Prepare training data using strategy
            let training_data = S::prepare_training_data(
                per_env_rollouts,
                config,
                hidden_config,
                reward_normalizer.as_deref(),
            );

            if training_data.n_transitions == 0 {
                consumed_epoch.fetch_add(1, Ordering::Release);
                continue;
            }

            // Training epochs
            let mut kl_early_stop = false;
            for _epoch in 0..config.n_epochs {
                if kl_early_stop {
                    break;
                }

                let current_env_steps = total_env_steps.load(Ordering::Relaxed);

                // Train using strategy
                let (updated_model, early_stop) = S::train_epoch(
                    model,
                    &mut optimizer,
                    &training_data,
                    config,
                    hidden_config,
                    &device,
                    current_env_steps,
                    initial_lr,
                );

                model = updated_model;
                kl_early_stop = early_stop;
            }

            // Publish updated weights
            match recorder.record(model.clone().into_record(), ()) {
                Ok(bytes) => {
                    bytes_slot.publish(bytes);
                }
                Err(e) => {
                    log::error!(
                        "Failed to serialize model weights (version {}): {:?}",
                        policy_version.load(Ordering::Relaxed),
                        e
                    );
                }
            }
            policy_version.fetch_add(1, Ordering::Release);
            consumed_epoch.fetch_add(1, Ordering::Release);
        }

        model
    }
}

// ============================================================================
// Convenience Constructors for Feed-Forward
// ============================================================================

impl<A, B, S> PPORunner<A, FeedForward, B, S>
where
    A: ActionPolicy<B>,
    B: AutodiffBackend,
    S: PPOTrainingStrategy<B, A, FeedForward>,
{
    /// Create a feed-forward PPO runner with default hidden config.
    pub fn feed_forward(config: PPOConfig) -> Self {
        Self::new(config, HiddenConfig::none())
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

use crate::algorithms::action_policy::{ContinuousPolicy, DiscretePolicy};
use super::ppo_strategies::{FeedForwardStrategy, RecurrentStrategy};

/// Feed-forward PPO runner with discrete actions.
pub type PPODiscrete<B> = PPORunner<DiscretePolicy, FeedForward, B, FeedForwardStrategy>;

/// Feed-forward PPO runner with continuous actions.
pub type PPOContinuous<B> = PPORunner<ContinuousPolicy, FeedForward, B, FeedForwardStrategy>;

/// Recurrent PPO runner with discrete actions.
pub type RecurrentPPODiscrete<B> = PPORunner<DiscretePolicy, Recurrent, B, RecurrentStrategy>;

/// Recurrent PPO runner with continuous actions.
pub type RecurrentPPOContinuous<B> = PPORunner<ContinuousPolicy, Recurrent, B, RecurrentStrategy>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_aliases_compile() {
        // These are compile-time checks - if they compile, the test passes
        fn _check_ppo_discrete<B: AutodiffBackend>(_: PPODiscrete<B>) {}
        fn _check_ppo_continuous<B: AutodiffBackend>(_: PPOContinuous<B>) {}
        fn _check_rec_ppo_discrete<B: AutodiffBackend>(_: RecurrentPPODiscrete<B>) {}
        fn _check_rec_ppo_continuous<B: AutodiffBackend>(_: RecurrentPPOContinuous<B>) {}
    }

    #[test]
    fn test_default_config() {
        use burn::backend::{Autodiff, NdArray};
        type B = Autodiff<NdArray<f32>>;

        let config = PPOConfig::default();

        // Feed-forward runner
        let _ff_runner: PPODiscrete<B> = PPORunner::feed_forward(config.clone());

        // Recurrent runner
        let hidden_config = HiddenConfig::lstm(64);
        let _rec_runner: RecurrentPPODiscrete<B> = PPORunner::new(config, hidden_config);
    }
}
