//! Distributed recurrent PPO runner with proper hidden state management.
//!
//! This runner correctly handles recurrent policies by:
//! 1. Persisting hidden states across timesteps in actors
//! 2. Resetting hidden states on episode termination
//! 3. Using sequence-based training with TBPTT in the learner
//!
//! # Key Differences from Feed-Forward Runner
//!
//! - Actor maintains hidden state across steps (not reset each step)
//! - Hidden states are reset when environments terminate
//! - Transitions include hidden state for sequence reconstruction
//! - Learner processes sequences, not shuffled individual transitions
//!
//! # Important Notes on Recurrent PPO
//!
//! Recurrent PPO has a fundamental tension: the hidden states used during
//! rollout collection differ from those during training (after model updates).
//! To mitigate this:
//! - Use fewer epochs (1-2) for recurrent policies
//! - Detach hidden states at sequence boundaries to prevent gradient explosion
//! - Consider shorter TBPTT lengths for stability

use burn::optim::{GradientsParams, Optimizer};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, ContinuousPolicy, DiscretePolicy, PolicyOutput};
use crate::algorithms::actor_critic::{ActorCritic, ActorCriticInference};
use crate::algorithms::gae::compute_gae;
use crate::algorithms::ppo::RecurrentPPOBuffer;
use crate::algorithms::temporal_policy::{HiddenStateType, Recurrent, RecurrentHidden, TemporalPolicy};
use crate::core::bytes_slot::{bytes_slot_with, SharedBytesSlot};
use crate::core::transition::{PPOTransition, RecurrentPPOTransition, Transition, Action};

use super::distributed_ppo_config::{DistributedPPOConfig, DistributedPPOStats};
use super::learner::VectorizedEnv;

/// Maximum number of recent episode rewards to keep for statistics.
/// Prevents unbounded memory growth during long training runs.
const MAX_RECENT_REWARDS: usize = 1000;

// ============================================================================
// DistributedRecurrentPPORunner
// ============================================================================

/// Distributed recurrent PPO runner with proper LSTM/GRU support.
///
/// This runner is specifically designed for recurrent policies and correctly:
/// - Maintains hidden state continuity across timesteps
/// - Resets hidden states on episode boundaries
/// - Implements sequence-based training (TBPTT)
pub struct DistributedRecurrentPPORunner<A, B>
where
    A: ActionPolicy<B>,
    B: AutodiffBackend,
{
    config: DistributedPPOConfig,
    _marker: PhantomData<(A, B)>,
}

impl<A, B> DistributedRecurrentPPORunner<A, B>
where
    A: ActionPolicy<B>,
    B: AutodiffBackend,
{
    /// Create a new distributed recurrent PPO runner.
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

    /// Run distributed recurrent PPO training.
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
        M: ActorCritic<B, A, Recurrent> + Clone + 'static,
        M::Record: Send + 'static,
        // Inner backend bounds for actor inference (no autodiff graphs)
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, Recurrent>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        Recurrent: TemporalPolicy<B::InnerBackend, Hidden = RecurrentHidden<B::InnerBackend>>,
        EF: Fn(usize, usize) -> E + Send + Sync + Clone + 'static,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action> + 'static,
        O: Optimizer<M, B> + 'static,
        F: Fn(&DistributedPPOStats),
    {
        let device = B::Device::default();
        let config = self.config.clone();

        println!("=== Distributed Recurrent PPO Runner ===");
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

        // Get hidden config from model's temporal policy
        let temporal = initial_model.temporal_policy();
        let hidden_config = <Recurrent as TemporalPolicy<B>>::hidden_config(&temporal);
        println!(
            "Recurrent: hidden_size={}, has_cell={}",
            hidden_config.hidden_size, hidden_config.has_cell
        );

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
        let buffer = Arc::new(RecurrentPPOBuffer::new(
            config.rollout_length,
            config.total_envs(),
        ));
        let shutdown = Arc::new(AtomicBool::new(false));
        let policy_version = Arc::new(AtomicU64::new(0));
        let total_env_steps = Arc::new(AtomicUsize::new(0));
        let total_episodes = Arc::new(AtomicUsize::new(0));
        let recent_rewards = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(MAX_RECENT_REWARDS)));
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
            let h_config = hidden_config.clone();

            let handle = std::thread::Builder::new()
                .name(format!("RecPPO-Actor-{}", actor_id))
                .spawn(move || {
                    Self::actor_thread::<M, MF, EF, E>(
                        actor_id,
                        &cfg,
                        &h_config,
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
        let h_config = hidden_config.clone();

        let learner_handle = std::thread::Builder::new()
            .name("RecPPO-Learner".to_string())
            .spawn(move || {
                Self::learner_thread::<M, O>(
                    &cfg,
                    &h_config,
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

    /// Actor thread with proper hidden state management.
    /// 1. Hidden state persists across steps (not reset each step)
    /// 2. Hidden state reset on episode termination
    /// 3. Hidden state stored with each transition
    ///
    /// Uses inner backend (non-autodiff) for inference to avoid computation graph
    /// accumulation. This is critical for maintaining consistent SPS throughout training.
    fn actor_thread<M, MF, EF, E>(
        actor_id: usize,
        config: &DistributedPPOConfig,
        hidden_config: &crate::algorithms::temporal_policy::HiddenConfig,
        model_factory: MF,
        env_factory: EF,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<RecurrentPPOBuffer>,
        shutdown: Arc<AtomicBool>,
        env_steps: Arc<AtomicUsize>,
        episodes: Arc<AtomicUsize>,
        rewards: Arc<Mutex<VecDeque<f32>>>,
        consumed_epoch: Arc<AtomicU64>,
    ) where
        MF: Fn(&B::Device) -> M,
        M: ActorCritic<B, A, Recurrent>,
        M::Record: Send + 'static,
        // Inner backend bounds for graph-free inference
        M::InnerModule: ActorCriticInference<B::InnerBackend, A, Recurrent>,
        A: ActionPolicy<B::InnerBackend, Action = <A as ActionPolicy<B>>::Action>,
        Recurrent: TemporalPolicy<B::InnerBackend, Hidden = RecurrentHidden<B::InnerBackend>>,
        EF: Fn(usize, usize) -> E,
        E: VectorizedEnv<<A as ActionPolicy<B>>::Action>,
    {
        // Create devices: autodiff for weight loading, inner for inference
        let device = B::Device::default();
        let inner_device = <B::InnerBackend as Backend>::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        // Create model using factory (on autodiff backend for weight loading)
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

        // Get inference model on inner backend (no autodiff - no computation graphs!)
        // This is the key optimization: forward passes won't accumulate orphaned graphs.
        let mut inference_model: M::InnerModule = model.valid();

        // Create environment
        let mut env = env_factory(actor_id, config.n_envs_per_actor);
        let obs_size = env.obs_size();
        let n_envs = config.n_envs_per_actor;
        let global_env_offset = actor_id * n_envs;

        // Initialize observations
        let mut obs_buffer = vec![0.0f32; n_envs * obs_size];
        env.write_observations(&mut obs_buffer);

        // ============================================================
        // FIX #1: Initialize hidden state ONCE, persist across steps
        // Hidden state on INNER backend for graph-free inference
        // ============================================================
        let mut hidden = RecurrentHidden::<B::InnerBackend>::new(
            n_envs,
            hidden_config.hidden_size,
            hidden_config.has_cell,
            &inner_device,
        );

        // Track episode state
        let mut episode_rewards = vec![0.0f32; n_envs];
        let mut sequence_ids = vec![0u64; n_envs]; // Unique ID per episode
        let mut steps_in_sequence = vec![0usize; n_envs];
        let mut is_sequence_start = vec![true; n_envs]; // First step after reset
        let mut next_sequence_id = (actor_id as u64) << 32; // Ensure unique across actors

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
                            // Refresh inference model with updated weights
                            inference_model = model.valid();
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

            // Save hidden state BEFORE forward pass (for storing in transition)
            let hidden_vecs: Vec<Vec<f32>> = (0..n_envs)
                .map(|i| hidden.get(i).to_vec())
                .collect();

            // Forward pass on INNER backend - NO computation graph accumulation!
            let obs_tensor =
                Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                    .reshape([n_envs, obs_size]);

            let output = inference_model.forward(obs_tensor, hidden.clone());

            // Sample actions and get values (also on inner backend - no graph overhead)
            let (actions, log_probs) = output.sample_actions(&inner_device);
            let values: Vec<f32> = output
                .values_flat()
                .into_data()
                .as_slice()
                .unwrap()
                .to_vec();

            // Update hidden state from output
            hidden = output.hidden;

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

            // Calculate bootstrap values if at rollout boundary
            // Uses inner backend for graph-free inference
            let mut bootstrap_values = vec![None; n_envs];
            if steps_this_epoch == config.rollout_length - 1 {
                let next_obs_tensor =
                    Tensor::<B::InnerBackend, 1>::from_floats(obs_buffer.as_slice(), &inner_device)
                        .reshape([n_envs, obs_size]);
                // hidden is already updated to the state corresponding to next_obs
                let output = inference_model.forward(next_obs_tensor, hidden.clone());
                let bootstrap_vals: Vec<f32> = output
                    .values_flat()
                    .into_data()
                    .as_slice()
                    .unwrap()
                    .to_vec();

                for i in 0..n_envs {
                    // Only bootstrap if not done (terminated or truncated by env)
                    if !step_result.dones[i] {
                        bootstrap_values[i] = Some(bootstrap_vals[i]);
                    }
                }
            }

            // Create recurrent transitions with hidden state
            let mut transitions = Vec::with_capacity(n_envs);
            for i in 0..n_envs {
                let action_floats = actions[i].as_floats();
                let action = if action_floats.len() == 1 {
                    Action::Discrete(action_floats[0] as u32)
                } else {
                    Action::Continuous(action_floats)
                };

                let ppo_transition = PPOTransition {
                    base: Transition {
                        state: current_obs[i].clone(),
                        action,
                        reward: step_result.rewards[i],
                        next_state: next_obs[i].clone(),
                        terminal: step_result.terminals[i],
                        truncated: step_result.dones[i] && !step_result.terminals[i],
                    },
                    log_prob: log_probs[i],
                    value: values[i],
                };

                let mut transition = RecurrentPPOTransition::from_ppo(
                    ppo_transition,
                    hidden_vecs[i].clone(),
                    sequence_ids[i],
                    steps_in_sequence[i],
                    is_sequence_start[i],
                );
                transition.bootstrap_value = bootstrap_values[i];
                transitions.push(transition);

                // Update sequence tracking
                steps_in_sequence[i] += 1;
                is_sequence_start[i] = false;
            }

            // Push to buffer
            buffer.push_batch(transitions, global_env_offset);

            // Track episode stats and handle terminations
            // CRITICAL: Separate done_indices (env reset) from terminal_indices (hidden reset)
            // Truncated episodes should NOT reset hidden state - the "memory" is still relevant
            let mut done_indices = Vec::new();
            let mut terminal_indices = Vec::new();
            for (i, (&reward, (&done, &terminal))) in step_result
                .rewards
                .iter()
                .zip(step_result.dones.iter().zip(step_result.terminals.iter()))
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
                    done_indices.push(i);

                    // Reset sequence tracking for this env
                    sequence_ids[i] = next_sequence_id;
                    next_sequence_id += 1;
                    steps_in_sequence[i] = 0;
                    is_sequence_start[i] = true;

                    // Track TRUE terminals separately for hidden state reset
                    if terminal {
                        terminal_indices.push(i);
                    }
                }
            }

            // Sanity check: terminals must be subset of dones
            debug_assert!(
                terminal_indices.iter().all(|idx| done_indices.contains(idx)),
                "terminal_indices must be subset of done_indices"
            );

            // Reset environments for ALL dones (both terminal and truncated)
            if !done_indices.is_empty() {
                env.reset_envs(&done_indices);
                env.write_observations(&mut obs_buffer);
            }

            // Reset hidden states ONLY for TRUE terminals
            // Truncated episodes keep their hidden state - the agent's learned memory
            // persists across the artificial episode boundary
            for &idx in &terminal_indices {
                hidden.reset(idx, &inner_device);
            }

            env_steps.fetch_add(n_envs, Ordering::Relaxed);
            steps_this_epoch += 1;
        }
    }

    /// Learner thread with sequence-based training (TBPTT).
    ///
    /// Key fix: Process sequences contiguously, don't shuffle within sequences.
    fn learner_thread<M, O>(
        config: &DistributedPPOConfig,
        hidden_config: &crate::algorithms::temporal_policy::HiddenConfig,
        mut model: M,
        mut optimizer: O,
        bytes_slot: SharedBytesSlot,
        buffer: Arc<RecurrentPPOBuffer>,
        shutdown: Arc<AtomicBool>,
        policy_version: Arc<AtomicU64>,
        consumed_epoch: Arc<AtomicU64>,
    ) -> M
    where
        M: ActorCritic<B, A, Recurrent>,
        M::Record: Send + 'static,
        O: Optimizer<M, B>,
    {
        let device = B::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let mut update_count = 0u64;

        while !shutdown.load(Ordering::Relaxed) {
            if !buffer.is_ready() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            let per_env_rollouts = buffer.consume();

            // Track loss for diagnostics
            let mut epoch_policy_loss = 0.0f32;
            let mut epoch_value_loss = 0.0f32;
            let mut epoch_entropy = 0.0f32;
            let mut epoch_updates = 0usize;

            // ============================================================
            // FIX #3: Sequence-based training with TBPTT
            // ============================================================

            // Process each environment's rollout as a sequence
            // We'll chunk long rollouts into smaller sequences for TBPTT
            let tbptt_length = config.rollout_length.min(32); // Truncation length for BPTT

            // Collect sequences for training
            let mut sequences: Vec<SequenceData> = Vec::new();

            for rollout in &per_env_rollouts {
                if rollout.is_empty() {
                    continue;
                }

                // Split rollout into TBPTT chunks, respecting episode boundaries
                let mut chunk_start = 0;
                while chunk_start < rollout.len() {
                    let chunk_end = (chunk_start + tbptt_length).min(rollout.len());

                    // Check if there's an episode boundary in this chunk
                    let mut actual_end = chunk_end;
                    for i in chunk_start..chunk_end {
                        if rollout[i].done() && i + 1 < chunk_end {
                            actual_end = i + 1;
                            break;
                        }
                    }

                    let chunk = &rollout[chunk_start..actual_end];
                    if !chunk.is_empty() {
                        // Determine bootstrap value V(s_T) for this chunk:
                        // - If last transition is terminal: bootstrap = 0
                        // - If there's a next transition: bootstrap = V(s_T) from next transition
                        // - If at rollout boundary: use precomputed bootstrap_value from transition
                        let last_transition = chunk.last().unwrap();
                        let next_value = if last_transition.terminal() {
                            // TRUE terminal state (episode ended) - bootstrap is 0
                            None
                        } else if actual_end < rollout.len() {
                            // Non-terminal and there's a next transition - use its value
                            Some(rollout[actual_end].value())
                        } else {
                            // At rollout boundary - use precomputed bootstrap from actor
                            // This is the proper V(s_T) computed with next_obs and current hidden
                            last_transition.bootstrap_value
                        };

                        sequences.push(SequenceData::from_transitions(chunk, next_value));
                    }

                    chunk_start = actual_end;
                }
            }

            if sequences.is_empty() {
                consumed_epoch.fetch_add(1, Ordering::Release);
                continue;
            }

            // Compute GAE for each sequence
            for seq in &mut sequences {
                // Use the stored next_value (V(s_T)) for bootstrap, or 0 if terminal/truncated
                let bootstrap = seq.next_value.unwrap_or(0.0);

                let (advantages, returns) = compute_gae(
                    &seq.rewards,
                    &seq.values,
                    &seq.dones,
                    bootstrap,
                    config.gamma,
                    config.gae_lambda,
                );
                seq.advantages = advantages;
                seq.returns = returns;
            }

            // Training epochs
            for _epoch in 0..config.n_epochs {
                // Shuffle sequences (not individual transitions!)
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                sequences.shuffle(&mut rng);

                // Process sequences in minibatches
                let n_sequences = sequences.len();
                let sequences_per_batch = (n_sequences / config.n_minibatches).max(1);
                for batch_start in (0..n_sequences).step_by(sequences_per_batch) {
                    let batch_end = (batch_start + sequences_per_batch).min(n_sequences);
                    let batch_sequences = &sequences[batch_start..batch_end];

                    let n_seqs = batch_sequences.len();
                    if n_seqs == 0 {
                        continue;
                    }

                    // Batched sequence processing
                    // Process all sequences in parallel by:
                    // 1. Padding to max length in batch
                    // 2. Batched forward through all timesteps
                    // 3. Masked loss computation
                    // 4. Single backward pass

                    let obs_size = batch_sequences[0].states[0].len();
                    let max_len = batch_sequences.iter().map(|s| s.states.len()).max().unwrap_or(0);

                    if max_len == 0 {
                        continue;
                    }

                    // Build padded batch tensors and mask
                    let mut obs_data = vec![0.0f32; n_seqs * max_len * obs_size];
                    let mut old_log_probs_data = vec![0.0f32; n_seqs * max_len];
                    let mut advantages_data = vec![0.0f32; n_seqs * max_len];
                    let mut returns_data = vec![0.0f32; n_seqs * max_len];
                    let mut mask_data = vec![0.0f32; n_seqs * max_len]; // 1.0 for valid, 0.0 for pad
                    let mut actions_by_timestep: Vec<Vec<A::Action>> = vec![Vec::with_capacity(n_seqs); max_len];
                    let mut dones_by_timestep: Vec<Vec<bool>> = vec![vec![false; n_seqs]; max_len];

                    // Initialize batched hidden state from sequences
                    let expected_hidden_size = if hidden_config.has_cell {
                        hidden_config.hidden_size * 2
                    } else {
                        hidden_config.hidden_size
                    };

                    let mut hidden = RecurrentHidden::<B>::new(
                        n_seqs,
                        hidden_config.hidden_size,
                        hidden_config.has_cell,
                        &device,
                    );

                    // Fill batch data (first pass - collect raw advantages)
                    for (seq_idx, seq) in batch_sequences.iter().enumerate() {
                        let seq_len = seq.states.len();

                        // Initialize hidden for this sequence
                        if seq.initial_hidden.len() == expected_hidden_size {
                            let h = RecurrentHidden::<B>::from_vec(
                                &seq.initial_hidden,
                                1,
                                &device,
                                hidden_config,
                            );
                            hidden.set(seq_idx, h.states[0].clone());
                        }

                        for t in 0..seq_len {
                            let batch_idx = seq_idx * max_len + t;
                            let obs_start = batch_idx * obs_size;
                            obs_data[obs_start..obs_start + obs_size].copy_from_slice(&seq.states[t]);
                            old_log_probs_data[batch_idx] = seq.old_log_probs[t];
                            advantages_data[batch_idx] = seq.advantages[t]; // Raw advantages
                            returns_data[batch_idx] = seq.returns[t];
                            mask_data[batch_idx] = 1.0;
                            actions_by_timestep[t].push(seq.get_action(t));
                            dones_by_timestep[t][seq_idx] = seq.dones[t];
                        }
                        // Pad actions for shorter sequences
                        for t in seq_len..max_len {
                            actions_by_timestep[t].push(seq.get_action(0)); // Dummy action for padding
                        }
                    }

                    // Normalize advantages across entire minibatch (not per-sequence)
                    // with defensive guards for numerical stability
                    if config.normalize_advantages {
                        // Compute mean and std only over valid (non-padded) entries
                        let valid_advs: Vec<f32> = advantages_data.iter()
                            .zip(mask_data.iter())
                            .filter(|(_, &m)| m > 0.5)
                            .map(|(&a, _)| a)
                            .collect();

                        if valid_advs.len() > 1 {
                            let mean = valid_advs.iter().sum::<f32>() / valid_advs.len() as f32;
                            let var = valid_advs.iter().map(|a| (a - mean).powi(2)).sum::<f32>()
                                / valid_advs.len() as f32;
                            // Epsilon guard prevents division by zero
                            let std = (var + 1e-8).sqrt();

                            // Normalize in place
                            for (adv, &m) in advantages_data.iter_mut().zip(mask_data.iter()) {
                                if m > 0.5 {
                                    *adv = (*adv - mean) / std;
                                }
                            }

                            // Debug check for numerical issues
                            #[cfg(debug_assertions)]
                            {
                                for (i, (&adv, &m)) in advantages_data.iter().zip(mask_data.iter()).enumerate() {
                                    if m > 0.5 {
                                        debug_assert!(
                                            adv.is_finite(),
                                            "Non-finite normalized advantage at index {}: {} (mean={}, std={})",
                                            i, adv, mean, std
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Process all sequences in parallel, timestep by timestep
                    let mut all_log_probs: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);
                    let mut all_entropies: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);
                    let mut all_values: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);

                    for t in 0..max_len {
                        // Gather obs for this timestep across all sequences: [n_seqs, obs_size]
                        let _obs_start = t * obs_size;
                        let obs_slice: Vec<f32> = (0..n_seqs)
                            .flat_map(|s| {
                                let idx = s * max_len * obs_size + t * obs_size;
                                obs_data[idx..idx + obs_size].iter().copied()
                            })
                            .collect();
                        let obs_t = Tensor::<B, 1>::from_floats(obs_slice.as_slice(), &device)
                            .reshape([n_seqs, obs_size]);

                        // Batched forward pass
                        let output = model.forward(obs_t, hidden);

                        // Get batched log_probs and values
                        let log_prob = output.policy.log_prob(&actions_by_timestep[t], &device);
                        let entropy = output.policy.entropy();
                        let value = output.values_flat();

                        all_log_probs.push(log_prob);
                        all_entropies.push(entropy);
                        all_values.push(value);

                        // Update hidden (batched)
                        hidden = output.hidden;

                        // Reset hidden for sequences that terminated at this timestep
                        for (seq_idx, &done) in dones_by_timestep[t].iter().enumerate() {
                            if done {
                                hidden.reset(seq_idx, &device);
                            }
                        }
                    }

                    // Stack: [max_len, n_seqs] -> flatten to [max_len * n_seqs]
                    // Then transpose conceptually to [n_seqs, max_len] for proper indexing
                    let log_probs: Tensor<B, 2> = Tensor::stack(all_log_probs, 0); // [max_len, n_seqs]
                    let entropies: Tensor<B, 2> = Tensor::stack(all_entropies, 0); // [max_len, n_seqs]
                    let values: Tensor<B, 2> = Tensor::stack(all_values, 0); // [max_len, n_seqs]

                    // Transpose to [n_seqs, max_len] then flatten
                    let log_probs: Tensor<B, 1> = log_probs.swap_dims(0, 1).flatten(0, 1);
                    let entropies: Tensor<B, 1> = entropies.swap_dims(0, 1).flatten(0, 1);
                    let values: Tensor<B, 1> = values.swap_dims(0, 1).flatten(0, 1);

                    // Create target tensors
                    let old_log_probs = Tensor::<B, 1>::from_floats(old_log_probs_data.as_slice(), &device);
                    let advantages = Tensor::<B, 1>::from_floats(advantages_data.as_slice(), &device);
                    let returns = Tensor::<B, 1>::from_floats(returns_data.as_slice(), &device);
                    let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device);

                    // Compute masked losses
                    let ratio = (log_probs.clone() - old_log_probs).exp();
                    let surr1 = ratio.clone() * advantages.clone();
                    let surr2 = ratio.clone().clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages;
                    let policy_loss_per_step = -surr1.min_pair(surr2);
                    let value_loss_per_step = (values - returns).powf_scalar(2.0);
                    let entropy_per_step = entropies;

                    // Apply mask and mean
                    let n_valid = mask.clone().sum();
                    let policy_loss = (policy_loss_per_step * mask.clone()).sum() / n_valid.clone();
                    let value_loss = (value_loss_per_step * mask.clone()).sum() / n_valid.clone();
                    let entropy_loss = -(entropy_per_step * mask).sum() / n_valid.clone();

                    // Combined loss (keep tensors for backward pass)
                    let total_loss = policy_loss.clone()
                        + value_loss.clone().mul_scalar(config.vf_coef)
                        + entropy_loss.clone().mul_scalar(config.entropy_coef);

                    // Diagnostic: compute ratio statistics to check if policy is updating
                    let ratio_data = ratio.clone().into_data();
                    let ratio_slice: &[f32] = ratio_data.as_slice().unwrap();
                    let mask_slice: &[f32] = mask_data.as_slice();
                    let valid_ratios: Vec<f32> = ratio_slice.iter()
                        .zip(mask_slice.iter())
                        .filter(|(_, &m)| m > 0.5)
                        .map(|(&r, _)| r)
                        .collect();
                    let ratio_mean = if !valid_ratios.is_empty() {
                        valid_ratios.iter().sum::<f32>() / valid_ratios.len() as f32
                    } else { 1.0 };
                    let ratio_std = if valid_ratios.len() > 1 {
                        let var = valid_ratios.iter().map(|r| (r - ratio_mean).powi(2)).sum::<f32>()
                            / valid_ratios.len() as f32;
                        var.sqrt()
                    } else { 0.0 };

                    // Log ratio diagnostics for EACH minibatch in first few updates
                    if update_count < 3 {
                        eprintln!(
                            "[RPPO DEBUG] Update {}.{} | Ratio mean: {:.6}, std: {:.6}",
                            update_count, epoch_updates, ratio_mean, ratio_std
                        );

                        if epoch_updates == 0 {
                            // Additional: check advantages statistics
                            let valid_advs: Vec<f32> = advantages_data.iter()
                                .zip(mask_data.iter())
                                .filter(|(_, &m)| m > 0.5)
                                .map(|(&a, _)| a)
                                .take(10)
                                .collect();
                            let valid_returns: Vec<f32> = returns_data.iter()
                                .zip(mask_data.iter())
                                .filter(|(_, &m)| m > 0.5)
                                .map(|(&r, _)| r)
                                .take(10)
                                .collect();
                            eprintln!(
                                "[RPPO DEBUG] Sample advantages: {:?}",
                                valid_advs
                            );
                            eprintln!(
                                "[RPPO DEBUG] Sample returns: {:?}",
                                valid_returns
                            );
                        }
                    }

                    let should_check_params = epoch_updates == 0 && update_count < 3;
                    epoch_updates += 1;

                    // Single backward pass for entire batch
                    // IMPORTANT: Call backward() BEFORE extracting loss values to avoid
                    // any potential autodiff interference
                    let grads = total_loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);

                    // Extract loss values AFTER backward (safe, no autodiff interference)
                    let policy_loss_val = policy_loss.into_data().as_slice::<f32>().unwrap()[0];
                    let value_loss_val = value_loss.into_data().as_slice::<f32>().unwrap()[0];
                    let entropy_val = -entropy_loss.into_data().as_slice::<f32>().unwrap()[0];
                    epoch_policy_loss += policy_loss_val;
                    epoch_value_loss += value_loss_val;
                    epoch_entropy += entropy_val;

                    // DEBUG: Check if optimizer is updating parameters
                    if should_check_params {
                        // Get a sample parameter value before update
                        let record_before = model.clone().into_record();
                        let bytes_before = recorder.record(record_before, ()).unwrap();
                        let param_sum_before: f32 = bytes_before.iter().map(|&b| b as f32).sum();

                        model = optimizer.step(config.learning_rate, model, grads);

                        // Get same parameter after update
                        let record_after = model.clone().into_record();
                        let bytes_after = recorder.record(record_after, ()).unwrap();
                        let param_sum_after: f32 = bytes_after.iter().map(|&b| b as f32).sum();

                        eprintln!(
                            "[RPPO DEBUG] Param check: before={:.1}, after={:.1}, diff={:.1}",
                            param_sum_before, param_sum_after, param_sum_after - param_sum_before
                        );
                    } else {
                        model = optimizer.step(config.learning_rate, model, grads);
                    }
                }
            }

            // Log loss diagnostics
            if epoch_updates > 0 {
                let avg_policy = epoch_policy_loss / epoch_updates as f32;
                let avg_value = epoch_value_loss / epoch_updates as f32;
                let avg_entropy = epoch_entropy / epoch_updates as f32;
                eprintln!(
                    "[RPPO] Update {:>3} | Policy: {:>7.4} | Value: {:>7.4} | Entropy: {:>5.3}",
                    update_count, avg_policy, avg_value, avg_entropy
                );

                // Additional diagnostics: check values and returns
                if update_count % 20 == 0 {
                    // Sample values and returns from the last batch
                    eprintln!(
                        "[RPPO DEBUG] n_sequences: {}, epoch_updates: {}",
                        sequences.len(), epoch_updates
                    );
                }
            }
            update_count += 1;

            // Publish updated weights
            if let Ok(bytes) = recorder.record(model.clone().into_record(), ()) {
                bytes_slot.publish(bytes);
            }
            policy_version.fetch_add(1, Ordering::Release);
            consumed_epoch.fetch_add(1, Ordering::Release);
        }

        model
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Sequence data for TBPTT training.
///
/// Stores actions as the raw `Action` enum to avoid generic type parameters.
/// Actions are converted to `A::Action` when computing log probabilities.
struct SequenceData {
    states: Vec<Vec<f32>>,
    actions: Vec<Action>,  // Store as Action enum, convert when needed
    rewards: Vec<f32>,
    values: Vec<f32>,
    dones: Vec<bool>,
    old_log_probs: Vec<f32>,
    initial_hidden: Vec<f32>,
    /// Bootstrap value V(s_T) for GAE computation.
    /// Some(v) if sequence doesn't end at episode boundary and next transition exists.
    /// None if sequence ends at episode boundary (done=true) or at rollout boundary.
    next_value: Option<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

impl SequenceData {
    /// Create sequence data from transitions with optional bootstrap value.
    ///
    /// # Arguments
    /// * `transitions` - The transitions in this sequence
    /// * `next_value` - Bootstrap value V(s_T) for non-terminal sequences
    fn from_transitions(transitions: &[RecurrentPPOTransition], next_value: Option<f32>) -> Self {
        let states: Vec<Vec<f32>> = transitions.iter().map(|t| t.state().to_vec()).collect();
        let actions: Vec<Action> = transitions.iter().map(|t| t.action().clone()).collect();
        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward()).collect();
        let values: Vec<f32> = transitions.iter().map(|t| t.value()).collect();
        let dones: Vec<bool> = transitions.iter().map(|t| t.done()).collect();
        let old_log_probs: Vec<f32> = transitions.iter().map(|t| t.log_prob()).collect();

        // Use the hidden state from the first transition as initial
        let initial_hidden = if let Some(first) = transitions.first() {
            first.hidden_state.clone()
        } else {
            Vec::new()
        };

        Self {
            states,
            actions,
            rewards,
            values,
            dones,
            old_log_probs,
            initial_hidden,
            next_value,
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    /// Convert stored Action to A::Action for log prob computation.
    fn get_action<AV: ActionValue>(&self, idx: usize) -> AV {
        match &self.actions[idx] {
            Action::Discrete(a) => AV::from_floats(&[*a as f32]),
            Action::Continuous(a) => AV::from_floats(a),
        }
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Distributed recurrent PPO with discrete actions.
pub type DistributedRecurrentPPODiscreteRunner<B> = DistributedRecurrentPPORunner<DiscretePolicy, B>;

/// Distributed recurrent PPO with continuous actions.
pub type DistributedRecurrentPPOContinuousRunner<B> = DistributedRecurrentPPORunner<ContinuousPolicy, B>;
