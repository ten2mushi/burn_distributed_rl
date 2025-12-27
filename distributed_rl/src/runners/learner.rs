//! Generic Learner for on-policy algorithms.
//!
//! The Learner is parameterized by three orthogonal axes:
//! - `A: ActionPolicy` - Discrete or continuous action handling
//! - `T: TemporalPolicy` - Feed-forward or recurrent structure
//! - `Alg: Algorithm` - Loss computation strategy (PPO, A2C, etc.)
//!
//! This design allows O(n+m+k) implementations instead of O(n×m×k).

use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::marker::PhantomData;

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::actor_critic::ActorCritic;
use crate::algorithms::algorithm::Algorithm;
use crate::algorithms::gae::{compute_gae_vectorized, normalize_advantages};
use crate::algorithms::temporal_policy::TemporalPolicy;

use super::rollout_storage::{
    extract_minibatch, generate_minibatches, ComputedValues, RolloutStorage,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Learner.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Number of parallel environments
    pub n_envs: usize,
    /// Steps per rollout per environment
    pub rollout_steps: usize,
    /// Training epochs per rollout
    pub epochs: usize,
    /// Minibatch size
    pub minibatch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Max gradient norm (for clipping)
    pub max_grad_norm: f32,
    /// Maximum total steps (stopping condition)
    pub max_steps: usize,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda
    pub gae_lambda: f32,
    /// Sequence length for recurrent training (TBPTT)
    pub sequence_length: usize,
    /// Whether to reset hidden state on terminal
    pub reset_hidden_on_terminal: bool,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            n_envs: 64,
            rollout_steps: 128,
            epochs: 4,
            minibatch_size: 256,
            learning_rate: 3e-4,
            max_grad_norm: 0.5,
            max_steps: 1_000_000,
            gamma: 0.99,
            gae_lambda: 0.95,
            sequence_length: 16,
            reset_hidden_on_terminal: true,
        }
    }
}

impl LearnerConfig {
    /// Create a new learner configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get rollout batch size (total transitions per rollout).
    pub fn rollout_size(&self) -> usize {
        self.n_envs * self.rollout_steps
    }

    /// Builder methods
    pub fn with_n_envs(mut self, n_envs: usize) -> Self {
        self.n_envs = n_envs;
        self
    }

    pub fn with_rollout_steps(mut self, steps: usize) -> Self {
        self.rollout_steps = steps;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn with_minibatch_size(mut self, size: usize) -> Self {
        self.minibatch_size = size;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_gae_lambda(mut self, lambda: f32) -> Self {
        self.gae_lambda = lambda;
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }
}

// ============================================================================
// Training Stats
// ============================================================================

/// Training statistics after each rollout.
#[derive(Debug, Clone)]
pub struct LearnerStats {
    /// Number of rollouts completed
    pub rollouts: usize,
    /// Total environment steps
    pub total_steps: usize,
    /// Total episodes completed
    pub episodes: usize,
    /// Average episode reward (recent)
    pub avg_reward: f32,
    /// Policy loss (last epoch)
    pub policy_loss: f32,
    /// Value loss (last epoch)
    pub value_loss: f32,
    /// Entropy (last epoch)
    pub entropy: f32,
}

// ============================================================================
// VectorizedEnv Trait
// ============================================================================

/// Trait for vectorized environments.
///
/// Provides a common interface for environments that run multiple
/// instances in parallel.
pub trait VectorizedEnv<A: ActionValue>: Send {
    /// Number of parallel environments.
    fn n_envs(&self) -> usize;

    /// Observation size.
    fn obs_size(&self) -> usize;

    /// Write current observations to buffer.
    fn write_observations(&self, buffer: &mut [f32]);

    /// Step all environments with given actions.
    ///
    /// Returns (rewards, dones, terminals) for each environment.
    fn step(&mut self, actions: &[A]) -> StepResult;

    /// Reset specific environments.
    fn reset_envs(&mut self, indices: &[usize]);
}

/// Result from stepping vectorized environment.
pub struct StepResult {
    /// Rewards for each environment
    pub rewards: Vec<f32>,
    /// Done flags (terminal or truncated)
    pub dones: Vec<bool>,
    /// Terminal flags (true terminal, not truncation)
    pub terminals: Vec<bool>,
}

// ============================================================================
// Learner
// ============================================================================

/// Generic learner for on-policy algorithms.
///
/// # Type Parameters
///
/// - `A`: Action policy (discrete or continuous)
/// - `T`: Temporal policy (feed-forward or recurrent)
/// - `Alg`: Algorithm (PPO, A2C, etc.)
/// - `B`: Backend type
pub struct Learner<A, T, Alg, B>
where
    A: Clone + Send + Sync + 'static,
    T: Clone + Send + Sync + 'static,
    Alg: Clone + Send + Sync + 'static,
    B: AutodiffBackend,
{
    config: LearnerConfig,
    algorithm: Alg,
    _marker: PhantomData<(A, T, B)>,
}

impl<A, T, Alg, B> Learner<A, T, Alg, B>
where
    A: ActionPolicy<B>,
    T: TemporalPolicy<B>,
    Alg: Algorithm<B>,
    B: AutodiffBackend,
{
    /// Create a new learner with given configuration and algorithm.
    pub fn new(config: LearnerConfig, algorithm: Alg) -> Self {
        Self {
            config,
            algorithm,
            _marker: PhantomData,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &LearnerConfig {
        &self.config
    }

    /// Run training loop.
    ///
    /// # Arguments
    ///
    /// - `model`: Actor-critic model to train
    /// - `env`: Vectorized environment
    /// - `optimizer`: Optimizer instance
    /// - `callback`: Called after each rollout with training stats
    ///
    /// # Returns
    ///
    /// The trained model.
    pub fn run<M, E, O>(
        &self,
        mut model: M,
        mut env: E,
        mut optimizer: O,
        callback: impl Fn(&LearnerStats),
    ) -> M
    where
        M: ActorCritic<B, A, T>,
        E: VectorizedEnv<A::Action>,
        O: Optimizer<M, B>,
        B::Device: Default + Clone,
        B::FloatElem: From<f32>,
    {
        let config = &self.config;
        let device = B::Device::default();
        let obs_size = env.obs_size();

        // Pre-allocate buffers
        let mut obs_buffer = vec![0.0f32; config.n_envs * obs_size];
        let mut storage: RolloutStorage<A::Action, T::Hidden> =
            RolloutStorage::new(config.n_envs, config.rollout_steps, obs_size);

        // Initialize hidden state for recurrent policies
        let mut hidden = model.initial_hidden(config.n_envs, &device);

        // Tracking
        let mut total_steps = 0usize;
        let mut rollout_count = 0usize;
        let mut episode_rewards: Vec<f32> = vec![0.0; config.n_envs];
        let mut completed_episodes = 0usize;
        let mut recent_rewards: Vec<f32> = Vec::with_capacity(100);

        // Training loop
        while total_steps < config.max_steps {
            storage.clear();

            // Phase 1: Collect rollout
            for _step in 0..config.rollout_steps {
                // Get observations
                env.write_observations(&mut obs_buffer);

                // Create observation tensor
                let obs_tensor = Tensor::<B, 1>::from_floats(&obs_buffer[..], &device)
                    .reshape([config.n_envs, obs_size]);

                // Forward pass
                let output = model.forward(obs_tensor, hidden.clone());

                // Sample actions
                let (actions, log_probs) = output.sample_actions(&device);

                // Extract values
                let values_data: burn::tensor::TensorData = output.values.clone().flatten::<1>(0, 1).into_data();
                let values: Vec<f32> = values_data.as_slice::<f32>().unwrap().to_vec();

                // Step environment
                let step_result = env.step(&actions);

                // Update episode rewards
                for (i, &reward) in step_result.rewards.iter().enumerate() {
                    episode_rewards[i] += reward;
                }

                // Handle terminal states
                let terminal_indices: Vec<usize> = step_result
                    .terminals
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &t)| if t { Some(i) } else { None })
                    .collect();

                if config.reset_hidden_on_terminal && !terminal_indices.is_empty() {
                    if T::is_recurrent() {
                        model.temporal_policy().handle_terminals(
                            &mut hidden,
                            &terminal_indices,
                            &device,
                        );
                    }
                }

                // Track completed episodes and collect indices for reset
                let done_indices: Vec<usize> = step_result
                    .dones
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &done)| {
                        if done {
                            recent_rewards.push(episode_rewards[i]);
                            if recent_rewards.len() > 100 {
                                recent_rewards.remove(0);
                            }
                            episode_rewards[i] = 0.0;
                            completed_episodes += 1;
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Reset completed environments
                if !done_indices.is_empty() {
                    env.reset_envs(&done_indices);
                }

                // Store transition
                storage.push_step(
                    &obs_buffer,
                    actions,
                    &step_result.rewards,
                    &step_result.dones,
                    &step_result.terminals,
                    &values,
                    &log_probs,
                    hidden.clone(),
                );

                // Update hidden state for next step
                hidden = output.hidden;
            }

            total_steps += config.rollout_size();

            // Get bootstrap values for GAE
            env.write_observations(&mut obs_buffer);
            let last_obs = Tensor::<B, 1>::from_floats(&obs_buffer[..], &device)
                .reshape([config.n_envs, obs_size]);
            let last_output = model.forward(last_obs, hidden.clone());
            let last_values_data: burn::tensor::TensorData = last_output.values.flatten::<1>(0, 1).into_data();
            let last_values: Vec<f32> = last_values_data.as_slice::<f32>().unwrap().to_vec();

            // Phase 2: Compute GAE
            let computed = self.compute_gae(&storage, &last_values);

            // Phase 3: Training epochs
            let (policy_loss, value_loss, entropy) = if T::is_recurrent() {
                self.train_recurrent(&mut model, &mut optimizer, &storage, &computed, &device)
            } else {
                self.train_feedforward(&mut model, &mut optimizer, &storage, &computed, &device)
            };

            rollout_count += 1;

            // Compute stats
            let avg_reward = if recent_rewards.is_empty() {
                0.0
            } else {
                recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32
            };

            let stats = LearnerStats {
                rollouts: rollout_count,
                total_steps,
                episodes: completed_episodes,
                avg_reward,
                policy_loss,
                value_loss,
                entropy,
            };

            callback(&stats);
        }

        model
    }

    /// Compute GAE for the rollout.
    fn compute_gae(
        &self,
        storage: &RolloutStorage<A::Action, T::Hidden>,
        last_values: &[f32],
    ) -> ComputedValues
    where
        T::Hidden: Clone,
    {
        let config = &self.config;

        // Compute GAE using flat storage arrays
        let (advantages, returns) = compute_gae_vectorized(
            &storage.rewards,
            &storage.values,
            &storage.dones,
            last_values,
            config.n_envs,
            config.gamma,
            config.gae_lambda,
        );

        // Normalize advantages (modifies in place)
        let mut norm_advantages = advantages;
        normalize_advantages(&mut norm_advantages);

        ComputedValues::new(norm_advantages, returns)
    }

    /// Train with feed-forward minibatches.
    fn train_feedforward<M, O>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        storage: &RolloutStorage<A::Action, T::Hidden>,
        computed: &ComputedValues,
        device: &B::Device,
    ) -> (f32, f32, f32)
    where
        M: ActorCritic<B, A, T>,
        O: Optimizer<M, B>,
        T::Hidden: Clone + Default,
    {
        let config = &self.config;
        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0;

        for _epoch in 0..config.epochs {
            let minibatches = generate_minibatches(storage.len(), config.minibatch_size);

            for mb_indices in minibatches {
                let mb_data = extract_minibatch(storage, computed, &mb_indices);

                // Create tensors
                let states = mb_data.states_tensor::<B>(device);
                let old_log_probs = mb_data.old_log_probs_tensor::<B>(device);
                let old_values = mb_data.old_values_tensor::<B>(device);
                let advantages = mb_data.advantages_tensor::<B>(device);
                let returns = mb_data.returns_tensor::<B>(device);

                // Forward pass
                let output = model.forward(states, T::Hidden::default());

                // Compute log probs and entropy from policy output
                let log_probs = output.policy.log_prob(&mb_data.actions, device);
                let entropy = output.policy.entropy();

                // Compute loss using algorithm
                let loss_output = self.algorithm.compute_loss(
                    log_probs,
                    entropy,
                    output.values,
                    old_log_probs,
                    old_values,
                    advantages,
                    returns,
                );

                total_policy_loss += loss_output.policy_loss;
                total_value_loss += loss_output.value_loss;
                total_entropy += loss_output.entropy;
                update_count += 1;

                // Backward pass
                let grads = loss_output.total_loss.backward();
                let grads = GradientsParams::from_grads(grads, model);

                // Optimizer step
                *model = optimizer.step(config.learning_rate, model.clone(), grads);
            }
        }

        let n = update_count.max(1) as f32;
        (
            total_policy_loss / n,
            total_value_loss / n,
            total_entropy / n,
        )
    }

    /// Train with recurrent TBPTT (Truncated Backpropagation Through Time).
    ///
    /// This implementation uses efficient fixed-chunk BPTT:
    /// 1. Processes rollout in fixed-length chunks (sequence_length)
    /// 2. All environments are batched together (batch_size = n_envs)
    /// 3. Accumulates loss tensor over chunk timesteps
    /// 4. Single backward pass per chunk (true BPTT)
    ///
    /// This achieves O(1) optimizer steps per chunk instead of O(sequence_length),
    /// providing ~16x speedup for sequence_length=16.
    fn train_recurrent<M, O>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        storage: &RolloutStorage<A::Action, T::Hidden>,
        computed: &ComputedValues,
        device: &B::Device,
    ) -> (f32, f32, f32)
    where
        M: ActorCritic<B, A, T>,
        O: Optimizer<M, B>,
        T::Hidden: Clone + Default,
    {
        let config = &self.config;
        let n_envs = config.n_envs;
        let obs_size = storage.obs_size;

        // Use fixed-length chunks for efficiency
        let seq_len = config.sequence_length.min(config.rollout_steps);
        let n_chunks = config.rollout_steps / seq_len;

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0;

        for _epoch in 0..config.epochs {
            for chunk_idx in 0..n_chunks {
                let start_step = chunk_idx * seq_len;

                // Get initial hidden state for this chunk
                // Use stored hidden state at chunk boundary if available
                let mut hidden = if start_step == 0 || storage.hidden_states.is_empty() {
                    model.initial_hidden(n_envs, device)
                } else {
                    storage.hidden_states[start_step].clone()
                };

                // Accumulate the loss TENSOR across chunk timesteps (key to BPTT!)
                let mut accumulated_loss: Option<Tensor<B, 1>> = None;
                let mut chunk_policy_loss = 0.0f32;
                let mut chunk_value_loss = 0.0f32;
                let mut chunk_entropy = 0.0f32;

                // Unroll through chunk timesteps - all n_envs batched together
                for t in 0..seq_len {
                    let step = start_step + t;
                    let step_start = step * n_envs;
                    let step_end = step_start + n_envs;

                    // Get data for this step (all environments batched)
                    let obs_start = step * n_envs * obs_size;
                    let obs_end = obs_start + n_envs * obs_size;

                    let states = Tensor::<B, 1>::from_floats(
                        &storage.states[obs_start..obs_end],
                        device,
                    )
                    .reshape([n_envs, obs_size]);

                    let old_log_probs = Tensor::<B, 1>::from_floats(
                        &storage.log_probs[step_start..step_end],
                        device,
                    );
                    let old_values = Tensor::<B, 1>::from_floats(
                        &storage.values[step_start..step_end],
                        device,
                    );
                    let advantages = Tensor::<B, 1>::from_floats(
                        &computed.advantages[step_start..step_end],
                        device,
                    );
                    let returns = Tensor::<B, 1>::from_floats(
                        &computed.returns[step_start..step_end],
                        device,
                    );

                    let actions = storage.actions[step_start..step_end].to_vec();

                    // Forward pass - hidden state propagates through time
                    let output = model.forward(states, hidden);
                    hidden = output.hidden;

                    // Compute log probs and entropy
                    let log_probs = output.policy.log_prob(&actions, device);
                    let entropy = output.policy.entropy();

                    // Compute loss - the total_loss tensor stays in autodiff graph!
                    let loss_output = self.algorithm.compute_loss(
                        log_probs,
                        entropy,
                        output.values,
                        old_log_probs,
                        old_values,
                        advantages,
                        returns,
                    );

                    // Accumulate the TENSOR for proper BPTT gradient flow
                    accumulated_loss = match accumulated_loss {
                        Some(acc) => Some(acc + loss_output.total_loss),
                        None => Some(loss_output.total_loss),
                    };

                    // Accumulate scalar metrics for logging
                    chunk_policy_loss += loss_output.policy_loss;
                    chunk_value_loss += loss_output.value_loss;
                    chunk_entropy += loss_output.entropy;
                }

                // Average loss over chunk timesteps
                let avg_loss = accumulated_loss.unwrap() / seq_len as f32;

                // Track metrics
                total_policy_loss += chunk_policy_loss / seq_len as f32;
                total_value_loss += chunk_value_loss / seq_len as f32;
                total_entropy += chunk_entropy / seq_len as f32;
                update_count += 1;

                // SINGLE backward pass - gradients flow through ALL chunk timesteps!
                let grads = avg_loss.backward();
                let grads = GradientsParams::from_grads(grads, model);

                // SINGLE optimizer step per chunk
                *model = optimizer.step(config.learning_rate, model.clone(), grads);
            }
        }

        let n = update_count.max(1) as f32;
        (
            total_policy_loss / n,
            total_value_loss / n,
            total_entropy / n,
        )
    }
}

// ============================================================================
// Type Aliases for Common Combinations
// ============================================================================

use crate::algorithms::action_policy::{ContinuousPolicy, DiscretePolicy};
use crate::algorithms::algorithm::PPOAlgorithm;
use crate::algorithms::temporal_policy::{FeedForward, Recurrent};

/// PPO with discrete actions and feed-forward policy.
pub type PPODiscrete<B> = Learner<DiscretePolicy, FeedForward, PPOAlgorithm, B>;

/// PPO with continuous actions and feed-forward policy.
pub type PPOContinuous<B> = Learner<ContinuousPolicy, FeedForward, PPOAlgorithm, B>;

/// PPO with discrete actions and recurrent (LSTM/GRU) policy.
pub type RecurrentPPODiscrete<B> = Learner<DiscretePolicy, Recurrent, PPOAlgorithm, B>;

/// PPO with continuous actions and recurrent policy.
pub type RecurrentPPOContinuous<B> = Learner<ContinuousPolicy, Recurrent, PPOAlgorithm, B>;
