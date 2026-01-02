//! Recurrent PPO training strategy with TBPTT.
//!
//! This strategy implements PPO training for recurrent policies (LSTM/GRU):
//! - Hidden state persisted across timesteps
//! - Hidden state reset on episode termination
//! - TBPTT (Truncated Backpropagation Through Time) sequence chunking
//! - Sequence-based minibatch training with masking
//!
//! # TBPTT Overview
//!
//! Truncated Backpropagation Through Time limits the gradient flow to
//! `tbptt_length` timesteps, preventing gradient explosion while still
//! allowing the network to learn temporal dependencies within each chunk.
//!
//! ```text
//! Rollout: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ...]
//!           |----chunk 1----|  |----chunk 2----|  ...
//!           └── TBPTT grad ─┘  └── TBPTT grad ─┘
//! ```

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use rand::seq::SliceRandom;

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::actor_critic::ActorCritic;
use crate::algorithms::gae::compute_gae;
use crate::algorithms::ppo::normalization::SharedRewardNormalizer;
use crate::algorithms::ppo::ppo_transition::{RecurrentHiddenData, PPOTransition};
use crate::algorithms::temporal_policy::{HiddenConfig, HiddenStateType, Recurrent, RecurrentHidden};
use crate::core::transition::{Action, Transition};
use crate::runners::ppo_config::{AdvantageNormalization, PPOConfig};

use super::{normalize_advantages, PPOTrainingStrategy, PreparedTrainingData, SequenceInfo};

// ============================================================================
// RecurrentStrategy
// ============================================================================

/// Recurrent PPO training strategy with TBPTT.
///
/// Implements PPO with:
/// - Hidden state storage in transitions
/// - Hidden state persistence across timesteps
/// - TBPTT sequence chunking
/// - Masked sequence-based minibatch training
#[derive(Debug, Clone, Copy, Default)]
pub struct RecurrentStrategy;

impl<B, A> PPOTrainingStrategy<B, A, Recurrent> for RecurrentStrategy
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
{
    type TransitionData = RecurrentHiddenData;

    fn create_transition(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
        hidden_data: Vec<f32>,
        sequence_id: u64,
        step_in_sequence: u32,
        is_sequence_start: bool,
    ) -> PPOTransition<Self::TransitionData> {
        PPOTransition::recurrent(
            base,
            log_prob,
            value,
            bootstrap_value,
            hidden_data,
            sequence_id,
            step_in_sequence,
            is_sequence_start,
        )
    }

    fn prepare_training_data(
        rollouts: Vec<Vec<PPOTransition<Self::TransitionData>>>,
        config: &PPOConfig,
        _hidden_config: &HiddenConfig,
        reward_normalizer: Option<&SharedRewardNormalizer>,
    ) -> PreparedTrainingData<A::Action> {
        let tbptt_length = config.rollout_length.min(config.tbptt_length);

        // =========================================================
        // PHASE 1: Compute GAE per-environment over FULL rollout
        // =========================================================
        // GAE computed once over entire trajectory,
        // not per TBPTT chunk. This ensures advantage estimates near chunk
        // boundaries can "see" rewards beyond the chunk.

        let mut env_advantages: Vec<Vec<f32>> = Vec::with_capacity(rollouts.len());
        let mut env_returns: Vec<Vec<f32>> = Vec::with_capacity(rollouts.len());

        for rollout in &rollouts {
            if rollout.is_empty() {
                env_advantages.push(Vec::new());
                env_returns.push(Vec::new());
                continue;
            }

            // Use raw rewards for GAE computation
            // Return normalization is applied AFTER GAE, not before
            let rewards: Vec<f32> = rollout.iter().map(|t| t.reward()).collect();
            let values: Vec<f32> = rollout.iter().map(|t| t.value).collect();
            let dones: Vec<bool> = rollout.iter().map(|t| t.done()).collect();

            // Bootstrap value from last transition
            let bootstrap = rollout
                .last()
                .and_then(|t| {
                    if t.terminal() {
                        Some(0.0)
                    } else {
                        t.bootstrap_value
                    }
                })
                .unwrap_or(0.0);

            // Compute GAE over the ENTIRE rollout
            let (advantages, returns) = compute_gae(
                &rewards,
                &values,
                &dones,
                bootstrap,
                config.gamma,
                config.gae_lambda,
            );

            env_advantages.push(advantages);
            env_returns.push(returns);
        }

        // =========================================================
        // PHASE 2: Create TBPTT sequences using pre-computed GAE
        // =========================================================

        let mut sequences: Vec<SequenceData> = Vec::new();

        for (env_idx, rollout) in rollouts.iter().enumerate() {
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
                    // Use PRE-COMPUTED advantages from full rollout GAE
                    let seq_advantages = env_advantages[env_idx][chunk_start..actual_end].to_vec();
                    let seq_returns = env_returns[env_idx][chunk_start..actual_end].to_vec();

                    sequences.push(SequenceData::from_transitions_with_gae(
                        chunk,
                        seq_advantages,
                        seq_returns,
                    ));
                }

                chunk_start = actual_end;
            }
        }

        if sequences.is_empty() {
            return PreparedTrainingData {
                states: Vec::new(),
                actions: Vec::new(),
                old_log_probs: Vec::new(),
                old_values: Vec::new(),
                advantages: Vec::new(),
                returns: Vec::new(),
                dones: Vec::new(),
                terminals: Vec::new(),
                obs_dim: 0,
                n_transitions: 0,
                sequence_info: None,
            };
        }

        // =========================================================
        // PHASE 3: Flatten sequences into training data
        // =========================================================

        let obs_dim = sequences
            .first()
            .and_then(|s| s.states.first())
            .map(|s| s.len())
            .unwrap_or(0);

        let mut all_states: Vec<f32> = Vec::new();
        let mut all_actions: Vec<A::Action> = Vec::new();
        let mut all_old_log_probs: Vec<f32> = Vec::new();
        let mut all_old_values: Vec<f32> = Vec::new();
        let mut all_advantages: Vec<f32> = Vec::new();
        let mut all_returns: Vec<f32> = Vec::new();
        let mut all_dones: Vec<bool> = Vec::new();
        let mut all_terminals: Vec<bool> = Vec::new();
        let mut initial_hidden_states: Vec<Vec<f32>> = Vec::new();
        let mut sequence_lengths: Vec<usize> = Vec::new();
        let mut sequence_starts: Vec<usize> = Vec::new();

        let mut current_idx = 0;
        for seq in &sequences {
            sequence_starts.push(current_idx);
            sequence_lengths.push(seq.states.len());
            initial_hidden_states.push(seq.initial_hidden.clone());

            for (i, state) in seq.states.iter().enumerate() {
                all_states.extend(state);
                all_actions.push(seq.get_action::<A::Action>(i));
                all_old_log_probs.push(seq.old_log_probs[i]);
                all_old_values.push(seq.old_values[i]);
                all_advantages.push(seq.advantages[i]);
                all_returns.push(seq.returns[i]);
                all_dones.push(seq.dones[i]);
                all_terminals.push(seq.terminals[i]);
                current_idx += 1;
            }
        }

        let n_transitions = all_actions.len();

        // Normalize returns by return std if enabled
        // This scales value function targets to have consistent magnitude
        let returns = if let Some(normalizer) = reward_normalizer {
            normalizer.normalize_returns(&all_returns)
        } else {
            all_returns
        };

        // Normalize advantages if using global normalization
        let advantages = if config.normalize_advantages
            && config.advantage_normalization == AdvantageNormalization::Global
        {
            normalize_advantages(all_advantages)
        } else {
            all_advantages
        };

        PreparedTrainingData {
            states: all_states,
            actions: all_actions,
            old_log_probs: all_old_log_probs,
            old_values: all_old_values,
            advantages,
            returns,
            dones: all_dones,
            terminals: all_terminals,
            obs_dim,
            n_transitions,
            sequence_info: Some(SequenceInfo {
                initial_hidden_states,
                sequence_lengths,
                sequence_starts,
                mask: None, // Built per-batch
            }),
        }
    }

    fn train_epoch<M, O>(
        mut model: M,
        optimizer: &mut O,
        data: &PreparedTrainingData<A::Action>,
        config: &PPOConfig,
        hidden_config: &HiddenConfig,
        device: &B::Device,
        current_env_steps: usize,
        initial_lr: f64,
    ) -> (M, bool)
    where
        M: AutodiffModule<B> + ActorCritic<B, A, Recurrent>,
        O: Optimizer<M, B>,
    {
        if data.n_transitions == 0 {
            return (model, false);
        }

        let seq_info = match &data.sequence_info {
            Some(info) => info,
            None => return (model, false),
        };

        let obs_dim = data.obs_dim;
        let n_sequences = seq_info.sequence_lengths.len();

        // Shuffle sequence order (not individual transitions!)
        let mut seq_indices: Vec<usize> = (0..n_sequences).collect();
        seq_indices.shuffle(&mut rand::thread_rng());

        // Process sequences in minibatches
        let sequences_per_batch = (n_sequences / config.n_minibatches).max(1);
        let mut kl_early_stop = false;

        for batch_start in (0..n_sequences).step_by(sequences_per_batch) {
            if kl_early_stop {
                break;
            }

            let batch_end = (batch_start + sequences_per_batch).min(n_sequences);
            let batch_seq_indices = &seq_indices[batch_start..batch_end];
            let n_seqs = batch_seq_indices.len();

            if n_seqs == 0 {
                continue;
            }

            // Find max length in this batch
            let max_len = batch_seq_indices
                .iter()
                .map(|&i| seq_info.sequence_lengths[i])
                .max()
                .unwrap_or(0);

            if max_len == 0 {
                continue;
            }

            // Build padded batch tensors and mask
            let mut obs_data = vec![0.0f32; n_seqs * max_len * obs_dim];
            let mut old_log_probs_data = vec![0.0f32; n_seqs * max_len];
            let mut old_values_data = vec![0.0f32; n_seqs * max_len];
            let mut advantages_data = vec![0.0f32; n_seqs * max_len];
            let mut returns_data = vec![0.0f32; n_seqs * max_len];
            let mut mask_data = vec![0.0f32; n_seqs * max_len];
            let mut actions_by_timestep: Vec<Vec<A::Action>> =
                vec![Vec::with_capacity(n_seqs); max_len];
            // CRITICAL: Use terminals, NOT dones, for hidden state reset during training
            // This ensures training mirrors collection behavior where:
            // - Terminal: Reset hidden (new episode, fresh memory)
            // - Truncated: Preserve hidden (same episode, memory continues)
            let mut terminals_by_timestep: Vec<Vec<bool>> = vec![vec![false; n_seqs]; max_len];

            // Initialize batched hidden state
            let expected_hidden_size = if hidden_config.has_cell {
                hidden_config.hidden_size * 2
            } else {
                hidden_config.hidden_size
            };

            let mut hidden = RecurrentHidden::<B>::new(
                n_seqs,
                hidden_config.hidden_size,
                hidden_config.has_cell,
                device,
            );

            // Fill batch data
            for (local_idx, &global_seq_idx) in batch_seq_indices.iter().enumerate() {
                let seq_start = seq_info.sequence_starts[global_seq_idx];
                let seq_len = seq_info.sequence_lengths[global_seq_idx];

                // Initialize hidden for this sequence
                let init_hidden = &seq_info.initial_hidden_states[global_seq_idx];
                if init_hidden.len() == expected_hidden_size {
                    let h = RecurrentHidden::<B>::from_vec(init_hidden, 1, device, hidden_config);
                    hidden.set(local_idx, h.states[0].clone());
                }

                for t in 0..seq_len {
                    let flat_idx = seq_start + t;
                    let batch_idx = local_idx * max_len + t;
                    let obs_start = batch_idx * obs_dim;

                    obs_data[obs_start..obs_start + obs_dim]
                        .copy_from_slice(&data.states[flat_idx * obs_dim..(flat_idx + 1) * obs_dim]);
                    old_log_probs_data[batch_idx] = data.old_log_probs[flat_idx];
                    old_values_data[batch_idx] = data.old_values[flat_idx];
                    advantages_data[batch_idx] = data.advantages[flat_idx];
                    returns_data[batch_idx] = data.returns[flat_idx];
                    mask_data[batch_idx] = 1.0;
                    actions_by_timestep[t].push(data.actions[flat_idx].clone());

                    // Use terminal flags (NOT dones) for hidden state reset
                    terminals_by_timestep[t][local_idx] = data.terminals[flat_idx];
                }

                // Pad actions for shorter sequences
                for t in seq_len..max_len {
                    actions_by_timestep[t].push(data.actions[seq_start].clone()); // Dummy
                }
            }

            // Per-minibatch advantage normalization if configured
            if config.normalize_advantages
                && config.advantage_normalization == AdvantageNormalization::PerMinibatch
            {
                let valid_advs: Vec<f32> = advantages_data
                    .iter()
                    .zip(mask_data.iter())
                    .filter(|(_, &m)| m > 0.5)
                    .map(|(&a, _)| a)
                    .collect();

                if valid_advs.len() > 1 {
                    let mean = valid_advs.iter().sum::<f32>() / valid_advs.len() as f32;
                    let var = valid_advs
                        .iter()
                        .map(|a| (a - mean).powi(2))
                        .sum::<f32>()
                        / valid_advs.len() as f32;
                    let std = (var + 1e-8).sqrt();

                    if mean.is_finite() && std.is_finite() {
                        for (adv, &m) in advantages_data.iter_mut().zip(mask_data.iter()) {
                            if m > 0.5 {
                                let norm = (*adv - mean) / std;
                                *adv = if norm.is_finite() { norm } else { 0.0 };
                            }
                        }
                    }
                }
            }

            // Process timestep by timestep (batched)
            let mut all_log_probs: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);
            let mut all_entropies: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);
            let mut all_values: Vec<Tensor<B, 1>> = Vec::with_capacity(max_len);

            for t in 0..max_len {
                // Gather obs for this timestep across all sequences
                let obs_slice: Vec<f32> = (0..n_seqs)
                    .flat_map(|s| {
                        let idx = s * max_len * obs_dim + t * obs_dim;
                        obs_data[idx..idx + obs_dim].iter().copied()
                    })
                    .collect();

                let obs_t = Tensor::<B, 1>::from_floats(obs_slice.as_slice(), device)
                    .reshape([n_seqs, obs_dim]);

                // Batched forward pass
                let output = model.forward(obs_t, hidden);

                // Get log_probs, entropy, values
                let log_prob = output.policy.log_prob(&actions_by_timestep[t], device);
                let entropy = output.policy.entropy();
                let value = output.values_flat();

                all_log_probs.push(log_prob);
                all_entropies.push(entropy);
                all_values.push(value);

                // Update hidden (batched)
                hidden = output.hidden;

                // Reset hidden for sequences that TRULY terminated at this timestep
                // CRITICAL: Use terminals, not dones, to preserve hidden across truncation
                for (seq_idx, &terminal) in terminals_by_timestep[t].iter().enumerate() {
                    if terminal {
                        hidden.reset(seq_idx, device);
                    }
                }
            }

            // Stack: [max_len, n_seqs] -> transpose to [n_seqs, max_len] -> flatten
            let log_probs: Tensor<B, 2> = Tensor::stack(all_log_probs, 0);
            let entropies: Tensor<B, 2> = Tensor::stack(all_entropies, 0);
            let values: Tensor<B, 2> = Tensor::stack(all_values, 0);

            let log_probs: Tensor<B, 1> = log_probs.swap_dims(0, 1).flatten(0, 1);
            let entropies: Tensor<B, 1> = entropies.swap_dims(0, 1).flatten(0, 1);
            let values: Tensor<B, 1> = values.swap_dims(0, 1).flatten(0, 1);

            // Create target tensors
            let old_log_probs =
                Tensor::<B, 1>::from_floats(old_log_probs_data.as_slice(), device);
            let old_values_tensor =
                Tensor::<B, 1>::from_floats(old_values_data.as_slice(), device);
            let advantages = Tensor::<B, 1>::from_floats(advantages_data.as_slice(), device);
            let returns = Tensor::<B, 1>::from_floats(returns_data.as_slice(), device);
            let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device);

            // Compute masked losses
            let log_ratio = log_probs.clone() - old_log_probs;
            let ratio = log_ratio.clone().exp();

            // Approximate KL divergence with masking
            let approx_kl = if config.target_kl.is_some() {
                let kl_per_step = (ratio.clone() - 1.0) - log_ratio;
                let kl_masked = kl_per_step * mask.clone();
                let kl_sum = kl_masked.sum().into_data().as_slice::<f32>().unwrap()[0];
                let n_valid = mask.clone().sum().into_data().as_slice::<f32>().unwrap()[0];
                if n_valid > 0.0 {
                    Some((kl_sum / n_valid) as f64)
                } else {
                    Some(0.0)
                }
            } else {
                None
            };

            // Clipped surrogate objective
            let surr1 = ratio.clone() * advantages.clone();
            let surr2 = ratio
                .clone()
                .clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                * advantages;
            let policy_loss_per_step = -surr1.min_pair(surr2);

            // Value loss (with optional clipping)
            let value_loss_per_step = if config.clip_vloss {
                let v_loss_unclipped = (values.clone() - returns.clone()).powf_scalar(2.0);
                let v_clipped = old_values_tensor.clone()
                    + (values - old_values_tensor).clamp(-config.clip_ratio, config.clip_ratio);
                let v_loss_clipped = (v_clipped - returns).powf_scalar(2.0);
                v_loss_unclipped.max_pair(v_loss_clipped).mul_scalar(0.5)
            } else {
                (values - returns).powf_scalar(2.0)
            };

            let entropy_per_step = entropies;

            // Apply mask and mean
            let n_valid = mask.clone().sum();
            let policy_loss = (policy_loss_per_step * mask.clone()).sum() / n_valid.clone();
            let value_loss = (value_loss_per_step * mask.clone()).sum() / n_valid.clone();
            let entropy_loss = -(entropy_per_step * mask).sum() / n_valid;

            // Total loss
            let total_loss = policy_loss
                + value_loss.mul_scalar(config.vf_coef)
                + entropy_loss.mul_scalar(config.entropy_coef);

            // Backward pass
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Learning rate (with optional annealing)
            let current_lr = if config.anneal_lr {
                let frac = 1.0 - (current_env_steps as f64 / config.max_env_steps as f64);
                initial_lr * frac.max(0.0)
            } else {
                initial_lr
            };

            // Optimizer step
            model = optimizer.step(current_lr, model, grads);

            // KL early stopping
            if let (Some(kl), Some(target)) = (approx_kl, config.target_kl) {
                if kl > target {
                    log::debug!(
                        "KL early stop triggered: approx_kl={:.4} > target_kl={:.4}",
                        kl,
                        target
                    );
                    kl_early_stop = true;
                }
            }
        }

        (model, kl_early_stop)
    }
}

// ============================================================================
// SequenceData Helper
// ============================================================================

/// Internal sequence data for TBPTT training.
struct SequenceData {
    states: Vec<Vec<f32>>,
    actions: Vec<Action>,
    #[allow(dead_code)]
    rewards: Vec<f32>,
    #[allow(dead_code)]
    values: Vec<f32>,
    dones: Vec<bool>,
    /// Terminal flags (TRUE episode termination only, NOT truncated).
    /// Used for hidden state reset during training to match collection behavior.
    terminals: Vec<bool>,
    old_log_probs: Vec<f32>,
    old_values: Vec<f32>,
    initial_hidden: Vec<f32>,
    #[allow(dead_code)]
    next_value: Option<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

impl SequenceData {
    /// Create sequence data from transitions.
    #[allow(dead_code)]
    fn from_transitions(
        transitions: &[PPOTransition<RecurrentHiddenData>],
        next_value: Option<f32>,
    ) -> Self {
        let states: Vec<Vec<f32>> = transitions.iter().map(|t| t.state().to_vec()).collect();
        let actions: Vec<Action> = transitions.iter().map(|t| t.action().clone()).collect();
        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward()).collect();
        let values: Vec<f32> = transitions.iter().map(|t| t.value).collect();
        let dones: Vec<bool> = transitions.iter().map(|t| t.done()).collect();
        let terminals: Vec<bool> = transitions.iter().map(|t| t.terminal()).collect();
        let old_log_probs: Vec<f32> = transitions.iter().map(|t| t.log_prob).collect();
        let old_values: Vec<f32> = transitions.iter().map(|t| t.value).collect();

        // Use hidden state from first transition
        let initial_hidden = transitions
            .first()
            .map(|t| t.hidden_data.data.clone())
            .unwrap_or_default();

        Self {
            states,
            actions,
            rewards,
            values,
            dones,
            terminals,
            old_log_probs,
            old_values,
            initial_hidden,
            next_value,
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    /// Create sequence data from transitions with pre-computed GAE.
    ///
    /// This is used when GAE is computed over the full rollout before chunking
    /// rather than per-chunk.
    fn from_transitions_with_gae(
        transitions: &[PPOTransition<RecurrentHiddenData>],
        advantages: Vec<f32>,
        returns: Vec<f32>,
    ) -> Self {
        let states: Vec<Vec<f32>> = transitions.iter().map(|t| t.state().to_vec()).collect();
        let actions: Vec<Action> = transitions.iter().map(|t| t.action().clone()).collect();
        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward()).collect();
        let values: Vec<f32> = transitions.iter().map(|t| t.value).collect();
        let dones: Vec<bool> = transitions.iter().map(|t| t.done()).collect();
        let terminals: Vec<bool> = transitions.iter().map(|t| t.terminal()).collect();
        let old_log_probs: Vec<f32> = transitions.iter().map(|t| t.log_prob).collect();
        let old_values: Vec<f32> = transitions.iter().map(|t| t.value).collect();

        // Use hidden state from first transition
        let initial_hidden = transitions
            .first()
            .map(|t| t.hidden_data.data.clone())
            .unwrap_or_default();

        Self {
            states,
            actions,
            rewards,
            values,
            dones,
            terminals,
            old_log_probs,
            old_values,
            initial_hidden,
            next_value: None, // Not needed with pre-computed GAE
            advantages,
            returns,
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_rec_transition(
        env_id: usize,
        step: usize,
        seq_id: u64,
        is_start: bool,
    ) -> PPOTransition<RecurrentHiddenData> {
        PPOTransition::recurrent(
            Transition::new_discrete(
                vec![env_id as f32, step as f32],
                0,
                1.0,
                vec![env_id as f32, (step + 1) as f32],
                false,
                false,
            ),
            -0.5,
            1.0,
            None,
            vec![0.1, 0.2, 0.3, 0.4], // hidden state
            seq_id,
            step as u32,
            is_start,
        )
    }

    fn make_terminal_rec_transition(
        env_id: usize,
        step: usize,
        seq_id: u64,
    ) -> PPOTransition<RecurrentHiddenData> {
        PPOTransition::recurrent(
            Transition::new_discrete(
                vec![env_id as f32, step as f32],
                0,
                1.0,
                vec![0.0, 0.0],
                true,
                false,
            ),
            -0.5,
            0.0,
            None,
            vec![0.1, 0.2, 0.3, 0.4],
            seq_id,
            step as u32,
            false,
        )
    }

    #[test]
    fn test_create_transition() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        let base = Transition::new_discrete(vec![1.0, 2.0], 0, 1.0, vec![2.0, 3.0], false, false);

        let t = <RecurrentStrategy as PPOTrainingStrategy<B, A, Recurrent>>::create_transition(
            base.clone(),
            -0.5,
            1.0,
            Some(0.9),
            vec![0.1, 0.2, 0.3, 0.4], // hidden state
            42,
            5,
            true,
        );

        assert_eq!(t.state(), &[1.0, 2.0]);
        assert_eq!(t.log_prob, -0.5);
        assert_eq!(t.value, 1.0);
        assert_eq!(t.bootstrap_value, Some(0.9));
        assert_eq!(t.sequence_id, 42);
        assert_eq!(t.step_in_sequence, 5);
        assert!(t.is_sequence_start);
        // Hidden state should be stored
        assert!(!t.hidden_data.data.is_empty());
    }

    #[test]
    fn test_prepare_training_data_basic() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        // Create rollouts with sequences
        let rollouts = vec![
            vec![
                make_rec_transition(0, 0, 100, true),
                make_rec_transition(0, 1, 100, false),
                make_terminal_rec_transition(0, 2, 100),
            ],
            vec![
                make_rec_transition(1, 0, 101, true),
                make_rec_transition(1, 1, 101, false),
            ],
        ];

        let config = PPOConfig::default().with_tbptt_length(32);
        let hidden_config = HiddenConfig::lstm(4);

        let data = <RecurrentStrategy as PPOTrainingStrategy<B, A, Recurrent>>::prepare_training_data(
            rollouts,
            &config,
            &hidden_config,
            None,
        );

        assert_eq!(data.n_transitions, 5);
        assert_eq!(data.obs_dim, 2);
        assert!(data.sequence_info.is_some());

        let seq_info = data.sequence_info.as_ref().unwrap();
        assert_eq!(seq_info.sequence_lengths.len(), 2); // 2 sequences (TBPTT chunks)
        assert_eq!(seq_info.initial_hidden_states.len(), 2);
    }

    #[test]
    fn test_tbptt_chunking() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        // Create a long rollout that should be chunked
        let rollout: Vec<_> = (0..10)
            .map(|i| make_rec_transition(0, i, 100, i == 0))
            .collect();

        let rollouts = vec![rollout];

        let config = PPOConfig::default().with_tbptt_length(4);
        let hidden_config = HiddenConfig::lstm(4);

        let data = <RecurrentStrategy as PPOTrainingStrategy<B, A, Recurrent>>::prepare_training_data(
            rollouts,
            &config,
            &hidden_config,
            None,
        );

        let seq_info = data.sequence_info.as_ref().unwrap();
        // 10 transitions with tbptt_length=4 should give 3 chunks: [0-3], [4-7], [8-9]
        assert_eq!(seq_info.sequence_lengths.len(), 3);
        assert_eq!(seq_info.sequence_lengths[0], 4);
        assert_eq!(seq_info.sequence_lengths[1], 4);
        assert_eq!(seq_info.sequence_lengths[2], 2);
    }

    #[test]
    fn test_empty_rollouts() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        let rollouts: Vec<Vec<PPOTransition<RecurrentHiddenData>>> = vec![vec![], vec![]];

        let config = PPOConfig::default();
        let hidden_config = HiddenConfig::lstm(4);

        let data = <RecurrentStrategy as PPOTrainingStrategy<B, A, Recurrent>>::prepare_training_data(
            rollouts,
            &config,
            &hidden_config,
            None,
        );

        assert_eq!(data.n_transitions, 0);
    }
}
