//! Feed-forward PPO training strategy.
//!
//! This strategy implements standard PPO training for stateless policies:
//! - No hidden state management (zero-cost abstraction with `()`)
//! - Transitions shuffled across all environments
//! - Standard minibatch training
//!
//! # Zero-Cost Abstraction
//!
//! Since `()` is a zero-sized type:
//! - `PPOTransition<()>` has no hidden data overhead
//! - Hidden state operations are no-ops that compile away
//! - Identical performance to non-generic implementation

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use rand::seq::SliceRandom;

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::actor_critic::ActorCritic;
use crate::algorithms::gae::compute_gae;
use crate::algorithms::ppo::normalization::SharedRewardNormalizer;
use crate::algorithms::ppo::ppo_transition::PPOTransition;
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig};
use crate::core::transition::Transition;
use crate::runners::ppo_config::PPOConfig;

use super::{normalize_advantages, PPOTrainingStrategy, PreparedTrainingData};

// ============================================================================
// FeedForwardStrategy
// ============================================================================

/// Feed-forward PPO training strategy.
///
/// Implements standard PPO with:
/// - No hidden state (zero-cost `()` type)
/// - Shuffle all transitions across environments
/// - Standard minibatch training loop
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedForwardStrategy;

impl<B, A> PPOTrainingStrategy<B, A, FeedForward> for FeedForwardStrategy
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
{
    type TransitionData = ();

    fn create_transition(
        base: Transition,
        log_prob: f32,
        value: f32,
        bootstrap_value: Option<f32>,
        _hidden_data: Vec<f32>,
        _sequence_id: u64,
        _step_in_sequence: u32,
        _is_sequence_start: bool,
    ) -> PPOTransition<Self::TransitionData> {
        // Feed-forward: no hidden state, no sequence tracking
        PPOTransition::feed_forward(base, log_prob, value, bootstrap_value)
    }

    fn prepare_training_data(
        rollouts: Vec<Vec<PPOTransition<Self::TransitionData>>>,
        config: &PPOConfig,
        _hidden_config: &HiddenConfig,
        reward_normalizer: Option<&SharedRewardNormalizer>,
    ) -> PreparedTrainingData<A::Action> {
        let mut all_states: Vec<f32> = Vec::new();
        let mut all_actions: Vec<A::Action> = Vec::new();
        let mut all_old_log_probs: Vec<f32> = Vec::new();
        let mut all_old_values: Vec<f32> = Vec::new();
        let mut all_advantages: Vec<f32> = Vec::new();
        let mut all_returns: Vec<f32> = Vec::new();
        let mut obs_dim: Option<usize> = None;

        // Process each environment's rollout independently for GAE
        for rollout in rollouts {
            if rollout.is_empty() {
                continue;
            }

            // Set obs_dim from first non-empty rollout
            if obs_dim.is_none() {
                obs_dim = Some(rollout[0].state().len());
            }

            // Use raw rewards for GAE computation
            // Return normalization is applied AFTER GAE, not before
            let rewards: Vec<f32> = rollout.iter().map(|t| t.reward()).collect();
            let values: Vec<f32> = rollout.iter().map(|t| t.value).collect();
            let dones: Vec<bool> = rollout.iter().map(|t| t.done()).collect();
            let terminals: Vec<bool> = rollout.iter().map(|t| t.terminal()).collect();

            // Bootstrap value determination:
            // 1. TRUE terminal: bootstrap = 0
            // 2. Truncated or ongoing: use precomputed V(s_{t+1})
            // 3. Fallback: use V(s_t)
            let last_transition = rollout.last().unwrap();
            let bootstrap = if terminals.last().copied().unwrap_or(true) {
                0.0
            } else if let Some(bv) = last_transition.bootstrap_value {
                bv
            } else {
                values.last().copied().unwrap_or(0.0)
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
                all_states.extend(t.state());

                // Convert action from transition format
                let action = match t.action() {
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

        let n_transitions = all_actions.len();
        let obs_dim = obs_dim.unwrap_or(0);

        // Normalize returns by return std if enabled
        // This scales value function targets to have consistent magnitude
        let returns = if let Some(normalizer) = reward_normalizer {
            normalizer.normalize_returns(&all_returns)
        } else {
            all_returns
        };

        // Normalize advantages if configured
        let advantages = if config.normalize_advantages {
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
            dones: Vec::new(),     // Feed-forward doesn't use dones for hidden reset
            terminals: Vec::new(), // Feed-forward doesn't use terminals for hidden reset
            obs_dim,
            n_transitions,
            sequence_info: None, // Feed-forward doesn't use sequences
        }
    }

    fn train_epoch<M, O>(
        mut model: M,
        optimizer: &mut O,
        data: &PreparedTrainingData<A::Action>,
        config: &PPOConfig,
        _hidden_config: &HiddenConfig,
        device: &B::Device,
        current_env_steps: usize,
        initial_lr: f64,
    ) -> (M, bool)
    where
        M: AutodiffModule<B> + ActorCritic<B, A, FeedForward>,
        O: Optimizer<M, B>,
    {
        if data.n_transitions == 0 {
            return (model, false);
        }

        let n_transitions = data.n_transitions;
        let obs_dim = data.obs_dim;

        // Create and shuffle minibatch indices
        // Shuffling is critical: reduces correlation, improves gradient quality
        let mut indices: Vec<usize> = (0..n_transitions).collect();
        indices.shuffle(&mut rand::thread_rng());

        let minibatch_size = config.minibatch_size();
        let mut kl_early_stop = false;

        for batch_start in (0..n_transitions).step_by(minibatch_size) {
            if kl_early_stop {
                break;
            }

            let batch_end = (batch_start + minibatch_size).min(n_transitions);
            let batch_indices = &indices[batch_start..batch_end];
            let batch_size = batch_indices.len();

            // Extract minibatch data
            let batch_states: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| &data.states[i * obs_dim..(i + 1) * obs_dim])
                .copied()
                .collect();
            let batch_actions: Vec<A::Action> = batch_indices
                .iter()
                .map(|&i| data.actions[i].clone())
                .collect();
            let batch_old_log_probs: Vec<f32> = batch_indices
                .iter()
                .map(|&i| data.old_log_probs[i])
                .collect();
            let batch_advantages: Vec<f32> = batch_indices
                .iter()
                .map(|&i| data.advantages[i])
                .collect();
            let batch_returns: Vec<f32> = batch_indices
                .iter()
                .map(|&i| data.returns[i])
                .collect();
            let batch_old_values: Vec<f32> = batch_indices
                .iter()
                .map(|&i| data.old_values[i])
                .collect();

            // Forward pass
            let obs_tensor = Tensor::<B, 1>::from_floats(batch_states.as_slice(), device)
                .reshape([batch_size, obs_dim]);
            let hidden = model.initial_hidden(batch_size, device);
            let output = model.forward(obs_tensor, hidden);

            // Compute log probs and entropy
            let log_probs = output.policy.log_prob(&batch_actions, device);
            let entropy = output.policy.entropy();
            let values = output.values_flat();

            // PPO loss computation
            let old_log_probs_tensor =
                Tensor::<B, 1>::from_floats(batch_old_log_probs.as_slice(), device);
            let advantages_tensor =
                Tensor::<B, 1>::from_floats(batch_advantages.as_slice(), device);
            let returns_tensor =
                Tensor::<B, 1>::from_floats(batch_returns.as_slice(), device);
            let old_values_tensor =
                Tensor::<B, 1>::from_floats(batch_old_values.as_slice(), device);

            // Ratio and log ratio for KL computation
            let log_ratio = log_probs.clone() - old_log_probs_tensor;
            let ratio = log_ratio.clone().exp();

            // Approximate KL divergence
            let approx_kl = if config.target_kl.is_some() {
                let kl_tensor = (ratio.clone() - 1.0) - log_ratio;
                let kl_val = kl_tensor.mean().into_data().as_slice::<f32>().unwrap()[0];
                Some(kl_val as f64)
            } else {
                None
            };

            // Clipped surrogate objective
            let surr1 = ratio.clone() * advantages_tensor.clone();
            let surr2 = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                * advantages_tensor;
            let policy_loss = -surr1.clone().min_pair(surr2).mean();

            // Value loss (with optional clipping)
            let value_loss = if config.clip_vloss {
                let v_loss_unclipped = (values.clone() - returns_tensor.clone()).powf_scalar(2.0);
                let v_clipped = old_values_tensor.clone()
                    + (values - old_values_tensor).clamp(-config.clip_ratio, config.clip_ratio);
                let v_loss_clipped = (v_clipped - returns_tensor).powf_scalar(2.0);
                v_loss_unclipped.max_pair(v_loss_clipped).mean().mul_scalar(0.5)
            } else {
                (values - returns_tensor).powf_scalar(2.0).mean()
            };

            // Entropy bonus
            let entropy_loss = -entropy.mean();

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
                        kl, target
                    );
                    kl_early_stop = true;
                }
            }
        }

        (model, kl_early_stop)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_ff_transition(env_id: usize, step: usize, reward: f32) -> PPOTransition<()> {
        PPOTransition::feed_forward(
            Transition::new_discrete(
                vec![env_id as f32, step as f32],
                0,
                reward,
                vec![env_id as f32, (step + 1) as f32],
                false,
                false,
            ),
            -0.5,
            1.0,
            None,
        )
    }

    fn make_terminal_transition(env_id: usize, step: usize) -> PPOTransition<()> {
        PPOTransition::feed_forward(
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
        )
    }

    #[test]
    fn test_create_transition() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        let base = Transition::new_discrete(vec![1.0, 2.0], 0, 1.0, vec![2.0, 3.0], false, false);

        let t = <FeedForwardStrategy as PPOTrainingStrategy<B, A, FeedForward>>::create_transition(
            base.clone(),
            -0.5,
            1.0,
            Some(0.9),
            vec![], // empty hidden for feed-forward
            0,
            0,
            false,
        );

        assert_eq!(t.state(), &[1.0, 2.0]);
        assert_eq!(t.log_prob, -0.5);
        assert_eq!(t.value, 1.0);
        assert_eq!(t.bootstrap_value, Some(0.9));
        assert_eq!(t.sequence_id, 0);
        assert_eq!(t.step_in_sequence, 0);
        assert!(!t.is_sequence_start);
    }

    #[test]
    fn test_prepare_training_data_basic() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        // Create simple rollouts
        let rollouts = vec![
            vec![
                make_ff_transition(0, 0, 1.0),
                make_ff_transition(0, 1, 1.0),
                make_terminal_transition(0, 2),
            ],
            vec![
                make_ff_transition(1, 0, 0.5),
                make_ff_transition(1, 1, 0.5),
            ],
        ];

        let config = PPOConfig::default();
        let hidden_config = HiddenConfig::none();

        let data = <FeedForwardStrategy as PPOTrainingStrategy<B, A, FeedForward>>::prepare_training_data(
            rollouts,
            &config,
            &hidden_config,
            None,
        );

        assert_eq!(data.n_transitions, 5);
        assert_eq!(data.obs_dim, 2);
        assert_eq!(data.actions.len(), 5);
        assert_eq!(data.advantages.len(), 5);
        assert_eq!(data.returns.len(), 5);
        assert!(data.sequence_info.is_none());
    }

    #[test]
    fn test_empty_rollouts() {
        use crate::algorithms::action_policy::DiscretePolicy;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray<f32>>;
        type A = DiscretePolicy;

        let rollouts: Vec<Vec<PPOTransition<()>>> = vec![vec![], vec![]];

        let config = PPOConfig::default();
        let hidden_config = HiddenConfig::none();

        let data = <FeedForwardStrategy as PPOTrainingStrategy<B, A, FeedForward>>::prepare_training_data(
            rollouts,
            &config,
            &hidden_config,
            None,
        );

        assert_eq!(data.n_transitions, 0);
    }
}
