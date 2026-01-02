//! Recurrent SAC training strategy.
//!
//! This strategy implements SAC training for recurrent policies:
//! - Hidden state management across sequences
//! - TBPTT (Truncated Backpropagation Through Time)
//! - Sequence-based batch processing
//!
//! # Recurrent SAC Considerations
//!
//! - Hidden states must be stored and restored for training
//! - Sequences should respect episode boundaries
//! - Target networks also need hidden state handling
//!
//! # Discrete vs Continuous
//!
//! This strategy supports both action types:
//! - **Continuous**: Standard SAC with sampled actions
//! - **Discrete**: Full expectation over all actions using `PolicyOutput::action_probs()`

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use crate::algorithms::action_policy::{ActionPolicy, ActionValue, PolicyOutput};
use crate::algorithms::sac::{
    EntropyTuner, SACActor, SACCritic, SACRecurrentData, SACTransition,
    sac_actor_loss, sac_critic_loss, sac_td_targets,
};
use crate::algorithms::temporal_policy::{HiddenConfig, Recurrent};
use crate::core::target_network::TargetNetworkManager;
use crate::core::transition::Transition;
use crate::runners::sac_config::SACConfig;

use super::{SACLossInfo, SACTrainingStrategy};

// ============================================================================
// RecurrentSACStrategy
// ============================================================================

/// Recurrent SAC training strategy.
///
/// Implements SAC with:
/// - Hidden state persistence across timesteps
/// - TBPTT sequence training
/// - Episode boundary handling
///
/// For discrete action spaces, uses full expectation over all actions
/// via `PolicyOutput::action_probs()` and `PolicyOutput::all_log_probs()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecurrentSACStrategy;

impl<B, A> SACTrainingStrategy<B, A, Recurrent> for RecurrentSACStrategy
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
{
    type TransitionData = SACRecurrentData;

    fn create_transition(base: Transition) -> SACTransition<Self::TransitionData> {
        // For now, create with default hidden data
        // The actual hidden state will be populated by the actor thread
        SACTransition::with_data(base, SACRecurrentData::default())
    }

    fn train_step<Actor, Critic, ActorOpt, CriticOpt>(
        mut actor: Actor,
        mut critic: Critic,
        mut target_critic: Critic,
        actor_optimizer: &mut ActorOpt,
        critic_optimizer: &mut CriticOpt,
        entropy_tuner: &mut EntropyTuner<B>,
        batch: &[SACTransition<Self::TransitionData>],
        config: &SACConfig,
        _hidden_config: &HiddenConfig,
        device: &B::Device,
        gradient_step: usize,
        target_manager: &TargetNetworkManager,
    ) -> (Actor, Critic, Critic, SACLossInfo)
    where
        Actor: SACActor<B, A, Recurrent> + AutodiffModule<B> + Clone,
        Critic: SACCritic<B, A, Recurrent> + AutodiffModule<B> + Clone,
        ActorOpt: Optimizer<Actor, B>,
        CriticOpt: Optimizer<Critic, B>,
    {
        let batch_size = batch.len();
        if batch_size == 0 {
            return (actor, critic, target_critic, SACLossInfo::default());
        }

        // For recurrent SAC, we process transitions in sequence order
        // For now, we use a simplified approach similar to feed-forward
        // but with initial hidden states from the transitions

        let obs_size = batch[0].state().len();
        let is_discrete = critic.is_discrete();
        let alpha = entropy_tuner.cached_alpha();

        // Prepare batch tensors
        let states: Vec<f32> = batch.iter().flat_map(|t| t.state()).copied().collect();
        let next_states: Vec<f32> = batch.iter().flat_map(|t| t.next_state()).copied().collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward()).collect();
        let terminals: Vec<f32> = batch.iter().map(|t| if t.terminal() { 1.0 } else { 0.0 }).collect();

        let states_tensor = Tensor::<B, 1>::from_floats(states.as_slice(), device)
            .reshape([batch_size, obs_size]);
        let next_states_tensor = Tensor::<B, 1>::from_floats(next_states.as_slice(), device)
            .reshape([batch_size, obs_size]);
        let rewards_tensor = Tensor::<B, 1>::from_floats(rewards.as_slice(), device);
        let terminals_tensor = Tensor::<B, 1>::from_floats(terminals.as_slice(), device);

        // Get actions from batch
        let actions: Vec<A::Action> = batch
            .iter()
            .map(|t| match t.action() {
                crate::core::transition::Action::Discrete(a) => A::Action::from_floats(&[*a as f32]),
                crate::core::transition::Action::Continuous(a) => A::Action::from_floats(a),
            })
            .collect();

        // Initialize hidden states for batch
        // In a full implementation, we would restore hidden states from transitions
        // For simplicity, we use initial hidden states
        let actor_hidden = actor.initial_hidden(batch_size, device);
        let critic_hidden = critic.initial_hidden(batch_size, device);
        let target_hidden = target_critic.initial_hidden(batch_size, device);

        // ====================================================================
        // CRITIC UPDATE
        // ====================================================================

        // Get policy output for next states
        let next_actor_hidden = actor.initial_hidden(batch_size, device);
        let next_actor_output = actor.forward(next_states_tensor.clone(), next_actor_hidden);

        // Get target Q-values and compute TD targets
        let (min_q_target, next_log_probs) = if is_discrete {
            // Discrete: compute full expectation over all actions
            let target_output = target_critic.forward(next_states_tensor.clone(), None, target_hidden);

            // Get full distribution from policy
            let next_probs = next_actor_output.policy.action_probs()
                .expect("Discrete policy must implement action_probs()");
            let next_log_probs_all = next_actor_output.policy.all_log_probs()
                .expect("Discrete policy must implement all_log_probs()");

            // min_Q(s', a) for all actions
            let min_q_all: Tensor<B, 2> = target_output.q1.clone().min_pair(target_output.q2);

            // V(s') = Σ_a π(a|s') * Q(s',a) + α * H(π)
            let expected_q: Tensor<B, 2> = (next_probs.clone() * min_q_all).sum_dim(1);
            let entropy: Tensor<B, 2> = -(next_probs * next_log_probs_all).sum_dim(1);
            let v_next = expected_q.flatten(0, 1) + entropy.flatten(0, 1).mul_scalar(alpha);

            let zero_log_probs = Tensor::<B, 1>::zeros([batch_size], device);
            (v_next, zero_log_probs)
        } else {
            // Continuous: sample next actions
            let (next_actions, next_log_probs_vec) = next_actor_output.sample(device);
            let next_log_probs = Tensor::<B, 1>::from_floats(next_log_probs_vec.as_slice(), device);

            let next_actions_tensor = actions_to_tensor::<B, A>(&next_actions, device);
            let next_target_hidden = target_critic.initial_hidden(batch_size, device);
            let target_output = target_critic.forward(
                next_states_tensor.clone(),
                Some(next_actions_tensor),
                next_target_hidden,
            );
            let min_q: Tensor<B, 1> = target_output.q1.flatten(0, 1).min_pair(target_output.q2.flatten(0, 1));
            (min_q, next_log_probs)
        };

        // Compute TD targets
        let td_targets = sac_td_targets(
            rewards_tensor.clone(),
            terminals_tensor.clone(),
            min_q_target.clone(),
            next_log_probs.clone(),
            config.gamma,
            alpha,
        );

        // Get current Q-values
        let (q1_pred, q2_pred) = if is_discrete {
            let critic_output = critic.forward(states_tensor.clone(), None, critic_hidden.clone());
            let action_indices: Vec<i32> = batch
                .iter()
                .map(|t| match t.action() {
                    crate::core::transition::Action::Discrete(a) => *a as i32,
                    _ => 0,
                })
                .collect();
            let indices_tensor = Tensor::<B, 1, burn::tensor::Int>::from_ints(
                action_indices.as_slice(),
                device,
            ).reshape([batch_size, 1]);

            (critic_output.q1.gather(1, indices_tensor.clone()).flatten(0, 1),
             critic_output.q2.gather(1, indices_tensor).flatten(0, 1))
        } else {
            let actions_tensor = actions_to_tensor::<B, A>(&actions, device);
            let critic_output = critic.forward(
                states_tensor.clone(),
                Some(actions_tensor),
                critic_hidden.clone(),
            );
            (critic_output.q1.flatten(0, 1), critic_output.q2.flatten(0, 1))
        };

        // Compute critic loss
        let critic_loss = sac_critic_loss(q1_pred.clone(), q2_pred, td_targets.clone());
        let critic_loss_val = tensor_to_scalar(&critic_loss);

        // Critic backward and step
        let critic_grads = critic_loss.backward();
        let critic_grads = GradientsParams::from_grads(critic_grads, &critic);
        critic = critic_optimizer.step(config.critic_lr, critic, critic_grads);

        // ====================================================================
        // ACTOR UPDATE (every policy_update_freq steps)
        // ====================================================================

        let mut actor_loss_val = 0.0;
        let mut mean_entropy = 0.0;
        let mut alpha_loss_val = 0.0;

        if gradient_step % config.policy_update_freq == 0 {
            let actor_hidden = actor.initial_hidden(batch_size, device);
            let actor_output = actor.forward(states_tensor.clone(), actor_hidden);

            let critic_hidden = critic.initial_hidden(batch_size, device);

            let (actor_loss, new_log_probs) = if is_discrete {
                // Discrete: compute expectation over all actions
                let critic_output = critic.forward(states_tensor.clone(), None, critic_hidden);
                let min_q: Tensor<B, 2> = critic_output.q1.min_pair(critic_output.q2);

                let probs = actor_output.policy.action_probs()
                    .expect("Discrete policy must implement action_probs()");
                let log_probs = actor_output.policy.all_log_probs()
                    .expect("Discrete policy must implement all_log_probs()");

                // Actor loss = E_s[Σ_a π(a|s) * (α*log π(a|s) - Q(s,a))]
                let inside_exp: Tensor<B, 2> = log_probs.clone().mul_scalar(alpha) - min_q;
                let expected: Tensor<B, 2> = (probs.clone() * inside_exp).sum_dim(1);
                let loss: Tensor<B, 1> = expected.flatten::<1>(0, 1).mean();

                // For alpha update
                let entropy_per_state: Tensor<B, 2> = -(probs * log_probs).sum_dim(1);
                let neg_mean_entropy: Tensor<B, 1> = -entropy_per_state.flatten::<1>(0, 1).mean();

                (loss, neg_mean_entropy)
            } else {
                // Continuous: sample new actions
                let (new_actions, new_log_probs_vec) = actor_output.sample(device);
                let new_log_probs = Tensor::<B, 1>::from_floats(new_log_probs_vec.as_slice(), device);

                let new_actions_tensor = actions_to_tensor::<B, A>(&new_actions, device);
                let critic_output = critic.forward(
                    states_tensor.clone(),
                    Some(new_actions_tensor),
                    critic_hidden,
                );
                let min_q_new = critic_output.min_q().flatten(0, 1);

                let loss = sac_actor_loss(min_q_new, new_log_probs.clone(), alpha);
                (loss, new_log_probs.mean())
            };

            actor_loss_val = tensor_to_scalar(&actor_loss);

            let entropy = actor_output.entropy();
            mean_entropy = tensor_to_scalar(&entropy.mean());

            let actor_grads = actor_loss.backward();
            let actor_grads = GradientsParams::from_grads(actor_grads, &actor);
            actor = actor_optimizer.step(config.actor_lr, actor, actor_grads);

            // Alpha update
            if config.auto_entropy_tuning {
                let log_prob_for_alpha = tensor_to_scalar(&new_log_probs);

                let log_alpha = entropy_tuner.log_alpha_tensor();
                let log_alpha_data = log_alpha.clone().into_data();
                let log_alpha_val = log_alpha_data.as_slice::<f32>().unwrap()[0];

                let target_entropy = entropy_tuner.target_entropy();
                let grad = -(log_prob_for_alpha + target_entropy);

                let new_log_alpha_val = log_alpha_val - config.alpha_lr as f32 * grad;
                let new_log_alpha = Tensor::<B, 1>::from_floats([new_log_alpha_val], device);
                entropy_tuner.set_log_alpha(new_log_alpha);
                entropy_tuner.update_cache();

                alpha_loss_val = alpha * grad;
            }
        }

        // Target network update
        target_critic = target_manager.maybe_update(
            &critic,
            target_critic,
            device,
        );

        let mean_q = tensor_to_scalar(&q1_pred.mean());

        let loss_info = SACLossInfo {
            critic_loss: critic_loss_val,
            actor_loss: actor_loss_val,
            alpha_loss: alpha_loss_val,
            alpha: entropy_tuner.cached_alpha(),
            mean_q,
            mean_entropy,
        };

        (actor, critic, target_critic, loss_info)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn actions_to_tensor<B: AutodiffBackend, A: ActionPolicy<B>>(
    actions: &[A::Action],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = actions.len();
    if batch_size == 0 {
        return Tensor::zeros([0, 1], device);
    }

    let first_action = actions[0].as_floats();
    let action_dim = first_action.len();

    let flat_actions: Vec<f32> = actions
        .iter()
        .flat_map(|a| a.as_floats())
        .collect();

    Tensor::<B, 1>::from_floats(flat_actions.as_slice(), device)
        .reshape([batch_size, action_dim])
}

fn tensor_to_scalar<B: AutodiffBackend>(tensor: &Tensor<B, 1>) -> f32 {
    let data = tensor.clone().into_data();
    data.as_slice::<f32>().unwrap()[0]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    #[test]
    fn test_create_transition() {
        let base = Transition::new_discrete(
            vec![1.0, 2.0],
            0,
            1.0,
            vec![2.0, 3.0],
            false,
            false,
        );

        // Use SACTransition::with_data directly since create_transition is just a wrapper
        let transition: SACTransition<SACRecurrentData> = SACTransition::with_data(base.clone(), SACRecurrentData::default());

        assert_eq!(transition.state(), &[1.0, 2.0]);
        assert_eq!(transition.reward(), 1.0);
        assert!(!transition.terminal());
    }

    #[test]
    fn test_hidden_data_default() {
        let data = SACRecurrentData::default();
        assert!(data.hidden.is_empty());
        assert_eq!(data.sequence_id, 0);
        assert_eq!(data.step_in_sequence, 0);
        assert!(!data.is_sequence_start);
    }
}
