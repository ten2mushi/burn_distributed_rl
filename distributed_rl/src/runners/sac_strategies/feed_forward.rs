//! Feed-forward SAC training strategy.
//!
//! This strategy implements standard SAC training for stateless policies:
//! - No hidden state management
//! - Standard batched training
//! - Separate actor/critic/alpha updates
//!
//! # Zero-Cost Abstraction
//!
//! Since `()` is a zero-sized type, `SACTransition<()>` has no hidden data overhead.
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
    EntropyTuner, SACActor, SACCritic, SACTransition,
    sac_actor_loss, sac_critic_loss, sac_td_targets,
};
use crate::algorithms::temporal_policy::{FeedForward, HiddenConfig};
use crate::core::target_network::TargetNetworkManager;
use crate::core::transition::Transition;
use crate::runners::sac_config::SACConfig;

use super::{SACLossInfo, SACTrainingStrategy};

// ============================================================================
// FeedForwardSACStrategy
// ============================================================================

/// Feed-forward SAC training strategy.
///
/// Implements standard SAC with:
/// - No hidden state (zero-cost `()` type)
/// - Batched training
/// - Separate network updates
///
/// For discrete action spaces, uses full expectation over all actions
/// via `PolicyOutput::action_probs()` and `PolicyOutput::all_log_probs()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedForwardSACStrategy;

impl<B, A> SACTrainingStrategy<B, A, FeedForward> for FeedForwardSACStrategy
where
    B: AutodiffBackend,
    A: ActionPolicy<B>,
{
    type TransitionData = ();

    fn create_transition(base: Transition) -> SACTransition<Self::TransitionData> {
        SACTransition::new(base)
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
        Actor: SACActor<B, A, FeedForward> + AutodiffModule<B> + Clone,
        Critic: SACCritic<B, A, FeedForward> + AutodiffModule<B> + Clone,
        ActorOpt: Optimizer<Actor, B>,
        CriticOpt: Optimizer<Critic, B>,
    {
        let batch_size = batch.len();
        if batch_size == 0 {
            return (actor, critic, target_critic, SACLossInfo::default());
        }

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

        // Get actions from batch for critic update
        let actions: Vec<A::Action> = batch
            .iter()
            .map(|t| match t.action() {
                crate::core::transition::Action::Discrete(a) => A::Action::from_floats(&[*a as f32]),
                crate::core::transition::Action::Continuous(a) => A::Action::from_floats(a),
            })
            .collect();

        // ====================================================================
        // CRITIC UPDATE
        // ====================================================================

        // 1. Get policy output for next states
        let hidden = actor.initial_hidden(batch_size, device);
        let next_actor_output = actor.forward(next_states_tensor.clone(), hidden);

        // 2. Get target Q-values and compute TD targets
        let target_hidden = target_critic.initial_hidden(batch_size, device);

        let (min_q_target, next_log_probs) = if is_discrete {
            // Discrete: compute full expectation over all actions
            // V(s') = Σ_a π(a|s') * (min_Q(s',a) - α*log π(a|s'))
            let target_output = target_critic.forward(next_states_tensor.clone(), None, target_hidden);

            // Get full distribution from policy
            let next_probs = next_actor_output.policy.action_probs()
                .expect("Discrete policy must implement action_probs()");
            let next_log_probs_all = next_actor_output.policy.all_log_probs()
                .expect("Discrete policy must implement all_log_probs()");

            // min_Q(s', a) for all actions
            let min_q_all: Tensor<B, 2> = target_output.q1.clone().min_pair(target_output.q2);

            // V(s') = Σ_a π(a|s') * (Q(s',a) - α*log π(a|s'))
            // = Σ_a π(a|s') * Q(s',a) - α * Σ_a π(a|s') * log π(a|s')
            // = E_a[Q] + α * H(π)
            let expected_q: Tensor<B, 2> = (next_probs.clone() * min_q_all).sum_dim(1);

            // Entropy term: H = -Σ_a π(a|s') * log π(a|s')
            let entropy: Tensor<B, 2> = -(next_probs * next_log_probs_all).sum_dim(1);

            // V(s') = E[Q] + α * H
            let v_next = expected_q.flatten(0, 1) + entropy.flatten(0, 1).mul_scalar(alpha);

            // For discrete, we incorporate entropy into V, so next_log_probs is zeros
            let zero_log_probs = Tensor::<B, 1>::zeros([batch_size], device);
            (v_next, zero_log_probs)
        } else {
            // Continuous: sample next actions and compute Q(s', a')
            let (next_actions, next_log_probs_vec) = next_actor_output.sample(device);
            let next_log_probs = Tensor::<B, 1>::from_floats(next_log_probs_vec.as_slice(), device);

            let next_actions_tensor = actions_to_tensor::<B, A>(&next_actions, device);
            let target_output = target_critic.forward(
                next_states_tensor.clone(),
                Some(next_actions_tensor),
                target_hidden,
            );
            let min_q: Tensor<B, 1> = target_output.q1.flatten(0, 1).min_pair(target_output.q2.flatten(0, 1));
            (min_q, next_log_probs)
        };

        // Compute TD targets: y = r + γ(1-d)(V(s') - α*log π) for continuous
        // For discrete, entropy is already in V(s'), so we pass zero log_probs
        let td_targets = sac_td_targets(
            rewards_tensor.clone(),
            terminals_tensor.clone(),
            min_q_target.clone(),
            next_log_probs.clone(),
            config.gamma,
            alpha,
        );

        // 3. Get current Q-values from online critic
        let critic_hidden = critic.initial_hidden(batch_size, device);
        let (q1_pred, q2_pred) = if is_discrete {
            // For discrete, gather Q-values at action indices
            let critic_output = critic.forward(states_tensor.clone(), None, critic_hidden);
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
                critic_hidden,
            );
            (critic_output.q1.flatten(0, 1), critic_output.q2.flatten(0, 1))
        };

        // 4. Compute critic loss
        let critic_loss = sac_critic_loss(q1_pred.clone(), q2_pred, td_targets.clone());
        let critic_loss_val = tensor_to_scalar(&critic_loss);

        // 5. Critic backward and step
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
            // 1. Get policy output for current states
            let actor_hidden = actor.initial_hidden(batch_size, device);
            let actor_output = actor.forward(states_tensor.clone(), actor_hidden);

            // 2. Compute actor loss based on action type
            let critic_hidden = critic.initial_hidden(batch_size, device);

            let (actor_loss, new_log_probs) = if is_discrete {
                // Discrete: J(π) = E_s[Σ_a π(a|s) * (α*log π(a|s) - Q(s,a))]
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

                // For alpha update, use mean entropy
                // H = -Σ_a π(a|s) * log π(a|s)
                let entropy_per_state: Tensor<B, 2> = -(probs * log_probs).sum_dim(1);
                let mean_ent: Tensor<B, 1> = entropy_per_state.flatten::<1>(0, 1).mean();

                // Return negative entropy as "log_probs" for alpha update
                // This ensures alpha increases when entropy is too low
                let neg_mean_entropy = -mean_ent.clone();

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

                // Actor loss = E[α*log π(a|s) - Q(s,a)]
                let loss = sac_actor_loss(min_q_new, new_log_probs.clone(), alpha);
                (loss, new_log_probs.mean())
            };

            actor_loss_val = tensor_to_scalar(&actor_loss);

            // 3. Compute entropy for logging
            let entropy = actor_output.entropy();
            mean_entropy = tensor_to_scalar(&entropy.mean());

            // 4. Actor backward and step
            let actor_grads = actor_loss.backward();
            let actor_grads = GradientsParams::from_grads(actor_grads, &actor);
            actor = actor_optimizer.step(config.actor_lr, actor, actor_grads);

            // ================================================================
            // ALPHA UPDATE (if auto-tuning)
            // ================================================================

            if config.auto_entropy_tuning {
                // For discrete: use mean negative entropy
                // For continuous: use mean log prob
                let log_prob_for_alpha = tensor_to_scalar(&new_log_probs);

                // Manual SGD step for log_alpha
                // L(α) = -α * (log π + H_target)
                let log_alpha = entropy_tuner.log_alpha_tensor();
                let log_alpha_data = log_alpha.clone().into_data();
                let log_alpha_val = log_alpha_data.as_slice::<f32>().unwrap()[0];

                let target_entropy = entropy_tuner.target_entropy();
                let grad = -(log_prob_for_alpha + target_entropy);

                let new_log_alpha_val = log_alpha_val - config.alpha_lr as f32 * grad;
                let new_log_alpha = Tensor::<B, 1>::from_floats([new_log_alpha_val], device);
                entropy_tuner.set_log_alpha(new_log_alpha);
                entropy_tuner.update_cache();

                alpha_loss_val = alpha * grad; // Approximate loss value
            }
        }

        // ====================================================================
        // TARGET NETWORK UPDATE
        // ====================================================================

        target_critic = target_manager.maybe_update(
            &critic,
            target_critic,
            device,
        );

        // Compute mean Q for logging
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

/// Convert actions to tensor for continuous action spaces.
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

/// Extract scalar from 1D tensor.
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

        // Use SACTransition::new directly since create_transition is just a wrapper
        let transition: SACTransition = SACTransition::new(base.clone());

        assert_eq!(transition.state(), &[1.0, 2.0]);
        assert_eq!(transition.reward(), 1.0);
        assert!(!transition.terminal());
    }

    #[test]
    fn test_actions_to_tensor() {
        use burn::backend::{Autodiff, NdArray};
        use crate::algorithms::action_policy::ContinuousAction;

        type B = Autodiff<NdArray<f32>>;

        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let actions = vec![
            ContinuousAction(vec![1.0, 2.0]),
            ContinuousAction(vec![3.0, 4.0]),
        ];

        let tensor = actions_to_tensor::<B, crate::algorithms::action_policy::ContinuousPolicy>(
            &actions,
            &device,
        );

        assert_eq!(tensor.dims(), [2, 2]);
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }
}
