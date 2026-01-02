//! Generic actor for distributed training.
//!
//! Actors collect experience from environments and push to buffers.
//! Supports both PPO (on-policy) and IMPALA (off-policy) workflows.
//!
//! # WGPU Thread Safety
//!
//! With WGPU backend, CubeCL streams provide automatic cross-thread
//! tensor synchronization. No explicit TensorSync is needed.

use crate::core::model_slot::ModelSlot;
use crate::core::transition::{Action, IMPALATransition, PPOTransition, Trajectory, Transition};
use crate::messages::{ActorMsg, ActorStats};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;

/// Actor configuration.
#[derive(Debug, Clone)]
pub struct ActorConfig {
    /// Actor ID (for logging and exploration differentiation)
    pub actor_id: usize,
    /// Number of parallel environments
    pub n_envs: usize,
    /// Steps between model downloads
    pub model_update_freq: usize,
    /// Number of actions (for random action generation)
    pub n_actions: usize,
    /// Exploration rate (epsilon)
    pub epsilon: f32,
    /// Whether to use epsilon-greedy exploration
    pub use_exploration: bool,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            actor_id: 0,
            n_envs: 64,
            model_update_freq: 100,
            n_actions: 2,
            epsilon: 0.01,
            use_exploration: true,
        }
    }
}

impl ActorConfig {
    /// Create config for a specific actor ID.
    pub fn for_actor(actor_id: usize) -> Self {
        Self {
            actor_id,
            ..Default::default()
        }
    }

    /// Set number of environments.
    pub fn with_n_envs(mut self, n_envs: usize) -> Self {
        self.n_envs = n_envs;
        self
    }

    /// Set model update frequency.
    pub fn with_model_update_freq(mut self, freq: usize) -> Self {
        self.model_update_freq = freq;
        self
    }

    /// Set number of actions.
    pub fn with_n_actions(mut self, n_actions: usize) -> Self {
        self.n_actions = n_actions;
        self
    }

    /// Set epsilon for exploration.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Actor handle for controlling spawned actor thread.
pub struct ActorHandle {
    /// Thread handle
    pub thread: std::thread::JoinHandle<()>,
    /// Channel to receive stats from actor
    pub stats_rx: Receiver<ActorStats>,
    /// Channel to send commands to actor
    pub cmd_tx: Sender<ActorMsg<()>>,
}

impl ActorHandle {
    /// Send stop command to actor.
    pub fn stop(&self) {
        let _ = self.cmd_tx.try_send(ActorMsg::Stop);
    }

    /// Get latest stats (non-blocking).
    pub fn get_stats(&self) -> Option<ActorStats> {
        self.stats_rx.try_recv().ok()
    }

    /// Wait for actor thread to finish.
    pub fn join(self) -> std::thread::Result<()> {
        self.thread.join()
    }
}

/// Generic actor that collects experience.
///
/// Actors run in their own thread and:
/// 1. Step vectorized environments
/// 2. Query policy for actions
/// 3. Push transitions to shared buffer
/// 4. Periodically update model from ModelSlot
pub struct Actor {
    config: ActorConfig,
}

impl Actor {
    /// Create a new actor with given configuration.
    pub fn new(config: ActorConfig) -> Self {
        Self { config }
    }

    /// Spawn actor thread for PPO.
    ///
    /// Returns actions, log_probs, and values from policy.
    /// Pushes PPOTransitions to the rollout buffer.
    ///
    /// # Type Parameters
    /// - `M`: Model type (Clone + Send, NOT Sync - Burn constraint)
    ///
    /// # WGPU Note
    /// With WGPU backend, policy_fn can perform tensor operations
    /// directly without synchronization - CubeCL handles it.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ppo<M, FEnv, FPolicy, FUpdate, FPush>(
        self,
        mut env_step_fn: FEnv,
        mut policy_fn: FPolicy,
        mut update_model_fn: FUpdate,
        mut push_fn: FPush,
        model_slot: Arc<ModelSlot<M>>,
        cmd_rx: Receiver<ActorMsg<M>>,
    ) -> ActorHandle
    where
        M: Clone + Send + 'static,
        FEnv: FnMut(&[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<bool>) + Send + 'static,
        FPolicy: FnMut(&[Vec<f32>]) -> (Vec<u32>, Vec<f32>, Vec<f32>) + Send + 'static,
        FUpdate: FnMut(M) + Send + 'static,
        FPush: FnMut(Vec<PPOTransition>, u64) + Send + 'static,
    {
        let config = self.config;
        let (stats_tx, stats_rx) = crossbeam_channel::bounded(100);
        let (cmd_tx, cmd_rx_internal) = crossbeam_channel::bounded(100);

        // Forward external commands
        let cmd_rx_clone = cmd_rx;
        std::thread::spawn(move || {
            while let Ok(msg) = cmd_rx_clone.recv() {
                if let Err(_) = cmd_tx.send(msg) {
                    break;
                }
            }
        });

        let thread = std::thread::Builder::new()
            .name(format!("PPO-Actor-{}", config.actor_id))
            .spawn(move || {
                let mut step = 0usize;
                let mut episodes = 0usize;
                let mut episode_rewards = vec![0.0f32; config.n_envs];
                let mut total_episode_reward = 0.0f32;
                let mut recent_episode_reward = 0.0f32;
                let mut obs: Vec<Vec<f32>> = Vec::new();
                let mut initialized = false;
                let mut last_model_check = 0usize;
                let mut current_policy_version = 0u64;
                let mut epsilon = config.epsilon;

                loop {
                    // Check for commands
                    if let Ok(msg) = cmd_rx_internal.try_recv() {
                        match msg {
                            ActorMsg::Stop => break,
                            ActorMsg::UpdateModel(model) => {
                                update_model_fn(model);
                                current_policy_version = model_slot.version();
                            }
                            ActorMsg::SetEpsilon(eps) => {
                                epsilon = eps;
                            }
                            ActorMsg::RequestStats => {
                                let avg_reward = if episodes > 0 {
                                    total_episode_reward / episodes as f32
                                } else {
                                    0.0
                                };
                                let _ = stats_tx.try_send(ActorStats {
                                    actor_id: config.actor_id,
                                    steps: step,
                                    episodes,
                                    avg_episode_reward: avg_reward,
                                    recent_episode_reward,
                                    ..Default::default()
                                });
                            }
                        }
                    }

                    // Periodic model update from ModelSlot
                    if step >= last_model_check + config.model_update_freq {
                        if let Some(model) = model_slot.take() {
                            update_model_fn(model);
                            current_policy_version = model_slot.version();
                        }
                        last_model_check = step;
                    }

                    // Get policy output
                    // WGPU: No synchronization needed - CubeCL streams handle it
                    let (actions, log_probs, values) = if !initialized || obs.is_empty() {
                        // Random actions for first step
                        let random_actions: Vec<u32> = (0..config.n_envs)
                            .map(|_| fastrand::u32(0..config.n_actions as u32))
                            .collect();
                        let zero_log_probs = vec![0.0f32; config.n_envs];
                        let zero_values = vec![0.0f32; config.n_envs];
                        (random_actions, zero_log_probs, zero_values)
                    } else {
                        let (mut actions, log_probs, values) = policy_fn(&obs);

                        // Apply epsilon-greedy exploration
                        if config.use_exploration && epsilon > 0.0 {
                            for i in 0..actions.len() {
                                if fastrand::f32() < epsilon {
                                    actions[i] = fastrand::u32(0..config.n_actions as u32);
                                }
                            }
                        }

                        (actions, log_probs, values)
                    };

                    // Step environment
                    let (next_obs, rewards, terminals, truncated) = env_step_fn(&actions);

                    // Create transitions and push
                    if initialized && !obs.is_empty() && obs.len() == next_obs.len() {
                        let transitions: Vec<PPOTransition> = obs
                            .iter()
                            .zip(actions.iter())
                            .zip(log_probs.iter())
                            .zip(values.iter())
                            .zip(rewards.iter())
                            .zip(next_obs.iter())
                            .zip(terminals.iter())
                            .zip(truncated.iter())
                            .map(|(((((((s, &a), &lp), &v), &r), ns), &t), &tr)| {
                                PPOTransition {
                                    base: Transition {
                                        state: s.clone(),
                                        action: Action::Discrete(a),
                                        reward: r,
                                        next_state: ns.clone(),
                                        terminal: t,
                                        truncated: tr,
                                    },
                                    log_prob: lp,
                                    value: v,
                                    bootstrap_value: None,
                                }
                            })
                            .collect();

                        push_fn(transitions, current_policy_version);
                    }

                    // Track episode stats
                    for (i, (&r, (&t, &tr))) in rewards
                        .iter()
                        .zip(terminals.iter().zip(truncated.iter()))
                        .enumerate()
                    {
                        episode_rewards[i] += r;
                        if t || tr {
                            total_episode_reward += episode_rewards[i];
                            recent_episode_reward = episode_rewards[i];
                            episodes += 1;
                            episode_rewards[i] = 0.0;
                        }
                    }

                    obs = next_obs;
                    initialized = true;
                    step += config.n_envs;

                    // Report stats periodically
                    if step % 10000 == 0 {
                        let avg_reward = if episodes > 0 {
                            total_episode_reward / episodes as f32
                        } else {
                            0.0
                        };
                        let _ = stats_tx.try_send(ActorStats {
                            actor_id: config.actor_id,
                            steps: step,
                            episodes,
                            avg_episode_reward: avg_reward,
                            recent_episode_reward,
                            ..Default::default()
                        });
                    }
                }
            })
            .expect("Failed to spawn PPO actor thread");

        // Create a dummy cmd_tx for the handle
        let (handle_cmd_tx, _) = crossbeam_channel::bounded(1);

        ActorHandle {
            thread,
            stats_rx,
            cmd_tx: handle_cmd_tx,
        }
    }

    /// Spawn actor thread for IMPALA.
    ///
    /// Collects trajectories with behavior policy information for V-trace.
    ///
    /// # Type Parameters
    /// - `M`: Model type (Clone + Send, NOT Sync - Burn constraint)
    ///
    /// # WGPU Note
    /// With WGPU backend, policy_fn can perform tensor operations
    /// directly without synchronization - CubeCL handles it.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_impala<M, FEnv, FPolicy, FUpdate, FPush>(
        self,
        mut env_step_fn: FEnv,
        mut policy_fn: FPolicy,
        mut update_model_fn: FUpdate,
        mut push_fn: FPush,
        model_slot: Arc<ModelSlot<M>>,
        cmd_rx: Receiver<ActorMsg<M>>,
        trajectory_length: usize,
    ) -> ActorHandle
    where
        M: Clone + Send + 'static,
        FEnv: FnMut(&[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<bool>) + Send + 'static,
        FPolicy: FnMut(&[Vec<f32>]) -> (Vec<u32>, Vec<f32>) + Send + 'static,
        FUpdate: FnMut(M) + Send + 'static,
        FPush: FnMut(Vec<Trajectory<IMPALATransition>>) + Send + 'static,
    {
        let config = self.config;
        let (stats_tx, stats_rx) = crossbeam_channel::bounded(100);
        let (cmd_tx, cmd_rx_internal) = crossbeam_channel::bounded(100);

        // Forward external commands
        let cmd_rx_clone = cmd_rx;
        std::thread::spawn(move || {
            while let Ok(msg) = cmd_rx_clone.recv() {
                if let Err(_) = cmd_tx.send(msg) {
                    break;
                }
            }
        });

        let thread = std::thread::Builder::new()
            .name(format!("IMPALA-Actor-{}", config.actor_id))
            .spawn(move || {
                let mut step = 0usize;
                let mut episodes = 0usize;
                let mut episode_rewards = vec![0.0f32; config.n_envs];
                let mut total_episode_reward = 0.0f32;
                let mut recent_episode_reward = 0.0f32;
                let mut obs: Vec<Vec<f32>> = Vec::new();
                let mut initialized = false;
                let mut last_model_check = 0usize;
                let mut current_policy_version = 0u64;

                // Per-environment trajectories
                let mut trajectories: Vec<Trajectory<IMPALATransition>> = (0..config.n_envs)
                    .map(|i| Trajectory::with_capacity(i, trajectory_length))
                    .collect();

                loop {
                    // Check for commands
                    if let Ok(msg) = cmd_rx_internal.try_recv() {
                        match msg {
                            ActorMsg::Stop => break,
                            ActorMsg::UpdateModel(model) => {
                                update_model_fn(model);
                                current_policy_version = model_slot.version();
                            }
                            ActorMsg::SetEpsilon(_) => {
                                // IMPALA doesn't use epsilon-greedy
                            }
                            ActorMsg::RequestStats => {
                                let avg_reward = if episodes > 0 {
                                    total_episode_reward / episodes as f32
                                } else {
                                    0.0
                                };
                                let _ = stats_tx.try_send(ActorStats {
                                    actor_id: config.actor_id,
                                    steps: step,
                                    episodes,
                                    avg_episode_reward: avg_reward,
                                    recent_episode_reward,
                                    ..Default::default()
                                });
                            }
                        }
                    }

                    // Periodic model update from ModelSlot
                    if step >= last_model_check + config.model_update_freq {
                        if let Some(model) = model_slot.take() {
                            update_model_fn(model);
                            current_policy_version = model_slot.version();
                        }
                        last_model_check = step;
                    }

                    // Get policy output
                    // WGPU: No synchronization needed - CubeCL streams handle it
                    let (actions, log_probs) = if !initialized || obs.is_empty() {
                        let random_actions: Vec<u32> = (0..config.n_envs)
                            .map(|_| fastrand::u32(0..config.n_actions as u32))
                            .collect();
                        let zero_log_probs = vec![0.0f32; config.n_envs];
                        (random_actions, zero_log_probs)
                    } else {
                        policy_fn(&obs)
                    };

                    // Step environment
                    let (next_obs, rewards, terminals, truncated) = env_step_fn(&actions);

                    // Add transitions to trajectories
                    if initialized && !obs.is_empty() && obs.len() == next_obs.len() {
                        for i in 0..config.n_envs {
                            let transition = IMPALATransition {
                                base: Transition {
                                    state: obs[i].clone(),
                                    action: Action::Discrete(actions[i]),
                                    reward: rewards[i],
                                    next_state: next_obs[i].clone(),
                                    terminal: terminals[i],
                                    truncated: truncated[i],
                                },
                                behavior_log_prob: log_probs[i],
                                policy_version: current_policy_version,
                            };
                            trajectories[i].push(transition);
                        }
                    }

                    // Check for complete trajectories
                    let mut complete_trajectories = Vec::new();
                    for i in 0..config.n_envs {
                        let should_push = trajectories[i].len() >= trajectory_length
                            || terminals[i]
                            || truncated[i];

                        if should_push && !trajectories[i].is_empty() {
                            // Set episode return if complete
                            if terminals[i] || truncated[i] {
                                trajectories[i].episode_return = Some(episode_rewards[i] + rewards[i]);
                            }

                            let traj = std::mem::replace(
                                &mut trajectories[i],
                                Trajectory::with_capacity(i, trajectory_length),
                            );
                            complete_trajectories.push(traj);
                        }
                    }

                    if !complete_trajectories.is_empty() {
                        push_fn(complete_trajectories);
                    }

                    // Track episode stats
                    for (i, (&r, (&t, &tr))) in rewards
                        .iter()
                        .zip(terminals.iter().zip(truncated.iter()))
                        .enumerate()
                    {
                        episode_rewards[i] += r;
                        if t || tr {
                            total_episode_reward += episode_rewards[i];
                            recent_episode_reward = episode_rewards[i];
                            episodes += 1;
                            episode_rewards[i] = 0.0;
                        }
                    }

                    obs = next_obs;
                    initialized = true;
                    step += config.n_envs;

                    // Report stats periodically
                    if step % 10000 == 0 {
                        let avg_reward = if episodes > 0 {
                            total_episode_reward / episodes as f32
                        } else {
                            0.0
                        };
                        let _ = stats_tx.try_send(ActorStats {
                            actor_id: config.actor_id,
                            steps: step,
                            episodes,
                            avg_episode_reward: avg_reward,
                            recent_episode_reward,
                            ..Default::default()
                        });
                    }
                }
            })
            .expect("Failed to spawn IMPALA actor thread");

        // Create a dummy cmd_tx for the handle
        let (handle_cmd_tx, _) = crossbeam_channel::bounded(1);

        ActorHandle {
            thread,
            stats_rx,
            cmd_tx: handle_cmd_tx,
        }
    }

    /// Spawn actor thread for continuous action spaces (PPO).
    ///
    /// Similar to spawn_ppo but handles continuous actions.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ppo_continuous<M, FEnv, FPolicy, FUpdate, FPush>(
        self,
        mut env_step_fn: FEnv,
        mut policy_fn: FPolicy,
        mut update_model_fn: FUpdate,
        mut push_fn: FPush,
        model_slot: Arc<ModelSlot<M>>,
        cmd_rx: Receiver<ActorMsg<M>>,
        action_dim: usize,
    ) -> ActorHandle
    where
        M: Clone + Send + 'static,
        FEnv: FnMut(&[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<bool>) + Send + 'static,
        FPolicy: FnMut(&[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) + Send + 'static,
        FUpdate: FnMut(M) + Send + 'static,
        FPush: FnMut(Vec<PPOTransition>, u64) + Send + 'static,
    {
        let config = self.config;
        let (stats_tx, stats_rx) = crossbeam_channel::bounded(100);
        let (cmd_tx, cmd_rx_internal) = crossbeam_channel::bounded(100);

        // Forward external commands
        let cmd_rx_clone = cmd_rx;
        std::thread::spawn(move || {
            while let Ok(msg) = cmd_rx_clone.recv() {
                if let Err(_) = cmd_tx.send(msg) {
                    break;
                }
            }
        });

        let thread = std::thread::Builder::new()
            .name(format!("PPO-Continuous-Actor-{}", config.actor_id))
            .spawn(move || {
                let mut step = 0usize;
                let mut episodes = 0usize;
                let mut episode_rewards = vec![0.0f32; config.n_envs];
                let mut total_episode_reward = 0.0f32;
                let mut recent_episode_reward = 0.0f32;
                let mut obs: Vec<Vec<f32>> = Vec::new();
                let mut initialized = false;
                let mut last_model_check = 0usize;
                let mut current_policy_version = 0u64;

                loop {
                    // Check for commands
                    if let Ok(msg) = cmd_rx_internal.try_recv() {
                        match msg {
                            ActorMsg::Stop => break,
                            ActorMsg::UpdateModel(model) => {
                                update_model_fn(model);
                                current_policy_version = model_slot.version();
                            }
                            ActorMsg::SetEpsilon(_) => {}
                            ActorMsg::RequestStats => {
                                let avg_reward = if episodes > 0 {
                                    total_episode_reward / episodes as f32
                                } else {
                                    0.0
                                };
                                let _ = stats_tx.try_send(ActorStats {
                                    actor_id: config.actor_id,
                                    steps: step,
                                    episodes,
                                    avg_episode_reward: avg_reward,
                                    recent_episode_reward,
                                    ..Default::default()
                                });
                            }
                        }
                    }

                    // Periodic model update
                    if step >= last_model_check + config.model_update_freq {
                        if let Some(model) = model_slot.take() {
                            update_model_fn(model);
                            current_policy_version = model_slot.version();
                        }
                        last_model_check = step;
                    }

                    // Get policy output
                    let (actions, log_probs, values) = if !initialized || obs.is_empty() {
                        // Random continuous actions
                        let random_actions: Vec<Vec<f32>> = (0..config.n_envs)
                            .map(|_| (0..action_dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect())
                            .collect();
                        let zero_log_probs = vec![0.0f32; config.n_envs];
                        let zero_values = vec![0.0f32; config.n_envs];
                        (random_actions, zero_log_probs, zero_values)
                    } else {
                        policy_fn(&obs)
                    };

                    // Step environment
                    let (next_obs, rewards, terminals, truncated) = env_step_fn(&actions);

                    // Create transitions and push
                    if initialized && !obs.is_empty() && obs.len() == next_obs.len() {
                        let transitions: Vec<PPOTransition> = obs
                            .iter()
                            .zip(actions.iter())
                            .zip(log_probs.iter())
                            .zip(values.iter())
                            .zip(rewards.iter())
                            .zip(next_obs.iter())
                            .zip(terminals.iter())
                            .zip(truncated.iter())
                            .map(|(((((((s, a), &lp), &v), &r), ns), &t), &tr)| {
                                PPOTransition {
                                    base: Transition {
                                        state: s.clone(),
                                        action: Action::Continuous(a.clone()),
                                        reward: r,
                                        next_state: ns.clone(),
                                        terminal: t,
                                        truncated: tr,
                                    },
                                    log_prob: lp,
                                    value: v,
                                    bootstrap_value: None,
                                }
                            })
                            .collect();

                        push_fn(transitions, current_policy_version);
                    }

                    // Track episode stats
                    for (i, (&r, (&t, &tr))) in rewards
                        .iter()
                        .zip(terminals.iter().zip(truncated.iter()))
                        .enumerate()
                    {
                        episode_rewards[i] += r;
                        if t || tr {
                            total_episode_reward += episode_rewards[i];
                            recent_episode_reward = episode_rewards[i];
                            episodes += 1;
                            episode_rewards[i] = 0.0;
                        }
                    }

                    obs = next_obs;
                    initialized = true;
                    step += config.n_envs;

                    // Report stats periodically
                    if step % 10000 == 0 {
                        let avg_reward = if episodes > 0 {
                            total_episode_reward / episodes as f32
                        } else {
                            0.0
                        };
                        let _ = stats_tx.try_send(ActorStats {
                            actor_id: config.actor_id,
                            steps: step,
                            episodes,
                            avg_episode_reward: avg_reward,
                            recent_episode_reward,
                            ..Default::default()
                        });
                    }
                }
            })
            .expect("Failed to spawn continuous PPO actor thread");

        let (handle_cmd_tx, _) = crossbeam_channel::bounded(1);

        ActorHandle {
            thread,
            stats_rx,
            cmd_tx: handle_cmd_tx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_config_default() {
        let config = ActorConfig::default();
        assert_eq!(config.actor_id, 0);
        assert_eq!(config.n_envs, 64);
        assert_eq!(config.model_update_freq, 100);
    }

    #[test]
    fn test_actor_config_builder() {
        let config = ActorConfig::for_actor(5)
            .with_n_envs(128)
            .with_model_update_freq(50)
            .with_n_actions(4)
            .with_epsilon(0.1);

        assert_eq!(config.actor_id, 5);
        assert_eq!(config.n_envs, 128);
        assert_eq!(config.model_update_freq, 50);
        assert_eq!(config.n_actions, 4);
        assert_eq!(config.epsilon, 0.1);
    }
}
