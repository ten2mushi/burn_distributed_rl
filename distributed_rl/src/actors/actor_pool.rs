//! Multi-actor pool for distributed training.
//!
//! Coordinates multiple actors for parallel experience collection.
//!
//! # Architecture
//!
//! The ActorPool manages N actor threads, each collecting experience
//! from M vectorized environments. With WGPU backend, each actor
//! can perform model inference via its own CubeCL stream.

use super::actor::{ActorConfig, ActorHandle};
use crate::messages::{ActorMsg, ActorStats};
use crossbeam_channel::Sender;

/// Configuration for actor pool.
#[derive(Debug, Clone)]
pub struct ActorPoolConfig {
    /// Number of actors
    pub n_actors: usize,
    /// Environments per actor
    pub n_envs_per_actor: usize,
    /// Model update frequency (in steps)
    pub model_update_freq: usize,
    /// Number of actions
    pub n_actions: usize,
}

impl Default for ActorPoolConfig {
    fn default() -> Self {
        Self {
            n_actors: 4,
            n_envs_per_actor: 64,
            model_update_freq: 100,
            n_actions: 2,
        }
    }
}

impl ActorPoolConfig {
    /// Total environments across all actors.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    /// Create ActorConfig for a specific actor ID.
    pub fn actor_config(&self, actor_id: usize) -> ActorConfig {
        ActorConfig {
            actor_id,
            n_envs: self.n_envs_per_actor,
            model_update_freq: self.model_update_freq,
            n_actions: self.n_actions,
            epsilon: 0.01,
            use_exploration: true,
        }
    }
}

/// Multi-actor pool coordinator.
///
/// Manages multiple actor threads for parallel experience collection.
/// Each actor runs in its own thread with its own vectorized environment.
pub struct ActorPool<M: Send + 'static> {
    handles: Vec<ActorHandle>,
    cmd_txs: Vec<Sender<ActorMsg<M>>>,
    config: ActorPoolConfig,
}

impl<M: Clone + Send + 'static> ActorPool<M> {
    /// Create a new actor pool (actors not yet spawned).
    pub fn new(config: ActorPoolConfig) -> Self {
        Self {
            handles: Vec::with_capacity(config.n_actors),
            cmd_txs: Vec::with_capacity(config.n_actors),
            config,
        }
    }

    /// Add an actor handle to the pool.
    pub fn add_actor(&mut self, handle: ActorHandle, cmd_tx: Sender<ActorMsg<M>>) {
        self.handles.push(handle);
        self.cmd_txs.push(cmd_tx);
    }

    /// Get number of actors.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Check if pool is empty.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Get pool configuration.
    pub fn config(&self) -> &ActorPoolConfig {
        &self.config
    }

    /// Send stop command to all actors.
    pub fn stop_all(&self) {
        for tx in &self.cmd_txs {
            let _ = tx.try_send(ActorMsg::Stop);
        }
    }

    /// Send model update to all actors.
    pub fn update_all_models(&self, model: M) {
        for tx in &self.cmd_txs {
            let _ = tx.try_send(ActorMsg::UpdateModel(model.clone()));
        }
    }

    /// Set epsilon for all actors.
    pub fn set_epsilon(&self, epsilon: f32) {
        for tx in &self.cmd_txs {
            let _ = tx.try_send(ActorMsg::SetEpsilon(epsilon));
        }
    }

    /// Request stats from all actors.
    pub fn request_stats(&self) {
        for tx in &self.cmd_txs {
            let _ = tx.try_send(ActorMsg::RequestStats);
        }
    }

    /// Collect stats from all actors (non-blocking).
    pub fn collect_stats(&self) -> Vec<ActorStats> {
        let mut all_stats = Vec::new();
        for handle in &self.handles {
            while let Some(stats) = handle.get_stats() {
                all_stats.push(stats);
            }
        }
        all_stats
    }

    /// Get aggregated stats.
    pub fn aggregate_stats(&self) -> ActorStats {
        let stats = self.collect_stats();
        if stats.is_empty() {
            return ActorStats::default();
        }

        let total_steps: usize = stats.iter().map(|s| s.steps).sum();
        let total_episodes: usize = stats.iter().map(|s| s.episodes).sum();
        let total_reward: f32 = stats.iter().map(|s| s.avg_episode_reward * s.episodes as f32).sum();

        ActorStats {
            actor_id: 0,  // Aggregate
            steps: total_steps,
            episodes: total_episodes,
            avg_episode_reward: if total_episodes > 0 {
                total_reward / total_episodes as f32
            } else {
                0.0
            },
            recent_episode_reward: stats.last().map(|s| s.recent_episode_reward).unwrap_or(0.0),
            ..Default::default()
        }
    }

    /// Join all actor threads.
    pub fn join_all(self) -> Vec<std::thread::Result<()>> {
        self.stop_all();
        self.handles
            .into_iter()
            .map(|h| h.join())
            .collect()
    }

    /// Check if any actor threads have finished.
    pub fn any_finished(&self) -> bool {
        self.handles.iter().any(|h| h.thread.is_finished())
    }

    /// Get number of finished actors.
    pub fn finished_count(&self) -> usize {
        self.handles.iter().filter(|h| h.thread.is_finished()).count()
    }
}

/// Builder for creating actor pools.
pub struct ActorPoolBuilder {
    config: ActorPoolConfig,
}

impl ActorPoolBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: ActorPoolConfig::default(),
        }
    }

    /// Set number of actors.
    pub fn n_actors(mut self, n: usize) -> Self {
        self.config.n_actors = n;
        self
    }

    /// Set environments per actor.
    pub fn n_envs_per_actor(mut self, n: usize) -> Self {
        self.config.n_envs_per_actor = n;
        self
    }

    /// Set model update frequency.
    pub fn model_update_freq(mut self, freq: usize) -> Self {
        self.config.model_update_freq = freq;
        self
    }

    /// Set number of actions.
    pub fn n_actions(mut self, n: usize) -> Self {
        self.config.n_actions = n;
        self
    }

    /// Build config.
    pub fn build_config(self) -> ActorPoolConfig {
        self.config
    }

    /// Build the pool.
    pub fn build<M: Clone + Send + 'static>(self) -> ActorPool<M> {
        ActorPool::new(self.config)
    }
}

impl Default for ActorPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_pool_new() {
        let pool: ActorPool<()> = ActorPool::new(ActorPoolConfig::default());
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_actor_pool_config_default() {
        let config = ActorPoolConfig::default();
        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 64);
        assert_eq!(config.total_envs(), 256);
    }

    #[test]
    fn test_actor_pool_builder() {
        let config = ActorPoolBuilder::new()
            .n_actors(8)
            .n_envs_per_actor(32)
            .model_update_freq(50)
            .n_actions(4)
            .build_config();

        assert_eq!(config.n_actors, 8);
        assert_eq!(config.n_envs_per_actor, 32);
        assert_eq!(config.model_update_freq, 50);
        assert_eq!(config.n_actions, 4);
    }

    #[test]
    fn test_actor_config_from_pool_config() {
        let pool_config = ActorPoolConfig {
            n_actors: 4,
            n_envs_per_actor: 128,
            model_update_freq: 200,
            n_actions: 6,
        };

        let actor_config = pool_config.actor_config(2);
        assert_eq!(actor_config.actor_id, 2);
        assert_eq!(actor_config.n_envs, 128);
        assert_eq!(actor_config.model_update_freq, 200);
        assert_eq!(actor_config.n_actions, 6);
    }
}
