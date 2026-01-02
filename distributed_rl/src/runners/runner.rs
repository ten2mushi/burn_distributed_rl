//! Training runner that coordinates actors and learner.
//!
//! The Runner is the main entry point for distributed RL training.
//! It manages:
//! - N actor threads collecting experience
//! - 1 learner thread performing gradient updates
//! - Shared buffer for experience transfer
//! - ModelSlot for model synchronization
//!
//! # WGPU Thread Safety
//!
//! With WGPU backend, CubeCL streams provide automatic cross-thread
//! tensor synchronization. No explicit synchronization needed.

use crate::actors::ActorPoolConfig;
use crate::core::model_slot::ModelSlot;
use crate::learner::{Learner, LearnerConfig};
use crate::messages::{LearnerMsg, LearnerStats};
use crossbeam_channel::{bounded, Receiver};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for distributed training runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Number of actor threads
    pub n_actors: usize,
    /// Environments per actor
    pub n_envs_per_actor: usize,
    /// Maximum total environment steps
    pub max_env_steps: usize,
    /// Maximum training steps
    pub max_train_steps: usize,
    /// Target reward for early stopping
    pub target_reward: Option<f32>,
    /// Model publish frequency (learner steps)
    pub publish_freq: usize,
    /// Stats logging frequency (seconds)
    pub log_interval_secs: f32,
    /// Number of actions
    pub n_actions: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            n_actors: 4,
            n_envs_per_actor: 64,
            max_env_steps: 10_000_000,
            max_train_steps: 100_000,
            target_reward: None,
            publish_freq: 10,
            log_interval_secs: 5.0,
            n_actions: 2,
        }
    }
}

impl RunnerConfig {
    /// Total environments across all actors.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    /// Create actor pool config.
    pub fn actor_pool_config(&self) -> ActorPoolConfig {
        ActorPoolConfig {
            n_actors: self.n_actors,
            n_envs_per_actor: self.n_envs_per_actor,
            model_update_freq: 100,
            n_actions: self.n_actions,
        }
    }

    /// Create learner config.
    pub fn learner_config(&self) -> LearnerConfig {
        LearnerConfig {
            publish_freq: self.publish_freq,
            stats_freq: 100,
            eval_freq: 1000,
            max_train_steps: self.max_train_steps,
        }
    }
}

/// Training statistics.
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total environment steps across all actors
    pub env_steps: usize,
    /// Total training steps
    pub train_steps: usize,
    /// Total episodes completed
    pub episodes: usize,
    /// Average episode reward
    pub avg_reward: f32,
    /// Recent episode reward
    pub recent_reward: f32,
    /// Average training loss
    pub avg_loss: f32,
    /// Environment steps per second
    pub env_steps_per_sec: f32,
    /// Training steps per second
    pub train_steps_per_sec: f32,
    /// Current policy version
    pub policy_version: u64,
}

/// Distributed training runner.
///
/// Coordinates multiple actor threads and a single learner thread
/// for distributed RL training.
pub struct Runner {
    config: RunnerConfig,
    shutdown: Arc<AtomicBool>,
    policy_version: Arc<AtomicU64>,
}

impl Runner {
    /// Create a new distributed runner.
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            policy_version: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get the shutdown flag.
    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutdown)
    }

    /// Get the policy version counter.
    pub fn policy_version(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.policy_version)
    }

    /// Signal shutdown.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Check if shutdown has been signaled.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get configuration.
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    /// Run training loop with provided closures.
    ///
    /// This is a simplified entry point that runs the main coordination loop.
    /// For full control, use spawn_actors and spawn_learner separately.
    #[allow(clippy::too_many_arguments)]
    pub fn run<M, B, FEnvFactory, FPolicyFactory, FTrain, FGetModel>(
        &self,
        _env_factory: FEnvFactory,
        _policy_factory: FPolicyFactory,
        train_fn: FTrain,
        get_model_fn: FGetModel,
        model_slot: Arc<ModelSlot<M>>,
        ready_rx: Receiver<()>,
    ) -> TrainingStats
    where
        M: Clone + Send + 'static,
        B: burn::tensor::backend::AutodiffBackend,
        FEnvFactory: Fn(usize) -> Box<dyn FnMut(&[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<bool>) + Send> + Send + Sync,
        FPolicyFactory: Fn(usize) -> Box<dyn FnMut(&[Vec<f32>]) -> (Vec<u32>, Vec<f32>, Vec<f32>) + Send> + Send + Sync,
        FTrain: FnMut() -> (f32, f32, f32, f32) + Send + 'static,
        FGetModel: FnMut() -> M + Send + 'static,
    {
        let start_time = Instant::now();
        let mut stats = TrainingStats::default();
        let mut last_log_time = Instant::now();

        // Create channels
        let (learner_cmd_tx, learner_cmd_rx) = bounded::<LearnerMsg<M>>(100);
        let (_stats_tx, _stats_rx) = bounded::<LearnerStats>(100);

        // Create shared state
        let shutdown = Arc::clone(&self.shutdown);
        let version = Arc::clone(&self.policy_version);

        // Spawn learner
        let learner_config = self.config.learner_config();
        let learner = Learner::new(learner_config);

        let learner_shutdown = Arc::clone(&shutdown);
        let learner_model_slot = Arc::clone(&model_slot);
        let learner_handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            |_model| { /* publish callback */ },
            learner_model_slot,
            learner_cmd_rx,
            ready_rx,
            learner_shutdown,
        );

        // Main coordination loop
        while !self.is_shutdown() {
            // Check training limits
            if stats.env_steps >= self.config.max_env_steps {
                println!("[Runner] Reached max env steps: {}", self.config.max_env_steps);
                break;
            }

            if stats.train_steps >= self.config.max_train_steps {
                println!("[Runner] Reached max train steps: {}", self.config.max_train_steps);
                break;
            }

            // Check target reward
            if let Some(target) = self.config.target_reward {
                if stats.avg_reward >= target {
                    println!("[Runner] Reached target reward: {:.2} >= {:.2}", stats.avg_reward, target);
                    break;
                }
            }

            // Collect learner stats
            if let Some(learner_stats) = learner_handle.get_stats() {
                stats.train_steps = learner_stats.train_steps;
                stats.avg_loss = learner_stats.avg_loss;
                stats.train_steps_per_sec = learner_stats.steps_per_second;
            }

            // Log periodically
            let elapsed_since_log = last_log_time.elapsed().as_secs_f32();
            if elapsed_since_log >= self.config.log_interval_secs {
                let total_elapsed = start_time.elapsed().as_secs_f32();
                stats.env_steps_per_sec = stats.env_steps as f32 / total_elapsed;
                stats.policy_version = version.load(Ordering::Relaxed);

                println!(
                    "[Runner] Steps: {:>8} | Train: {:>6} | Reward: {:>7.2} | Loss: {:>6.4} | SPS: {:>6.0}",
                    stats.env_steps,
                    stats.train_steps,
                    stats.avg_reward,
                    stats.avg_loss,
                    stats.env_steps_per_sec,
                );

                last_log_time = Instant::now();
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        // Shutdown
        self.shutdown();
        let _ = learner_cmd_tx.try_send(LearnerMsg::Stop);
        let _ = learner_handle.join();

        stats
    }
}

/// Builder for Runner.
pub struct RunnerBuilder {
    config: RunnerConfig,
}

impl RunnerBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: RunnerConfig::default(),
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

    /// Set maximum environment steps.
    pub fn max_env_steps(mut self, steps: usize) -> Self {
        self.config.max_env_steps = steps;
        self
    }

    /// Set maximum training steps.
    pub fn max_train_steps(mut self, steps: usize) -> Self {
        self.config.max_train_steps = steps;
        self
    }

    /// Set target reward.
    pub fn target_reward(mut self, reward: f32) -> Self {
        self.config.target_reward = Some(reward);
        self
    }

    /// Set model publish frequency.
    pub fn publish_freq(mut self, freq: usize) -> Self {
        self.config.publish_freq = freq;
        self
    }

    /// Set log interval.
    pub fn log_interval_secs(mut self, secs: f32) -> Self {
        self.config.log_interval_secs = secs;
        self
    }

    /// Set number of actions.
    pub fn n_actions(mut self, n: usize) -> Self {
        self.config.n_actions = n;
        self
    }

    /// Build the runner.
    pub fn build(self) -> Runner {
        Runner::new(self.config)
    }
}

impl Default for RunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_runner_config() {
        let config = RunnerConfig {
            n_actors: 8,
            n_envs_per_actor: 32,
            max_env_steps: 5_000_000,
            ..Default::default()
        };

        assert_eq!(config.total_envs(), 256);
    }

    #[test]
    fn test_distributed_runner_builder() {
        let runner = RunnerBuilder::new()
            .n_actors(4)
            .n_envs_per_actor(64)
            .max_train_steps(50000)
            .target_reward(450.0)
            .build();

        assert_eq!(runner.config().n_actors, 4);
        assert_eq!(runner.config().n_envs_per_actor, 64);
        assert_eq!(runner.config().max_train_steps, 50000);
        assert_eq!(runner.config().target_reward, Some(450.0));
    }

    #[test]
    fn test_shutdown_flag() {
        let runner = Runner::new(RunnerConfig::default());
        assert!(!runner.is_shutdown());
        runner.shutdown();
        assert!(runner.is_shutdown());
    }
}
