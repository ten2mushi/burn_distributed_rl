//! Learner component for gradient updates.
//!
//! The learner consumes experience from buffers and performs gradient updates.
//! Model updates are published via ModelSlot to actors.
//!
//! # WGPU Thread Safety
//!
//! With WGPU backend, the learner can perform tensor operations directly
//! without synchronization. CubeCL streams provide automatic cross-thread
//! tensor synchronization.
//!
//! # Thread Safety and Lifecycle
//!
//! Each spawned learner creates:
//! 1. A main learner thread that processes training
//! 2. A command forwarding thread that bridges external commands
//!
//! Both threads respect the shutdown flag and terminate cleanly when:
//! - The shutdown flag is set to true
//! - The stop command is received via LearnerHandle
//! - max_train_steps is reached (if configured)

use crate::core::model_slot::ModelSlot;
use crate::messages::{LearnerMsg, LearnerStats};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Learner configuration.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Model publish frequency (in train steps)
    pub publish_freq: usize,
    /// Stats reporting frequency (in train steps)
    pub stats_freq: usize,
    /// Evaluation frequency (in train steps)
    pub eval_freq: usize,
    /// Maximum training steps (0 = unlimited)
    pub max_train_steps: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            publish_freq: 10,
            stats_freq: 100,
            eval_freq: 1000,
            max_train_steps: 0,
        }
    }
}

impl LearnerConfig {
    /// Create config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model publish frequency.
    ///
    /// # Panics
    ///
    /// Panics if `freq` is 0. Use a positive value to publish every N steps.
    pub fn with_publish_freq(mut self, freq: usize) -> Self {
        assert!(freq > 0, "publish_freq must be > 0 to avoid division by zero");
        self.publish_freq = freq;
        self
    }

    /// Set stats reporting frequency.
    ///
    /// # Panics
    ///
    /// Panics if `freq` is 0. Use a positive value to report stats every N steps.
    pub fn with_stats_freq(mut self, freq: usize) -> Self {
        assert!(freq > 0, "stats_freq must be > 0 to avoid division by zero");
        self.stats_freq = freq;
        self
    }

    /// Set evaluation frequency.
    ///
    /// # Panics
    ///
    /// Panics if `freq` is 0. Use a positive value to evaluate every N steps.
    pub fn with_eval_freq(mut self, freq: usize) -> Self {
        assert!(freq > 0, "eval_freq must be > 0 to avoid division by zero");
        self.eval_freq = freq;
        self
    }

    /// Set maximum training steps.
    ///
    /// Use 0 for unlimited training (will run until stop command or shutdown).
    pub fn with_max_train_steps(mut self, steps: usize) -> Self {
        self.max_train_steps = steps;
        self
    }

    /// Validate the configuration and return any issues.
    ///
    /// Returns `Ok(())` if valid, or an error message describing the issue.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.publish_freq == 0 {
            return Err("publish_freq must be > 0");
        }
        if self.stats_freq == 0 {
            return Err("stats_freq must be > 0");
        }
        if self.eval_freq == 0 {
            return Err("eval_freq must be > 0");
        }
        Ok(())
    }
}

/// Learner handle for controlling spawned learner thread.
///
/// This handle provides methods to:
/// - Send commands to the learner (stop, request stats, etc.)
/// - Receive stats updates from the learner
/// - Wait for the learner thread to finish
///
/// # Command Delivery
///
/// Commands are delivered through a properly connected channel.
/// The `stop()` method sends a Stop command that the learner will
/// process on its next command check (typically within 100ms).
pub struct LearnerHandle {
    /// Thread handle for the main learner thread
    pub thread: std::thread::JoinHandle<()>,
    /// Channel to receive stats from learner
    pub stats_rx: Receiver<LearnerStats>,
    /// Channel to send commands to learner - properly connected to internal channel
    pub cmd_tx: Sender<LearnerMsg<()>>,
    /// Counter for dropped stats (for diagnostics)
    pub dropped_stats: Arc<AtomicUsize>,
}

impl LearnerHandle {
    /// Send stop command to learner.
    ///
    /// This is a non-blocking operation. The learner will process the stop
    /// command on its next loop iteration (typically within 100ms).
    ///
    /// Returns `true` if the command was sent successfully, `false` if the
    /// channel is full or disconnected.
    pub fn stop(&self) -> bool {
        self.cmd_tx.try_send(LearnerMsg::Stop).is_ok()
    }

    /// Send stop command and wait for learner to finish.
    ///
    /// This is a blocking operation that waits for the learner thread to exit.
    pub fn stop_and_wait(self) -> std::thread::Result<()> {
        let _ = self.cmd_tx.try_send(LearnerMsg::Stop);
        self.thread.join()
    }

    /// Get latest stats (non-blocking).
    ///
    /// Returns the oldest available stats from the queue, or None if empty.
    pub fn get_stats(&self) -> Option<LearnerStats> {
        self.stats_rx.try_recv().ok()
    }

    /// Drain all available stats from the channel.
    ///
    /// Returns all stats that were buffered, in order from oldest to newest.
    pub fn drain_stats(&self) -> Vec<LearnerStats> {
        let mut stats = Vec::new();
        while let Ok(s) = self.stats_rx.try_recv() {
            stats.push(s);
        }
        stats
    }

    /// Request stats from the learner.
    ///
    /// Sends a RequestStats command. Stats will be available via `get_stats()`
    /// after the learner processes this command.
    pub fn request_stats(&self) -> bool {
        self.cmd_tx.try_send(LearnerMsg::RequestStats).is_ok()
    }

    /// Check if the learner thread is still running.
    pub fn is_running(&self) -> bool {
        !self.thread.is_finished()
    }

    /// Get the number of stats that were dropped due to full channel.
    pub fn get_dropped_stats_count(&self) -> usize {
        self.dropped_stats.load(Ordering::Relaxed)
    }

    /// Wait for learner thread to finish.
    pub fn join(self) -> std::thread::Result<()> {
        self.thread.join()
    }
}

/// Calculate steps per second safely, handling edge cases.
///
/// Returns a clamped value to prevent extreme/infinite results from
/// near-zero elapsed times.
#[inline]
fn safe_steps_per_second(train_step: usize, elapsed_secs: f32) -> f32 {
    const MIN_ELAPSED: f32 = 1e-6; // 1 microsecond minimum
    const MAX_RATE: f32 = 1e9;     // 1 billion steps/sec maximum (reasonable bound)

    if elapsed_secs < MIN_ELAPSED {
        0.0
    } else {
        let rate = train_step as f32 / elapsed_secs;
        rate.clamp(0.0, MAX_RATE)
    }
}

/// Generic learner that performs gradient updates.
///
/// Learners run in their own thread and:
/// 1. Wait for buffer to be ready
/// 2. Sample batches and compute loss
/// 3. Perform gradient updates
/// 4. Publish updated models to ModelSlot
pub struct Learner {
    config: LearnerConfig,
}

impl Learner {
    /// Create a new learner with given configuration.
    pub fn new(config: LearnerConfig) -> Self {
        Self { config }
    }

    /// Spawn learner thread for PPO.
    ///
    /// Consumes rollouts from buffer, computes GAE, and performs gradient updates.
    ///
    /// # Type Parameters
    /// - `M`: Model type (Clone + Send, NOT Sync - Burn constraint)
    ///
    /// # WGPU Note
    /// With WGPU backend, train_fn can perform tensor operations
    /// directly without synchronization - CubeCL handles it.
    ///
    /// # Panics
    ///
    /// Panics if the config contains invalid frequency values (0).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ppo<M, FTrain, FGetModel, FPublish>(
        self,
        mut train_fn: FTrain,
        mut get_model_fn: FGetModel,
        mut publish_fn: FPublish,
        model_slot: Arc<ModelSlot<M>>,
        cmd_rx: Receiver<LearnerMsg<M>>,
        ready_rx: Receiver<()>,
        shutdown: Arc<AtomicBool>,
    ) -> LearnerHandle
    where
        M: Clone + Send + 'static,
        FTrain: FnMut() -> (f32, f32, f32, f32) + Send + 'static,  // (total, policy, value, entropy)
        FGetModel: FnMut() -> M + Send + 'static,
        FPublish: FnMut(M) + Send + 'static,
    {
        // Validate config before spawning (defensive: catch invalid configs early)
        if let Err(e) = self.config.validate() {
            panic!("Invalid LearnerConfig: {}", e);
        }

        let config = self.config;
        let (stats_tx, stats_rx) = crossbeam_channel::bounded(100);

        // Create command channel - CRITICAL: cmd_tx is cloned for the handle
        // so that LearnerHandle.cmd_tx can send commands to cmd_rx_internal
        let (cmd_tx, cmd_rx_internal) = crossbeam_channel::bounded::<LearnerMsg<()>>(100);

        // Clone cmd_tx BEFORE moving it to the forwarding thread
        // This clone will be used in LearnerHandle so stop() actually works
        let handle_cmd_tx = cmd_tx.clone();

        // Track dropped stats for diagnostics
        let dropped_stats = Arc::new(AtomicUsize::new(0));
        let dropped_stats_clone = dropped_stats.clone();

        // Forward external commands from cmd_rx (typed as M) to cmd_rx_internal (typed as ())
        // The forwarding thread respects the shutdown flag for clean termination
        let shutdown_fwd = shutdown.clone();
        std::thread::Builder::new()
            .name("PPO-CmdForward".to_string())
            .spawn(move || {
                loop {
                    // Check shutdown flag periodically
                    if shutdown_fwd.load(Ordering::Relaxed) {
                        break;
                    }

                    // Use recv_timeout to allow periodic shutdown checks
                    match cmd_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(msg) => {
                            // Convert typed message to unit-typed for internal channel
                            let converted = match msg {
                                LearnerMsg::Stop => LearnerMsg::Stop,
                                LearnerMsg::Pause => LearnerMsg::Pause,
                                LearnerMsg::Resume => LearnerMsg::Resume,
                                LearnerMsg::RequestStats => LearnerMsg::RequestStats,
                                LearnerMsg::UpdateLearningRate(lr) => LearnerMsg::UpdateLearningRate(lr),
                                LearnerMsg::_Phantom(_) => continue,
                            };
                            if cmd_tx.send(converted).is_err() {
                                // Internal receiver disconnected, exit
                                break;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            // Continue to check shutdown flag
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            // External sender disconnected, exit
                            break;
                        }
                    }
                }
            })
            .expect("Failed to spawn command forwarding thread");

        let thread = std::thread::Builder::new()
            .name("PPO-Learner".to_string())
            .spawn(move || {
                let mut train_step = 0usize;
                let mut total_loss = 0.0f32;
                let mut total_policy_loss = 0.0f32;
                let mut total_value_loss = 0.0f32;
                let mut total_entropy = 0.0f32;
                let mut loss_count = 0usize;
                let start_time = Instant::now();

                // Helper to send stats with drop tracking
                let send_stats = |stats: LearnerStats, stats_tx: &Sender<LearnerStats>, dropped: &AtomicUsize| {
                    if let Err(TrySendError::Full(_)) = stats_tx.try_send(stats) {
                        dropped.fetch_add(1, Ordering::Relaxed);
                    }
                };

                loop {
                    // Check for shutdown (highest priority)
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Check max training steps
                    if config.max_train_steps > 0 && train_step >= config.max_train_steps {
                        break;
                    }

                    // Check for commands (process all pending commands)
                    while let Ok(msg) = cmd_rx_internal.try_recv() {
                        match msg {
                            LearnerMsg::Stop => {
                                // Exit immediately on stop command
                                return;
                            }
                            LearnerMsg::Pause => { /* Not implemented */ }
                            LearnerMsg::Resume => { /* Not implemented */ }
                            LearnerMsg::RequestStats => {
                                let elapsed = start_time.elapsed().as_secs_f32();
                                let stats = LearnerStats {
                                    train_steps: train_step,
                                    avg_loss: if loss_count > 0 { total_loss / loss_count as f32 } else { 0.0 },
                                    avg_policy_loss: if loss_count > 0 { total_policy_loss / loss_count as f32 } else { 0.0 },
                                    avg_value_loss: if loss_count > 0 { total_value_loss / loss_count as f32 } else { 0.0 },
                                    avg_entropy: if loss_count > 0 { total_entropy / loss_count as f32 } else { 0.0 },
                                    steps_per_second: safe_steps_per_second(train_step, elapsed),
                                    ..Default::default()
                                };
                                send_stats(stats, &stats_tx, &dropped_stats_clone);
                            }
                            LearnerMsg::UpdateLearningRate(_) => {
                                // Learning rate update handled by train_fn closure
                            }
                            LearnerMsg::_Phantom(_) => unreachable!(),
                        }
                    }

                    // Wait for rollout to be ready
                    match ready_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) => {
                            // Perform training
                            // WGPU: No synchronization needed - CubeCL streams handle it
                            let (loss, policy_loss, value_loss, entropy) = train_fn();

                            train_step += 1;

                            // Accumulate losses (handle NaN/Inf gracefully)
                            if loss.is_finite() {
                                total_loss += loss;
                            }
                            if policy_loss.is_finite() {
                                total_policy_loss += policy_loss;
                            }
                            if value_loss.is_finite() {
                                total_value_loss += value_loss;
                            }
                            if entropy.is_finite() {
                                total_entropy += entropy;
                            }
                            loss_count += 1;

                            // Publish model periodically (config.publish_freq guaranteed > 0)
                            if train_step % config.publish_freq == 0 {
                                let model = get_model_fn();
                                model_slot.publish(model.clone());
                                publish_fn(model);
                            }

                            // Report stats periodically (config.stats_freq guaranteed > 0)
                            if train_step % config.stats_freq == 0 && loss_count > 0 {
                                let elapsed = start_time.elapsed().as_secs_f32();
                                let stats = LearnerStats {
                                    train_steps: train_step,
                                    avg_loss: total_loss / loss_count as f32,
                                    avg_policy_loss: total_policy_loss / loss_count as f32,
                                    avg_value_loss: total_value_loss / loss_count as f32,
                                    avg_entropy: total_entropy / loss_count as f32,
                                    steps_per_second: safe_steps_per_second(train_step, elapsed),
                                    ..Default::default()
                                };
                                send_stats(stats, &stats_tx, &dropped_stats_clone);

                                // Reset accumulators
                                total_loss = 0.0;
                                total_policy_loss = 0.0;
                                total_value_loss = 0.0;
                                total_entropy = 0.0;
                                loss_count = 0;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            // No data ready, continue (allows command processing)
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            // Channel closed, exit gracefully
                            break;
                        }
                    }
                }
            })
            .expect("Failed to spawn PPO learner thread");

        // CRITICAL FIX: Use the cloned cmd_tx that is connected to cmd_rx_internal
        // Previously this created a new disconnected channel, breaking stop()
        LearnerHandle {
            thread,
            stats_rx,
            cmd_tx: handle_cmd_tx,
            dropped_stats,
        }
    }

    /// Spawn learner thread for IMPALA.
    ///
    /// Consumes trajectories, computes V-trace, and performs gradient updates.
    ///
    /// # Type Parameters
    /// - `M`: Model type (Clone + Send, NOT Sync - Burn constraint)
    ///
    /// # WGPU Note
    /// With WGPU backend, train_fn can perform tensor operations
    /// directly without synchronization - CubeCL handles it.
    ///
    /// # Panics
    ///
    /// Panics if the config contains invalid frequency values (0).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_impala<M, FTrain, FGetModel, FPublish, FIsReady>(
        self,
        mut train_fn: FTrain,
        mut get_model_fn: FGetModel,
        mut publish_fn: FPublish,
        mut is_ready_fn: FIsReady,
        model_slot: Arc<ModelSlot<M>>,
        cmd_rx: Receiver<LearnerMsg<M>>,
        shutdown: Arc<AtomicBool>,
        version_counter: Arc<AtomicU64>,
    ) -> LearnerHandle
    where
        M: Clone + Send + 'static,
        FTrain: FnMut() -> Option<(f32, f32, f32, f32)> + Send + 'static,
        FGetModel: FnMut() -> M + Send + 'static,
        FPublish: FnMut(M, u64) + Send + 'static,  // (model, version)
        FIsReady: FnMut() -> bool + Send + 'static,
    {
        // Validate config before spawning (defensive: catch invalid configs early)
        if let Err(e) = self.config.validate() {
            panic!("Invalid LearnerConfig: {}", e);
        }

        let config = self.config;
        let (stats_tx, stats_rx) = crossbeam_channel::bounded(100);

        // Create command channel - CRITICAL: cmd_tx is cloned for the handle
        let (cmd_tx, cmd_rx_internal) = crossbeam_channel::bounded::<LearnerMsg<()>>(100);

        // Clone cmd_tx BEFORE moving it to the forwarding thread
        let handle_cmd_tx = cmd_tx.clone();

        // Track dropped stats for diagnostics
        let dropped_stats = Arc::new(AtomicUsize::new(0));
        let dropped_stats_clone = dropped_stats.clone();

        // Forward external commands with proper shutdown handling
        let shutdown_fwd = shutdown.clone();
        std::thread::Builder::new()
            .name("IMPALA-CmdForward".to_string())
            .spawn(move || {
                loop {
                    if shutdown_fwd.load(Ordering::Relaxed) {
                        break;
                    }

                    match cmd_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(msg) => {
                            let converted = match msg {
                                LearnerMsg::Stop => LearnerMsg::Stop,
                                LearnerMsg::Pause => LearnerMsg::Pause,
                                LearnerMsg::Resume => LearnerMsg::Resume,
                                LearnerMsg::RequestStats => LearnerMsg::RequestStats,
                                LearnerMsg::UpdateLearningRate(lr) => LearnerMsg::UpdateLearningRate(lr),
                                LearnerMsg::_Phantom(_) => continue,
                            };
                            if cmd_tx.send(converted).is_err() {
                                break;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .expect("Failed to spawn IMPALA command forwarding thread");

        let thread = std::thread::Builder::new()
            .name("IMPALA-Learner".to_string())
            .spawn(move || {
                let mut train_step = 0usize;
                let mut total_loss = 0.0f32;
                let mut total_policy_loss = 0.0f32;
                let mut total_value_loss = 0.0f32;
                let mut total_entropy = 0.0f32;
                let mut loss_count = 0usize;
                let start_time = Instant::now();

                // Helper to send stats with drop tracking
                let send_stats = |stats: LearnerStats, stats_tx: &Sender<LearnerStats>, dropped: &AtomicUsize| {
                    if let Err(TrySendError::Full(_)) = stats_tx.try_send(stats) {
                        dropped.fetch_add(1, Ordering::Relaxed);
                    }
                };

                loop {
                    // Check for shutdown (highest priority)
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Check max training steps
                    if config.max_train_steps > 0 && train_step >= config.max_train_steps {
                        break;
                    }

                    // Check for commands (process all pending)
                    while let Ok(msg) = cmd_rx_internal.try_recv() {
                        match msg {
                            LearnerMsg::Stop => return,
                            LearnerMsg::Pause => { /* Not implemented */ }
                            LearnerMsg::Resume => { /* Not implemented */ }
                            LearnerMsg::RequestStats => {
                                let elapsed = start_time.elapsed().as_secs_f32();
                                let stats = LearnerStats {
                                    train_steps: train_step,
                                    avg_loss: if loss_count > 0 { total_loss / loss_count as f32 } else { 0.0 },
                                    avg_policy_loss: if loss_count > 0 { total_policy_loss / loss_count as f32 } else { 0.0 },
                                    avg_value_loss: if loss_count > 0 { total_value_loss / loss_count as f32 } else { 0.0 },
                                    avg_entropy: if loss_count > 0 { total_entropy / loss_count as f32 } else { 0.0 },
                                    steps_per_second: safe_steps_per_second(train_step, elapsed),
                                    ..Default::default()
                                };
                                send_stats(stats, &stats_tx, &dropped_stats_clone);
                            }
                            LearnerMsg::UpdateLearningRate(_) => {
                                // Learning rate update handled by train_fn closure
                            }
                            LearnerMsg::_Phantom(_) => unreachable!(),
                        }
                    }

                    // Check if buffer is ready
                    if !is_ready_fn() {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }

                    // Perform training
                    // WGPU: No synchronization needed - CubeCL streams handle it
                    if let Some((loss, policy_loss, value_loss, entropy)) = train_fn() {
                        train_step += 1;

                        // Accumulate losses (handle NaN/Inf gracefully)
                        if loss.is_finite() {
                            total_loss += loss;
                        }
                        if policy_loss.is_finite() {
                            total_policy_loss += policy_loss;
                        }
                        if value_loss.is_finite() {
                            total_value_loss += value_loss;
                        }
                        if entropy.is_finite() {
                            total_entropy += entropy;
                        }
                        loss_count += 1;

                        // Publish model with new version (config.publish_freq guaranteed > 0)
                        if train_step % config.publish_freq == 0 {
                            let model = get_model_fn();
                            let version = version_counter.fetch_add(1, Ordering::Relaxed) + 1;
                            model_slot.publish(model.clone());
                            publish_fn(model, version);
                        }

                        // Report stats periodically (config.stats_freq guaranteed > 0)
                        if train_step % config.stats_freq == 0 && loss_count > 0 {
                            let elapsed = start_time.elapsed().as_secs_f32();
                            let stats = LearnerStats {
                                train_steps: train_step,
                                avg_loss: total_loss / loss_count as f32,
                                avg_policy_loss: total_policy_loss / loss_count as f32,
                                avg_value_loss: total_value_loss / loss_count as f32,
                                avg_entropy: total_entropy / loss_count as f32,
                                steps_per_second: safe_steps_per_second(train_step, elapsed),
                                ..Default::default()
                            };
                            send_stats(stats, &stats_tx, &dropped_stats_clone);

                            // Reset accumulators
                            total_loss = 0.0;
                            total_policy_loss = 0.0;
                            total_value_loss = 0.0;
                            total_entropy = 0.0;
                            loss_count = 0;
                        }
                    }
                }
            })
            .expect("Failed to spawn IMPALA learner thread");

        // CRITICAL FIX: Use the cloned cmd_tx that is connected to cmd_rx_internal
        LearnerHandle {
            thread,
            stats_rx,
            cmd_tx: handle_cmd_tx,
            dropped_stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.publish_freq, 10);
        assert_eq!(config.stats_freq, 100);
        assert_eq!(config.eval_freq, 1000);
    }

    #[test]
    fn test_learner_config_builder() {
        let config = LearnerConfig::new()
            .with_publish_freq(50)
            .with_stats_freq(200)
            .with_max_train_steps(10000);

        assert_eq!(config.publish_freq, 50);
        assert_eq!(config.stats_freq, 200);
        assert_eq!(config.max_train_steps, 10000);
    }
}
