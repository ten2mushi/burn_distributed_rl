//! Training loggers for distributed training.
//!
//! Provides different logging backends for training metrics.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Training snapshot for logging.
#[derive(Debug, Clone)]
pub struct TrainingSnapshot {
    /// Current training step.
    pub step: usize,
    /// Total environment steps.
    pub env_steps: usize,
    /// Number of completed episodes.
    pub episodes: usize,
    /// Average episode reward.
    pub avg_reward: f32,
    /// Policy loss (if applicable).
    pub policy_loss: f32,
    /// Value function loss (if applicable).
    pub value_loss: f32,
    /// Entropy (if applicable).
    pub entropy: f32,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Gradient norm (if available).
    pub gradient_norm: Option<f32>,
}

impl TrainingSnapshot {
    /// Create a new training snapshot.
    pub fn new(step: usize, env_steps: usize, episodes: usize, avg_reward: f32) -> Self {
        Self {
            step,
            env_steps,
            episodes,
            avg_reward,
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            learning_rate: 0.0,
            gradient_norm: None,
        }
    }

    /// Set loss values.
    pub fn with_losses(mut self, policy_loss: f32, value_loss: f32, entropy: f32) -> Self {
        self.policy_loss = policy_loss;
        self.value_loss = value_loss;
        self.entropy = entropy;
        self
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set gradient norm.
    pub fn with_gradient_norm(mut self, norm: f32) -> Self {
        self.gradient_norm = Some(norm);
        self
    }
}

/// Logger trait for different logging backends.
pub trait MetricsLogger: Send {
    /// Log a training snapshot.
    fn log(&mut self, snapshot: &TrainingSnapshot);

    /// Flush any buffered output.
    fn flush(&mut self);
}

/// Console logger with pretty formatting.
pub struct ConsoleLogger {
    log_interval: usize,
    last_log_step: usize,
    start_time: Instant,
    show_header: bool,
}

impl ConsoleLogger {
    /// Create a new console logger.
    ///
    /// # Arguments
    ///
    /// * `log_interval` - Steps between log entries
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            last_log_step: 0,
            start_time: Instant::now(),
            show_header: true,
        }
    }

    /// Reset the start time.
    pub fn reset_timer(&mut self) {
        self.start_time = Instant::now();
    }

    fn print_header(&self) {
        println!(
            "{:>8} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "Step", "EnvSteps", "Episodes", "Reward", "Policy", "Value", "Entropy", "FPS"
        );
        println!("{}", "-".repeat(86));
    }
}

impl MetricsLogger for ConsoleLogger {
    fn log(&mut self, snapshot: &TrainingSnapshot) {
        // Check if we should log at this step
        if snapshot.step < self.last_log_step + self.log_interval {
            return;
        }

        // Print header on first log
        if self.show_header {
            self.print_header();
            self.show_header = false;
        }

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let fps = if elapsed > 0.0 {
            snapshot.env_steps as f32 / elapsed
        } else {
            0.0
        };

        println!(
            "{:>8} {:>10} {:>8} {:>10.2} {:>10.4} {:>10.4} {:>10.4} {:>8.0}",
            snapshot.step,
            snapshot.env_steps,
            snapshot.episodes,
            snapshot.avg_reward,
            snapshot.policy_loss,
            snapshot.value_loss,
            snapshot.entropy,
            fps
        );

        self.last_log_step = snapshot.step;
    }

    fn flush(&mut self) {
        // stdout is typically line-buffered, so nothing to do
    }
}

/// CSV file logger for analysis.
pub struct CSVLogger {
    writer: BufWriter<File>,
    start_time: Instant,
}

impl CSVLogger {
    /// Create a new CSV logger.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(
            writer,
            "step,env_steps,episodes,avg_reward,policy_loss,value_loss,entropy,learning_rate,gradient_norm,elapsed_secs,fps"
        )?;

        Ok(Self {
            writer,
            start_time: Instant::now(),
        })
    }

    /// Reset the start time.
    pub fn reset_timer(&mut self) {
        self.start_time = Instant::now();
    }
}

impl MetricsLogger for CSVLogger {
    fn log(&mut self, snapshot: &TrainingSnapshot) {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let fps = if elapsed > 0.0 {
            snapshot.env_steps as f32 / elapsed
        } else {
            0.0
        };

        let grad_norm_str = snapshot
            .gradient_norm
            .map(|n| n.to_string())
            .unwrap_or_default();

        let _ = writeln!(
            self.writer,
            "{},{},{},{:.4},{:.6},{:.6},{:.6},{:.8},{},{:.2},{:.2}",
            snapshot.step,
            snapshot.env_steps,
            snapshot.episodes,
            snapshot.avg_reward,
            snapshot.policy_loss,
            snapshot.value_loss,
            snapshot.entropy,
            snapshot.learning_rate,
            grad_norm_str,
            elapsed,
            fps
        );
    }

    fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}

impl Drop for CSVLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Multi-logger that writes to multiple backends.
pub struct MultiLogger {
    loggers: Vec<Box<dyn MetricsLogger>>,
}

impl MultiLogger {
    /// Create a new multi-logger.
    pub fn new() -> Self {
        Self {
            loggers: Vec::new(),
        }
    }

    /// Add a logger.
    pub fn add<L: MetricsLogger + 'static>(mut self, logger: L) -> Self {
        self.loggers.push(Box::new(logger));
        self
    }
}

impl Default for MultiLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsLogger for MultiLogger {
    fn log(&mut self, snapshot: &TrainingSnapshot) {
        for logger in &mut self.loggers {
            logger.log(snapshot);
        }
    }

    fn flush(&mut self) {
        for logger in &mut self.loggers {
            logger.flush();
        }
    }
}

/// Progress bar style logger for interactive use.
pub struct ProgressLogger {
    total_steps: usize,
    log_interval: usize,
    last_log_step: usize,
    start_time: Instant,
}

impl ProgressLogger {
    /// Create a new progress logger.
    pub fn new(total_steps: usize, log_interval: usize) -> Self {
        Self {
            total_steps,
            log_interval,
            last_log_step: 0,
            start_time: Instant::now(),
        }
    }

    fn render_bar(&self, progress: f32, width: usize) -> String {
        let filled = (progress * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
    }
}

impl MetricsLogger for ProgressLogger {
    fn log(&mut self, snapshot: &TrainingSnapshot) {
        if snapshot.step < self.last_log_step + self.log_interval {
            return;
        }

        let progress = snapshot.step as f32 / self.total_steps as f32;
        let progress = progress.min(1.0);
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Estimate remaining time
        let eta_secs = if progress > 0.0 {
            elapsed / progress * (1.0 - progress)
        } else {
            0.0
        };

        let bar = self.render_bar(progress, 30);
        let percent = (progress * 100.0) as usize;

        print!(
            "\r{} {:>3}% | Step {:>8}/{} | Reward {:>7.2} | ETA {:>6.0}s",
            bar,
            percent,
            snapshot.step,
            self.total_steps,
            snapshot.avg_reward,
            eta_secs
        );
        let _ = std::io::stdout().flush();

        self.last_log_step = snapshot.step;
    }

    fn flush(&mut self) {
        println!(); // Move to new line when done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_snapshot() {
        let snapshot = TrainingSnapshot::new(100, 1000, 50, 150.0)
            .with_losses(0.5, 0.3, 0.1)
            .with_learning_rate(3e-4);

        assert_eq!(snapshot.step, 100);
        assert_eq!(snapshot.env_steps, 1000);
        assert_eq!(snapshot.episodes, 50);
        assert!((snapshot.avg_reward - 150.0).abs() < 0.01);
        assert!((snapshot.policy_loss - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_console_logger() {
        let mut logger = ConsoleLogger::new(10);

        // First log should print (step 0 + interval = 10, but 0 < 10, so won't print)
        let snapshot1 = TrainingSnapshot::new(5, 500, 10, 50.0);
        logger.log(&snapshot1); // Won't print (5 < 0 + 10)

        let snapshot2 = TrainingSnapshot::new(10, 1000, 20, 100.0);
        logger.log(&snapshot2); // Will print (10 >= 0 + 10)
    }

    #[test]
    fn test_multi_logger() {
        let console = ConsoleLogger::new(10);
        let mut multi = MultiLogger::new().add(console);

        let snapshot = TrainingSnapshot::new(10, 1000, 20, 100.0);
        multi.log(&snapshot);
        multi.flush();
    }
}
