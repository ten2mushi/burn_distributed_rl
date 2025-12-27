//! Model checkpointing for distributed training.
//!
//! Provides automatic saving and loading of model checkpoints during training,
//! with support for best-model tracking and checkpoint cleanup.

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::Backend;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Configuration for the checkpointer.
#[derive(Debug, Clone)]
pub struct CheckpointerConfig {
    /// Directory to store checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Steps between checkpoint saves.
    pub save_interval: usize,
    /// Number of recent checkpoints to keep (0 = keep all).
    pub keep_last_n: usize,
    /// Whether to track and save the best model.
    pub save_best: bool,
}

impl Default for CheckpointerConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            save_interval: 10_000,
            keep_last_n: 5,
            save_best: true,
        }
    }
}

impl CheckpointerConfig {
    /// Create a new config with specified checkpoint directory.
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            ..Default::default()
        }
    }

    /// Set the save interval.
    pub fn with_save_interval(mut self, interval: usize) -> Self {
        self.save_interval = interval;
        self
    }

    /// Set the number of checkpoints to keep.
    pub fn with_keep_last_n(mut self, n: usize) -> Self {
        self.keep_last_n = n;
        self
    }

    /// Enable or disable best model tracking.
    pub fn with_save_best(mut self, save_best: bool) -> Self {
        self.save_best = save_best;
        self
    }
}

/// Error type for checkpointing operations.
#[derive(Debug)]
pub enum CheckpointError {
    /// IO error during save/load.
    Io(io::Error),
    /// Burn recorder error.
    Recorder(String),
    /// No checkpoints found.
    NoCheckpoints,
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "IO error: {}", e),
            CheckpointError::Recorder(e) => write!(f, "Recorder error: {}", e),
            CheckpointError::NoCheckpoints => write!(f, "No checkpoints found"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self {
        CheckpointError::Io(e)
    }
}

/// Checkpoint metadata.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Path to the checkpoint file.
    pub path: PathBuf,
    /// Step at which checkpoint was saved.
    pub step: usize,
    /// Optional metric value (e.g., reward).
    pub metric: Option<f32>,
}

/// Model checkpointer for training.
///
/// Handles saving model checkpoints at regular intervals,
/// tracking the best model, and cleaning up old checkpoints.
pub struct Checkpointer {
    config: CheckpointerConfig,
    best_metric: f32,
    checkpoint_history: Vec<CheckpointInfo>,
}

impl Checkpointer {
    /// Create a new checkpointer.
    ///
    /// Creates the checkpoint directory if it doesn't exist.
    pub fn new(config: CheckpointerConfig) -> Result<Self, CheckpointError> {
        fs::create_dir_all(&config.checkpoint_dir)?;

        Ok(Self {
            config,
            best_metric: f32::NEG_INFINITY,
            checkpoint_history: Vec::new(),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &CheckpointerConfig {
        &self.config
    }

    /// Check if it's time to save a checkpoint.
    pub fn should_save(&self, step: usize) -> bool {
        step > 0 && step % self.config.save_interval == 0
    }

    /// Save a model checkpoint.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `step` - Current training step
    /// * `metric` - Optional metric value (e.g., average reward)
    pub fn save<B: Backend, M: Module<B>>(
        &mut self,
        model: &M,
        step: usize,
        metric: Option<f32>,
    ) -> Result<PathBuf, CheckpointError> {
        let filename = format!("checkpoint_{:08}.bin", step);
        let path = self.config.checkpoint_dir.join(&filename);

        // Save using Burn's BinFileRecorder
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(&path, &recorder)
            .map_err(|e| CheckpointError::Recorder(e.to_string()))?;

        // Track checkpoint
        let info = CheckpointInfo {
            path: path.clone(),
            step,
            metric,
        };
        self.checkpoint_history.push(info);

        // Save best model if this is the best so far
        if self.config.save_best {
            if let Some(m) = metric {
                if m > self.best_metric {
                    self.best_metric = m;
                    let best_path = self.config.checkpoint_dir.join("best.bin");
                    model
                        .clone()
                        .save_file(&best_path, &recorder)
                        .map_err(|e| CheckpointError::Recorder(e.to_string()))?;
                }
            }
        }

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(path)
    }

    /// Load a model from a checkpoint file.
    ///
    /// In Burn 0.19+, you need to provide a model template to load into.
    /// The template is typically created with `Model::new(&device)`.
    pub fn load<B: Backend, M: Module<B>>(
        &self,
        model_template: M,
        path: &Path,
        device: &B::Device,
    ) -> Result<M, CheckpointError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        model_template
            .load_file(path, &recorder, device)
            .map_err(|e| CheckpointError::Recorder(e.to_string()))
    }

    /// Load the best model.
    ///
    /// Requires a model template to load into.
    pub fn load_best<B: Backend, M: Module<B>>(
        &self,
        model_template: M,
        device: &B::Device,
    ) -> Result<M, CheckpointError> {
        let best_path = self.config.checkpoint_dir.join("best.bin");
        if !best_path.exists() {
            return Err(CheckpointError::NoCheckpoints);
        }
        self.load(model_template, &best_path, device)
    }

    /// Load the latest checkpoint.
    ///
    /// Returns the model and the step it was saved at.
    /// Requires a model template to load into.
    pub fn load_latest<B: Backend, M: Module<B>>(
        &self,
        model_template: M,
        device: &B::Device,
    ) -> Result<(M, usize), CheckpointError> {
        let latest = self.find_latest_checkpoint()?;
        let model = self.load(model_template, &latest.path, device)?;
        Ok((model, latest.step))
    }

    /// Find the latest checkpoint in the checkpoint directory.
    pub fn find_latest_checkpoint(&self) -> Result<CheckpointInfo, CheckpointError> {
        let mut checkpoints: Vec<_> = fs::read_dir(&self.config.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("checkpoint_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        if checkpoints.is_empty() {
            return Err(CheckpointError::NoCheckpoints);
        }

        // Sort by filename (which includes step number)
        checkpoints.sort_by_key(|e| e.path());

        let latest = checkpoints.pop().unwrap();
        let path = latest.path();

        // Extract step from filename
        let step = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("checkpoint_"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(CheckpointInfo {
            path,
            step,
            metric: None,
        })
    }

    /// List all checkpoints in the directory.
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointInfo>, CheckpointError> {
        let mut checkpoints: Vec<CheckpointInfo> = fs::read_dir(&self.config.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                let filename = path.file_name()?.to_str()?;
                if filename.starts_with("checkpoint_") && filename.ends_with(".bin") {
                    let step = filename
                        .strip_prefix("checkpoint_")?
                        .strip_suffix(".bin")?
                        .parse()
                        .ok()?;
                    Some(CheckpointInfo {
                        path,
                        step,
                        metric: None,
                    })
                } else {
                    None
                }
            })
            .collect();

        checkpoints.sort_by_key(|c| c.step);
        Ok(checkpoints)
    }

    /// Get the current best metric value.
    pub fn best_metric(&self) -> f32 {
        self.best_metric
    }

    /// Cleanup old checkpoints, keeping only the last N.
    fn cleanup_old_checkpoints(&mut self) -> Result<(), CheckpointError> {
        if self.config.keep_last_n == 0 {
            return Ok(()); // Keep all
        }

        // Remove from history if we have too many
        while self.checkpoint_history.len() > self.config.keep_last_n {
            let old = self.checkpoint_history.remove(0);
            // Only remove if it's not the best checkpoint
            if old.path.file_name().and_then(|n| n.to_str()) != Some("best.bin") {
                let _ = fs::remove_file(&old.path);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_checkpointer_config() {
        let config = CheckpointerConfig::new("./test_ckpts")
            .with_save_interval(5000)
            .with_keep_last_n(3)
            .with_save_best(false);

        assert_eq!(config.checkpoint_dir, PathBuf::from("./test_ckpts"));
        assert_eq!(config.save_interval, 5000);
        assert_eq!(config.keep_last_n, 3);
        assert!(!config.save_best);
    }

    #[test]
    fn test_should_save() {
        let dir = tempdir().unwrap();
        let config = CheckpointerConfig::new(dir.path()).with_save_interval(100);
        let checkpointer = Checkpointer::new(config).unwrap();

        assert!(!checkpointer.should_save(0));
        assert!(!checkpointer.should_save(50));
        assert!(checkpointer.should_save(100));
        assert!(!checkpointer.should_save(150));
        assert!(checkpointer.should_save(200));
    }

    #[test]
    fn test_checkpoint_dir_creation() {
        let dir = tempdir().unwrap();
        let subdir = dir.path().join("nested/checkpoints");

        let config = CheckpointerConfig::new(&subdir);
        let _checkpointer = Checkpointer::new(config).unwrap();

        assert!(subdir.exists());
    }
}
