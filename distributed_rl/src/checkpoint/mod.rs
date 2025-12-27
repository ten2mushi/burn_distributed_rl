//! Model checkpointing module.
//!
//! Provides checkpoint saving and loading for model persistence during training.
//!
//! ## Features
//!
//! - Automatic checkpoint saving at configurable intervals
//! - Best model tracking based on a metric (e.g., reward)
//! - Automatic cleanup of old checkpoints
//! - Resume training from latest or best checkpoint
//!
//! ## Example
//!
//! ```rust,ignore
//! use burn_rl::distributed::checkpoint::{Checkpointer, CheckpointerConfig};
//!
//! let config = CheckpointerConfig::new("./checkpoints")
//!     .with_save_interval(10_000)
//!     .with_keep_last_n(5)
//!     .with_save_best(true);
//!
//! let mut checkpointer = Checkpointer::new(config)?;
//!
//! // In training loop:
//! if checkpointer.should_save(step) {
//!     checkpointer.save(&model, step, Some(avg_reward))?;
//! }
//!
//! // Resume training:
//! let (model, step) = checkpointer.load_latest(&device)?;
//! ```

pub mod checkpointer;

pub use checkpointer::{
    Checkpointer,
    CheckpointerConfig,
    CheckpointInfo,
    CheckpointError,
};
