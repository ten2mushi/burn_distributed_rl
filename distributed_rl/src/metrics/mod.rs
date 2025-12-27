//! Training metrics and logging for Distributed.
//!
//! ## Metrics
//!
//! - [`TrainingMetrics`]: Thread-safe counters for training progress
//! - [`SharedTrainingMetrics`]: Arc wrapper for multi-threaded access
//!
//! ## Loggers
//!
//! - [`ConsoleLogger`]: Pretty-printed console output
//! - [`CSVLogger`]: CSV file logging for analysis
//! - [`ProgressLogger`]: Progress bar with ETA
//! - [`MultiLogger`]: Combine multiple loggers

pub mod training_metrics;
pub mod logger;

pub use training_metrics::{TrainingMetrics, SharedTrainingMetrics, training_metrics};
pub use logger::{
    TrainingSnapshot,
    MetricsLogger,
    ConsoleLogger,
    CSVLogger,
    MultiLogger,
    ProgressLogger,
};
