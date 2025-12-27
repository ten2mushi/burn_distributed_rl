//! Distributed IMPALA algorithm components.
//!
//! This module provides the IMPALA-specific implementation for distributed training:
//! - `IMPALAConfig`: Unified configuration for all IMPALA settings
//! - `IMPALABuffer`: Off-policy trajectory buffer with FIFO semantics
//! - `DistributedIMPALA`: Algorithm implementation with V-trace correction

mod config;
mod impala_buffer;
mod distributed_impala;

#[cfg(test)]
mod tests;

pub use config::{IMPALAConfig, IMPALAStats};
pub use impala_buffer::{IMPALABuffer, IMPALABufferConfig, IMPALABatch};
pub use distributed_impala::{DistributedIMPALA, IMPALAProcessedBatch};
