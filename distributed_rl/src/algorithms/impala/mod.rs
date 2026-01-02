//! IMPALA algorithm components.
//!
//! This module provides the IMPALA-specific implementation:
//! - `IMPALAConfig`: Configuration for all IMPALA settings
//! - `IMPALABuffer`: Off-policy trajectory buffer with FIFO semantics
//! - `IMPALA`: Algorithm implementation with V-trace correction

mod config;
mod impala_buffer;
mod impala;

#[cfg(test)]
mod tests;

pub use config::{IMPALAConfig, IMPALAStats};
pub use impala_buffer::{IMPALABuffer, IMPALABufferConfig, IMPALABatch};
pub use impala::{IMPALA, IMPALAProcessedBatch};
