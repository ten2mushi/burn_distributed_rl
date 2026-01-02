//! Frequency Band Implementations
//!
//! This module provides concrete implementations of the `FrequencyBand` trait
//! for various RF bands:
//!
//! - [`UHFBand`] - Ultra High Frequency (300 MHz - 3 GHz)
//! - [`SHFBand`] - Super High Frequency (3 - 30 GHz)
//! - Future bands (VLF, LF, MF, HF, VHF, EHF) as stubs

pub mod uhf;
pub mod shf;
pub mod future;

pub use uhf::UHFBand;
pub use shf::SHFBand;

// Re-export future band stubs when implemented
pub use future::{VLFBand, LFBand, MFBand, HFBand, VHFBand, EHFBand};
