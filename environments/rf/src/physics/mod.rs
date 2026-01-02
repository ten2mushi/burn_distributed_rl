//! Physics and Channel Models
//!
//! This module provides realistic RF propagation physics including:
//! - Path loss models (FSPL, log-distance, Hata, COST-231)
//! - Fading models (Rayleigh, Rician, shadowing)
//! - Multi-tap multipath (ITU EPA, EVA, ETU)
//! - Atmospheric effects (O2/H2O absorption, rain attenuation)
//! - Doppler effects (shift, micro-Doppler, coherence time)
//! - Ionospheric propagation (HF skip, MUF/LUF, diurnal variation)
//!
//! All functions are SIMD-optimized for 8-wide f32 vectors.

pub mod path_loss;
pub mod fading;
pub mod multipath;
pub mod atmospheric;
pub mod doppler;
pub mod channel;
pub mod ionospheric;

pub use path_loss::*;
pub use fading::*;
pub use multipath::*;
pub use atmospheric::*;
pub use doppler::*;
pub use channel::*;
pub use ionospheric::*;
