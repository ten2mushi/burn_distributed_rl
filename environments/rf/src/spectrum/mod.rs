//! Spectrum Module
//!
//! Provides spectral synthesis functionality for the RF environment:
//! - PSD (Power Spectral Density) buffer management
//! - Spectral pattern functions (rectangular, Gaussian, OFDM, chirp)
//! - Noise floor models (thermal, man-made per ITU-R P.372)
//!
//! ## Type-Safe API
//!
//! This module uses Curry-Howard compliant types to guarantee correctness:
//! - `ValidatedPsd` - PSD buffer with non-negative power guarantee
//! - `ValidatedFrequencyGrid` - Frequency grid with valid bounds
//! - `PositivePower` - Non-negative power values
//! - `Hertz` - Validated frequency values
//!
//! All pattern and noise functions use these types to prevent invalid
//! states at compile time.

pub mod psd;
pub mod patterns;
pub mod noise;

// Type-safe exports (primary API)
pub use psd::ValidatedPsd;

pub use patterns::{
    add_rect_pattern,
    add_gaussian_pattern,
    add_ofdm_pattern,
    add_chirp_pattern,
    add_chirp_pattern_centered,
    dbm_to_watts,
    watts_to_dbm,
    dbm_to_power,
};

pub use noise::{
    NoiseEnvironment,
    add_thermal_noise,
    add_thermal_noise_default,
    add_man_made_noise,
    add_noise_floor,
    noise_floor_dbm,
    calculate_snr_db,
    thermal_noise_psd,
};

#[cfg(feature = "simd")]
pub use patterns::{
    add_rect_pattern_simd,
    add_gaussian_pattern_simd,
};

#[cfg(feature = "simd")]
pub use noise::add_noise_floor_simd;
