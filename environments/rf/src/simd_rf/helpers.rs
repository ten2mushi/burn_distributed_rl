//! SIMD Helper Utilities
//!
//! Memory access patterns and geometric utilities for RF environment simulation.
//! These functions handle strided data access for SoA (Struct-of-Arrays) layouts
//! and common geometric calculations.

use std::simd::{cmp::SimdPartialOrd, f32x8, u32x8};

use super::math::simd_sqrt;

// ============================================================================
// Strided Memory Access
// ============================================================================

/// Load 8 f32 values with stride from a flat array
///
/// Used for loading data from SoA layouts where values for 8 environments
/// are interleaved with other data.
///
/// # Arguments
/// * `arr` - Source array
/// * `base` - Starting index
/// * `stride` - Distance between consecutive values
///
/// # Safety
/// Caller must ensure `base + 7 * stride < arr.len()`
#[inline]
pub fn load_f32_simd(arr: &[f32], base: usize, stride: usize) -> f32x8 {
    debug_assert!(
        base + 7 * stride < arr.len(),
        "load_f32_simd: out of bounds"
    );

    f32x8::from_array([
        arr[base],
        arr[base + stride],
        arr[base + 2 * stride],
        arr[base + 3 * stride],
        arr[base + 4 * stride],
        arr[base + 5 * stride],
        arr[base + 6 * stride],
        arr[base + 7 * stride],
    ])
}

/// Store 8 f32 values with stride to a flat array
///
/// # Arguments
/// * `arr` - Destination array
/// * `base` - Starting index
/// * `stride` - Distance between consecutive values
/// * `values` - SIMD vector to store
///
/// # Safety
/// Caller must ensure `base + 7 * stride < arr.len()`
#[inline]
pub fn store_f32_simd(arr: &mut [f32], base: usize, stride: usize, values: f32x8) {
    debug_assert!(
        base + 7 * stride < arr.len(),
        "store_f32_simd: out of bounds"
    );

    let vals: [f32; 8] = values.into();
    arr[base] = vals[0];
    arr[base + stride] = vals[1];
    arr[base + 2 * stride] = vals[2];
    arr[base + 3 * stride] = vals[3];
    arr[base + 4 * stride] = vals[4];
    arr[base + 5 * stride] = vals[5];
    arr[base + 6 * stride] = vals[6];
    arr[base + 7 * stride] = vals[7];
}

/// Load 8 contiguous f32 values from a flat array
///
/// More efficient than strided load when data is contiguous.
///
/// # Safety
/// Caller must ensure `base + 8 <= arr.len()`
#[inline]
pub fn load_f32_simd_contiguous(arr: &[f32], base: usize) -> f32x8 {
    debug_assert!(base + 8 <= arr.len(), "load_f32_simd_contiguous: out of bounds");

    f32x8::from_slice(&arr[base..base + 8])
}

/// Store 8 contiguous f32 values to a flat array
///
/// # Safety
/// Caller must ensure `base + 8 <= arr.len()`
#[inline]
pub fn store_f32_simd_contiguous(arr: &mut [f32], base: usize, values: f32x8) {
    debug_assert!(
        base + 8 <= arr.len(),
        "store_f32_simd_contiguous: out of bounds"
    );

    let vals: [f32; 8] = values.into();
    arr[base..base + 8].copy_from_slice(&vals);
}

/// Load 8 u32 values with stride from a flat array
#[inline]
pub fn load_u32_simd(arr: &[u32], base: usize, stride: usize) -> u32x8 {
    debug_assert!(
        base + 7 * stride < arr.len(),
        "load_u32_simd: out of bounds"
    );

    u32x8::from_array([
        arr[base],
        arr[base + stride],
        arr[base + 2 * stride],
        arr[base + 3 * stride],
        arr[base + 4 * stride],
        arr[base + 5 * stride],
        arr[base + 6 * stride],
        arr[base + 7 * stride],
    ])
}

/// Store 8 u32 values with stride to a flat array
#[inline]
pub fn store_u32_simd(arr: &mut [u32], base: usize, stride: usize, values: u32x8) {
    debug_assert!(
        base + 7 * stride < arr.len(),
        "store_u32_simd: out of bounds"
    );

    let vals: [u32; 8] = values.into();
    arr[base] = vals[0];
    arr[base + stride] = vals[1];
    arr[base + 2 * stride] = vals[2];
    arr[base + 3 * stride] = vals[3];
    arr[base + 4 * stride] = vals[4];
    arr[base + 5 * stride] = vals[5];
    arr[base + 6 * stride] = vals[6];
    arr[base + 7 * stride] = vals[7];
}

/// Load 8 u8 values as u32 with stride from a flat array
#[inline]
pub fn load_u8_as_u32_simd(arr: &[u8], base: usize, stride: usize) -> u32x8 {
    debug_assert!(
        base + 7 * stride < arr.len(),
        "load_u8_as_u32_simd: out of bounds"
    );

    u32x8::from_array([
        arr[base] as u32,
        arr[base + stride] as u32,
        arr[base + 2 * stride] as u32,
        arr[base + 3 * stride] as u32,
        arr[base + 4 * stride] as u32,
        arr[base + 5 * stride] as u32,
        arr[base + 6 * stride] as u32,
        arr[base + 7 * stride] as u32,
    ])
}

// ============================================================================
// Geometric Utilities
// ============================================================================

/// Position vector type alias (x, y, z coordinates for 8 environments)
pub type Position3D = (f32x8, f32x8, f32x8);

/// Velocity vector type alias
pub type Velocity3D = (f32x8, f32x8, f32x8);

/// Calculate 3D Euclidean distance between two positions
///
/// d = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
#[inline]
pub fn calc_distance_3d(pos1: Position3D, pos2: Position3D) -> f32x8 {
    let dx = pos2.0 - pos1.0;
    let dy = pos2.1 - pos1.1;
    let dz = pos2.2 - pos1.2;

    simd_sqrt(dx * dx + dy * dy + dz * dz)
}

/// Calculate 2D Euclidean distance (ignoring z coordinate)
///
/// d = sqrt((x2-x1)² + (y2-y1)²)
#[inline]
pub fn calc_distance_2d(x1: f32x8, y1: f32x8, x2: f32x8, y2: f32x8) -> f32x8 {
    let dx = x2 - x1;
    let dy = y2 - y1;

    simd_sqrt(dx * dx + dy * dy)
}

/// Calculate squared 3D distance (more efficient when sqrt not needed)
#[inline]
pub fn calc_distance_sq_3d(pos1: Position3D, pos2: Position3D) -> f32x8 {
    let dx = pos2.0 - pos1.0;
    let dy = pos2.1 - pos1.1;
    let dz = pos2.2 - pos1.2;

    dx * dx + dy * dy + dz * dz
}

/// Calculate radial velocity (velocity component along line from pos to rx_pos)
///
/// Used for Doppler shift calculations. Positive = approaching, negative = receding.
///
/// # Arguments
/// * `pos` - Position of moving entity
/// * `vel` - Velocity of moving entity
/// * `rx_pos` - Position of receiver
///
/// # Returns
/// Radial velocity (m/s), positive if approaching
#[inline]
pub fn calc_radial_velocity(pos: Position3D, vel: Velocity3D, rx_pos: Position3D) -> f32x8 {
    // Unit vector from entity to receiver
    let dx = rx_pos.0 - pos.0;
    let dy = rx_pos.1 - pos.1;
    let dz = rx_pos.2 - pos.2;

    let dist = simd_sqrt(dx * dx + dy * dy + dz * dz);

    // Avoid division by zero
    let eps = f32x8::splat(1e-10);
    let inv_dist = f32x8::splat(1.0) / (dist + eps);

    let ux = dx * inv_dist;
    let uy = dy * inv_dist;
    let uz = dz * inv_dist;

    // Dot product of velocity with unit vector
    vel.0 * ux + vel.1 * uy + vel.2 * uz
}

/// Calculate Doppler frequency shift
///
/// fd = (v_radial / c) * f_carrier
///
/// # Arguments
/// * `radial_velocity` - Radial velocity in m/s (positive = approaching)
/// * `carrier_freq` - Carrier frequency in Hz
///
/// # Returns
/// Doppler shift in Hz (positive = frequency increase)
#[inline]
pub fn calc_doppler_shift(radial_velocity: f32x8, carrier_freq: f32x8) -> f32x8 {
    let c = f32x8::splat(crate::constants::SPEED_OF_LIGHT);
    (radial_velocity / c) * carrier_freq
}

/// Calculate azimuth angle from origin to target
///
/// Returns angle in radians from positive x-axis, range [-π, π]
#[inline]
pub fn calc_azimuth(
    origin_x: f32x8,
    origin_y: f32x8,
    target_x: f32x8,
    target_y: f32x8,
) -> f32x8 {
    let dx = target_x - origin_x;
    let dy = target_y - origin_y;

    super::math::simd_atan2(dy, dx)
}

/// Calculate elevation angle from origin to target
///
/// Returns angle in radians from horizontal plane
#[inline]
pub fn calc_elevation(
    origin: Position3D,
    target: Position3D,
) -> f32x8 {
    let dx = target.0 - origin.0;
    let dy = target.1 - origin.1;
    let dz = target.2 - origin.2;

    let horizontal_dist = simd_sqrt(dx * dx + dy * dy);

    super::math::simd_atan2(dz, horizontal_dist)
}

/// Normalize a 3D vector to unit length
#[inline]
pub fn normalize_3d(v: (f32x8, f32x8, f32x8)) -> (f32x8, f32x8, f32x8) {
    let mag = simd_sqrt(v.0 * v.0 + v.1 * v.1 + v.2 * v.2);
    let eps = f32x8::splat(1e-10);
    let inv_mag = f32x8::splat(1.0) / (mag + eps);

    (v.0 * inv_mag, v.1 * inv_mag, v.2 * inv_mag)
}

/// Dot product of two 3D vectors
#[inline]
pub fn dot_3d(a: (f32x8, f32x8, f32x8), b: (f32x8, f32x8, f32x8)) -> f32x8 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

// ============================================================================
// Frequency Utilities
// ============================================================================

/// Calculate frequency bin index from frequency value
///
/// # Arguments
/// * `freq` - Frequency in Hz
/// * `freq_min` - Minimum frequency of spectrum
/// * `freq_max` - Maximum frequency of spectrum
/// * `num_bins` - Number of frequency bins
///
/// # Returns
/// Bin index (clamped to valid range)
#[inline]
pub fn freq_to_bin(
    freq: f32x8,
    freq_min: f32x8,
    freq_max: f32x8,
    num_bins: f32x8,
) -> f32x8 {
    let normalized = (freq - freq_min) / (freq_max - freq_min);
    let bin = normalized * num_bins;

    // Clamp to valid range
    super::math::simd_clamp(bin, f32x8::splat(0.0), num_bins - f32x8::splat(1.0))
}

/// Calculate frequency from bin index
///
/// # Arguments
/// * `bin` - Bin index
/// * `freq_min` - Minimum frequency of spectrum
/// * `freq_max` - Maximum frequency of spectrum
/// * `num_bins` - Number of frequency bins
///
/// # Returns
/// Frequency in Hz (center of bin)
#[inline]
pub fn bin_to_freq(
    bin: f32x8,
    freq_min: f32x8,
    freq_max: f32x8,
    num_bins: f32x8,
) -> f32x8 {
    let normalized = (bin + f32x8::splat(0.5)) / num_bins;
    freq_min + normalized * (freq_max - freq_min)
}

/// Calculate frequency overlap ratio between two bands
///
/// Returns overlap fraction in [0, 1]
#[inline]
pub fn calc_freq_overlap(
    f1_center: f32x8,
    f1_bw: f32x8,
    f2_center: f32x8,
    f2_bw: f32x8,
) -> f32x8 {
    let half_bw1 = f1_bw * f32x8::splat(0.5);
    let half_bw2 = f2_bw * f32x8::splat(0.5);

    let low1 = f1_center - half_bw1;
    let high1 = f1_center + half_bw1;
    let low2 = f2_center - half_bw2;
    let high2 = f2_center + half_bw2;

    // Overlap region
    let overlap_low = super::math::simd_select(low1.simd_gt(low2), low1, low2);
    let overlap_high = super::math::simd_select(high1.simd_lt(high2), high1, high2);

    let overlap_width = overlap_high - overlap_low;

    // Clamp to [0, 1] - negative overlap means no overlap
    let zero = f32x8::splat(0.0);
    let overlap_positive = super::math::simd_select(overlap_width.simd_gt(zero), overlap_width, zero);

    // Normalize by minimum bandwidth
    let min_bw = super::math::simd_select(f1_bw.simd_lt(f2_bw), f1_bw, f2_bw);
    let eps = f32x8::splat(1e-10);

    overlap_positive / (min_bw + eps)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-5;

    #[test]
    fn test_load_store_roundtrip() {
        let mut arr = vec![0.0f32; 24];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as f32;
        }

        // Test contiguous load/store
        let loaded = load_f32_simd_contiguous(&arr, 0);
        let expected = f32x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let loaded_arr: [f32; 8] = loaded.into();
        let expected_arr: [f32; 8] = expected.into();

        for i in 0..8 {
            assert!(
                (loaded_arr[i] - expected_arr[i]).abs() < TOLERANCE,
                "contiguous load lane {}: expected {}, got {}",
                i,
                expected_arr[i],
                loaded_arr[i]
            );
        }
    }

    #[test]
    fn test_strided_load() {
        let arr: Vec<f32> = (0..24).map(|i| i as f32).collect();

        // Load every 3rd element starting from index 0
        let loaded = load_f32_simd(&arr, 0, 3);
        let expected = f32x8::from_array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]);

        let loaded_arr: [f32; 8] = loaded.into();
        let expected_arr: [f32; 8] = expected.into();

        for i in 0..8 {
            assert!(
                (loaded_arr[i] - expected_arr[i]).abs() < TOLERANCE,
                "strided load lane {}: expected {}, got {}",
                i,
                expected_arr[i],
                loaded_arr[i]
            );
        }
    }

    #[test]
    fn test_strided_store() {
        let mut arr = vec![0.0f32; 24];
        let values = f32x8::from_array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        store_f32_simd(&mut arr, 0, 3, values);

        assert!((arr[0] - 10.0).abs() < TOLERANCE);
        assert!((arr[3] - 20.0).abs() < TOLERANCE);
        assert!((arr[6] - 30.0).abs() < TOLERANCE);
        assert!((arr[21] - 80.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_calc_distance_3d() {
        let pos1 = (
            f32x8::splat(0.0),
            f32x8::splat(0.0),
            f32x8::splat(0.0),
        );
        let pos2 = (
            f32x8::splat(3.0),
            f32x8::splat(4.0),
            f32x8::splat(0.0),
        );

        let dist = calc_distance_3d(pos1, pos2);
        let dist_arr: [f32; 8] = dist.into();

        // 3-4-5 triangle, distance should be 5
        for (i, d) in dist_arr.iter().enumerate() {
            assert!(
                (*d - 5.0).abs() < TOLERANCE,
                "distance lane {}: expected 5.0, got {}",
                i,
                d
            );
        }
    }

    #[test]
    fn test_calc_radial_velocity() {
        // Entity at origin moving toward receiver at (100, 0, 0)
        let pos = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
        let vel = (f32x8::splat(10.0), f32x8::splat(0.0), f32x8::splat(0.0)); // Moving in +x
        let rx_pos = (f32x8::splat(100.0), f32x8::splat(0.0), f32x8::splat(0.0));

        let v_radial = calc_radial_velocity(pos, vel, rx_pos);
        let v_arr: [f32; 8] = v_radial.into();

        // Should be +10 (approaching)
        for (i, v) in v_arr.iter().enumerate() {
            assert!(
                (*v - 10.0).abs() < TOLERANCE,
                "radial velocity lane {}: expected 10.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_calc_radial_velocity_receding() {
        // Entity at origin moving away from receiver at (100, 0, 0)
        let pos = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
        let vel = (f32x8::splat(-10.0), f32x8::splat(0.0), f32x8::splat(0.0)); // Moving in -x
        let rx_pos = (f32x8::splat(100.0), f32x8::splat(0.0), f32x8::splat(0.0));

        let v_radial = calc_radial_velocity(pos, vel, rx_pos);
        let v_arr: [f32; 8] = v_radial.into();

        // Should be -10 (receding)
        for (i, v) in v_arr.iter().enumerate() {
            assert!(
                (*v + 10.0).abs() < TOLERANCE,
                "radial velocity lane {}: expected -10.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_freq_overlap_full() {
        // Same band = 100% overlap
        let overlap = calc_freq_overlap(
            f32x8::splat(2.4e9),
            f32x8::splat(20e6),
            f32x8::splat(2.4e9),
            f32x8::splat(20e6),
        );

        let overlap_arr: [f32; 8] = overlap.into();

        for (i, o) in overlap_arr.iter().enumerate() {
            assert!(
                (*o - 1.0).abs() < 0.01,
                "full overlap lane {}: expected 1.0, got {}",
                i,
                o
            );
        }
    }

    #[test]
    fn test_freq_overlap_none() {
        // Non-overlapping bands
        let overlap = calc_freq_overlap(
            f32x8::splat(2.4e9),
            f32x8::splat(20e6),
            f32x8::splat(5.0e9),
            f32x8::splat(20e6),
        );

        let overlap_arr: [f32; 8] = overlap.into();

        for (i, o) in overlap_arr.iter().enumerate() {
            assert!(
                *o < 0.01,
                "no overlap lane {}: expected ~0, got {}",
                i,
                o
            );
        }
    }
}
