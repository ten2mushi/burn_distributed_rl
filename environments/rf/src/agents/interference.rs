//! Interference Matrix Computation
//!
//! Calculates interference between all agents based on frequency overlap,
//! distance, and channel effects.

use super::multi_agent::MultiAgentState;

// ============================================================================
// Frequency Overlap Calculation
// ============================================================================

/// Calculate frequency overlap between two signals
///
/// Returns overlap ratio in range [0, 1]:
/// - 0.0 = no overlap (disjoint bands)
/// - 1.0 = complete overlap (same center and bandwidth)
///
/// # Arguments
/// * `f1` - Center frequency of signal 1 (Hz)
/// * `bw1` - Bandwidth of signal 1 (Hz)
/// * `f2` - Center frequency of signal 2 (Hz)
/// * `bw2` - Bandwidth of signal 2 (Hz)
pub fn frequency_overlap(f1: f32, bw1: f32, f2: f32, bw2: f32) -> f32 {
    // Calculate band edges
    let low1 = f1 - bw1 / 2.0;
    let high1 = f1 + bw1 / 2.0;
    let low2 = f2 - bw2 / 2.0;
    let high2 = f2 + bw2 / 2.0;

    // Calculate overlap region
    let overlap_low = low1.max(low2);
    let overlap_high = high1.min(high2);

    if overlap_high <= overlap_low {
        // No overlap
        0.0
    } else {
        // Overlap width relative to target signal bandwidth
        let overlap_width = overlap_high - overlap_low;
        // Normalize by the smaller bandwidth (more conservative)
        let min_bw = bw1.min(bw2);
        if min_bw > 0.0 {
            (overlap_width / min_bw).min(1.0)
        } else {
            0.0
        }
    }
}

/// Calculate frequency overlap considering both signals' bandwidths
///
/// Returns the fraction of signal 2's power that overlaps with signal 1
pub fn frequency_overlap_asymmetric(f1: f32, bw1: f32, f2: f32, bw2: f32) -> f32 {
    let low1 = f1 - bw1 / 2.0;
    let high1 = f1 + bw1 / 2.0;
    let low2 = f2 - bw2 / 2.0;
    let high2 = f2 + bw2 / 2.0;

    let overlap_low = low1.max(low2);
    let overlap_high = high1.min(high2);

    if overlap_high <= overlap_low || bw2 <= 0.0 {
        0.0
    } else {
        ((overlap_high - overlap_low) / bw2).min(1.0)
    }
}

// ============================================================================
// Interference Matrix
// ============================================================================

/// Interference matrix for a single environment
///
/// Stores interference levels between all pairs of agents.
#[derive(Clone, Debug)]
pub struct InterferenceMatrix {
    /// Number of jammers
    pub num_jammers: usize,
    /// Number of CRs
    pub num_crs: usize,

    /// Jammer to CR interference (dBm): [num_jammers × num_crs]
    /// Received power at each CR from each jammer
    pub jammer_to_cr: Vec<f32>,

    /// CR to CR interference (dBm): [num_crs × num_crs]
    /// Mutual interference between CRs (diagonal = self signal)
    pub cr_to_cr: Vec<f32>,

    /// Jammer success flags: [num_jammers × num_crs]
    /// True if jammer successfully disrupts CR
    pub jam_success: Vec<bool>,

    /// CR SINR values (dB): [num_crs]
    pub cr_sinr: Vec<f32>,

    /// Total interference at each CR (dBm): [num_crs]
    pub cr_interference: Vec<f32>,
}

impl InterferenceMatrix {
    /// Create empty interference matrix
    pub fn new(num_jammers: usize, num_crs: usize) -> Self {
        Self {
            num_jammers,
            num_crs,
            jammer_to_cr: vec![f32::NEG_INFINITY; num_jammers * num_crs],
            cr_to_cr: vec![f32::NEG_INFINITY; num_crs * num_crs],
            jam_success: vec![false; num_jammers * num_crs],
            cr_sinr: vec![0.0; num_crs],
            cr_interference: vec![f32::NEG_INFINITY; num_crs],
        }
    }

    /// Get index into jammer_to_cr array
    #[inline]
    pub fn j2c_idx(&self, jammer: usize, cr: usize) -> usize {
        debug_assert!(jammer < self.num_jammers);
        debug_assert!(cr < self.num_crs);
        jammer * self.num_crs + cr
    }

    /// Get index into cr_to_cr array
    #[inline]
    pub fn c2c_idx(&self, cr1: usize, cr2: usize) -> usize {
        debug_assert!(cr1 < self.num_crs);
        debug_assert!(cr2 < self.num_crs);
        cr1 * self.num_crs + cr2
    }

    /// Get jammer to CR interference (dBm)
    #[inline]
    pub fn jammer_interference(&self, jammer: usize, cr: usize) -> f32 {
        self.jammer_to_cr[self.j2c_idx(jammer, cr)]
    }

    /// Set jammer to CR interference (dBm)
    #[inline]
    pub fn set_jammer_interference(&mut self, jammer: usize, cr: usize, power_dbm: f32) {
        let idx = self.j2c_idx(jammer, cr);
        self.jammer_to_cr[idx] = power_dbm;
    }

    /// Get CR to CR interference (dBm)
    #[inline]
    pub fn cr_interference_from(&self, from_cr: usize, to_cr: usize) -> f32 {
        self.cr_to_cr[self.c2c_idx(from_cr, to_cr)]
    }

    /// Set CR to CR interference (dBm)
    #[inline]
    pub fn set_cr_interference(&mut self, from_cr: usize, to_cr: usize, power_dbm: f32) {
        let idx = self.c2c_idx(from_cr, to_cr);
        self.cr_to_cr[idx] = power_dbm;
    }

    /// Check if jammer successfully jams CR
    #[inline]
    pub fn is_jammed(&self, jammer: usize, cr: usize) -> bool {
        self.jam_success[self.j2c_idx(jammer, cr)]
    }

    /// Set jam success flag
    #[inline]
    pub fn set_jammed(&mut self, jammer: usize, cr: usize, success: bool) {
        let idx = self.j2c_idx(jammer, cr);
        self.jam_success[idx] = success;
    }

    /// Get SINR for CR
    #[inline]
    pub fn sinr(&self, cr: usize) -> f32 {
        self.cr_sinr[cr]
    }

    /// Count how many CRs a jammer successfully jams
    pub fn jammer_victim_count(&self, jammer: usize) -> usize {
        (0..self.num_crs)
            .filter(|&cr| self.is_jammed(jammer, cr))
            .count()
    }

    /// Count how many jammers are jamming a CR
    pub fn jammers_affecting_cr(&self, cr: usize) -> usize {
        (0..self.num_jammers)
            .filter(|&j| self.is_jammed(j, cr))
            .count()
    }

    /// Check if CR is jammed by any jammer
    pub fn cr_is_jammed(&self, cr: usize) -> bool {
        (0..self.num_jammers).any(|j| self.is_jammed(j, cr))
    }

    /// Get total interference power at CR from all sources (linear)
    pub fn total_interference_linear(&self, cr: usize) -> f32 {
        let mut total = 0.0;

        // Add jammer interference
        for j in 0..self.num_jammers {
            let power_dbm = self.jammer_interference(j, cr);
            if power_dbm > -200.0 {
                total += db_to_linear(power_dbm);
            }
        }

        // Add CR-to-CR interference (from other CRs)
        for other_cr in 0..self.num_crs {
            if other_cr != cr {
                let power_dbm = self.cr_interference_from(other_cr, cr);
                if power_dbm > -200.0 {
                    total += db_to_linear(power_dbm);
                }
            }
        }

        total
    }

    /// Compute interference matrix from agent state
    pub fn compute(
        state: &MultiAgentState,
        env: usize,
        noise_floor_dbm: f32,
        sinr_threshold_db: f32,
    ) -> Self {
        let mut matrix = Self::new(state.num_jammers, state.num_crs);

        // Compute jammer to CR interference
        for j in 0..state.num_jammers {
            let j_idx = state.jammer_idx(env, j);
            let j_freq = state.jammer_freq[j_idx];
            let j_bw = state.jammer_bandwidth[j_idx];
            let j_power = state.jammer_power[j_idx];
            let j_x = state.jammer_x[j_idx];
            let j_y = state.jammer_y[j_idx];

            for c in 0..state.num_crs {
                let c_idx = state.cr_idx(env, c);
                let c_freq = state.cr_freq[c_idx];
                let c_bw = state.cr_bandwidth[c_idx];
                let c_x = state.cr_x[c_idx];
                let c_y = state.cr_y[c_idx];

                // Calculate frequency overlap
                let overlap = frequency_overlap(j_freq, j_bw, c_freq, c_bw);

                if overlap > 0.0 {
                    // Calculate distance
                    let dx = j_x - c_x;
                    let dy = j_y - c_y;
                    let distance = (dx * dx + dy * dy).sqrt().max(1.0); // Minimum 1m

                    // Simple FSPL (free space path loss)
                    let freq_ghz = j_freq / 1e9;
                    let fspl_db = 20.0 * (distance.log10()) + 20.0 * (freq_ghz.log10()) + 32.45;

                    // Received interference power
                    let rx_power = j_power - fspl_db + 10.0 * overlap.log10();

                    matrix.set_jammer_interference(j, c, rx_power);
                }
            }
        }

        // Compute CR to CR interference (self and mutual)
        for c1 in 0..state.num_crs {
            let c1_idx = state.cr_idx(env, c1);
            let c1_freq = state.cr_freq[c1_idx];
            let c1_bw = state.cr_bandwidth[c1_idx];
            let c1_power = state.cr_power[c1_idx];
            let c1_x = state.cr_x[c1_idx];
            let c1_y = state.cr_y[c1_idx];

            for c2 in 0..state.num_crs {
                let c2_idx = state.cr_idx(env, c2);
                let c2_freq = state.cr_freq[c2_idx];
                let c2_bw = state.cr_bandwidth[c2_idx];
                let c2_x = state.cr_x[c2_idx];
                let c2_y = state.cr_y[c2_idx];

                if c1 == c2 {
                    // Self-signal (for SINR calculation)
                    // Assume fixed link distance for CR
                    let link_distance: f32 = 100.0; // 100m typical link distance
                    let freq_ghz = c1_freq / 1e9;
                    let fspl_db =
                        20.0 * link_distance.log10() + 20.0 * freq_ghz.log10() + 32.45;
                    let rx_signal = c1_power - fspl_db;
                    matrix.set_cr_interference(c1, c2, rx_signal);
                } else {
                    // Mutual interference
                    let overlap = frequency_overlap(c1_freq, c1_bw, c2_freq, c2_bw);

                    if overlap > 0.0 {
                        let dx = c1_x - c2_x;
                        let dy = c1_y - c2_y;
                        let distance = (dx * dx + dy * dy).sqrt().max(1.0);

                        let freq_ghz = c1_freq / 1e9;
                        let fspl_db =
                            20.0 * distance.log10() + 20.0 * freq_ghz.log10() + 32.45;
                        let rx_power = c1_power - fspl_db + 10.0 * overlap.log10();

                        matrix.set_cr_interference(c1, c2, rx_power);
                    }
                }
            }
        }

        // Calculate SINR and jam success for each CR
        for c in 0..state.num_crs {
            // Signal power (from diagonal)
            let signal_linear = db_to_linear(matrix.cr_interference_from(c, c));

            // Total interference + noise
            let interference_linear = matrix.total_interference_linear(c);
            let noise_linear = db_to_linear(noise_floor_dbm);
            let total_interference = interference_linear + noise_linear;

            // Calculate SINR
            let sinr_linear = signal_linear / total_interference.max(1e-20);
            let sinr_db = 10.0 * sinr_linear.log10();

            matrix.cr_sinr[c] = sinr_db;
            matrix.cr_interference[c] = linear_to_db(total_interference);

            // Check jam success for each jammer
            for j in 0..state.num_jammers {
                let jammer_power = matrix.jammer_interference(j, c);
                if jammer_power > noise_floor_dbm && sinr_db < sinr_threshold_db {
                    // Check if this jammer contributes significantly
                    let jammer_linear = db_to_linear(jammer_power);
                    if jammer_linear > 0.1 * interference_linear {
                        matrix.set_jammed(j, c, true);
                    }
                }
            }
        }

        matrix
    }
}

// ============================================================================
// Batch Interference Computation
// ============================================================================

/// Compute interference matrices for all environments
pub fn compute_all_interference(
    state: &MultiAgentState,
    noise_floor_dbm: f32,
    sinr_threshold_db: f32,
) -> Vec<InterferenceMatrix> {
    (0..state.num_envs)
        .map(|env| InterferenceMatrix::compute(state, env, noise_floor_dbm, sinr_threshold_db))
        .collect()
}

// ============================================================================
// SINR Calculation
// ============================================================================

/// Calculate SINR given signal power and interference powers
///
/// All values in dBm
pub fn calculate_sinr_db(signal_dbm: f32, interference_dbm_list: &[f32], noise_floor_dbm: f32) -> f32 {
    let signal_linear = db_to_linear(signal_dbm);

    let interference_linear: f32 = interference_dbm_list
        .iter()
        .filter(|&&p| p > -200.0)
        .map(|&p| db_to_linear(p))
        .sum();

    let noise_linear = db_to_linear(noise_floor_dbm);
    let total_interference = interference_linear + noise_linear;

    let sinr_linear = signal_linear / total_interference.max(1e-20);
    10.0 * sinr_linear.log10()
}

/// Calculate link budget (received signal power)
///
/// Uses simplified free-space path loss model
pub fn calculate_link_budget(tx_power_dbm: f32, distance_m: f32, freq_hz: f32) -> f32 {
    if distance_m <= 0.0 || freq_hz <= 0.0 {
        return tx_power_dbm;
    }

    let freq_ghz = freq_hz / 1e9;
    let distance = distance_m.max(1.0);

    // FSPL in dB
    let fspl_db = 20.0 * distance.log10() + 20.0 * freq_ghz.log10() + 32.45;

    tx_power_dbm - fspl_db
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dBm to linear power (mW)
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

/// Convert linear power (mW) to dBm
#[inline]
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        10.0 * linear.log10()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_overlap_complete() {
        // Same frequency and bandwidth = complete overlap
        let overlap = frequency_overlap(2.45e9, 20e6, 2.45e9, 20e6);
        // Relaxed tolerance due to f32 precision at large frequencies
        assert!((overlap - 1.0).abs() < 1e-4, "overlap = {}", overlap);
    }

    #[test]
    fn test_frequency_overlap_none() {
        // Disjoint bands = no overlap
        let overlap = frequency_overlap(2.40e9, 20e6, 2.48e9, 20e6);
        assert!(overlap.abs() < 1e-6);
    }

    #[test]
    fn test_frequency_overlap_partial() {
        // 50% overlap
        let f1 = 2.45e9;
        let bw = 20e6;
        let f2 = f1 + bw / 2.0; // Shifted by half bandwidth

        let overlap = frequency_overlap(f1, bw, f2, bw);

        // Should be 50% overlap
        assert!(overlap > 0.4 && overlap < 0.6, "Overlap = {}", overlap);
    }

    #[test]
    fn test_interference_matrix_indexing() {
        let matrix = InterferenceMatrix::new(2, 3);

        // J2C indexing: row-major [jammer × cr]
        assert_eq!(matrix.j2c_idx(0, 0), 0);
        assert_eq!(matrix.j2c_idx(0, 2), 2);
        assert_eq!(matrix.j2c_idx(1, 0), 3);
        assert_eq!(matrix.j2c_idx(1, 2), 5);

        // C2C indexing: row-major [cr × cr]
        assert_eq!(matrix.c2c_idx(0, 0), 0);
        assert_eq!(matrix.c2c_idx(0, 2), 2);
        assert_eq!(matrix.c2c_idx(2, 0), 6);
        assert_eq!(matrix.c2c_idx(2, 2), 8);
    }

    #[test]
    fn test_interference_matrix_set_get() {
        let mut matrix = InterferenceMatrix::new(2, 2);

        matrix.set_jammer_interference(0, 1, -50.0);
        assert!((matrix.jammer_interference(0, 1) - (-50.0)).abs() < 0.01);

        matrix.set_cr_interference(1, 0, -60.0);
        assert!((matrix.cr_interference_from(1, 0) - (-60.0)).abs() < 0.01);

        matrix.set_jammed(0, 1, true);
        assert!(matrix.is_jammed(0, 1));
        assert!(!matrix.is_jammed(0, 0));
    }

    #[test]
    fn test_jammer_victim_count() {
        let mut matrix = InterferenceMatrix::new(2, 3);

        // Jammer 0 jams CRs 0 and 2
        matrix.set_jammed(0, 0, true);
        matrix.set_jammed(0, 2, true);

        assert_eq!(matrix.jammer_victim_count(0), 2);
        assert_eq!(matrix.jammer_victim_count(1), 0);
    }

    #[test]
    fn test_sinr_calculation() {
        // Signal at -50 dBm
        let signal = -50.0;

        // Two interferers at -60 dBm each
        let interference = vec![-60.0, -60.0];

        // Noise floor at -100 dBm
        let noise = -100.0;

        let sinr = calculate_sinr_db(signal, &interference, noise);

        // Signal is 10 dB above each interferer
        // Two interferers combine to ~-57 dBm (3 dB higher)
        // SINR should be around 7 dB
        assert!(sinr > 5.0 && sinr < 9.0, "SINR = {} dB", sinr);
    }

    #[test]
    fn test_link_budget() {
        let tx_power = 30.0; // 30 dBm = 1W
        let distance = 100.0; // 100 meters
        let freq = 2.4e9; // 2.4 GHz

        let rx_power = calculate_link_budget(tx_power, distance, freq);

        // FSPL at 100m, 2.4 GHz should be about 80 dB
        // So rx_power should be around -50 dBm
        assert!(rx_power > -60.0 && rx_power < -40.0, "RX power = {} dBm", rx_power);
    }

    #[test]
    fn test_total_interference_linear() {
        let mut matrix = InterferenceMatrix::new(2, 2);

        // Set some interference values
        matrix.set_jammer_interference(0, 0, -50.0);
        matrix.set_jammer_interference(1, 0, -50.0);
        matrix.set_cr_interference(1, 0, -60.0);

        let total = matrix.total_interference_linear(0);

        // Two -50 dBm sources + one -60 dBm source
        // Linear: 10^-5 + 10^-5 + 10^-6 = 2.1e-5
        let expected = 2.0e-5 + 1e-6;
        assert!((total - expected).abs() < 1e-7, "Total = {}, expected = {}", total, expected);
    }

    #[test]
    fn test_interference_matrix_compute() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Set jammer state
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;
        state.jammer_power[j_idx] = 30.0;
        state.jammer_x[j_idx] = 0.0;
        state.jammer_y[j_idx] = 0.0;

        // Set CR state on same frequency but 50m away
        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.45e9;
        state.cr_bandwidth[c_idx] = 20e6;
        state.cr_power[c_idx] = 20.0;
        state.cr_x[c_idx] = 50.0;
        state.cr_y[c_idx] = 0.0;

        let matrix = InterferenceMatrix::compute(&state, 0, -100.0, 10.0);

        // Jammer should cause interference
        let jammer_power = matrix.jammer_interference(0, 0);
        assert!(jammer_power > -100.0, "Jammer power = {} dBm", jammer_power);

        // SINR should be computed
        let sinr = matrix.sinr(0);
        assert!(sinr.is_finite(), "SINR = {}", sinr);
    }

    #[test]
    fn test_cr_is_jammed() {
        let mut matrix = InterferenceMatrix::new(2, 2);

        // Initially no one is jammed
        assert!(!matrix.cr_is_jammed(0));
        assert!(!matrix.cr_is_jammed(1));

        // Jam CR 0 by jammer 1
        matrix.set_jammed(1, 0, true);
        assert!(matrix.cr_is_jammed(0));
        assert!(!matrix.cr_is_jammed(1));
    }
}
