//! Frequency Domain Types
//!
//! Types for validated frequency ranges and grids that enforce
//! physical constraints and prevent invalid frequency configurations.

use super::dimensional::Hertz;
use super::primitives::PositiveF32;
use std::num::NonZeroUsize;

// ============================================================================
// FrequencyRange: Validated min < max Range
// ============================================================================

/// A validated frequency range where min < max is guaranteed.
///
/// # Invariant
/// `min < max` (both positive)
///
/// This type prevents invalid configurations such as:
/// - Negative frequencies
/// - Empty ranges (min >= max)
/// - Zero-width ranges
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyRange {
    min: Hertz,
    max: Hertz,
}

impl FrequencyRange {
    /// Try to create a frequency range.
    /// Returns `None` if min >= max.
    #[inline]
    pub fn try_new(min: Hertz, max: Hertz) -> Option<Self> {
        (min.as_hz() < max.as_hz()).then(|| Self { min, max })
    }

    /// Create a frequency range, panicking if invalid.
    #[inline]
    pub fn new(min: Hertz, max: Hertz) -> Self {
        Self::try_new(min, max).expect("FrequencyRange requires min < max")
    }

    /// Create from raw Hz values.
    #[inline]
    pub fn from_hz(min_hz: f32, max_hz: f32) -> Option<Self> {
        let min = Hertz::try_new(min_hz)?;
        let max = Hertz::try_new(max_hz)?;
        Self::try_new(min, max)
    }

    /// Create from MHz values.
    #[inline]
    pub fn from_mhz(min_mhz: f32, max_mhz: f32) -> Option<Self> {
        let min = Hertz::from_mhz(min_mhz)?;
        let max = Hertz::from_mhz(max_mhz)?;
        Self::try_new(min, max)
    }

    /// Create from GHz values.
    #[inline]
    pub fn from_ghz(min_ghz: f32, max_ghz: f32) -> Option<Self> {
        let min = Hertz::from_ghz(min_ghz)?;
        let max = Hertz::from_ghz(max_ghz)?;
        Self::try_new(min, max)
    }

    /// Get minimum frequency.
    #[inline]
    pub fn min(&self) -> Hertz {
        self.min
    }

    /// Get maximum frequency.
    #[inline]
    pub fn max(&self) -> Hertz {
        self.max
    }

    /// Get bandwidth (max - min), guaranteed positive.
    #[inline]
    pub fn bandwidth(&self) -> Hertz {
        // Safe: we know max > min from construction
        Hertz::new(self.max.as_hz() - self.min.as_hz())
    }

    /// Get center frequency.
    #[inline]
    pub fn center(&self) -> Hertz {
        Hertz::new((self.min.as_hz() + self.max.as_hz()) / 2.0)
    }

    /// Check if a frequency is within this range (inclusive).
    #[inline]
    pub fn contains(&self, freq: Hertz) -> bool {
        freq.as_hz() >= self.min.as_hz() && freq.as_hz() <= self.max.as_hz()
    }

    /// Check if another range overlaps with this one.
    #[inline]
    pub fn overlaps(&self, other: &FrequencyRange) -> bool {
        self.min.as_hz() <= other.max.as_hz() && other.min.as_hz() <= self.max.as_hz()
    }

    /// Get the intersection with another range, if any.
    #[inline]
    pub fn intersection(&self, other: &FrequencyRange) -> Option<FrequencyRange> {
        let new_min = self.min.as_hz().max(other.min.as_hz());
        let new_max = self.max.as_hz().min(other.max.as_hz());

        FrequencyRange::from_hz(new_min, new_max)
    }

    /// Expand the range by a factor (centered).
    #[inline]
    pub fn expand(&self, factor: PositiveF32) -> Option<FrequencyRange> {
        let center = self.center().as_hz();
        let half_bw = self.bandwidth().as_hz() / 2.0;
        let new_half_bw = half_bw * factor.get();

        FrequencyRange::from_hz(center - new_half_bw, center + new_half_bw)
    }
}

// ============================================================================
// ValidatedFrequencyGrid: PSD Grid with Invariants
// ============================================================================

/// A validated frequency grid for PSD operations.
///
/// # Invariants
/// - Range is valid (min < max)
/// - num_bins > 0
/// - All bin centers are within range
///
/// This replaces the legacy `FrequencyGrid` with type-safe construction.
#[derive(Clone, Debug)]
pub struct ValidatedFrequencyGrid {
    /// The frequency range covered by this grid.
    range: FrequencyRange,
    /// Number of frequency bins (guaranteed > 0).
    num_bins: NonZeroUsize,
    /// Frequency resolution (Hz per bin).
    resolution: Hertz,
    /// Pre-computed bin center frequencies.
    bin_centers: Vec<Hertz>,
}

impl ValidatedFrequencyGrid {
    /// Create a new validated frequency grid.
    ///
    /// # Arguments
    /// * `range` - The frequency range to cover
    /// * `num_bins` - Number of bins (must be > 0)
    ///
    /// # Returns
    /// A validated grid with pre-computed bin centers.
    pub fn new(range: FrequencyRange, num_bins: NonZeroUsize) -> Self {
        let n = num_bins.get() as f32;
        let resolution_hz = range.bandwidth().as_hz() / n;
        let resolution = Hertz::new(resolution_hz);

        // Pre-compute bin centers
        let bin_centers: Vec<Hertz> = (0..num_bins.get())
            .map(|i| {
                let center_hz = range.min().as_hz() + (i as f32 + 0.5) * resolution_hz;
                Hertz::new(center_hz)
            })
            .collect();

        Self {
            range,
            num_bins,
            resolution,
            bin_centers,
        }
    }

    /// Create from raw parameters.
    #[inline]
    pub fn from_params(freq_min_hz: f32, freq_max_hz: f32, num_bins: usize) -> Option<Self> {
        let range = FrequencyRange::from_hz(freq_min_hz, freq_max_hz)?;
        let num_bins = NonZeroUsize::new(num_bins)?;
        Some(Self::new(range, num_bins))
    }

    /// Get the frequency range.
    #[inline]
    pub fn range(&self) -> &FrequencyRange {
        &self.range
    }

    /// Get minimum frequency.
    #[inline]
    pub fn freq_min(&self) -> Hertz {
        self.range.min()
    }

    /// Get maximum frequency.
    #[inline]
    pub fn freq_max(&self) -> Hertz {
        self.range.max()
    }

    /// Get number of bins.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.num_bins.get()
    }

    /// Get the NonZeroUsize bin count.
    #[inline]
    pub fn num_bins_nonzero(&self) -> NonZeroUsize {
        self.num_bins
    }

    /// Get frequency resolution.
    #[inline]
    pub fn resolution(&self) -> Hertz {
        self.resolution
    }

    /// Get reference to pre-computed bin centers.
    #[inline]
    pub fn bin_centers(&self) -> &[Hertz] {
        &self.bin_centers
    }

    /// Get bin center at index.
    /// Returns None if index is out of bounds.
    #[inline]
    pub fn bin_center(&self, index: usize) -> Option<Hertz> {
        self.bin_centers.get(index).copied()
    }

    /// Convert frequency to bin index.
    ///
    /// Frequencies outside the range are clamped to the nearest valid bin.
    #[inline]
    pub fn freq_to_bin(&self, freq: Hertz) -> usize {
        let normalized = (freq.as_hz() - self.range.min().as_hz())
            / (self.range.max().as_hz() - self.range.min().as_hz());
        let bin = (normalized * self.num_bins.get() as f32) as usize;
        bin.min(self.num_bins.get().saturating_sub(1))
    }

    /// Convert bin index to center frequency.
    #[inline]
    pub fn bin_to_freq(&self, bin: usize) -> Hertz {
        self.bin_center(bin).unwrap_or(self.range.min())
    }

    /// Check if frequency is within grid range.
    #[inline]
    pub fn in_range(&self, freq: Hertz) -> bool {
        self.range.contains(freq)
    }

    /// Get bin bandwidth (alias for resolution).
    #[inline]
    pub fn bin_bandwidth(&self) -> Hertz {
        self.resolution
    }

    /// Calculate the bin range (start, end exclusive) covered by a frequency range.
    #[inline]
    pub fn bins_for_range(&self, freq_range: &FrequencyRange) -> (usize, usize) {
        let start = self.freq_to_bin(freq_range.min());
        let end = (self.freq_to_bin(freq_range.max()) + 1).min(self.num_bins.get());
        (start, end)
    }

    /// Get bin center frequencies as raw f32 slice for SIMD operations.
    #[inline]
    pub fn bin_centers_raw(&self) -> Vec<f32> {
        self.bin_centers.iter().map(|h| h.as_hz()).collect()
    }

    /// Get bin centers starting at given index as SIMD vector.
    #[cfg(feature = "simd")]
    pub fn bin_centers_simd(&self, start_bin: usize) -> std::simd::f32x8 {
        use std::simd::f32x8;
        let mut centers = [0.0f32; 8];
        for i in 0..8 {
            let bin = start_bin + i;
            centers[i] = if bin < self.num_bins.get() {
                self.bin_centers[bin].as_hz()
            } else {
                self.range.max().as_hz()
            };
        }
        f32x8::from_array(centers)
    }
}

// ============================================================================
// BinIndex: Type-Safe Bin Indexing (Optional)
// ============================================================================

/// A validated bin index within a frequency grid.
///
/// This type ensures the index is within bounds at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinIndex {
    index: usize,
    max_bins: NonZeroUsize,
}

impl BinIndex {
    /// Try to create a bin index.
    #[inline]
    pub fn try_new(index: usize, grid: &ValidatedFrequencyGrid) -> Option<Self> {
        (index < grid.num_bins()).then(|| Self {
            index,
            max_bins: grid.num_bins_nonzero(),
        })
    }

    /// Get the raw index.
    #[inline]
    pub fn get(&self) -> usize {
        self.index
    }

    /// Get the next bin, if within bounds.
    #[inline]
    pub fn next(&self) -> Option<Self> {
        let new_index = self.index + 1;
        (new_index < self.max_bins.get()).then(|| Self {
            index: new_index,
            max_bins: self.max_bins,
        })
    }

    /// Get the previous bin, if within bounds.
    #[inline]
    pub fn prev(&self) -> Option<Self> {
        self.index.checked_sub(1).map(|new_index| Self {
            index: new_index,
            max_bins: self.max_bins,
        })
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_range_construction() {
        let range = FrequencyRange::from_ghz(1.0, 2.0).unwrap();
        assert!((range.min().as_ghz() - 1.0).abs() < 1e-6);
        assert!((range.max().as_ghz() - 2.0).abs() < 1e-6);

        // Invalid: min >= max
        assert!(FrequencyRange::from_ghz(2.0, 1.0).is_none());
        assert!(FrequencyRange::from_ghz(1.0, 1.0).is_none());
    }

    #[test]
    fn test_frequency_range_bandwidth() {
        let range = FrequencyRange::from_mhz(100.0, 200.0).unwrap();
        assert!((range.bandwidth().as_mhz() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_range_center() {
        let range = FrequencyRange::from_ghz(1.0, 3.0).unwrap();
        assert!((range.center().as_ghz() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_range_contains() {
        let range = FrequencyRange::from_ghz(1.0, 2.0).unwrap();

        assert!(range.contains(Hertz::from_ghz(1.5).unwrap()));
        assert!(range.contains(Hertz::from_ghz(1.0).unwrap())); // inclusive
        assert!(range.contains(Hertz::from_ghz(2.0).unwrap())); // inclusive
        assert!(!range.contains(Hertz::from_ghz(0.5).unwrap()));
        assert!(!range.contains(Hertz::from_ghz(2.5).unwrap()));
    }

    #[test]
    fn test_frequency_range_overlaps() {
        let r1 = FrequencyRange::from_ghz(1.0, 2.0).unwrap();
        let r2 = FrequencyRange::from_ghz(1.5, 2.5).unwrap();
        let r3 = FrequencyRange::from_ghz(2.5, 3.0).unwrap();

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_frequency_range_intersection() {
        let r1 = FrequencyRange::from_ghz(1.0, 2.0).unwrap();
        let r2 = FrequencyRange::from_ghz(1.5, 2.5).unwrap();

        let intersection = r1.intersection(&r2).unwrap();
        assert!((intersection.min().as_ghz() - 1.5).abs() < 1e-6);
        assert!((intersection.max().as_ghz() - 2.0).abs() < 1e-6);

        // No intersection
        let r3 = FrequencyRange::from_ghz(3.0, 4.0).unwrap();
        assert!(r1.intersection(&r3).is_none());
    }

    #[test]
    fn test_validated_grid_creation() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap();

        assert_eq!(grid.num_bins(), 1000);
        assert!((grid.freq_min().as_hz() - 1e9).abs() < 1.0);
        assert!((grid.freq_max().as_hz() - 2e9).abs() < 1.0);

        // Invalid: zero bins
        assert!(ValidatedFrequencyGrid::from_params(1e9, 2e9, 0).is_none());

        // Invalid: min >= max
        assert!(ValidatedFrequencyGrid::from_params(2e9, 1e9, 100).is_none());
    }

    #[test]
    fn test_validated_grid_resolution() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap();
        let expected_resolution = 1e9 / 1000.0; // 1 MHz
        assert!((grid.resolution().as_hz() - expected_resolution).abs() < 1.0);
    }

    #[test]
    fn test_validated_grid_bin_centers() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap();

        // First bin center should be at min + resolution/2
        let first_center = grid.bin_center(0).unwrap();
        let expected = 1e9 + (1e9 / 1000.0) / 2.0;
        assert!((first_center.as_hz() - expected).abs() < 1.0);

        // Last bin center
        let last_center = grid.bin_center(999).unwrap();
        let expected = 2e9 - (1e9 / 1000.0) / 2.0;
        assert!((last_center.as_hz() - expected).abs() < 1.0);

        // Out of bounds
        assert!(grid.bin_center(1000).is_none());
    }

    #[test]
    fn test_validated_grid_freq_to_bin() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap();

        assert_eq!(grid.freq_to_bin(Hertz::new(1e9)), 0);
        assert_eq!(grid.freq_to_bin(Hertz::new(1.5e9)), 500);
        assert_eq!(grid.freq_to_bin(Hertz::new(2e9)), 999); // clamped

        // Below range
        assert_eq!(grid.freq_to_bin(Hertz::new(0.5e9)), 0);
    }

    #[test]
    fn test_validated_grid_bins_for_range() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap();
        let sub_range = FrequencyRange::from_hz(1.2e9, 1.5e9).unwrap();

        let (start, end) = grid.bins_for_range(&sub_range);
        assert_eq!(start, 200);
        assert_eq!(end, 501); // exclusive
    }

    #[test]
    fn test_bin_index() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 100).unwrap();

        let idx = BinIndex::try_new(50, &grid).unwrap();
        assert_eq!(idx.get(), 50);

        assert!(idx.next().is_some());
        assert!(idx.prev().is_some());

        // Out of bounds
        assert!(BinIndex::try_new(100, &grid).is_none());

        // Edge cases
        let first = BinIndex::try_new(0, &grid).unwrap();
        assert!(first.prev().is_none());

        let last = BinIndex::try_new(99, &grid).unwrap();
        assert!(last.next().is_none());
    }
}

// ============================================================================
// Frequency Bands - ITU Radio Spectrum Allocation
// ============================================================================

/// Propagation characteristics for different frequency bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationType {
    /// Earth-ionosphere waveguide (VLF)
    Waveguide,
    /// Ground wave + sky wave (LF/MF)
    GroundWave,
    /// Ionospheric skip/reflection (HF)
    Ionospheric,
    /// Line of sight (VHF/UHF/SHF)
    LineOfSight,
    /// Line of sight with atmospheric absorption (EHF)
    LineOfSightAtmospheric,
}

/// Trait for frequency band properties.
pub trait FrequencyBand {
    /// Minimum frequency of the band in Hz.
    fn min_freq_hz(&self) -> f32;

    /// Maximum frequency of the band in Hz.
    fn max_freq_hz(&self) -> f32;

    /// Default propagation type for this band.
    fn propagation_type(&self) -> PropagationType;

    /// Band name (e.g., "HF", "VHF").
    fn name(&self) -> &'static str;

    /// Check if a frequency is within this band.
    fn contains(&self, freq_hz: f32) -> bool {
        freq_hz >= self.min_freq_hz() && freq_hz <= self.max_freq_hz()
    }

    /// Get the band as a FrequencyRange.
    fn as_range(&self) -> Option<FrequencyRange> {
        FrequencyRange::from_hz(self.min_freq_hz(), self.max_freq_hz())
    }
}

/// Very Low Frequency (VLF): 3-30 kHz
/// Used for submarine communication, time signals.
/// Propagates via Earth-ionosphere waveguide.
#[derive(Debug, Clone, Copy, Default)]
pub struct VLFBand;

impl FrequencyBand for VLFBand {
    fn min_freq_hz(&self) -> f32 { 3e3 }
    fn max_freq_hz(&self) -> f32 { 30e3 }
    fn propagation_type(&self) -> PropagationType { PropagationType::Waveguide }
    fn name(&self) -> &'static str { "VLF" }
}

/// Low Frequency (LF): 30-300 kHz
/// Used for NDB beacons, LORAN, longwave broadcast.
/// Propagates via ground wave + nighttime sky wave.
#[derive(Debug, Clone, Copy, Default)]
pub struct LFBand;

impl FrequencyBand for LFBand {
    fn min_freq_hz(&self) -> f32 { 30e3 }
    fn max_freq_hz(&self) -> f32 { 300e3 }
    fn propagation_type(&self) -> PropagationType { PropagationType::GroundWave }
    fn name(&self) -> &'static str { "LF" }
}

/// Medium Frequency (MF): 300 kHz - 3 MHz
/// Used for AM broadcast, maritime, aviation NDBs.
/// Ground wave dominant by day, sky wave at night.
#[derive(Debug, Clone, Copy, Default)]
pub struct MFBand;

impl FrequencyBand for MFBand {
    fn min_freq_hz(&self) -> f32 { 300e3 }
    fn max_freq_hz(&self) -> f32 { 3e6 }
    fn propagation_type(&self) -> PropagationType { PropagationType::GroundWave }
    fn name(&self) -> &'static str { "MF" }
}

/// High Frequency (HF): 3-30 MHz
/// Used for shortwave, amateur, aviation, military.
/// Ionospheric skip enables worldwide propagation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HFBand;

impl FrequencyBand for HFBand {
    fn min_freq_hz(&self) -> f32 { 3e6 }
    fn max_freq_hz(&self) -> f32 { 30e6 }
    fn propagation_type(&self) -> PropagationType { PropagationType::Ionospheric }
    fn name(&self) -> &'static str { "HF" }
}

/// Very High Frequency (VHF): 30-300 MHz
/// Used for FM broadcast, TV, aviation, amateur.
/// Line of sight with tropospheric scatter.
#[derive(Debug, Clone, Copy, Default)]
pub struct VHFBand;

impl FrequencyBand for VHFBand {
    fn min_freq_hz(&self) -> f32 { 30e6 }
    fn max_freq_hz(&self) -> f32 { 300e6 }
    fn propagation_type(&self) -> PropagationType { PropagationType::LineOfSight }
    fn name(&self) -> &'static str { "VHF" }
}

/// Ultra High Frequency (UHF): 300 MHz - 3 GHz
/// Used for cellular, WiFi, GPS, TV.
/// Line of sight with multipath.
#[derive(Debug, Clone, Copy, Default)]
pub struct UHFBand;

impl FrequencyBand for UHFBand {
    fn min_freq_hz(&self) -> f32 { 300e6 }
    fn max_freq_hz(&self) -> f32 { 3e9 }
    fn propagation_type(&self) -> PropagationType { PropagationType::LineOfSight }
    fn name(&self) -> &'static str { "UHF" }
}

/// Super High Frequency (SHF): 3-30 GHz
/// Used for satellite, radar, microwave links.
/// Line of sight, rain fade significant.
#[derive(Debug, Clone, Copy, Default)]
pub struct SHFBand;

impl FrequencyBand for SHFBand {
    fn min_freq_hz(&self) -> f32 { 3e9 }
    fn max_freq_hz(&self) -> f32 { 30e9 }
    fn propagation_type(&self) -> PropagationType { PropagationType::LineOfSight }
    fn name(&self) -> &'static str { "SHF" }
}

/// Extremely High Frequency (EHF): 30-300 GHz
/// Used for 5G mmWave, point-to-point links.
/// Severe atmospheric absorption (O2 at 60 GHz, H2O at 183 GHz).
#[derive(Debug, Clone, Copy, Default)]
pub struct EHFBand;

impl FrequencyBand for EHFBand {
    fn min_freq_hz(&self) -> f32 { 30e9 }
    fn max_freq_hz(&self) -> f32 { 300e9 }
    fn propagation_type(&self) -> PropagationType { PropagationType::LineOfSightAtmospheric }
    fn name(&self) -> &'static str { "EHF" }
}

/// Determine the ITU frequency band for a given frequency.
#[inline]
pub fn frequency_to_band(freq_hz: f32) -> Option<&'static str> {
    if freq_hz < 3e3 { None }
    else if freq_hz < 30e3 { Some("VLF") }
    else if freq_hz < 300e3 { Some("LF") }
    else if freq_hz < 3e6 { Some("MF") }
    else if freq_hz < 30e6 { Some("HF") }
    else if freq_hz < 300e6 { Some("VHF") }
    else if freq_hz < 3e9 { Some("UHF") }
    else if freq_hz < 30e9 { Some("SHF") }
    else if freq_hz < 300e9 { Some("EHF") }
    else { None }
}

/// Determine the propagation type for a given frequency.
#[inline]
pub fn frequency_to_propagation(freq_hz: f32) -> PropagationType {
    if freq_hz < 30e3 { PropagationType::Waveguide }
    else if freq_hz < 3e6 { PropagationType::GroundWave }
    else if freq_hz < 30e6 { PropagationType::Ionospheric }
    else if freq_hz < 30e9 { PropagationType::LineOfSight }
    else { PropagationType::LineOfSightAtmospheric }
}

// ============================================================================
// Frequency Band Tests
// ============================================================================

#[cfg(test)]
mod band_tests {
    use super::*;

    #[test]
    fn test_hf_band() {
        let hf = HFBand;
        assert_eq!(hf.name(), "HF");
        assert_eq!(hf.min_freq_hz(), 3e6);
        assert_eq!(hf.max_freq_hz(), 30e6);
        assert_eq!(hf.propagation_type(), PropagationType::Ionospheric);

        assert!(hf.contains(14e6)); // 20m amateur
        assert!(!hf.contains(2e6)); // Below HF
        assert!(!hf.contains(50e6)); // VHF
    }

    #[test]
    fn test_all_bands_contiguous() {
        // Verify bands are contiguous
        let vlf = VLFBand;
        let lf = LFBand;
        let mf = MFBand;
        let hf = HFBand;
        let vhf = VHFBand;
        let uhf = UHFBand;
        let shf = SHFBand;
        let ehf = EHFBand;

        assert!((vlf.max_freq_hz() - lf.min_freq_hz()).abs() < 1.0);
        assert!((lf.max_freq_hz() - mf.min_freq_hz()).abs() < 1.0);
        assert!((mf.max_freq_hz() - hf.min_freq_hz()).abs() < 1.0);
        assert!((hf.max_freq_hz() - vhf.min_freq_hz()).abs() < 1.0);
        assert!((vhf.max_freq_hz() - uhf.min_freq_hz()).abs() < 1.0);
        assert!((uhf.max_freq_hz() - shf.min_freq_hz()).abs() < 1.0);
        assert!((shf.max_freq_hz() - ehf.min_freq_hz()).abs() < 1.0);
    }

    #[test]
    fn test_frequency_to_band() {
        assert_eq!(frequency_to_band(10e3), Some("VLF"));
        assert_eq!(frequency_to_band(100e3), Some("LF"));
        assert_eq!(frequency_to_band(1e6), Some("MF"));
        assert_eq!(frequency_to_band(14e6), Some("HF"));
        assert_eq!(frequency_to_band(100e6), Some("VHF"));
        assert_eq!(frequency_to_band(900e6), Some("UHF"));
        assert_eq!(frequency_to_band(10e9), Some("SHF"));
        assert_eq!(frequency_to_band(60e9), Some("EHF"));
    }

    #[test]
    fn test_frequency_to_propagation() {
        assert_eq!(frequency_to_propagation(10e3), PropagationType::Waveguide);
        assert_eq!(frequency_to_propagation(1e6), PropagationType::GroundWave);
        assert_eq!(frequency_to_propagation(14e6), PropagationType::Ionospheric);
        assert_eq!(frequency_to_propagation(900e6), PropagationType::LineOfSight);
        assert_eq!(frequency_to_propagation(60e9), PropagationType::LineOfSightAtmospheric);
    }

    #[test]
    fn test_band_as_range() {
        let hf = HFBand;
        let range = hf.as_range().unwrap();

        assert!((range.min().as_hz() - 3e6).abs() < 1.0);
        assert!((range.max().as_hz() - 30e6).abs() < 1.0);
    }
}
