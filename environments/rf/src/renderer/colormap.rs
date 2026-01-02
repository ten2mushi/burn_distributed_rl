//! Colormap implementations for visualization.
//!
//! Uses interpolation between key colors for smooth gradients.

/// Trait for mapping scalar values to RGB colors.
pub trait Colormap: Send + Sync {
    /// Map a value in [0, 1] to an RGB color.
    fn map(&self, value: f32) -> [u8; 3];

    /// Map a value in [0, 1] to an ARGB u32 (for minifb).
    fn map_argb(&self, value: f32) -> u32 {
        let [r, g, b] = self.map(value);
        0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }
}

/// Viridis colormap - perceptually uniform, colorblind-friendly.
pub struct Viridis;

/// Inferno colormap - high contrast, good for power displays.
pub struct Inferno;

/// Grayscale colormap - simple black to white.
pub struct Grayscale;

/// Turbo colormap - rainbow-like but more perceptually uniform.
pub struct Turbo;

/// Plasma colormap - similar to Inferno but with purple-pink tones.
pub struct Plasma;

// Key colors for interpolation (simpler and more maintainable than 256-entry LUTs)

// Viridis: dark purple -> blue -> teal -> green -> yellow
const VIRIDIS_KEYS: &[(f32, [u8; 3])] = &[
    (0.00, [68, 1, 84]),      // Dark purple
    (0.25, [59, 82, 139]),    // Blue
    (0.50, [33, 145, 140]),   // Teal
    (0.75, [94, 201, 98]),    // Green
    (1.00, [253, 231, 37]),   // Yellow
];

// Inferno: black -> purple -> red -> orange -> yellow -> white
const INFERNO_KEYS: &[(f32, [u8; 3])] = &[
    (0.00, [0, 0, 4]),        // Black
    (0.20, [40, 11, 84]),     // Dark purple
    (0.40, [101, 21, 110]),   // Purple
    (0.60, [182, 55, 84]),    // Red
    (0.80, [243, 132, 48]),   // Orange
    (1.00, [252, 255, 164]),  // Light yellow
];

// Turbo: blue -> cyan -> green -> yellow -> red
const TURBO_KEYS: &[(f32, [u8; 3])] = &[
    (0.00, [48, 18, 59]),     // Dark blue
    (0.20, [69, 138, 252]),   // Blue
    (0.40, [40, 200, 220]),   // Cyan
    (0.50, [90, 220, 100]),   // Green
    (0.60, [170, 220, 50]),   // Yellow-green
    (0.80, [250, 170, 50]),   // Orange
    (1.00, [122, 4, 3]),      // Dark red
];

// Plasma: blue-purple -> pink -> orange -> yellow
const PLASMA_KEYS: &[(f32, [u8; 3])] = &[
    (0.00, [13, 8, 135]),     // Dark blue
    (0.25, [126, 3, 168]),    // Purple
    (0.50, [204, 71, 120]),   // Pink
    (0.75, [248, 149, 64]),   // Orange
    (1.00, [240, 249, 33]),   // Yellow
];

/// Interpolate between key colors.
fn interpolate_colormap(keys: &[(f32, [u8; 3])], value: f32) -> [u8; 3] {
    let v = value.clamp(0.0, 1.0);

    // Find the two keys to interpolate between
    let mut lower_idx = 0;
    for (i, &(t, _)) in keys.iter().enumerate() {
        if t <= v {
            lower_idx = i;
        } else {
            break;
        }
    }

    let upper_idx = (lower_idx + 1).min(keys.len() - 1);

    if lower_idx == upper_idx {
        return keys[lower_idx].1;
    }

    let (t0, c0) = keys[lower_idx];
    let (t1, c1) = keys[upper_idx];

    // Linear interpolation
    let t = (v - t0) / (t1 - t0);

    [
        (c0[0] as f32 + t * (c1[0] as f32 - c0[0] as f32)) as u8,
        (c0[1] as f32 + t * (c1[1] as f32 - c0[1] as f32)) as u8,
        (c0[2] as f32 + t * (c1[2] as f32 - c0[2] as f32)) as u8,
    ]
}

impl Colormap for Viridis {
    fn map(&self, value: f32) -> [u8; 3] {
        interpolate_colormap(VIRIDIS_KEYS, value)
    }
}

impl Colormap for Inferno {
    fn map(&self, value: f32) -> [u8; 3] {
        interpolate_colormap(INFERNO_KEYS, value)
    }
}

impl Colormap for Grayscale {
    fn map(&self, value: f32) -> [u8; 3] {
        let v = (value.clamp(0.0, 1.0) * 255.0) as u8;
        [v, v, v]
    }
}

impl Colormap for Turbo {
    fn map(&self, value: f32) -> [u8; 3] {
        interpolate_colormap(TURBO_KEYS, value)
    }
}

impl Colormap for Plasma {
    fn map(&self, value: f32) -> [u8; 3] {
        interpolate_colormap(PLASMA_KEYS, value)
    }
}

/// Available colormap types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapType {
    #[default]
    Viridis,
    Inferno,
    Grayscale,
    Turbo,
    Plasma,
}

impl ColormapType {
    /// Get the colormap implementation.
    pub fn get(&self) -> Box<dyn Colormap> {
        match self {
            ColormapType::Viridis => Box::new(Viridis),
            ColormapType::Inferno => Box::new(Inferno),
            ColormapType::Grayscale => Box::new(Grayscale),
            ColormapType::Turbo => Box::new(Turbo),
            ColormapType::Plasma => Box::new(Plasma),
        }
    }

    /// Map a value directly using this colormap type.
    pub fn map(&self, value: f32) -> [u8; 3] {
        match self {
            ColormapType::Viridis => Viridis.map(value),
            ColormapType::Inferno => Inferno.map(value),
            ColormapType::Grayscale => Grayscale.map(value),
            ColormapType::Turbo => Turbo.map(value),
            ColormapType::Plasma => Plasma.map(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viridis_bounds() {
        let cm = Viridis;
        let start = cm.map(0.0);
        let end = cm.map(1.0);
        // Start should be dark purple
        assert!(start[0] < 100 && start[2] > 50);
        // End should be bright yellow
        assert!(end[0] > 200 && end[1] > 200);
    }

    #[test]
    fn test_colormap_clamp() {
        let cm = Viridis;
        // Values outside [0, 1] should be clamped
        assert_eq!(cm.map(-0.5), cm.map(0.0));
        assert_eq!(cm.map(1.5), cm.map(1.0));
    }

    #[test]
    fn test_grayscale_monotonic() {
        let cm = Grayscale;
        let mut prev = 0u8;
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let [r, g, b] = cm.map(v);
            assert_eq!(r, g);
            assert_eq!(g, b);
            assert!(r >= prev);
            prev = r;
        }
    }

    #[test]
    fn test_argb_conversion() {
        let cm = Viridis;
        let argb = cm.map_argb(0.0);
        // Should have alpha = 0xFF
        assert_eq!((argb >> 24) & 0xFF, 0xFF);
    }

    #[test]
    fn test_interpolation_midpoint() {
        // At 0.5, should be approximately teal for Viridis
        let color = Viridis.map(0.5);
        // Teal-ish: moderate R, higher G, high B
        assert!(color[1] > 100); // Green component should be significant
    }
}
