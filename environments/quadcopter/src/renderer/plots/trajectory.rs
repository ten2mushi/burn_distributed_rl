//! Trajectory trail rendering utilities.

use crate::renderer::config::FadeStyle;
use crate::renderer::history::DroneTrajectory;

/// Compute alpha value for a point in the trajectory.
pub fn compute_alpha(
    point_index: usize,
    total_points: usize,
    fade_style: FadeStyle,
    min_alpha: f32,
) -> f32 {
    if total_points == 0 {
        return min_alpha;
    }

    let t = point_index as f32 / total_points as f32;

    match fade_style {
        FadeStyle::Linear => min_alpha + (1.0 - min_alpha) * t,
        FadeStyle::Exponential => min_alpha + (1.0 - min_alpha) * t * t,
        FadeStyle::None => 1.0,
    }
}

/// Get interpolated color with alpha.
pub fn blend_color(color: [u8; 3], alpha: f32, bg: [u8; 3]) -> [u8; 3] {
    let inv_alpha = 1.0 - alpha;
    [
        (color[0] as f32 * alpha + bg[0] as f32 * inv_alpha) as u8,
        (color[1] as f32 * alpha + bg[1] as f32 * inv_alpha) as u8,
        (color[2] as f32 * alpha + bg[2] as f32 * inv_alpha) as u8,
    ]
}

/// Sample trajectory at regular intervals for smooth rendering.
pub fn sample_trajectory(
    trajectory: &DroneTrajectory,
    max_points: usize,
) -> Vec<[f32; 3]> {
    let positions: Vec<_> = trajectory.iter().cloned().collect();

    if positions.len() <= max_points {
        return positions;
    }

    // Subsample evenly
    let step = positions.len() as f32 / max_points as f32;
    (0..max_points)
        .map(|i| {
            let idx = (i as f32 * step) as usize;
            positions[idx.min(positions.len() - 1)]
        })
        .collect()
}
