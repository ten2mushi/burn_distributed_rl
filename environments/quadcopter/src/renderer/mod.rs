//! Real-time 3D visualization for quadcopter SIMD environments.
//!
//! This module provides visualization capabilities for the quadcopter environment,
//! enabling real-time display of all parallel drone simulations in a unified 3D scene.
//!
//! # Features
//!
//! - `render`: Base rendering (PNG export via plotters)
//! - `render-realtime`: Live window visualization (minifb)
//! - `render-gif`: Animated GIF recording
//!
//! # Architecture
//!
//! Uses the Snapshot-History-Renderer pattern:
//! 1. `MultiEnvSnapshot` captures state of all environments at each timestep
//! 2. `RollingHistory` maintains temporal buffer of snapshots and trajectories
//! 3. `RealtimeWindow` renders the scene using perspective projection
//!
//! # Example
//!
//! ```rust,ignore
//! use quadcopter_env::renderer::{
//!     MultiEnvSnapshot, RollingHistory, RealtimeWindow, VisualizationConfig,
//! };
//!
//! let mut history = RollingHistory::new(8, 200);
//! let mut window = RealtimeWindow::new("Training Visualization", config)?;
//!
//! while window.is_open() {
//!     env.step(&actions);
//!     history.push(MultiEnvSnapshot::capture(&env, step));
//!     window.update(&history)?;
//! }
//! ```

pub mod config;
pub mod snapshot;
pub mod history;
pub mod projection;
pub mod drone_shape;

pub mod backends;
pub mod plots;

#[cfg(feature = "render-realtime")]
pub mod realtime;

// Re-exports for convenience
pub use config::{
    CameraConfig, DroneStyleConfig, SceneConfig, TrajectoryConfig, VisualizationConfig,
};
pub use snapshot::{DroneSnapshot, MultiEnvSnapshot};
pub use history::{DroneTrajectory, RollingHistory};
pub use projection::PerspectiveProjection;

#[cfg(feature = "render-realtime")]
pub use realtime::RealtimeWindow;

/// Environment colors for up to 8 parallel simulations.
/// Each drone gets a distinct, visually separable color.
pub const ENV_COLORS: [[u8; 3]; 8] = [
    [220, 50, 50],   // Red
    [255, 150, 0],   // Orange
    [230, 200, 0],   // Yellow
    [50, 180, 50],   // Green
    [0, 200, 200],   // Cyan
    [50, 100, 220],  // Blue
    [150, 50, 200],  // Purple
    [220, 100, 150], // Pink
];

/// Get color for a specific environment index.
#[inline]
pub fn env_color(idx: usize) -> [u8; 3] {
    ENV_COLORS[idx % ENV_COLORS.len()]
}
