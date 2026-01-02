//! RF Environment Renderer
//!
//! Visualization subsystem for the RF spectrum environment, providing:
//!
//! - **Waterfall spectrograms**: Time-frequency heatmaps of PSD evolution
//! - **Spectrum snapshots**: Instantaneous PSD plots with entity markers
//! - **Entity position maps**: 2D bird's-eye view of the RF world
//! - **Frequency timelines**: Agent frequency traces over time
//!
//! # Features
//!
//! This module is feature-gated to avoid unnecessary dependencies:
//!
//! - `render`: Core rendering via plotters (PNG output)
//! - `render-terminal`: ASCII art visualization
//! - `render-html`: Interactive HTML/Plotly.js dashboards
//! - `render-realtime`: Live window via minifb
//! - `render-gif`: Animated GIF export
//!
//! # Example
//!
//! ```ignore
//! use rf_environment::renderer::{RFRenderer, RenderConfig, PlotType};
//!
//! // Create renderer with default config
//! let mut renderer = RFRenderer::new(RenderConfig::default());
//!
//! // During training loop
//! for step in 0..total_steps {
//!     let (state, rewards) = env.step(&actions);
//!
//!     // Capture current state
//!     renderer.capture(&env);
//!
//!     // Render periodically
//!     if renderer.should_render(step) {
//!         renderer.render_all_plots("./renders", &format!("step_{:06}", step))?;
//!     }
//! }
//! ```

// Core modules - always available
mod colormap;
mod config;
mod error;
mod history;
mod snapshot;

// Re-export core types
pub use colormap::{Colormap, ColormapType, Grayscale, Inferno, Plasma, Turbo, Viridis};
pub use config::{
    BackendType, EntityMapConfig, FreqUnit, GifConfig, PlotType, RealtimeConfig, RenderConfig,
    SpectrumConfig, TimelineConfig, TimeUnit, WaterfallConfig,
};
pub use error::{RenderError, RenderResult};
pub use history::{AgentTraces, CollisionEvent, EntityTrace, HistoryStats, PsdMatrix, RollingHistory};
pub use snapshot::{AgentSnapshot, AgentType, EntityIcon, EntitySnapshot, EntityType, EnvSnapshot};

// Backend modules
#[cfg(feature = "render")]
pub mod backends;

// Plot modules
#[cfg(feature = "render")]
pub mod plots;

// Main renderer API (requires render feature)
#[cfg(feature = "render")]
mod renderer;

#[cfg(feature = "render")]
pub use renderer::RFRenderer;

// Real-time window (requires render-realtime feature)
#[cfg(feature = "render-realtime")]
mod realtime;

#[cfg(feature = "render-realtime")]
pub use realtime::RealtimeWindow;

// Re-export minifb::Key for examples that need keyboard input
#[cfg(feature = "render-realtime")]
pub use minifb::Key;

// GIF export (requires render-gif feature)
#[cfg(feature = "render-gif")]
mod gif_recorder;

#[cfg(feature = "render-gif")]
pub use gif_recorder::GifRecorder;

// Terminal backend (requires render-terminal feature)
#[cfg(feature = "render-terminal")]
mod terminal;

#[cfg(feature = "render-terminal")]
pub use terminal::TerminalRenderer;

// HTML backend (requires render-html feature)
#[cfg(feature = "render-html")]
mod html;

#[cfg(feature = "render-html")]
pub use html::HtmlRenderer;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        AgentSnapshot, AgentType, ColormapType, EntitySnapshot, EntityType, EnvSnapshot, FreqUnit,
        PlotType, RenderConfig, RenderError, RenderResult, RollingHistory,
    };

    #[cfg(feature = "render")]
    pub use super::RFRenderer;

    #[cfg(feature = "render-realtime")]
    pub use super::RealtimeWindow;

    #[cfg(feature = "render-gif")]
    pub use super::GifRecorder;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports_available() {
        // Ensure core types are exported
        let _config = RenderConfig::default();
        let _colormap = ColormapType::Viridis;
        let _plot = PlotType::Waterfall;
    }

    #[test]
    fn test_history_integration() {
        let mut history = RollingHistory::new(10);

        let snapshot = EnvSnapshot::new(
            0,
            vec![1.0, 2.0, 3.0],
            vec![1e9, 2e9, 3e9],
            -100.0,
            (1000.0, 1000.0),
            (1e9, 3e9),
        );

        history.push(snapshot);
        assert_eq!(history.len(), 1);
    }
}
