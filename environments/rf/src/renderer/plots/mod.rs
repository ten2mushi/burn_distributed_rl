//! Plot implementations for RF environment visualization.
//!
//! This module provides high-level plot generation functions that
//! use the rendering backends to produce visualizations.

mod waterfall;
mod spectrum;
mod entity_map;
mod timeline;

pub use waterfall::WaterfallPlot;
pub use spectrum::SpectrumPlot;
pub use entity_map::EntityMapPlot;
pub use timeline::TimelinePlot;

use crate::renderer::{PlotType, RenderConfig, RenderResult, RollingHistory};
use crate::renderer::backends::RenderBackend;

/// Generate a plot based on plot type.
pub fn render_plot(
    plot_type: PlotType,
    history: &RollingHistory,
    config: &RenderConfig,
) -> RenderResult<Vec<u8>> {
    use crate::renderer::backends::ImageBackend;

    let backend = ImageBackend::new(config.image_size.0, config.image_size.1);

    match plot_type {
        PlotType::Waterfall => {
            let matrix = history.psd_matrix();
            backend.render_waterfall_to_buffer(&matrix, &config.waterfall)
        }
        PlotType::SpectrumSnapshot => {
            let snapshot = history.latest().ok_or(crate::renderer::RenderError::EmptyHistory)?;
            backend.render_spectrum(&snapshot, &config.spectrum)
        }
        PlotType::EntityMap => {
            let snapshot = history.latest().ok_or(crate::renderer::RenderError::EmptyHistory)?;
            backend.render_entity_map(&snapshot, &config.entity_map)
        }
        PlotType::FrequencyTimeline => {
            let traces = history.agent_freq_traces();
            let collisions = history.find_collisions(config.timeline.collision_threshold_hz);
            backend.render_timeline(&traces, &collisions, &config.timeline)
        }
    }
}
