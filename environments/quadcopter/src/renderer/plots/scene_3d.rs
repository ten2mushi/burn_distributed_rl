//! 3D scene composition and rendering.

use crate::renderer::backends::{ImageBackend, RenderResult};
use crate::renderer::config::VisualizationConfig;
use crate::renderer::history::RollingHistory;
use crate::renderer::projection::PerspectiveProjection;

/// Render a complete 3D scene frame.
pub fn render_scene(
    backend: &mut ImageBackend,
    history: &RollingHistory,
    projection: &PerspectiveProjection,
    config: &VisualizationConfig,
) -> RenderResult<()> {
    backend.render(history, projection, config)
}
