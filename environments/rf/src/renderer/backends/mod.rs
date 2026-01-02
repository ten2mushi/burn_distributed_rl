//! Rendering backends for different output formats.
//!
//! Each backend implements the same rendering logic but outputs
//! to different targets (PNG, terminal, HTML, etc.).

mod image;

pub use image::ImageBackend;

/// Trait for rendering backends.
pub trait RenderBackend {
    /// The output type produced by this backend.
    type Output;

    /// Render a waterfall spectrogram.
    fn render_waterfall(
        &self,
        psd_matrix: &crate::renderer::PsdMatrix,
        config: &crate::renderer::WaterfallConfig,
    ) -> crate::renderer::RenderResult<Self::Output>;

    /// Render a spectrum snapshot.
    fn render_spectrum(
        &self,
        snapshot: &crate::renderer::EnvSnapshot,
        config: &crate::renderer::SpectrumConfig,
    ) -> crate::renderer::RenderResult<Self::Output>;

    /// Render an entity position map.
    fn render_entity_map(
        &self,
        snapshot: &crate::renderer::EnvSnapshot,
        config: &crate::renderer::EntityMapConfig,
    ) -> crate::renderer::RenderResult<Self::Output>;

    /// Render a frequency timeline.
    fn render_timeline(
        &self,
        traces: &crate::renderer::AgentTraces,
        collisions: &[crate::renderer::CollisionEvent],
        config: &crate::renderer::TimelineConfig,
    ) -> crate::renderer::RenderResult<Self::Output>;
}
