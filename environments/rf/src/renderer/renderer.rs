//! Main RFRenderer API.
//!
//! Provides the primary interface for capturing environment state
//! and rendering visualizations during training.

use std::fs;
use std::path::Path;

use crate::renderer::{
    backends::ImageBackend,
    config::{PlotType, RenderConfig},
    error::{RenderError, RenderResult},
    history::RollingHistory,
    snapshot::EnvSnapshot,
};

/// Main renderer for RF environment visualization.
///
/// # Example
///
/// ```ignore
/// use rf_environment::renderer::{RFRenderer, RenderConfig};
///
/// let mut renderer = RFRenderer::new(RenderConfig::default());
///
/// // During training
/// for step in 0..total_steps {
///     let (state, rewards) = env.step(&actions);
///
///     renderer.capture(&env);
///
///     if renderer.should_render(step) {
///         renderer.render_all_plots("./renders", &format!("step_{:06}", step))?;
///     }
/// }
/// ```
pub struct RFRenderer {
    /// Renderer configuration.
    config: RenderConfig,
    /// Rolling history buffer.
    history: RollingHistory,
    /// Image backend for PNG rendering.
    image_backend: ImageBackend,
    /// Current step counter.
    current_step: u64,
}

impl RFRenderer {
    /// Create a new renderer with the given configuration.
    pub fn new(config: RenderConfig) -> Self {
        let history = RollingHistory::new(config.history_capacity);
        let image_backend = ImageBackend::new(config.image_size.0, config.image_size.1);

        Self {
            config,
            history,
            image_backend,
            current_step: 0,
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &RenderConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut RenderConfig {
        &mut self.config
    }

    /// Get the history buffer.
    pub fn history(&self) -> &RollingHistory {
        &self.history
    }

    /// Get the current step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Capture a snapshot of the environment state.
    ///
    /// This should be called once per step to build up history
    /// for waterfall and timeline plots.
    pub fn capture(&mut self, snapshot: EnvSnapshot) {
        self.current_step = snapshot.step;
        self.history.push(snapshot);
    }

    /// Check if rendering should occur at this step.
    ///
    /// Returns true if `render_every > 0` and `step % render_every == 0`.
    pub fn should_render(&self, step: u64) -> bool {
        if self.config.render_every == 0 {
            false
        } else {
            step % self.config.render_every == 0
        }
    }

    /// Render all configured plots to separate PNG files.
    ///
    /// Files are named: `{prefix}_{plot_suffix}.png`
    ///
    /// # Arguments
    ///
    /// * `dir` - Output directory
    /// * `prefix` - Filename prefix (e.g., "step_000100")
    ///
    /// # Returns
    ///
    /// List of paths to created files.
    pub fn render_all_plots(
        &self,
        dir: impl AsRef<Path>,
        prefix: &str,
    ) -> RenderResult<Vec<std::path::PathBuf>> {
        let dir = dir.as_ref();

        // Ensure directory exists
        fs::create_dir_all(dir)?;

        let mut paths = Vec::new();

        for plot_type in &self.config.plots {
            let filename = format!("{}_{}.png", prefix, plot_type.suffix());
            let path = dir.join(&filename);

            self.render_plot_to_file(*plot_type, &path)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Render a specific plot to a file.
    pub fn render_plot_to_file(&self, plot_type: PlotType, path: impl AsRef<Path>) -> RenderResult<()> {
        match plot_type {
            PlotType::Waterfall => {
                let matrix = self.history.psd_matrix();
                if matrix.num_time_steps == 0 {
                    return Err(RenderError::EmptyHistory);
                }
                self.image_backend
                    .render_waterfall_to_file(&matrix, &self.config.waterfall, path)
            }
            PlotType::SpectrumSnapshot => {
                let snapshot = self
                    .history
                    .latest()
                    .ok_or(RenderError::EmptyHistory)?;
                self.image_backend
                    .render_spectrum_to_file(snapshot, &self.config.spectrum, path)
            }
            PlotType::EntityMap => {
                let snapshot = self
                    .history
                    .latest()
                    .ok_or(RenderError::EmptyHistory)?;
                self.image_backend
                    .render_entity_map_to_file(snapshot, &self.config.entity_map, path)
            }
            PlotType::FrequencyTimeline => {
                let traces = self.history.agent_freq_traces();
                let collisions = self
                    .history
                    .find_collisions(self.config.timeline.collision_threshold_hz);
                self.image_backend
                    .render_timeline_to_file(&traces, &collisions, &self.config.timeline, path)
            }
        }
    }

    /// Render a specific plot to RGB bytes.
    pub fn render_plot_to_bytes(&self, plot_type: PlotType) -> RenderResult<Vec<u8>> {
        use crate::renderer::backends::RenderBackend;

        match plot_type {
            PlotType::Waterfall => {
                let matrix = self.history.psd_matrix();
                if matrix.num_time_steps == 0 {
                    return Err(RenderError::EmptyHistory);
                }
                self.image_backend.render_waterfall(&matrix, &self.config.waterfall)
            }
            PlotType::SpectrumSnapshot => {
                let snapshot = self
                    .history
                    .latest()
                    .ok_or(RenderError::EmptyHistory)?;
                self.image_backend.render_spectrum(snapshot, &self.config.spectrum)
            }
            PlotType::EntityMap => {
                let snapshot = self
                    .history
                    .latest()
                    .ok_or(RenderError::EmptyHistory)?;
                self.image_backend.render_entity_map(snapshot, &self.config.entity_map)
            }
            PlotType::FrequencyTimeline => {
                let traces = self.history.agent_freq_traces();
                let collisions = self
                    .history
                    .find_collisions(self.config.timeline.collision_threshold_hz);
                self.image_backend.render_timeline(&traces, &collisions, &self.config.timeline)
            }
        }
    }

    /// Clear the history buffer.
    ///
    /// Call this when resetting the environment between episodes
    /// if you want to start fresh visualizations.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_step = 0;
    }

    /// Get history statistics.
    pub fn stats(&self) -> crate::renderer::HistoryStats {
        self.history.stats()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get number of snapshots in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

impl Default for RFRenderer {
    fn default() -> Self {
        Self::new(RenderConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::EnvSnapshot;

    fn make_test_snapshot(step: u64) -> EnvSnapshot {
        EnvSnapshot::new(
            step,
            vec![0.001; 64],
            (0..64).map(|i| 1e9 + i as f32 * 10e6).collect(),
            -100.0,
            (1000.0, 1000.0),
            (1e9, 2e9),
        )
    }

    #[test]
    fn test_renderer_creation() {
        let renderer = RFRenderer::new(RenderConfig::default());
        assert!(renderer.is_empty());
        assert_eq!(renderer.history_len(), 0);
    }

    #[test]
    fn test_capture_and_history() {
        let mut renderer = RFRenderer::new(RenderConfig::default());

        renderer.capture(make_test_snapshot(0));
        renderer.capture(make_test_snapshot(1));
        renderer.capture(make_test_snapshot(2));

        assert_eq!(renderer.history_len(), 3);
        assert_eq!(renderer.current_step(), 2);
    }

    #[test]
    fn test_should_render() {
        let config = RenderConfig::default().render_every(100);
        let renderer = RFRenderer::new(config);

        // Step 0 triggers render (0 % 100 == 0)
        assert!(renderer.should_render(0));
        assert!(!renderer.should_render(50));
        assert!(renderer.should_render(100));
        assert!(renderer.should_render(200));
        assert!(!renderer.should_render(201));
    }

    #[test]
    fn test_reset() {
        let mut renderer = RFRenderer::new(RenderConfig::default());

        renderer.capture(make_test_snapshot(0));
        renderer.capture(make_test_snapshot(1));

        renderer.reset();

        assert!(renderer.is_empty());
        assert_eq!(renderer.current_step(), 0);
    }
}
