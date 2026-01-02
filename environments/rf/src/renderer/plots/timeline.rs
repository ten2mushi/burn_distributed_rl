//! Agent frequency timeline plot.
//!
//! Shows jammer and CR frequency traces over time with collision markers.

use crate::renderer::TimelineConfig;

/// Timeline plot generator.
pub struct TimelinePlot {
    config: TimelineConfig,
}

impl TimelinePlot {
    /// Create a new timeline plot with the given configuration.
    pub fn new(config: TimelineConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TimelineConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: TimelineConfig) {
        self.config = config;
    }
}

impl Default for TimelinePlot {
    fn default() -> Self {
        Self::new(TimelineConfig::default())
    }
}
