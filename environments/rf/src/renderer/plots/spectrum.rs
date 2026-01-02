//! Spectrum snapshot plot.
//!
//! Instantaneous PSD vs frequency with entity/agent markers.

use crate::renderer::SpectrumConfig;

/// Spectrum snapshot plot generator.
pub struct SpectrumPlot {
    config: SpectrumConfig,
}

impl SpectrumPlot {
    /// Create a new spectrum plot with the given configuration.
    pub fn new(config: SpectrumConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SpectrumConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: SpectrumConfig) {
        self.config = config;
    }
}

impl Default for SpectrumPlot {
    fn default() -> Self {
        Self::new(SpectrumConfig::default())
    }
}
