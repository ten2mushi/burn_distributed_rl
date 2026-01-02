//! Waterfall spectrogram plot.
//!
//! Time-frequency heatmap showing PSD evolution over time.

use crate::renderer::WaterfallConfig;

/// Waterfall plot generator.
pub struct WaterfallPlot {
    config: WaterfallConfig,
}

impl WaterfallPlot {
    /// Create a new waterfall plot with the given configuration.
    pub fn new(config: WaterfallConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &WaterfallConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: WaterfallConfig) {
        self.config = config;
    }
}

impl Default for WaterfallPlot {
    fn default() -> Self {
        Self::new(WaterfallConfig::default())
    }
}
