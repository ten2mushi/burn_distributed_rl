//! Entity position map plot.
//!
//! 2D bird's-eye view showing entities and agents with icons.

use crate::renderer::EntityMapConfig;

/// Entity map plot generator.
pub struct EntityMapPlot {
    config: EntityMapConfig,
}

impl EntityMapPlot {
    /// Create a new entity map plot with the given configuration.
    pub fn new(config: EntityMapConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &EntityMapConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: EntityMapConfig) {
        self.config = config;
    }
}

impl Default for EntityMapPlot {
    fn default() -> Self {
        Self::new(EntityMapConfig::default())
    }
}
