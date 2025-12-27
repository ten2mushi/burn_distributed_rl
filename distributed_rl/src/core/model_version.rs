//! Model versioning for importance sampling in IMPALA.
//!
//! In distributed training, actors may collect experience with stale policies.
//! Model versioning tracks which policy version generated each trajectory,
//! enabling V-trace off-policy correction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Atomic version counter for models.
///
/// Thread-safe counter that tracks model versions. Each time the learner
/// updates the policy, it increments the version counter.
#[derive(Debug)]
pub struct VersionCounter {
    version: AtomicU64,
}

impl VersionCounter {
    /// Create a new version counter starting at 0.
    pub fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
        }
    }

    /// Increment and return new version.
    ///
    /// Uses SeqCst ordering for strict consistency across threads.
    pub fn increment(&self) -> u64 {
        self.version.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get current version without incrementing.
    pub fn current(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Reset to version 0.
    pub fn reset(&self) {
        self.version.store(0, Ordering::SeqCst);
    }
}

impl Default for VersionCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Versioned model wrapper.
///
/// Associates a model with its version number, enabling importance
/// sampling calculations in V-trace.
#[derive(Clone)]
pub struct VersionedModel<M: Clone> {
    /// The model
    pub model: M,
    /// Version when this model was created
    pub version: u64,
}

impl<M: Clone> VersionedModel<M> {
    /// Create a new versioned model.
    pub fn new(model: M, version: u64) -> Self {
        Self { model, version }
    }

    /// Get model reference.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get model version.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Check if this model is older than a given version.
    pub fn is_stale(&self, current_version: u64) -> bool {
        self.version < current_version
    }

    /// Calculate version lag (how many versions behind).
    pub fn lag(&self, current_version: u64) -> u64 {
        current_version.saturating_sub(self.version)
    }
}

impl<M: Clone + std::fmt::Debug> std::fmt::Debug for VersionedModel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionedModel")
            .field("version", &self.version)
            .field("model", &self.model)
            .finish()
    }
}

/// Shared version counter (thread-safe reference).
pub type SharedVersionCounter = Arc<VersionCounter>;

/// Create a new shared version counter.
pub fn version_counter() -> SharedVersionCounter {
    Arc::new(VersionCounter::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_counter_new() {
        let counter = VersionCounter::new();
        assert_eq!(counter.current(), 0);
    }

    #[test]
    fn test_version_counter_increment() {
        let counter = VersionCounter::new();
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.current(), 2);
    }

    #[test]
    fn test_version_counter_reset() {
        let counter = VersionCounter::new();
        counter.increment();
        counter.increment();
        counter.reset();
        assert_eq!(counter.current(), 0);
    }

    #[test]
    fn test_versioned_model() {
        let model = vec![1.0f32, 2.0, 3.0];
        let versioned = VersionedModel::new(model.clone(), 5);

        assert_eq!(versioned.version(), 5);
        assert_eq!(versioned.model(), &model);
    }

    #[test]
    fn test_versioned_model_staleness() {
        let model = vec![1.0f32];
        let versioned = VersionedModel::new(model, 5);

        assert!(!versioned.is_stale(5));
        assert!(!versioned.is_stale(4));
        assert!(versioned.is_stale(6));
        assert!(versioned.is_stale(10));
    }

    #[test]
    fn test_versioned_model_lag() {
        let model = vec![1.0f32];
        let versioned = VersionedModel::new(model, 5);

        assert_eq!(versioned.lag(5), 0);
        assert_eq!(versioned.lag(10), 5);
        assert_eq!(versioned.lag(3), 0); // saturating_sub
    }

    #[test]
    fn test_shared_version_counter() {
        let counter = version_counter();
        let counter2 = Arc::clone(&counter);

        counter.increment();
        counter2.increment();

        assert_eq!(counter.current(), 2);
        assert_eq!(counter2.current(), 2);
    }
}
