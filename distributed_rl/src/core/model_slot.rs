//! Single-slot model transfer mechanism for distributed training.
//!
//! Provides memory-efficient model transfer between Learner and Actor threads
//! without accumulation of stale models. Uses swap semantics to ensure
//! at most one pending update exists at any time.
//!
//! # Design
//!
//! This pattern is specifically designed for Burn models which are:
//! - `Send`: Can be transferred between threads
//! - NOT `Sync`: Cannot be shared between threads (due to `Param<T>` containing `OnceCell`)
//!
//! # Memory Characteristics
//!
//! Unlike bounded channels that can queue multiple models, `ModelSlot`
//! uses swap semantics: new publications overwrite pending updates.
//!
//! ```text
//! Memory invariant: slot.pending <= 1 model at all times
//! ```

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Single-slot model container for efficient cross-thread model transfer.
///
/// This pattern ensures:
/// - Maximum 2 models in memory: Actor's current + 1 pending update
/// - No accumulation of stale updates (overwrites)
///
/// # Thread Safety
///
/// Uses `Mutex<Option<M>>` which requires `M: Send` (NOT `M: Sync`).
/// This is critical for Burn models which are `Send` but NOT `Sync` due to
/// `Param<T>` containing `OnceCell`.
pub struct ModelSlot<M> {
    pending: Mutex<Option<M>>,
    /// Current model version
    version: AtomicU64,
    /// Counter for total models published
    published_count: AtomicUsize,
    /// Counter for models dropped (overwritten before being taken)
    dropped_count: AtomicUsize,
    /// Counter for models successfully taken by actor
    taken_count: AtomicUsize,
}

impl<M> ModelSlot<M> {
    /// Create a new empty model slot.
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(None),
            version: AtomicU64::new(0),
            published_count: AtomicUsize::new(0),
            dropped_count: AtomicUsize::new(0),
            taken_count: AtomicUsize::new(0),
        }
    }

    /// Create a model slot with an initial model.
    pub fn with_initial(model: M) -> Self {
        Self {
            pending: Mutex::new(Some(model)),
            version: AtomicU64::new(1),
            published_count: AtomicUsize::new(1),
            dropped_count: AtomicUsize::new(0),
            taken_count: AtomicUsize::new(0),
        }
    }

    /// Get current model version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Get debug statistics: (published, dropped, taken)
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.published_count.load(Ordering::Relaxed),
            self.dropped_count.load(Ordering::Relaxed),
            self.taken_count.load(Ordering::Relaxed),
        )
    }
}

impl<M> Default for ModelSlot<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Clone + Send> ModelSlot<M> {
    /// Publish a new model, overwriting any pending update.
    ///
    /// Called by the Learner thread after training steps.
    /// Any existing pending model is dropped (no accumulation).
    /// Returns true if a pending model was overwritten (dropped).
    pub fn publish(&self, model: M) -> bool {
        let mut guard = self.pending.lock();
        let was_pending = guard.is_some();
        if was_pending {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
        }
        *guard = Some(model);
        self.version.fetch_add(1, Ordering::Release);
        self.published_count.fetch_add(1, Ordering::Relaxed);
        was_pending
    }

    /// Publish a model with explicit version number.
    ///
    /// Useful when version is managed externally (e.g., by VersionCounter).
    pub fn publish_versioned(&self, model: M, version: u64) -> bool {
        let mut guard = self.pending.lock();
        let was_pending = guard.is_some();
        if was_pending {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
        }
        *guard = Some(model);
        self.version.store(version, Ordering::Release);
        self.published_count.fetch_add(1, Ordering::Relaxed);
        was_pending
    }

    /// Take the pending model, leaving the slot empty.
    ///
    /// Called by the Actor thread to receive updated parameters.
    /// Returns `None` if no update is pending.
    pub fn take(&self) -> Option<M> {
        let mut guard = self.pending.lock();
        let result = guard.take();
        if result.is_some() {
            self.taken_count.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Take the pending model along with its version.
    ///
    /// Returns `None` if no update is pending.
    pub fn take_versioned(&self) -> Option<(M, u64)> {
        let mut guard = self.pending.lock();
        let result = guard.take();
        if result.is_some() {
            self.taken_count.fetch_add(1, Ordering::Relaxed);
            let version = self.version.load(Ordering::Acquire);
            result.map(|m| (m, version))
        } else {
            None
        }
    }

    /// Check if an update is pending without taking it.
    pub fn has_pending(&self) -> bool {
        self.pending.lock().is_some()
    }

    /// Clone the pending model without removing it.
    ///
    /// Useful for evaluation or inspection without affecting
    /// the Actor's update flow.
    pub fn peek(&self) -> Option<M> {
        self.pending.lock().clone()
    }

    /// Peek the pending model with its version.
    pub fn peek_versioned(&self) -> Option<(M, u64)> {
        let guard = self.pending.lock();
        guard.clone().map(|m| (m, self.version.load(Ordering::Acquire)))
    }
}

/// Thread-safe shared model slot.
pub type SharedModelSlot<M> = Arc<ModelSlot<M>>;

/// Create a new shared model slot.
pub fn model_slot<M>() -> SharedModelSlot<M> {
    Arc::new(ModelSlot::new())
}

/// Create a new shared model slot with an initial model.
pub fn model_slot_with<M>(model: M) -> SharedModelSlot<M> {
    Arc::new(ModelSlot::with_initial(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct MockModel {
        version: u32,
    }

    #[test]
    fn test_publish_and_take() {
        let slot = ModelSlot::new();

        // Initially empty
        assert!(slot.take().is_none());
        assert!(!slot.has_pending());
        assert_eq!(slot.version(), 0);

        // Publish model
        slot.publish(MockModel { version: 1 });
        assert!(slot.has_pending());
        assert_eq!(slot.version(), 1);

        // Take model
        let model = slot.take();
        assert_eq!(model, Some(MockModel { version: 1 }));
        assert!(!slot.has_pending());

        // Take again returns None
        assert!(slot.take().is_none());
    }

    #[test]
    fn test_overwrite_pending() {
        let slot = ModelSlot::new();

        // Publish v1
        slot.publish(MockModel { version: 1 });
        assert_eq!(slot.version(), 1);

        // Publish v2 (overwrites v1)
        let dropped = slot.publish(MockModel { version: 2 });
        assert!(dropped);
        assert_eq!(slot.version(), 2);

        // Publish v3 (overwrites v2)
        let dropped = slot.publish(MockModel { version: 3 });
        assert!(dropped);
        assert_eq!(slot.version(), 3);

        // Only v3 should be available
        let model = slot.take();
        assert_eq!(model, Some(MockModel { version: 3 }));
        assert!(slot.take().is_none());

        // Check stats
        let (published, dropped_count, taken) = slot.stats();
        assert_eq!(published, 3);
        assert_eq!(dropped_count, 2);
        assert_eq!(taken, 1);
    }

    #[test]
    fn test_peek_does_not_consume() {
        let slot = ModelSlot::new();
        slot.publish(MockModel { version: 42 });

        // Peek multiple times
        assert_eq!(slot.peek(), Some(MockModel { version: 42 }));
        assert_eq!(slot.peek(), Some(MockModel { version: 42 }));

        // Model still available for take
        assert_eq!(slot.take(), Some(MockModel { version: 42 }));
    }

    #[test]
    fn test_versioned_operations() {
        let slot = ModelSlot::new();

        // Publish with explicit version
        slot.publish_versioned(MockModel { version: 1 }, 100);
        assert_eq!(slot.version(), 100);

        // Take versioned
        let (model, version) = slot.take_versioned().unwrap();
        assert_eq!(model, MockModel { version: 1 });
        assert_eq!(version, 100);
    }

    #[test]
    fn test_shared_model_slot() {
        let slot = model_slot::<MockModel>();
        let slot_clone = Arc::clone(&slot);

        // Publisher side
        slot.publish(MockModel { version: 100 });

        // Consumer side (using clone)
        assert_eq!(slot_clone.take(), Some(MockModel { version: 100 }));
    }

    #[test]
    fn test_initial_model() {
        let slot = ModelSlot::with_initial(MockModel { version: 0 });
        assert!(slot.has_pending());
        assert_eq!(slot.version(), 1);

        let model = slot.take();
        assert_eq!(model, Some(MockModel { version: 0 }));
    }
}
