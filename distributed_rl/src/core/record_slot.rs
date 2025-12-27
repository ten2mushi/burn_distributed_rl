//! Record-based model transfer for WGPU multi-threading.
//!
//! Unlike ModelSlot which transfers full models, RecordSlot transfers
//! serialized model records (weights). This is essential for WGPU backend
//! where models contain GPU handles that aren't `Send`.
//!
//! # Architecture
//!
//! ```text
//! Learner Thread                           Actor Thread
//! ┌──────────────────┐                     ┌──────────────────┐
//! │ model.train()    │                     │ local_model      │
//! │       ↓          │                     │       ↑          │
//! │ model.into_record() ────RecordSlot───→ │ load_record()    │
//! └──────────────────┘                     └──────────────────┘
//! ```
//!
//! # Why Records?
//!
//! - **Models (WGPU)**: Contain GPU device handles → NOT `Send`
//! - **Records**: Pure data (TensorData) → `Send + Sync`
//!
//! Each actor thread creates its own model instance on its own CubeCL stream,
//! then loads weights from records published by the learner.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Single-slot record container for efficient cross-thread weight transfer.
///
/// This pattern ensures:
/// - Each thread has its own model (no shared GPU handles)
/// - Only serialized weights cross thread boundaries
/// - No accumulation of stale updates (overwrites)
///
/// # Thread Safety
///
/// Records (model weights) are pure data, so they are `Send + Sync`.
/// This enables safe transfer between WGPU threads.
pub struct RecordSlot<R> {
    pending: Mutex<Option<R>>,
    /// Current record version
    version: AtomicU64,
    /// Counter for total records published
    published_count: AtomicUsize,
    /// Counter for records dropped (overwritten before being taken)
    dropped_count: AtomicUsize,
    /// Counter for records successfully taken by actor
    taken_count: AtomicUsize,
}

impl<R> RecordSlot<R> {
    /// Create a new empty record slot.
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(None),
            version: AtomicU64::new(0),
            published_count: AtomicUsize::new(0),
            dropped_count: AtomicUsize::new(0),
            taken_count: AtomicUsize::new(0),
        }
    }

    /// Create a record slot with an initial record.
    pub fn with_initial(record: R) -> Self {
        Self {
            pending: Mutex::new(Some(record)),
            version: AtomicU64::new(1),
            published_count: AtomicUsize::new(1),
            dropped_count: AtomicUsize::new(0),
            taken_count: AtomicUsize::new(0),
        }
    }

    /// Get current record version.
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

    /// Check if an update is pending without taking it.
    pub fn has_pending(&self) -> bool {
        self.pending.lock().is_some()
    }
}

impl<R> Default for RecordSlot<R> {
    fn default() -> Self {
        Self::new()
    }
}

// Records only need Send for cross-thread transfer
impl<R: Send> RecordSlot<R> {
    /// Publish a new record, overwriting any pending update.
    ///
    /// Called by the Learner thread after training steps.
    /// Any existing pending record is dropped (no accumulation).
    /// Returns true if a pending record was overwritten (dropped).
    pub fn publish(&self, record: R) -> bool {
        let mut guard = self.pending.lock();
        let was_pending = guard.is_some();
        if was_pending {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
        }
        *guard = Some(record);
        self.version.fetch_add(1, Ordering::Release);
        self.published_count.fetch_add(1, Ordering::Relaxed);
        was_pending
    }

    /// Publish a record with explicit version number.
    pub fn publish_versioned(&self, record: R, version: u64) -> bool {
        let mut guard = self.pending.lock();
        let was_pending = guard.is_some();
        if was_pending {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
        }
        *guard = Some(record);
        self.version.store(version, Ordering::Release);
        self.published_count.fetch_add(1, Ordering::Relaxed);
        was_pending
    }

    /// Take the pending record, leaving the slot empty.
    ///
    /// Called by Actor threads to receive updated weights.
    /// Returns `None` if no update is pending.
    pub fn take(&self) -> Option<R> {
        let mut guard = self.pending.lock();
        let result = guard.take();
        if result.is_some() {
            self.taken_count.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Take the pending record along with its version.
    pub fn take_versioned(&self) -> Option<(R, u64)> {
        let mut guard = self.pending.lock();
        let result = guard.take();
        if result.is_some() {
            self.taken_count.fetch_add(1, Ordering::Relaxed);
            let version = self.version.load(Ordering::Acquire);
            result.map(|r| (r, version))
        } else {
            None
        }
    }
}

// Clone only needed for peek operations
impl<R: Clone + Send> RecordSlot<R> {
    /// Clone the pending record without removing it.
    pub fn peek(&self) -> Option<R> {
        self.pending.lock().clone()
    }

    /// Peek the pending record with its version.
    pub fn peek_versioned(&self) -> Option<(R, u64)> {
        let guard = self.pending.lock();
        guard.clone().map(|r| (r, self.version.load(Ordering::Acquire)))
    }
}

/// Thread-safe shared record slot.
pub type SharedRecordSlot<R> = Arc<RecordSlot<R>>;

/// Create a new shared record slot.
pub fn record_slot<R>() -> SharedRecordSlot<R> {
    Arc::new(RecordSlot::new())
}

/// Create a new shared record slot with an initial record.
pub fn record_slot_with<R>(record: R) -> SharedRecordSlot<R> {
    Arc::new(RecordSlot::with_initial(record))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct MockRecord {
        weights: Vec<f32>,
    }

    #[test]
    fn test_publish_and_take() {
        let slot = RecordSlot::new();

        // Initially empty
        assert!(slot.take().is_none());
        assert!(!slot.has_pending());
        assert_eq!(slot.version(), 0);

        // Publish record
        slot.publish(MockRecord { weights: vec![1.0, 2.0] });
        assert!(slot.has_pending());
        assert_eq!(slot.version(), 1);

        // Take record
        let record = slot.take();
        assert_eq!(record, Some(MockRecord { weights: vec![1.0, 2.0] }));
        assert!(!slot.has_pending());
    }

    #[test]
    fn test_overwrite_pending() {
        let slot = RecordSlot::new();

        slot.publish(MockRecord { weights: vec![1.0] });
        let dropped = slot.publish(MockRecord { weights: vec![2.0] });
        assert!(dropped);

        let record = slot.take();
        assert_eq!(record, Some(MockRecord { weights: vec![2.0] }));

        let (published, dropped_count, taken) = slot.stats();
        assert_eq!(published, 2);
        assert_eq!(dropped_count, 1);
        assert_eq!(taken, 1);
    }

    #[test]
    fn test_shared_record_slot() {
        let slot = record_slot::<MockRecord>();
        let slot_clone = Arc::clone(&slot);

        slot.publish(MockRecord { weights: vec![42.0] });
        assert_eq!(slot_clone.take(), Some(MockRecord { weights: vec![42.0] }));
    }
}
