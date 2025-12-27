//! Bytes-based weight transfer for WGPU multi-threading.
//!
//! WGPU model records may not implement `Sync` due to lazy initialization
//! internals. This module provides a bytes-based alternative where weights
//! are serialized to `Vec<u8>` which is always `Send + Sync`.
//!
//! # Usage
//!
//! ```text
//! Learner Thread                           Actor Thread
//! ┌──────────────────┐                     ┌──────────────────┐
//! │ model.train()    │                     │ local_model      │
//! │       ↓          │                     │       ↑          │
//! │ into_record()    │                     │ load_record()    │
//! │       ↓          │                     │       ↑          │
//! │ recorder.record()│                     │ recorder.load()  │
//! │       ↓          │                     │       ↑          │
//! │   Vec<u8>  ─────────BytesSlot────────→ │    Vec<u8>       │
//! └──────────────────┘                     └──────────────────┘
//! ```

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe slot for transferring serialized model weights as bytes.
///
/// `Vec<u8>` is always `Send + Sync`, making this suitable for any backend
/// including WGPU where model records may not be `Sync`.
pub struct BytesSlot {
    bytes: Mutex<Option<Vec<u8>>>,
    version: AtomicU64,
}

impl BytesSlot {
    /// Create a new empty bytes slot.
    pub fn new() -> Self {
        Self {
            bytes: Mutex::new(None),
            version: AtomicU64::new(0),
        }
    }

    /// Create a bytes slot with initial bytes.
    pub fn with_initial(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Mutex::new(Some(bytes)),
            version: AtomicU64::new(1),
        }
    }

    /// Get current version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Publish new bytes, overwriting any pending data.
    pub fn publish(&self, bytes: Vec<u8>) {
        let mut guard = self.bytes.lock();
        *guard = Some(bytes);
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Take the pending bytes, leaving the slot empty.
    pub fn take(&self) -> Option<Vec<u8>> {
        self.bytes.lock().take()
    }

    /// Clone the pending bytes without removing them.
    pub fn get(&self) -> Option<Vec<u8>> {
        self.bytes.lock().clone()
    }

    /// Check if bytes are pending.
    pub fn has_pending(&self) -> bool {
        self.bytes.lock().is_some()
    }
}

impl Default for BytesSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared bytes slot.
pub type SharedBytesSlot = Arc<BytesSlot>;

/// Create a new shared bytes slot.
pub fn bytes_slot() -> SharedBytesSlot {
    Arc::new(BytesSlot::new())
}

/// Create a new shared bytes slot with initial bytes.
pub fn bytes_slot_with(bytes: Vec<u8>) -> SharedBytesSlot {
    Arc::new(BytesSlot::with_initial(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_slot_basic() {
        let slot = BytesSlot::new();
        assert!(slot.take().is_none());
        assert_eq!(slot.version(), 0);

        slot.publish(vec![1, 2, 3]);
        assert_eq!(slot.version(), 1);
        assert!(slot.has_pending());

        let bytes = slot.take();
        assert_eq!(bytes, Some(vec![1, 2, 3]));
        assert!(!slot.has_pending());
    }

    #[test]
    fn test_bytes_slot_get() {
        let slot = BytesSlot::with_initial(vec![42]);

        // get() doesn't remove
        assert_eq!(slot.get(), Some(vec![42]));
        assert_eq!(slot.get(), Some(vec![42]));

        // take() removes
        assert_eq!(slot.take(), Some(vec![42]));
        assert!(slot.get().is_none());
    }

    #[test]
    fn test_shared_bytes_slot() {
        let slot = bytes_slot_with(vec![1, 2, 3]);
        let slot2 = Arc::clone(&slot);

        assert_eq!(slot.get(), Some(vec![1, 2, 3]));
        slot2.publish(vec![4, 5, 6]);
        assert_eq!(slot.get(), Some(vec![4, 5, 6]));
    }
}
