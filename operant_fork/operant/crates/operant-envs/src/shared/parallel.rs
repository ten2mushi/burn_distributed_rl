//! Utilities for parallel execution with rayon.

/// A wrapper around raw pointers that asserts they're safe to share between threads.
///
/// # Safety
/// The caller must ensure that:
/// 1. The underlying data lives for the duration of all parallel operations
/// 2. Each thread accesses only its assigned non-overlapping index range
/// 3. No other code accesses the data while parallel operations are in progress
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(*mut T);

impl<T> SyncPtr<T> {
    /// Create a new SyncPtr from a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure that parallel access to the underlying data
    /// is safe (i.e., each thread operates on non-overlapping ranges).
    #[inline(always)]
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Get the raw pointer.
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }

    /// Access element at offset.
    ///
    /// # Safety
    /// The caller must ensure the index is within bounds and that
    /// no other thread is accessing the same index.
    #[inline(always)]
    pub unsafe fn add(self, count: usize) -> *mut T {
        self.0.add(count)
    }
}

// SAFETY: We assert that SyncPtr is only used in contexts where
// parallel access is safe (non-overlapping index ranges)
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
