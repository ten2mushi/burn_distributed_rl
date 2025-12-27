//! Lock-free shared replay buffer using crossbeam-deque.
//!
//! This module provides a high-performance lock-free buffer for collecting
//! experience data from multiple actor threads and sampling for training.
//!
//! # Design
//!
//! - Actors push transitions to lock-free Injector (non-blocking)
//! - Periodic consolidation moves data to RingBuffer for sampling
//! - Learner samples from consolidated storage (read-lock only)
//!
//! # Data Flow
//!
//! ```text
//! Actor 0 ─┐
//! Actor 1 ─┼──> Injector (lock-free) ──> Consolidator ──> RingStorage
//! Actor N ─┘                                                   │
//!                                                              │
//!                                                              v
//!                                                          Learner
//!                                                          (sampling)
//! ```

use crossbeam_deque::{Injector, Steal};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free replay buffer with periodic consolidation.
///
/// Data flow:
/// 1. Actor pushes to lock-free Injector (non-blocking)
/// 2. Background Consolidator moves data to RingStorage (periodic)
/// 3. Learner samples from RingStorage (read-lock only)
pub struct SharedBuffer<T: Send> {
    /// Actor pushes here (lock-free MPSC)
    injector: Injector<T>,

    /// Consolidated ring storage for sampling
    storage: RwLock<RingStorage<T>>,

    /// Current size of consolidated storage (atomic for lock-free queries)
    size: AtomicUsize,

    /// Approximate size of injector queue (for debugging)
    injector_size: AtomicUsize,

    /// Maximum capacity
    capacity: usize,
}

/// Ring buffer storage for consolidated transitions.
struct RingStorage<T> {
    data: Vec<Option<T>>,
    head: usize,
    len: usize,
}

impl<T> RingStorage<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: (0..capacity).map(|_| None).collect(),
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, item: T) {
        let capacity = self.data.len();
        let idx = (self.head + self.len) % capacity;
        self.data[idx] = Some(item);
        if self.len < capacity {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % capacity;
        }
    }

    fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len {
            return None;
        }
        let actual_idx = (self.head + idx) % self.data.len();
        self.data[actual_idx].as_ref()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        for item in &mut self.data {
            *item = None;
        }
        self.head = 0;
        self.len = 0;
    }
}

impl<T: Send + Clone> SharedBuffer<T> {
    /// Create a new shared buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            injector: Injector::new(),
            storage: RwLock::new(RingStorage::new(capacity)),
            size: AtomicUsize::new(0),
            injector_size: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Actor adds a single transition (non-blocking).
    ///
    /// Data is pushed to the lock-free injector.
    /// Consolidation is handled by the background Consolidator thread.
    pub fn push(&self, item: T) {
        self.injector.push(item);
        self.injector_size.fetch_add(1, Ordering::Relaxed);
    }

    /// Actor adds batch of transitions (non-blocking).
    ///
    /// Data is pushed to the lock-free injector.
    /// Consolidation is handled by the background Consolidator thread.
    pub fn push_batch(&self, batch: Vec<T>) {
        let count = batch.len();
        for item in batch {
            self.injector.push(item);
        }
        self.injector_size.fetch_add(count, Ordering::Relaxed);
    }

    /// Learner samples batch (read-lock only, clones data).
    ///
    /// This method is non-blocking - it only acquires a read lock.
    /// Consolidation is handled by the background Consolidator thread.
    pub fn sample(&self, n: usize) -> Option<Vec<T>> {
        let guard = self.storage.read();
        let len = guard.len();

        if len == 0 || n == 0 {
            return None;
        }

        let mut rng = fastrand::Rng::new();
        let samples: Vec<T> = (0..n)
            .filter_map(|_| {
                let idx = rng.usize(0..len);
                guard.get(idx).cloned()
            })
            .collect();

        if samples.is_empty() {
            None
        } else {
            Some(samples)
        }
    }

    /// Sample all data from buffer (for on-policy algorithms like PPO).
    ///
    /// Returns all consolidated data and clears the buffer.
    pub fn drain(&self) -> Vec<T> {
        let mut guard = self.storage.write();
        let mut result = Vec::with_capacity(guard.len());

        for i in 0..guard.len() {
            if let Some(item) = guard.get(i).cloned() {
                result.push(item);
            }
        }

        guard.clear();
        self.size.store(0, Ordering::Relaxed);

        result
    }

    /// Current buffer size (consolidated storage).
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Approximate size of pending items in injector queue.
    /// This is useful for debugging to detect consolidation lag.
    pub fn injector_len(&self) -> usize {
        self.injector_size.load(Ordering::Relaxed)
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get buffer utilization as fraction (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.capacity as f32
    }

    /// Try to consolidate without blocking.
    ///
    /// Returns immediately if the write lock is not available.
    /// This is the primary consolidation method used by the background Consolidator.
    pub fn try_consolidate(&self) -> usize {
        // Non-blocking: try to acquire write lock, return immediately if unavailable
        if let Some(mut guard) = self.storage.try_write() {
            let mut count = 0;
            const MAX_CONSOLIDATE: usize = 1000;
            const MAX_RETRIES: usize = 50;

            let mut retries = 0;
            loop {
                if count >= MAX_CONSOLIDATE {
                    break;
                }

                match self.injector.steal() {
                    Steal::Success(item) => {
                        guard.push(item);
                        count += 1;
                        retries = 0;
                    }
                    Steal::Empty => break,
                    Steal::Retry => {
                        retries += 1;
                        if retries >= MAX_RETRIES {
                            break;
                        }
                        continue;
                    }
                }
            }

            if count > 0 {
                self.injector_size.fetch_sub(count, Ordering::Relaxed);
                let new_size = guard.len().min(self.capacity);
                self.size.store(new_size, Ordering::Relaxed);
            }

            count
        } else {
            0
        }
    }

    /// Force consolidation from injector to storage (blocking).
    ///
    /// This method blocks until the write lock is acquired.
    /// Use `try_consolidate()` for non-blocking consolidation.
    pub fn force_consolidate(&self) -> usize {
        let mut count = 0;
        let mut guard = self.storage.write();

        const MAX_CONSOLIDATE: usize = 1000;
        const MAX_RETRIES: usize = 100;
        let mut retries = 0;

        loop {
            if count >= MAX_CONSOLIDATE {
                break;
            }

            match self.injector.steal() {
                Steal::Success(item) => {
                    guard.push(item);
                    count += 1;
                    retries = 0;
                }
                Steal::Empty => break,
                Steal::Retry => {
                    retries += 1;
                    if retries >= MAX_RETRIES {
                        std::thread::yield_now();
                        break;
                    }
                    continue;
                }
            }
        }

        if count > 0 {
            self.injector_size.fetch_sub(count, Ordering::Relaxed);
            let new_size = guard.len().min(self.capacity);
            self.size.store(new_size, Ordering::Relaxed);
        }

        count
    }

    /// Clear all data from the buffer.
    pub fn clear(&self) {
        // Drain injector
        loop {
            match self.injector.steal() {
                Steal::Success(_) => continue,
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        self.injector_size.store(0, Ordering::Relaxed);

        // Clear storage
        let mut guard = self.storage.write();
        guard.clear();
        self.size.store(0, Ordering::Relaxed);
    }
}

/// Thread-safe shared buffer.
pub type SharedReplayBuffer<T> = Arc<SharedBuffer<T>>;

/// Create a new shared replay buffer.
pub fn shared_buffer<T: Send + Clone>(capacity: usize) -> SharedReplayBuffer<T> {
    Arc::new(SharedBuffer::new(capacity))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_sample() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(100);

        // Push items
        for i in 0..50 {
            buffer.push(i);
        }

        buffer.force_consolidate();
        assert_eq!(buffer.len(), 50);

        // Sample
        let samples = buffer.sample(10).unwrap();
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_push_batch() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(100);
        buffer.push_batch((0..30).collect());
        buffer.force_consolidate();
        assert_eq!(buffer.len(), 30);
    }

    #[test]
    fn test_ring_overflow() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(10);
        buffer.push_batch((0..20).collect());
        buffer.force_consolidate();
        assert_eq!(buffer.len(), 10); // Capped at capacity
    }

    #[test]
    fn test_drain() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(100);
        buffer.push_batch((0..50).collect());
        buffer.force_consolidate();

        let data = buffer.drain();
        assert_eq!(data.len(), 50);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_utilization() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(100);
        buffer.push_batch((0..50).collect());
        buffer.force_consolidate();
        assert!((buffer.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let buffer: SharedBuffer<i32> = SharedBuffer::new(100);
        buffer.push_batch((0..50).collect());
        buffer.force_consolidate();
        assert_eq!(buffer.len(), 50);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.injector_len(), 0);
    }

    #[test]
    fn test_shared_buffer_arc() {
        let buffer = shared_buffer::<i32>(100);
        let buffer_clone = Arc::clone(&buffer);

        buffer.push(42);
        buffer.force_consolidate();

        let samples = buffer_clone.sample(1).unwrap();
        assert!(!samples.is_empty());
    }
}
