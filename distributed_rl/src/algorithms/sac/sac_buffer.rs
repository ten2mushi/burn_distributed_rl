//! SAC-specific uniform replay buffer.
//!
//! Key differences from IMPALA's FIFO buffer:
//! - **Uniform random sampling** instead of FIFO
//! - **Ring buffer** semantics (overwrite oldest when full)
//! - **Individual transitions** instead of trajectories
//!
//! The buffer uses a lock-free injection queue for actor pushes
//! and a ring buffer for storage with O(1) random access.

use crate::core::experience_buffer::{ExperienceBuffer, OffPolicyBuffer};
use crossbeam_queue::SegQueue;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::sac_transition::SACTransitionTrait;

// ============================================================================
// Buffer Configuration
// ============================================================================

/// Configuration for SAC replay buffer.
#[derive(Debug, Clone)]
pub struct SACBufferConfig {
    /// Maximum number of transitions to store.
    pub capacity: usize,
    /// Minimum transitions before training starts.
    pub min_size: usize,
    /// Batch size for sampling.
    pub batch_size: usize,
}

impl Default for SACBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 1_000_000,
            min_size: 10_000,
            batch_size: 256,
        }
    }
}

impl SACBufferConfig {
    /// Create a new buffer config.
    pub fn new(capacity: usize, min_size: usize, batch_size: usize) -> Self {
        Self {
            capacity,
            min_size,
            batch_size,
        }
    }

    /// Create config with default SAC settings.
    pub fn sac_default() -> Self {
        Self::default()
    }

    /// Builder pattern: set capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Builder pattern: set minimum size.
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size;
        self
    }

    /// Builder pattern: set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

// ============================================================================
// Ring Buffer (Internal)
// ============================================================================

/// Ring buffer with O(1) insert and random access.
///
/// Overwrites oldest elements when capacity is reached.
struct RingBuffer<T> {
    /// Storage vector.
    buffer: Vec<T>,
    /// Capacity of the buffer.
    capacity: usize,
    /// Next position to write (circular).
    write_pos: usize,
    /// Current number of valid items.
    len: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with given capacity.
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            len: 0,
        }
    }

    /// Push an item to the buffer, overwriting oldest if full.
    fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity {
            // Still growing, push to end
            self.buffer.push(item);
            self.len = self.buffer.len();
        } else {
            // Full, overwrite at write position
            self.buffer[self.write_pos] = item;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.len = self.len.min(self.capacity);
    }

    /// Get an item by index.
    #[inline]
    fn get(&self, idx: usize) -> &T {
        debug_assert!(idx < self.len, "Index out of bounds: {} >= {}", idx, self.len);
        &self.buffer[idx]
    }

    /// Current number of items.
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all items.
    fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.len = 0;
    }

    /// Capacity.
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }
}

// ============================================================================
// SAC Replay Buffer
// ============================================================================

/// Thread-safe uniform replay buffer for SAC.
///
/// Actors push transitions via a lock-free MPMC queue.
/// The learner samples uniformly at random from the consolidated storage.
///
/// # Key Properties
/// - **Lock-free push**: Actors can push concurrently without blocking
/// - **Uniform sampling**: Each transition has equal probability of being sampled
/// - **Ring buffer**: O(1) insertion, oldest transitions evicted when full
/// - **Lazy consolidation**: Pending items moved to storage on sample/ready check
pub struct SACBuffer<Trans: SACTransitionTrait> {
    config: SACBufferConfig,
    /// Lock-free MPMC queue for actor pushes.
    pending: SegQueue<Trans>,
    /// Main storage (ring buffer).
    storage: RwLock<RingBuffer<Trans>>,
    /// Current size (atomic for fast reads).
    size: AtomicUsize,
    /// Pending items not yet consolidated.
    pending_size: AtomicUsize,
}

impl<Trans: SACTransitionTrait> SACBuffer<Trans> {
    /// Create a new SAC replay buffer.
    pub fn new(config: SACBufferConfig) -> Self {
        Self {
            pending: SegQueue::new(),
            storage: RwLock::new(RingBuffer::new(config.capacity)),
            size: AtomicUsize::new(0),
            pending_size: AtomicUsize::new(0),
            config,
        }
    }

    /// Push a single transition (lock-free).
    pub fn push_transition(&self, transition: Trans) {
        self.pending.push(transition);
        self.pending_size.fetch_add(1, Ordering::Release);
    }

    /// Push a batch of transitions (lock-free).
    pub fn push_transitions(&self, transitions: Vec<Trans>) {
        let count = transitions.len();
        for trans in transitions {
            self.pending.push(trans);
        }
        self.pending_size.fetch_add(count, Ordering::Release);
    }

    /// Sample a uniform random batch.
    ///
    /// Returns `None` if buffer has fewer than `batch_size` items.
    pub fn sample(&self, batch_size: usize) -> Option<Vec<Trans>> {
        // Consolidate pending first
        self.do_consolidate();

        let storage = self.storage.read();
        if storage.len() < batch_size {
            return None;
        }

        // Uniform random sampling with replacement
        let mut samples = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = fastrand::usize(..storage.len());
            samples.push(storage.get(idx).clone());
        }

        Some(samples)
    }

    /// Sample without replacement (for testing/debugging).
    ///
    /// Returns `None` if buffer has fewer than `batch_size` items.
    pub fn sample_without_replacement(&self, batch_size: usize) -> Option<Vec<Trans>> {
        self.do_consolidate();

        let storage = self.storage.read();
        if storage.len() < batch_size {
            return None;
        }

        // Fisher-Yates shuffle on indices
        let mut indices: Vec<usize> = (0..storage.len()).collect();
        for i in 0..batch_size {
            let j = fastrand::usize(i..indices.len());
            indices.swap(i, j);
        }

        let samples: Vec<Trans> = indices[..batch_size]
            .iter()
            .map(|&idx| storage.get(idx).clone())
            .collect();

        Some(samples)
    }

    /// Check if ready for training.
    pub fn is_training_ready(&self) -> bool {
        self.do_consolidate();
        self.size.load(Ordering::Acquire) >= self.config.min_size
    }

    /// Check if buffer has enough samples for a batch.
    pub fn has_batch(&self) -> bool {
        self.do_consolidate();
        self.size.load(Ordering::Acquire) >= self.config.batch_size
    }

    /// Sample a batch using the configured batch size.
    ///
    /// Convenience method that uses `config.batch_size`.
    pub fn sample_batch(&self) -> Option<Vec<Trans>> {
        self.sample(self.config.batch_size)
    }

    /// Consolidate pending transitions into main storage.
    fn do_consolidate(&self) {
        let mut storage = self.storage.write();
        let mut count = 0;

        // Drain all pending items from the MPMC queue
        while let Some(trans) = self.pending.pop() {
            storage.push(trans);
            count += 1;
        }

        if count > 0 {
            let pending = self.pending_size.load(Ordering::Acquire);
            self.pending_size.fetch_sub(count.min(pending), Ordering::Release);
            self.size.store(storage.len(), Ordering::Release);
        }
    }

    /// Get the current size (consolidated).
    pub fn current_size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Get the pending size (not yet consolidated).
    pub fn pending_size(&self) -> usize {
        self.pending_size.load(Ordering::Relaxed)
    }

    /// Get buffer utilization (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        self.size.load(Ordering::Relaxed) as f32 / self.config.capacity as f32
    }

    /// Get the configuration.
    pub fn config(&self) -> &SACBufferConfig {
        &self.config
    }
}

// ============================================================================
// ExperienceBuffer Trait Implementation
// ============================================================================

impl<Trans: SACTransitionTrait> ExperienceBuffer for SACBuffer<Trans> {
    type Item = Trans;

    fn push(&self, item: Self::Item) {
        self.push_transition(item);
    }

    fn push_batch(&self, items: Vec<Self::Item>) {
        self.push_transitions(items);
    }

    fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    fn capacity(&self) -> usize {
        self.config.capacity
    }

    fn consolidate(&self) {
        self.do_consolidate();
    }

    fn clear(&self) {
        // Drain pending queue
        while self.pending.pop().is_some() {}
        self.pending_size.store(0, Ordering::Release);

        // Clear storage
        let mut storage = self.storage.write();
        storage.clear();
        self.size.store(0, Ordering::Release);
    }
}

// ============================================================================
// OffPolicyBuffer Trait Implementation
// ============================================================================

impl<Trans: SACTransitionTrait> OffPolicyBuffer for SACBuffer<Trans> {
    fn sample(&self, batch_size: usize) -> Option<Vec<Self::Item>> {
        SACBuffer::sample(self, batch_size)
    }

    fn pending_len(&self) -> usize {
        self.pending_size.load(Ordering::Relaxed)
    }
}

// Note: SACBuffer is automatically Send + Sync because:
// - Injector<T> is Send + Sync when T: Send
// - RwLock<T> is Send + Sync when T: Send
// - AtomicUsize is Send + Sync

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::sac::sac_transition::SACTransition;

    fn make_transition(state_val: f32) -> SACTransition {
        SACTransition::new_discrete(
            vec![state_val],
            0,
            1.0,
            vec![state_val + 1.0],
            false,
        )
    }

    #[test]
    fn test_ring_buffer_push_and_get() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);

        rb.push(1);
        rb.push(2);
        assert_eq!(rb.len(), 2);
        assert_eq!(*rb.get(0), 1);
        assert_eq!(*rb.get(1), 2);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);

        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // Overwrites 1

        assert_eq!(rb.len(), 3);
        // After overflow, buffer contains [4, 2, 3] at positions [0, 1, 2]
        assert_eq!(*rb.get(0), 4);
        assert_eq!(*rb.get(1), 2);
        assert_eq!(*rb.get(2), 3);
    }

    #[test]
    fn test_sac_buffer_new() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer: SACBuffer<SACTransition> = SACBuffer::new(config);

        assert!(buffer.is_empty());
        assert!(!buffer.is_training_ready());
    }

    #[test]
    fn test_sac_buffer_push_and_consolidate() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer = SACBuffer::new(config);

        // Push some transitions
        for i in 0..10 {
            buffer.push_transition(make_transition(i as f32));
        }

        // Before consolidation, size is 0
        assert_eq!(buffer.current_size(), 0);
        assert_eq!(buffer.pending_size(), 10);

        // Consolidate
        buffer.consolidate();

        assert_eq!(buffer.current_size(), 10);
        assert_eq!(buffer.pending_size(), 0);
    }

    #[test]
    fn test_sac_buffer_sample() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer = SACBuffer::new(config);

        // Push transitions
        for i in 0..10 {
            buffer.push_transition(make_transition(i as f32));
        }

        // Sample (will consolidate automatically)
        let batch = buffer.sample(3);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);

        // Verify all transitions are valid
        let batch2 = buffer.sample(3).unwrap();
        for t in batch2 {
            assert!(t.state()[0] >= 0.0 && t.state()[0] < 10.0);
        }
    }

    #[test]
    fn test_sac_buffer_sample_insufficient() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 10,
        };
        let buffer = SACBuffer::new(config);

        // Push only 5 transitions
        for i in 0..5 {
            buffer.push_transition(make_transition(i as f32));
        }

        // Try to sample 10 - should fail
        let batch = buffer.sample(10);
        assert!(batch.is_none());
    }

    #[test]
    fn test_sac_buffer_training_ready() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::new(config);

        // Push 5 transitions - not ready
        for i in 0..5 {
            buffer.push_transition(make_transition(i as f32));
        }
        assert!(!buffer.is_training_ready());

        // Push 5 more - now ready
        for i in 5..10 {
            buffer.push_transition(make_transition(i as f32));
        }
        assert!(buffer.is_training_ready());
    }

    #[test]
    fn test_sac_buffer_clear() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer = SACBuffer::new(config);

        // Add data
        for i in 0..10 {
            buffer.push_transition(make_transition(i as f32));
        }
        buffer.consolidate();
        assert_eq!(buffer.current_size(), 10);

        // Clear
        buffer.clear();
        assert_eq!(buffer.current_size(), 0);
        assert_eq!(buffer.pending_size(), 0);
        assert!(!buffer.is_training_ready());
    }

    #[test]
    fn test_sac_buffer_capacity_enforcement() {
        let config = SACBufferConfig {
            capacity: 5,  // Small capacity
            min_size: 2,
            batch_size: 2,
        };
        let buffer = SACBuffer::new(config);

        // Push more than capacity
        for i in 0..10 {
            buffer.push_transition(make_transition(i as f32));
        }
        buffer.consolidate();

        // Buffer should be at capacity
        assert_eq!(buffer.current_size(), 5);
    }

    #[test]
    fn test_sac_buffer_utilization() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer = SACBuffer::new(config);

        // Push 50 transitions
        for i in 0..50 {
            buffer.push_transition(make_transition(i as f32));
        }
        buffer.consolidate();

        assert!((buffer.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sac_buffer_sample_without_replacement() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 5,
        };
        let buffer = SACBuffer::new(config);

        // Push transitions with unique states
        for i in 0..10 {
            buffer.push_transition(make_transition(i as f32));
        }

        let batch = buffer.sample_without_replacement(5).unwrap();
        assert_eq!(batch.len(), 5);

        // Check all samples are unique (by state value)
        let mut states: Vec<f32> = batch.iter().map(|t| t.state()[0]).collect();
        states.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..states.len() {
            assert!(states[i] != states[i - 1], "Found duplicate in sample without replacement");
        }
    }

    #[test]
    fn test_experience_buffer_trait() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer: SACBuffer<SACTransition> = SACBuffer::new(config);

        // Use trait methods
        buffer.push(make_transition(1.0));
        buffer.push_batch(vec![make_transition(2.0), make_transition(3.0)]);

        buffer.consolidate();
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.capacity(), 100);
    }

    #[test]
    fn test_off_policy_buffer_trait() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 5,
            batch_size: 3,
        };
        let buffer: SACBuffer<SACTransition> = SACBuffer::new(config);

        // Push data
        for i in 0..10 {
            buffer.push(make_transition(i as f32));
        }

        // Use OffPolicyBuffer trait methods
        let batch = OffPolicyBuffer::sample(&buffer, 3);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);

        assert_eq!(buffer.pending_len(), 0); // Consolidated by sample
    }
}
