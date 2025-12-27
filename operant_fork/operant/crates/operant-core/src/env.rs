//! Environment traits for high-performance parallel RL.
//!
//! This module provides:
//! - [`Environment`] trait for vectorized environment implementations
//! - [`StepResult`] for efficient step result access
//! - [`ResetMask`] for selective environment reset with O(k) iteration

use std::fmt::Debug;

// ============================================================================
// StepResult - Zero-copy step result access
// ============================================================================

/// Result of a single step, avoiding multiple buffer writes.
///
/// Provides zero-copy access to environment state after stepping.
/// Use with [`Environment::step_no_reset_with_result`].
#[derive(Debug)]
pub struct StepResult<'a> {
    /// Flat observation buffer (AoS layout: [obs0, obs1, ...])
    pub observations: &'a [f32],
    /// Reward for each environment
    pub rewards: &'a [f32],
    /// Terminal flags (1 = terminated, 0 = not)
    pub terminals: &'a [u8],
    /// Truncation flags (1 = truncated, 0 = not)
    pub truncations: &'a [u8],
    /// Number of parallel environments
    pub num_envs: usize,
    /// Observation size per environment
    pub obs_size: usize,
}

impl<'a> StepResult<'a> {
    /// Get observation for a specific environment.
    #[inline]
    pub fn obs(&self, env_idx: usize) -> &[f32] {
        debug_assert!(env_idx < self.num_envs, "env_idx out of bounds");
        let start = env_idx * self.obs_size;
        &self.observations[start..start + self.obs_size]
    }

    /// Check if environment terminated.
    #[inline]
    pub fn is_terminal(&self, env_idx: usize) -> bool {
        self.terminals[env_idx] != 0
    }

    /// Check if environment was truncated.
    #[inline]
    pub fn is_truncated(&self, env_idx: usize) -> bool {
        self.truncations[env_idx] != 0
    }

    /// Check if episode ended (terminal or truncated).
    #[inline]
    pub fn is_done(&self, env_idx: usize) -> bool {
        self.is_terminal(env_idx) || self.is_truncated(env_idx)
    }

    /// Create a ResetMask from this step result's terminal/truncation flags.
    pub fn to_reset_mask(&self) -> ResetMask {
        ResetMask::from_done_flags(self.terminals, self.truncations)
    }
}

// ============================================================================
// ResetMask - Efficient bitmask for selective reset
// ============================================================================

/// Bitmask for selective environment reset.
///
/// Each bit represents one environment (1 = reset, 0 = keep).
/// Uses u64 chunks for efficient 64-env-at-a-time processing.
///
/// # Example
///
/// ```rust,ignore
/// // After step_no_reset, create mask from terminal/truncation flags
/// let mask = ResetMask::from_done_flags(&terminals, &truncations);
///
/// // Check if any environments need reset
/// if mask.any() {
///     env.reset_envs(&mask, new_seed);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ResetMask {
    /// Packed bitmask chunks (64 environments per u64)
    chunks: Vec<u64>,
    /// Total number of environments
    num_envs: usize,
}

impl ResetMask {
    /// Create an empty mask (no environments to reset).
    pub fn new(num_envs: usize) -> Self {
        let num_chunks = (num_envs + 63) / 64;
        Self {
            chunks: vec![0u64; num_chunks],
            num_envs,
        }
    }

    /// Create mask from terminal/truncation buffers.
    ///
    /// An environment is marked for reset if either terminal or truncated.
    pub fn from_done_flags(terminals: &[u8], truncations: &[u8]) -> Self {
        debug_assert_eq!(terminals.len(), truncations.len());
        let num_envs = terminals.len();
        let num_chunks = (num_envs + 63) / 64;
        let mut chunks = vec![0u64; num_chunks];

        for (i, (&t, &tr)) in terminals.iter().zip(truncations.iter()).enumerate() {
            if t != 0 || tr != 0 {
                chunks[i / 64] |= 1u64 << (i % 64);
            }
        }

        Self { chunks, num_envs }
    }

    /// Create mask from terminal buffer only.
    pub fn from_terminals(terminals: &[u8]) -> Self {
        let num_envs = terminals.len();
        let num_chunks = (num_envs + 63) / 64;
        let mut chunks = vec![0u64; num_chunks];

        for (i, &t) in terminals.iter().enumerate() {
            if t != 0 {
                chunks[i / 64] |= 1u64 << (i % 64);
            }
        }

        Self { chunks, num_envs }
    }

    /// Check if any environments need reset.
    #[inline]
    pub fn any(&self) -> bool {
        self.chunks.iter().any(|&c| c != 0)
    }

    /// Count how many environments need reset.
    pub fn count(&self) -> usize {
        self.chunks.iter().map(|c| c.count_ones() as usize).sum()
    }

    /// Get the number of environments this mask covers.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the raw chunks (for advanced usage).
    #[inline]
    pub fn chunks(&self) -> &[u64] {
        &self.chunks
    }

    /// Set a specific environment for reset.
    #[inline]
    pub fn set(&mut self, env_idx: usize) {
        debug_assert!(env_idx < self.num_envs);
        self.chunks[env_idx / 64] |= 1u64 << (env_idx % 64);
    }

    /// Clear a specific environment from reset.
    #[inline]
    pub fn clear(&mut self, env_idx: usize) {
        debug_assert!(env_idx < self.num_envs);
        self.chunks[env_idx / 64] &= !(1u64 << (env_idx % 64));
    }

    /// Check if a specific environment is marked for reset.
    #[inline]
    pub fn is_set(&self, env_idx: usize) -> bool {
        debug_assert!(env_idx < self.num_envs);
        (self.chunks[env_idx / 64] >> (env_idx % 64)) & 1 != 0
    }

    /// Iterate over environment indices that need reset.
    ///
    /// Uses efficient bit iteration with `trailing_zeros()` for O(k) iteration
    /// where k is the number of set bits (environments to reset).
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.chunks.iter().enumerate().flat_map(|(chunk_idx, &chunk)| {
            let base = chunk_idx * 64;
            let max_bit = if chunk_idx == self.chunks.len() - 1 {
                self.num_envs - base
            } else {
                64
            };
            BitIter::new(chunk, max_bit).map(move |bit| base + bit)
        })
    }
}

/// Efficient bit iterator using `trailing_zeros()`.
///
/// Iterates over set bits in O(k) where k is the number of set bits.
struct BitIter {
    remaining: u64,
    max_bit: usize,
}

impl BitIter {
    fn new(bits: u64, max_bit: usize) -> Self {
        Self {
            remaining: bits,
            max_bit,
        }
    }
}

impl Iterator for BitIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            let bit = self.remaining.trailing_zeros() as usize;
            if bit >= self.max_bit {
                self.remaining = 0;
                None
            } else {
                self.remaining &= self.remaining - 1; // Clear lowest set bit
                Some(bit)
            }
        }
    }
}

// ============================================================================
// LogData trait
// ============================================================================

/// Log data that environments can return for tracking metrics.
pub trait LogData: Clone + Debug + Default {
    /// Merge another log into this one (for aggregation).
    fn merge(&mut self, other: &Self);

    /// Clear/reset the log counters.
    fn clear(&mut self);

    /// Get the number of episodes recorded.
    fn episode_count(&self) -> f32;
}

/// Trait for parallel/batched environments with SoA memory layout.
///
/// All environments in Operant are vectorized by default - this is the standard
/// interface for implementing custom environments. Vectorization enables processing
/// multiple environment instances simultaneously for maximum throughput.
///
/// # Example
///
/// ```rust,ignore
/// use operant_core::Environment;
///
/// struct MyEnv {
///     num_envs: usize,
///     // ... other fields
/// }
///
/// impl Environment for MyEnv {
///     fn num_envs(&self) -> usize {
///         self.num_envs
///     }
///     // ... implement other methods
/// }
/// ```
pub trait Environment {
    /// Returns the number of parallel environments.
    fn num_envs(&self) -> usize;

    /// Returns the observation size per environment.
    fn observation_size(&self) -> usize;

    /// Returns the number of discrete actions, or None for continuous.
    fn num_actions(&self) -> Option<usize>;

    /// Reset all environments with deterministic seeding.
    fn reset(&mut self, seed: u64);

    /// Step all environments (includes auto-reset for done envs).
    fn step(&mut self, actions: &[f32]);

    /// Write observations to buffer.
    fn write_observations(&self, buffer: &mut [f32]);

    /// Write rewards to buffer.
    fn write_rewards(&self, buffer: &mut [f32]);

    /// Write terminal flags to buffer.
    fn write_terminals(&self, buffer: &mut [u8]);

    /// Write truncation flags to buffer.
    fn write_truncations(&self, buffer: &mut [u8]);

    // ========================================================================
    // NEW: Non-auto-reset API for value-based RL (DQN, C51, SAC, etc.)
    // ========================================================================

    /// Step all environments WITHOUT auto-reset.
    ///
    /// Unlike [`step`], this preserves terminal flags and terminal observations,
    /// allowing correct TD learning for value-based RL algorithms.
    ///
    /// After calling this method:
    /// - [`write_terminals`] returns accurate terminal flags
    /// - [`write_observations`] returns terminal observations (not reset)
    /// - Caller MUST call [`reset_envs`] before the next step
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// env.step_no_reset(&actions);
    ///
    /// // Read terminal observations BEFORE reset
    /// env.write_observations(&mut obs_buffer);
    /// env.write_terminals(&mut terminals);
    ///
    /// // Reset terminated environments
    /// let mask = ResetMask::from_terminals(&terminals);
    /// if mask.any() {
    ///     env.reset_envs(&mask, new_seed);
    /// }
    /// ```
    fn step_no_reset(&mut self, actions: &[f32]) {
        // Default implementation panics - must be overridden
        let _ = actions;
        unimplemented!(
            "step_no_reset not implemented for this environment. \
             Check supports_no_reset() before calling."
        )
    }

    /// Step without auto-reset, returning all results in one struct.
    ///
    /// This combines [`step_no_reset`] with all write operations into a single
    /// call, providing zero-copy access to results.
    ///
    /// # Returns
    ///
    /// A [`StepResult`] containing references to internal buffers.
    fn step_no_reset_with_result(&mut self, actions: &[f32]) -> StepResult<'_> {
        let _ = actions;
        unimplemented!(
            "step_no_reset_with_result not implemented for this environment. \
             Check supports_no_reset() before calling."
        )
    }

    /// Reset specific environments identified by a bitmask.
    ///
    /// Only environments with their bit set in the mask will be reset.
    /// This is used after [`step_no_reset`] to reset terminated environments.
    ///
    /// # Arguments
    ///
    /// * `mask` - A [`ResetMask`] indicating which environments to reset
    /// * `seed` - Base seed for RNG (combined with env index for determinism)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mask = ResetMask::from_done_flags(&terminals, &truncations);
    /// if mask.any() {
    ///     env.reset_envs(&mask, seed);
    /// }
    /// ```
    fn reset_envs(&mut self, mask: &ResetMask, seed: u64) {
        let _ = (mask, seed);
        unimplemented!(
            "reset_envs not implemented for this environment. \
             Check supports_no_reset() before calling."
        )
    }

    /// Returns true if this environment supports the non-auto-reset API.
    ///
    /// Use this for runtime feature detection before calling [`step_no_reset`]
    /// or [`reset_envs`].
    fn supports_no_reset(&self) -> bool {
        false
    }
}
