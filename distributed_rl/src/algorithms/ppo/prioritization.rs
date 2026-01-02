//! Prioritized experience sampling for PPO.
//!
//! This module provides utilities for sampling minibatches based on priority
//! (e.g., TD error, advantage magnitude) rather than uniformly. This can help
//! focus learning on the most informative experiences.
//!
//! # Theory
//!
//! Standard PPO samples uniformly from the rollout buffer. Prioritized sampling
//! weights samples by their "importance" - typically the magnitude of TD error
//! or advantage. This helps the agent learn faster from surprising or important
//! transitions.
//!
//! # Importance Sampling
//!
//! To correct for the bias introduced by non-uniform sampling, we use importance
//! sampling weights:
//!
//! ```text
//! w_i = (1 / (N * P(i)))^β
//! ```
//!
//! Where:
//! - N is the buffer size
//! - P(i) is the probability of sampling transition i
//! - β controls the amount of correction (0 = no correction, 1 = full correction)
//!
//! # Usage
//!
//! ```ignore
//! use distributed_rl::algorithms::ppo::prioritization::{
//!     PrioritizedSampler, PrioritizationConfig,
//! };
//!
//! let config = PrioritizationConfig::new()
//!     .with_alpha(0.6)  // Priority exponent
//!     .with_beta(0.4);  // Importance sampling exponent
//!
//! let mut sampler = PrioritizedSampler::new(buffer_size, config);
//!
//! // Update priorities based on TD errors
//! sampler.update_priorities(&indices, &td_errors);
//!
//! // Sample a minibatch
//! let (indices, weights) = sampler.sample(batch_size);
//! ```

use rand::Rng;

/// Configuration for prioritized sampling.
#[derive(Debug, Clone)]
pub struct PrioritizationConfig {
    /// Priority exponent (alpha).
    /// 0 = uniform sampling, 1 = fully prioritized.
    /// Typical value: 0.6
    pub alpha: f32,
    /// Importance sampling exponent (beta).
    /// 0 = no correction, 1 = full correction.
    /// Often annealed from 0.4 to 1.0 during training.
    pub beta: f32,
    /// Whether to use TD error (true) or advantage magnitude (false) for priority.
    pub use_td_error: bool,
    /// Small constant for numerical stability.
    pub epsilon: f32,
}

impl Default for PrioritizationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            beta: 0.4,
            use_td_error: true,
            epsilon: 1e-6,
        }
    }
}

impl PrioritizationConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the priority exponent (alpha).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the importance sampling exponent (beta).
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set whether to use TD error for priorities.
    pub fn with_use_td_error(mut self, use_td_error: bool) -> Self {
        self.use_td_error = use_td_error;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Result of a prioritized sample.
#[derive(Debug, Clone)]
pub struct PrioritizedSample {
    /// Indices of sampled transitions.
    pub indices: Vec<usize>,
    /// Importance sampling weights for each sample.
    pub weights: Vec<f32>,
}

/// Prioritized sampler using a sum tree for efficient sampling.
///
/// The sum tree allows O(log N) sampling and priority updates.
#[derive(Debug, Clone)]
pub struct PrioritizedSampler {
    /// Sum tree for efficient prioritized sampling.
    tree: SumTree,
    /// Configuration.
    config: PrioritizationConfig,
    /// Maximum priority seen so far (for new transitions).
    max_priority: f32,
    /// Current beta value (can be annealed).
    current_beta: f32,
}

impl PrioritizedSampler {
    /// Create a new prioritized sampler.
    pub fn new(capacity: usize, config: PrioritizationConfig) -> Self {
        Self {
            tree: SumTree::new(capacity),
            current_beta: config.beta,
            config,
            max_priority: 1.0,
        }
    }

    /// Add a new transition with maximum priority.
    pub fn add(&mut self, idx: usize) {
        let priority = self.max_priority.powf(self.config.alpha);
        self.tree.update(idx, priority);
    }

    /// Update priorities based on new values (e.g., TD errors).
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f32]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            let p = (priority.abs() + self.config.epsilon).powf(self.config.alpha);
            self.tree.update(idx, p);
            self.max_priority = self.max_priority.max(priority.abs() + self.config.epsilon);
        }
    }

    /// Sample a batch of transitions with importance sampling weights.
    pub fn sample(&self, batch_size: usize) -> PrioritizedSample {
        let mut rng = rand::thread_rng();
        let total = self.tree.total();

        if total == 0.0 {
            // Fallback to uniform sampling if tree is empty
            return PrioritizedSample {
                indices: (0..batch_size).collect(),
                weights: vec![1.0; batch_size],
            };
        }

        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        // Stratified sampling: divide total range into segments
        let segment_size = total / batch_size as f32;

        // Find minimum probability for normalization
        let min_prob = self.tree.min_nonzero() / total;
        let max_weight = (self.tree.len() as f32 * min_prob).powf(-self.current_beta);

        for i in 0..batch_size {
            // Sample uniformly within segment
            let low = segment_size * i as f32;
            let high = segment_size * (i + 1) as f32;
            let sample = rng.gen_range(low..high);

            // Find corresponding index
            let (idx, priority) = self.tree.get(sample);
            indices.push(idx);

            // Compute importance sampling weight
            let prob = priority / total;
            let weight = (self.tree.len() as f32 * prob).powf(-self.current_beta);
            weights.push(weight / max_weight); // Normalize by max weight
        }

        PrioritizedSample { indices, weights }
    }

    /// Sample uniformly (ignores priorities).
    pub fn sample_uniform(&self, batch_size: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let len = self.tree.len();
        (0..batch_size)
            .map(|_| rng.gen_range(0..len))
            .collect()
    }

    /// Anneal beta towards 1.0.
    pub fn anneal_beta(&mut self, progress: f32) {
        // Linear interpolation from initial beta to 1.0
        self.current_beta = self.config.beta + (1.0 - self.config.beta) * progress.clamp(0.0, 1.0);
    }

    /// Get current beta value.
    pub fn beta(&self) -> f32 {
        self.current_beta
    }

    /// Get the configuration.
    pub fn config(&self) -> &PrioritizationConfig {
        &self.config
    }

    /// Get the tree capacity.
    pub fn capacity(&self) -> usize {
        self.tree.capacity()
    }
}

/// Sum tree data structure for efficient prioritized sampling.
///
/// The sum tree stores priorities in leaves and maintains partial sums in
/// internal nodes, allowing O(log N) sampling and updates.
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Tree nodes: first half are internal nodes, second half are leaves.
    tree: Vec<f32>,
    /// Capacity (number of leaves).
    capacity: usize,
}

impl SumTree {
    /// Create a new sum tree with the given capacity.
    pub fn new(capacity: usize) -> Self {
        // Tree has 2 * capacity - 1 nodes
        let tree_size = 2 * capacity - 1;
        Self {
            tree: vec![0.0; tree_size],
            capacity,
        }
    }

    /// Update the priority of a leaf.
    pub fn update(&mut self, idx: usize, priority: f32) {
        // Leaf index in tree array
        let tree_idx = self.capacity - 1 + idx;

        // Compute change
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;

        // Propagate up
        let mut current = tree_idx;
        while current > 0 {
            current = (current - 1) / 2;
            self.tree[current] += change;
        }
    }

    /// Sample an index based on a value in [0, total).
    pub fn get(&self, value: f32) -> (usize, f32) {
        let mut value = value;
        let mut idx = 0;

        // Traverse down the tree
        while idx < self.capacity - 1 {
            let left = 2 * idx + 1;
            let right = left + 1;

            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }

        // Convert tree index to data index
        let data_idx = idx - (self.capacity - 1);
        (data_idx, self.tree[idx])
    }

    /// Get total priority sum.
    pub fn total(&self) -> f32 {
        self.tree[0]
    }

    /// Get minimum non-zero priority.
    pub fn min_nonzero(&self) -> f32 {
        self.tree[(self.capacity - 1)..]
            .iter()
            .filter(|&&p| p > 0.0)
            .cloned()
            .fold(f32::MAX, f32::min)
    }

    /// Get number of leaves.
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.total() == 0.0
    }
}

/// Simple priority buffer that tracks advantages for prioritization.
#[derive(Debug, Clone)]
pub struct PriorityBuffer {
    /// Priorities for each transition.
    priorities: Vec<f32>,
    /// Configuration.
    config: PrioritizationConfig,
}

impl PriorityBuffer {
    /// Create a new priority buffer.
    pub fn new(capacity: usize, config: PrioritizationConfig) -> Self {
        Self {
            priorities: vec![1.0; capacity],
            config,
        }
    }

    /// Update priorities from advantages.
    pub fn update_from_advantages(&mut self, advantages: &[f32]) {
        for (i, &adv) in advantages.iter().enumerate() {
            if i < self.priorities.len() {
                self.priorities[i] = (adv.abs() + self.config.epsilon).powf(self.config.alpha);
            }
        }
    }

    /// Update priorities from TD errors.
    pub fn update_from_td_errors(&mut self, td_errors: &[f32]) {
        for (i, &td) in td_errors.iter().enumerate() {
            if i < self.priorities.len() {
                self.priorities[i] = (td.abs() + self.config.epsilon).powf(self.config.alpha);
            }
        }
    }

    /// Get sampling weights for the given indices.
    pub fn get_weights(&self, indices: &[usize], beta: f32) -> Vec<f32> {
        let total: f32 = self.priorities.iter().sum();
        if total == 0.0 {
            return vec![1.0; indices.len()];
        }

        let n = self.priorities.len() as f32;

        // Find max weight for normalization
        let min_prob = self.priorities
            .iter()
            .filter(|&&p| p > 0.0)
            .cloned()
            .fold(f32::MAX, f32::min) / total;
        let max_weight = (n * min_prob).powf(-beta);

        indices
            .iter()
            .map(|&idx| {
                let prob = self.priorities[idx] / total;
                let weight = (n * prob).powf(-beta);
                weight / max_weight
            })
            .collect()
    }

    /// Sample indices based on priorities.
    pub fn sample(&self, batch_size: usize) -> Vec<usize> {
        let total: f32 = self.priorities.iter().sum();
        if total == 0.0 {
            // Fallback to sequential
            return (0..batch_size.min(self.priorities.len())).collect();
        }

        let mut rng = rand::thread_rng();
        let mut indices = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let sample = rng.gen_range(0.0..total);
            let mut cumsum = 0.0;

            for (i, &p) in self.priorities.iter().enumerate() {
                cumsum += p;
                if sample <= cumsum {
                    indices.push(i);
                    break;
                }
            }
        }

        // Pad if needed
        while indices.len() < batch_size {
            indices.push(rng.gen_range(0..self.priorities.len()));
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prioritization_config() {
        let config = PrioritizationConfig::new()
            .with_alpha(0.7)
            .with_beta(0.5);

        assert_eq!(config.alpha, 0.7);
        assert_eq!(config.beta, 0.5);
    }

    #[test]
    fn test_sum_tree_basic() {
        let mut tree = SumTree::new(4);

        // Set priorities
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert!((tree.total() - 10.0).abs() < 1e-6);
        assert_eq!(tree.len(), 4);
    }

    #[test]
    fn test_sum_tree_sampling() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        // Sample at different values
        let (idx0, _) = tree.get(0.5);  // Should get index 0
        assert_eq!(idx0, 0);

        let (idx1, _) = tree.get(2.5);  // Should get index 1
        assert_eq!(idx1, 1);

        let (idx2, _) = tree.get(5.5);  // Should get index 2
        assert_eq!(idx2, 2);

        let (idx3, _) = tree.get(9.5);  // Should get index 3
        assert_eq!(idx3, 3);
    }

    #[test]
    fn test_prioritized_sampler() {
        let config = PrioritizationConfig::new();
        let mut sampler = PrioritizedSampler::new(100, config);

        // Add transitions
        for i in 0..100 {
            sampler.add(i);
        }

        // Update priorities
        let indices: Vec<usize> = (0..10).collect();
        let priorities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        sampler.update_priorities(&indices, &priorities);

        // Sample
        let sample = sampler.sample(32);
        assert_eq!(sample.indices.len(), 32);
        assert_eq!(sample.weights.len(), 32);

        // Weights should be positive and normalized
        for &w in &sample.weights {
            assert!(w > 0.0);
            assert!(w <= 1.0);
        }
    }

    #[test]
    fn test_beta_annealing() {
        let config = PrioritizationConfig::new().with_beta(0.4);
        let mut sampler = PrioritizedSampler::new(100, config);

        assert_eq!(sampler.beta(), 0.4);

        sampler.anneal_beta(0.5);
        assert!((sampler.beta() - 0.7).abs() < 0.01); // 0.4 + 0.6 * 0.5 = 0.7

        sampler.anneal_beta(1.0);
        assert!((sampler.beta() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_buffer() {
        let config = PrioritizationConfig::new();
        let mut buffer = PriorityBuffer::new(10, config);

        // Update from advantages
        let advantages: Vec<f32> = (0..10).map(|i| i as f32).collect();
        buffer.update_from_advantages(&advantages);

        // Sample
        let indices = buffer.sample(5);
        assert_eq!(indices.len(), 5);

        // Get weights
        let weights = buffer.get_weights(&indices, 0.4);
        assert_eq!(weights.len(), 5);
    }
}
