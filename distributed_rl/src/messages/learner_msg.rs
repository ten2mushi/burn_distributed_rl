//! Messages for Learner thread.
//!
//! # API Design
//!
//! `LearnerMsg<M>` uses a phantom type parameter for API consistency with `ActorMsg<M>`.
//! The `Clone` implementation is manual to avoid requiring `M: Clone`, since `M` is never
//! actually stored (only in `PhantomData`).

use std::marker::PhantomData;

/// Messages sent to the learner thread from Coordinator.
///
/// Type parameter M is unused but kept for API consistency with ActorMsg<M>.
/// This allows unified handling of message channels in coordinator code.
///
/// # Clone Behavior
///
/// `LearnerMsg<M>` implements `Clone` unconditionally (no `M: Clone` bound required)
/// because the model type `M` is only used as a phantom type marker and is never
/// actually cloned.
#[derive(Debug)]
pub enum LearnerMsg<M> {
    /// Stop the learner gracefully.
    Stop,

    /// Pause training (keep thread alive).
    Pause,

    /// Resume training after pause.
    Resume,

    /// Request statistics.
    RequestStats,

    /// Update learning rate dynamically.
    UpdateLearningRate(f64),

    /// Phantom for model type (unused, for API consistency).
    #[doc(hidden)]
    _Phantom(PhantomData<M>),
}

// Manual Clone implementation without M: Clone bound.
// PhantomData<M> is always Clone regardless of M's bounds.
impl<M> Clone for LearnerMsg<M> {
    fn clone(&self) -> Self {
        match self {
            LearnerMsg::Stop => LearnerMsg::Stop,
            LearnerMsg::Pause => LearnerMsg::Pause,
            LearnerMsg::Resume => LearnerMsg::Resume,
            LearnerMsg::RequestStats => LearnerMsg::RequestStats,
            LearnerMsg::UpdateLearningRate(lr) => LearnerMsg::UpdateLearningRate(*lr),
            LearnerMsg::_Phantom(_) => LearnerMsg::_Phantom(PhantomData),
        }
    }
}

/// Statistics reported by the learner.
#[derive(Debug, Clone, Default)]
pub struct LearnerStats {
    /// Total training steps completed.
    pub train_steps: usize,

    /// Average total loss.
    pub avg_loss: f32,

    /// Average policy loss.
    pub avg_policy_loss: f32,

    /// Average value loss.
    pub avg_value_loss: f32,

    /// Average entropy.
    pub avg_entropy: f32,

    /// Training steps per second.
    pub steps_per_second: f32,

    /// Current model version.
    pub model_version: u64,

    /// Buffer utilization (fraction full).
    pub buffer_utilization: f32,
}

impl LearnerStats {
    /// Create new learner stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with training step results.
    pub fn record_step(
        &mut self,
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
    ) {
        self.train_steps += 1;
        self.avg_loss = total_loss;
        self.avg_policy_loss = policy_loss;
        self.avg_value_loss = value_loss;
        self.avg_entropy = entropy;
    }

    /// Update model version.
    pub fn set_model_version(&mut self, version: u64) {
        self.model_version = version;
    }

    /// Update buffer utilization.
    pub fn set_buffer_utilization(&mut self, utilization: f32) {
        self.buffer_utilization = utilization;
    }

    /// Update steps per second.
    pub fn set_steps_per_second(&mut self, sps: f32) {
        self.steps_per_second = sps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type that deliberately does NOT implement Clone
    struct NonCloneModel {
        #[allow(dead_code)]
        data: Vec<u8>,
    }

    #[test]
    fn test_learner_msg_clone_without_m_clone() {
        // This test verifies that LearnerMsg<M> can be cloned even when M doesn't implement Clone.
        // This is important for API flexibility - users shouldn't be forced to make their
        // model types Clone just to use the message system.
        let msg: LearnerMsg<NonCloneModel> = LearnerMsg::Stop;
        let cloned = msg.clone();
        assert!(matches!(cloned, LearnerMsg::Stop));

        let msg: LearnerMsg<NonCloneModel> = LearnerMsg::UpdateLearningRate(0.001);
        let cloned = msg.clone();
        assert!(matches!(cloned, LearnerMsg::UpdateLearningRate(lr) if (lr - 0.001).abs() < 1e-9));
    }

    #[test]
    fn test_learner_msg_all_variants_clone() {
        // Verify all variants can be cloned
        let stop: LearnerMsg<()> = LearnerMsg::Stop;
        let pause: LearnerMsg<()> = LearnerMsg::Pause;
        let resume: LearnerMsg<()> = LearnerMsg::Resume;
        let request: LearnerMsg<()> = LearnerMsg::RequestStats;
        let lr: LearnerMsg<()> = LearnerMsg::UpdateLearningRate(0.01);
        let phantom: LearnerMsg<()> = LearnerMsg::_Phantom(PhantomData);

        assert!(matches!(stop.clone(), LearnerMsg::Stop));
        assert!(matches!(pause.clone(), LearnerMsg::Pause));
        assert!(matches!(resume.clone(), LearnerMsg::Resume));
        assert!(matches!(request.clone(), LearnerMsg::RequestStats));
        assert!(matches!(lr.clone(), LearnerMsg::UpdateLearningRate(_)));
        assert!(matches!(phantom.clone(), LearnerMsg::_Phantom(_)));
    }

    #[test]
    fn test_learner_stats_record_step() {
        let mut stats = LearnerStats::new();

        stats.record_step(0.8, 0.5, 0.3, 0.1);
        assert_eq!(stats.train_steps, 1);
        assert_eq!(stats.avg_policy_loss, 0.5);
        assert_eq!(stats.avg_value_loss, 0.3);

        stats.record_step(0.6, 0.4, 0.2, 0.15);
        assert_eq!(stats.train_steps, 2);
        assert_eq!(stats.avg_policy_loss, 0.4);
    }

    #[test]
    fn test_learner_stats_model_version() {
        let mut stats = LearnerStats::new();
        stats.set_model_version(42);
        assert_eq!(stats.model_version, 42);
    }
}
