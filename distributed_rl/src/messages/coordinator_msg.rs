//! Messages sent to the Coordinator.

use super::{ActorStats, LearnerStats, EvalResult};

/// Messages sent to the coordinator from workers.
///
/// The coordinator aggregates information from all components
/// and makes decisions about training progress.
#[derive(Debug, Clone)]
pub enum CoordinatorMsg {
    /// Actor reports statistics.
    ActorStats(ActorStats),

    /// Learner reports statistics.
    LearnerStats(LearnerStats),

    /// Evaluation result.
    EvalResult(EvalResult),

    /// Actor thread finished (either stopped or panicked).
    ActorFinished {
        actor_id: usize,
        reason: FinishReason,
    },

    /// Learner thread finished.
    LearnerFinished {
        reason: FinishReason,
    },

    /// Checkpoint saved.
    CheckpointSaved {
        step: usize,
        path: String,
    },
}

/// Reason why a thread finished.
#[derive(Debug, Clone)]
pub enum FinishReason {
    /// Normal shutdown after Stop message.
    Stopped,

    /// Thread panicked.
    Panicked(String),

    /// Training completed (reached max steps).
    Completed,
}

impl CoordinatorMsg {
    /// Create actor stats message.
    pub fn actor_stats(stats: ActorStats) -> Self {
        Self::ActorStats(stats)
    }

    /// Create learner stats message.
    pub fn learner_stats(stats: LearnerStats) -> Self {
        Self::LearnerStats(stats)
    }

    /// Create eval result message.
    pub fn eval_result(result: EvalResult) -> Self {
        Self::EvalResult(result)
    }
}
