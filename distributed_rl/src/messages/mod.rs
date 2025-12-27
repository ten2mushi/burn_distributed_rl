//! Message-passing infrastructure for distributed RL.
//!
//! This module defines the messages used for communication between
//! distributed training components (Actors, Learner, Coordinator, Evaluator).
//!
//! # Architecture
//!
//! ```text
//!                     +-------------------+
//!                     |    Coordinator    |
//!                     +-------------------+
//!                            |
//!          +-----------------+------------------+
//!          |                 |                  |
//!          v                 v                  v
//!    +----------+      +----------+      +------------+
//!    | Actor    |      | Learner  |      | Evaluator  |
//!    | Inbox    |      | Inbox    |      | Inbox      |
//!    +----------+      +----------+      +------------+
//! ```

mod actor_msg;
mod learner_msg;
mod coordinator_msg;
mod eval_msg;

#[cfg(test)]
mod tests;

pub use actor_msg::{ActorMsg, ActorStats};
pub use learner_msg::{LearnerMsg, LearnerStats};
pub use coordinator_msg::{CoordinatorMsg, FinishReason};
pub use eval_msg::{EvalMsg, EvalResult};
