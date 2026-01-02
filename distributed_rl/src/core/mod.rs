//! Core types and abstractions for Distributed.

pub mod bytes_slot;
pub mod episode_state;
pub mod experience_buffer;
pub mod model_slot;
pub mod model_version;
pub mod record_slot;
pub mod recurrent;
pub mod running_stats;
pub mod shared_buffer;
pub mod target_network;
pub mod transition;

pub use bytes_slot::{BytesSlot, SharedBytesSlot, bytes_slot, bytes_slot_with};
pub use experience_buffer::{
    ExperienceBuffer, OnPolicyBuffer, OffPolicyBuffer,
    SharedExperienceBuffer, BufferConfig,
};
pub use model_slot::{ModelSlot, SharedModelSlot, model_slot, model_slot_with};
pub use record_slot::{RecordSlot, SharedRecordSlot, record_slot, record_slot_with};
pub use shared_buffer::{SharedBuffer, SharedReplayBuffer, shared_buffer};
pub use recurrent::{
    GruCell, GruCellConfig, HiddenState, LstmCellConfig, LstmCellWrapper, RecurrentCell,
    RecurrentCellType,
};
pub use running_stats::{RunningMeanStd, SharedRunningMeanStd, RunningScalarStats};
pub use target_network::{
    SoftUpdatable, soft_update, hard_copy,
    TargetNetworkConfig, TargetNetworkManager, ExponentialMovingAverage,
};
pub use episode_state::EpisodeState;
