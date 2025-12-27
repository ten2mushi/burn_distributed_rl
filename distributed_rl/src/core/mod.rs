//! Core types and abstractions for Distributed.

pub mod bytes_slot;
pub mod experience_buffer;
pub mod model_slot;
pub mod model_version;
pub mod record_slot;
pub mod recurrent;
pub mod shared_buffer;
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
