//! Preset reward configurations for common tasks.
//!
//! These presets return concrete tuple types for zero-cost composition.

use super::components::*;
use crate::constants::HOVER_RPM;

/// Type alias for the standard hover reward composition.
pub type HoverReward = (
    PositionError,
    VelocityError,
    AttitudePenalty,
    AngularVelocityPenalty,
    ActionMagnitudePenalty,
    ActionRatePenalty,
    AliveBonus,
);

/// Type alias for the tracking reward composition.
pub type TrackingReward = (
    PositionError,
    VelocityError,
    AttitudePenalty,
    AngularVelocityPenalty,
    ActionMagnitudePenalty,
    ActionRatePenalty,
    AliveBonus,
);

/// Standard hover reward: position error + stability penalties.
///
/// Weights optimized for hovering at a fixed target position.
///
/// # Components
/// - Position error: 1.0 (primary objective)
/// - Velocity error: 0.1 (encourage stillness)
/// - Attitude penalty: 0.5 (stay level)
/// - Angular velocity penalty: 0.01 (minimize spin)
/// - Action magnitude penalty: 0.001 (efficient thrust)
/// - Action rate penalty: 0.0001 (smooth control)
/// - Alive bonus: 0.1 (survival reward)
pub fn hover() -> HoverReward {
    (
        PositionError { weight: 1.0 },
        VelocityError { weight: 0.1 },
        AttitudePenalty { weight: 0.5 },
        AngularVelocityPenalty { weight: 0.01 },
        ActionMagnitudePenalty {
            weight: 0.001,
            reference_rpm: HOVER_RPM,
        },
        ActionRatePenalty { weight: 0.0001 },
        AliveBonus { bonus: 0.1 },
    )
}

/// Tracking reward: position + velocity errors for trajectory following.
///
/// Weights optimized for tracking moving targets.
///
/// # Components
/// - Position error: 1.0 (primary objective)
/// - Velocity error: 0.5 (match target velocity)
/// - Attitude penalty: 0.2 (allow tilting for movement)
/// - Angular velocity penalty: 0.01 (minimize spin)
/// - Action magnitude penalty: 0.001 (efficient thrust)
/// - Action rate penalty: 0.0001 (smooth control)
/// - Alive bonus: 0.1 (survival reward)
pub fn tracking() -> TrackingReward {
    (
        PositionError { weight: 1.0 },
        VelocityError { weight: 0.5 },
        AttitudePenalty { weight: 0.2 },
        AngularVelocityPenalty { weight: 0.01 },
        ActionMagnitudePenalty {
            weight: 0.001,
            reference_rpm: HOVER_RPM,
        },
        ActionRatePenalty { weight: 0.0001 },
        AliveBonus { bonus: 0.1 },
    )
}

/// Aggressive reward: emphasizes position with minimal constraints.
///
/// Allows aggressive maneuvers to reach the target quickly.
pub fn aggressive() -> HoverReward {
    (
        PositionError { weight: 2.0 },
        VelocityError { weight: 0.0 },
        AttitudePenalty { weight: 0.1 },
        AngularVelocityPenalty { weight: 0.0 },
        ActionMagnitudePenalty {
            weight: 0.0,
            reference_rpm: HOVER_RPM,
        },
        ActionRatePenalty { weight: 0.0 },
        AliveBonus { bonus: 0.05 },
    )
}

/// Smooth reward: emphasizes stability and smooth control.
///
/// Produces conservative, stable flight behavior.
pub fn smooth() -> HoverReward {
    (
        PositionError { weight: 0.5 },
        VelocityError { weight: 0.2 },
        AttitudePenalty { weight: 1.0 },
        AngularVelocityPenalty { weight: 0.1 },
        ActionMagnitudePenalty {
            weight: 0.01,
            reference_rpm: HOVER_RPM,
        },
        ActionRatePenalty { weight: 0.01 },
        AliveBonus { bonus: 0.2 },
    )
}

/// Energy-efficient reward: minimizes power consumption.
///
/// Penalizes high thrust and frequent action changes.
pub fn energy_efficient() -> HoverReward {
    (
        PositionError { weight: 0.8 },
        VelocityError { weight: 0.1 },
        AttitudePenalty { weight: 0.3 },
        AngularVelocityPenalty { weight: 0.02 },
        ActionMagnitudePenalty {
            weight: 0.1,
            reference_rpm: HOVER_RPM,
        },
        ActionRatePenalty { weight: 0.01 },
        AliveBonus { bonus: 0.15 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reward::RewardComponent;
    use crate::state::QuadcopterState;

    #[test]
    fn test_hover_preset() {
        let reward_fn = hover();
        let state = QuadcopterState::new(1, 0);
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_tracking_preset() {
        let reward_fn = tracking();
        let state = QuadcopterState::new(1, 0);
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_aggressive_preset() {
        let reward_fn = aggressive();
        let state = QuadcopterState::new(1, 0);
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_smooth_preset() {
        let reward_fn = smooth();
        let state = QuadcopterState::new(1, 0);
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_energy_efficient_preset() {
        let reward_fn = energy_efficient();
        let state = QuadcopterState::new(1, 0);
        let reward = reward_fn.compute(&state, 0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_hover_matches_expected_weights() {
        let h = hover();
        assert_eq!(h.0.weight, 1.0); // position
        assert_eq!(h.1.weight, 0.1); // velocity
        assert_eq!(h.2.weight, 0.5); // attitude
        assert_eq!(h.3.weight, 0.01); // angular velocity
        assert_eq!(h.4.weight, 0.001); // action magnitude
        assert_eq!(h.5.weight, 0.0001); // action rate
        assert_eq!(h.6.bonus, 0.1); // alive bonus
    }
}
