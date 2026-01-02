//! Training Configuration Presets
//!
//! Pre-configured setups for common training scenarios.

use super::actions::ActionConfig;
use super::multi_agent::MultiAgentConfig;
use super::rewards::{CRRewardConfig, JammerRewardConfig, RewardConfig};

// ============================================================================
// Self-Play Configuration
// ============================================================================

/// Configuration for self-play training
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Whether agents act simultaneously or alternating
    pub simultaneous: bool,
    /// Fraction of time to freeze jammer policy (for CR curriculum)
    pub jammer_freeze_ratio: f32,
    /// Fraction of time to freeze CR policy (for jammer curriculum)
    pub cr_freeze_ratio: f32,
    /// Whether to use a policy pool for diverse opponents
    pub use_policy_pool: bool,
    /// Size of policy pool
    pub pool_size: usize,
    /// How often to add current policy to pool (steps)
    pub pool_update_interval: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simultaneous: true,
            jammer_freeze_ratio: 0.0,
            cr_freeze_ratio: 0.0,
            use_policy_pool: false,
            pool_size: 10,
            pool_update_interval: 10000,
        }
    }
}

impl SelfPlayConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable alternating play
    pub fn with_alternating(mut self) -> Self {
        self.simultaneous = false;
        self
    }

    /// Set jammer freeze ratio for CR curriculum
    pub fn with_jammer_freeze(mut self, ratio: f32) -> Self {
        self.jammer_freeze_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set CR freeze ratio for jammer curriculum
    pub fn with_cr_freeze(mut self, ratio: f32) -> Self {
        self.cr_freeze_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Enable policy pool
    pub fn with_policy_pool(mut self, size: usize) -> Self {
        self.use_policy_pool = true;
        self.pool_size = size;
        self
    }
}

// ============================================================================
// Team Reward Configuration
// ============================================================================

/// Configuration for team-based rewards
#[derive(Clone, Debug)]
pub struct TeamRewardConfig {
    /// Weight for team coordination bonus
    pub coordination_weight: f32,
    /// Bonus for synchronized jamming across multiple CRs
    pub sync_jam_bonus: f32,
    /// Bonus for CRs using different frequencies (diversity)
    pub diversity_bonus: f32,
    /// Penalty for friendly fire (jammers interfering with each other)
    pub friendly_fire_penalty: f32,
}

impl Default for TeamRewardConfig {
    fn default() -> Self {
        Self {
            coordination_weight: 0.1,
            sync_jam_bonus: 0.5,
            diversity_bonus: 0.2,
            friendly_fire_penalty: 0.3,
        }
    }
}

impl TeamRewardConfig {
    /// Fully cooperative team rewards
    pub fn cooperative() -> Self {
        Self {
            coordination_weight: 0.5,
            sync_jam_bonus: 1.0,
            diversity_bonus: 0.5,
            friendly_fire_penalty: 1.0,
        }
    }

    /// Competitive (individual) rewards
    pub fn competitive() -> Self {
        Self {
            coordination_weight: 0.0,
            sync_jam_bonus: 0.0,
            diversity_bonus: 0.0,
            friendly_fire_penalty: 0.0,
        }
    }
}

// ============================================================================
// Environment Presets
// ============================================================================

/// Complete training scenario configuration
#[derive(Clone, Debug)]
pub struct ScenarioConfig {
    /// Multi-agent configuration
    pub multi_agent: MultiAgentConfig,
    /// Reward configuration
    pub rewards: RewardConfig,
    /// Self-play configuration
    pub self_play: SelfPlayConfig,
    /// Team reward configuration
    pub team: TeamRewardConfig,
    /// Description of this scenario
    pub description: String,
}

impl ScenarioConfig {
    /// Dense urban scenario
    ///
    /// High interference, multiple agents, challenging environment
    pub fn dense_urban() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(2)
                .with_crs(4)
                .with_history_length(16)
                .with_action_config(
                    ActionConfig::new()
                        .with_freq_range(2.4e9, 2.5e9)
                        .with_power_range(0.0, 30.0)
                        .with_bandwidth_range(5e6, 40e6),
                ),
            rewards: RewardConfig::new()
                .with_jammer(JammerRewardConfig::efficient())
                .with_cr(CRRewardConfig::cautious()),
            self_play: SelfPlayConfig::new(),
            team: TeamRewardConfig::default(),
            description: "Dense urban: 2 jammers vs 4 CRs, high interference".to_string(),
        }
    }

    /// Fast hopping scenario
    ///
    /// CRs must hop quickly to avoid reactive jammers
    pub fn fast_hopping() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(1)
                .with_crs(1)
                .with_history_length(32)
                .with_action_config(
                    ActionConfig::new()
                        .with_freq_range(2.4e9, 2.5e9)
                        .with_power_range(-10.0, 20.0)
                        .with_bandwidth_range(1e6, 20e6),
                )
                .with_sinr_threshold(5.0), // Lower threshold
            rewards: RewardConfig::new()
                .with_jammer(JammerRewardConfig::aggressive())
                .with_cr(CRRewardConfig {
                    switching_penalty: 0.0, // No penalty for switching
                    throughput_weight: 1.5,
                    ..CRRewardConfig::default()
                }),
            self_play: SelfPlayConfig::new().with_jammer_freeze(0.3),
            team: TeamRewardConfig::competitive(),
            description: "Fast hopping: 1v1, frequent frequency changes".to_string(),
        }
    }

    /// Adversarial training scenario
    ///
    /// Zero-sum game with policy pool
    pub fn adversarial_training() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(1)
                .with_crs(1)
                .with_history_length(8),
            rewards: RewardConfig::new()
                .with_zero_sum(true)
                .with_jammer(JammerRewardConfig::default())
                .with_cr(CRRewardConfig::default()),
            self_play: SelfPlayConfig::new()
                .with_policy_pool(20)
                .with_alternating(),
            team: TeamRewardConfig::competitive(),
            description: "Adversarial: 1v1, zero-sum, policy pool".to_string(),
        }
    }

    /// Simple baseline scenario
    ///
    /// 1 jammer vs 1 CR for basic testing
    pub fn simple_baseline() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(1)
                .with_crs(1)
                .with_history_length(4),
            rewards: RewardConfig::default(),
            self_play: SelfPlayConfig::default(),
            team: TeamRewardConfig::competitive(),
            description: "Simple baseline: 1v1, default settings".to_string(),
        }
    }

    /// Team coordination scenario
    ///
    /// Multiple jammers must coordinate to jam multiple CRs
    pub fn team_coordination() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(3)
                .with_crs(3)
                .with_history_length(8),
            rewards: RewardConfig::new()
                .with_jammer(JammerRewardConfig {
                    disruption_weight: 0.5,
                    target_match_weight: 1.0,
                    ..JammerRewardConfig::default()
                })
                .with_cr(CRRewardConfig::default()),
            self_play: SelfPlayConfig::new(),
            team: TeamRewardConfig::cooperative(),
            description: "Team coordination: 3v3, cooperative rewards".to_string(),
        }
    }

    /// Wideband scenario
    ///
    /// Large frequency range, wideband signals
    pub fn wideband() -> Self {
        Self {
            multi_agent: MultiAgentConfig::new()
                .with_jammers(2)
                .with_crs(2)
                .with_action_config(
                    ActionConfig::new()
                        .with_freq_range(1e9, 6e9) // 1-6 GHz
                        .with_power_range(-10.0, 40.0)
                        .with_bandwidth_range(10e6, 100e6),
                ),
            rewards: RewardConfig::default(),
            self_play: SelfPlayConfig::default(),
            team: TeamRewardConfig::default(),
            description: "Wideband: 2v2, 1-6 GHz, wide bandwidth".to_string(),
        }
    }
}

// ============================================================================
// Curriculum Learning Stages
// ============================================================================

/// Stage in curriculum learning
#[derive(Clone, Debug)]
pub struct CurriculumStage {
    /// Stage name
    pub name: String,
    /// Scenario configuration for this stage
    pub config: ScenarioConfig,
    /// Number of training steps in this stage
    pub steps: usize,
    /// Minimum performance to advance (normalized 0-1)
    pub advancement_threshold: f32,
}

/// Curriculum for progressive training
#[derive(Clone, Debug)]
pub struct Curriculum {
    /// Stages in order
    pub stages: Vec<CurriculumStage>,
    /// Current stage index
    pub current_stage: usize,
}

impl Curriculum {
    /// Create jammer curriculum (easy to hard CRs)
    pub fn jammer_curriculum() -> Self {
        Self {
            stages: vec![
                CurriculumStage {
                    name: "Static CR".to_string(),
                    config: ScenarioConfig {
                        multi_agent: MultiAgentConfig::new()
                            .with_jammers(1)
                            .with_crs(1),
                        self_play: SelfPlayConfig::new().with_cr_freeze(1.0),
                        ..ScenarioConfig::simple_baseline()
                    },
                    steps: 50000,
                    advancement_threshold: 0.8,
                },
                CurriculumStage {
                    name: "Slow Hopping CR".to_string(),
                    config: ScenarioConfig {
                        multi_agent: MultiAgentConfig::new()
                            .with_jammers(1)
                            .with_crs(1),
                        self_play: SelfPlayConfig::new().with_cr_freeze(0.5),
                        ..ScenarioConfig::simple_baseline()
                    },
                    steps: 100000,
                    advancement_threshold: 0.7,
                },
                CurriculumStage {
                    name: "Fast Hopping CR".to_string(),
                    config: ScenarioConfig::fast_hopping(),
                    steps: 200000,
                    advancement_threshold: 0.6,
                },
                CurriculumStage {
                    name: "Adversarial CR".to_string(),
                    config: ScenarioConfig::adversarial_training(),
                    steps: 500000,
                    advancement_threshold: 0.5,
                },
            ],
            current_stage: 0,
        }
    }

    /// Create CR curriculum (easy to hard jammers)
    pub fn cr_curriculum() -> Self {
        Self {
            stages: vec![
                CurriculumStage {
                    name: "No Jammer".to_string(),
                    config: ScenarioConfig {
                        multi_agent: MultiAgentConfig::new()
                            .with_jammers(0)
                            .with_crs(1),
                        ..ScenarioConfig::simple_baseline()
                    },
                    steps: 20000,
                    advancement_threshold: 0.9,
                },
                CurriculumStage {
                    name: "Random Jammer".to_string(),
                    config: ScenarioConfig {
                        multi_agent: MultiAgentConfig::new()
                            .with_jammers(1)
                            .with_crs(1),
                        self_play: SelfPlayConfig::new().with_jammer_freeze(1.0),
                        ..ScenarioConfig::simple_baseline()
                    },
                    steps: 50000,
                    advancement_threshold: 0.8,
                },
                CurriculumStage {
                    name: "Learning Jammer".to_string(),
                    config: ScenarioConfig::simple_baseline(),
                    steps: 200000,
                    advancement_threshold: 0.6,
                },
                CurriculumStage {
                    name: "Expert Jammer".to_string(),
                    config: ScenarioConfig::adversarial_training(),
                    steps: 500000,
                    advancement_threshold: 0.5,
                },
            ],
            current_stage: 0,
        }
    }

    /// Get current stage configuration
    pub fn current_config(&self) -> &ScenarioConfig {
        &self.stages[self.current_stage].config
    }

    /// Advance to next stage if threshold met
    ///
    /// Returns true if advanced
    pub fn try_advance(&mut self, performance: f32) -> bool {
        if self.current_stage < self.stages.len() - 1
            && performance >= self.stages[self.current_stage].advancement_threshold
        {
            self.current_stage += 1;
            true
        } else {
            false
        }
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.current_stage >= self.stages.len() - 1
    }

    /// Reset to first stage
    pub fn reset(&mut self) {
        self.current_stage = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_presets() {
        // Just verify they don't panic
        let _ = ScenarioConfig::dense_urban();
        let _ = ScenarioConfig::fast_hopping();
        let _ = ScenarioConfig::adversarial_training();
        let _ = ScenarioConfig::simple_baseline();
        let _ = ScenarioConfig::team_coordination();
        let _ = ScenarioConfig::wideband();
    }

    #[test]
    fn test_curriculum_advancement() {
        let mut curriculum = Curriculum::jammer_curriculum();

        assert_eq!(curriculum.current_stage, 0);

        // Below threshold
        assert!(!curriculum.try_advance(0.5));
        assert_eq!(curriculum.current_stage, 0);

        // At threshold
        assert!(curriculum.try_advance(0.8));
        assert_eq!(curriculum.current_stage, 1);

        // Can't advance past last
        curriculum.current_stage = curriculum.stages.len() - 1;
        assert!(!curriculum.try_advance(1.0));
    }

    #[test]
    fn test_self_play_config() {
        let config = SelfPlayConfig::new()
            .with_alternating()
            .with_policy_pool(15);

        assert!(!config.simultaneous);
        assert!(config.use_policy_pool);
        assert_eq!(config.pool_size, 15);
    }

    #[test]
    fn test_team_reward_presets() {
        let coop = TeamRewardConfig::cooperative();
        let comp = TeamRewardConfig::competitive();

        assert!(coop.coordination_weight > comp.coordination_weight);
        assert!(coop.sync_jam_bonus > comp.sync_jam_bonus);
    }
}
