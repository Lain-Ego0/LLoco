"""Source-parity checks for the staged Go2 migration."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.mdp.observations import single_frame_noise_bounds


def test_only_completed_trot_is_registered() -> None:
  tasks = set(list_tasks())
  assert "Unitree-Go2-Trot-Flat" in tasks
  incomplete = {
    "Unitree-Go2-Jump-Flat",
    "Unitree-Go2-Spring-Jump-Flat",
    "Unitree-Go2-Handstand-Flat",
  }
  assert tasks.isdisjoint(incomplete)


def test_trot_timing_initial_state_and_action() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  robot = cfg.scene.entities["robot"]
  action = cfg.actions["joint_pos"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 24.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert robot.init_state.pos == (0.0, 0.0, 0.42)
  assert robot.init_state.joint_pos[".*thigh_joint"] == 0.8
  assert robot.init_state.joint_pos[".*calf_joint"] == -1.5
  assert action.scale == 0.25
  assert action.delay_min_lag == 1
  assert action.delay_max_lag == 3


def test_trot_observation_layout_noise_and_history() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  actor = cfg.observations["actor"].terms["history"]
  critic = cfg.observations["critic"].terms["history"]
  assert actor.func.frame_dim == 47
  assert actor.func.history_length == 10
  assert critic.func.frame_dim == 68
  assert critic.func.history_length == 3
  assert actor.params["add_noise"] is True
  assert actor.noise is None
  noise_max = single_frame_noise_bounds()[1]
  assert len(noise_max) == 47
  assert noise_max[:5] == (0.0,) * 5
  assert noise_max[5:8] == (0.05,) * 3
  assert noise_max[8:11] == (0.1,) * 3
  assert noise_max[11:23] == (0.01,) * 12
  assert noise_max[23:35] == (0.07500000000000001,) * 12
  assert noise_max[35:47] == (0.0,) * 12
  play = load_env_cfg("Unitree-Go2-Trot-Flat", play=True)
  assert play.observations["actor"].terms["history"].params["add_noise"] is False


def test_trot_reward_names_weights_and_gates() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  expected = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 2.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -2.0,
    "torques": -0.0001,
    "dof_acc": -2.5e-7,
    "collision": -1.0,
    "action_rate": -0.01,
    "stand_still": -1.0,
    "base_height": -5.0,
    "trot": 0.8,
    "feet_clearance": 0.1,
    "default_hip_pos": -0.2,
    "default_pos": -0.1,
    "contact_without_command": 1.0,
  }
  assert {name: term.weight for name, term in cfg.rewards.items()} == expected
  assert cfg.rewards["tracking_lin_vel"].params["sigma"] == 0.25
  assert cfg.rewards["trot"].params["cycle_time"] == 0.5
  assert cfg.rewards["base_height"].params["target_height"] == 0.29
  assert cfg.rewards["feet_clearance"].params["target_foot_height"] == 0.06
  assert cfg.scale_rewards_by_dt


def test_trot_commands_events_termination_and_curriculum() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  command = cfg.commands["twist"]
  assert command.resampling_time_range == (5.0, 5.0)
  assert command.ranges.lin_vel_x == (-1.0, 1.0)
  assert command.ranges.lin_vel_y == (-1.0, 1.0)
  assert command.ranges.ang_vel_z == (-1.0, 1.0)
  assert cfg.events["push_robot"].interval_range_s == (4.0, 4.0)
  assert cfg.events["push_robot"].is_global_time
  assert cfg.events["friction"].params["ranges"] == (0.2, 1.2)
  assert cfg.events["base_mass"].params["ranges"] == (-1.0, 2.0)
  assert cfg.events["link_mass"].params["ranges"] == (0.9, 1.1)
  assert cfg.events["motor_zero_offset"].params["bias_range"] == (-0.035, 0.035)
  assert cfg.terminations["base_contact"].params["force_threshold"] == 1.0
  assert set(cfg.curriculum) == {"command_velocity"}


def test_trot_ppo_parameters() -> None:
  cfg = load_rl_cfg("Unitree-Go2-Trot-Flat")
  assert cfg.seed == 1
  assert cfg.num_steps_per_env == 24
  assert cfg.max_iterations == 15_000
  assert cfg.save_interval == 100
  assert cfg.clip_actions == 100.0
  assert cfg.algorithm.learning_rate == 1.0e-5
  assert cfg.algorithm.num_learning_epochs == 5
  assert cfg.algorithm.num_mini_batches == 4
  assert cfg.algorithm.entropy_coef == 0.01
  assert cfg.actor.hidden_dims == (512, 256, 128)
  assert cfg.critic.hidden_dims == (512, 256, 128)
  assert not cfg.actor.obs_normalization
  assert not cfg.critic.obs_normalization


def test_existing_go2_velocity_tasks_unchanged() -> None:
  flat = load_env_cfg("Unitree-Go2-Flat")
  rough = load_env_cfg("Unitree-Go2-Rough")
  assert "history" not in flat.observations["actor"].terms
  assert "trot" not in rough.rewards
