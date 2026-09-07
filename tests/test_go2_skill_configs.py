"""Source-parity checks for the staged Go2 migration."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.hand_stand.mdp.observations import handstand_noise_bounds
from lloco.tasks.go2_skills.rear_stand.mdp.observations import rear_stand_noise_bounds
from lloco.tasks.go2_skills.trot.mdp.observations import single_frame_noise_bounds


def test_only_completed_staged_skills_are_registered() -> None:
  tasks = set(list_tasks())
  assert "Unitree-Go2-Trot-Flat" in tasks
  assert "Unitree-Go2-Jump-Flat" in tasks
  assert "Unitree-Go2-Rear-Stand-Flat" in tasks
  assert "Unitree-Go2-Handstand-Flat" in tasks
  assert "Unitree-Go2-Spring-Jump-Flat" in tasks
  incomplete = set()
  assert tasks.isdisjoint(incomplete)


def test_spring_jump_source_configuration() -> None:
  cfg = load_env_cfg("Unitree-Go2-Spring-Jump-Flat")
  actor = cfg.observations["actor"].terms["history"]
  critic = cfg.observations["critic"].terms["history"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 5.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.scene.entities["robot"].init_state.pos == (0.0, 0.0, 0.39)
  assert actor.func.frame_dim == 47 and actor.func.history_length == 10
  assert critic.func.frame_dim == 65 and critic.func.history_length == 3
  assert cfg.actions["joint_pos"].delay_min_lag == 1
  assert cfg.actions["joint_pos"].delay_max_lag == 3
  assert cfg.commands["jump_target"].ranges.lin_vel_x == (0.8, 1.2)
  assert cfg.events["friction"].params["num_buckets"] == 64
  assert set(cfg.rewards) == {
    "before_setting", "line_z", "flight", "base_height_flight",
    "base_height_stance", "orientation", "dof_pos", "dof_hip_pos",
    "ang_vel_xy", "torques", "dof_pos_limits", "dof_vel_limits",
    "dof_vel", "collision", "action_rate", "land_pos", "tracking_lin_vel",
    "line_vel_stance", "feet_contact_forces", "foot_clearance",
  }
  rl = load_rl_cfg("Unitree-Go2-Spring-Jump-Flat")
  assert rl.max_iterations == 50_000
  assert rl.algorithm.learning_rate == 1.0e-5
  assert rl.algorithm.class_name == (
    "lloco.tasks.go2_skills.spring_jump.mdp.symmetry:SourceSymmetricPPO"
  )


def test_backflip_source_configuration() -> None:
  cfg = load_env_cfg("Unitree-Go2-Backflip-Flat")
  actor = cfg.observations["actor"].terms["history"]
  critic = cfg.observations["critic"].terms["history"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 4.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert actor.func.frame_dim == 47 and actor.func.history_length == 10
  assert critic.func.frame_dim == 50 and critic.func.history_length == 3
  assert cfg.actions["joint_pos"].delay_min_lag == 1
  assert cfg.actions["joint_pos"].delay_max_lag == 3
  assert cfg.commands["flip"].ranges.lin_vel_x == (0.0, 0.0)
  assert cfg.events["friction"].params["num_buckets"] == 64
  assert cfg.events["friction"].params["low"] == 0.2
  assert cfg.events["friction"].params["high"] == 1.25
  assert {name: term.weight for name, term in cfg.rewards.items()} == {
    "before_setting": 5.0, "line_z": 25.0, "angle_y": 10.0,
    "base_height_flight": 5.0, "base_height_stance": 10.0,
    "orientation": 10.0, "orientation_before": 2.0, "dof_pos": -0.2,
    "line_vel_stance": -1.0, "ang_vel_xy": -0.2, "torques": -0.0001,
    "dof_pos_limits": -10.0, "dof_vel_limits": -2.0, "dof_vel": -0.001,
    "collision": -10.0, "action_rate": -0.01, "feet_contact_forces": -0.1,
    "land_pos": 1.0, "symmetric_joints": -0.3, "default_hip_pos": -0.5,
  }
  rl = load_rl_cfg("Unitree-Go2-Backflip-Flat")
  assert rl.seed == 1
  assert rl.max_iterations == 50_000
  assert rl.num_steps_per_env == 24
  assert rl.algorithm.learning_rate == 1.0e-5
  assert rl.algorithm.symmetry_cfg is None


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


def test_jump_timing_initial_state_action_and_observations() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat")
  robot = cfg.scene.entities["robot"]
  action = cfg.actions["joint_pos"]
  actor = cfg.observations["actor"].terms["history"]
  critic = cfg.observations["critic"].terms["history"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 24.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert robot.init_state.pos == (0.0, 0.0, 0.42)
  assert robot.init_state.joint_pos["FL_hip_joint"] == 0.1
  assert robot.init_state.joint_pos["FR_hip_joint"] == -0.1
  assert robot.init_state.joint_pos["RL_thigh_joint"] == 1.0
  assert robot.init_state.joint_pos["RR_thigh_joint"] == 1.0
  assert action.scale == 0.25
  assert action.delay_min_lag == 1
  assert action.delay_max_lag == 3
  assert actor.func.frame_dim == 47
  assert actor.func.history_length == 10
  assert critic.func.frame_dim == 70
  assert critic.func.history_length == 3


def test_jump_rewards_commands_events_and_ppo() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat")
  expected = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 2.0,
    "lin_vel_z": 0.05,
    "ang_vel_xy": 0.2,
    "orientation": 0.6,
    "torques": -0.0002,
    "dof_acc": -5.5e-4,
    "base_height": 1.0,
    "feet_air_time": 1.0,
    "collision": -1.0,
    "action_rate": -0.01,
    "stand_still": -1.0,
    "default_pos": -0.1,
    "default_hip_pos": 0.3,
    "feet_contact_forces": -0.01,
    "jump": 2.0,
    "feet_clearance": 0.5,
    "contact_without_command": 1.0,
  }
  assert {name: term.weight for name, term in cfg.rewards.items()} == expected
  assert cfg.rewards["jump"].params["cycle_time"] == 1.5
  assert cfg.rewards["feet_clearance"].params["max_height"] == 0.05
  assert cfg.rewards["base_height"].params["target_height"] == 0.3
  assert cfg.commands["twist"].resampling_time_range == (5.0, 5.0)
  assert cfg.events["friction"].params["num_buckets"] == 256
  assert cfg.events["base_mass"].params["ranges"] == (-1.0, 1.0)
  assert cfg.terminations["base_contact"].params["force_threshold"] == 1.0
  rl = load_rl_cfg("Unitree-Go2-Jump-Flat")
  assert rl.seed == 1
  assert rl.max_iterations == 15_000
  assert rl.num_steps_per_env == 24
  assert rl.algorithm.learning_rate == 1.0e-4
  assert rl.actor.hidden_dims == (512, 256, 128)
  assert rl.critic.hidden_dims == (512, 256, 128)


def test_rear_stand_timing_initial_state_action_and_observations() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  robot = cfg.scene.entities["robot"]
  action = cfg.actions["joint_pos"]
  actor = cfg.observations["actor"].terms["frame"]
  critic = cfg.observations["critic"].terms["frame"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 20.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert robot.init_state.pos == (0.0, 0.0, 0.42)
  assert robot.init_state.joint_pos["FL_hip_joint"] == 0.1
  assert robot.init_state.joint_pos["FR_hip_joint"] == -0.1
  assert robot.init_state.joint_pos["RL_thigh_joint"] == 1.0
  assert action.scale == 0.25
  assert action.delay_min_lag == 0
  assert action.delay_max_lag == 3
  assert action.delay_update_period == 4
  assert actor.func.frame_dim == 45
  assert actor.func.history_length == 1
  assert critic.func.frame_dim == 86
  assert critic.func.history_length == 1
  assert actor.params["add_noise"] is True
  assert len(rear_stand_noise_bounds()[1]) == 45


def test_rear_stand_rewards_commands_events_and_ppo() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  expected = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 2.5,
    "lin_vel_z": 0.2,
    "ang_vel_xy": 0.2,
    "rear_stand_orientation": -1.0,
    "torques": -0.0002,
    "dof_acc": -2.5e-7,
    "base_height": 1.5,
    "rear_stand_feet_on_air": 0.4,
    "collision": -2.0,
    "action_rate": -0.05,
    "default_pos": -0.1,
    "default_hip_pos": -0.1,
    "feet_clearance": 0.4,
    "ang_xz": -0.5,
    "contact": 0.3,
    "symmetric_joints": -0.1,
    "orientation_symmetry": -0.5,
    "feet_height_symmetry": -0.2,
    "rear_stand_feet_height_exp": 5.0,
    "default_pos_reward": 0.5,
    "dof_pos_limits": -2.0,
    "alive": 1.0,
  }
  assert {name: term.weight for name, term in cfg.rewards.items()} == expected
  command = cfg.commands["twist"]
  assert command.resampling_time_range == (10.0, 10.0)
  assert command.heading_command
  assert command.rel_heading_envs == 1.0
  assert command.ranges.lin_vel_x == (-0.2, 0.6)
  assert command.ranges.lin_vel_y == (-0.0, 0.0)
  assert command.ranges.ang_vel_z == (-0.4, 0.4)
  assert command.ranges.heading == (-3.14, 3.14)
  assert cfg.events["push_robot"].interval_range_s == (8.0, 8.0)
  assert cfg.events["reset_robot_joints"].params["scale_range"] == (0.5, 1.5)
  assert cfg.events["friction"].params["ranges"] == (0.2, 1.2)
  assert cfg.events["base_mass"].params["ranges"] == (-1.0, 2.0)
  assert cfg.events["base_com"].params["ranges"][0] == (-0.05, 0.05)
  assert cfg.events["joint_friction"].params["ranges"] == (0.01, 0.1)
  assert cfg.events["joint_damping"].params["ranges"] == (0.0, 0.1)
  assert cfg.events["joint_armature"].params["ranges"] == (0.003, 0.08)
  assert cfg.rewards["collision"].params["sensor_name"] == ("thigh_calf_ground_contact")
  assert {sensor.name for sensor in cfg.scene.sensors} == {
    "feet_ground_contact",
    "thigh_calf_ground_contact",
    "base_ground_contact",
  }
  assert cfg.curriculum == {}
  rl = load_rl_cfg("Unitree-Go2-Rear-Stand-Flat")
  assert rl.seed == 1
  assert rl.max_iterations == 15_000
  assert rl.num_steps_per_env == 24
  assert rl.algorithm.learning_rate == 1.0e-3
  assert rl.algorithm.class_name == (
    "lloco.tasks.go2_skills.rear_stand.mdp.symmetry:SourceSymmetricPPO"
  )
  assert rl.algorithm.symmetry_cfg == {
    "data_augmentation_func": (
      "lloco.tasks.go2_skills.rear_stand.mdp.symmetry:rear_stand_symmetry"
    ),
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  assert rl.actor.hidden_dims == (512, 256, 128)
  assert rl.critic.hidden_dims == (512, 256, 128)


def test_handstand_source_parity_configuration() -> None:
  cfg = load_env_cfg("Unitree-Go2-Handstand-Flat")
  robot = cfg.scene.entities["robot"]
  action = cfg.actions["joint_pos"]
  actor = cfg.observations["actor"].terms["frame"]
  critic = cfg.observations["critic"].terms["frame"]
  assert cfg.scene.num_envs == 4096
  assert cfg.episode_length_s == 20.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert robot.init_state.pos == (0.0, 0.0, 0.42)
  assert robot.init_state.joint_pos["FL_thigh_joint"] == 0.8
  assert robot.init_state.joint_pos["RL_thigh_joint"] == 1.0
  assert action.scale == 0.25
  assert action.delay_min_lag == 0
  assert action.delay_max_lag == 3
  assert actor.func.frame_dim == 48
  assert actor.func.history_length == 1
  assert critic.func.frame_dim == 89
  assert critic.func.history_length == 1
  assert handstand_noise_bounds()[1][:3] == (0.0, 0.0, 0.0)
  assert handstand_noise_bounds()[1][3:] == rear_stand_noise_bounds()[1]

  expected = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 2.5,
    "tracking_lin_vel_zero": -0.2,
    "tracking_ang_vel_zero": -0.2,
    "lin_vel_z": 0.2,
    "ang_vel_xy": 0.2,
    "handstand_orientation": -1.0,
    "torques": -0.0002,
    "dof_acc": -2.5e-7,
    "base_height": 1.0,
    "handstand_feet_on_air": 0.4,
    "collision": -1.0,
    "action_rate": -0.05,
    "default_pos": -0.05,
    "default_hip_pos": -0.1,
    "feet_clearance": 0.4,
    "ang_xz": -0.5,
    "contact": 0.3,
    "feet_air_time": 2.0,
    "symmetric_joints": -0.1,
    "handstand_feet_height_exp": 5.0,
    "default_pos_reward": 0.5,
    "alive": 1.0,
    "termination": -5.0,
    "from_zero_guidance": 1.0,
  }
  assert {name: term.weight for name, term in cfg.rewards.items()} == expected
  assert cfg.rewards["base_height"].params["target_height"] == 0.47
  assert cfg.rewards["from_zero_guidance"].params == {
    "target_steps": 9_600,
    "fade_steps": 4_800,
    "initial_foot_height": 0.022,
    "target_foot_height": 0.67,
    "initial_base_height": 0.30,
    "target_base_height": 0.47,
  }
  command = cfg.commands["twist"]
  assert command.resampling_time_range == (5.0, 5.0)
  assert not command.heading_command
  assert command.rel_heading_envs == 0.0
  assert command.ranges.lin_vel_x == (-0.4, 0.4)
  assert command.ranges.lin_vel_y == (-0.0, 0.0)
  assert command.ranges.ang_vel_z == (-0.4, 0.4)
  assert cfg.events["push_robot"].params == {
    "max_push_vel_xy": 1.0,
    "max_push_ang_vel": 1.0,
  }
  assert cfg.events["joint_friction"].params["low"] == 0.01
  assert cfg.events["joint_friction"].params["high"] == 0.2
  assert cfg.events["joint_damping"].params["low"] == 0.0
  assert cfg.events["joint_damping"].params["high"] == 0.2
  assert cfg.events["joint_armature"].params["ranges"] == (0.005, 0.015)
  assert cfg.events["restitution_label"].params["attribute_name"] == (
    "_handstand_restitution"
  )
  rl = load_rl_cfg("Unitree-Go2-Handstand-Flat")
  assert rl.seed == 1
  assert rl.max_iterations == 15_000
  assert rl.num_steps_per_env == 24
  assert rl.algorithm.learning_rate == 1.0e-3
  assert rl.algorithm.symmetry_cfg is None
