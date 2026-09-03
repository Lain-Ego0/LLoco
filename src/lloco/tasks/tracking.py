"""Motion-tracking task registrations for G1 variants."""

from collections.abc import Callable

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from lloco.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  G1_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
  get_g1_robot_cfg,
)
from lloco.tasks.rl import make_ppo_runner_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def make_g1_tracking_env_cfg(
  robot_cfg: Callable[[], EntityCfg],
  action_scale: dict[str, float],
  *,
  reduced_dof: bool = False,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a G1 tracking task using the mjlab 1.6 tracking MDP."""
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": robot_cfg()}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = action_scale

  hand_name = "wrist_roll_rubber_hand" if reduced_dof else "wrist_yaw_link"
  command = cfg.commands["motion"]
  assert isinstance(command, MotionCommandCfg)
  command.anchor_body_name = "torso_link"
  command.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    f"left_{hand_name}",
    "right_shoulder_roll_link",
    "right_elbow_link",
    f"right_{hand_name}",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    f"left_{hand_name}",
    f"right_{hand_name}",
  )
  cfg.viewer.body_name = "torso_link"

  if not has_state_estimation:
    actor_terms = {
      name: term
      for name, term in cfg.observations["actor"].terms.items()
      if name not in {"motion_anchor_pos_b", "base_lin_vel"}
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    command.pose_range = {}
    command.velocity_range = {}
    command.sampling_mode = "start"
  return cfg


def _register_tracking_tasks(
  name: str,
  robot_cfg: Callable[[], EntityCfg],
  action_scale: dict[str, float],
  *,
  reduced_dof: bool,
) -> None:
  runner_cfg = make_ppo_runner_cfg(
    f"{name.lower().replace('-', '_')}_tracking",
    max_iterations=30_000,
    entropy_coef=0.005,
    save_interval=500,
  )
  for has_state_estimation, suffix in (
    (True, "Tracking"),
    (False, "Tracking-No-State-Estimation"),
  ):
    kwargs = {
      "reduced_dof": reduced_dof,
      "has_state_estimation": has_state_estimation,
    }
    register_mjlab_task(
      task_id=f"Unitree-{name}-{suffix}",
      env_cfg=make_g1_tracking_env_cfg(robot_cfg, action_scale, **kwargs),
      play_env_cfg=make_g1_tracking_env_cfg(
        robot_cfg, action_scale, play=True, **kwargs
      ),
      rl_cfg=runner_cfg,
      runner_cls=MotionTrackingOnPolicyRunner,
    )


_register_tracking_tasks("G1", get_g1_robot_cfg, G1_ACTION_SCALE, reduced_dof=False)
_register_tracking_tasks(
  "G1-23Dof",
  get_g1_23dof_robot_cfg,
  G1_23DOF_ACTION_SCALE,
  reduced_dof=True,
)
