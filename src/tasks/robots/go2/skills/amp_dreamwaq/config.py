"""AMP-DreamWaQ environment config layered on source-faithful DreamWaQ."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import mdp as env_mdp
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RecorderTermCfg,
  RewardTermCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.rl import RslRlPpoWithSymmetryAlgorithmCfg

from ..dreamwaq.config import make_dreamwaq_env_cfg, make_dreamwaq_runner_cfg
from ..shared import rl as shared_rl
from . import commands, mdp


@dataclass
class AmpDreamWaQAlgorithmCfg(RslRlPpoWithSymmetryAlgorithmCfg):
  amp_replay_buffer_size: int = 1_000_000
  amp_num_preload_transitions: int = 2_000_000
  amp_reward_coef: float = 0.5
  amp_discr_hidden_dims: tuple[int, ...] = (1024, 512)
  min_normalized_std: float = 0.05


def make_amp_dreamwaq_env_cfg(*, play: bool = False):
  cfg = make_dreamwaq_env_cfg(play=play)
  cfg = deepcopy(cfg)
  # Source AMP differs from DreamWaQ in lateral command range, task rewards,
  # base-mass range and 0.01 joint-position observation noise.
  cfg.commands["twist"] = commands.AmpDreamWaQVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.0,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.6, 0.6),
      ang_vel_z=(-1.0, 1.0),
      heading=(-3.14, 3.14),
    ),
  )
  cfg.events["base_mass"].params["ranges"] = (-1.0, 3.0)
  cfg.rewards["tracking_lin_vel"].weight = 1.0
  cfg.rewards["tracking_ang_vel"].weight = 0.5
  cfg.rewards["lin_vel_z"].weight = -2.0
  cfg.rewards["torques"].weight = -1.0e-5
  cfg.rewards["base_height"].params["target_height"] = 0.35
  cfg.rewards["rear_hip_limit"] = RewardTermCfg(func=mdp.rear_hip_limit, weight=-1.0)
  # In MuJoCo the random AMP policy can terminate after ~15 steps and avoid
  # the source task's net-negative shaping return.  The Gym task assigns zero
  # terminal cost and happens not to fall into this basin under PhysX.  These
  # two backend-adaptation terms remove that early-termination optimum; they
  # do not alter the target pose, commands, observations, or AMP objective.
  # Keep a small backend survival bridge: the terminal cost prevents the
  # early-fall loophole, while a large per-step alive bonus creates a static
  # policy that ignores velocity commands in MuJoCo.
  cfg.rewards["alive"] = RewardTermCfg(func=env_mdp.is_alive, weight=0.1)
  cfg.rewards["termination"] = RewardTermCfg(func=shared_rl.terminal_cost, weight=-5.0)
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={"state": ObservationTermCfg(func=mdp.amp_state)}, enable_corruption=False
  )
  cfg.observations["actor"].terms["frame"].params["joint_position_noise"] = 0.01
  cfg.recorders["amp_terminal_state"] = RecorderTermCfg(
    func=mdp.AmpTerminalStateRecorder
  )
  return cfg


def make_amp_dreamwaq_runner_cfg():
  # The dedicated AMP PPO class is installed by this task, not the source fork.
  cfg = make_dreamwaq_runner_cfg()
  cfg.experiment_name = "go2_amp_dreamwaq"
  cfg.max_iterations = 20_000
  cfg.save_interval = 500
  cfg.algorithm = AmpDreamWaQAlgorithmCfg(**vars(cfg.algorithm))
  cfg.algorithm.class_name = (
    "src.tasks.robots.go2.skills.amp_dreamwaq.rl:AmpDreamWaQPPO"
  )
  return cfg
