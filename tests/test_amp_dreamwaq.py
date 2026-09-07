"""Focused regression tests for the AMP-DreamWaQ data path."""

from typing import Any, cast

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.amp_dreamwaq.motion import Go2AmpMotionLoader
from lloco.tasks.go2_skills.amp_dreamwaq.rl import (
  _AmpReplayBuffer,
  _RunningMeanStd,
)


def test_amp_source_configuration() -> None:
  env = load_env_cfg("Unitree-Go2-AMP-DreamWaQ-Rough")
  runner = cast(Any, load_rl_cfg("Unitree-Go2-AMP-DreamWaQ-Rough"))
  command = cast(Any, env.commands["twist"])
  assert env.observations["actor"].terms["frame"].params["joint_position_noise"] == .01
  assert "amp_terminal_state" in env.recorders
  assert env.rewards["alive"].weight == 1.0
  assert env.rewards["termination"].weight == -5.0
  assert command.ranges.lin_vel_y == (-.6, .6)
  assert runner.algorithm.amp_replay_buffer_size == 1_000_000
  assert runner.algorithm.amp_num_preload_transitions == 2_000_000
  assert runner.algorithm.amp_discr_hidden_dims == (1024, 512)
  assert runner.algorithm.min_normalized_std == .05


def test_expert_state_uses_policy_dt_interpolation() -> None:
  frames = torch.arange(4, dtype=torch.float32).view(4, 1).repeat(1, 49)
  state = Go2AmpMotionLoader._state_at_time(
    frames, torch.tensor([.02]), trajectory_length=.12
  )
  next_state = Go2AmpMotionLoader._state_at_time(
    frames, torch.tensor([.04]), trajectory_length=.12
  )
  # Source phase is time / trajectory_length * num_frames.  A 0.02 s
  # transition therefore advances 2/3 of a 0.04 s data frame, not one frame.
  assert torch.allclose(next_state - state, torch.full((1, 31), 2 / 3))


def test_amp_replay_wraps_and_retains_previous_iterations() -> None:
  replay = _AmpReplayBuffer(4, torch.device("cpu"))
  replay.insert(torch.zeros((2, 31)), torch.ones((2, 31)))
  replay.insert(torch.full((3, 31), 2.0), torch.full((3, 31), 3.0))
  assert replay.size == 4
  assert replay.position == 1
  assert (replay.states == 0.0).all(1).sum() == 1
  assert (replay.states == 2.0).all(1).sum() == 3


def test_amp_running_statistics_normalize_and_clip() -> None:
  normalizer = _RunningMeanStd(2)
  normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 5.0]]))
  normalized = normalizer.normalize(torch.tensor([[2.0, 4.0], [1e6, -1e6]]))
  # The source starts RunningMeanStd with count=1e-4, so its first mean is
  # intentionally a few 1e-4 away from the exact batch mean.
  assert torch.allclose(normalized[0], torch.zeros(2), atol=3e-4)
  assert torch.equal(normalized[1], torch.tensor([10.0, -10.0]))
