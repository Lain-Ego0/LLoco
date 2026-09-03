"""Explicit task/profile playback workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab.scripts.play import PlayConfig, run_play

from lainloco.experiments import resolve_experiment

PlaybackAgent = Literal["zero", "random", "trained"]
ViewerBackend = Literal["auto", "native", "viser"]


@dataclass(frozen=True, slots=True)
class PlaybackPlan:
  """Validated local playback selection."""

  task_id: str
  profile_id: str
  registry_task_id: str
  agent: PlaybackAgent
  checkpoint: Path | None
  num_envs: int
  device: str | None
  viewer: ViewerBackend


def build_playback_plan(
  *,
  task_id: str,
  profile_id: str,
  agent: PlaybackAgent = "trained",
  checkpoint: str | Path | None = None,
  num_envs: int = 1,
  device: str | None = None,
  viewer: ViewerBackend = "auto",
) -> PlaybackPlan:
  """Resolve a local playback run without starting a viewer."""
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  binding = resolve_experiment(task_id, profile_id)
  checkpoint_path = (
    Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
  )
  if agent == "trained":
    if checkpoint_path is None:
      raise ValueError("trained playback requires --checkpoint")
    if not checkpoint_path.is_file():
      raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
  elif checkpoint_path is not None:
    raise ValueError("--checkpoint is only valid with --agent trained")
  return PlaybackPlan(
    task_id=task_id,
    profile_id=profile_id,
    registry_task_id=binding.registry_task_id,
    agent=agent,
    checkpoint=checkpoint_path,
    num_envs=num_envs,
    device=device,
    viewer=viewer,
  )


def launch_playback(plan: PlaybackPlan) -> None:
  """Launch playback through mjlab's maintained viewer workflow."""
  run_play(
    plan.registry_task_id,
    PlayConfig(
      agent=plan.agent,
      checkpoint_file=str(plan.checkpoint) if plan.checkpoint is not None else None,
      num_envs=plan.num_envs,
      device=plan.device,
      viewer=plan.viewer,
    ),
  )
