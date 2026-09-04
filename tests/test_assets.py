"""Regression tests for LLoco-owned robot descriptions."""

from collections.abc import Callable
from pathlib import Path

import pytest
from mjlab.entity import Entity, EntityCfg

from lloco.assets.robots import (
  get_a2_robot_cfg,
  get_as2_robot_cfg,
  get_g1_23dof_robot_cfg,
  get_g1_robot_cfg,
  get_go2_robot_cfg,
  get_h1_2_robot_cfg,
  get_h2_robot_cfg,
  get_r1_robot_cfg,
)
from lloco.motion_conversion import (
  G1_23DOF_JOINT_NAMES,
  G1_JOINT_NAMES,
  resolve_output_path,
)


@pytest.mark.parametrize(
  "robot_cfg",
  (
    get_a2_robot_cfg,
    get_as2_robot_cfg,
    get_g1_robot_cfg,
    get_g1_23dof_robot_cfg,
    get_go2_robot_cfg,
    get_h1_2_robot_cfg,
    get_h2_robot_cfg,
    get_r1_robot_cfg,
  ),
)
def test_robot_model_compiles(robot_cfg: Callable[[], EntityCfg]) -> None:
  model = Entity(robot_cfg()).spec.compile()
  assert model.nbody > 1
  assert model.njnt > 1


def test_motion_conversion_joint_counts() -> None:
  assert len(G1_JOINT_NAMES) == 29
  assert len(G1_23DOF_JOINT_NAMES) == 23


def test_motion_output_path_defaults_to_input_directory(tmp_path) -> None:
  input_file = tmp_path / "motions" / "dance.csv"
  assert resolve_output_path(input_file, "converted") == (
    input_file.parent / "converted.npz"
  )


def test_motion_output_path_accepts_explicit_path(tmp_path) -> None:
  input_file = tmp_path / "motions" / "dance.csv"
  output_file = tmp_path / "exports" / "dance.npz"
  assert resolve_output_path(input_file, output_file) == output_file


def test_motion_output_path_appends_npz_without_replacing_suffix(tmp_path) -> None:
  input_file = tmp_path / "motions" / "dance.csv"
  output_file = tmp_path / "exports" / "dance.v1"
  assert resolve_output_path(input_file, output_file) == Path(f"{output_file}.npz")
