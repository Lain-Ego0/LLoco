"""Regression tests for LLoco-owned robot descriptions."""

from collections.abc import Callable

import pytest

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
from mjlab.entity import Entity, EntityCfg


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
