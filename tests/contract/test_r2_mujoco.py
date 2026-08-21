"""R2 CPU simulation evidence against the Customized MJLab 1.6 dependency set."""

from __future__ import annotations

from pathlib import Path

import pytest
from robolab_core import run_simulation_smoke

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "robots/firedog2.2.SLDASM/model/firedog2_2.xml"
JOINTS = [f"{leg}{index if index else ''}_joint" for leg in ("RF", "RR", "LR", "LF") for index in range(4)]


def test_firedog_load_reset_named_action_observation() -> None:
    result = run_simulation_smoke(MODEL, JOINTS)
    assert result["load"] is True
    assert result["reset"]["time"] == 0.0
    assert result["actionDimension"] == 16
    assert result["observation"]["beforeDimension"] == 32
    assert result["observation"]["afterDimension"] == 32
    assert result["observation"]["finite"] is True
    assert result["observation"]["valuesChanged"] is True
