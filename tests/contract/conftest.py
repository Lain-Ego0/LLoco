"""Shared fixtures and source-path setup for contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SKILL_CATALOG = REPO_ROOT.parent / "RoboLab-Skill"

for src in (
    REPO_ROOT / "packages/schemas/src",
    REPO_ROOT / "packages/core/src",
    REPO_ROOT / "packages/mjlab_adapter/src",
    REPO_ROOT / "services/api/src",
    REPO_ROOT / "apps/cli/src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_fixture(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def g1_profile() -> dict:
    return load_fixture("robot_profile.g1.yaml")


@pytest.fixture()
def g1_joint_set() -> dict:
    return load_fixture("joint_set.g1_29dof.yaml")


@pytest.fixture()
def motion_skill() -> dict:
    return load_fixture("skill.motion.g1_velocity.yaml")


@pytest.fixture()
def platform_skill() -> dict:
    return load_fixture("skill.platform.mjcf_inspector.yaml")


@pytest.fixture()
def agent_skill() -> dict:
    return load_fixture("skill.agent.robot_onboarding.yaml")


@pytest.fixture()
def g1_velocity_package_dir() -> Path:
    """Return the real g1_velocity package directory or skip when absent."""
    package = SKILL_CATALOG / "skills/motion/g1_velocity"
    if not package.is_dir():
        pytest.skip("同级 RoboLab-Skill checkout 不存在（D-029 开发发现路径）")
    return package
