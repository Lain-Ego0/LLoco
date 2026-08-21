"""Shared fixtures and source-path setup for contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
for src in (
    REPO_ROOT / "packages/schemas/src",
    REPO_ROOT / "packages/core/src",
    REPO_ROOT / "services/api/src",
    REPO_ROOT / "apps/cli/src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_fixture(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def reference_profile() -> dict:
    return load_fixture("robot_profile.reference_biped.yaml")


@pytest.fixture()
def reference_joint_set() -> dict:
    return load_fixture("joint_set.reference_biped.yaml")


@pytest.fixture()
def motion_skill() -> dict:
    return load_fixture("skill.motion.reference_biped.yaml")


@pytest.fixture()
def platform_skill() -> dict:
    return load_fixture("skill.platform.mjcf_inspector.yaml")


@pytest.fixture()
def agent_skill() -> dict:
    return load_fixture("skill.agent.robot_onboarding.yaml")
