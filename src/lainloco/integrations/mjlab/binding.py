"""Runtime binding between a LainLoco experiment and mjlab factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lainloco.core import ExperimentSpec

ConfigFactory = Callable[..., Any]
MetadataValue = list | str | float
MetadataFactory = Callable[..., Mapping[str, MetadataValue]]


@dataclass(frozen=True, slots=True)
class MjlabExperimentBinding:
  """An ExperimentSpec plus the factories needed by the mjlab adapter."""

  binding_id: str
  experiment: ExperimentSpec
  registry_task_id: str
  canonical_task_id: str | None
  env_factory: ConfigFactory
  rl_factory: ConfigFactory
  runner_cls: type
  distillation: bool = False
  metadata_factory: MetadataFactory | None = None

  @property
  def registered_ids(self) -> tuple[str, ...]:
    if self.canonical_task_id is None:
      return (self.registry_task_id,)
    return (self.canonical_task_id, self.registry_task_id)

  @property
  def legacy_task_id(self) -> str:
    """Compatibility name for the underlying mjlab registry task."""
    return self.registry_task_id

  def deployment_metadata(
    self, actor: object, *, recurrent: bool
  ) -> dict[str, MetadataValue]:
    if self.metadata_factory is None:
      return {}
    return dict(self.metadata_factory(actor, recurrent=recurrent))
