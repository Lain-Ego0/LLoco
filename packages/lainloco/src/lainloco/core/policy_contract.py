"""Versioned policy interface shared by training and deployment."""

from dataclasses import dataclass

from ._validation import require_text


@dataclass(frozen=True, slots=True)
class ObservationField:
  """One ordered field in a policy observation contract."""

  name: str
  width: int

  def __post_init__(self) -> None:
    require_text(self.name, "observation field name")
    if self.width <= 0:
      raise ValueError("observation field width must be positive")


@dataclass(frozen=True, slots=True)
class RecurrentStateSpec:
  """Shape of a recurrent policy state, excluding batch size."""

  layers: int
  hidden_width: int

  def __post_init__(self) -> None:
    if self.layers <= 0 or self.hidden_width <= 0:
      raise ValueError("recurrent state dimensions must be positive")


@dataclass(frozen=True, slots=True)
class PolicyContract:
  """Versioned interface shared by training, export, and deployment."""

  contract_version: str
  robot_id: str
  task_id: str
  joint_order: tuple[str, ...]
  action_dim: int
  action_scale: tuple[float, ...]
  observation_fields: tuple[ObservationField, ...]
  history_length: int
  history_order: str
  history_reset: str
  normalization: str
  recurrent_state: RecurrentStateSpec | None
  control_dt: float
  conditional_fields: tuple[ObservationField, ...] = ()

  def __post_init__(self) -> None:
    require_text(self.contract_version, "contract_version")
    if self.action_dim != len(self.joint_order):
      raise ValueError("action_dim does not match joint_order")
    if self.action_dim != len(self.action_scale):
      raise ValueError("action_dim does not match action_scale")
    if self.history_length < 0:
      raise ValueError("history_length cannot be negative")
    if self.control_dt <= 0:
      raise ValueError("control_dt must be positive")
    field_names = tuple(field.name for field in self.observation_fields)
    conditional_names = tuple(field.name for field in self.conditional_fields)
    if len(set(field_names)) != len(field_names):
      raise ValueError("observation field names must be unique")
    if len(set(conditional_names)) != len(conditional_names):
      raise ValueError("conditional field names must be unique")
    if self.recurrent_state is not None and self.conditional_fields:
      raise ValueError("recurrent contracts cannot also declare conditional fields")

  @property
  def observation_dim(self) -> int:
    return sum(field.width for field in self.observation_fields)

  @property
  def conditional_dim(self) -> int:
    return sum(field.width for field in self.conditional_fields)
