"""Thin CLI adapters that register LLoco tasks before delegating to mjlab."""


def _register_tasks() -> None:
  import lloco.tasks  # noqa: F401


def train() -> None:
  """Train an LLoco or built-in mjlab task."""
  _register_tasks()
  from mjlab.scripts.train import main

  main()


def play() -> None:
  """Evaluate an LLoco or built-in mjlab task."""
  _register_tasks()
  from mjlab.scripts.play import main

  main()


def list_envs() -> None:
  """List all registered LLoco and built-in mjlab tasks."""
  _register_tasks()
  from mjlab.scripts.list_envs import main

  main()


def csv_to_npz() -> None:
  """Convert a G1 motion CSV to a local tracking NPZ."""
  import mjlab
  import tyro

  from lloco.motion_conversion import main

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
