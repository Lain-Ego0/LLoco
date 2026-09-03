from typing import Any, Self

class MjSpec:
  @staticmethod
  def from_file(filename: str, assets: dict[str, bytes] | None = None) -> Self: ...
  def compile(self) -> Any: ...

viewer: Any
