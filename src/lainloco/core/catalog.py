"""Typed, duplicate-rejecting catalogs for LainLoco domain objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from types import MappingProxyType
from typing import Generic, TypeVar

T = TypeVar("T")


class Catalog(Generic[T]):
  """An immutable catalog keyed by an explicit ID getter."""

  def __init__(self, values: Iterable[T], id_of: Callable[[T], str]) -> None:
    items: dict[str, T] = {}
    for value in values:
      item_id = id_of(value)
      if item_id in items:
        raise ValueError(f"Duplicate catalog ID: {item_id}")
      items[item_id] = value
    self._items = MappingProxyType(items)

  def __contains__(self, item_id: object) -> bool:
    return item_id in self._items

  def __iter__(self) -> Iterator[str]:
    return iter(self._items)

  def __len__(self) -> int:
    return len(self._items)

  def get(self, item_id: str) -> T:
    try:
      return self._items[item_id]
    except KeyError as exc:
      choices = ", ".join(sorted(self._items))
      raise KeyError(
        f"Unknown catalog ID {item_id!r}; choose one of: {choices}"
      ) from exc

  def ids(self) -> tuple[str, ...]:
    return tuple(sorted(self._items))

  def values(self) -> tuple[T, ...]:
    return tuple(self._items[item_id] for item_id in self.ids())
