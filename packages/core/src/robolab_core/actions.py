"""One small, shared action registry for CLI, API and Agent adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Action:
    id: str
    description: str
    handler: Callable[[Mapping[str, Any]], Any]
    input_schema: Mapping[str, Any]


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if action.id in self._actions:
            raise ValueError(f"Action 已注册: {action.id}")
        self._actions[action.id] = action

    def describe(self) -> list[dict[str, Any]]:
        return [{"id": action.id, "description": action.description, "inputSchema": dict(action.input_schema)} for action in self._actions.values()]

    def invoke(self, action_id: str, parameters: Mapping[str, Any]) -> Any:
        try:
            action = self._actions[action_id]
        except KeyError as error:
            raise KeyError(f"未知 Action: {action_id}") from error
        return action.handler(parameters)
