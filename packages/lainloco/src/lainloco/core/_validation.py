"""Shared validation helpers for dependency-light domain specifications."""


def require_text(value: str, field_name: str) -> None:
  if not value or value.strip() != value:
    raise ValueError(f"{field_name} must be non-empty and have no outer whitespace")
