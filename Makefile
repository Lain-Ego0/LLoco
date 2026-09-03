.PHONY: sync format lint type test check list

sync:
	uv sync --extra cu128

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check

type:
	uv run pyright

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

check: lint type test

list:
	uv run list-envs --keyword Unitree
