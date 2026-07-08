.PHONY: sync lint format test check

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest -q

check: lint test
	uv run ruff format --check .
	git diff --check
