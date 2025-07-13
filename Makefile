.PHONY: install
install:
	uv sync

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache .coverage

# require: uv run jupytext --set-formats ipynb,py:percent src/notebook/main.ipynb
.PHONY: sync
sync:
	uv run jupytext --sync src/notebook/main.py

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check --fix

.PHONY: mypy
mypy:
	uv run mypy *.py

.PHONY: ty
ty:
	uv run ty check *.py

.PHONY: hello
hello:
	PYTHONPATH=. uv run ml-cli hello

