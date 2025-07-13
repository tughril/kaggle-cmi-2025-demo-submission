# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- `make install` or `uv sync` - Install dependencies and set up virtual environment

### Code Quality
- `make format` - Format code with ruff
- `make lint` - Lint and auto-fix code with ruff
- `make mypy` - Type check with mypy (checks `*.py` files in root)
- `make ty` - Type check with ty (checks `*.py` files in root)

### CLI Testing
- `make hello` - Test CLI with hello command (uses `PYTHONPATH=. uv run ml-cli hello`)

### Notebook Development
- `make sync` - Sync notebook and Python file using jupytext
- Before first sync, run: `uv run jupytext --set-formats ipynb,py:percent src/notebook/main.ipynb`

### Cleanup
- `make clean` - Remove cache directories and virtual environment

## Architecture

This is a machine learning template project using:
- **uv** for Python package management and virtual environment
- **jupytext** for notebook-Python file synchronization (ipynb ↔ py:percent format)
- **ruff** for formatting and linting
- **mypy/ty** for type checking
- **typer** for CLI interface
- **PyTorch ecosystem** with torchvision for ML models
- **Standard ML stack**: numpy, pandas, scikit-learn, matplotlib, seaborn, opencv-python

### Project Structure
- `src/cmd/main.py` - CLI entry point with typer commands (hello, train, predict)
- `src/notebook/` - Jupyter notebook development with jupytext sync
- Entry point: `ml-cli` command defined in pyproject.toml scripts

The main development happens in `src/notebook/` where Jupyter notebooks are kept in sync with Python files using jupytext's percent format. This allows for version control of notebook content while maintaining notebook interactivity.

The CLI provides commands for training models and making predictions, with placeholders for actual ML implementation.