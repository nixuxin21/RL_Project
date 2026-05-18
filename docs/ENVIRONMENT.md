# Environment

This project is frozen against Python `3.13.2` for the current reproduction
package. The machine-readable constraints are:

- `.python-version`: preferred local interpreter version for pyenv/asdf-style tools.
- `pyproject.toml`: Python version range and direct runtime/training/test dependencies.
- `requirements-lock.txt`: full pinned environment observed during the audit.

Recommended clean setup:

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
make test
./.venv/bin/python -m pytest
make mainline-audit
```

Fresh-clone verification status:

- On 2026-05-18, a temporary clone under `/tmp` with an independent Python
  `3.13.2` virtual environment successfully installed `requirements.txt` and
  passed `make quick-audit`.
- The repository also has `.github/workflows/checks.yml`, which runs
  `make quick-audit PYTHON=python` on push and pull request.

For a lighter resolver-driven setup, install the direct pinned dependencies,
including the standard `pytest` and high-signal `ruff` check entry points:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

`stable-baselines3[extra]` pulls in training and logging dependencies. If PyTorch
wheel resolution differs across CPU/GPU platforms, use the official PyTorch
wheel index for the target platform while keeping the same `torch==2.11.0`
version for reproduction runs.
