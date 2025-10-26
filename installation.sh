#!/usr/bin/env bash
# installation.sh
# POSIX-friendly installer for Pictura Quantica (Git Bash / WSL / macOS / Linux).
# Usage:
#   ./installation.sh        # create venv, install deps, and run the app
#   ./installation.sh --no-run

set -euo pipefail

NO_RUN=0
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    -n|--no-run)
      NO_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--no-run]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

echo "Pictura Quantica POSIX installer"

# Prefer `python` on PATH; fallback to python3 if available
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Error: python (3.10+) not found on PATH." >&2
  exit 1
fi

# Create venv
echo "Creating virtual environment 'venv'..."
$PY -m venv venv

# Activate venv (POSIX)
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "Error: venv activation script not found at venv/bin/activate" >&2
  exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
$PY -m pip install --upgrade pip

# Install dependencies
packages=(
  numpy
  pillow
  opencv-python
  scipy
  scikit-learn
  qiskit
  qiskit-machine-learning
  matplotlib
  pylatexenc
  seaborn
  joblib
  PyQt6
)

echo "Installing packages into venv..."
$PY -m pip install -U "${packages[@]}"

# Run the app unless requested not to
if [[ $NO_RUN -eq 0 ]]; then
  if [[ -f source/main.py ]]; then
    echo "Launching the application (source/main.py)..."
    (cd source && $PY main.py)
  else
    echo "Warning: source/main.py not found — skipping launch." >&2
  fi
else
  echo "Setup complete. Skipping app launch (--no-run)."
fi

echo "Done."