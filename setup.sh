#!/usr/bin/env bash
# Create the virtual environment (psdenv) and install dependencies.
set -euo pipefail
cd "$(dirname "$0")"
[ -d psdenv ] || "${PYTHON:-python3}" -m venv psdenv
psdenv/bin/python -m pip install --upgrade pip
psdenv/bin/python -m pip install -r requirements.txt
