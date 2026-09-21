#!/usr/bin/env bash
# Run GloriosaAI, setting up the virtual environment first if needed.
set -euo pipefail
cd "$(dirname "$0")"
[ -d psdenv ] || ./setup.sh
exec psdenv/bin/python main.py
