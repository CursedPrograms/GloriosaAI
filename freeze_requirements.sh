#!/usr/bin/env bash
# Pin the exact versions of the current environment into requirements-lock.txt.
set -euo pipefail
cd "$(dirname "$0")"
[ -d psdenv ] && source psdenv/bin/activate
pip freeze > requirements-lock.txt
echo "requirements-lock.txt written."
