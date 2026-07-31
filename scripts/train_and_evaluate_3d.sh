#!/usr/bin/env bash
set -euo pipefail

exec python scripts/train_and_evaluate_3d.py "$@"
