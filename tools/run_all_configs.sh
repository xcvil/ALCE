#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_all_configs.sh [configs_dir] [-- extra args forwarded to run.py]
#
# Examples:
#   bash tools/run_all_configs.sh
#   bash tools/run_all_configs.sh configs/new
#   bash tools/run_all_configs.sh configs/new -- --quick_test 10

CONFIG_DIR="${1:-configs/asqa}"
shift || true

# If user provided `--`, drop it and forward the rest to run.py
if [[ "${1:-}" == "--" ]]; then
  shift
fi

shopt -s nullglob
configs=( "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml )

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "[ERROR] No config files found in: $CONFIG_DIR" >&2
  exit 2
fi

for cfg in "${configs[@]}"; do
  echo "========== Running: $cfg =========="
  python run.py --config "$cfg" "$@"
done


