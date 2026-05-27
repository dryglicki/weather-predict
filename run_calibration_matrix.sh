#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

mkdir -p logs/calibration_matrix

runs=(
  "no_interval_no_pit:--no-interval-calibration --no-pit-calibration"
  "interval_only:--interval-calibration --no-pit-calibration"
  "pit_only:--no-interval-calibration --pit-calibration"
  "interval_and_pit:--interval-calibration --pit-calibration"
)

for run_spec in "${runs[@]}"; do
  run_name="${run_spec%%:*}"
  run_args="${run_spec#*:}"
  log_path="logs/calibration_matrix/${run_name}.log"

  echo "Running ${run_name}: python predictor.py ${run_args}"
  python predictor.py ${run_args} 2>&1 | tee "${log_path}"
done
