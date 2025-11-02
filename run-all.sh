#!/usr/bin/env bash
# A small runner for simulations. This version avoids creating a virtualenv
# by default (use SKIP_PIP=1 to skip installing requirements). Outputs are
# logged to logs/ with timestamps.

set -euo pipefail
IFS=$'\n\t'

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "Starting run-all.sh at $(date -u)"

echo "Skipping pip install step (removed per user request)." | tee "$LOG_DIR/setup_$TS.log"

echo "Running baseline simulation (ideal + offset)..."
python3 bpsk_sim.py > "$LOG_DIR/bpsk_$TS.log" 2>&1 || { echo "bpsk_sim failed; see $LOG_DIR/bpsk_$TS.log"; exit 1; }

echo "Running Gardner TED simulation..."
python3 ted_sim.py > "$LOG_DIR/ted_$TS.log" 2>&1 || { echo "ted_sim failed; see $LOG_DIR/ted_$TS.log"; exit 1; }

echo "✅ All simulations complete."
echo "Logs saved to: $LOG_DIR/"
echo "Check plots/ for: eye_clean.png, eye_offset.png, eye_ted.png, ber_curve.png, spectrum.png, ted_error.png"