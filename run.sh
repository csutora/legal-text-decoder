#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

echo "============================================================"
echo "[run.sh] Legal Text Decoder - Full Pipeline"
echo "[run.sh] Starting at $(date --iso-8601=seconds)"
echo "============================================================"

echo ""
echo "[run.sh] Step 1: Data Preprocessing"
echo "------------------------------------------------------------"
python src/01-data-preprocessing.py

echo ""
echo "[run.sh] Step 2: Baseline Model"
echo "------------------------------------------------------------"
python src/baseline.py

echo ""
echo "[run.sh] Step 3: Training Deep Learning Model"
echo "------------------------------------------------------------"
python src/02-training.py

echo ""
echo "[run.sh] Step 4: Model Evaluation"
echo "------------------------------------------------------------"
python src/03-evaluation.py

echo ""
echo "[run.sh] Step 5: Inference Examples"
echo "------------------------------------------------------------"
python src/04-inference.py

echo ""
echo "============================================================"
echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
echo "============================================================"
