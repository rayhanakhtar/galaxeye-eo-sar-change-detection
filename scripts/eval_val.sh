#!/usr/bin/env bash
set -euo pipefail
python -m src.eval --data_path dataset/val --weights checkpoints/best.pth --config config.yaml --output_dir results_val
