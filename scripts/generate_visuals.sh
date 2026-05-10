#!/usr/bin/env bash
set -euo pipefail
python -m src.visualize --data_path dataset/val --weights checkpoints/best.pth --config config.yaml --output results_val --num_samples 10
python -m src.visualize --data_path dataset/test --weights checkpoints/best.pth --config config.yaml --output results_test --num_samples 10
