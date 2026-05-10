@echo off
python -m src.eval --data_path dataset/test --weights checkpoints/best.pth --config config.yaml --output_dir results_test
