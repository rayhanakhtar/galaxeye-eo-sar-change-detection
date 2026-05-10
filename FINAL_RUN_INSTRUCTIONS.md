# Final Run Instructions

These steps should be executed on a GPU-capable machine after copying the repository and dataset.

## 1. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the Full Model

```bash
python -m src.train --config config.yaml
```

Expected output:
- `checkpoints/best.pth`
- `logs/training.log`

## 3. Evaluate Validation Split

```bash
python -m src.eval --data_path dataset/val --weights checkpoints/best.pth --config config.yaml --output_dir results_val
```

Expected output:
- `results_val/evaluation_metrics.json`

## 4. Evaluate Test Split

```bash
python -m src.eval --data_path dataset/test --weights checkpoints/best.pth --config config.yaml --output_dir results_test
```

Expected output:
- `results_test/evaluation_metrics.json`

## 5. Generate Qualitative Visualizations

```bash
python -m src.visualize --data_path dataset/val --weights checkpoints/best.pth --config config.yaml --output results_val --num_samples 10
python -m src.visualize --data_path dataset/test --weights checkpoints/best.pth --config config.yaml --output results_test --num_samples 10
```

Expected output:
- `results_val/predictions.png`
- `results_test/predictions.png`

## 6. Update Report Assets

After the full run, replace the debug metrics in:
- `reports/technical_report_draft.md`
- `reports/metrics_table_debug.md`
- `results_debug_summary.md`

with final numbers from:
- `results_val/evaluation_metrics.json`
- `results_test/evaluation_metrics.json`

## 7. Prepare Submission

- Export the report draft to PDF
- Upload `checkpoints/best.pth` to a public link
- Add the link to `README.md`
- Assemble the final ZIP contents listed in `SUBMISSION_CHECKLIST.md`
