# Binary Change Detection on EO-SAR Image Pairs

## Project Overview

This project addresses **binary change detection** on paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery. The goal is to classify each pixel as changed (1) or unchanged (0) between pre-event and post-event image pairs.

## Task Summary

- **Input**: Co-registered pre-event and post-event image pairs (EO + SAR)
- **Output**: Binary pixel-level change mask (0 = No-Change, 1 = Change)
- **Dataset**: 
  - Train: 2,781 samples
  - Validation: 334 samples
  - Test: 77 samples (50% of full test set)
- **Label Remapping** (Required):
  - Background (0) → No-Change (0)
  - Intact (1) → No-Change (0)
  - Damaged (2) → Change (1)
  - Destroyed (3) → Change (1)

## Data Structure

```
dataset/
├── train/
│   ├── pre-event/     # 2781 TIF images
│   ├── post-event/   # 2781 TIF images
│   └── target/       # 2781 TIF masks
├── val/
│   ├── pre-event/    # 334 TIF images
│   ├── post-event/   # 334 TIF images
│   └── target/       # 334 TIF masks
└── test/
    ├── pre-event/    # 77 TIF images
    ├── post-event/   # 77 TIF images
    └── target/       # 77 TIF masks
```

## Key Challenges

1. **Multi-modal Fusion**: EO pre-event images are RGB while SAR post-event images are grayscale and physically different modalities
2. **Class Imbalance**: Change pixels are a severe minority class in all splits
3. **Generalization**: The model must work across multiple disaster scenes with limited labeled data

## Evaluation Metrics

- Intersection over Union (IoU) for change class
- Precision (change class)
- Recall (change class)
- F1 Score (change class)
- Confusion Matrix

---

## Roadmap

### Phase 1: Data Exploration & Analysis
- [x] Load and visualize sample image pairs (EO + SAR + Target)
- [x] Analyze image dimensions, channels, and data types
- [x] Compute class distribution (change vs no-change ratio)
- [x] Identify unique disaster events/scenes in dataset
- [x] Analyze spatial resolution and sensor characteristics

### Phase 2: Literature Survey
- [x] Review change detection methods in remote sensing
- [x] Study EO-SAR fusion approaches
- [x] Explore architectures: Siamese networks, UNet variants, attention mechanisms
- [x] Research handling of class imbalance (weighted loss, focal loss, etc.)
- [x] Identify state-of-the-art methods and pretrained backbones

### Phase 3: Data Pipeline Implementation
- [x] Create dataset class with loading and preprocessing
- [x] Implement label remapping logic
- [x] Build data augmentation pipeline (geometric + spectral)
- [x] Create train/val/test data loaders
- [x] Handle multi-modal input (EO + SAR concatenation/fusion)

### Phase 4: Model Architecture Design
- [x] Design baseline: UNet with dual-encoder EO-SAR backbone
- [x] Implement attention-gated decoder fusion
- [ ] Try different backbones: ResNet, EfficientNet (pretrained on ImageNet)
- [x] Implement skip connections and feature fusion strategies

### Phase 5: Training Strategy
- [x] Select loss function (Focal + Dice)
- [x] Configure optimizer (AdamW with cosine scheduling)
- [x] Set up training loop with validation monitoring
- [x] Implement early stopping and model checkpointing
- [x] Track metrics: IoU, Precision, Recall, F1

### Phase 6: Evaluation & Analysis
- [x] Compute all metrics on validation set (debug checkpoint)
- [x] Compute all metrics on test set (debug checkpoint)
- [x] Generate confusion matrix (debug checkpoint)
- [x] Create qualitative visualizations (debug checkpoint)
- [x] Analyze error patterns and failure modes (debug checkpoint)

### Phase 7: Report Writing
- [x] Write Abstract (draft)
- [x] Document Literature Survey (draft)
- [x] Detail Methodology with rationale (draft)
- [x] Present Results with tables and figures (debug draft)
- [ ] Propose Future Work
- [x] Write Conclusion (draft)
- [x] Include Time and Resource Log

### Phase 8: Submission Preparation
- [x] Create config.yaml with all hyperparameters
- [x] Prepare requirements.txt
- [x] Write comprehensive README.md
- [ ] Train final model and upload checkpoint (Google Drive/HuggingFace)
- [ ] Create ZIP file: checkpoint + PDF report + time log
- [ ] Push code to public GitHub repository

---

## Current Baseline

- Model: `DualEncoderEOSARUNet` with separate ResNet34 encoders for EO and SAR
- Loss: `FocalDiceLoss`
- Output: 2-channel logits for binary segmentation with change-class metrics
- Input pipeline: EO RGB + SAR grayscale, resized to `256x256`, with mandatory label remapping

## Current Status

- Core implementation, full training, evaluation, visualization, and report generation are complete
- Final checkpoint: `checkpoints/best.pth`
- Final validation outputs: `results_val_final/`
- Final test outputs: `results_test_final/`
- Final submission report sources: `reports/final_technical_report.md`, `reports/final_technical_report.pdf`
- Debug checkpoint and verification outputs are still available in `checkpoints_debug/`, `results_debug_val/`, and `results_debug_test/`

## Quick Start Commands

```bash
# Setup environment
conda create -n eosar_cd python=3.10
conda activate eosar_cd
pip install -r requirements.txt

# Train model
python -m src.train --config config.yaml

# Run a short debug verification
python -m src.train --config debug_config.yaml

# Evaluate on validation or test set
python -m src.eval --data_path dataset/val --weights checkpoints/best.pth --config config.yaml --output_dir results_val_final
python -m src.eval --data_path dataset/test --weights checkpoints/best.pth --config config.yaml --output_dir results_test_final

# Evaluate the debug checkpoint
python -m src.eval --data_path dataset/val --weights checkpoints_debug/best.pth --config debug_config.yaml --output_dir results_debug_val
python -m src.eval --data_path dataset/test --weights checkpoints_debug/best.pth --config debug_config.yaml --output_dir results_debug_test

# Visualize predictions
python -m src.visualize --data_path dataset/val --weights checkpoints/best.pth --output results_val_final --config config.yaml
python -m src.visualize --data_path dataset/test --weights checkpoints/best.pth --output results_test_final --config config.yaml

# Lint and format checks
python -m ruff check src
python -m black --check src

# Run lightweight tests
pytest tests -q
```

## Helper Scripts

- Windows batch files: `scripts/*.bat`
- Shell scripts: `scripts/*.sh`
- Final GPU handoff steps: `FINAL_RUN_INSTRUCTIONS.md`
- Submission tracker: `SUBMISSION_CHECKLIST.md`

## Expected Deliverables

1. **GitHub Repository**: Public repo with all source code, config, README
2. **Model Weights**: Public download link in README (Google Drive/HuggingFace)
3. **Technical Report (PDF)**: Comprehensive report with all sections
4. **ZIP File**: FirstName_LastName_GalaxEye.zip containing weights + report + logs

## Final Artifacts Available Locally

- Final checkpoint: `checkpoints/best.pth`
- Validation metrics: `results_val_final/evaluation_metrics.json`
- Test metrics: `results_test_final/evaluation_metrics.json`
- Validation visualizations: `results_val_final/predictions.png`
- Test visualizations: `results_test_final/predictions.png`
- Final report markdown: `reports/final_technical_report.md`
- Final report PDF: `reports/final_technical_report.pdf`

## Pending External Submission Items

- Paste the public submission ZIP link in the submission form.
- Public GitHub repository link: `https://github.com/rayhanakhtar/galaxeye-eo-sar-change-detection`

## Model Weights

- Final checkpoint included locally at `checkpoints/best.pth`

## Public Repository

- GitHub repository: `https://github.com/rayhanakhtar/galaxeye-eo-sar-change-detection`

---

## Notes

- Use only provided dataset (no external remote sensing data)
- Pretrained backbones (ImageNet) are permitted
- LLM/coding assistants allowed for implementation
- Document all design decisions with rationale in report
