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
- [ ] Load and visualize sample image pairs (EO + SAR + Target)
- [ ] Analyze image dimensions, channels, and data types
- [ ] Compute class distribution (change vs no-change ratio)
- [ ] Identify unique disaster events/scenes in dataset
- [ ] Analyze spatial resolution and sensor characteristics

### Phase 2: Literature Survey
- [ ] Review change detection methods in remote sensing
- [ ] Study EO-SAR fusion approaches
- [ ] Explore architectures: Siamese networks, UNet variants, attention mechanisms
- [ ] Research handling of class imbalance (weighted loss, focal loss, etc.)
- [ ] Identify state-of-the-art methods and pretrained backbones

### Phase 3: Data Pipeline Implementation
- [ ] Create dataset class with loading and preprocessing
- [ ] Implement label remapping logic
- [ ] Build data augmentation pipeline (geometric + spectral)
- [ ] Create train/val/test data loaders
- [ ] Handle multi-modal input (EO + SAR concatenation/fusion)

### Phase 4: Model Architecture Design
- [ ] Design baseline: UNet with Siamese encoder
- [ ] Experiment with attention mechanisms (CBAM, self-attention)
- [ ] Try different backbones: ResNet, EfficientNet (pretrained on ImageNet)
- [ ] Implement skip connections and feature fusion strategies

### Phase 5: Training Strategy
- [ ] Select loss function (handle class imbalance: Focal Loss, Dice Loss, Tversky Loss)
- [ ] Configure optimizer (Adam/AdamW with learning rate scheduling)
- [ ] Set up training loop with validation monitoring
- [ ] Implement early stopping and model checkpointing
- [ ] Track metrics: IoU, Precision, Recall, F1

### Phase 6: Evaluation & Analysis
- [ ] Compute all metrics on validation set
- [ ] Compute all metrics on test set
- [ ] Generate confusion matrix
- [ ] Create qualitative visualizations (success + failure cases)
- [ ] Analyze error patterns and failure modes

### Phase 7: Report Writing
- [ ] Write Abstract
- [ ] Document Literature Survey
- [ ] Detail Methodology with rationale
- [ ] Present Results with tables and figures
- [ ] Propose Future Work
- [ ] Write Conclusion
- [ ] Include Time and Resource Log

### Phase 8: Submission Preparation
- [ ] Create config.yaml with all hyperparameters
- [ ] Prepare requirements.txt
- [ ] Write comprehensive README.md
- [ ] Train final model and upload checkpoint (Google Drive/HuggingFace)
- [ ] Create ZIP file: checkpoint + PDF report + time log
- [ ] Push code to public GitHub repository

---

## Current Baseline

- Model: `DualEncoderEOSARUNet` with separate ResNet34 encoders for EO and SAR
- Loss: `FocalDiceLoss`
- Output: 2-channel logits for binary segmentation with change-class metrics
- Input pipeline: EO RGB + SAR grayscale, resized to `256x256`, with mandatory label remapping

## Quick Start Commands

```bash
# Setup environment
conda create -n eosar_cd python=3.10
conda activate eosar_cd
pip install -r requirements.txt

# Train model
python -m src.train --config config.yaml

# Evaluate on validation or test set
python -m src.eval --data_path dataset/val --weights checkpoints/best.pth --config config.yaml

# Visualize predictions
python -m src.visualize --data_path dataset/val --weights checkpoints/best.pth --output results --config config.yaml
```

## Expected Deliverables

1. **GitHub Repository**: Public repo with all source code, config, README
2. **Model Weights**: Public download link in README (Google Drive/HuggingFace)
3. **Technical Report (PDF)**: Comprehensive report with all sections
4. **ZIP File**: FirstName_LastName_GalaxEye.zip containing weights + report + logs

---

## Notes

- Use only provided dataset (no external remote sensing data)
- Pretrained backbones (ImageNet) are permitted
- LLM/coding assistants allowed for implementation
- Document all design decisions with rationale in report
