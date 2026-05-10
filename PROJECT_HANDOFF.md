# Project Handoff Documentation

**From:** MiniMax M2.5 Free (AI Agent)  
**To:** GPT-5.5 (AI Agent)  
**Project:** EO-SAR Binary Change Detection for GalaxEye Technical Assignment  
**Date:** Created during development

---

## 1. PROJECT OVERVIEW

### Task Description
Binary change detection on paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery. The goal is to classify each pixel as changed (1) or unchanged (0) between pre-event and post-event image pairs.

### Key Context
- **Position:** Satellite AI Research Intern assignment
- **Deadline:** 13 May 2026, 11:59 PM IST
- **Submission:** GitHub repository + ZIP file with model weights + PDF report
- **Constraints:** Use only provided dataset (no external data), pretrained backbones allowed

---

## 2. DATASET STRUCTURE

### Data Location
```
dataset/
├── train/
│   ├── pre-event/     # 2781 TIF images (RGB, 1024x1024x3)
│   ├── post-event/   # 2781 TIF images (Grayscale, 1024x1024)
│   └── target/       # 2781 TIF masks (Binary: 0 or 1)
├── val/
│   ├── pre-event/    # 334 images
│   ├── post-event/   # 334 images
│   └── target/       # 334 masks
└── test/
    ├── pre-event/    # 77 images (50% of full test set)
    ├── post-event/   # 77 images
    └── target/       # 77 masks
```

### Key Findings from Data Exploration (Phase 1)

1. **Image Properties:**
   - Pre-event (EO): RGB 3-channel, 1024×1024, uint8, range [0-218]
   - Post-event (SAR): Grayscale 1-channel, 1024×1024, uint8, range [0-255]
   - Target: Binary (already remapped to 0/1)

2. **Class Distribution (SEVERE IMBALANCE):**
   - Train: No-Change 98.43%, Change 1.57% (62.71:1 ratio)
   - Val: No-Change 97.80%, Change 2.20% (44.45:1 ratio)
   - Test: No-Change 99.25%, Change 0.75% (131.54:1 ratio)

3. **Scene Distribution:**
   - 8 unique disaster scenes (scene_01 to scene_08)
   - Scene 06: 836 samples, Scene 08: 1001 samples (largest)
   - Scene 04-05: ~60 samples each (smallest)

4. **Label Remapping (MANDATORY):**
   - Original: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
   - Remapped: 0=No-Change (0,1), 1=Change (2,3)

---

## 3. ARCHITECTURE DESIGN

### Proposed Model: SiameseUNet (Dual-Modal)

**Architecture:**
```
SiameseUNet
├── EO Encoder (Pre-event): Pretrained ResNet34, 3 channels → 512 features
├── SAR Encoder (Post-event): Pretrained ResNet34, 1 channel → 512 features
├── Fusion: Concatenate features at each scale (C64, C128, C256, C512)
├── Decoder: 
│   ├── Decoder4: 1024 → 512 → 256
│   ├── Decoder3: 512 → 256 → 128
│   ├── Decoder2: 256 → 128 → 64
│   └── Decoder1: 64 → 64 → 32
├── Attention Gates in each decoder block
└── Output: Binary change mask (2 classes)
```

### Key Components

1. **AttentionGate Module:**
   - Gating signal from decoder
   - Skip connection from encoder
   - Sigmoid attention weighting
   - Refines features to highlight changed regions

2. **ConvBlock:**
   - Conv2d + BatchNorm + ReLU

3. **DecoderBlock:**
   - Upsample (ConvTranspose2d)
   - Attention gate
   - Concatenation + Conv block

### Design Rationale (from Literature Survey)
- Siamese architecture is standard for change detection (SNUNet-CD, MRA-SNet)
- Separate encoders for EO/SAR handle different modalities
- Pretrained ResNet34 enables transfer learning with limited data (2781 samples)
- Attention gates suppress irrelevant features, highlight changes
- UNet-style decoder preserves spatial detail

---

## 4. DATA PIPELINE

### ChangeDetectionDataset Class (src/data/dataset.py)

**Features:**
- Loads TIF images using tifffile
- Resizes to 256×256 (configurable)
- Label remapping (0,1→0, 2,3→1)
- Channel-wise normalization:
  - EO: ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
  - SAR: mean [0.5], std [0.25]
- Returns dict: {"pre_img", "post_img", "target", "filename"}

---

## 5. LOSS FUNCTIONS

### FocalDiceLoss (src/utils/losses.py)

**Purpose:** Handle severe class imbalance (62:1 to 131:1)

**Components:**
1. **Focal Loss (γ=2.0):** Down-weights well-classified, focuses on hard examples
2. **Dice Loss:** Directly optimizes for IoU
3. **Combined:** 0.5 × Focal + 0.5 × Dice

---

## 6. TRAINING CONFIGURATION (config.yaml)

```yaml
data:
  dataset_path: "dataset"
  image_size: 256
  train_split: "train"
  val_split: "val"
  test_split: "test"

model:
  name: "SiameseUNet"
  backbone: "resnet34"
  pretrained: true
  num_classes: 2
  eo_channels: 3
  sar_channels: 1

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping:
    patience: 10
    min_delta: 0.001
  loss:
    type: "focal_dice"
    focal_weight: 0.5
    dice_weight: 0.5
    focal_gamma: 2.0

checkpoint:
  save_dir: "checkpoints"
  save_best: true
  monitor: "val_iou"
  mode: "max"

seed: 42
device: "cuda"
```

---

## 7. CODEBASE STRUCTURE

```
project/
├── AGENTS.md                    # Guidelines for AI agents
├── README.md                    # Project overview & roadmap
├── config.yaml                  # Hyperparameters
├── requirements.txt             # Dependencies
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Phase 1: Data analysis
│   └── 02_literature_survey.ipynb # Phase 2: Literature & architecture
├── src/
│   ├── __init__.py
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   ├── visualize.py              # Visualization script
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # ChangeDetectionDataset
│   ├── models/
│   │   ├── __init__.py
│   │   └── siamese_unet.py       # SiameseUNet model
│   └── utils/
│       ├── __init__.py
│       └── losses.py             # FocalDiceLoss
├── dataset/                      # Data (gitignored)
├── checkpoints/                  # Model weights (gitignored)
└── results/                      # Visualizations
```

---

## 8. GIT HISTORY

### Commits on `develop` branch:
1. `57bc355` - Initial commit: README.md and AGENTS.md
2. `9f3e8c8` - Add .gitignore for dataset
3. `8e41ad1` - Add Phase 1 data exploration notebook
4. `7f6d2ce` - Add Phase 2: Literature survey + Siamese UNet implementation

### Branch: `main` (original commit only)
- Initial setup files

---

## 9. REMAINING WORK (from README.md Roadmap)

### Phase 3: Data Pipeline Implementation ✅ DONE
- [x] Create dataset class
- [x] Implement label remapping
- [ ] Build augmentation pipeline (flip, rotate, color jitter)
- [x] Create data loaders
- [x] Handle multi-modal input

### Phase 4: Model Architecture Design ✅ DONE
- [x] Design Siamese UNet baseline
- [x] Attention mechanisms (AttentionGate)
- [x] Pretrained ResNet34 backbone
- [x] Skip connections

### Phase 5: Training Strategy
- [ ] Run training with config.yaml
- [ ] Implement early stopping
- [ ] Checkpoint based on val IoU

### Phase 6: Evaluation & Analysis
- [ ] Compute IoU, Precision, Recall, F1 on val and test
- [ ] Generate confusion matrix
- [ ] Create visualizations (success + failure cases)

### Phase 7: Report Writing
- [ ] Write Abstract, Literature Survey, Methodology
- [ ] Present Results with tables/figures
- [ ] Propose Future Work, Conclusion
- [ ] Include Time and Resource Log

### Phase 8: Submission Preparation
- [ ] Train final model
- [ ] Upload checkpoint (Google Drive/HuggingFace)
- [ ] Create ZIP file
- [ ] Push to public GitHub

---

## 10. KEY DESIGN DECISIONS

1. **Dual-Modal Encoders:** Separate encoders for EO and SAR because they have different channel counts (3 vs 1) and different physical properties

2. **Focal + Dice Loss:** Combined loss chosen because:
   - Focal: handles class imbalance by focusing on hard examples
   - Dice: optimizes directly for IoU metric
   - Combined: best of both worlds (proven in literature)

3. **ResNet34 Backbone:** Chosen because:
   - Good balance of performance vs compute
   - Pretrained ImageNet weights available
   - 2781 training samples is limited for larger models

4. **256×256 Input Size:** Downsampled from 1024×1024 to fit in GPU memory and speed up training

5. **Attention Gates:** Included based on MRA-SNet paper showing improved change detection with attention

---

## 11. NEXT STEPS FOR GPT-5.5

1. **Test the code:** Run training to verify everything works
2. **Fix any issues:** Debug any errors in data loading, model, training
3. **Run full training:** Train model on full dataset
4. **Evaluate:** Compute all metrics on val and test sets
5. **Create visualizations:** Generate success/failure case images
6. **Write report:** Complete technical report (PDF)
7. **Prepare submission:** Upload weights, create ZIP, push to GitHub

---

## 12. FILES TO READ FOR FULL CONTEXT

1. **AGENTS.md** - Agent guidelines and code style
2. **README.md** - Project overview and roadmap
3. **notebooks/01_data_exploration.ipynb** - Data analysis findings
4. **notebooks/02_literature_survey.ipynb** - Architecture design rationale
5. **config.yaml** - All hyperparameters
6. **src/models/siamese_unet.py** - Model architecture
7. **src/data/dataset.py** - Data pipeline
8. **src/utils/losses.py** - Loss functions
9. **src/train.py** - Training loop

---

*End of Project Handoff*