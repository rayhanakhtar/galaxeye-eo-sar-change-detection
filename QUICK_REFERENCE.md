# Quick Reference for GPT-5.5

## Project Status: Phase 2 Complete (Ready for Phase 5: Training)

### What Has Been Done
1. **Phase 1 - Data Exploration:** Analyzed dataset, found severe class imbalance (62:1 to 131:1)
2. **Phase 2 - Literature Survey & Architecture:** Designed SiameseUNet model

### What Needs to Be Done
1. Run training with `python -m src.train --config config.yaml`
2. Evaluate on test set
3. Write technical report
4. Submit

---

## One-Line Summary
Binary change detection on EO-SAR satellite images using SiameseUNet with dual ResNet34 encoders and Focal+Dice loss to handle 62:1 class imbalance.

---

## Key Commands
```bash
# Train model
python -m src.train --config config.yaml

# Evaluate
python -m src.eval --data_path dataset/test --weights checkpoints/best.pth

# Visualize
python -m src.visualize --data_path dataset/test --weights checkpoints/best.pth --output results
```

---

## Critical Notes
- **Label Remapping:** Already applied in dataset.py (0,1→0, 2,3→1)
- **Class Imbalance:** Handled by FocalDiceLoss (0.5 focal + 0.5 dice)
- **EO vs SAR:** Pre-event is RGB (3ch), Post-event is grayscale (1ch)
- **Pretrained:** ResNet34 ImageNet weights used

---

## File Locations
- Config: `config.yaml`
- Model: `src/models/siamese_unet.py`
- Dataset: `src/data/dataset.py`
- Loss: `src/utils/losses.py`
- Train: `src/train.py`
- Eval: `src/eval.py`

---

## Checkpoints & Results
- Directory: `checkpoints/`
- Directory: `results/`

*(Full context in PROJECT_HANDOFF.md)*