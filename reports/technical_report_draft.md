# Technical Report Draft

## Abstract

This project tackles binary change detection on paired EO-SAR satellite imagery for the GalaxEye technical assignment. The final implemented baseline is a dual-encoder EO-SAR UNet with separate ResNet34 backbones for RGB EO pre-event imagery and grayscale SAR post-event imagery, feature fusion across scales, and an attention-gated decoder. The current repository has a complete training, evaluation, and visualization pipeline with mandatory label remapping, class-imbalance-aware loss, and change-class metrics. Debug-only validation confirms that the code path is working end to end; full experiment results remain pending on a GPU-capable machine.

## Literature Survey

- Siamese and pseudo-Siamese encoder-decoder architectures are strong baselines for change detection because they preserve modality-specific representations while enabling feature comparison.
- EO-SAR fusion requires modality-aware handling because optical and radar imagery differ in channel structure, noise, and physical signal characteristics.
- Severe foreground imbalance motivates hybrid losses such as Focal + Dice.

## Methodology

### Problem Setup

- Input: pre-event EO image `(1024, 1024, 3)` and post-event SAR image `(1024, 1024)`
- Output: binary change mask
- Label remapping: `{0, 1} -> 0`, `{2, 3} -> 1`

### Data Pipeline

- EO images are loaded as RGB and SAR images as single-channel grayscale.
- Images are resized to `256x256`.
- EO and SAR are normalized separately.
- Binary masks are resized with nearest-neighbor interpolation.
- Spatial augmentations are applied jointly to EO, SAR, and mask.

### Model

- Model: `DualEncoderEOSARUNet`
- EO encoder: pretrained ResNet34, 3-channel input
- SAR encoder: pretrained ResNet34 adapted to 1-channel input by averaging pretrained RGB first-layer weights
- Fusion: concatenation of EO and SAR features at multiple scales
- Decoder: attention-gated UNet-style decoder
- Output: 2-channel logits for binary segmentation

### Optimization

- Loss: `FocalDiceLoss`
- Optimizer: `AdamW`
- Scheduler: cosine annealing
- Early stopping and best-checkpoint saving based on validation IoU

## Debug Results

These are pipeline-verification results from `checkpoints_debug/best.pth`, trained for one epoch on four samples.

### Validation

- IoU: `0.0220`
- Precision: `0.0220`
- Recall: `1.0000`
- F1: `0.0431`

### Test

- IoU: `0.0076`
- Precision: `0.0076`
- Recall: `1.0000`
- F1: `0.0150`

### Interpretation

- The debug checkpoint predicts change too aggressively, producing perfect recall but very poor precision.
- This is expected from a tiny debug-only run and should not be treated as final performance.

## Figures and Artifacts

- Validation predictions: `results_debug_val/predictions.png`
- Test predictions: `results_debug_test/predictions.png`
- Validation confusion matrix: `results_debug_val/confusion_matrix.png`
- Test confusion matrix: `results_debug_test/confusion_matrix.png`
- Debug analysis summary: `results_debug_summary.md`

## Future Work

- Run full training on a GPU-capable machine using `config.yaml`
- Tune thresholding, loss weights, and sampling strategies for class imbalance
- Compare the current ResNet34 baseline against stronger backbones or feature-fusion variants
- Add nodata-aware masking directly into optimization and metric computation if needed

## Conclusion

The repository now contains a complete and reproducible baseline for EO-SAR binary change detection, including data loading, model definition, training, evaluation, and visualization. The implementation has been debug-validated successfully, and the next meaningful milestone is full experiment execution on appropriate hardware.
