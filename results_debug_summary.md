# Phase 6 Debug Analysis

Current analysis uses the debug checkpoint `checkpoints_debug/best.pth` trained for 1 epoch on 4 samples, so these findings validate pipeline behavior rather than final model quality.

## VAL Split
- IoU: 0.0220
- Precision: 0.0220
- Recall: 1.0000
- F1: 0.0431
- Confusion matrix: TN=0, FP=21407432, FN=0, TP=481592
- Error pattern: the model predicts change on most pixels, producing perfect recall but extremely poor precision.

## TEST Split
- IoU: 0.0076
- Precision: 0.0076
- Recall: 1.0000
- F1: 0.0150
- Confusion matrix: TN=0, FP=5008151, FN=0, TP=38121
- Error pattern: the model predicts change on most pixels, producing perfect recall but extremely poor precision.

## Failure Mode Interpretation
- Severe class imbalance and the tiny debug training set bias the model toward unstable decision boundaries.
- The current checkpoint is useful only for verifying the Phase 6 tooling: metrics, confusion matrices, and qualitative plots are being produced correctly.
- A meaningful Phase 6 analysis requires a fully trained checkpoint from `config.yaml` on a GPU-capable machine.