# Agent Guidelines for EO-SAR Change Detection Project

This file provides guidelines for AI agents working on the binary change detection project.

## Project Context

- **Task**: Binary change detection on paired EO-SAR satellite imagery
- **Goal**: Pixel-level classification (Change vs No-Change)
- **Dataset**: Train (2781), Val (334), Test (77) samples
- **Framework**: PyTorch with potential use of segmentation libraries

## Directory Structure

```
project/
├── AGENTS.md
├── README.md
├── config.yaml                 # Hyperparameters
├── requirements.txt           # Dependencies
├── src/
│   ├── data/                  # Dataset classes
│   ├── models/                # Model architectures
│   ├── utils/                 # Helper functions
│   ├── train.py               # Training script
│   ├── eval.py                # Evaluation script
│   └── visualize.py           # Visualization script
├── dataset/
│   ├── train/{pre-event,post-event,target}/
│   ├── val/{pre-event,post-event,target}/
│   └── test/{pre-event,post-event,target}/
├── checkpoints/               # Saved model weights
└── results/                   # Evaluation outputs
```

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (uses config.yaml)
python -m src.train --config config.yaml

# Evaluate on test set
python -m src.eval --data_path dataset/test --weights checkpoints/best.pth

# Run single test
python -m pytest tests/ -v -k "test_name_here"

# Lint code
python -m ruff check src/
python -m black --check src/

# Format code
python -m black src/
python -m isort src/
```

## Code Style Guidelines

### Imports
- Use absolute imports: `from src.data.dataset import EODataset`
- Group imports: stdlib, third-party, local
- Use `__init__.py` for package exports
- No wildcard imports

### Formatting
- Line length: 100 characters max
- 4 spaces for indentation (no tabs)
- Use Black formatter
- Sort imports with isort (profile: black)

### Types
- Use type hints for all function signatures
- Prefer explicit types over TypeVar
- Document complex types in docstrings
- Example:
  ```python
  def load_image(path: str) -> torch.Tensor:
      """Load and preprocess image.

      Args:
          path: Path to image file.

      Returns:
          Tensor of shape (C, H, W).
      """
  ```

### Naming Conventions
- **Files**: snake_case (e.g., `data_loader.py`)
- **Classes**: PascalCase (e.g., `ChangeDetectionDataset`)
- **Functions/methods**: snake_case (e.g., `compute_iou`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `NUM_CLASSES`)
- **Variables**: snake_case (e.g., `learning_rate`)

### Error Handling
- Use custom exceptions for domain-specific errors
- Catch specific exceptions, not bare `except`
- Include context in error messages
- Use logging instead of print for debugging

### ML-Specific Guidelines

**Dataset Classes**:
- Inherit from `torch.utils.data.Dataset`
- Implement `__len__` and `__getitem__`
- Handle label remapping (0,1 → 0, 2,3 → 1)
- Return dict with keys: `pre_img`, `post_img`, `target`, `filename`

**Model Architecture**:
- Use Siamese encoder for change detection
- Support pretrained backbones (ResNet50, EfficientNet)
- Implement proper device placement (GPU/CPU)

**Training**:
- Use config.yaml for all hyperparameters
- Implement early stopping based on validation IoU
- Save checkpoints with metrics in filename
- Log training progress (tensorboard/wandb)

**Evaluation**:
- Compute IoU, Precision, Recall, F1 for change class
- Generate confusion matrix
- Save qualitative visualizations

## Key Implementation Notes

1. **Multi-modal Input**: Images have both EO (optical) and SAR (radar) channels
   - Handle different physical properties and noise profiles
   - Consider channel-wise normalization

2. **Class Imbalance**: Change pixels are minority class
   - Use Focal Loss, Dice Loss, or weighted cross-entropy
   - Report class distribution in analysis

3. **Label Remapping** (MANDATORY):
   - Original: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
   - Remapped: 0=No-Change (0,1), 1=Change (2,3)

4. **Test Set**: Only 50% provided; model must generalize

## Testing Guidelines

- Unit tests for data loading and transforms
- Integration tests for training loop
- Test on small subset before full training
- Use pytest fixtures for reusable setup

## Documentation Requirements

- Docstrings for all public functions (Google style)
- README with setup instructions
- Config comments explaining hyperparameters
- Inline comments for complex logic

## Common Pitfalls to Avoid

1. Forgetting to apply label remapping before training/evaluation
2. Not handling GPU/CPU device placement correctly
3. Memory issues with large images (use tiling or resizing)
4. Data leakage between train/val/test
5. Not normalizing SAR images differently from EO

## Output Expectations

For submission, ensure:
- Checkpoints uploaded to public link (Google Drive/HuggingFace)
- Metrics reported on both validation and test splits
- Qualitative visualizations showing success/failure cases
- Clean, reproducible code with no hardcoded paths