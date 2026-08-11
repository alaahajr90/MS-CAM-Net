# MS-CAM-Net Stage 3: Multi-Task Stress Classification

This directory contains the Stage 3 implementation used for binary stress-state recognition and three-class stress-level recognition.

## Targets

The UBFC-Phys task mapping is:

- T1 -> Rest: state = 0, level = 0
- T2 -> Stress / Low Stress: state = 1, level = 1
- T3 -> Stress / High Stress: state = 1, level = 2

## Architecture

Stage 3 loads the trained Stage 2 physiological model and selectively fine-tunes the transferred 3D encoder, Memory-Attention stacks, and Long-Range Refinement module. The refined 64-D temporal representation is summarized by concatenating temporal mean and temporal maximum pooling, resulting in a 128-D descriptor.

The shared stress bridge is:

- Linear 128 -> 128
- LayerNorm
- GELU
- Dropout 0.4
- Linear 128 -> 64
- GELU

The binary head predicts Rest versus Stress. The three-class head predicts Rest, Low Stress, and High Stress and uses dropout 0.5.

## Training Objective

The reported Stage 3 objective is:

`L = 2.0 * L_state + 5.0 * L_level + 0.1 * L_rPPG`

Both classification losses use cross-entropy with label smoothing 0.1. The auxiliary rPPG term uses mean squared error.

## Optimization

- AdamW with AMSGrad
- Initial learning rate: 2e-4
- Weight decay: 0.05
- Physical batch size: 2
- Gradient accumulation: 4 mini-batches
- Effective optimization batch size: up to 8
- Gradient clipping: max norm 1.0
- Warm-up: 3 epochs
- Early-stopping patience: 15 epochs for development folds

## Evaluation Protocol

The fixed subject-independent test cohort is:

`s1, s10, s11, s12, s13, s14, s15, s16`

These subjects must not appear in Stage 1, Stage 2, or Stage 3 development, model selection, early stopping, or hyperparameter tuning.

Classification metrics are Accuracy, weighted Precision, weighted Recall, weighted F1-score, and AUC. Classification MAE is not computed.

## Development Cross-Validation

```bash
python train_stage3.py \
  --development-dir /path/to/processed_data/development \
  --stage1-dir /path/to/MS-CAM-Net/stage1 \
  --stage2-dir /path/to/MS-CAM-Net/stage2 \
  --stage2-weights /path/to/stage2/final_model/final_stage2_model.pth \
  --output-dir /path/to/stage3_results \
  --final-train
```

The script performs 10-fold subject-exclusive development cross-validation. The best checkpoint in each fold is selected using the joint validation loss. With 48 development subjects, eight folds contain 43 training and 5 validation subjects and two folds contain 44 training and 4 validation subjects.

When `--final-train` is specified, a final Stage 3 model is trained using only the 48 development subjects for 100 epochs by default. The fixed test cohort is never read by the training script.

## Final Test Evaluation

```bash
python evaluate_stage3_test.py \
  --test-dir /path/to/processed_data/test \
  --stage1-dir /path/to/MS-CAM-Net/stage1 \
  --stage2-dir /path/to/MS-CAM-Net/stage2 \
  --stage2-weights /path/to/stage2/final_model/final_stage2_model.pth \
  --stage3-weights /path/to/stage3_results/final_model/final_stage3_model.pth \
  --output-dir /path/to/stage3_test_results
```

The evaluation script verifies that the test directory contains exactly the eight fixed test subjects and evaluates them once. It saves metrics, confusion matrices as numeric arrays, predictions, probabilities, and the shared 64-D features.

## Visualization

```bash
python visualize_stage3_results.py \
  --results /path/to/stage3_test_results/test_predictions_and_features.npz \
  --output-dir /path/to/stage3_test_results/figures
```

This creates the binary and three-class confusion matrices and a t-SNE visualization from the saved test features. It does not search for or alter a test subset to match a target accuracy.
