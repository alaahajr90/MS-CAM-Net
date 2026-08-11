# MS-CAM-Net — Stage 2: Supervised Physiological Refinement

This directory contains the reproducibility implementation of Stage 2 described in the manuscript.

## Scope

Stage 2 receives the Stage 1 3D-CNN encoder and discards the Stage 1 projection head. The transferred encoder is selectively fine-tuned. Hierarchical P5, P6, P7, and Pout features are processed by RoPE-enhanced Memory-Attention modules, integrated into a common temporal representation, refined by a six-block long-range module, and used for:

- primary rPPG waveform reconstruction;
- direct heart-rate (HR) estimation;
- direct heart-rate-variability (HRV) estimation.

Auxiliary rPPG heads are active only for the deep-supervision term used during Stage 2 training. Respiratory-rate estimation is not part of the reported Stage 2 implementation.

## Data isolation

Use only the `development/` directory produced by the repository preprocessing script. The fixed subject-independent test subjects must not be used for Stage 2 training, validation, early stopping, hyperparameter selection, or checkpoint selection.

The expected development pool contains 48 subjects. The fixed test subjects are:

`s1, s10, s11, s12, s13, s14, s15, s16`

## Manuscript-aligned training objective

The implementation follows:

`L_S2 = alpha_e * (5 L_HR + 2 L_HRV + 2 L_sig + 0.1 L_aux)`

with `alpha_e = min(1, e/10)` for one-based epoch `e`.

- `L_sig`: MSE between reconstructed rPPG and synchronized reference BVP after temporal resampling.
- `L_HRV`: MSE for direct HRV estimation.
- `L_HR`: Huber loss with delta 10 plus the manuscript's error-dependent absolute-error penalty.
- `L_aux`: mean MSE across auxiliary deep-supervision rPPG heads.

## Optimization

Default settings:

- AdamW
- base learning rate: `5e-5`
- encoder learning rate: `0.05 * LR`
- rPPG reconstruction heads: `1.2 * LR`
- vital-sign head: `1.5 * LR`
- K-fold weight decay: `0.01`
- final-development training weight decay: `0.05`
- batch size: `4`
- early stopping patience: `5`, based on validation HR MAE

## 10-fold development run

```bash
python train_stage2.py \
  --development-dir /path/to/processed_data/development \
  --stage1-dir /path/to/MS-CAM-Net/stage1 \
  --stage1-weights /path/to/stage1/final_stage1_encoder_last.pth \
  --output-dir /path/to/results/stage2_10fold \
  --num-folds 10 \
  --final-train
```

With 48 development subjects, 10-fold splitting yields eight 43/5 train/validation folds and two 44/4 folds.

## Five-fold physiological validation

The manuscript reports an additional five-fold subject-exclusive physiological validation. Train the corresponding checkpoints first:

```bash
python train_stage2.py \
  --development-dir /path/to/processed_data/development \
  --stage1-dir /path/to/MS-CAM-Net/stage1 \
  --stage1-weights /path/to/stage1/final_stage1_encoder_last.pth \
  --output-dir /path/to/results/stage2_5fold \
  --num-folds 5
```

Then run complete-session evaluation:

```bash
python evaluate_stage2_physiology.py \
  --development-dir /path/to/processed_data/development \
  --stage1-dir /path/to/MS-CAM-Net/stage1 \
  --stage1-weights /path/to/stage1/final_stage1_encoder_last.pth \
  --fold-root /path/to/results/stage2_5fold \
  --output-dir /path/to/results/stage2_physiology
```

The evaluator uses overlapping 20-s windows with 50% overlap, reconstructs session-level signals by Hann-weighted overlap-add, retains T1/T2/T3 as separate sessions, and aggregates direct HR/HRV predictions within a session using the median.

## Important reproducibility note

This repository does not redistribute UBFC-Phys or the dlib landmark model. Obtain those resources from their official sources and comply with their licenses/terms.
