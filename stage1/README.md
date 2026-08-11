# MS-CAM-Net — Stage 1 Reproducibility Code

This directory contains the preprocessing and Stage 1 self-supervised pre-training code used to reproduce the Stage 1 protocol described for MS-CAM-Net.

## Protocol implemented here

- UBFC-Phys videos are processed at 35 frames/s.
- Each recording is divided into 60 s clips (2100 frames).
- Consecutive clips use a 30 s step (1050 frames), giving 50% temporal overlap.
- Faces are detected with the dlib frontal-face detector and 68-point landmark model.
- The upper ROI boundary is extended by 0.7 times the interocular distance to include the forehead.
- Facial ROIs are converted from BGR to RGB and resized to 64 × 64 pixels.
- If face/landmark detection fails after a valid ROI has been observed, the most recent valid ROI coordinates are reused.
- Stage 1 samples a 128-frame temporal segment from each 60 s clip.
- No color jitter, hue shift, saturation perturbation, channel-wise color transform, or frame-wise brightness perturbation is applied.
- Stage 1 uses a 3D-CNN encoder and a two-layer projection head.
- The Stage 1 objective combines a margin-adjusted contrastive log-ratio loss with temporal consistency.
- Default hyperparameters: Adam, learning rate 3e-4, batch size 16, up to 100 epochs, patience 10, temperature 0.07, negative-logit margin 0.3, and temporal-consistency weight 0.25.

## Strict subject-independent test exclusion

The following subjects are reserved as the fixed subject-independent test cohort:

`s1, s10, s11, s12, s13, s14, s15, s16`

They are separated before clip generation. Stage 1 reads only the 48-subject `development` partition. The test partition is never loaded for Stage 1 training, validation, early stopping, hyperparameter selection, or checkpoint selection.

## Files

- `extract_clips.py`: subject-partitioned clip extraction and BVP synchronization.
- `data_stage1.py`: Stage 1 dataset and 128-frame temporal sampling.
- `model_stage1.py`: 3D-CNN encoder, projection head, contrastive loss, and temporal-consistency loss.
- `train_stage1.py`: 10-fold subject-exclusive Stage 1 training and checkpoint selection.
- `visualization_utils.py`: loss-curve plotting only. It intentionally does not interpret Stage 1 embeddings as reconstructed rPPG.

## Preprocessing

Run preprocessing once to create separate development and test directories:

```bash
python extract_clips.py \
  --dataset-root /path/to/UBFC-Phys \
  --predictor-path /path/to/shape_predictor_68_face_landmarks.dat \
  --output-root /path/to/processed_ubfc_phys \
  --partition both
```

The generated structure is:

```text
processed_ubfc_phys/
├── development/
├── test/
└── preprocessing_manifest.json
```

Stage 1 must use only `processed_ubfc_phys/development/`.

## Stage 1 training

```bash
python train_stage1.py \
  --development-dir /path/to/processed_ubfc_phys/development \
  --output-dir /path/to/stage1_results \
  --num-workers 4
```

The script creates ten subject-exclusive development folds. With 48 subjects, eight validation folds contain five subjects and two validation folds contain four subjects. The corresponding training folds contain 43 or 44 subjects.

For each fold, the encoder checkpoint with the lowest validation Stage 1 objective is saved. Early stopping uses patience 10.

After the ten folds, the default script trains one development-only transfer encoder on all 48 development subjects. Its training duration is the rounded median of the ten validation-selected fold epochs. This final training does not load or evaluate the fixed test partition.

The compatibility checkpoint for downstream code is:

```text
stage1_results/final_stage1_encoder_last.pth
```

## Dataset availability

The UBFC-Phys dataset is not redistributed in this repository. Users should obtain the dataset from its official source and comply with its license and data-use terms.
