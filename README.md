MS-CAM-Net

Official implementation of MS-CAM-Net (Multi-Stage Contrastive Attention Mechanism Network) for facial-video-based non-contact stress recognition.

The framework follows a three-stage learning pipeline:

1. Stage 1 — Spatiotemporal Contrastive Pre-training
   - Self-supervised representation learning from facial video clips.
   - Uses a 3D-CNN encoder, projection head, contrastive learning, and temporal-consistency regularization.
   - No stress labels or physiological targets are used for the Stage 1 optimization objective.

2. Stage 2 — Supervised rPPG Refinement and Vital-Sign Estimation
   - Transfers the Stage 1 encoder.
   - Uses hierarchical encoder features, Memory-Attention modules with rotary positional encoding, and long-range temporal refinement.
   - Reconstructs an rPPG waveform and estimates heart rate (HR) and heart-rate variability (HRV).

3. Stage 3 — Multi-Task Stress Classification
   - Transfers the physiologically refined representation from Stage 2.
   - Uses dual-statistic temporal pooling, a shared stress representation bridge, and two task-specific classification heads.
   - Performs:
     - Binary Stress State Recognition: Rest vs. Stress.
     - Three-Class Stress Level Recognition: Rest vs. Low Stress vs. High Stress.
   - Retains auxiliary rPPG supervision during training.

---

 Repository Structure

```text
MS-CAM-Net/
│
├── README.md
├── requirements.txt
├── requirements_full.txt
├── .gitignore
│
├── preprocessing/
│   └── extract_clips.py
│
├── stage1/
│   ├── __init__.py
│   ├── data_stage1.py
│   ├── model_stage1.py
│   ├── train_stage1.py
│   └── visualization_utils.py
│
├── stage2/
│   ├── __init__.py
│   ├── dataset_stage2.py
│   ├── model_parts.py
│   ├── stage2_model.py
│   ├── losses_stage2.py
│   ├── train_stage2.py
│   ├── evaluate_stage2_physiology.py
│   └── utils/
│       ├── __init__.py
│       └── physiology.py
│
└── stage3/
    ├── __init__.py
    ├── dataset_stage3.py
    ├── model_stage3.py
    ├── losses_stage3.py
    ├── metrics_stage3.py
    ├── trainer_stage3.py
    ├── train_stage3.py
    ├── evaluate_stage3_test.py
    └── visualize_stage3_results.py
````

---

 Dataset

The experiments were conducted using the UBFC-Phys dataset.

The dataset contains facial videos and synchronized physiological recordings from 56 subjects performing three tasks:

 T1: Rest
 T2: Speech / Low Stress
 T3: Arithmetic / High Stress

The original UBFC-Phys data are not redistributed in this repository.

Users must obtain the dataset separately and provide the local dataset path when running the preprocessing script.

---

 Data Partitioning

A subject-independent protocol is used to prevent subject-level leakage.

The fixed subject-independent test cohort contains:


s1
s10
s11
s12
s13
s14
s15
s16
```

These eight subjects are excluded from:

 Stage 1 self-supervised pre-training
 Stage 2 supervised physiological refinement
 Stage 3 stress-classification training
 development cross-validation
 hyperparameter selection
 early stopping
 checkpoint selection

The remaining 48 subjects constitute the development pool.

For the original development protocol, the 48 subjects are evaluated using 10-fold subject-exclusive cross-validation.

Because 48 is not exactly divisible by 10:

 eight folds contain 43 training subjects and 5 validation subjects;
 two folds contain 44 training subjects and 4 validation subjects.

---

 Preprocessing

The facial videos are processed using the following configuration:

```text
Video frame rate        : 35 fps
Clip length             : 2100 frames
Clip duration           : 60 s
Step size               : 1050 frames
Temporal overlap        : 50%
ROI size                : 64 x 64 pixels
Facial landmarks        : dlib 68-point landmark detector
Forehead extension      : 0.7 x interocular distance
Color format            : RGB
```

When face or landmark detection fails for an individual frame, the most recent valid ROI coordinates are reused.

The preprocessing script stores the processed clips separately for development and fixed test subjects.

Example:

```bash
python preprocessing/extract_clips.py \
  --dataset-root "/path/to/UBFC-Phys" \
  --predictor-path "/path/to/shape_predictor_68_face_landmarks.dat" \
  --output-root "/path/to/processed_data" \
  --partition both
```

Expected structure:

```text
processed_data/
├── development/
└── test/
```

---

 Environment

The experiments were implemented using Python and PyTorch.

The tested software environment included the following principal packages:

Python              : 3.9.23
PyTorch             : 2.5.1
TorchVision         : 0.20.1
TorchAudio          : 2.5.1
NumPy               : 1.26.4
Pandas              : 2.2.2
SciPy               : 1.13.1
Scikit-learn        : 1.4.2
OpenCV-Python       : 4.9.0.80
dlib                : 19.24.0
NeuroKit2           : 0.2.7
Matplotlib          : 3.8.4
Seaborn             : 0.13.2
Pillow              : 11.1.0
PyWavelets          : 1.6.0
tqdm                : 4.66.4
h5py                : 3.10.0
PyYAML              : 6.0.2
joblib              : 1.5.1


The reported experiments were executed on an:

NVIDIA GeForce RTX 3060
12 GB VRAM


 Stage 1: Spatiotemporal Contrastive Pre-training

Stage 1 performs self-supervised representation learning using only facial-video information.

A random temporal segment of 128 consecutive frames is sampled from every processed 60-s clip.

Two identical views of the segment are passed through the shared 3D-CNN encoder.

No color jitter, hue modification, saturation perturbation, frame-wise brightness perturbation, or channel-wise color transformation is applied.

Main settings:

Optimizer                    : Adam
Learning rate                : 3e-4
Physical batch size          : 16
Maximum epochs               : 100
Gradient accumulation        : No
Temperature                  : 0.07
Negative-logit margin        : 0.3
Temporal-consistency weight  : 0.25
Temporal segment length      : 128 frames
```

The Stage 1 objective combines:

 margin-adjusted contrastive learning;
 temporal-consistency regularization.

Run Stage 1 using:

python stage1/train_stage1.py \
  --development-dir "/path/to/processed_data/development" \
  --output-dir "/path/to/results/stage1" \
  --num-workers 4
  
The Stage 1 projection head is used only during contrastive pre-training.

Only the trained encoder is transferred to Stage 2.

---

 Stage 2: Supervised rPPG Refinement and Vital-Sign Estimation

Stage 2 transfers the pretrained Stage 1 encoder and performs physiologically supervised refinement.

The projection head from Stage 1 is discarded.

Hierarchical features are extracted from:

P5
P6
P7
Pout

Each hierarchy is processed by a dedicated Memory-Attention module.

The refined hierarchical features are integrated and processed by a six-block long-range refinement module.

The unified representation is then used for:

 rPPG reconstruction;
 HR estimation;
 HRV estimation.

Main optimization settings:


Optimizer                    : AdamW
Base learning rate           : 5e-5
Encoder learning rate        : 0.05 x base LR
rPPG head learning rate      : 1.2 x base LR
Vitals head learning rate    : 1.5 x base LR

K-fold weight decay          : 0.01
Final-training weight decay  : 0.05
Early-stopping patience      : 5
Checkpoint criterion         : validation HR MAE
```

The Stage 2 objective is:

LStage2 =
alpha_epoch 
(
    5.0  LHR
  + 2.0  LHRV
  + 2.0  Lsignal
  + 0.1  Laux
)

where:


alpha_epoch = min(1, epoch / 10)

The first ten epochs therefore provide a progressive warm-up of the physiological objectives.

Run Stage 2 using:

python stage2/train_stage2.py \
  --development-dir "/path/to/processed_data/development" \
  --stage1-dir "/path/to/MS-CAM-Net/stage1" \
  --stage1-weights "/path/to/stage1_encoder_checkpoint.pth" \
  --output-dir "/path/to/results/stage2" \
  --num-folds 10


 Stage 2 Physiological Evaluation

The Stage 2 physiological evaluation includes:

```text
Direct HR MAE
Direct HR RMSE
Direct HR Pearson correlation

Direct HRV MAE
Direct HRV RMSE
Direct HRV Pearson correlation

Raw waveform Pearson correlation
Polarity-corrected Pearson correlation
Polarity/lag-aware Pearson correlation
Artifact-masked waveform NRMSE
Cardiac-band spectral SNR
Spectral coherence

HR-from-rPPG MAE
HR-from-rPPG Pearson correlation

RMSSD-from-rPPG MAE
RMSSD-from-rPPG Pearson correlation
```

The session-level physiological analysis uses:

Window length         : 20 s
Window overlap        : 50%
Reconstruction        : weighted overlap-add
Session separation    : T1, T2, and T3 processed independently
HR/HRV aggregation    : median within each complete session
```

Example:

python stage2/evaluate_stage2_physiology.py \
  --development-dir "/path/to/processed_data/development" \
  --stage1-dir "/path/to/MS-CAM-Net/stage1" \
  --stage1-weights "/path/to/stage1_encoder_checkpoint.pth" \
  --fold-root "/path/to/results/stage2" \
  --output-dir "/path/to/results/stage2_physiology"
```

---

 Stage 3: Multi-Task Stress Classification

Stage 3 transfers the physiologically refined modules learned during Stage 2.

During Stage 3 fine-tuning, the following transferred modules are unfrozen:

```text
3D encoder
Memory-Attention stacks
Long-Range Refinement module
```

The refined temporal representation has 64 channels.

Dual-statistic temporal pooling is applied using:


Global Average Pooling
+
Global Maximum Pooling

The two 64-dimensional descriptors are concatenated:

```text
64 + 64 = 128 dimensions
```

The resulting feature is processed by the Shared Stress Bridge:

```text
128 -> 128
LayerNorm
GELU
Dropout(0.4)
128 -> 64
GELU
```

The 64-dimensional shared representation is used by two independent task-specific heads.

 Binary Stress-State Head

```text
64 -> 32 -> 2
```

Classes:

```text
0 = Rest
1 = Stress
```

 Three-Class Stress-Level Head

```text
64 -> 128
LayerNorm
GELU
Dropout(0.5)
128 -> 64
GELU
64 -> 3
```

Classes:

```text
0 = Rest
1 = Low Stress
2 = High Stress
```

Task mapping:

```text
T1 -> State = 0, Level = 0
T2 -> State = 1, Level = 1
T3 -> State = 1, Level = 2
```

---

 Stage 3 Training Objective

The Stage 3 objective combines:

 binary stress-state classification;
 three-class stress-level classification;
 auxiliary rPPG reconstruction.

The classification losses use cross-entropy with label smoothing:

```text
Label smoothing = 0.1
```

The complete Stage 3 objective is:

```text
LStage3 =
2.0  Lstate
+
5.0  Llevel
+
0.1  LrPPG
```

Main optimization settings:

```text
Optimizer                     : AdamW
AMSGrad                       : Enabled
Initial learning rate         : 2e-4
Warm-up                       : 3 epochs
Physical batch size           : 2
Gradient accumulation         : 4
Effective optimization batch  : up to 8
Gradient clipping             : 1.0
Weight decay                  : 0.05
Label smoothing               : 0.1
Early-stopping patience       : 15
```

Run Stage 3 using:

```bash
python stage3/train_stage3.py \
  --development-dir "/path/to/processed_data/development" \
  --stage1-dir "/path/to/MS-CAM-Net/stage1" \
  --stage2-dir "/path/to/MS-CAM-Net/stage2" \
  --stage2-weights "/path/to/stage2_checkpoint.pth" \
  --output-dir "/path/to/results/stage3" \
  --final-train
```


 Final Subject-Independent Test Evaluation

The fixed test cohort must not be used during model training or model selection.

After training is complete, run:

```bash
python stage3/evaluate_stage3_test.py \
  --test-dir "/path/to/processed_data/test" \
  --stage1-dir "/path/to/MS-CAM-Net/stage1" \
  --stage2-dir "/path/to/MS-CAM-Net/stage2" \
  --stage2-weights "/path/to/stage2_checkpoint.pth" \
  --stage3-weights "/path/to/stage3_checkpoint.pth" \
  --output-dir "/path/to/results/stage3_test"
```

The classification evaluation includes:

```text
Accuracy
Weighted Precision
Weighted Recall
Weighted F1-score
AUC
```

Classification MAE is not used.

---

 Visualization

The Stage 3 visualization utility can generate:

 binary confusion matrix;
 three-class confusion matrix;
 t-SNE representation of the learned feature space.

Example:

```bash
python stage3/visualize_stage3_results.py \
  --input "/path/to/results/stage3_test/test_predictions_and_features.npz" \
  --output-dir "/path/to/results/stage3_test/figures"
```

---

 Reproducibility Notes

To avoid subject-level information leakage:

1. Subject partitioning must be performed before model training.
2. All clips from the same participant must remain in the same subject partition.
3. The fixed test subjects must not be used for:

    training;
    self-supervised pre-training;
    hyperparameter optimization;
    early stopping;
    learning-rate scheduling;
    checkpoint selection.
4. The fixed test set must be evaluated only after the final model checkpoint has been selected.
5. The original fixed test evaluation and repeated random subject-exclusive evaluations must be treated as separate experimental protocols.
6. The UBFC-Phys dataset itself is not redistributed by this repository.