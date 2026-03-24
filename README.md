# EEG Stress Detection: Enhanced CBGG (Kaggle, TensorFlow/Keras)

## 1. Project Goal
This project classifies EEG stress state into 3 classes:
- Low (0)
- Medium (1)
- High (2)

It implements an enhanced CBGG pipeline inspired by Roy et al. (2023), with additional hand-crafted EEG features and an ablation study to measure the contribution of each feature group.

## 2. End-to-End Data Flow
The full data flow is:

1. Load raw EEG and labels from MAT files.
2. Align tensor shape into segment-major format.
3. Expand subject-level labels to segment-level labels.
4. Apply wavelet decomposition (DWT) per channel.
5. Compute per-band features (CBGG + added feature set).
6. Concatenate features into one vector per segment.
7. Normalize features fold-wise (no leakage).
8. Train CBGG deep model using stratified cross-validation.
9. Aggregate fold predictions for final metrics and plots.
10. Run ablation experiments.
11. Save all artifacts to Kaggle working directory.

## 3. Runtime and Infrastructure (Kaggle)
### 3.1 Paths
- Input data root: /kaggle/input
- Outputs: /kaggle/working/results
- Checkpoints: /kaggle/working/checkpoints

### 3.2 GPU and training acceleration
The notebook configures:
- MirroredStrategy for multi-GPU data parallel training when multiple GPUs are available.
- Mixed precision (mixed_float16) to accelerate Tensor Core operations on T4.
- XLA JIT for graph-level optimization.
- Batch size scaling by number of replicas.
- Learning-rate scaling with replica count.

This means on dual T4 runtime, global batch and optimizer step settings adapt automatically.

## 4. Dataset Loading and Label Construction
### 4.1 Input files
The loader searches recursively under /kaggle/input for:
- dataset.mat
- class_012.mat

If both are found in the same folder, that pair is preferred.

### 4.2 Shape normalization
Raw EEG is transformed into:
- X shape: (n_segments, n_channels, n_samples)

This standardization is critical because all downstream signal processing assumes segment-major indexing.

### 4.3 Subject-to-segment label expansion
The provided labels are subject-level, while training occurs at segment-level.
The notebook computes how many segments belong to each subject and expands labels via repeat operations.

Result:
- One label per EEG segment.

## 5. Wavelet Decomposition (DWT)
### 5.1 Why DWT is used
EEG is non-stationary. DWT captures localized time-frequency behavior better than plain FFT-only summaries.

### 5.2 Wavelet choice
- Wavelet family: db4 (Daubechies-4)

Why db4:
- Commonly used in EEG literature.
- Good compact support and smoothness tradeoff.
- Effective for transient and oscillatory EEG components.

### 5.3 Decomposition level
The notebook computes maximum safe level from signal length and uses a bounded level (typically up to 3) for stable feature extraction.

### 5.4 Band outputs
Depending on selected level, resulting coefficient sets represent coarse-to-fine frequency content (approximation + detail sub-bands).

## 6. Feature Engineering
Features are computed per band and per channel, then flattened and concatenated.

### 6.1 Original CBGG features
For each band/channel:
- Mean: central tendency of coefficients.
- Variance: signal dispersion/energy spread.
- Skewness: asymmetry of coefficient distribution.
- Kurtosis: tail/heaviness and peakedness.
- Power: mean squared magnitude (energy proxy).

Interpretation:
- Together they describe distribution shape + energy profile of each EEG band.

### 6.2 Added feature set (novel extension)
For each band/channel:
- Differential Entropy (DE): entropy-like complexity measure for near-Gaussian assumptions.
- Hjorth Activity: variance (signal power in time domain).
- Hjorth Mobility: mean frequency tendency via first derivative dynamics.
- Hjorth Complexity: waveform shape complexity relative to a pure sine.
- Katz Fractal Dimension (KFD): geometric complexity/irregularity of EEG trajectory.

Cross-band/channel ratios:
- High-band to low-band power ratio.
- Mid-band to low-band power ratio (when available).

Why these were added:
- They encode non-linear complexity and relative spectral balance not fully captured by simple moments.

### 6.3 Final feature matrix
Depending on flags:
- Baseline: CBGG-only features.
- Extended: CBGG + added features.
- Optional: added features only.

## 7. Normalization and Leakage Control
A MinMaxScaler is fitted inside each training fold and applied to validation fold using the train-fit scaler only.

This prevents data leakage from validation into training statistics and gives a realistic estimate of generalization.

## 8. Model Architecture (CBGG)
Input tensor per sample:
- (n_features, 1)

Sequence of layers:
1. Conv1D(128, kernel_size=1)
2. Softmax activation
3. MaxPooling1D(pool_size=1)
4. Bidirectional LSTM(64, return_sequences=True)
5. GRU(32, return_sequences=True)
6. GRU(16, return_sequences=False)
7. Dropout(0.2)
8. Dense(n_classes, softmax)

### Why each block is used
- Conv1D(1x1): channel-wise projection/mixing over feature axis.
- BiLSTM: models forward and backward temporal dependencies in the transformed sequence.
- Stacked GRUs: compact recurrent refinement with lower parameter cost than pure LSTM stacks.
- Dropout: regularization against overfitting.
- Final softmax: class probability distribution.

### Mixed precision stability detail
Final Dense output is forced to float32 dtype for numerically stable softmax/loss when using mixed_float16 globally.

## 9. Training Procedure
### 9.1 Cross-validation
- Stratified K-fold (typically 10 folds) at segment level.
- Class balance preserved across folds.

### 9.2 Optimizer and loss
- Adam optimizer.
- Categorical cross-entropy.
- One-hot encoded targets.

### 9.3 Class imbalance handling
Per-fold class weights are computed from training labels and passed to fit().

### 9.4 Learning schedule and stopping
- ReduceLROnPlateau lowers LR when val_loss stalls.
- EarlyStopping monitors val_loss with patience/min_delta.
- Best weights restored before evaluation.

This combination improves speed and avoids wasting epochs after convergence.

## 10. Evaluation and Reporting
Per fold and overall:
- Accuracy
- Weighted F1
- Multi-class ROC-AUC (OvR)

Global outputs:
- Combined confusion matrix.
- Classification report.
- ROC curves per class.
- Convergence curves per fold.

Paper-style metrics table:
- Precision
- Sensitivity (Recall)
- Specificity
- F1
- Accuracy
- Positive likelihood ratio (+LR)
- Negative likelihood ratio (-LR)

## 11. Ablation Study
Three configurations are compared:
1. CBGG features only (baseline)
2. Added features only
3. CBGG + added features (enhanced)

Purpose:
- Quantify incremental value of the added feature family.
- Validate that accuracy gains are due to feature enrichment, not randomness.

## 12. Saved Artifacts
All artifacts are written under /kaggle/working/results (or checkpoints folder), including:
- cv_results.json
- cv_results_partial.json
- metrics_table.csv
- ablation_results.json
- final_summary.json
- convergence_curves.png
- confusion_matrix.png
- roc_curve.png
- ablation_chart.png

## 13. Practical Execution Order
Run notebook sequentially from top to bottom:
1. Configuration and runtime setup
2. Kaggle path checks
3. Data loading
4. Shape and label expansion
5. DWT
6. Feature extraction
7. Main CV training
8. Metrics and plots
9. Ablation
10. Final summary

## 14. What Each Stage Tells You Scientifically
- DWT stage: where discriminative time-frequency content exists.
- CBGG moment features: distributional shifts across stress levels.
- Entropy/Hjorth/KFD: complexity and dynamical irregularity changes with stress.
- Ratios: relative dominance of high vs low-frequency energy.
- Confusion matrix: which stress classes overlap behaviorally.
- ROC-AUC per class: separability quality per class boundary.
- Ablation: direct evidence for contribution of feature families.

## 15. Reproducibility Notes
- Global random seeds are fixed.
- Fold splitting is deterministic via RANDOM_SEED.
- Output paths are deterministic.
- Early stopping and LR scheduling reduce variance from over-training.

## 16. Limitations and Good Next Improvements
- Segment-level split can inflate estimates if subject leakage exists in ordering; subject-independent CV can be added for stricter generalization testing.
- Add confidence intervals via repeated CV seeds.
- Add calibration checks (ECE/Brier) for probability reliability.
- Add SHAP or permutation importance on engineered features for interpretability.

## 17. Quick Troubleshooting
- If only one GPU is shown, runtime may not provide dual GPU; code still works.
- If MAT files are not found, verify dataset attachment under /kaggle/input.
- If memory issues occur, reduce BASE_BATCH_SIZE.
- If training is unstable, lower EFFECTIVE_LR or increase EarlyStopping patience slightly.
