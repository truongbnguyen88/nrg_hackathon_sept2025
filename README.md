# FlexTrack Challenge 2025 Solution

This repository contains my solution for the **AIcrowd FlexTrack Challenge 2025**, completed during a one-week data science hackathon in **September 2025**.

## Challenge overview

The competition focused on a two-stage energy demand response problem:

1. **Classification task**: predict the demand response flag at each timestamp with three possible classes: `-1`, `0`, and `+1`.
2. **Regression task**: estimate the demand response capacity during an event relative to a baseline.

Challenge link: https://www.aicrowd.com/challenges/flextrack-challenge-2025/

## Repository structure

### Core code
- `helper_modules.py`: feature engineering and preprocessing utilities for both phase 1 and phase 2 models.
- `net_architecture.py`: PyTorch neural network architecture for the phase 1 multiclass classifier.

### Notebooks
- `data_investigation.ipynb`: exploratory data analysis and dataset investigation.
- `phase1_train.ipynb`: training workflow for the phase 1 classifier.
- `phase2_train.ipynb`: baseline phase 2 regression training workflow.
- `phase2_train_method2.ipynb`: alternative experimentation for phase 2 modeling.
- `phase2_train_method3.ipynb`: additional phase 2 training experiments.
- `predict.ipynb`: inference pipeline for the main prediction workflow.
- `predict_method3.ipynb`: inference workflow for one of the alternative phase 2 methods.

### Data assets
The `datasets/` directory includes local copies of the challenge data and supporting files, including:
- phase 1 train and test files
- an extended training file
- a partial test file
- a sample/random submission format

### Saved models
The `models/` directory contains trained artifacts used by the pipeline:
- `phase1_nn_model.pth`
- `phase1_scaler.pkl`
- `phase2_xgb_clf_nz_model.pkl`
- `phase2_xgb_reg_nz_model.pkl`

## Modeling approach

### Phase 1: multiclass classification
For the demand response flag prediction task, I built a **3-class neural network in PyTorch**.

Key ideas:
- engineered calendar, weather, and cyclical time features
- used batch normalization and dropout for regularization
- tuned training settings such as learning rate, batch size, early stopping, and learning-rate scheduling
- used **focal loss** to improve learning on minority classes
- remapped the target label `-1` to `2` internally for model training

The network definition is implemented in `net_architecture.py` as a multilayer feedforward architecture with hidden layers sized `64 -> 128 -> 64 -> 32`.

### Phase 2: demand response capacity regression
For the capacity prediction task, I used a **two-step XGBoost-based approach**.

Because the regression target contains many zeros, the second stage is decomposed into:
- a classifier to estimate whether the target is zero or non-zero
- a regressor to predict the capacity magnitude for non-zero cases

The final prediction is computed as:

`y_hat = P_nonzero * y_pred`

where:
- `P_nonzero` is produced by the non-zero classifier
- `y_pred` is produced by the regression model

## Feature engineering

A substantial part of the project is in feature engineering and leakage-aware preprocessing.

Highlights from `helper_modules.py`:
- weather-derived features such as heating/cooling degree indicators and radiation transforms
- timestamp-based features including quarter, month, week, day, hour, weekend flags, and month/quarter boundaries
- leak-safe lag and rolling statistics for:
  - building power
  - horizontal radiation
  - dry bulb temperature
- anomaly, baseline, and percentage-change features
- one-hot encoding utilities for phase 2 target conditioning

The phase 2 preprocessing pipeline uses a **30-minute freeze window** to avoid target leakage when building lagged and rolling features.

## End-to-end prediction pipeline

The overall scoring pipeline is:

1. preprocess raw inputs and generate engineered features
2. use the neural network to predict the demand response flag
3. pass the predicted flag into the second-stage feature set
4. use `phase2_xgb_clf_nz_model.pkl` to estimate `P_nonzero`
5. use `phase2_xgb_reg_nz_model.pkl` to estimate `y_pred`
6. compute final output as `y_hat = P_nonzero * y_pred`

## Results and reflections

- The phase 1 multiclass classifier performed well on scoring data and achieved a strong geometric mean score.
- The phase 2 regression task was more challenging and ultimately underperformed relative to the top leaderboard entries.
- My best regression result was around **1.7 MAE**, while the leading score at the time I stopped was approximately **1.1x**.

## Notes

- Most of the work in this repository is notebook-driven experimentation.
- The repository is primarily composed of **Jupyter Notebook** and **Python** code.
- Model artifacts are committed for reproducibility and inference reuse.

## Future improvements

If I continue this project, I would likely explore:
- stronger ensembling for phase 2 regression
- better handling of zero-inflated targets
- site-aware or group-aware temporal features
- more systematic validation for the two-stage pipeline
- calibration of the phase 1 classifier outputs before phase 2 modeling
