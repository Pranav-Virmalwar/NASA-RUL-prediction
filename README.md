# Aircraft Engine Remaining Useful Life Prediction

An LSTM-based regression project for predicting the **remaining useful life
(RUL)** of turbofan engines from the NASA C-MAPSS FD001 simulated run-to-failure
dataset. The maintained implementation is a Python package and command-line
workflow; the original exploratory notebook is retained as a historical
reference.

## Problem and motivation

Predictive maintenance estimates how long an asset can continue operating
before failure. For each FD001 test engine, the task is to estimate the number
of operating cycles remaining after its final observed cycle. The dataset
contains multivariate sensor readings and operating settings from simulated
engines.

This repository focuses on FD001, which represents one operating condition and
one fault mode. Although files for FD002–FD004 are present in the original
workspace, the maintained data loader and CLI support FD001 only.

## Key features

- Validates FD001 sensor-file dimensions, numeric values, engine IDs, cycle
  ordering, and the number of test RUL labels.
- Splits training data by **engine before** constructing overlapping
  sequences, keeping validation engines separate from training engines.
- Fits feature scaling using training-engine rows only.
- Trains a configurable, three-layer LSTM with the original default network
  sizes and learning-rate schedule.
- Reports two RMSE protocols and the asymmetric C-MAPSS S-score.
- Saves the trained Keras model and a JSON sidecar containing its feature
  order, training configuration, and fitted scaler statistics.

## Architecture and workflow

The implementation is divided into modules with distinct responsibilities:

1. `rul_prediction.config` defines the 26-column C-MAPSS schema, the 14 model
   features, and validated training defaults.
2. `rul_prediction.data` loads and checks the FD001 files, creates an
   engine-level training/validation split, fits a `StandardScaler`, and
   generates labeled training windows and recent test windows.
3. `rul_prediction.model` constructs and trains the TensorFlow/Keras network.
4. `rul_prediction.metrics` aggregates test-window predictions and computes
   RMSE and the C-MAPSS asymmetric S-score.
5. `rul_prediction.cli` coordinates loading, preprocessing, training,
   evaluation, and saving outputs. Run it with `python -m rul_prediction`.

## Methodology and defaults

Unless overridden by command-line arguments, the workflow uses:

| Setting | Default |
| --- | ---: |
| Dataset subset | FD001 |
| Input features | `cycles`, `T24`, `T30`, `T50`, `Nf`, `Nc`, `Ps30`, `phi`, `NRf`, `NRc`, `BPR`, `htBleed`, `W31`, `W32` |
| Sequence length / stride | 30 cycles / 1 cycle |
| Training RUL target cap | 125 cycles |
| Test windows | Up to 5 most recent windows per engine |
| Validation split | 20% of training engines |
| Split/random seed | 83 |
| Network | LSTM(128) → LSTM(64) → LSTM(32) → Dense(96) → Dense(128) → Dense(1) |
| Loss / optimizer | Mean squared error / Adam |
| Learning rate | 0.001 for the first 5 epochs, then 0.0001 |
| Epochs / batch size | 10 / 128 |

Each training-window target is the RUL at the last cycle in that window,
computed from the engine's run-to-failure trajectory and capped at 125. The
test set has no run-to-failure labels in its sensor file; its separate RUL file
provides one label per engine for evaluation.

The validation split is made at engine level before windows are built. This
avoids leakage from highly overlapping windows of the same engine. The scaler
is fitted only on training-engine rows and then applied to validation and test
rows.

## Technologies

- Python 3.10 or 3.11
- NumPy 1.26.4
- pandas 2.2.3
- scikit-learn 1.5.2
- TensorFlow 2.15.1 / Keras
- pytest 8.3.3

Exact package versions are pinned in [`requirements.txt`](requirements.txt).

## Repository structure

```text
.
├── rul_prediction/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── model.py
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_model.py
├── CMaps/                              # local dataset; text files are gitignored
├── rul-prediction-using-lstm-for-aircraft-engine.ipynb
├── FD001_LSTM_piecewise_RMSE_14.5002.h5 # historical model artifact
├── requirements.txt
└── README.md
```

The original internship report and certificate are personal documents, not
runtime dependencies, and are excluded by [`.gitignore`](.gitignore). The
archive `archive (5).zip` duplicates the extracted dataset and is also
excluded. The HDF5 model and C-MAPSS text files are retained in the local
workspace but ignored by Git.

## Prerequisites and installation

Install Python 3.10 or 3.11. TensorFlow is pinned to 2.15.1; use a supported
Python version and a virtual environment to isolate its dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On systems where `python3.11` is unavailable, use the equivalent Python 3.10
executable to create the environment.

## Dataset requirements

The loader expects these three whitespace-delimited FD001 files under the
directory passed to `--data-dir`:

```text
CMaps/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

Each train/test row must have 26 numeric columns: engine ID, cycle, three
operational settings, and 21 sensor measurements. The RUL file must contain
one non-negative value per test engine. The loader validates those shapes and
checks that test engine IDs are consecutive and align with the label count.

The original workspace includes these files locally, but C-MAPSS text files
are excluded from Git. If they are not present in your checkout, obtain the
dataset from NASA's [Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
or another authorized source and place the FD001 files in `CMaps/`. The
repository also includes
[`CMaps/readme.txt`](CMaps/readme.txt) and the cited
[Damage Propagation Modeling paper](CMaps/Damage%20Propagation%20Modeling.pdf).
Check the dataset's current terms and attribution requirements before
redistributing its files.

## Configuration

The CLI exposes these options:

```text
--data-dir
--model-output
--epochs
--batch-size
--seed
--window-length
--early-rul-cap
--test-windows
--validation-fraction
```

Defaults are specified in `rul_prediction.config.TrainingConfig` and mirrored
by the CLI. Invalid window sizes, fractions, learning rates, model sizes, and
training lengths are rejected. The window stride, network layer sizes, and
learning-rate schedule are currently configured in code rather than exposed
as CLI options.

## Run the project

From the repository root, with the virtual environment active and the
FD001 files available:

```bash
python -m rul_prediction \
  --data-dir CMaps \
  --model-output artifacts/fd001_lstm.keras
```

The command trains and evaluates the model, then writes:

- `artifacts/fd001_lstm.keras` — trained Keras model.
- `artifacts/fd001_lstm.preprocessing.json` — feature order, configuration,
  and scaler statistics needed to reproduce the input transformation.

The output directory is created when needed and ignored by Git. For example,
to run a shorter training smoke test:

```bash
python -m rul_prediction --data-dir CMaps --epochs 1
```

That command still evaluates on the FD001 test set; the resulting scores are
not a substitute for a properly configured experiment.

## Evaluation and results

The CLI reports:

- **Mean-window RMSE:** compares each engine's true RUL with the arithmetic
  mean of its up-to-five most recent window predictions.
- **Last-window RMSE:** compares true RUL with the prediction from the
  engine's final available window.
- **C-MAPSS S-score:** applies an asymmetric penalty to final-window
  predictions. Lower is better; underprediction and overprediction use
  different penalty scales.

The original notebook contains historical outputs of **14.5002** mean-window
RMSE, **15.1006** last-window RMSE, and **436.027** last-window S-score. These
are outputs saved with the original exploratory notebook, not results
reproduced by the maintained package. Its preprocessing and validation
procedure differs from the package implementation, so these values should not
be treated as a direct benchmark of a fresh run.

## Tests

Run the automated tests from the repository root:

```bash
python -m pytest
```

The test suite covers target capping, window boundaries, malformed data,
engine-level split behavior, test-window aggregation, metric validation, and
model forward-pass and save/load behavior. Tests do not assert model-quality
benchmarks.

## Limitations and future work

- Only FD001 is supported by the maintained loader and CLI; FD002–FD004 are
  not currently trained or evaluated through this workflow.
- The CLI evaluates on the FD001 test labels after training. For model
  selection, a separate validation protocol and final untouched test
  evaluation should be maintained.
- Training can vary across hardware and TensorFlow execution environments
  even when seeds are set.
- The original notebook remains exploratory and includes Kaggle-specific
  paths and historical results; the package CLI is the maintained entry
  point.
- The repository has no selected software license. Add one only after
  confirming rights to distribute the source, data, and included material.

Potential next steps include adding support for the other C-MAPSS subsets,
recording repeat-run variability, adding a separate inference command that
loads the saved model and scaler metadata, and documenting a verified
reproducible experiment protocol.

## License and attribution

No project software license has been selected. Dataset redistribution rights
and third-party document terms should be reviewed before publishing or
reusing those files. Cite A. Saxena, K. Goebel, D. Simon, and N. Eklund,
“Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,”
Proceedings of the 1st International Conference on Prognostics and Health
Management (PHM08), Denver, Colorado, October 2008.
