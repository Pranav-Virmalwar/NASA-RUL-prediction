"""C-MAPSS FD001 loading, validation, scaling, and sequence construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_COLUMNS, FEATURE_COLUMNS, TrainingConfig


@dataclass
class PreparedData:
    """Model-ready sequences and metadata for one FD001 training run."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    true_rul: np.ndarray
    test_windows_per_engine: list[int]
    scaler: StandardScaler
    training_engine_ids: np.ndarray
    validation_engine_ids: np.ndarray


def _read_sensor_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required C-MAPSS file not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(DATA_COLUMNS):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns; expected {len(DATA_COLUMNS)}"
        )
    frame.columns = DATA_COLUMNS
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} is empty or contains non-finite values")
    _validate_cycles(frame, str(path))
    return frame


def _validate_cycles(frame: pd.DataFrame, label: str) -> None:
    unit_ids = frame["unit_ID"].to_numpy(dtype=float)
    cycles = frame["cycles"].to_numpy(dtype=float)
    if np.any(unit_ids <= 0) or np.any(unit_ids != np.floor(unit_ids)):
        raise ValueError(f"{label} contains invalid engine IDs")
    if np.any(cycles <= 0) or np.any(cycles != np.floor(cycles)):
        raise ValueError(f"{label} contains invalid cycle numbers")
    for unit_id, group in frame.groupby("unit_ID", sort=False):
        values = group["cycles"].to_numpy()
        if np.any(np.diff(values) <= 0):
            raise ValueError(
                f"{label} has non-increasing cycle numbers for engine {int(unit_id)}"
            )


def load_fd001(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load and validate the FD001 train, test, and test-RUL files."""
    data_dir = Path(data_dir)
    train = _read_sensor_file(data_dir / "train_FD001.txt")
    test = _read_sensor_file(data_dir / "test_FD001.txt")
    rul_path = data_dir / "RUL_FD001.txt"
    if not rul_path.is_file():
        raise FileNotFoundError(f"Required C-MAPSS file not found: {rul_path}")

    rul_frame = pd.read_csv(rul_path, sep=r"\s+", header=None)
    if rul_frame.shape[1] != 1 or rul_frame.empty:
        raise ValueError(f"{rul_path} must contain exactly one RUL value per row")
    true_rul = rul_frame.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(true_rul).all() or np.any(true_rul < 0):
        raise ValueError(f"{rul_path} contains invalid RUL values")

    test_engine_ids = np.sort(test["unit_ID"].unique().astype(int))
    if len(test_engine_ids) != len(true_rul):
        raise ValueError(
            f"{rul_path} contains {len(true_rul)} labels for "
            f"{len(test_engine_ids)} test engines"
        )
    expected_ids = np.arange(1, len(test_engine_ids) + 1)
    if not np.array_equal(test_engine_ids, expected_ids):
        raise ValueError("FD001 test engine IDs must be consecutive and start at 1")
    return train, test, true_rul


def process_targets(data_length: int, early_rul_cap: Optional[int] = None) -> np.ndarray:
    """Create cycle-level RUL labels, optionally capped at the configured value."""
    if data_length < 1:
        raise ValueError("data_length must be at least 1")
    if early_rul_cap is not None and early_rul_cap < 1:
        raise ValueError("early_rul_cap must be at least 1")
    targets = np.arange(data_length - 1, -1, -1, dtype=float)
    if early_rul_cap is not None:
        np.minimum(targets, early_rul_cap, out=targets)
    return targets


def make_windows(
    input_data: np.ndarray,
    window_length: int,
    shift: int = 1,
    target_data: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert a single engine's ordered data into fixed-size sliding windows."""
    values = np.asarray(input_data, dtype=float)
    if values.ndim != 2:
        raise ValueError("input_data must be a two-dimensional array")
    if window_length < 1 or shift < 1:
        raise ValueError("window_length and shift must be at least 1")
    if values.shape[0] < window_length:
        raise ValueError(
            f"sequence has {values.shape[0]} rows; window_length is {window_length}"
        )
    if not np.isfinite(values).all():
        raise ValueError("input_data contains non-finite values")

    targets = None if target_data is None else np.asarray(target_data, dtype=float)
    if targets is not None:
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("target_data must be one-dimensional and match input_data")
        if not np.isfinite(targets).all():
            raise ValueError("target_data contains non-finite values")

    starts = range(0, len(values) - window_length + 1, shift)
    windows = np.stack([values[start : start + window_length] for start in starts])
    labels = None
    if targets is not None:
        labels = np.asarray(
            [targets[start + window_length - 1] for start in range(
                0, len(values) - window_length + 1, shift
            )],
            dtype=float,
        )
    return windows, labels


def prepare_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    true_rul: np.ndarray,
    config: TrainingConfig,
) -> PreparedData:
    """Split by engine, fit scaling on training engines, then construct windows."""
    _validate_frame(train, "training data")
    _validate_frame(test, "test data")
    if not set(DATA_COLUMNS).issubset(train.columns) or not set(DATA_COLUMNS).issubset(
        test.columns
    ):
        raise ValueError("train and test must contain the complete C-MAPSS schema")
    if not set(FEATURE_COLUMNS).issubset(train.columns):
        raise ValueError("training data is missing one or more model features")

    labels = np.asarray(true_rul, dtype=float)
    test_engine_ids = np.sort(test["unit_ID"].unique().astype(int))
    if labels.ndim != 1 or len(labels) != len(test_engine_ids):
        raise ValueError("true_rul must have one value per test engine")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("true_rul contains invalid values")

    engine_ids = np.sort(train["unit_ID"].unique().astype(int))
    if len(engine_ids) < 2:
        raise ValueError("at least two training engines are required for validation")
    rng = np.random.default_rng(config.random_seed)
    shuffled_ids = rng.permutation(engine_ids)
    validation_count = max(1, int(np.ceil(len(engine_ids) * config.validation_fraction)))
    if validation_count >= len(engine_ids):
        raise ValueError("validation split leaves no training engines")
    validation_ids = np.sort(shuffled_ids[:validation_count])
    training_ids = np.sort(shuffled_ids[validation_count:])

    scaler = StandardScaler()
    scaler.fit(train.loc[train["unit_ID"].isin(training_ids), FEATURE_COLUMNS])
    scaled_train = train.astype({column: float for column in FEATURE_COLUMNS}).copy()
    scaled_test = test.astype({column: float for column in FEATURE_COLUMNS}).copy()
    scaled_train.loc[:, FEATURE_COLUMNS] = scaler.transform(train.loc[:, FEATURE_COLUMNS])
    scaled_test.loc[:, FEATURE_COLUMNS] = scaler.transform(test.loc[:, FEATURE_COLUMNS])

    train_x, train_y = _windows_for_engines(scaled_train, training_ids, config)
    validation_x, validation_y = _windows_for_engines(
        scaled_train, validation_ids, config
    )
    test_x: list[np.ndarray] = []
    test_windows_per_engine: list[int] = []
    for engine_id in test_engine_ids:
        engine = scaled_test.loc[scaled_test["unit_ID"] == engine_id]
        windows, _ = make_windows(
            engine.loc[:, FEATURE_COLUMNS].to_numpy(),
            config.window_length,
            config.window_shift,
        )
        windows = windows[-config.test_windows :]
        test_x.append(windows)
        test_windows_per_engine.append(len(windows))

    return PreparedData(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=np.concatenate(test_x),
        true_rul=labels,
        test_windows_per_engine=test_windows_per_engine,
        scaler=scaler,
        training_engine_ids=training_ids,
        validation_engine_ids=validation_ids,
    )


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError(f"{label} is empty or contains non-finite numeric values")
    if "unit_ID" not in frame or "cycles" not in frame:
        raise ValueError(f"{label} must include unit_ID and cycles")
    _validate_cycles(frame, label)


def _windows_for_engines(
    frame: pd.DataFrame,
    engine_ids: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets_by_engine: list[np.ndarray] = []
    for engine_id in engine_ids:
        engine = frame.loc[frame["unit_ID"] == engine_id]
        features = engine.loc[:, FEATURE_COLUMNS].to_numpy()
        targets = process_targets(len(engine), config.early_rul_cap)
        windows, window_targets = make_windows(
            features, config.window_length, config.window_shift, targets
        )
        assert window_targets is not None
        targets_by_engine.append(window_targets)
        sequences.append(windows)
    return np.concatenate(sequences), np.concatenate(targets_by_engine)
