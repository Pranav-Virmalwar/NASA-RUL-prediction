"""Evaluation metrics used by the notebook's RUL workflow."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Evaluation:
    """Test-set predictions and metrics for both supported window protocols."""

    mean_window_predictions: np.ndarray
    last_window_predictions: np.ndarray
    mean_window_rmse: float
    last_window_rmse: float
    last_window_s_score: float


def rmse(true_rul: np.ndarray, predicted_rul: np.ndarray) -> float:
    """Return root mean squared error after validating corresponding vectors."""
    truth, prediction = _paired_vectors(true_rul, predicted_rul)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def nasa_s_score(true_rul: np.ndarray, predicted_rul: np.ndarray) -> float:
    """Return the C-MAPSS asymmetric score; lower values indicate less penalty."""
    truth, prediction = _paired_vectors(true_rul, predicted_rul)
    difference = prediction - truth
    penalties = np.where(
        difference < 0,
        np.expm1(-difference / 13.0),
        np.expm1(difference / 10.0),
    )
    return float(np.sum(penalties))


def evaluate_predictions(
    true_rul: np.ndarray,
    window_predictions: np.ndarray,
    windows_per_engine: list[int],
) -> Evaluation:
    """Aggregate test-window predictions and calculate the notebook metrics."""
    predictions = np.asarray(window_predictions, dtype=float)
    truth = np.asarray(true_rul, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("window_predictions must be a finite one-dimensional array")
    if not windows_per_engine or any(count < 1 for count in windows_per_engine):
        raise ValueError("windows_per_engine must contain positive window counts")
    if truth.ndim != 1 or len(truth) != len(windows_per_engine):
        raise ValueError("true_rul must have one value per test engine")
    if not np.isfinite(truth).all():
        raise ValueError("true_rul must contain finite values")
    if sum(windows_per_engine) != len(predictions):
        raise ValueError("window prediction count does not match windows_per_engine")

    mean_predictions = []
    last_predictions = []
    offset = 0
    for count in windows_per_engine:
        engine_predictions = predictions[offset : offset + count]
        mean_predictions.append(float(np.mean(engine_predictions)))
        last_predictions.append(float(engine_predictions[-1]))
        offset += count

    mean_values = np.asarray(mean_predictions)
    last_values = np.asarray(last_predictions)
    return Evaluation(
        mean_window_predictions=mean_values,
        last_window_predictions=last_values,
        mean_window_rmse=rmse(truth, mean_values),
        last_window_rmse=rmse(truth, last_values),
        last_window_s_score=nasa_s_score(truth, last_values),
    )


def _paired_vectors(
    true_rul: np.ndarray, predicted_rul: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(true_rul, dtype=float)
    prediction = np.asarray(predicted_rul, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("true_rul and predicted_rul must be equally sized vectors")
    if len(truth) == 0 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("RUL vectors must be non-empty and finite")
    return truth, prediction
