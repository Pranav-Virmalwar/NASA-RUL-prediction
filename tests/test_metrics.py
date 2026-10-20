import numpy as np
import pytest

from rul_prediction.metrics import evaluate_predictions, nasa_s_score, rmse


def test_rmse_and_s_score_perfect_predictions() -> None:
    actual = np.array([10.0, 20.0])
    np.testing.assert_equal(rmse(actual, actual), 0.0)
    np.testing.assert_equal(nasa_s_score(actual, actual), 0.0)


def test_s_score_increases_with_error_and_validates_vector_shapes() -> None:
    actual = np.array([10.0])
    assert nasa_s_score(actual, np.array([12.0])) > nasa_s_score(actual, actual)
    with pytest.raises(ValueError, match="equally sized vectors"):
        rmse(np.array([1.0]), np.array([1.0, 2.0]))


def test_evaluation_aggregates_recent_windows_per_engine() -> None:
    result = evaluate_predictions(
        true_rul=np.array([10.0, 20.0]),
        window_predictions=np.array([8.0, 10.0, 18.0, 22.0]),
        windows_per_engine=[2, 2],
    )
    np.testing.assert_allclose(result.mean_window_predictions, [9.0, 20.0])
    np.testing.assert_allclose(result.last_window_predictions, [10.0, 22.0])
    assert result.mean_window_rmse == pytest.approx(np.sqrt(0.5))
    assert result.last_window_rmse == pytest.approx(np.sqrt(2.0))


def test_evaluation_rejects_inconsistent_window_counts() -> None:
    with pytest.raises(ValueError, match="window prediction count"):
        evaluate_predictions(np.array([10.0]), np.array([1.0, 2.0]), [1])
    with pytest.raises(ValueError, match="one value per test engine"):
        evaluate_predictions(np.array(10.0), np.array([1.0]), [1])
