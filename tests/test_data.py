import numpy as np
import pandas as pd
import pytest

from rul_prediction.config import DATA_COLUMNS, FEATURE_COLUMNS, TrainingConfig
from rul_prediction.data import (
    load_fd001,
    make_windows,
    prepare_data,
    process_targets,
)


def _engine_frame(engine_id: int, cycles: int) -> pd.DataFrame:
    values = []
    for cycle in range(1, cycles + 1):
        row = [float(engine_id), float(cycle)]
        row.extend(float(engine_id + cycle + index) for index in range(24))
        values.append(row)
    return pd.DataFrame(values, columns=DATA_COLUMNS)


def test_process_targets_caps_early_rul() -> None:
    np.testing.assert_array_equal(
        process_targets(5, early_rul_cap=3),
        np.array([3.0, 3.0, 2.0, 1.0, 0.0]),
    )


@pytest.mark.parametrize(
    ("length", "window", "shift", "expected"),
    [(5, 3, 1, 3), (5, 3, 2, 2), (3, 3, 1, 1)],
)
def test_make_windows_counts_and_end_targets(length, window, shift, expected) -> None:
    values = np.arange(length * 2, dtype=float).reshape(length, 2)
    targets = np.arange(length, dtype=float)
    windows, labels = make_windows(values, window, shift, targets)
    assert windows.shape == (expected, window, 2)
    assert labels is not None
    np.testing.assert_array_equal(labels, [window - 1 + i * shift for i in range(expected)])


def test_make_windows_rejects_too_short_sequences() -> None:
    with pytest.raises(ValueError, match="window_length"):
        make_windows(np.ones((2, 1)), window_length=3)


def test_prepare_data_splits_by_engine_and_keeps_test_windows_aligned() -> None:
    train = pd.concat([_engine_frame(i, 7) for i in range(1, 6)], ignore_index=True)
    test = pd.concat([_engine_frame(i, 5) for i in range(1, 4)], ignore_index=True)
    config = TrainingConfig(window_length=3, test_windows=2, random_seed=7)

    prepared = prepare_data(train, test, np.array([4, 5, 6]), config)

    assert set(prepared.training_engine_ids).isdisjoint(prepared.validation_engine_ids)
    assert len(prepared.training_engine_ids) + len(prepared.validation_engine_ids) == 5
    assert prepared.train_x.shape[1:] == (3, len(FEATURE_COLUMNS))
    assert prepared.validation_x.shape[1:] == (3, len(FEATURE_COLUMNS))
    assert prepared.test_x.shape[0] == 6
    assert prepared.test_windows_per_engine == [2, 2, 2]
    np.testing.assert_array_equal(prepared.true_rul, [4, 5, 6])


def test_prepare_data_rejects_rul_count_mismatch() -> None:
    train = pd.concat([_engine_frame(i, 5) for i in range(1, 4)], ignore_index=True)
    test = _engine_frame(1, 5)
    with pytest.raises(ValueError, match="one value per test engine"):
        prepare_data(train, test, np.array([1, 2]), TrainingConfig(window_length=3))


def test_prepare_data_rejects_non_increasing_cycles() -> None:
    train = pd.concat([_engine_frame(i, 5) for i in range(1, 4)], ignore_index=True)
    train.loc[1, "cycles"] = 1
    test = _engine_frame(1, 5)
    with pytest.raises(ValueError, match="non-increasing"):
        prepare_data(train, test, np.array([1]), TrainingConfig(window_length=3))


def test_load_fd001_reads_whitespace_files_and_checks_label_count(tmp_path) -> None:
    train = pd.concat([_engine_frame(i, 3) for i in range(1, 3)], ignore_index=True)
    test = _engine_frame(1, 3)
    np.savetxt(tmp_path / "train_FD001.txt", train.to_numpy(), fmt="%.1f")
    np.savetxt(tmp_path / "test_FD001.txt", test.to_numpy(), fmt="%.1f")
    (tmp_path / "RUL_FD001.txt").write_text("7\n")

    loaded_train, loaded_test, true_rul = load_fd001(tmp_path)

    assert len(loaded_train) == 6
    assert len(loaded_test) == 3
    np.testing.assert_array_equal(true_rul, [7])

    (tmp_path / "RUL_FD001.txt").write_text("7\n8\n")
    with pytest.raises(ValueError, match="labels for 1 test engines"):
        load_fd001(tmp_path)
