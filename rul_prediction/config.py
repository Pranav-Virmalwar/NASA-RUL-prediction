"""Configuration and dataset schema for the FD001 workflow."""

from dataclasses import dataclass


DATA_COLUMNS = (
    "unit_ID",
    "cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
)

FEATURE_COLUMNS = (
    "cycles",
    "T24",
    "T30",
    "T50",
    "Nf",
    "Nc",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "htBleed",
    "W31",
    "W32",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the notebook's FD001 LSTM workflow."""

    window_length: int = 30
    window_shift: int = 1
    early_rul_cap: int = 125
    test_windows: int = 5
    validation_fraction: float = 0.2
    random_seed: int = 83
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    reduced_learning_rate: float = 0.0001
    learning_rate_drop_epoch: int = 5
    lstm_units: tuple[int, int, int] = (128, 64, 32)
    dense_units: tuple[int, int] = (96, 128)

    def __post_init__(self) -> None:
        if self.window_length < 1:
            raise ValueError("window_length must be at least 1")
        if self.window_shift < 1:
            raise ValueError("window_shift must be at least 1")
        if self.early_rul_cap < 1:
            raise ValueError("early_rul_cap must be at least 1")
        if self.test_windows < 1:
            raise ValueError("test_windows must be at least 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")
        if self.learning_rate <= 0 or self.reduced_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if len(self.lstm_units) != 3 or any(units < 1 for units in self.lstm_units):
            raise ValueError("lstm_units must contain three positive layer sizes")
        if len(self.dense_units) != 2 or any(units < 1 for units in self.dense_units):
            raise ValueError("dense_units must contain two positive layer sizes")
