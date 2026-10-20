"""TensorFlow model definition and training loop."""

from typing import Any

from .config import TrainingConfig
from .data import PreparedData


def create_compiled_model(config: TrainingConfig, feature_count: int) -> Any:
    """Build the three-layer LSTM architecture used by the original notebook."""
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for model training; install requirements.txt"
        ) from error

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.window_length, feature_count)),
            tf.keras.layers.LSTM(
                config.lstm_units[0], return_sequences=True, activation="tanh"
            ),
            tf.keras.layers.LSTM(
                config.lstm_units[1], return_sequences=True, activation="tanh"
            ),
            tf.keras.layers.LSTM(config.lstm_units[2], activation="tanh"),
            tf.keras.layers.Dense(config.dense_units[0], activation="relu"),
            tf.keras.layers.Dense(config.dense_units[1], activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    )
    return model


def train_model(
    model: Any, prepared_data: PreparedData, config: TrainingConfig
) -> Any:
    """Fit the model with the notebook's batch size and learning-rate schedule."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for model training; install requirements.txt"
        ) from error

    def scheduler(epoch: int, current_learning_rate: float) -> float:
        del current_learning_rate
        if epoch < config.learning_rate_drop_epoch:
            return config.learning_rate
        return config.reduced_learning_rate

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    return model.fit(
        prepared_data.train_x,
        prepared_data.train_y,
        epochs=config.epochs,
        validation_data=(prepared_data.validation_x, prepared_data.validation_y),
        callbacks=[callback],
        batch_size=config.batch_size,
        verbose=2,
        shuffle=True,
    )
