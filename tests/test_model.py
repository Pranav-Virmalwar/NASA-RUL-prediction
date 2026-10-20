import numpy as np
import tensorflow as tf

from rul_prediction.config import TrainingConfig
from rul_prediction.model import create_compiled_model


def test_model_input_output_and_keras_round_trip(tmp_path) -> None:
    config = TrainingConfig(
        window_length=5,
        lstm_units=(4, 3, 2),
        dense_units=(3, 2),
    )
    model = create_compiled_model(config, feature_count=14)
    values = np.zeros((2, 5, 14), dtype=np.float32)

    assert model.input_shape == (None, 5, 14)
    assert model(values).shape == (2, 1)

    model_path = tmp_path / "model.keras"
    model.save(model_path)
    loaded = tf.keras.models.load_model(model_path)
    assert loaded.input_shape == (None, 5, 14)
    np.testing.assert_allclose(loaded(values).numpy(), model(values).numpy())
