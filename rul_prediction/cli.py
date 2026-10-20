"""Command-line entry point for training and evaluating the FD001 model."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import FEATURE_COLUMNS, TrainingConfig
from .data import load_fd001, prepare_data
from .metrics import evaluate_predictions
from .model import create_compiled_model, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the FD001 turbofan RUL LSTM."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("CMaps"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/fd001_lstm.keras"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--early-rul-cap", type=int, default=125)
    parser.add_argument("--test-windows", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_seed=args.seed,
            window_length=args.window_length,
            early_rul_cap=args.early_rul_cap,
            test_windows=args.test_windows,
            validation_fraction=args.validation_fraction,
        )
        train, test, true_rul = load_fd001(args.data_dir)
        prepared_data = prepare_data(train, test, true_rul, config)
        np.random.seed(config.random_seed)
        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError(
                "TensorFlow is required for model training; install requirements.txt"
            ) from error
        tf.keras.utils.set_random_seed(config.random_seed)

        model = create_compiled_model(config, len(FEATURE_COLUMNS))
        train_model(model, prepared_data, config)
        predictions = model.predict(prepared_data.test_x, verbose=0).reshape(-1)
        evaluation = evaluate_predictions(
            prepared_data.true_rul,
            predictions,
            prepared_data.test_windows_per_engine,
        )

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_output)
        preprocessing_path = args.model_output.with_suffix(".preprocessing.json")
        preprocessing_path.write_text(
            json.dumps(
                {
                    "feature_columns": FEATURE_COLUMNS,
                    "training_config": asdict(config),
                    "scaler": {
                        "mean": prepared_data.scaler.mean_.tolist(),
                        "scale": prepared_data.scaler.scale_.tolist(),
                        "variance": prepared_data.scaler.var_.tolist(),
                    },
                },
                indent=2,
            )
            + "\n"
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))

    print(f"Training engines: {len(prepared_data.training_engine_ids)}")
    print(f"Validation engines: {len(prepared_data.validation_engine_ids)}")
    print(
        "RMSE (mean of up to configured recent windows): "
        f"{evaluation.mean_window_rmse:.4f}"
    )
    print(f"RMSE (last window): {evaluation.last_window_rmse:.4f}")
    print(f"NASA S-score (last window; lower is better): {evaluation.last_window_s_score:.4f}")
    print(f"Model saved to: {args.model_output}")
    print(f"Preprocessing saved to: {preprocessing_path}")
    return 0
