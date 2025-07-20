import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import kaggle_evaluation.cmi_inference_server


def aggregate_features_by_sequence(
    df: pl.DataFrame, feature_cols: list[str], include_labels: bool = True
) -> pl.DataFrame:
    """
    Aggregate features by sequence_id using statistical measures.
    Returns DataFrame with one row per sequence.
    """
    # Group by sequence_id and calculate statistics
    agg_exprs = []
    for col in feature_cols:
        agg_exprs.extend(
            [
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).max().alias(f"{col}_max"),
            ]
        )

    # Include labels and metadata only if they exist (for training data)
    if include_labels and "gesture" in df.columns:
        agg_exprs.extend(
            [
                pl.col("gesture").first().alias("gesture"),
                pl.col("subject").first().alias("subject"),
                pl.col("sequence_type").first().alias("sequence_type"),
            ]
        )
    elif "subject" in df.columns:
        # For test data, include subject if available
        agg_exprs.append(pl.col("subject").first().alias("subject"))

    aggregated = df.group_by("sequence_id").agg(agg_exprs)
    return aggregated


# Model configuration embedded in code
MODEL_PATH = Path("models/lightgbm_sequence_model.txt")

# Gesture class mapping (same order as training)
GESTURE_CLASSES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Drink from bottle/cup",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Feel around in tray and pull out an object",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Glasses on/off",
    "Neck - pinch skin",
    "Neck - scratch",
    "Pinch knee/leg skin",
    "Pull air toward your face",
    "Scratch knee/leg skin",
    "Text on phone",
    "Wave hello",
    "Write name in air",
    "Write name on leg",
]

# Original feature columns (332 features)
FEATURE_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
    "thm_1",
    "thm_2",
    "thm_3",
    "thm_4",
    "thm_5",
] + [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

# Aggregated feature columns (1328 features = 332 * 4 statistics)
AGG_FEATURE_COLS = []
for col in FEATURE_COLS:
    AGG_FEATURE_COLS.extend([f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max"])

# Global variables to store loaded model
model = None


def load_model():
    """Load the trained model."""
    global model

    if model is None:
        print("Loading LightGBM model...")
        model = lgb.Booster(model_file=str(MODEL_PATH))
        print(f"Model loaded with {len(AGG_FEATURE_COLS)} features")
        print(f"Classes: {len(GESTURE_CLASSES)}")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predict gesture for a single sequence.
    Returns the predicted gesture name.
    """
    global model

    # Load model on first call
    if model is None:
        load_model()

    # Aggregate features by sequence
    df_agg = aggregate_features_by_sequence(
        sequence, FEATURE_COLS, include_labels=False
    )

    # Prepare features (same preprocessing as training)
    X = df_agg.select(AGG_FEATURE_COLS).fill_null(-1)

    # Predict
    pred_proba = model.predict(X.to_numpy())
    pred_class = int(
        np.argmax(pred_proba, axis=1)[0]
    )  # Get first prediction and convert to int

    # Convert class index back to gesture name
    predicted_gesture = GESTURE_CLASSES[pred_class]

    # Ensure return value is a string
    result = str(predicted_gesture)

    return result


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "./data/test.csv",
            "./data/test_demographics.csv",
        )
    )
