import os
import pickle
from datetime import datetime

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


def aggregate_features_by_sequence(
    df: pl.DataFrame, feature_cols: list[str], include_labels: bool = True
) -> pl.DataFrame:
    """
    Aggregate features by sequence_id using statistical measures.
    Returns DataFrame with one row per sequence.
    """
    print(f"Aggregating {len(feature_cols)} features by sequence_id...")

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

    print(f"Aggregated data shape: {aggregated.shape}")
    print(f"Original sequences: {df.select('sequence_id').n_unique()}")
    print(f"Aggregated sequences: {len(aggregated)}")

    return aggregated


def do_job(
    sample_size: int | None = None, log_model: bool = False, save_model: bool = False
) -> None:
    # Load data
    print("Loading data...")
    df = pl.read_csv("data/train.csv")
    print(f"Data shape: {df.shape}")

    # Log data info
    mlflow.log_param("total_rows", df.shape[0])
    mlflow.log_param("total_features", df.shape[1])

    # Sample data for quick test
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, seed=42)
        print(f"Using sample of {sample_size} rows")

    # Prepare features
    non_feature_cols = [
        "row_id",
        "sequence_type",
        "sequence_id",
        "sequence_counter",
        "subject",
        "orientation",
        "behavior",
        "phase",
        "gesture",
    ]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Aggregate features by sequence
    df_agg = aggregate_features_by_sequence(df, feature_cols)

    # Prepare aggregated features
    agg_feature_cols = []
    for col in feature_cols:
        agg_feature_cols.extend(
            [f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max"]
        )

    X = df_agg.select(agg_feature_cols).fill_null(-1)

    # Prepare target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_agg.select("gesture").to_series())

    print(f"Aggregated features: {X.shape[1]} (from {len(feature_cols)} original)")
    print(f"Sequences: {len(df_agg)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Unique gestures: {df_agg.select('gesture').unique().to_series().to_list()}")

    # Log model parameters
    mlflow.log_param("num_features", X.shape[1])
    mlflow.log_param("num_classes", len(np.unique(y)))
    mlflow.log_param("split_strategy", "sequence_based")
    mlflow.log_param("aggregation_method", "statistical")

    # Split data by sequence (hold out entire sequences)
    unique_sequences = df_agg.select("sequence_id").unique().to_series().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_sequences)

    split_idx = int(len(unique_sequences) * 0.8)
    train_sequences = unique_sequences[:split_idx]
    test_sequences = unique_sequences[split_idx:]

    train_mask = df_agg.select("sequence_id").to_series().is_in(train_sequences)
    test_mask = df_agg.select("sequence_id").to_series().is_in(test_sequences)

    X_train = X.filter(train_mask)
    X_test = X.filter(test_mask)
    y_train = y[train_mask]
    y_test = y[test_mask]

    mlflow.log_param("train_sequences", len(train_sequences))
    mlflow.log_param("test_sequences", len(test_sequences))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # Train model
    print("Training LightGBM model...")
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(np.unique(y)),
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }

    # Log hyperparameters
    mlflow.log_params(params)
    mlflow.log_param("num_boost_round", 100)

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Log model
    if log_model:
        mlflow.lightgbm.log_model(model, "model")

    # Define BFRB behaviors for evaluation and model saving
    bfrb_behaviors = [
        "Forehead - pull hairline",
        "Eyelash - pull hair",
        "Above ear - pull hair",
        "Eyebrow - pull hair",
        "Cheek - pinch skin",
        "Neck - pinch skin",
        "Pinch knee/leg skin",
        "Neck - scratch",
        "Forehead - scratch",
        "Scratch knee/leg skin",
    ]

    # Save model for Kaggle submission
    if save_model:
        os.makedirs("models", exist_ok=True)

        # Save LightGBM model
        model_path = "models/lightgbm_sequence_model.txt"
        model.save_model(model_path)
        print(f"Model saved to: {model_path}")

        # Save label encoder and feature information
        metadata = {
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
            "agg_feature_cols": agg_feature_cols,
            "bfrb_behaviors": bfrb_behaviors,
            "non_feature_cols": non_feature_cols,
        }

        metadata_path = "models/model_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Model metadata saved to: {metadata_path}")

        # Save preprocessing function code (for reference)
        with open("models/preprocessing_info.txt", "w") as f:
            f.write("# Preprocessing steps for Kaggle submission:\n")
            f.write("# 1. Load test data\n")
            f.write(
                "# 2. Aggregate features by sequence_id using mean, std, min, max\n"
            )
            f.write("# 3. Apply same feature selection and fill_null(-1)\n")
            f.write("# 4. Predict and return one prediction per sequence_id\n")
            f.write(f"# Original features: {len(feature_cols)}\n")
            f.write(f"# Aggregated features: {len(agg_feature_cols)}\n")
            f.write(f"# Classes: {len(label_encoder.classes_)}\n")
        print("Preprocessing info saved to: models/preprocessing_info.txt")

    # Validate test data format and aggregate by sequence
    print("\n=== Test Data Validation ===")
    test_df = pl.read_csv("data/test.csv")
    print(f"Test data shape: {test_df.shape}")

    # Aggregate test data by sequence
    test_df_agg = aggregate_features_by_sequence(
        test_df, feature_cols, include_labels=False
    )
    test_X_agg = test_df_agg.select(agg_feature_cols).fill_null(-1)

    print(f"Aggregated test data shape: {test_X_agg.shape}")

    # Test model prediction on aggregated test data
    print("\nTesting model on aggregated test data...")
    try:
        test_pred_proba = model.predict(test_X_agg)
        test_pred = np.argmax(test_pred_proba, axis=1)
        print(f"Test predictions shape: {test_pred.shape}")
        print(f"Test predictions sample: {test_pred[:10]}")
        print(f"Unique test sequences: {len(test_df_agg)}")
    except Exception as e:
        print(f"Error predicting on test data: {e}")

    # Predict
    print("Making predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate BFRB vs non-BFRB binary F1

    # Get BFRB class indices
    bfrb_indices = []
    for behavior in bfrb_behaviors:
        if behavior in label_encoder.classes_:
            bfrb_indices.append(label_encoder.transform([behavior])[0])

    # Binary classification: BFRB vs non-BFRB
    y_test_binary = np.isin(y_test, bfrb_indices).astype(int)
    y_pred_binary = np.isin(y_pred, bfrb_indices).astype(int)

    binary_f1 = f1_score(y_test_binary, y_pred_binary)

    # 9-class F1 (8 BFRB + 1 non-BFRB)
    y_test_9class = y_test.copy()
    y_pred_9class = y_pred.copy()

    # Map non-BFRB classes to a single class
    non_bfrb_class = max(bfrb_indices) + 1 if bfrb_indices else 8
    y_test_9class[~np.isin(y_test_9class, bfrb_indices)] = non_bfrb_class
    y_pred_9class[~np.isin(y_pred_9class, bfrb_indices)] = non_bfrb_class

    macro_f1 = f1_score(y_test_9class, y_pred_9class, average="macro")

    # Competition score
    competition_score = (binary_f1 + macro_f1) / 2

    # Log metrics
    mlflow.log_metric("binary_f1", binary_f1)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("competition_score", competition_score)

    print("\n=== Results ===")
    print(f"Binary F1 (BFRB vs non-BFRB): {binary_f1:.4f}")
    print(f"Macro F1 (9 classes): {macro_f1:.4f}")
    print(f"Competition Score: {competition_score:.4f}")

    # Feature importance (using aggregated feature names)
    feature_importance = pl.DataFrame(
        {"feature": agg_feature_cols, "importance": model.feature_importance()}
    ).sort("importance", descending=True)

    # Log feature importance
    for row in feature_importance.head(10).iter_rows(named=True):
        mlflow.log_metric(f"feature_importance_{row['feature']}", row["importance"])

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def main():
    print("=== CMI LightGBM Simple Model ===")

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"first-implementation-{now}"

    os.environ["MLFLOW_DISABLED"] = "true"
    mlflow.set_experiment("lightgbm")
    with mlflow.start_run(run_name=run_name):
        do_job(save_model=True)


if __name__ == "__main__":
    main()
