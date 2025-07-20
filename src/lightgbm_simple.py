from datetime import datetime

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


def validate_test_data():
    """Validate that the model can handle test data format"""
    print("\n=== Test Data Validation ===")

    # Load test data
    test_df = pl.read_csv("data/test.csv")
    print(f"Test data shape: {test_df.shape}")

    # Check for missing columns that exist in train but not test
    train_df = pl.read_csv("data/train.csv")
    missing_cols = set(train_df.columns) - set(test_df.columns)
    print(f"Columns in train but not test: {missing_cols}")

    # Check for null values in test data
    null_counts = test_df.null_count()
    print("Null values in test data:")
    for row in null_counts.iter_rows(named=True):
        for col, count in row.items():
            if count > 0:
                print(f"{col}: {count}")

    # Check IMU-only rows (50% of test data)
    imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    temp_cols = [col for col in test_df.columns if col.startswith("thm_")]
    tof_cols = [col for col in test_df.columns if col.startswith("tof_")]

    # Check if all temp and tof columns are null for each row
    imu_only_mask = (
        test_df.select(temp_cols + tof_cols)
        .select(pl.all_horizontal(pl.all().is_null()))
        .to_series()
    )
    imu_only_count = imu_only_mask.sum()
    print(
        f"IMU-only rows: {imu_only_count}/{len(test_df)} ({imu_only_count / len(test_df) * 100:.1f}%)"
    )

    # Prepare test features using same logic as training
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
    test_feature_cols = [col for col in test_df.columns if col not in non_feature_cols]
    test_X = test_df.select(test_feature_cols).fill_null(-1)

    print(f"Test features shape: {test_X.shape}")
    print(f"Test features: {len(test_feature_cols)}")

    return test_X


def do_job(sample_size: int | None = None, log_model: bool = False) -> None:
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

    X = df.select(feature_cols).fill_null(-1)

    # Prepare target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df.select("gesture").to_series())

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Unique gestures: {df.select('gesture').unique().to_series().to_list()}")

    # Log model parameters
    mlflow.log_param("num_features", X.shape[1])
    mlflow.log_param("num_classes", len(np.unique(y)))
    mlflow.log_param("split_strategy", "sequence_based")

    # Split data by sequence (hold out entire sequences)
    unique_sequences = df.select("sequence_id").unique().to_series().to_list()
    np.random.seed(42)
    np.random.shuffle(unique_sequences)

    split_idx = int(len(unique_sequences) * 0.8)
    train_sequences = unique_sequences[:split_idx]
    test_sequences = unique_sequences[split_idx:]

    train_mask = df.select("sequence_id").to_series().is_in(train_sequences)
    test_mask = df.select("sequence_id").to_series().is_in(test_sequences)

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

    # Validate test data format
    test_X = validate_test_data()

    # Test model prediction on test data format
    print("\nTesting model on test data format...")
    try:
        test_pred_proba = model.predict(test_X)
        test_pred = np.argmax(test_pred_proba, axis=1)
        print(f"Test predictions shape: {test_pred.shape}")
        print(f"Test predictions sample: {test_pred[:10]}")
    except Exception as e:
        print(f"Error predicting on test data: {e}")

    # Predict
    print("Making predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate BFRB vs non-BFRB binary F1
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

    # Feature importance
    feature_importance = pl.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importance()}
    ).sort("importance", descending=True)

    # Log feature importance
    for i, row in enumerate(feature_importance.head(10).iter_rows(named=True)):
        mlflow.log_metric(f"feature_importance_{row['feature']}", row["importance"])

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def main():
    print("=== CMI LightGBM Simple Model ===")

    # Start MLflow run
    mlflow.set_experiment("lightgbm")

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"first-implementation-{now}"

    with mlflow.start_run(run_name=run_name):
        do_job()


if __name__ == "__main__":
    main()
