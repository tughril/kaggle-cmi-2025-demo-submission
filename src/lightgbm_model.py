import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class CMILightGBMModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def load_data(self, data_path):
        """Load train.csv data"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df

    def preprocess_data(self, df):
        """Preprocess data for LightGBM"""
        print("Preprocessing data...")

        # Remove non-feature columns
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
        X = df[feature_cols].copy()

        # Handle missing values
        X = X.fillna(-1)

        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(-1)

        # Create target labels
        # Binary classification: BFRB vs non-BFRB
        bfrb_behaviors = [
            "hair_pulling",
            "skin_picking",
            "scratching_skin",
            "scratching_head",
            "nail_biting",
            "lip_biting",
            "cheek_biting",
            "teeth_grinding",
        ]

        # Multi-class target (18 classes)
        y_multi = self.label_encoder.fit_transform(df["gesture"])

        # Binary target (BFRB vs non-BFRB)
        y_binary = df["gesture"].isin(bfrb_behaviors).astype(int)

        self.feature_names = X.columns.tolist()

        print(f"Feature shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y_multi))}")
        print(f"Class distribution (binary): {np.bincount(y_binary)}")
        print(f"BFRB behaviors: {bfrb_behaviors}")
        print(f"Unique gestures: {df['gesture'].unique()}")

        return X, y_multi, y_binary

    def feature_engineering(self, X):
        """Create additional features"""
        print("Creating additional features...")

        # IMU features
        imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
        if all(col in X.columns for col in imu_cols):
            X["acc_magnitude"] = np.sqrt(
                X["acc_x"] ** 2 + X["acc_y"] ** 2 + X["acc_z"] ** 2
            )
            X["rot_magnitude"] = np.sqrt(
                X["rot_w"] ** 2 + X["rot_x"] ** 2 + X["rot_y"] ** 2 + X["rot_z"] ** 2
            )

        # Temperature features
        temp_cols = [col for col in X.columns if col.startswith("thm_")]
        if temp_cols:
            X["temp_mean"] = X[temp_cols].mean(axis=1)
            X["temp_std"] = X[temp_cols].std(axis=1)

        # Distance features
        tof_cols = [col for col in X.columns if col.startswith("tof_")]
        if tof_cols:
            X["tof_mean"] = X[tof_cols].mean(axis=1)
            X["tof_std"] = X[tof_cols].std(axis=1)
            X["tof_nonzero_count"] = (X[tof_cols] > 0).sum(axis=1)

        return X

    def train_model(self, X, y, params=None):
        """Train LightGBM model"""
        if params is None:
            params = {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": 0,
            }

        params["num_class"] = len(np.unique(y))

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        # Train model
        self.model = lgb.train(params, train_data, num_boost_round=100)

        return self.model

    def evaluate_binary_f1(self, y_true_multi, y_pred_multi):
        """Calculate binary F1 score (BFRB vs non-BFRB)"""
        # Convert multi-class predictions to binary
        bfrb_classes = list(range(8))  # First 8 classes are BFRB

        y_true_binary = np.isin(y_true_multi, bfrb_classes).astype(int)
        y_pred_binary = np.isin(y_pred_multi, bfrb_classes).astype(int)

        return f1_score(y_true_binary, y_pred_binary)

    def evaluate_macro_f1(self, y_true_multi, y_pred_multi):
        """Calculate macro F1 score for 9 classes (8 BFRB + 1 non-BFRB)"""
        # Convert to 9-class problem: 8 BFRB classes + 1 combined non-BFRB class
        y_true_9class = y_true_multi.copy()
        y_pred_9class = y_pred_multi.copy()

        # Combine non-BFRB classes (8-17) into single class (8)
        y_true_9class[y_true_9class >= 8] = 8
        y_pred_9class[y_pred_9class >= 8] = 8

        return f1_score(y_true_9class, y_pred_9class, average="macro")

    def calculate_competition_score(self, y_true_multi, y_pred_multi):
        """Calculate competition score: (Binary F1 + Macro F1) / 2"""
        binary_f1 = self.evaluate_binary_f1(y_true_multi, y_pred_multi)
        macro_f1 = self.evaluate_macro_f1(y_true_multi, y_pred_multi)

        competition_score = (binary_f1 + macro_f1) / 2

        return competition_score, binary_f1, macro_f1

    def cross_validate(self, X, y, n_splits=3):
        """Perform cross-validation"""
        print(f"Performing {n_splits}-fold cross-validation...")

        # Use smaller sample for faster validation
        sample_size = min(50000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y[sample_indices]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = []
        binary_f1_scores = []
        macro_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
            print(f"Fold {fold + 1}/{n_splits}")

            X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
            y_train, y_val = y_sample[train_idx], y_sample[val_idx]

            # Train model with fewer rounds
            params = {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": 0,
                "num_class": len(np.unique(y_sample)),
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=100)

            # Predict
            y_pred = model.predict(X_val)
            y_pred_class = np.argmax(y_pred, axis=1)

            # Calculate scores
            comp_score, binary_f1, macro_f1 = self.calculate_competition_score(
                y_val, y_pred_class
            )

            scores.append(comp_score)
            binary_f1_scores.append(binary_f1)
            macro_f1_scores.append(macro_f1)

            print(f"  Competition Score: {comp_score:.4f}")
            print(f"  Binary F1: {binary_f1:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")

        print("\nCross-validation results:")
        print(
            f"Competition Score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})"
        )
        print(
            f"Binary F1: {np.mean(binary_f1_scores):.4f} (+/- {np.std(binary_f1_scores) * 2:.4f})"
        )
        print(
            f"Macro F1: {np.mean(macro_f1_scores):.4f} (+/- {np.std(macro_f1_scores) * 2:.4f})"
        )

        return scores, binary_f1_scores, macro_f1_scores

    def run_validation(self, data_path):
        """Run full validation pipeline"""
        print("=== CMI LightGBM Model Validation ===")

        # Load data
        df = self.load_data(data_path)

        # Preprocess
        X, y_multi, y_binary = self.preprocess_data(df)

        # Feature engineering
        X = self.feature_engineering(X)

        # Cross-validation
        scores, binary_f1_scores, macro_f1_scores = self.cross_validate(X, y_multi)

        # Final model training
        print("\nTraining final model...")
        final_model = self.train_model(X, y_multi)

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": final_model.feature_importance()}
        ).sort_values("importance", ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        return {
            "competition_scores": scores,
            "binary_f1_scores": binary_f1_scores,
            "macro_f1_scores": macro_f1_scores,
            "feature_importance": feature_importance,
            "model": final_model,
        }


if __name__ == "__main__":
    # Run validation
    model = CMILightGBMModel()
    results = model.run_validation("data/train.csv")
