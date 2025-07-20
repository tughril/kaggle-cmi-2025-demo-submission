import os
import pickle
from datetime import datetime

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None, max_length=None):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length or max(len(seq) for seq in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Pad or truncate sequence to max_length
        if len(sequence) < self.max_length:
            # Pad with zeros
            padded = np.zeros((self.max_length, sequence.shape[1]))
            padded[: len(sequence)] = sequence
            sequence = padded
        else:
            # Truncate
            sequence = sequence[: self.max_length]

        sequence = torch.FloatTensor(sequence)

        if self.labels is not None:
            return sequence, torch.LongTensor([self.labels[idx]])[0]
        return sequence


class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(CNN1D, self).__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


def prepare_sequence_data(
    df: pl.DataFrame, feature_cols: list[str], include_labels: bool = True
):
    """
    Prepare data as sequences (no aggregation) for CNN training.
    Returns list of sequences, one per sequence_id.
    """
    print(f"Preparing sequence data with {len(feature_cols)} features...")

    sequences = []
    labels = []
    sequence_ids = []

    # Group by sequence_id
    for seq_id, group in df.group_by("sequence_id"):
        # Extract features and fill nulls
        seq_data = group.select(feature_cols).fill_null(-1).to_numpy()
        sequences.append(seq_data)
        sequence_ids.append(seq_id[0])

        if include_labels and "gesture" in group.columns:
            # Take the first gesture label (should be same for entire sequence)
            label = group.select("gesture").to_series()[0]
            labels.append(label)

    print(f"Prepared {len(sequences)} sequences")
    print(
        f"Sequence lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}, avg={np.mean([len(s) for s in sequences]):.1f}"
    )

    if include_labels:
        return sequences, labels, sequence_ids
    return sequences, sequence_ids


def do_job(
    sample_size: int | None = None, log_model: bool = False, save_model: bool = False
) -> None:
    # Use MPS (Metal Performance Shaders) on Apple Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df = pl.read_csv("data/train.csv")
    print(f"Data shape: {df.shape}")

    # Log data info
    mlflow.log_param("total_rows", df.shape[0])
    mlflow.log_param("total_features", df.shape[1])
    mlflow.log_param("device", str(device))

    # Sample data for quick test
    if sample_size is not None and len(df) > sample_size:
        # Sample by sequence to maintain sequence integrity
        unique_sequences = df.select("sequence_id").unique().to_series().to_list()
        sample_sequences = np.random.choice(
            unique_sequences,
            min(sample_size // 100, len(unique_sequences)),
            replace=False,
        )
        df = df.filter(pl.col("sequence_id").is_in(sample_sequences))
        print(
            f"Using sample with {len(df)} rows from {len(sample_sequences)} sequences"
        )

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

    # Prepare sequence data (no aggregation)
    sequences, labels, sequence_ids = prepare_sequence_data(df, feature_cols)

    # Prepare target encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"Features per timestep: {len(feature_cols)}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Unique gestures: {np.unique(labels)}")

    # Log model parameters
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("num_classes", len(np.unique(y)))
    mlflow.log_param("split_strategy", "sequence_based")
    mlflow.log_param("model_type", "1D_CNN")

    # Split data by sequence (hold out entire sequences)
    unique_seq_ids = list(set(sequence_ids))
    np.random.seed(42)
    np.random.shuffle(unique_seq_ids)

    split_idx = int(len(unique_seq_ids) * 0.8)
    train_seq_ids = unique_seq_ids[:split_idx]
    test_seq_ids = unique_seq_ids[split_idx:]

    # Create train/test splits
    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []

    for i, seq_id in enumerate(sequence_ids):
        if seq_id in train_seq_ids:
            train_sequences.append(sequences[i])
            train_labels.append(y[i])
        else:
            test_sequences.append(sequences[i])
            test_labels.append(y[i])

    mlflow.log_param("train_sequences", len(train_sequences))
    mlflow.log_param("test_sequences", len(test_sequences))

    # Create datasets and dataloaders
    max_length = max(len(seq) for seq in sequences)
    mlflow.log_param("max_sequence_length", max_length)

    train_dataset = SequenceDataset(train_sequences, train_labels, max_length)
    test_dataset = SequenceDataset(test_sequences, test_labels, max_length)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_classes = len(np.unique(y))
    model = CNN1D(input_dim=len(feature_cols), num_classes=num_classes).to(device)

    # Training parameters
    learning_rate = 0.001
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # Log hyperparameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("optimizer", "Adam")

    # Training loop
    print("Training CNN model...")
    model.train()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # Evaluation
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    y_test = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Define BFRB behaviors for evaluation
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

    # Calculate BFRB vs non-BFRB binary F1
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

    # Save model for Kaggle submission
    if save_model:
        os.makedirs("models", exist_ok=True)

        # Save PyTorch model
        model_path = "models/cnn1d_sequence_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": len(feature_cols),
                    "num_classes": num_classes,
                    "max_length": max_length,
                },
            },
            model_path,
        )
        print(f"Model saved to: {model_path}")

        # Save label encoder and feature information
        metadata = {
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
            "bfrb_behaviors": bfrb_behaviors,
            "non_feature_cols": non_feature_cols,
            "max_length": max_length,
            "model_type": "1D_CNN",
        }

        metadata_path = "models/cnn1d_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Model metadata saved to: {metadata_path}")

    # Test data validation
    print("\n=== Test Data Validation ===")
    test_df = pl.read_csv("data/test.csv")
    print(f"Test data shape: {test_df.shape}")

    # Prepare test sequences
    test_sequences, test_seq_ids = prepare_sequence_data(
        test_df, feature_cols, include_labels=False
    )
    test_dataset_eval = SequenceDataset(test_sequences, max_length=max_length)
    test_loader_eval = DataLoader(
        test_dataset_eval, batch_size=batch_size, shuffle=False
    )

    print(f"Test sequences: {len(test_sequences)}")

    # Test model prediction
    print("\nTesting model on test data...")
    try:
        model.eval()
        test_preds = []
        with torch.no_grad():
            for data in test_loader_eval:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_preds.extend(pred.cpu().numpy())

        print(f"Test predictions shape: {len(test_preds)}")
        print(f"Test predictions sample: {test_preds[:10]}")
    except Exception as e:
        print(f"Error predicting on test data: {e}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def main():
    print("=== CMI 1D CNN Model ===")

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"cnn1d-implementation-{now}"

    os.environ["MLFLOW_DISABLED"] = "true"
    mlflow.set_experiment("cnn1d")
    with mlflow.start_run(run_name=run_name):
        do_job(save_model=True)


if __name__ == "__main__":
    main()
