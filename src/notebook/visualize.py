# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Missing Values Visualization
#
# Analyze missing value patterns in Kaggle CMI 2025 competition data.
# Test data has 50% IMU-only samples with temperature/distance sensors as null.


# %%
def load_data():
    """Load training data"""
    train = pd.read_csv("../../data/train.csv")
    return train


# %%
def visualize_missing_values(df):
    """Visualize missing value patterns"""
    # Calculate missing value percentages
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100

    # Extract columns with missing values
    missing_data = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if len(missing_data) == 0:
        print("No missing values found")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Missing value percentages (bar chart)
    ax1 = axes[0, 0]
    missing_data.plot(kind="bar", ax=ax1, color="coral")
    ax1.set_title("Missing Value Percentages", fontsize=14)
    ax1.set_ylabel("Missing Percentage (%)")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Missing value heatmap (sampled)
    ax2 = axes[0, 1]
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df

    missing_cols = missing_data.index.tolist()
    if len(missing_cols) > 0:
        sns.heatmap(sample_df[missing_cols].isnull(), cbar=True, ax=ax2, cmap="viridis")
        ax2.set_title(
            f"Missing Value Patterns (Sample: {len(sample_df)} rows)", fontsize=14
        )

    # 3. Missing values by sensor type
    ax3 = axes[1, 0]
    sensor_missing = {}

    # IMU sensors (acc, rot)
    imu_cols = [col for col in df.columns if col.startswith(("acc_", "rot_"))]
    if imu_cols:
        sensor_missing["IMU"] = df[imu_cols].isnull().any(axis=1).sum()

    # Temperature sensors (thm)
    temp_cols = [col for col in df.columns if col.startswith("thm_")]
    if temp_cols:
        sensor_missing["Temperature"] = df[temp_cols].isnull().any(axis=1).sum()

    # Distance sensors (tof)
    tof_cols = [col for col in df.columns if col.startswith("tof_")]
    if tof_cols:
        sensor_missing["Distance (ToF)"] = df[tof_cols].isnull().any(axis=1).sum()

    if sensor_missing:
        sensors = list(sensor_missing.keys())
        counts = list(sensor_missing.values())
        ax3.bar(
            sensors, counts, color=["skyblue", "lightgreen", "salmon"][: len(sensors)]
        )
        ax3.set_title("Missing Values by Sensor Type", fontsize=14)
        ax3.set_ylabel("Rows with Missing Values")

    # 4. Missing value count distribution
    ax4 = axes[1, 1]
    missing_pattern = df.isnull().sum(axis=1)
    pattern_counts = missing_pattern.value_counts().sort_index()

    ax4.bar(pattern_counts.index, pattern_counts.values, color="lightcoral")
    ax4.set_title("Distribution of Missing Value Counts", fontsize=14)
    ax4.set_xlabel("Missing Values per Row")
    ax4.set_ylabel("Number of Rows")

    plt.tight_layout()
    plt.show()

    # Print detailed missing value information
    print("\n=== Missing Value Summary ===")
    print(f"Total rows: {len(df):,}")
    print(f"Columns with missing values: {len(missing_data)}")
    print(f"Rows with missing values: {df.isnull().any(axis=1).sum():,}")
    print(f"Complete rows: {df.dropna().shape[0]:,}")

    if len(missing_data) > 0:
        print("\nTop 10 columns with missing values:")
        for col, pct in missing_data.head(10).items():
            print(f"  {col}: {pct:.2f}% ({int(pct * len(df) / 100):,} rows)")


# %%
def analyze_missing_by_sequence(df):
    """Analyze missing value patterns by sequence"""
    if "sequence_id" not in df.columns:
        print("sequence_id column not found")
        return

    # Missing value percentage by sequence
    seq_missing = (
        df.groupby("sequence_id")
        .apply(lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns)) * 100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))

    # Top 20 sequences with most missing values
    plt.subplot(1, 2, 1)
    seq_missing.head(20).plot(kind="bar", color="orange")
    plt.title("Top 20 Sequences with Most Missing Values", fontsize=14)
    plt.xlabel("Sequence ID")
    plt.ylabel("Missing Percentage (%)")
    plt.xticks(rotation=45)

    # Distribution of missing percentages
    plt.subplot(1, 2, 2)
    plt.hist(seq_missing, bins=30, color="lightblue", alpha=0.7)
    plt.title("Distribution of Missing Percentages by Sequence", fontsize=14)
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Number of Sequences")

    plt.tight_layout()
    plt.show()

    print(f"\n=== Sequence-wise Missing Value Analysis ===")
    print(f"Total sequences: {len(seq_missing)}")
    print(f"Sequences with no missing values: {(seq_missing == 0).sum()}")
    print(f"Sequences with missing values: {(seq_missing > 0).sum()}")
    print(f"Average missing percentage: {seq_missing.mean():.2f}%")


# %%
def analyze_sensor_completeness(df):
    """Analyze sensor data completeness patterns"""
    sensor_groups = {
        "IMU": [col for col in df.columns if col.startswith(("acc_", "rot_"))],
        "Temperature": [col for col in df.columns if col.startswith("thm_")],
        "Distance": [col for col in df.columns if col.startswith("tof_")],
    }

    # Calculate completeness for each sensor group
    completeness = {}
    for sensor_type, cols in sensor_groups.items():
        if cols:
            # Check if ALL sensors in group are present (not null)
            completeness[sensor_type] = (~df[cols].isnull().any(axis=1)).sum()

    plt.figure(figsize=(10, 6))

    # Sensor completeness comparison
    plt.subplot(1, 2, 1)
    sensor_names = list(completeness.keys())
    complete_counts = list(completeness.values())
    colors = ["skyblue", "lightgreen", "salmon"][: len(sensor_names)]

    bars = plt.bar(sensor_names, complete_counts, color=colors)
    plt.title("Complete Sensor Data by Type", fontsize=14)
    plt.ylabel("Rows with Complete Data")

    # Add percentage labels on bars
    total_rows = len(df)
    for bar, count in zip(bars, complete_counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + total_rows * 0.01,
            f"{count:,}\n({count / total_rows * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    # Sensor combination patterns
    plt.subplot(1, 2, 2)

    # Create combination labels
    pattern_counts = {}
    for _, row in df.iterrows():
        pattern = []
        for sensor_type, cols in sensor_groups.items():
            if cols and not row[cols].isnull().any():
                pattern.append(sensor_type[0])  # Use first letter

        pattern_key = "".join(sorted(pattern)) if pattern else "None"
        pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

    # Plot combination patterns
    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())

    plt.bar(patterns, counts, color="lightcoral")
    plt.title("Sensor Combination Patterns", fontsize=14)
    plt.xlabel("Sensor Combinations (I=IMU, T=Temp, D=Distance)")
    plt.ylabel("Number of Rows")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print(f"\n=== Sensor Completeness Analysis ===")
    for sensor_type, count in completeness.items():
        pct = count / len(df) * 100
        print(f"{sensor_type}: {count:,} complete rows ({pct:.1f}%)")


# %%
# Main execution
if __name__ == "__main__":
    print("Loading data...")
    train_df = load_data()

    print(f"Data shape: {train_df.shape}")
    print("\nStarting missing value visualization...")

    # Basic missing value visualization
    visualize_missing_values(train_df)

    # Sequence-wise missing value analysis
    analyze_missing_by_sequence(train_df)

    # Sensor completeness analysis
    analyze_sensor_completeness(train_df)
