# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
# plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")


def load_data():
    """データを読み込む"""
    train_df = pd.read_csv("../../data/train.csv")
    train_demo_df = pd.read_csv("../../data/train_demographics.csv")
    return train_df, train_demo_df


def basic_info(df):
    """データの基本情報を表示"""
    print("=== データ基本情報 ===")
    print(f"データ形状: {df.shape}")
    print(f"シーケンス数: {df['sequence_id'].nunique()}")
    print(f"被験者数: {df['subject'].nunique()}")
    print("\n=== ジェスチャー分布 ===")
    print(df["gesture"].value_counts())
    print("\n=== 体勢分布 ===")
    print(df["orientation"].value_counts())


def plot_gesture_distribution(df):
    """ジェスチャーの分布を可視化"""
    plt.figure(figsize=(12, 8))

    # ジェスチャー分布
    plt.subplot(2, 2, 1)
    gesture_counts = df.groupby("sequence_id")["gesture"].first().value_counts()
    plt.pie(gesture_counts.values, labels=gesture_counts.index, autopct="%1.1f%%")
    plt.title("ジェスチャー分布")

    # Target vs Non-target
    plt.subplot(2, 2, 2)
    seq_type_counts = df.groupby("sequence_id")["sequence_type"].first().value_counts()
    plt.bar(seq_type_counts.index, seq_type_counts.values)
    plt.title("Target vs Non-target分布")
    plt.ylabel("シーケンス数")

    # 体勢分布
    plt.subplot(2, 2, 3)
    orientation_counts = df.groupby("sequence_id")["orientation"].first().value_counts()
    plt.barh(range(len(orientation_counts)), orientation_counts.values)
    plt.yticks(range(len(orientation_counts)), orientation_counts.index)
    plt.title("体勢分布")
    plt.xlabel("シーケンス数")

    # 行動フェーズ分布
    plt.subplot(2, 2, 4)
    behavior_counts = df["behavior"].value_counts()
    plt.bar(behavior_counts.index, behavior_counts.values)
    plt.title("行動フェーズ分布")
    plt.ylabel("データポイント数")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("gesture_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensor_overview(df):
    """センサーデータの概要を可視化"""
    # 加速度センサーの列を取得
    acc_cols = [col for col in df.columns if col.startswith("acc_")]
    rot_cols = [col for col in df.columns if col.startswith("rot_")]
    thm_cols = [col for col in df.columns if col.startswith("thm_")]

    plt.figure(figsize=(15, 10))

    # 加速度データ
    plt.subplot(2, 3, 1)
    for col in acc_cols:
        plt.plot(df[col].iloc[:1000], alpha=0.7, label=col)
    plt.title("加速度センサー (最初の1000ポイント)")
    plt.xlabel("時間")
    plt.ylabel("加速度 (m/s²)")
    plt.legend()

    # 回転データ
    plt.subplot(2, 3, 2)
    for col in rot_cols:
        plt.plot(df[col].iloc[:1000], alpha=0.7, label=col)
    plt.title("回転センサー (最初の1000ポイント)")
    plt.xlabel("時間")
    plt.ylabel("回転")
    plt.legend()

    # 温度センサー
    plt.subplot(2, 3, 3)
    for col in thm_cols:
        plt.plot(df[col].iloc[:1000], alpha=0.7, label=col)
    plt.title("温度センサー (最初の1000ポイント)")
    plt.xlabel("時間")
    plt.ylabel("温度 (°C)")
    plt.legend()

    # センサーデータの欠損値確認
    plt.subplot(2, 3, 4)
    missing_data = df.isnull().sum()
    missing_sensors = missing_data[missing_data > 0]
    if len(missing_sensors) > 0:
        plt.bar(range(len(missing_sensors)), missing_sensors.values)
        plt.xticks(range(len(missing_sensors)), missing_sensors.index, rotation=90)
        plt.title("センサーデータ欠損値")
        plt.ylabel("欠損値数")
    else:
        plt.text(0.5, 0.5, "欠損値なし", ha="center", va="center")
        plt.title("センサーデータ欠損値")

    # 加速度の相関行列
    plt.subplot(2, 3, 5)
    corr_matrix = df[acc_cols + rot_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("IMUセンサー相関")

    # 温度センサーの相関行列
    plt.subplot(2, 3, 6)
    thm_corr = df[thm_cols].corr()
    sns.heatmap(thm_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("温度センサー相関")

    plt.tight_layout()
    plt.savefig("sensor_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sequence_example(df, sequence_id=None):
    """特定のシーケンスの詳細な時系列プロット"""
    if sequence_id is None:
        # ランダムにシーケンスを選択
        sequence_id = df["sequence_id"].iloc[0]

    seq_data = df[df["sequence_id"] == sequence_id].copy()
    seq_data = seq_data.sort_values("sequence_counter")

    gesture = seq_data["gesture"].iloc[0]
    orientation = seq_data["orientation"].iloc[0]

    plt.figure(figsize=(15, 12))

    # 加速度データ
    plt.subplot(4, 1, 1)
    acc_cols = [col for col in df.columns if col.startswith("acc_")]
    for col in acc_cols:
        plt.plot(
            seq_data["sequence_counter"],
            seq_data[col],
            label=col,
            marker="o",
            markersize=2,
        )
    plt.title(f"加速度データ - {gesture} ({orientation})")
    plt.xlabel("シーケンス内時間")
    plt.ylabel("加速度 (m/s²)")
    plt.legend()
    plt.grid(True)

    # 回転データ
    plt.subplot(4, 1, 2)
    rot_cols = [col for col in df.columns if col.startswith("rot_")]
    for col in rot_cols:
        plt.plot(
            seq_data["sequence_counter"],
            seq_data[col],
            label=col,
            marker="o",
            markersize=2,
        )
    plt.title("回転データ")
    plt.xlabel("シーケンス内時間")
    plt.ylabel("回転")
    plt.legend()
    plt.grid(True)

    # 温度データ
    plt.subplot(4, 1, 3)
    thm_cols = [col for col in df.columns if col.startswith("thm_")]
    for col in thm_cols:
        plt.plot(
            seq_data["sequence_counter"],
            seq_data[col],
            label=col,
            marker="o",
            markersize=2,
        )
    plt.title("温度データ")
    plt.xlabel("シーケンス内時間")
    plt.ylabel("温度 (°C)")
    plt.legend()
    plt.grid(True)

    # 行動フェーズのハイライト
    plt.subplot(4, 1, 4)
    behaviors = seq_data["behavior"].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(behaviors)))

    for i, behavior in enumerate(behaviors):
        behavior_mask = seq_data["behavior"] == behavior
        if behavior_mask.any():
            indices = seq_data[behavior_mask]["sequence_counter"]
            plt.scatter(
                indices,
                [i] * len(indices),
                c=[colors[i]],
                label=behavior,
                s=50,
                alpha=0.7,
            )

    plt.title("行動フェーズ")
    plt.xlabel("シーケンス内時間")
    plt.ylabel("フェーズ")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"sequence_example_{sequence_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

    return sequence_id


def analyze_gesture_differences(df):
    """ジェスチャー間の違いを分析"""
    acc_cols = [col for col in df.columns if col.startswith("acc_")]

    # ジェスチャーごとの加速度統計
    gesture_stats = []
    for gesture in df["gesture"].unique():
        if pd.isna(gesture):
            continue
        gesture_data = df[df["gesture"] == gesture]

        stats = {
            "gesture": gesture,
            "sequence_type": gesture_data["sequence_type"].iloc[0],
            "count": len(gesture_data),
            "avg_acc_magnitude": np.sqrt(
                (gesture_data[acc_cols] ** 2).sum(axis=1)
            ).mean(),
        }

        for col in acc_cols:
            stats[f"{col}_mean"] = gesture_data[col].mean()
            stats[f"{col}_std"] = gesture_data[col].std()

        gesture_stats.append(stats)

    stats_df = pd.DataFrame(gesture_stats)

    plt.figure(figsize=(15, 10))

    # 加速度の大きさの比較
    plt.subplot(2, 2, 1)
    target_stats = stats_df[stats_df["sequence_type"] == "Target"]
    non_target_stats = stats_df[stats_df["sequence_type"] == "Non-Target"]

    plt.bar(
        range(len(target_stats)),
        target_stats["avg_acc_magnitude"],
        alpha=0.7,
        label="Target (BFRB)",
        color="red",
    )
    plt.bar(
        range(len(target_stats), len(target_stats) + len(non_target_stats)),
        non_target_stats["avg_acc_magnitude"],
        alpha=0.7,
        label="Non-Target",
        color="blue",
    )

    all_gestures = list(target_stats["gesture"]) + list(non_target_stats["gesture"])
    plt.xticks(range(len(all_gestures)), all_gestures, rotation=45)
    plt.title("ジェスチャー別平均加速度の大きさ")
    plt.ylabel("平均加速度の大きさ")
    plt.legend()

    # Target vs Non-Target の箱ひげ図
    plt.subplot(2, 2, 2)
    target_mags = []
    non_target_mags = []

    for _, row in stats_df.iterrows():
        if row["sequence_type"] == "Target":
            target_mags.append(row["avg_acc_magnitude"])
        else:
            non_target_mags.append(row["avg_acc_magnitude"])

    plt.boxplot([target_mags, non_target_mags], labels=["Target", "Non-Target"])
    plt.title("Target vs Non-Target 加速度分布")
    plt.ylabel("平均加速度の大きさ")

    # X軸加速度の分布
    plt.subplot(2, 2, 3)
    plt.scatter(
        stats_df[stats_df["sequence_type"] == "Target"]["acc_x_mean"],
        stats_df[stats_df["sequence_type"] == "Target"]["acc_x_std"],
        alpha=0.7,
        label="Target",
        color="red",
        s=60,
    )
    plt.scatter(
        stats_df[stats_df["sequence_type"] == "Non-Target"]["acc_x_mean"],
        stats_df[stats_df["sequence_type"] == "Non-Target"]["acc_x_std"],
        alpha=0.7,
        label="Non-Target",
        color="blue",
        s=60,
    )
    plt.xlabel("X軸加速度平均")
    plt.ylabel("X軸加速度標準偏差")
    plt.title("X軸加速度の平均 vs 標準偏差")
    plt.legend()

    # 統計サマリー表示
    plt.subplot(2, 2, 4)
    plt.axis("off")
    summary_text = f"""
    統計サマリー:
    
    Total Gestures: {len(stats_df)}
    Target (BFRB): {len(target_stats)}
    Non-Target: {len(non_target_stats)}
    
    平均加速度の大きさ:
    Target: {np.mean(target_mags):.3f} ± {np.std(target_mags):.3f}
    Non-Target: {np.mean(non_target_mags):.3f} ± {np.std(non_target_mags):.3f}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

    plt.tight_layout()
    plt.savefig("gesture_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    return stats_df


def main():
    """メイン実行関数"""
    print("データ可視化を開始します...")

    # データ読み込み
    train_df, train_demo_df = load_data()

    # 基本情報表示
    basic_info(train_df)

    # 可視化実行
    print("\n1. ジェスチャー分布を可視化中...")
    plot_gesture_distribution(train_df)

    print("2. センサーデータ概要を可視化中...")
    plot_sensor_overview(train_df)

    print("3. シーケンス例を可視化中...")
    sequence_id = plot_sequence_example(train_df)
    print(f"表示したシーケンス: {sequence_id}")

    print("4. ジェスチャー間の違いを分析中...")
    stats_df = analyze_gesture_differences(train_df)

    print("\n可視化完了! 生成されたファイル:")
    print("- gesture_distribution.png")
    print("- sensor_overview.png")
    print(f"- sequence_example_{sequence_id}.png")
    print("- gesture_analysis.png")

    return train_df, train_demo_df, stats_df


main()

# %%
