# データセット説明

## 概要

このデータセットは、Helios wrist-worn deviceで収集されたセンサーデータを使用して、body-focused repetitive behaviors (BFRBs) と通常のジェスチャーを分類するためのものです。

## データファイル構成

### 1. train.csv / test.csv
メインのセンサーデータファイル（時系列データ）

#### 基本情報
- **データ構造**: 時系列センサーデータ（行ごとに1つの時点）
- **シーケンス構成**: Transition → Pause → Gesture の3段階（詳細は下記）
- **ジェスチャー種類**: 8種類のBFRB-like + 10種類のnon-BFRB-like
- **体勢**: 座位、前傾座位、仰臥位、側臥位の4種類

#### シーケンスの3段階構成

各実験シーケンスは以下の3つのフェーズで構成されています：

**1. Transition（移行期）**
- **内容**: 安静状態から目標位置への手の移動
- **例**: 膝の上にあった手を頬に向けて動かす
- **特徴**: 大きな動作、加速度・回転の変化が顕著
- **データ活用**: 動作開始の検出、個人の動作パターン分析

**2. Pause（一時停止）**
- **内容**: 目標位置で何もしない短い静止期間
- **例**: 手が頬の近くで静止している状態
- **特徴**: 比較的安定、センサー値の変動が小さい
- **データ活用**: ベースライン値の取得、個人差の正規化に重要

**3. Gesture（ジェスチャー実行）**
- **内容**: 実際のBFRB-likeまたはnon-BFRB-likeジェスチャーの実行
- **例**: 頬の皮膚をつまむ、スマホでテキスト入力する
- **特徴**: ジェスチャー固有のパターン、分類の主要対象
- **データ活用**: メインの分類ターゲット、特徴抽出の中心

#### 実験の流れの具体例

**例: "Cheek - Pinch skin"（頬の皮膚をつまむ）の場合**

1. **Transition**: 膝の上の手 → 頬の近くまで移動（2-3秒）
2. **Pause**: 頬の近くで静止（1-2秒）
3. **Gesture**: 頬の皮膚をつまむ動作を実行（2-3秒）

**センサーデータの変化パターン**
- **Transition**: acc値が大きく変化、rot値も変動
- **Pause**: 全センサー値が比較的安定
- **Gesture**: ジェスチャー特有の細かい動作パターンが出現

## データからのシーケンス取得方法

### 基本的なシーケンス抽出

```python
import pandas as pd

# データの読み込み
df = pd.read_csv('train.csv')

# 特定のシーケンスIDのデータを取得
sequence_id = 'SEQ_000001'
sequence_data = df[df['sequence_id'] == sequence_id].copy()

# sequence_counterで時系列順にソート
sequence_data = sequence_data.sort_values('sequence_counter')

print(f"シーケンス長: {len(sequence_data)}")
print(f"フェーズ構成: {sequence_data['behavior'].unique()}")
```

### フェーズ別データの分離

```python
# 各フェーズのデータを分離
transition_data = sequence_data[sequence_data['behavior'] == 'Transition']
pause_data = sequence_data[sequence_data['behavior'] == 'Pause']
gesture_data = sequence_data[sequence_data['behavior'] == 'Gesture']

print(f"Transition: {len(transition_data)}行")
print(f"Pause: {len(pause_data)}行")
print(f"Gesture: {len(gesture_data)}行")

# ジェスチャーラベルの取得（train.csvのみ）
gesture_label = sequence_data['gesture'].iloc[0]
sequence_type = sequence_data['sequence_type'].iloc[0]
```

### センサーデータの抽出

```python
# センサーデータのカラム名を定義
sensor_columns = {
    'imu_acc': ['acc_x', 'acc_y', 'acc_z'],
    'imu_rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
    'thermopile': [f'thm_{i}' for i in range(1, 6)],
    'tof': [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
}

# 各センサーデータを抽出
imu_acc = sequence_data[sensor_columns['imu_acc']].values
imu_rot = sequence_data[sensor_columns['imu_rot']].values
thermopile = sequence_data[sensor_columns['thermopile']].values
tof = sequence_data[sensor_columns['tof']].values

print(f"IMU加速度: {imu_acc.shape}")  # (時系列長, 3)
print(f"IMU回転: {imu_rot.shape}")    # (時系列長, 4)
print(f"温度: {thermopile.shape}")    # (時系列長, 5)
print(f"距離: {tof.shape}")           # (時系列長, 320)
```

### 全シーケンスの一括処理

```python
def extract_all_sequences(df):
    """全シーケンスを辞書形式で取得"""
    sequences = {}
    
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].sort_values('sequence_counter')
        
        sequences[seq_id] = {
            'data': seq_data,
            'gesture': seq_data['gesture'].iloc[0] if 'gesture' in seq_data.columns else None,
            'sequence_type': seq_data['sequence_type'].iloc[0] if 'sequence_type' in seq_data.columns else None,
            'subject': seq_data['subject'].iloc[0],
            'orientation': seq_data['orientation'].iloc[0] if 'orientation' in seq_data.columns else None,
            'length': len(seq_data),
            'phases': {
                'transition': seq_data[seq_data['behavior'] == 'Transition'],
                'pause': seq_data[seq_data['behavior'] == 'Pause'],
                'gesture': seq_data[seq_data['behavior'] == 'Gesture']
            }
        }
    
    return sequences

# 使用例
sequences = extract_all_sequences(df)
print(f"総シーケンス数: {len(sequences)}")
```

### 機械学習用データセット作成

```python
def create_ml_dataset(sequences, target_phase='gesture'):
    """機械学習用のデータセットを作成"""
    X = []
    y = []
    sequence_ids = []
    
    for seq_id, seq_info in sequences.items():
        if seq_info['gesture'] is None:  # test.csvの場合
            continue
            
        # 特定フェーズのデータを取得
        if target_phase == 'all':
            phase_data = seq_info['data']
        else:
            phase_data = seq_info['phases'][target_phase]
        
        if len(phase_data) == 0:
            continue
            
        # センサーデータを結合
        features = []
        for col_group in sensor_columns.values():
            features.append(phase_data[col_group].values)
        
        X.append(np.concatenate(features, axis=1))
        y.append(seq_info['gesture'])
        sequence_ids.append(seq_id)
    
    return X, y, sequence_ids

# ジェスチャーフェーズのみでデータセット作成
X, y, seq_ids = create_ml_dataset(sequences, target_phase='gesture')
print(f"データセット形状: {len(X)}シーケンス")
```

### パディング・正規化の処理

```python
from sklearn.preprocessing import LabelEncoder
import numpy as np

def pad_sequences(sequences, max_length=None):
    """シーケンスを同じ長さにパディング"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            # 最後の値で後方パディング
            padding = np.repeat(seq[-1:], max_length - len(seq), axis=0)
            seq = np.concatenate([seq, padding])
        elif len(seq) > max_length:
            # 切り詰め
            seq = seq[:max_length]
        padded.append(seq)
    
    return np.array(padded)

# ラベルエンコーディング
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# パディング適用
X_padded = pad_sequences(X)
print(f"パディング後: {X_padded.shape}")  # (サンプル数, 時系列長, 特徴量数)
```

### シーケンス特徴量の抽出パターン

```python
# パターン1: ジェスチャーフェーズのみ
gesture_features = create_ml_dataset(sequences, 'gesture')

# パターン2: 全フェーズ結合
full_features = create_ml_dataset(sequences, 'all')

# パターン3: Pauseをベースラインとした差分
def create_baseline_normalized_dataset(sequences):
    X = []
    y = []
    
    for seq_id, seq_info in sequences.items():
        if seq_info['gesture'] is None:
            continue
            
        pause_data = seq_info['phases']['pause']
        gesture_data = seq_info['phases']['gesture']
        
        if len(pause_data) == 0 or len(gesture_data) == 0:
            continue
            
        # Pauseの平均値をベースラインとして使用
        baseline = pause_data[sensor_columns['imu_acc'] + sensor_columns['imu_rot']].mean()
        
        # ジェスチャーデータからベースラインを引く
        normalized = gesture_data[sensor_columns['imu_acc'] + sensor_columns['imu_rot']] - baseline
        
        X.append(normalized.values)
        y.append(seq_info['gesture'])
    
    return X, y
```

#### カラム説明

**識別子・メタデータ**
- `row_id`: 行の一意識別子
- `sequence_id`: シーケンスのバッチID（各シーケンスは1つのジェスチャー実行）
- `sequence_counter`: シーケンス内での行番号（時系列の順序）
- `subject`: 参加者の一意ID
- `behavior`: 現在のフェーズ（"Transition" / "Pause" / "Gesture"）

**ターゲット変数（train.csvのみ）**
- `gesture`: 予測対象のジェスチャー名（詳細は下記参照）
- `sequence_type`: "target"（BFRB-like）または "non-target"（non-BFRB-like）
- `orientation`: 参加者の体勢（"sitting" / "sitting_leaning_forward" / "lying_on_back" / "lying_on_side"）

**センサーデータ**

*IMU（慣性測定装置）*
- `acc_x`, `acc_y`, `acc_z`: 3軸線形加速度（m/s²）
- `rot_w`, `rot_x`, `rot_y`, `rot_z`: 4元数による3D空間での姿勢データ

*温度センサー（Thermopile）*
- `thm_1` ~ `thm_5`: 5つの非接触温度センサーの値（℃）
- 赤外線放射を検出して体温を測定

*距離センサー（Time-of-Flight）*
- `tof_[1-5]_v[0-63]`: 5つの距離センサー、各8x8=64ピクセル
- 赤外線光の反射時間で距離を測定
- 値の範囲: 0-254（未校正センサー値）
- 反射なし（近くに物体がない）場合: -1
- ピクセル配置: 左上から右へ、行ごとに右下まで

**ToFセンサーの詳細仕様**

*センサー配置*
- **5つのセンサー**: `tof_1` ～ `tof_5`
- **各センサー**: 8x8=64ピクセルの2D距離マップ
- **総データ次元**: 5 × 64 = 320次元
- **ピクセル命名**: `tof_{センサーID}_v{ピクセルID}`
  - センサーID: 1-5
  - ピクセルID: 0-63（8×8グリッドを行優先で番号付け）

*ピクセル配置パターン*
```
8x8グリッドの番号付け例（tof_1の場合）:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ v0  │ v1  │ v2  │ v3  │ v4  │ v5  │ v6  │ v7  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v8  │ v9  │ v10 │ v11 │ v12 │ v13 │ v14 │ v15 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v16 │ v17 │ v18 │ v19 │ v20 │ v21 │ v22 │ v23 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v24 │ v25 │ v26 │ v27 │ v28 │ v29 │ v30 │ v31 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v32 │ v33 │ v34 │ v35 │ v36 │ v37 │ v38 │ v39 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v40 │ v41 │ v42 │ v43 │ v44 │ v45 │ v46 │ v47 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v48 │ v49 │ v50 │ v51 │ v52 │ v53 │ v54 │ v55 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ v56 │ v57 │ v58 │ v59 │ v60 │ v61 │ v62 │ v63 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

*データ値の解釈*
- **0-254**: 未校正の距離値（単位不明、生センサー値）
- **-1**: 反射なし/無効値
  - 物体が検出範囲外
  - センサーエラー
  - 光の吸収率が高い表面
- **典型的な値分布**:
  - 近距離（手のひら等）: 50-150
  - 中距離（顔・体）: 100-200
  - 遠距離/背景: 200-254
  - 無効: -1

*センサーの物理的特性*
- **測定原理**: 赤外線レーザーの飛行時間（Time-of-Flight）測定
- **視野角**: 各センサー約25-45度（推定）
- **測定範囲**: 数cm～数mの範囲（正確な仕様不明）
- **分解能**: ミリメートル単位の距離変化を検出可能
- **フレームレート**: IMUと同期（詳細不明）

*Time-of-Flight技術の特徴と利点*

**測定原理の詳細**
- **光子の往復時間測定**: 赤外線レーザーパルスが物体に反射して戻るまでの時間を測定
- **直接距離計算**: `距離 = (光速 × 往復時間) / 2`
- **非接触測定**: 物理的接触なしで正確な距離を測定
- **実時間処理**: 高速な距離マッピングが可能

**他の距離センサーとの比較**

| 技術 | ToF | 超音波 | ステレオカメラ | LiDAR |
|------|-----|--------|---------------|-------|
| **測定精度** | ±数mm | ±数cm | 画像依存 | ±数mm |
| **測定速度** | 高速 | 低速 | 中速 | 高速 |
| **環境光影響** | 低 | なし | 高 | 低 |
| **透明物体** | 検出困難 | 検出困難 | 検出困難 | 検出困難 |
| **消費電力** | 中 | 低 | 高 | 高 |
| **サイズ** | 小型 | 中型 | 大型 | 大型 |

**ToFセンサーの独特な特性**

*利点*
- **高精度**: ミリメートル単位の距離測定
- **高速応答**: リアルタイム動作検出
- **低遅延**: 光速による瞬時測定
- **3D空間認識**: 複数ピクセルによる立体把握
- **環境ロバスト**: 照明条件に比較的影響されない
- **小型化可能**: ウェアラブルデバイスに搭載可能

*制限事項*
- **反射率依存**: 黒色・光沢面で精度低下
- **多重反射**: 複雑な形状で誤測定
- **相互干渉**: 複数ToFセンサー間の干渉
- **温度依存**: 高温環境で性能変化
- **最小距離制限**: 近すぎる物体は測定困難

**BFRBジェスチャー検出での優位性**

*手の動作検出*
- **微細動作**: 皮膚をつまむ、髪を引く等の細かい動作
- **近距離精密測定**: 手と顔の距離変化（数cm精度）
- **形状変化**: 指の動き、手の形状変化の検出
- **接触検出**: 手が体に触れる瞬間の特定

*時系列パターン解析*
- **動作軌跡**: 手の移動パスの3D追跡
- **速度プロファイル**: 動作の速さ・加速度パターン
- **周期性検出**: 反復的な動作パターン
- **接触パターン**: 接触の強さ・時間・頻度

*空間的特徴*
```python
# ToFデータから抽出可能な特徴例
def extract_gesture_features(tof_sequence):
    features = {}
    
    # 1. 距離統計
    features['min_distance'] = np.min(tof_sequence)
    features['max_distance'] = np.max(tof_sequence) 
    features['mean_distance'] = np.mean(tof_sequence)
    features['distance_range'] = features['max_distance'] - features['min_distance']
    
    # 2. 動作強度
    features['motion_intensity'] = np.std(np.diff(tof_sequence, axis=0))
    features['motion_frequency'] = count_motion_peaks(tof_sequence)
    
    # 3. 空間分布
    features['contact_area'] = count_close_pixels(tof_sequence, threshold=50)
    features['hand_shape_variance'] = analyze_hand_shape(tof_sequence)
    
    # 4. 時系列パターン
    features['approach_speed'] = calculate_approach_velocity(tof_sequence)
    features['contact_duration'] = measure_contact_time(tof_sequence)
    
    return features
```

**センサー融合での活用**

*IMU + ToF*
- **動作意図**: IMUで大まかな動き、ToFで詳細な接触
- **フェーズ検出**: IMUで移動開始、ToFで接触開始
- **誤検出削減**: 両センサーの一致性チェック

*温度 + ToF*
- **接触確認**: 温度上昇 + 距離減少 = 確実な接触
- **接触強度**: 温度変化の速度で接触の強さを推定
- **皮膚検出**: 体温パターンと距離パターンの組み合わせ

**実際のジェスチャーでの応用例**

*「頬の皮膚をつまむ」の場合*
```python
def analyze_cheek_pinch(tof_data):
    # 段階1: 手の接近（距離減少）
    approach_phase = detect_distance_decrease(tof_data)
    
    # 段階2: 皮膚接触（最小距離到達）
    contact_moment = find_minimum_distance(tof_data)
    
    # 段階3: つまみ動作（小さな距離振動）
    pinch_pattern = detect_oscillation(tof_data, contact_moment)
    
    # 段階4: 手の離脱（距離増加）
    release_phase = detect_distance_increase(tof_data)
    
    return {
        'approach_duration': len(approach_phase),
        'contact_force': estimate_pinch_strength(pinch_pattern),
        'pinch_frequency': count_pinch_cycles(pinch_pattern),
        'total_duration': len(tof_data)
    }
```

*空間配置の推定*
```python
# 5つのセンサーの配置例（推定）
sensor_layout = {
    'tof_1': '中央',     # 手首正面
    'tof_2': '左側',     # 手首左側
    'tof_3': '右側',     # 手首右側
    'tof_4': '上側',     # 手首上側
    'tof_5': '下側'      # 手首下側
}
```

*データ処理のポイント*
- **欠損値処理**: -1を適切に処理（0埋め、補間、マスキング等）
- **正規化**: 0-254の範囲を[0,1]に正規化
- **2D構造活用**: 8x8画像として畳み込み処理可能
- **センサー間融合**: 5つのセンサーからの3D空間復元
- **時系列解析**: 動作による距離変化パターンの抽出

*具体的な活用例*
```python
# ToFデータの抽出と前処理
def process_tof_data(df):
    tof_columns = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    tof_data = df[tof_columns].values
    
    # -1を0に置換
    tof_data[tof_data == -1] = 0
    
    # 5つのセンサー別に8x8画像に変形
    tof_images = tof_data.reshape(-1, 5, 8, 8)
    
    # 正規化
    tof_images = tof_images / 254.0
    
    return tof_images

# 空間特徴の抽出
def extract_spatial_features(tof_images):
    features = []
    
    for sensor_idx in range(5):
        sensor_img = tof_images[:, sensor_idx, :, :]
        
        # 中央値（手との距離）
        features.append(np.median(sensor_img, axis=(1,2)))
        
        # 分散（動作の激しさ）
        features.append(np.var(sensor_img, axis=(1,2)))
        
        # エッジ強度（形状変化）
        edge_strength = np.sum(np.abs(np.diff(sensor_img, axis=1)), axis=(1,2))
        features.append(edge_strength)
    
    return np.column_stack(features)
```

### 2. train_demographics.csv / test_demographics.csv
参加者の人口統計・身体特性データ

#### カラム説明
- `subject`: 参加者ID（メインデータとの結合キー）
- `adult_child`: 大人(1) / 子供(0)　※18歳以上が大人
- `age`: 年齢（歳）
- `sex`: 生物学的性別　女性(0) / 男性(1)
- `handedness`: 利き手　左手(0) / 右手(1)
- `height_cm`: 身長（cm）
- `shoulder_to_wrist_cm`: 肩から手首までの距離（cm）
- `elbow_to_wrist_cm`: 肘から手首までの距離（cm）

## ジェスチャー種類

### BFRB-like Gestures（ターゲット行動）
1. **Above ear - Pull hair**: 耳上の髪を引く
2. **Forehead - Pull hairline**: 額の生え際を引く
3. **Forehead - Scratch**: 額をかく
4. **Eyebrow - Pull hair**: 眉毛の毛を引く
5. **Eyelash - Pull hair**: まつ毛を引く
6. **Neck - Pinch skin**: 首の皮膚をつまむ
7. **Neck - Scratch**: 首をかく
8. **Cheek - Pinch skin**: 頬の皮膚をつまむ

### Non-BFRB-like Gestures（非ターゲット行動）
1. **Drink from bottle/cup**: ボトル・カップから飲む
2. **Glasses on/off**: メガネをかける/外す
3. **Pull air toward your face**: 顔に向けて空気を送る
4. **Pinch knee/leg skin**: 膝・脚の皮膚をつまむ
5. **Scratch knee/leg skin**: 膝・脚の皮膚をかく
6. **Write name on leg**: 脚に名前を書く
7. **Text on phone**: スマホでテキスト入力
8. **Feel around in tray and pull out an object**: トレイを探って物を取る
9. **Write name in air**: 空中に名前を書く
10. **Wave hello**: 手を振る

## データの特徴

### センサー配置
- **装着位置**: 利き手の手首
- **センサー数**: IMU 1個、温度センサー 5個、距離センサー 5個
- **データ欠損**: センサー通信障害により一部データが欠損する場合あり

### テストデータの特徴
- **サイズ**: 約3,500シーケンス
- **センサー構成**: 
  - 半分: IMUのみ（温度・距離センサーはnull値）
  - 半分: 全センサー
- **目的**: 追加センサーの価値評価

### キャリブレーションの必要性
- **個人差**: 年齢、性別、身体サイズによる大きな変動
- **体勢差**: 4種類の体勢による動作パターンの違い
- **推奨手法**: 
  - 個人ごとの正規化
  - シーケンス内安静時基準の正規化
  - 身体測定値による重み付け

## 評価指標

**最終スコア** = (Binary F1 + Macro F1) / 2

### 評価指標の詳細

#### 1. Binary F1（50%の重み）
- **対象**: BFRB-like vs Non-BFRB-like の二値分類
- **処理**: 全てのNon-BFRB-like ジェスチャー（10種類）を1つのクラスとして統合
- **意味**: BFRB検出の確実性（医学的価値）
- **重要性**: 医療診断における偽陽性・偽陰性の最小化

#### 2. Macro F1（50%の重み）
- **対象**: 8種類のBFRB-like個別分類 + 1つのnon-target統合クラス = 9クラス
- **処理**: 各クラスのF1スコアを等しく重み付けして平均
- **意味**: BFRB種別の正確性（治療方針決定）
- **重要性**: クラス不均衡に対してロバスト、少数クラスも等しく評価

### 評価の具体例

```python
# 実際のジェスチャー
true_gestures = [
    'Cheek - Pinch skin',     # BFRB-like
    'Text on phone',          # Non-BFRB-like  
    'Above ear - Pull hair',  # BFRB-like
    'Wave hello',             # Non-BFRB-like
    'Neck - Scratch'          # BFRB-like
]

# 予測結果
pred_gestures = [
    'Cheek - Pinch skin',     # ✓ 完全正解
    'Drink from bottle/cup',  # ✓ Non-BFRB-like（種類違いだが二値分類では正解）
    'Neck - Scratch',         # ✓ BFRB-like（種類違いだが二値分類では正解）  
    'Wave hello',             # ✓ 完全正解
    'Text on phone'           # ✗ Non-BFRB-like（二値分類では間違い）
]

# Binary F1計算用変換
true_binary = [1, 0, 1, 0, 1]    # 1=BFRB-like, 0=Non-BFRB-like
pred_binary = [1, 0, 1, 0, 0]    # Binary F1: 4/5で約0.8

# Macro F1計算用変換  
true_macro = ['Cheek - Pinch skin', 'non_target', 'Above ear - Pull hair', 'non_target', 'Neck - Scratch']
pred_macro = ['Cheek - Pinch skin', 'non_target', 'Neck - Scratch', 'non_target', 'non_target']
# 個別BFRB分類: 2/3正解、non_target: 2/2正解
```

### 戦略への影響

#### 両方が等しく重要
- **Binary F1不足**: BFRB見逃し（医学的リスク）
- **Macro F1不足**: 治療法選択の精度低下
- **バランス必須**: 片方に偏ると最終スコア大幅減

#### 効果的なアプローチ
```python
# アプローチ1: 階層的分類
def hierarchical_classification(sensor_data):
    # Step 1: BFRB vs Non-BFRB （Binary F1向上）
    binary_prob = binary_classifier.predict_proba(sensor_data)
    
    if binary_prob[1] > threshold:  # BFRB-likeと判定
        # Step 2: BFRB詳細分類 （Macro F1向上）
        gesture = bfrb_classifier.predict(sensor_data)
    else:
        gesture = 'non_target'  # Non-BFRB統合
    
    return gesture

# アプローチ2: マルチタスク学習
class MultiTaskModel(nn.Module):
    def forward(self, x):
        shared_features = self.encoder(x)
        
        # 両方の目的を同時最適化
        binary_output = self.binary_head(shared_features)
        gesture_output = self.gesture_head(shared_features)
        
        return binary_output, gesture_output
    
    def loss_function(self, binary_pred, gesture_pred, binary_true, gesture_true):
        binary_loss = F.cross_entropy(binary_pred, binary_true)
        gesture_loss = F.cross_entropy(gesture_pred, gesture_true)
        
        # 両方を等しく重視
        return 0.5 * binary_loss + 0.5 * gesture_loss
```

### Non-BFRB-like Gesturesの扱い

#### 学習時の重要性
- **10種類の区別が有効**: 汎化性能向上、特徴学習の多様性
- **個別ラベルで学習**: Binary分類器の判別境界改善

#### 予測時の統合
- **Macro F1では統合**: 10種類全て→ "non_target"
- **Binary F1では重要**: BFRB検出精度に直結

```python
# 学習戦略の例
def training_strategy():
    # Phase 1: 全18クラスで詳細学習
    model.fit(X_train, y_train_18_classes)
    
    # Phase 2: Binary分類ファインチューニング  
    binary_labels = convert_to_binary(y_train_18_classes)
    model.finetune(X_train, binary_labels)
    
    # Phase 3: 評価指標特化調整
    final_model = optimize_for_combined_metric(model)
    
    return final_model
```

### 実装のポイント

```python
def evaluation_aware_prediction(model_output):
    """評価指標を考慮した予測戦略"""
    
    # 両方の指標を同時に考慮
    binary_confidence = get_binary_confidence(model_output)
    gesture_confidence = get_gesture_confidence(model_output)
    
    if binary_confidence < threshold_conservative:
        # Binary F1を重視した保守的予測
        return predict_conservatively(model_output)
    elif gesture_confidence > threshold_confident:
        # Macro F1を重視した積極的予測  
        return predict_specific_gesture(model_output)
    else:
        # バランス重視
        return predict_balanced(model_output)
```

**結論**: BFRB-like の詳細分類とNon-BFRB-like の検出、両方が最終スコアに等しく影響します。医学的価値（BFRB検出）と実用的価値（詳細分類）の両方を満たすバランス型アプローチが必須です。

## Train/Test データの重要な違い

### 利用可能カラムの差分

**Train.csvのみに存在（予測時使用不可）:**
- `sequence_type`: "Target"（BFRB-like）/ "Non-Target"（non-BFRB-like）の分類
- `orientation`: 体勢情報（"Seated Lean Non Dom - FACE DOWN"など）
- `behavior`: **フェーズ情報**（"Transition" / "Pause" / "Gesture"）
- `phase`: 追加のフェーズ詳細情報
- `gesture`: **予測ターゲット**（18種類のジェスチャー名）

**Test.csvで利用可能:**
- `row_id`, `sequence_id`, `sequence_counter`, `subject`: 識別子情報
- `acc_x`, `acc_y`, `acc_z`: IMU加速度データ
- `rot_w`, `rot_x`, `rot_y`, `rot_z`: IMU回転データ
- `thm_1`〜`thm_5`: 温度センサーデータ
- `tof_1_v0`〜`tof_5_v63`: 距離センサーデータ（320次元）

### データサイズの違い
- **Train.csv**: 574,945行（実際のセンサーデータ）
- **Test.csv**: 107行（サンプルデータ、実際のテストは評価API経由）

### 重要な制約事項

#### 1. フェーズ情報が使用不可
```python
# ❌ 学習時のみ可能（テストでは不可）
pause_data = train_df[train_df['behavior'] == 'Pause']
gesture_data = train_df[train_df['behavior'] == 'Gesture']
baseline = pause_data[sensor_cols].mean()

# ✅ テスト環境対応の代替手法
# A. フェーズ推定
estimated_phases = estimate_phases_from_activity(sensor_data)

# B. 時系列位置による推定
phase_by_position = estimate_phase_by_sequence_position(sensor_data)

# C. フェーズ非依存アプローチ
features = extract_sequence_features(sensor_data)  # 全体から特徴抽出
```

#### 2. 体勢情報が使用不可
```python
# ❌ 学習時のみ可能
orientation_specific_model = models[orientation]

# ✅ テスト環境対応
# 体勢に依存しない汎用モデルまたは体勢推定が必要
unified_model = create_orientation_agnostic_model()
```

#### 3. BFRB/non-BFRB事前情報なし
```python
# ❌ 学習時のみ可能
if sequence_type == 'Target':
    use_bfrb_specific_processing()

# ✅ テスト環境対応  
# 事前情報なしで直接18クラス分類
predicted_gesture = model.predict(sensor_data)
```

### Cross-Validation設計への影響

```python
def realistic_cv_split(train_df):
    """テスト環境を正確に模擬したCV分割"""
    
    # テストで利用可能なカラムのみ使用
    available_columns = [
        'row_id', 'sequence_id', 'sequence_counter', 'subject',
        'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',
        'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'
    ] + [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    
    X = train_df[available_columns]
    y = train_df['gesture']
    
    # behavior, orientation, sequence_type は一切使用しない
    return train_test_split(X, y, test_size=0.2, 
                           stratify=y, random_state=42)
```

### モデル設計への影響

```python
class TestCompatibleModel(nn.Module):
    """テスト環境で動作可能なモデル設計"""
    
    def __init__(self):
        super().__init__()
        
        # ✅ センサーデータのみで動作
        self.sensor_encoder = SensorEncoder(
            imu_dim=7,        # acc(3) + rot(4)
            temp_dim=5,       # thm_1〜5
            tof_dim=320       # tof_1_v0〜tof_5_v63
        )
        
        # ✅ フェーズ推定器（オプション）
        self.phase_estimator = PhaseEstimator()
        
        # ✅ 18クラス直接分類
        self.gesture_classifier = nn.Linear(hidden_dim, 18)
        
        # ❌ 使用不可な設計例
        # self.orientation_embedding = nn.Embedding(4, embed_dim)
        # self.sequence_type_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, sensor_data):
        # センサーデータのみで予測
        features = self.sensor_encoder(sensor_data)
        
        # 必要に応じてフェーズ推定
        estimated_phases = self.phase_estimator(sensor_data)
        
        return self.gesture_classifier(features)
```

### 推奨対応戦略

#### Phase 1: 制約を考慮したベースライン
1. **フェーズ非依存モデル**の構築
2. **センサーデータのみ**での特徴抽出
3. **統合モデル**（体勢・BFRB種別非依存）

#### Phase 2: 推定による情報復元
1. **フェーズ推定器**の開発
2. **体勢推定**の実装
3. **動的モデル選択**

#### Phase 3: 高度な適応手法
1. **ドメイン適応**
2. **自己教師学習**によるフェーズ学習
3. **アンサンブル**での不確実性対応

### 実用的なコード例

```python
# 学習時：制約を模擬
def train_with_test_constraints():
    train_df = pd.read_csv('train.csv')
    
    # テスト環境をシミュレート
    test_like_features = train_df.drop([
        'sequence_type', 'orientation', 'behavior', 'phase', 'gesture'
    ], axis=1)
    
    targets = train_df['gesture']
    
    # フェーズ情報なしで学習
    model = create_phase_agnostic_model()
    model.fit(test_like_features, targets)
    
    return model

# 予測時：API対応
def predict_with_api():
    import kaggle_evaluation
    
    env = kaggle_evaluation.make_env()
    model = load_trained_model()
    
    for test_sequence, submission_df in env.iter_test():
        # test_sequenceにはセンサーデータのみ
        prediction = model.predict(test_sequence)
        submission_df['gesture'] = prediction
        env.predict(submission_df)
```

## 技術的制約

- **提出方法**: Python evaluation API経由
- **推論**: シーケンス単位での逐次処理
- **実行時間**: CPU/GPU共に9時間以内
- **外部データ**: 公開データ・事前学習モデル使用可能
- **重要**: フェーズ・体勢・BFRB種別情報は予測時使用不可
