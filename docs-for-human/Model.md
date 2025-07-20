# モデル選択ガイド

## コンペ特性とモデル選択

### このコンペの特徴
- **時系列センサーデータ**：IMU（7次元）+ 温度（5次元）+ 距離（320次元）
- **マルチフェーズ構造**：Transition → Pause → Gesture
- **二重評価指標**：Binary F1 + Macro F1の平均
- **実行時間制約**：9時間以内（Notebook Competition）
- **センサー構成差**：テストの半分はIMUのみ

### 推奨モデル選択戦略
1. **Phase 1**: 堅実なベースライン構築（CNN系）
2. **Phase 2**: 高性能モデル導入（Transformer系）
3. **Phase 3**: アンサンブル最適化

## Tier 1: 最優先モデル（実績・安定性重視）

### 1. ResNet1D
**最も推奨される堅実な選択肢**

```python
class ResNet1D(nn.Module):
    def __init__(self, input_channels=327, num_classes=18):
        super().__init__()
        # 初期畳み込み
        self.initial_conv = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNetブロック
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 分類ヘッド
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.binary_head = nn.Linear(512, 2)      # BFRB vs non-BFRB
        self.gesture_head = nn.Linear(512, 18)    # 詳細ジェスチャー
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        return self.relu(out)
```

**特徴:**
- **実績豊富**: 時系列分類で広く使用され、安定した性能
- **効率的**: 学習・推論速度が高速
- **実装容易**: PyTorchで豊富な実装例
- **勾配消失対策**: Skip connectionで深いネットワーク学習可能

**適用場面**: ベースライン構築、高速実験、アンサンブルの一部

### 2. EfficientNet1D
**効率性と性能のバランス**

```python
class EfficientNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Conv1d(327, 32, kernel_size=3, padding=1)
        
        # MBConv blocks with different expansion ratios
        self.blocks = nn.ModuleList([
            MBConvBlock1D(32, 16, expand_ratio=1, kernel_size=3),
            MBConvBlock1D(16, 24, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock1D(24, 40, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock1D(40, 80, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock1D(80, 112, expand_ratio=6, kernel_size=5),
            MBConvBlock1D(112, 192, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock1D(192, 320, expand_ratio=6, kernel_size=3)
        ])
        
        # Head
        self.conv_head = nn.Conv1d(320, 1280, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(1280, 18)

class MBConvBlock1D(nn.Module):
    """Mobile Inverted Residual Block for 1D"""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv1d(in_channels, hidden_dim, 1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze-and-excitation
        layers.append(SEBlock1D(hidden_dim))
        
        # Pointwise projection
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y
```

**特徴:**
- **高効率**: パラメータ数に対して高い性能
- **モバイル最適化**: 推論速度が高速
- **Squeeze-and-Excitation**: チャンネル間の重要度学習
- **複合スケーリング**: 深さ・幅・解像度のバランス最適化

### 3. CNN-LSTM Hybrid
**局所パターンと時系列依存の両方を捉える**

```python
class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, input_channels=327, lstm_hidden=256, num_classes=18):
        super().__init__()
        
        # CNN feature extractor
        self.cnn_layers = nn.Sequential(
            # Multi-scale convolutions
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(256, lstm_hidden, batch_first=True, 
                           bidirectional=True, num_layers=2, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_hidden*2, 8, dropout=0.1)
        
        # Classification heads
        self.binary_head = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        self.gesture_head = nn.Sequential(
            nn.Linear(lstm_hidden*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, channels, time) -> (batch, time, channels)
        cnn_features = self.cnn_layers(x.transpose(1, 2))
        cnn_features = cnn_features.transpose(1, 2)  # Back to (batch, time, features)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # Attention pooling
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        global_feature = attn_out.mean(dim=1)
        
        # Classification
        binary_out = self.binary_head(global_feature)
        gesture_out = self.gesture_head(global_feature)
        
        return binary_out, gesture_out
```

## Tier 2: 高性能モデル（計算コスト高）

### 4. Transformer (Vision Transformer 1D)
**最高性能を狙う選択肢**

```python
class GestureTransformer(nn.Module):
    def __init__(self, input_dim=327, d_model=512, nhead=8, num_layers=6, 
                 max_seq_len=1000, num_classes=18):
        super().__init__()
        
        # Input projection and positional encoding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Phase embeddings (Transition, Pause, Gesture)
        self.phase_embedding = nn.Embedding(3, d_model)
        
        # Sensor-specific projections
        self.sensor_projections = nn.ModuleDict({
            'imu': nn.Linear(7, d_model//4),
            'thermopile': nn.Linear(5, d_model//4),
            'tof': nn.Linear(320, d_model//2)
        })
        
        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-sensor attention
        self.cross_sensor_attention = nn.MultiheadAttention(d_model, nhead//2)
        
        # Classification heads with different pooling strategies
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 2)
        )
        
        self.gesture_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x, phase_ids=None):
        batch_size, seq_len, _ = x.shape
        
        # Sensor-specific processing
        imu_features = self.sensor_projections['imu'](x[:, :, :7])
        thm_features = self.sensor_projections['thermopile'](x[:, :, 7:12])
        tof_features = self.sensor_projections['tof'](x[:, :, 12:])
        
        # Combine sensor features
        combined_features = torch.cat([imu_features, thm_features, tof_features], dim=-1)
        
        # Add positional encoding
        x = self.positional_encoding(combined_features)
        
        # Add phase information if available
        if phase_ids is not None:
            phase_emb = self.phase_embedding(phase_ids)
            x = x + phase_emb
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.transpose(0, 1)  # Back to (batch, seq_len, d_model)
        
        # Use CLS token for classification
        cls_output = transformer_out[:, 0]
        
        # Classification
        binary_out = self.binary_head(cls_output)
        gesture_out = self.gesture_head(cls_output)
        
        return binary_out, gesture_out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)
```

**特徴:**
- **最高性能**: 複雑なパターン学習能力
- **注意機構**: センサー間・時間間の関係性学習
- **スケーラブル**: データ量に応じて性能向上
- **事前学習活用**: 大規模事前学習モデルの利用可能

### 5. ConvNeXt1D
**最新CNN（2022年）、Transformerに匹敵する性能**

```python
class ConvNeXt1D(nn.Module):
    def __init__(self, input_channels=327, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], num_classes=18):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm1d(dims[0])
        )
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock1D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
            # Downsampling between stages
            if i < 3:
                downsample = nn.Sequential(
                    LayerNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
                self.stages.append(downsample)
        
        # Head
        self.norm = LayerNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.mean(dim=-1)
        x = self.norm(x)
        x = self.head(x)
        
        return x

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        
        # Depthwise convolution with large kernel
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1d(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        
        x = input + self.drop_path(x)
        return x

class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x
```

## Tier 3: 特殊用途・実験的

### 6. TCN (Temporal Convolutional Networks)
**因果性を保持した時系列モデル**

```python
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1)*dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        y = self.network(x)
        return self.linear(y[:, :, -1])  # 最後の時刻の出力

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
```

### 7. WaveNet Style
**非常に長い時系列での文脈捕捉**

```python
class WaveNet1D(nn.Module):
    def __init__(self, input_channels=327, residual_channels=64, 
                 dilation_channels=64, skip_channels=64, num_classes=18):
        super().__init__()
        
        self.start_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Dilated convolution blocks
        self.dilations = [2**i for i in range(10)] * 3  # 3サイクル
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for dilation in self.dilations:
            dilated_conv = nn.Conv1d(residual_channels, dilation_channels, 2, 
                                   dilation=dilation, padding=dilation)
            residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
            skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)
            
            self.dilated_convs.append(dilated_conv)
            self.residual_convs.append(residual_conv)
            self.skip_convs.append(skip_conv)
        
        self.end_conv_1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.end_conv_2 = nn.Conv1d(skip_channels, num_classes, 1)
        
    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []
        
        for dilated_conv, residual_conv, skip_conv in zip(
            self.dilated_convs, self.residual_convs, self.skip_convs):
            
            # Gated activation
            dilated_out = dilated_conv(x)
            tanh_out = torch.tanh(dilated_out)
            sigmoid_out = torch.sigmoid(dilated_out)
            gated = tanh_out * sigmoid_out
            
            # Residual connection
            residual_out = residual_conv(gated)
            x = x + residual_out
            
            # Skip connection
            skip_out = skip_conv(gated)
            skip_connections.append(skip_out)
        
        # Combine skip connections
        skip_sum = sum(skip_connections)
        
        # Final layers
        out = torch.relu(skip_sum)
        out = self.end_conv_1(out)
        out = torch.relu(out)
        out = self.end_conv_2(out)
        
        return out.mean(dim=-1)  # Global average pooling
```

## マルチタスク学習アーキテクチャ

### Multi-Task Gesture Model
**Binary F1 + Macro F1を同時最適化**

```python
class MultiTaskGestureModel(nn.Module):
    def __init__(self, base_model='resnet', num_aux_tasks=3):
        super().__init__()
        
        # Base encoder (共有特徴抽出器)
        if base_model == 'resnet':
            self.encoder = ResNet1D(input_channels=327, num_classes=512)
        elif base_model == 'transformer':
            self.encoder = GestureTransformer(num_classes=512)
        
        # メインタスクヘッド
        self.binary_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # BFRB vs non-BFRB
        )
        
        self.gesture_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 18)  # 18種類のジェスチャー
        )
        
        # 補助タスクヘッド（汎化性能向上）
        self.orientation_head = nn.Linear(512, 4)      # 体勢予測
        self.subject_head = nn.Linear(512, 100)        # 被験者識別（仮想的）
        self.phase_head = nn.Linear(512, 3)            # フェーズ予測
        
        # Attention weights for task balancing
        self.task_weights = nn.Parameter(torch.ones(5))
        
    def forward(self, x, return_attention=False):
        # 共有特徴抽出
        features = self.encoder(x)
        
        # タスク別予測
        outputs = {
            'binary': self.binary_head(features),
            'gesture': self.gesture_head(features),
            'orientation': self.orientation_head(features),
            'subject': self.subject_head(features),
            'phase': self.phase_head(features)
        }
        
        if return_attention:
            outputs['task_weights'] = torch.softmax(self.task_weights, dim=0)
        
        return outputs
    
    def compute_loss(self, outputs, targets):
        # 動的重み付き損失
        weights = torch.softmax(self.task_weights, dim=0)
        
        losses = {
            'binary': F.cross_entropy(outputs['binary'], targets['binary']),
            'gesture': F.cross_entropy(outputs['gesture'], targets['gesture']),
            'orientation': F.cross_entropy(outputs['orientation'], targets['orientation']),
            'subject': F.cross_entropy(outputs['subject'], targets['subject']),
            'phase': F.cross_entropy(outputs['phase'], targets['phase'])
        }
        
        # 重み付き合計損失
        total_loss = sum(weights[i] * loss for i, loss in enumerate(losses.values()))
        
        return total_loss, losses, weights
```

## アンサンブル戦略

### Heterogeneous Ensemble
**異なるアーキテクチャの組み合わせ**

```python
class HeterogeneousEnsemble:
    def __init__(self):
        self.models = {
            'resnet1d': ResNet1D(),
            'efficientnet1d': EfficientNet1D(),
            'transformer': GestureTransformer(),
            'cnn_lstm': CNN_LSTM_Hybrid(),
            'convnext1d': ConvNeXt1D()
        }
        
        # 学習可能な重み
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
        # メタ学習器
        self.meta_learner = nn.Sequential(
            nn.Linear(len(self.models) * 18, 128),  # 各モデルの出力確率を結合
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 18)
        )
        
    def forward(self, x):
        model_outputs = []
        
        # 各モデルの予測
        for model_name, model in self.models.items():
            with torch.no_grad():
                output = model(x)
                prob = torch.softmax(output, dim=-1)
                model_outputs.append(prob)
        
        # Simple weighted ensemble
        weights = torch.softmax(self.ensemble_weights, dim=0)
        simple_ensemble = sum(w * out for w, out in zip(weights, model_outputs))
        
        # Meta-learner ensemble
        meta_input = torch.cat(model_outputs, dim=-1)
        meta_output = self.meta_learner(meta_input)
        
        return {
            'simple_ensemble': simple_ensemble,
            'meta_ensemble': meta_output,
            'individual_outputs': model_outputs
        }
```

### Temporal Ensemble
**異なる時間窓での予測結合**

```python
class TemporalEnsemble:
    def __init__(self, base_model, window_sizes=[50, 100, 150, 200]):
        self.base_model = base_model
        self.window_sizes = window_sizes
        self.ensemble_weights = nn.Parameter(torch.ones(len(window_sizes)))
        
    def forward(self, x):
        seq_len = x.size(1)
        predictions = []
        
        for window_size in self.window_sizes:
            if seq_len >= window_size:
                # 異なる窓サイズでサンプリング
                step = max(1, seq_len // window_size)
                windowed_x = x[:, ::step, :][:, :window_size, :]
            else:
                # パディングまたは繰り返し
                repeat_times = (window_size + seq_len - 1) // seq_len
                windowed_x = x.repeat(1, repeat_times, 1)[:, :window_size, :]
            
            pred = self.base_model(windowed_x)
            predictions.append(torch.softmax(pred, dim=-1))
        
        # 重み付きアンサンブル
        weights = torch.softmax(self.ensemble_weights, dim=0)
        final_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        return final_pred
```

## 推奨実装順序

### Phase 1: ベースライン構築（1-2週間）
```python
implementation_order = [
    "1. ResNet1D - 基本実装",
    "2. データ前処理パイプライン構築", 
    "3. 評価指標実装（Binary F1 + Macro F1）",
    "4. Cross-validation環境構築",
    "5. ベースライン性能測定"
]
```

### Phase 2: 性能向上（2-3週間）
```python
improvement_steps = [
    "6. EfficientNet1D実装・比較",
    "7. CNN-LSTM Hybrid実装",
    "8. マルチフェーズ活用（Transition/Pause/Gesture）",
    "9. Pause正規化による個人差対応",
    "10. マルチタスク学習導入"
]
```

### Phase 3: 高度化（2-3週間）
```python
advanced_steps = [
    "11. Transformer実装",
    "12. ConvNeXt1D実装", 
    "13. Heterogeneous Ensemble構築",
    "14. メタ学習器による動的重み付け",
    "15. Test Time Augmentation"
]
```

### Phase 4: 最終最適化（1週間）
```python
optimization_steps = [
    "16. ハイパーパラメータ最適化",
    "17. モデル軽量化（9時間制約対応）",
    "18. アンサンブル重み最適化",
    "19. 最終検証・提出準備"
]
```

## 性能予測・選択指針

### 予想性能レンジ（CV Score）
```python
expected_performance = {
    "ResNet1D（単体）": "0.75-0.80",
    "EfficientNet1D（単体）": "0.76-0.81", 
    "CNN-LSTM（単体）": "0.77-0.82",
    "Transformer（単体）": "0.80-0.85",
    "ConvNeXt1D（単体）": "0.78-0.83",
    "MultiTask Learning": "+0.02-0.03 boost",
    "3-Model Ensemble": "+0.03-0.05 boost",
    "5-Model Ensemble": "+0.04-0.06 boost"
}
```

### 計算コスト vs 性能
```python
efficiency_analysis = {
    "ResNet1D": {"performance": "High", "speed": "Fast", "memory": "Low"},
    "EfficientNet1D": {"performance": "High", "speed": "Fast", "memory": "Low"},
    "CNN-LSTM": {"performance": "High", "speed": "Medium", "memory": "Medium"},
    "Transformer": {"performance": "Highest", "speed": "Slow", "memory": "High"},
    "ConvNeXt1D": {"performance": "High", "speed": "Medium", "memory": "Medium"}
}
```

このガイドに従って段階的に実装を進めることで、効率的にコンペで高いスコアを目指せます。特に時間制約（9時間）を考慮して、まずは軽量で安定したモデルから始めることを強く推奨します。
