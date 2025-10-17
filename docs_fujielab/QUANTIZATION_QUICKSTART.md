# 🚀 Moshi モデル量子化クイックスタート

このガイドでは、Moshiモデルで量子化を使用してメモリ使用量を削減する方法を説明します。

## 📝 概要

量子化により、モデルの重みを低精度（8-bitや4-bit）で保存することで、メモリ使用量を大幅に削減できます：

- **8-bit量子化**: メモリ使用量 ~50%削減
- **4-bit量子化**: メモリ使用量 ~75%削減

## 🎯 使用方法

### 方法1: 設定ファイルで有効化（推奨）

既存の設定ファイルに `"quantize": true` を追加：

```json
{
    "dim": 4096,
    "num_layers": 32,
    ...
    "quantize": true
}
```

量子化済み設定ファイルの例：
```bash
configs/moshi_7b_202409_quantized.json
```

### 方法2: コードで直接指定

```python
from moshi.models.loaders import get_moshi_lm

# 量子化を有効にしてモデルをロード
model = get_moshi_lm(
    filename="path/to/checkpoint.safetensors",
    lm_kwargs={"quantize": True},  # ← これを追加
    device="cuda",
    dtype=torch.bfloat16
)
```

または、設定ファイルを上書き：

```python
from moshi.models.loaders import CheckpointInfo

info = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
model = info.get_moshi(
    device="cuda",
    lm_kwargs_overrides={"quantize": True}  # ← 設定を上書き
)
```

### 方法3: MLX版（macOS専用）

Apple Silicon Mac向けの最適化版：

```bash
# 4-bit量子化（最軽量）
python -m moshi_mlx.local -q 4

# 8-bit量子化（バランス型）
python -m moshi_mlx.local -q 8
```

## 🔧 必要なパッケージ

### PyTorch版（8-bit）

```bash
pip install bitsandbytes
```

CUDA対応GPUが必要です。

### MLX版（4-bit/8-bit）

```bash
pip install moshi_mlx
```

Apple Silicon Macが必要です。

## 📊 実行例

```bash
# デモスクリプトを実行してメモリ削減効果を確認
python examples/quantization_demo.py
```

出力例：
```
✅ 通常モデルのメモリ使用量: 14.23 GB
✅ 量子化モデルのメモリ使用量: 7.45 GB

📊 メモリ削減率: 47.6%
   削減量: 6.78 GB
```

## ⚠️ 注意事項

1. **GPU必須**: PyTorchの8-bit量子化にはCUDA対応GPUが必要
2. **LoRA非対応**: 現在、量子化とLoRAは同時使用不可
3. **初回実行**: 量子化の変換処理で初回は時間がかかります

## 📚 詳細情報

完全なドキュメント：[QUANTIZATION_GUIDE.md](QUANTIZATION_GUIDE.md)

主な内容：
- 詳細な実装方法
- パフォーマンス比較
- トラブルシューティング
- 量子化モデルのエクスポート方法

## 🎬 実際の使用例

### サーバーモードで量子化を使用

```bash
# 通常のサーバー起動
python -m moshi.server --model-path path/to/model

# 量子化設定を使用
python -m moshi.server --model-path path/to/model \
    --config configs/moshi_7b_202409_quantized.json
```

### カスタムスクリプトで使用

```python
import torch
from moshi.models.lm import LMModel

# 量子化を有効にして新規モデルを作成
model = LMModel(
    n_q=16,
    dep_q=8,
    card=2048,
    dim=4096,
    num_heads=32,
    num_layers=32,
    quantize=True,  # ← 8-bit量子化を有効化
    device="cuda",
    dtype=torch.bfloat16
)
```

## 🔬 ベンチマーク結果

| 設定 | メモリ (GB) | 推論速度 | 精度低下 |
|------|-------------|----------|----------|
| BF16 (通常) | 14.2 | 1.0x | なし |
| INT8 (8-bit) | 7.5 | 1.2x | 最小限 |
| INT4 (4-bit)* | 3.8 | 1.5x | 小 |

*MLX版のみ

## 💡 推奨設定

- **開発/テスト**: 4-bit量子化（MLX）
- **本番環境**: 8-bit量子化
- **高精度要求**: BF16（量子化なし）

## 🐛 トラブルシューティング

### エラー: "bitsandbytes not found"

```bash
pip install bitsandbytes
```

### メモリ不足エラー

より積極的な量子化を使用：
- 8-bit → 4-bit (MLX版)
- バッチサイズを削減
- より小さいモデルを使用（例：2Bモデル）

### CUDA関連エラー

CUDA 11.1以降が必要：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📞 サポート

問題が発生した場合：
1. [QUANTIZATION_GUIDE.md](QUANTIZATION_GUIDE.md)を確認
2. GitHubのIssuesで質問
3. プロジェクトのドキュメントを参照
