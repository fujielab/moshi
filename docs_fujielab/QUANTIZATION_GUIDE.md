# Moshi モデルの量子化ガイド

このプロジェクトには既に4-bit/8-bit量子化の機能が実装されており、メモリ使用量を大幅に削減できます。

## 📊 量子化のメリット

- **メモリ使用量削減**: 8-bit量子化で約50%、4-bit量子化で約75%のメモリ削減
- **推論速度向上**: 量子化により推論速度が向上する場合がある
- **精度の維持**: 適切な量子化では精度の低下は最小限

## 🛠️ 使用可能な量子化オプション

### 1. PyTorch版（8-bit量子化）

`bitsandbytes`ライブラリを使用した8-bit量子化が利用可能です。

#### インストール

```bash
pip install bitsandbytes
```

#### コードでの使用方法

```python
from moshi.models.loaders import get_moshi_lm

# 量子化を有効にしてモデルをロード
model = get_moshi_lm(
    filename="path/to/model.safetensors",
    lm_kwargs={"quantize": True},  # 量子化を有効化
    device="cuda",
    dtype=torch.bfloat16
)
```

または、既存のモデルを量子化：

```python
from moshi.utils.quantize import replace_linear_with_qlinear
from moshi.models import LMModel

# モデルを作成
model = LMModel(quantize=False, ...)

# 後から量子化を適用
replace_linear_with_qlinear(model)
```

#### 設定ファイルでの使用

`configs/`フォルダ内のJSON設定ファイルに以下を追加：

```json
{
  "quantize": true,
  ...その他の設定
}
```

### 2. MLX版（4-bit/8-bit量子化） - macOS専用

Apple Silicon Mac向けのMLX実装では、4-bitと8-bit両方の量子化がサポートされています。

#### インストール

```bash
pip install moshi_mlx
```

#### 使用方法

```bash
# 4-bit量子化で実行（最も軽量）
python -m moshi_mlx.local -q 4

# 8-bit量子化で実行（バランス型）
python -m moshi_mlx.local -q 8

# Webインターフェース版
python -m moshi_mlx.local_web -q 4
```

#### 既存モデルの量子化

```bash
# MLXモデルを量子化してエクスポート
python scripts/quantize_mlx.py path/to/model.safetensors \
  --out model.q8.safetensors \
  --bits 8 \
  --group-size 64

# 4-bit量子化
python scripts/quantize_mlx.py path/to/model.safetensors \
  --out model.q4.safetensors \
  --bits 4 \
  --group-size 32
```

### 3. Rust版（量子化モデルの読み込み）

Rust実装でも量子化されたモデルの読み込みをサポートしています。

```toml
# Cargo.toml
[dependencies]
candle-transformers = { version = "...", features = ["quantized"] }
```

## 📝 実装例

### 例1: サーバーで量子化モデルを使用

```python
from moshi.models.loaders import CheckpointInfo

# Hugging Faceから量子化済みモデルをロード
info = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
model = info.get_moshi(
    device="cuda",
    lm_kwargs_overrides={"quantize": True}
)

# または直接設定
info = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-q8")
model = info.get_moshi(device="cuda")
```

### 例2: カスタムトレーニングスクリプトで使用

```python
import torch
from moshi.models.lm import LMModel

# 量子化を有効にしてモデルを作成
model = LMModel(
    n_q=16,
    dep_q=8,
    card=2048,
    dim=4096,
    quantize=True,  # 8-bit量子化を有効化
    device="cuda",
    dtype=torch.bfloat16
)

# モデルはすでに量子化されている
print(f"Model ready with quantization")
```

### 例3: 量子化モデルをHugging Faceにエクスポート

```bash
# PyTorchモデルを8-bit量子化してアップロード
python scripts/export_quantized.py \
  kyutai/moshiko-pytorch-bf16 \
  your-username/moshiko-q8
```

## ⚙️ 詳細設定

### bitsandbytes量子化の仕組み

`QLinear`クラスが通常の`nn.Linear`レイヤーを置き換え：

```python
# moshi/utils/quantize.py より
class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        # 重みを8-bit整数に量子化
        CB, SCB, _ = bnbF.int8_vectorwise_quant(
            weight.data.to(torch.float16)
        )
        self.weight = nn.Parameter(CB, requires_grad=False)
        self.weight_scb = nn.Parameter(SCB, requires_grad=False)
```

### MLX量子化の仕組み

```python
# moshi_mlx での使用例
import mlx.nn as nn

# モデルを量子化
nn.quantize(model, bits=8, group_size=64)  # 8-bit
nn.quantize(model, bits=4, group_size=32)  # 4-bit
```

## 🔍 パフォーマンス比較

| モデル | メモリ使用量 | 推論速度 | 精度 |
|--------|--------------|----------|------|
| FP32 (フル精度) | 100% | ベースライン | 100% |
| BF16 (半精度) | 50% | ~1.5x | ~99.9% |
| INT8 (8-bit) | 25% | ~2x | ~99.5% |
| INT4 (4-bit) | 12.5% | ~3x | ~98% |

※ 実際のパフォーマンスはハードウェアと設定に依存します

## ⚠️ 注意事項

1. **LoRAとの互換性**: 現在、量子化とLoRAは同時に使用できません
   ```python
   assert not lm_kwargs.get("quantize"), (
       "LoRA and quantization are incompatible for now."
   )
   ```

2. **GPU要件**: PyTorchの8-bit量子化にはCUDA対応GPUが必要

3. **データ型の整合性**: 量子化後はモデルのdtypeを変更しないでください

4. **初回実行**: 量子化の初回実行時は変換に時間がかかる場合があります

## 🎯 推奨される使用シナリオ

- **開発/テスト環境**: 4-bit量子化（最小メモリ）
- **本番環境（高精度）**: 8-bit量子化（バランス型）
- **GPU制約環境**: BF16 + 8-bit量子化
- **macOS（Apple Silicon）**: MLX版の4-bit/8-bit量子化

## 📚 関連ファイル

- `moshi/utils/quantize.py` - PyTorch量子化実装
- `moshi_mlx/moshi_mlx/` - MLX量子化実装
- `scripts/export_quantized.py` - 量子化モデルエクスポート
- `scripts/quantize_mlx.py` - MLXモデル量子化
- `rust/moshi-core/src/nn.rs` - Rust量子化サポート

## 🐛 トラブルシューティング

### "bitsandbytes not found"

```bash
pip install bitsandbytes
```

### CUDA関連エラー

bitsandbytesはCUDA 11.1以降が必要です：

```bash
# CUDA互換性を確認
python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー

より積極的な量子化を試してください：
- 8-bit → 4-bit (MLX)
- バッチサイズを削減
- より小さいモデルバリアントを使用

## 📖 参考資料

- [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes)
- [MLX documentation](https://ml-explore.github.io/mlx/)
- [Hugging Face Model Hub](https://huggingface.co/kyutai)
