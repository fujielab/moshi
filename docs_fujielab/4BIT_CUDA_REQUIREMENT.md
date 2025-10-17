# ⚠️ 4-bit量子化の重要な注意事項

## CUDA必須

**4-bit量子化には必ずCUDA対応GPUが必要です。**

### なぜCUDAが必要？

`bitsandbytes`の4-bit量子化（NF4）は、CUDA固有の最適化を使用しており、CPU上では動作しません。

```python
# このエラーが発生する場合:
NotImplementedError: Device type not supported for FP4 quantization: cpu
```

### 解決方法

#### オプション1: GPUを使用（推奨）

```bash
# CUDA対応GPUで実行
python -m moshi.server \
  --hf-repo nu-dialogue/j-moshi-ext \
  --host 0.0.0.0 \
  --device cuda \
  --config configs/moshi_7b_202409_4bit.json
```

#### オプション2: 8-bit量子化を使用

8-bit量子化はCPUでも動作します（ただし遅い）：

```bash
python -m moshi.server \
  --hf-repo nu-dialogue/j-moshi-ext \
  --host 0.0.0.0 \
  --config configs/moshi_7b_202409_quantized.json  # 8-bit
```

#### オプション3: 量子化なし

```bash
python -m moshi.server \
  --hf-repo nu-dialogue/j-moshi-ext \
  --host 0.0.0.0
  # 量子化設定なし
```

## 実装の詳細

### 自動CUDA移動

`QLinear4bit`は自動的に以下を処理します：

1. 重みがCPU上にある場合、一時的にCUDAに移動
2. CUDA上で4-bit量子化を実行
3. 量子化後、元のデバイスに戻す

```python
# utils/quantize.py より
if weight.device.type != 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit quantization requires CUDA, but CUDA is not available. "
            "Please use 8-bit quantization instead or run on a GPU."
        )
    weight = weight.cuda()  # 一時的にCUDAへ

# 量子化実行
quant_weight, quant_state = quantize_4bit(weight, ...)

# 元のデバイスに戻す
if original_device.type != 'cuda':
    quant_weight = quant_weight.to(original_device)
```

## 比較表

| 量子化 | CUDA要件 | メモリ削減 | CPU動作 | 速度 |
|--------|---------|-----------|---------|------|
| **4-bit** | **必須** | ~87.5% | ❌ 不可 | 最速 |
| **8-bit** | 推奨 | ~75% | ✅ 可能（遅い） | 速い |
| **なし** | 不要 | 0% | ✅ 可能 | 通常 |

## エラー対処ガイド

### エラー: "CUDA is not available"

```
RuntimeError: 4-bit quantization requires CUDA, but CUDA is not available.
```

**解決策:**
1. CUDA対応GPUが搭載されているか確認
2. CUDAドライバがインストールされているか確認
3. PyTorchがCUDAサポート付きでインストールされているか確認:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### エラー: "Device type not supported for FP4 quantization: cpu"

```
NotImplementedError: Device type not supported for FP4 quantization: cpu
```

**解決策:**
- 8-bit量子化を使用（`configs/moshi_7b_202409_quantized.json`）
- またはGPUで実行

### エラー: "CUDA out of memory"

```
RuntimeError: CUDA out of memory
```

**解決策:**
1. より小さいバッチサイズを使用
2. より小さいモデルを使用（2Bなど）
3. コンテキスト長を短縮
4. 4-bit量子化を使用（既に使用中の場合は、他のプロセスを終了）

## 推奨環境

### 最小要件（4-bit量子化）

- **GPU**: CUDA対応（計算能力6.0以上推奨）
- **VRAM**: 4GB以上（7Bモデルの場合）
- **CUDA**: 11.1以降
- **PyTorch**: 2.0以降（CUDAサポート付き）

### 推奨構成

- **GPU**: NVIDIA A100、L4、T4など
- **VRAM**: 8GB以上
- **CUDA**: 12.x
- **PyTorch**: 最新版

## まとめ

✅ **4-bit量子化を使用する場合: 必ずGPUで実行**

❌ **CPUのみの環境: 8-bit量子化または量子化なしを使用**

📊 **メモリが限られている場合:**
- GPU利用可能 → 4-bit量子化
- GPUなし → 8-bit量子化（遅いが動作する）
- メモリ十分 → 量子化なし（最高精度）
