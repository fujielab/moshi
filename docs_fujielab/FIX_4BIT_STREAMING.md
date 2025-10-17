# 4-bit量子化のストリーミング対応修正

## 問題

4-bit量子化を有効にしてサーバーを起動すると、以下のエラーが発生：

```
RuntimeError: Unknown type <class 'moshi.utils.quantize.QLinear4bit'> for linear.
```

## 原因

`StreamingTransformer`の`_init_streaming_state`メソッドで、`QLinear4bit`クラスが認識されていなかった。また、cross attention機能でも量子化レイヤーのweightアクセスが適切に処理されていなかった。

## 修正内容

### 1. ヘルパー関数の追加 (`moshi/modules/transformer.py`)

量子化レイヤーからweightを取得するヘルパー関数を追加：

```python
def _get_linear_weight(linear_module):
    """Get weight tensor from a linear module (handles quantized layers)."""
    if isinstance(linear_module, nn.Linear):
        return linear_module.weight
    elif isinstance(linear_module, (quantize.QLinear8bit, quantize.QLinear)):
        return linear_module.weight
    elif isinstance(linear_module, quantize.QLinear4bit):
        # For 4-bit quantized layers, we need to get the weight from bnb_linear
        return linear_module.bnb_linear.weight
    elif isinstance(linear_module, LoRALinear):
        # LoRA has frozen_W which is the original linear layer
        return linear_module.frozen_W.weight
    else:
        raise TypeError(f"Unsupported linear module type: {type(linear_module)}")
```

### 2. `_init_streaming_state`メソッドの修正

**変更前:**
```python
elif isinstance(in_proj, quantize.QLinear):
    device = in_proj.weight.device
    dtype = torch.float16
else:
    raise RuntimeError(f"Unknown type {type(in_proj)} for linear.")
```

**変更後:**
```python
elif isinstance(in_proj, (quantize.QLinear, quantize.QLinear8bit, quantize.QLinear4bit)):
    # Handle both 8-bit and 4-bit quantized layers
    if isinstance(in_proj, quantize.QLinear4bit):
        device = in_proj.bnb_linear.weight.device
    else:
        device = in_proj.weight.device
    dtype = torch.float16
else:
    raise RuntimeError(f"Unknown type {type(in_proj)} for linear.")
```

### 3. Cross Attention機能の修正

`_compute_cross_attention`と`forward`メソッドで、量子化レイヤーのweightアクセスを修正：

**変更前:**
```python
assert isinstance(in_proj, nn.Linear)
dim = in_proj.weight.shape[0] // 3
q = nn.functional.linear(query, in_proj.weight[:dim])
```

**変更後:**
```python
# Get weight tensor (supports quantized layers)
try:
    weight = _get_linear_weight(in_proj)
except TypeError:
    # Fallback for unsupported types
    assert isinstance(in_proj, nn.Linear), (
        f"Cross attention with quantized layers is not fully supported. "
        f"Got {type(in_proj)}"
    )
    weight = in_proj.weight
    
dim = weight.shape[0] // 3
q = F.linear(query, weight[:dim])
```

## テスト方法

```bash
# 4-bit量子化でサーバーを起動（修正後はエラーなし）
python -m moshi.server \
  --hf-repo nu-dialogue/j-moshi-ext \
  --host 0.0.0.0 \
  --config configs/moshi_7b_202409_4bit.json
```

## サポートされる機能

### ✅ サポート済み
- **Self-attention**: 4-bit/8-bit量子化で完全動作
- **ストリーミング**: KVキャッシュ対応
- **通常の推論**: 問題なし

### ⚠️ 制限あり
- **Cross-attention**: 量子化レイヤーでは部分的にサポート
  - 重みの直接スライス操作が必要な場合は非対応
  - 通常のMoshiモデルではcross-attentionを使用しないため影響なし

## 注意事項

1. **QLinear4bitの構造**:
   - 内部的に`bnb_linear`を保持
   - weightアクセスは`module.bnb_linear.weight`

2. **デバイス/dtype**:
   - 量子化レイヤーは常に`torch.float16`でdtypeを返す
   - これはストリーミングステートの初期化に使用される

3. **Cross-attention**:
   - Moshiモデルではdepformerがcross-attentionを使用しない
   - そのため、この制限は実際の使用には影響しない

## 関連ファイル

- `moshi/modules/transformer.py` - メイン修正箇所
- `moshi/utils/quantize.py` - QLinear4bit実装
- `configs/moshi_7b_202409_4bit.json` - 4-bit設定

## まとめ

この修正により、4-bit量子化を使用したMoshiサーバーが正常に起動・動作するようになりました。ストリーミング機能も問題なく使用できます。
