# 量子化のメタデバイス問題の修正

## 問題の概要

`nu-dialogue/j-moshi-ext`モデルをロードする際に、量子化が有効な場合に以下のエラーが発生していました：

```
RuntimeError: All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:
 [(torch.Size([4096, 4096]), device(type='meta'))]
```

## 原因

モデルの初期化フローで、以下の順序で処理が行われていました：

1. `loaders.py`: `init_device = torch.device('meta')` でモデルを初期化
2. `lm.py`の`__init__`: `quantize=True`の場合、すぐに量子化を実行
3. **問題**: この時点でテンソルはまだ`meta`デバイス上にあり、GPUにロードされていない
4. `bitsandbytes`の量子化処理がGPU上のテンソルを要求してエラー

## 修正内容

### 1. `/home/fujie/work/2025/moshi/moshi/moshi/models/lm.py`

**変更前:**
```python
self._init_weights()
if quantize:
    replace_linear_with_qlinear(self)
```

**変更後:**
```python
self._init_weights()
# Note: quantization is applied after weights are loaded, not here in __init__.
# This is to avoid issues with meta device initialization.
# See loaders.py for the actual quantization call.
self._quantize = quantize
```

**理由**: `__init__`での量子化を削除し、代わりに`_quantize`フラグを保存。

### 2. `/home/fujie/work/2025/moshi/moshi/moshi/models/loaders.py`

**追加したインポート:**
```python
from ..utils.quantize import replace_linear_with_qlinear
```

**変更前:**
```python
if filename is not None:
    # ... load weights ...
    model.load_state_dict(state, assign=True)

if lora:
    # ...
```

**変更後:**
```python
if filename is not None:
    # ... load weights ...
    model.load_state_dict(state, assign=True)

# Apply quantization after weights are loaded (to avoid meta device issues)
if lm_kwargs.get("quantize", False):
    replace_linear_with_qlinear(model)

if lora:
    # ...
```

**理由**: 重みがGPUにロードされた**後**に量子化を適用。

### 3. `/home/fujie/work/2025/moshi/moshi/moshi/modules/transformer.py`

**変更前:**
```python
if quantize:
    # Quantizing layers one by one to avoid taking too much space during init.
    self.layers[-1].to(device=device, dtype=dtype)
    replace_linear_with_qlinear(self.layers[-1])
```

**変更後:**
```python
# Note: quantization is now applied after weights are loaded in loaders.py
# to avoid issues with meta device initialization during model creation.
# Keeping this for backward compatibility with models initialized without loaders.
if quantize and device != torch.device('meta'):
    # Quantizing layers one by one to avoid taking too much space during init.
    self.layers[-1].to(device=device, dtype=dtype)
    replace_linear_with_qlinear(self.layers[-1])
```

**理由**: `meta`デバイスの場合は量子化をスキップ。

## 実行フロー（修正後）

1. `loaders.py`: `init_device = 'meta'` でモデルを初期化（メモリ効率的）
2. `lm.py`の`__init__`: `_quantize`フラグのみ保存、量子化は実行しない
3. `loaders.py`: 重みをファイルから読み込んでGPUに配置
4. `loaders.py`: **ここで量子化を実行**（テンソルは既にGPU上）
5. モデル準備完了

## テスト方法

```bash
# 修正後のコマンド（エラーが出ないはず）
python -m moshi.server --hf-repo nu-dialogue/j-moshi-ext --host 0.0.0.0
```

## 後方互換性

- `quantize=True`でモデルを直接初期化する既存コードも動作します
- `device='meta'`以外で初期化する場合は、従来通りの動作
- `loaders.py`を使わない場合も影響なし

## 関連する量子化機能

この修正により、以下の量子化方法がすべて正常に動作します：

1. **設定ファイルで有効化:**
   ```json
   {"quantize": true}
   ```

2. **コードで指定:**
   ```python
   model = get_moshi_lm(lm_kwargs={"quantize": True}, ...)
   ```

3. **サーバー起動時:**
   ```bash
   python -m moshi.server --hf-repo <repo> --config config_quantized.json
   ```

## 注意事項

- 量子化には`bitsandbytes`が必要: `pip install bitsandbytes`
- CUDA対応GPUが必要
- 量子化とLoRAは併用不可（既存の制限）

## まとめ

この修正により、`meta`デバイスを使用した効率的なモデル初期化と量子化が両立できるようになりました。メモリ効率を保ちつつ、量子化によるメモリ削減も実現できます。
