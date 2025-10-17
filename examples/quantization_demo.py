#!/usr/bin/env python3
"""
量子化されたMoshiモデルを使用した簡単な推論の例

このスクリプトは、8-bit量子化を使用してメモリ使用量を削減します。
"""

import torch
from moshi.models.loaders import CheckpointInfo


def main():
    print("=" * 60)
    print("Moshi モデル量子化デモ")
    print("=" * 60)
    
    # Hugging Faceリポジトリを指定
    hf_repo = "kyutai/moshiko-pytorch-bf16"
    
    print(f"\n✅ リポジトリから設定を読み込み中: {hf_repo}")
    info = CheckpointInfo.from_hf_repo(hf_repo)
    
    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ デバイス: {device}")
    
    if device == "cpu":
        print("\n⚠️  注意: CPUでは量子化の効果が限定的です。")
        print("   GPUの使用を推奨します。")
    
    # 量子化なしでモデルをロード（比較用）
    print("\n" + "-" * 60)
    print("1. 通常のモデルをロード中...")
    print("-" * 60)
    
    try:
        model_normal = info.get_moshi(
            device=device,
            lm_kwargs_overrides={"quantize": False}
        )
        
        # メモリ使用量を計算
        param_size_normal = sum(p.nelement() * p.element_size() 
                                for p in model_normal.parameters())
        buffer_size_normal = sum(b.nelement() * b.element_size() 
                                 for b in model_normal.buffers())
        memory_normal = (param_size_normal + buffer_size_normal) / 1024**3
        
        print(f"✅ 通常モデルのメモリ使用量: {memory_normal:.2f} GB")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        memory_normal = None
    
    # 量子化ありでモデルをロード
    print("\n" + "-" * 60)
    print("2. 量子化モデル（8-bit）をロード中...")
    print("-" * 60)
    
    try:
        # bitsandbytesのインストールチェック
        try:
            import bitsandbytes
            print("✅ bitsandbytes がインストール済み")
        except ImportError:
            print("❌ bitsandbytes がインストールされていません")
            print("   インストール: pip install bitsandbytes")
            return
        
        model_quantized = info.get_moshi(
            device=device,
            lm_kwargs_overrides={"quantize": True}  # 量子化を有効化
        )
        
        # メモリ使用量を計算
        param_size_quantized = sum(p.nelement() * p.element_size() 
                                   for p in model_quantized.parameters())
        buffer_size_quantized = sum(b.nelement() * b.element_size() 
                                    for b in model_quantized.buffers())
        memory_quantized = (param_size_quantized + buffer_size_quantized) / 1024**3
        
        print(f"✅ 量子化モデルのメモリ使用量: {memory_quantized:.2f} GB")
        
        # 比較
        if memory_normal is not None:
            reduction = ((memory_normal - memory_quantized) / memory_normal) * 100
            print(f"\n📊 メモリ削減率: {reduction:.1f}%")
            print(f"   削減量: {memory_normal - memory_quantized:.2f} GB")
        
        # モデル情報
        print("\n" + "-" * 60)
        print("モデル情報:")
        print("-" * 60)
        print(f"  - コードブック数: {model_quantized.n_q}")
        print(f"  - Depformerコードブック: {model_quantized.dep_q}")
        print(f"  - 次元: {model_quantized.dim}")
        print(f"  - カード: {model_quantized.card}")
        
        # 量子化レイヤーのカウント
        from moshi.utils.quantize import QLinear
        quantized_layers = sum(1 for m in model_quantized.modules() 
                               if isinstance(m, QLinear))
        total_linear_layers = sum(1 for m in model_quantized.modules() 
                                  if isinstance(m, (torch.nn.Linear, QLinear)))
        
        print(f"\n  - 量子化レイヤー数: {quantized_layers} / {total_linear_layers}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
