#!/usr/bin/env python3
"""
4-bit量子化のデモスクリプト

メモリ使用量を比較します。
"""

import torch
from moshi.models.loaders import CheckpointInfo


def format_memory(bytes_val):
    """メモリサイズを読みやすい形式にフォーマット"""
    gb = bytes_val / (1024 ** 3)
    return f"{gb:.2f} GB"


def get_model_memory(model):
    """モデルのメモリ使用量を計算"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return param_size + buffer_size


def main():
    print("=" * 70)
    print("Moshi 4-bit量子化デモ")
    print("=" * 70)
    
    # 設定
    hf_repo = "kyutai/moshiko-pytorch-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("\n⚠️  警告: CPUモードでは量子化の効果が限定的です")
        print("   GPUの使用を推奨します\n")
        return
    
    print(f"\n✅ デバイス: {device}")
    print(f"✅ リポジトリ: {hf_repo}\n")
    
    # リポジトリ情報を取得
    print("📥 モデル情報を取得中...")
    info = CheckpointInfo.from_hf_repo(hf_repo)
    
    results = []
    
    # 1. 通常モデル（量子化なし）
    print("\n" + "=" * 70)
    print("1️⃣  通常モデル（量子化なし - BF16）")
    print("=" * 70)
    try:
        model_normal = info.get_moshi(
            device=device,
            lm_kwargs_overrides={"quantize": False}
        )
        memory_normal = get_model_memory(model_normal)
        print(f"✅ メモリ使用量: {format_memory(memory_normal)}")
        results.append(("BF16 (通常)", memory_normal, 1.0))
        del model_normal
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ エラー: {e}")
        memory_normal = None
    
    # 2. 8-bit量子化
    print("\n" + "=" * 70)
    print("2️⃣  8-bit量子化（INT8）")
    print("=" * 70)
    try:
        model_8bit = info.get_moshi(
            device=device,
            lm_kwargs_overrides={
                "quantize": True,
                "quantize_bits": 8
            }
        )
        memory_8bit = get_model_memory(model_8bit)
        reduction_8bit = ((memory_normal - memory_8bit) / memory_normal * 100) if memory_normal else 0
        print(f"✅ メモリ使用量: {format_memory(memory_8bit)}")
        if memory_normal:
            print(f"📊 削減率: {reduction_8bit:.1f}%")
        results.append(("INT8 (8-bit)", memory_8bit, memory_normal / memory_8bit if memory_normal else 0))
        del model_8bit
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"   bitsandbytesのインストールが必要: pip install bitsandbytes")
        memory_8bit = None
    
    # 3. 4-bit量子化
    print("\n" + "=" * 70)
    print("3️⃣  4-bit量子化（INT4 NF4） ⭐ 最軽量")
    print("=" * 70)
    try:
        model_4bit = info.get_moshi(
            device=device,
            lm_kwargs_overrides={
                "quantize": True,
                "quantize_bits": 4
            }
        )
        memory_4bit = get_model_memory(model_4bit)
        reduction_4bit = ((memory_normal - memory_4bit) / memory_normal * 100) if memory_normal else 0
        print(f"✅ メモリ使用量: {format_memory(memory_4bit)}")
        if memory_normal:
            print(f"📊 削減率: {reduction_4bit:.1f}%")
        results.append(("INT4 (4-bit)", memory_4bit, memory_normal / memory_4bit if memory_normal else 0))
        del model_4bit
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        memory_4bit = None
    
    # 結果の比較表
    if results:
        print("\n" + "=" * 70)
        print("📊 結果のまとめ")
        print("=" * 70)
        print(f"{'モード':<20} {'メモリ使用量':<15} {'削減率':<10} {'速度向上':<10}")
        print("-" * 70)
        
        baseline_memory = results[0][1]
        for name, memory, speedup in results:
            reduction = ((baseline_memory - memory) / baseline_memory * 100)
            speedup_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"
            print(f"{name:<20} {format_memory(memory):<15} {reduction:>6.1f}% {speedup_str:>10}")
    
    print("\n" + "=" * 70)
    print("💡 推奨事項")
    print("=" * 70)
    print("• 本番環境（高精度）: 8-bit量子化")
    print("• メモリ制約が厳しい: 4-bit量子化")
    print("• 開発/テスト: 4-bit量子化")
    print("• 精度最優先: 量子化なし（BF16）")
    
    print("\n" + "=" * 70)
    print("✅ デモ完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
