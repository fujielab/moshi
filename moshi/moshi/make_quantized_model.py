# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from .client_utils import log
from .models import loaders


def main():
    parser = argparse.ArgumentParser(
        description="Create a quantized model checkpoint that can be loaded without re-quantization"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory where quantized model will be saved"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into, defaults to Moshiko. Use this to select a different pre-trained model."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to a local tokenizer file."
    )
    parser.add_argument(
        "--moshi-weight",
        type=str,
        help="Path to a local checkpoint file for Moshi."
    )
    parser.add_argument(
        "--mimi-weight",
        type=str,
        help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--lora-weight",
        type=str,
        default=None,
        help="Path to a local checkpoint file for LoRA."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a local config file."
    )
    parser.add_argument(
        "-q", "--quantize",
        type=int,
        choices=[4, 8],
        required=True,
        help="Quantize model weights to 4 or 8 bits."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run quantization, defaults to 'cuda'."
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Use float16 instead of bfloat16."
    )
    parser.add_argument(
        "--no-fuse-lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
        help="Do not fuse LoRA layers into Linear layers."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("info", "Retrieving checkpoint information")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        lora_weights=args.lora_weight,
        config_path=args.config_path
    )

    log("info", "Loading and quantizing Moshi model")
    lm_kwargs_overrides = {
        "quantize": True,
        "quantize_bits": args.quantize
    }
    lm = checkpoint_info.get_moshi(
        device=args.device,
        dtype=args.dtype,
        fuse_lora=args.fuse_lora,
        lm_kwargs_overrides=lm_kwargs_overrides
    )
    log("info", "Moshi model loaded and quantized")

    # Save the quantized model state dict in the format expected by get_moshi_lm
    moshi_path = output_dir / "moshi_quantized.pth"
    log("info", f"Saving quantized Moshi model to {moshi_path}")
    # Save in the checkpoint format expected by the loader
    torch.save({
        "fsdp_best_state": {
            "model": lm.state_dict()
        }
    }, moshi_path)
    log("info", "Quantized Moshi model saved")

    # Copy or save Mimi model
    log("info", "Loading Mimi model")
    mimi = checkpoint_info.get_mimi(device=args.device)
    mimi_path = output_dir / "mimi.pth"
    log("info", f"Saving Mimi model to {mimi_path}")
    # Save in the format expected by get_mimi (with "model" key)
    torch.save({"model": mimi.state_dict()}, mimi_path)
    log("info", "Mimi model saved")

    # Copy tokenizer
    tokenizer_path = output_dir / "tokenizer.spm"
    log("info", f"Copying tokenizer to {tokenizer_path}")
    if checkpoint_info.tokenizer is not None:
        shutil.copy(checkpoint_info.tokenizer, tokenizer_path)
    log("info", "Tokenizer copied")

    # Create a config file with all necessary information
    config = {
        "model_type": checkpoint_info.model_type,
        "quantize": True,
        "quantize_bits": args.quantize,
        "moshi_weight": str(moshi_path.absolute()),
        "mimi_weight": str(mimi_path.absolute()),
        "tokenizer": str(tokenizer_path.absolute()),
        "dtype": "float16" if args.dtype == torch.float16 else "bfloat16",
        "original_hf_repo": args.hf_repo,
        "lm_gen_config": checkpoint_info.lm_gen_config,
    }

    # Save original config if it exists
    if checkpoint_info.raw_config is not None:
        config["original_config"] = checkpoint_info.raw_config
    if checkpoint_info.lm_config is not None:
        config["lm_config"] = checkpoint_info.lm_config

    config_path = output_dir / "config.json"
    log("info", f"Saving configuration to {config_path}")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log("info", "Configuration saved")

    log("info", f"Quantized model successfully created in {output_dir}")
    log("info", f"To use this model, run:")
    log("info", f"  python -m moshi.server --quantized-model-dir {output_dir} --host 0.0.0.0")


if __name__ == "__main__":
    with torch.no_grad():
        main()
