# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization based on bitsandbytes, supporting 4-bit and 8-bit quantization."""

import torch
from torch import nn
import torch.nn.functional as F


class QLinear8bit(nn.Module):
    """8-bit quantized linear layer using bitsandbytes."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        from bitsandbytes import functional as bnbF  # type: ignore
        weight = linear.weight
        assert weight.data.dtype.is_floating_point
        assert linear.bias is None
        CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))  # type: ignore
        self.weight = nn.Parameter(CB, requires_grad=False)
        self.weight_scb = nn.Parameter(SCB, requires_grad=False)

    def forward(self, x):
        import bitsandbytes as bnb  # type: ignore
        state = bnb.MatmulLtState()
        state.CB = self.weight  # type: ignore
        assert isinstance(state.CB, torch.Tensor)
        state.SCB = self.weight_scb  # type: ignore
        assert isinstance(state.SCB, torch.Tensor)
        if state.SCB.dtype != torch.float:
            raise RuntimeError(
                "Expected `weight_scb` to have type float, but got bfloat16. "
                "When using quantized models, care should be taken not to change the dtype of "
                "the model once initialized.")
        assert state.SCB.dtype == torch.float, state.SCB.dtype
        state.has_fp16_weights = False
        y = bnb.matmul(x.half(), state.CB, state=state)
        assert isinstance(y, torch.Tensor)
        return y


class QLinear4bit(nn.Module):
    """4-bit quantized linear layer using bitsandbytes.
    
    Note: This uses a simpler approach - storing quantized weights and dequantizing
    during forward pass. This is more compatible with model loading.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        try:
            import bitsandbytes as bnb  # type: ignore
            from bitsandbytes.functional import quantize_4bit, dequantize_4bit
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. "
                "Install it with: pip install bitsandbytes"
            )
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        original_device = linear.weight.device
        
        # 4-bit quantization requires CUDA
        # Move weight to CUDA if not already there
        weight = linear.weight.data
        if weight.device.type != 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit quantization requires CUDA, but CUDA is not available. "
                    "Please use 8-bit quantization instead or run on a GPU."
                )
            # Temporarily move to CUDA for quantization
            weight = weight.cuda()
        
        # Ensure weight is in float32 for quantization
        weight = weight.to(torch.float32)
        
        # Perform 4-bit quantization (must be on CUDA)
        quant_weight, quant_state = quantize_4bit(
            weight,
            quant_type="nf4",
            compress_statistics=True,
        )
        
        # Move back to original device if needed
        if original_device.type != 'cuda':
            quant_weight = quant_weight.to(original_device)
            if hasattr(quant_state, 'absmax') and quant_state.absmax is not None:
                quant_state.absmax = quant_state.absmax.to(original_device)
            if hasattr(quant_state, 'code') and quant_state.code is not None:
                quant_state.code = quant_state.code.to(original_device)
        
        # Store quantized weight and state
        self.register_buffer('quant_weight', quant_weight)
        # quant_state contains statistics needed for dequantization
        self.quant_state = quant_state
        
        self.compute_dtype = torch.bfloat16
        self.quantize_4bit = quantize_4bit
        self.dequantize_4bit = dequantize_4bit
        
    def forward(self, x):
        # Dequantize weights on-the-fly
        # Note: quant_weight is a buffer registered as Tensor
        quant_weight_tensor = self.quant_weight
        assert isinstance(quant_weight_tensor, torch.Tensor)
        weight_deq = self.dequantize_4bit(quant_weight_tensor, self.quant_state)
        weight_deq = weight_deq.to(self.compute_dtype).to(x.device)
        
        # Perform linear operation
        return F.linear(x.to(self.compute_dtype), weight_deq)


# Alias for backward compatibility
QLinear = QLinear8bit


def replace_linear_with_qlinear(module, bits=8):
    """Recursively replace all Linear layers with QLinear layers.
    
    Args:
        module: The module to quantize
        bits: Number of bits for quantization (4 or 8). Default is 8.
    """
    if bits not in [4, 8]:
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    
    QLinearClass = QLinear4bit if bits == 4 else QLinear8bit
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, QLinearClass(child))
        elif isinstance(child, (QLinear8bit, QLinear4bit)):
            # Slight issue with the way we implement things: the scale param
            # might get casted with the rest of the model to bfloat16, although
            # we most likely want to keep it as float. For the LM model we might call this function twice,
            # first layer by layer to avoid too big of a memory usage, and second, at the end
            # of the LM init, after all other modules are initialized and properly dtyped.
            # In any case that should happen before loading the state dict to avoid a loss of precision.
            if isinstance(child, QLinear8bit):
                child.float()
        else:
            replace_linear_with_qlinear(child, bits=bits)

