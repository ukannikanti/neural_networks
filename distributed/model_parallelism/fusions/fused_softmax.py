# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn

from utils.enums import AttnMaskType
from utils.utils import get_default_causal_mask

class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Args:
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        attn_mask_type,
        mask_func,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.attn_mask_type = attn_mask_type
        self.mask_func = mask_func
        self.scale = scale


    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass of softmax with masked input.

        In case attn_mask_type is causal the mask is generated and None can be passed.
        A user-defined mask is only needed when attn_mask_type is not causal.
        """
        # [b, np, sq, sk]
        assert input.dim() == 4
        return self.forward_torch_softmax(input, mask)


    def forward_torch_softmax(self, input, mask):
        if self.scale is not None:
            input = input * self.scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            # If sq == 1 then either KV cache is used or one-element context is passed
            # so keeping mask=None in this case; subsequent code should handle it
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)
        return probs
