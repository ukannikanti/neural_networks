# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum

class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
