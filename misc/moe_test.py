import torch
from functools import partial
from megablocks.layers.arguments import Arguments
from megablocks.layers import moe
import torch


bs = 1
sl = 9
hs = 512
num_experts = 3
top_k = 1

init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
args = Arguments(
        hidden_size=hs,
        ffn_hidden_size=hs * 2,
        moe_num_experts=num_experts,
        moe_capacity_factor=1,
        moe_top_k=top_k,
        init_method=init_method)

x = torch.randn(sl, bs, hs).half().cuda()

moe = moe.MoE(args)
out, _ = moe(x)

print(x.shape, out.shape)