from models.encoder import TransformerEncoderLayer
from torchinfo import summary
import torch

encoder = TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward = 1024)
summary(encoder, input_size=torch.rand((1, 768)).shape)