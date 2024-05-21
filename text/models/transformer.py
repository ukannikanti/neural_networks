import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, Any, Union, Callable
from models.encoder import TransformerEncoderLayer, EncoderStacks
from models.decoder import TransformerDecoderLayer, TransformerDecoder
from torch.nn.modules.normalization import LayerNorm
import math
import torch.optim as optim

def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(src: Tensor, batch_first: bool
) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        # batched: B, S, E if batch_first else S, B, E
        seq_len_pos = 1 if batch_first else 0
        return src_size[seq_len_pos]

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.05, max_seq_len: int = 5000, device = torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_dim, 2)* math.log(10000) / embedding_dim)
        pos = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        pos_embedding = torch.zeros((max_seq_len, embedding_dim), device)

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, device = torch.device('cpu')):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, device=device)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
class TransformerModel(nn.Module):
    r"""A transformer model.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """

    def __init__(
        self,
        src_vocab_size:int,
        target_vocab_size:int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=torch.device('cpu'),
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        
        # token embedding
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model, device=device)
        self.tgt_embedding = TokenEmbedding(target_vocab_size, d_model, device=device)

        # positional enocoding
        self.positional_embedding = PositionalEncoding(d_model, device=device)

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        encoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.encoder = EncoderStacks(encoder_layer, num_encoder_layers, encoder_norm)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        decoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )
        
        # apply linear layer to match with the target size.
        self.linear_layer = nn.Linear(d_model, target_vocab_size)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        
        # check batch shapes
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        
        # embeddings to convert the input tokens and output tokens to vectors of dimension d_model
        src_embeddings = self.positional_embedding(self.src_embedding(src))
        tgt_embeddings = self.positional_embedding(self.tgt_embedding(tgt)) 

        # check the d_model shape for src & tgt
        if src_embeddings.size(-1) != self.d_model or tgt_embeddings.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        
        # pass through the encoder
        encoder_output = self.encoder(
            src_embeddings,
            src_mask=src_mask,
            padding_mask=src_padding_mask,
            is_causal=src_is_causal,
        )
        
        # pass through the decoder
        decoder_output = self.decoder(
            tgt_embeddings,
            encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        
        output = self.linear_layer(decoder_output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

