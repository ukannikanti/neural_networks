import torch
from torch import nn, Tensor
import math 
from models.transformer import TransformerEncoderLayer, EncoderStacks
from typing import Optional, Any, Union, Callable
from torch.nn import functional as F

class BertConfig():
     def __init__(
        self,
        vocab_size=30522,
        d_model=768,
        num_encoder_layers=12,
        num_attention_heads=12,
        pad_token_id=0,
        n_segments=2,
        hidden_dropout_prob=0.1,
        hidden_act =  F.relu,
        layer_norm_eps:float = 1e-12, 
        norm_first:bool = True,
        batch_first:bool = True,
        bias:bool = True,
        add_pooling_layer: bool = True
    ):
        self.vocab_size=vocab_size,
        self.d_model=d_model,
        self.num_encoder_layer=num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.pad_token_id = pad_token_id
        self.n_segments = n_segments
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.batch_first = batch_first
        self.bias = bias
        self.add_pooling_layer = add_pooling_layer
       
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.05, max_seq_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_dim, 2)* math.log(10000) / embedding_dim)
        pos = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        pos_embedding = torch.zeros((max_seq_len, embedding_dim))

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        print("\n pos_embedding: ", self.pos_embedding[:token_embedding.size(0), :])
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__()
        # create a lookup tables that stores embeddings of a fixed dictionary and size.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.n_segments, config.d_model)
        self.position_embeddings = PositionalEncoding(config.d_model)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_seq_ids: torch.Tensor,
        input_seq_segment_ids: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.word_embeddings(input_seq_ids)
        segment_embeddings = self.segment_embeddings(input_seq_segment_ids)
        embeddings = self.position_embeddings(inputs_embeds + segment_embeddings)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.encoder_layer = TransformerEncoderLayer(
            config.d_model,
            config.num_attention_heads,
            config.d_model,
            config.hidden_dropout_prob,
            config.hidden_act,
            config.layer_norm_eps,
            config.batch_first,
            config.norm_first,
            config.bias
        )

        encoder_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps, bias=config.bias
        )
        self.encoder = EncoderStacks(self.encoder_layer, config.num_encoder_layer, encoder_norm)

    def forward(
        self,
        src_embeddings: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
    ) :
       return self.encoder(
            src_embeddings,
            src_mask=src_mask,
            padding_mask=src_padding_mask,
            is_causal=src_is_causal,
        )

class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.dim_feedforward, config.dim_feedforward)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if config.add_pooling_layer else None
        self.config = config

    def forward(self, input_seq_ids: torch.Tensor, input_seq_segment_ids: torch.Tensor, 
                src_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None,):
        embeddings = self.embeddings(input_seq_ids, input_seq_segment_ids)
        encoder_output = self.encoder(embeddings, src_mask, src_padding_mask, src_is_causal)
        pooled_output = self.pooler(encoder_output) if self.config.add_pooling_layer is not None else None
        return (pooled_output, encoder_output)

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.d_model, config.layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_scores = self.decoder(hidden_states)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertForPreTraining(nn.Module):
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

    def forward(
        self,
        input_seq_ids: torch.Tensor,
        input_seq_segment_ids: torch.Tensor, 
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_seq_ids,
            input_seq_segment_ids,
            src_mask,
            src_padding_mask,
            src_is_causal
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        return (prediction_scores, seq_relationship_score)
