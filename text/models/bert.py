import torch
from torch import nn
import math
from transformer import TransformerEncoderLayer, EncoderStacks
from torch.nn.modules.normalization import LayerNorm

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len:int):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embedding_dim).float()
        pe.require_grad = False

        for pos in range(max_seq_len):   
            # for each dimension of the each position
            for i in range(0, embedding_dim, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   

    def forward(self):
        return self.pe


class BertEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, embedding_dim, seq_len, n_segments, padding_idx, dropout=0.05):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx) # token embedding
        self.segment_embeddings = nn.Embedding(n_segments, embedding_dim) # segment embedding
        self.position_embeddings = PositionalEmbedding(embedding_dim=embedding_dim, max_seq_len=seq_len) # position embedding
        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids) + self.segment_embeddings(self.n_segments) + self.position_embeddings()
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BERTModel(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=768, n_layers=12, heads=12, dropout=0.1, layer_norm_eps: float = 1e-5,
                 bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.heads = heads
        self.feed_forward_hidden = embedding_dim * 4
        self.embedding = BertEmbedding(vocab_size, embedding_dim, 0, 64, 1, dropout=0.05)
        encoder_layer = TransformerEncoderLayer(embedding_dim, heads, self.feed_forward_hidden, dropout, batch_first=True)
        encoder_norm = LayerNorm(embedding_dim, eps=layer_norm_eps, bias=bias)
        self.encoder = EncoderStacks(encoder_layer, n_layers, encoder_norm)

    def forward(self, input_ids):
        # embedding the indexed sequence to sequence of vectors
        embedding_inputs = self.embedding(input_ids)
        output = self.encoder(embedding_inputs)
        return output

if __name__ == "__main__":
    # run the bert model with const inputs. 
    model = BERTModel(1000, 512, 12, 8)
    model(torch.rand([1, 2, 512]))