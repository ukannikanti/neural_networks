import torch
from config.transformer_config import TransformerConfig
from layers.attention import SelfAttention
from utils.enums import AttnMaskType
from fusions.fused_bias_dropout import get_bias_dropout_add
from layers.mlp import MLP
from core import parallel_state

class TransformerLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number + self._get_layer_offset()
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        factory_kwargs = {"device": torch.device('cuda'), "dtype": torch.float}
        self.input_layernorm = torch.nn.modules.normalization.LayerNorm(
            self.config.hidden_size, self.config.layernorm_epsilon,  **factory_kwargs
        )
        
        ## [Module 2: SelfAttention]
        self.self_attention = SelfAttention(config=self.config, layer_number=layer_number)

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = MLP(config=self.config)

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask
        )

        # MLP.
        mlp_output_with_bias = self.mlp(attention_output)

        return mlp_output_with_bias

   