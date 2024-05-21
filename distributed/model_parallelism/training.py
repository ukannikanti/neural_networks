from core.initialize import initialize
from config.arguments import core_transformer_config_from_args
from config.global_vars import get_args
from layers.transformer_layer import TransformerLayer
from layers.parallel_linear import ColumnParallelLinear
import torch
from utils.enums import AttnMaskType

def train():
     print("Initializing the distributed environment to train large language models.. 20B+ Params")
     initialize()
     args = get_args()
     config = core_transformer_config_from_args(args)
     transformer_layer = TransformerLayer(config=config, layer_number=1)
     input = torch.rand(8, 1, 512).to(torch.cuda.current_device())
     output, bias = transformer_layer.forward(input, torch.ones(1).bool().to(torch.cuda.current_device()))
     print(output.shape)
     # column_parallel = ColumnParallelLinear(
     #        512,
     #        1536,
     #        config=config,
     #        init_method=config.init_method,
     #        gather_output=False
     #    )
     # output, _ = column_parallel.forward(input)

     # Define loss function
     # loss_function = torch.nn.MSELoss()

     # # Define optimizer
     # optimizer = torch.optim.SGD(column_parallel.parameters(), lr=0.01)
     
     # # Compute loss
     # loss = loss_function(output, torch.rand(8, 1, 1).to(torch.cuda.current_device()))

     # # Backward pass
     # optimizer.zero_grad()  # Zero the gradients to clear them before backward pass
     # loss.backward()  # Backpropagate the loss
     

train()


