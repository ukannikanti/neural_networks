import torch
from datetime import timedelta
from core.parallel_state import (
    model_parallel_is_initialized, 
    initialize_model_parallel, 
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size)
from config.arguments import parse_args
from config.global_vars import get_args, set_global_variables

def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()
    device_count = torch.cuda.device_count()

    if torch.distributed.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                order='tp-pp-dp',
            )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{get_pipeline_model_parallel_world_size()}"
                )


def initialize(
    extra_args_provider=None,
    ignore_unknown_args=False):
    # Make sure cuda is available.
    assert torch.cuda.is_available(), "Requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # Pytorch distributed.
    _initialize_distributed()