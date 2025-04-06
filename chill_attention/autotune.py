import itertools
import logging

import torch
import triton

logger = logging.getLogger(__name__)


_a100_fwd_default_config = {
    (torch.bfloat16, 16): (64, 64, 4, 4, False),
    (torch.bfloat16, 32): (16, 64, 4, 3, False),
    (torch.bfloat16, 64): (64, 64, 4, 3, False),
    (torch.bfloat16, 128): (32, 64, 4, 3, False),
    (torch.bfloat16, 256): (32, 64, 4, 2, True),
    #
    (torch.float16, 16): (64, 64, 4, 3, False),
    (torch.float16, 32): (16, 16, 1, 3, True),
    (torch.float16, 64): (16, 64, 1, 3, False),
    (torch.float16, 128): (32, 64, 4, 2, True),
    (torch.float16, 256): (32, 64, 4, 2, True),
    #
    (torch.float32, 16): (16, 64, 2, 1, True),
    (torch.float32, 32): (16, 64, 2, 1, True),
    (torch.float32, 64): (16, 32, 2, 2, False),
    (torch.float32, 128): (16, 64, 4, 2, False),
    (torch.float32, 256): (16, 64, 4, 2, True),
}
_h100_fwd_default_config = {
    (torch.bfloat16, 16): (16, 32, 1, 4, True),
    (torch.bfloat16, 32): (16, 16, 1, 4, True),
    (torch.bfloat16, 64): (16, 64, 2, 2, True),
    (torch.bfloat16, 128): (64, 64, 4, 1, True),
    (torch.bfloat16, 256): (32, 64, 4, 2, False),
    #
    (torch.float16, 16): (16, 32, 1, 3, False),
    (torch.float16, 32): (16, 16, 1, 2, False),
    (torch.float16, 64): (64, 64, 4, 2, False),
    (torch.float16, 128): (64, 64, 4, 1, True),
    (torch.float16, 256): (32, 64, 4, 2, True),
    #
    (torch.float32, 16): (16, 64, 2, 1, True),
    (torch.float32, 32): (16, 64, 2, 2, False),
    (torch.float32, 64): (16, 64, 2, 2, True),
}

_a100_bwd_default_config = {
    (torch.bfloat16, 16): (32, 16, 1, 3, True),
    (torch.bfloat16, 32): (64, 16, 4, 1, True),
    (torch.bfloat16, 64): (64, 16, 4, 1, False),
    (torch.bfloat16, 128): (32, 16, 2, 1, False),
    (torch.bfloat16, 256): (32, 16, 4, 1, False),
    #
    (torch.float16, 16): (32, 16, 1, 3, True),
    (torch.float16, 32): (64, 16, 4, 1, True),
    (torch.float16, 64): (64, 16, 4, 1, False),
    (torch.float16, 128): (32, 16, 2, 1, True),
    (torch.float16, 256): (32, 16, 4, 1, False),
    #
    (torch.float32, 16): (32, 32, 4, 1, False),
    (torch.float32, 32): (32, 32, 4, 2, True),
    (torch.float32, 64): (32, 32, 4, 1, True),
    (torch.float32, 128): (16, 16, 1, 1, True),
    (torch.float32, 256): (16, 16, 4, 1, True),
}
_h100_bwd_default_config = {
    (torch.bfloat16, 16): (32, 16, 1, 3, False),
    (torch.bfloat16, 32): (64, 16, 4, 3, False),
    (torch.bfloat16, 64): (64, 16, 4, 1, True),
    (torch.bfloat16, 128): (32, 32, 4, 1, False),
    (torch.bfloat16, 256): (32, 16, 4, 1, False),
    #
    (torch.float16, 16): (32, 16, 1, 3, False),
    (torch.float16, 32): (64, 16, 4, 3, False),
    (torch.float16, 64): (64, 16, 4, 1, True),
    (torch.float16, 128): (32, 32, 4, 1, False),
    (torch.float16, 256): (32, 16, 4, 1, False),
    #
    (torch.float32, 16): (32, 32, 4, 1, True),
    (torch.float32, 32): (32, 32, 4, 2, False),
    (torch.float32, 64): (32, 32, 4, 2, False),
    (torch.float32, 128): (32, 16, 2, 1, True),
    (torch.float32, 256): (32, 16, 4, 1, True),
}


def _get_default_config_fwd(head_dim, dtype) -> tuple[int, int, int, int, bool]:
    default_config = (16, 16, 4, 1, True)

    if torch.cuda.get_device_capability() >= (9, 0):  # H100
        default_config = _h100_fwd_default_config.get((dtype, head_dim), default_config)
    elif torch.cuda.get_device_capability() >= (8, 0):  # A100
        default_config = _a100_fwd_default_config.get((dtype, head_dim), default_config)

    return default_config


def _get_default_config_bwd(head_dim, dtype) -> tuple[int, int, int, int, bool]:
    default_config = (16, 16, 4, 1, False)

    if torch.cuda.get_device_capability() >= (9, 0):  # H100
        default_config = _h100_bwd_default_config.get((dtype, head_dim), default_config)
    elif torch.cuda.get_device_capability() >= (8, 0):  # A100
        default_config = _a100_bwd_default_config.get((dtype, head_dim), default_config)

    return default_config


def strides(t: torch.Tensor, expected_size=None):
    assert t is not None
    if expected_size is not None:
        assert t.ndim == expected_size
    return [t.stride(i) for i in range(t.ndim)]


def _get_forward_autotune_configs(head_dim, dtype):
    TILE_Q_SIZE, TILE_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = (
        _get_default_config_fwd(head_dim, dtype)
    )
    additional_q = [TILE_Q_SIZE * 2, TILE_Q_SIZE // 2]
    additional_k = [TILE_K_SIZE * 2, TILE_K_SIZE // 2]

    valid_size = lambda x: x >= 16 and x < 256
    additional_q = [TILE_Q_SIZE] + list(filter(valid_size, additional_q))
    additional_k = [TILE_K_SIZE] + list(filter(valid_size, additional_k))

    additional_pipe = [PIPELINING]
    if PIPELINING < 2 and (dtype != torch.float32 or head_dim < 128):
        additional_pipe.append(PIPELINING + 1)

    warps = [N_WARPS // 2, N_WARPS * 2]
    valid_size = lambda x: x >= 1 and x <= 4
    warps = [N_WARPS] + list(filter(valid_size, warps))

    results = []
    for q, k, pipe, TENSORS_PRELOAD, warp in itertools.product(
        additional_q, additional_k, additional_pipe, [True, False], warps
    ):
        results.append(
            triton.Config(
                dict(
                    TILE_Q_SIZE=q,
                    TILE_K_SIZE=k,
                    PIPELINING=pipe,
                    TENSORS_PRELOAD=TENSORS_PRELOAD,
                ),
                num_warps=warp,
                num_stages=pipe,
            )
        )
    return results


def _get_backward_autotune_configs(head_dim, dtype):
    TILE_DQ_Q_SIZE, TILE_DQ_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = (
        _get_default_config_bwd(head_dim, dtype=dtype)
    )
    additional_q = [TILE_DQ_Q_SIZE * 2, TILE_DQ_Q_SIZE // 2, TILE_DQ_Q_SIZE // 4]
    additional_k = [TILE_DQ_K_SIZE * 2, TILE_DQ_K_SIZE // 2, TILE_DQ_K_SIZE // 4]

    valid_size = lambda x: x >= 16 and x <= 64
    additional_q = [TILE_DQ_Q_SIZE] + list(filter(valid_size, additional_q))
    additional_k = [TILE_DQ_K_SIZE] + list(filter(valid_size, additional_k))

    additional_pipe = [PIPELINING]
    warps = [N_WARPS // 2]
    valid_size = lambda x: x >= 1 and x <= 8
    warps = [N_WARPS] + list(filter(valid_size, warps))

    results = []
    for q, k, pipe, TENSORS_PRELOAD, warp in itertools.product(
        additional_q, additional_k, additional_pipe, [True, False], warps
    ):
        results.append(
            triton.Config(
                dict(
                    TILE_DQ_Q_SIZE=q,
                    TILE_DQ_K_SIZE=k,
                    TILE_DK_Q_SIZE=q,
                    TILE_DK_K_SIZE=k,
                    PIPELINING=pipe,
                    TENSORS_PRELOAD=TENSORS_PRELOAD,
                ),
                num_warps=warp,
                num_stages=pipe,
            )
        )
    return results


def _prune_notfitting_configs(configs, kernel, args, kwargs, grid):
    def predicate(config: triton.Config):
        try:
            kwargs_config = (
                kwargs
                | {k: v for k, v in config.kwargs.items()}
                | dict(
                    num_stages=config.num_stages,
                    num_warps=config.num_warps,
                )
            )
            kernel_run = kernel
            while isinstance(kernel_run, triton.runtime.Heuristics):
                for v, heur in kernel_run.values.items():
                    kwargs_config[v] = heur(
                        {**dict(zip(kernel_run.arg_names, args)), **kwargs_config}
                    )
                kernel_run = kernel_run.fn
            kernel_run.warmup(
                *args,
                **kwargs_config,
                grid=grid,
            )
            return True
        except Exception as e:
            if "OutOfResources" in str(e):
                return False
            raise

    return list(filter(predicate, configs))
