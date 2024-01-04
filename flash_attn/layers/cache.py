# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Cache layers."""

from dragon.vm import torch
from dragon.vm.torch import autograd


class ApplyKVCache(object):
    """Apply KVCache."""

    @staticmethod
    def apply(k, v, cache_k, cache_v, seq_len, data_format="NHLD"):
        return autograd.Function.apply(
            "KVCache",
            k.device,
            [k, v, cache_k, cache_v, seq_len],
            outputs=[None, None],
            data_format=data_format,
        )

    autograd.Function.register(
        "KVCache",
        lambda **kwargs: {
            "check_device": False,  # ``seq_len`` is a cpu tensor.
            "data_format": kwargs.get("data_format", "NHLD"),
        },
    )


def apply_kv_cache(k, v, cache_k, cache_v, seq_len, data_format="NHLD"):
    """Write keys and values to cache then return the appended buffer.

    Parameters
    ----------
    k : dragon.vm.torch.Tensor
        The key tensor.
    v : dragon.vm.torch.Tensor
        The value tensor.
    cache_k : dragon.vm.torch.Tensor
        The cache key tensor.
    cache_v : dragon.vm.torch.Tensor
        The cache value tensor.
    seq_len : Union[int, dragon.vm.torch.Tensor]
        The filled sequence length of cache.
    data_format : str, optional, default='NHLD'
        ``'NHLD'`` or ``'NLHD'``.

    Returns
    -------
    Tuple[dragon.vm.torch.Tensor, dragon.vm.torch.Tensor]
        The output key and value tensor.

    """
    if isinstance(seq_len, int):
        seq_len = torch.tensor([seq_len], dtype=torch.int32)
    return ApplyKVCache.apply(k, v, cache_k, cache_v, seq_len, data_format)
