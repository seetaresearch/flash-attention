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
"""Flash Attention API."""

from dragon.vm import torch
from dragon.vm.torch import autograd


class FlashAttnFunc(object):
    """Apply flash attention."""

    @staticmethod
    def apply(q, k, v, dropout_p=0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        return autograd.Function.apply(
            "FlashAttn",
            q.device,
            [q, k, v],
            softmax_scale=float(softmax_scale),
            dropout=float(dropout_p),
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )

    autograd.Function.register(
        "FlashAttn",
        lambda **kwargs: {
            "dropout": kwargs.get("dropout", 0.0),
            "softmax_scale": kwargs.get("softmax_scale", 1.0),
            "causal": kwargs.get("causal", False),
            "window_size_left": kwargs.get("window_size_left", -1),
            "window_size_right": kwargs.get("window_size_right", -1),
        },
    )


class FlashAttnWithKVCacheFunc(object):
    """Apply flash attention."""

    @staticmethod
    def apply(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        return autograd.Function.apply(
            "FlashAttn",
            q.device,
            [q, k_cache, v_cache, k, v, cache_seqlens],
            softmax_scale=float(softmax_scale),
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )


def flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False, window_size=(-1, -1)):
    """Compute flash attention.

    Parameters
    ----------
    q : dragon.vm.torch.Tensor
        The Q tensor.
    k : dragon.vm.torch.Tensor
        The K tensor.
    v : dragon.vm.torch.Tensor
        The V tensor.
    dropout_p : float, optional, default=0
        The probability to zero an attention element.
    softmax_scale : float, optional
        The scale factor to softmax.
    causal : bool, optional, default=False
        Apply causal masking or not.
    window_size : Tuple[int, int], optional
        The window size for local attention.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, causal, window_size)


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """Compute flash attention with packed KV.

    Parameters
    ----------
    q : dragon.vm.torch.Tensor
        The Q tensor.
    kv : dragon.vm.torch.Tensor
        The KV tensor.
    dropout_p : float, optional, default=0
        The probability to zero an attention element.
    softmax_scale : float, optional
        The scale factor to softmax.
    causal : bool, optional, default=False
        Apply causal masking or not.
    window_size : Tuple[int, int], optional
        The window size for local attention.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    k, v = kv.unbind(dim=2)
    return FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, causal, window_size)


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """Compute flash attention with packed QKV.

    Parameters
    ----------
    qkv : dragon.vm.torch.Tensor
        The QKV tensor.
    dropout_p : float, optional, default=0
        The probability to zero an attention element.
    softmax_scale : float, optional
        The scale factor to softmax.
    causal : bool, optional, default=False
        Apply causal masking or not.
    window_size : Tuple[int, int], optional
        The window size for local attention.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    q, k, v = qkv.unbind(dim=2)
    return FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, causal, window_size)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """Compute flash attention with KVCache.

    Parameters
    ----------
    q : dragon.vm.torch.Tensor
        The Q tensor.
    k_cache : dragon.vm.torch.Tensor
        The cache K tensor.
    v_cache : dragon.vm.torch.Tensor
        The cache V tensor.
    k : dragon.vm.torch.Tensor
        The new K tensor.
    v : dragon.vm.torch.Tensor
        The new V tensor.
    softmax_scale : float, optional
        The scale factor to softmax.
    causal : bool, optional, default=False
        Apply causal masking or not.
    window_size : Tuple[int, int], optional
        The window size for local attention.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = q.new_full((q.shape[0],), cache_seqlens, dtype=torch.int32)
    return FlashAttnWithKVCacheFunc.apply(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        softmax_scale,
        causal,
        window_size,
    )
