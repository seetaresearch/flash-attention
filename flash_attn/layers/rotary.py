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
"""Rotary layers."""

from dragon.vm.torch import autograd


class ApplyRotaryEmb(object):
    """Apply rotary embedding."""

    @staticmethod
    def apply(q, k, cos, sin, inplace=False, data_format="NHLD"):
        return autograd.Function.apply(
            "Rotary",
            q.device,
            [q, k, cos, sin],
            outputs=[q, k] if inplace else [None, None],
            data_format=data_format,
        )

    autograd.Function.register(
        "Rotary",
        lambda **kwargs: {
            "data_format": kwargs.get("data_format", "NHLD"),
        },
    )


def apply_rotary_emb(q, k, cos, sin, inplace=False, data_format="NHLD"):
    """Apply rotary embedding.

    Parameters
    ----------
    q : dragon.vm.torch.Tensor
        The query tensor.
    k : dragon.vm.torch.Tensor
        The key tensor.
    cos : dragon.vm.torch.Tensor
        The cosine weight tensor.
    sin : dragon.vm.torch.Tensor
        The sine weight tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.
    data_format : str, optional, default='NHLD'
        ``'NHLD'`` or ``'NLHD'``.

    Returns
    -------
    Tuple[dragon.vm.torch.Tensor, dragon.vm.torch.Tensor]
        The output query and key tensor.

    """
    return ApplyRotaryEmb.apply(q, k, cos, sin, inplace, data_format)
