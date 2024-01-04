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
"""RMS normalization operator."""

from dragon.vm import torch
from dragon.vm.torch import autograd
from dragon.vm.torch import nn


class RMSNormFn(object):
    """Apply RMS normalization."""

    @staticmethod
    def apply(x, weight, epsilon=1e-5):
        return autograd.Function.apply("RMSNorm", x.device, [x, weight], epsilon=float(epsilon))

    autograd.Function.register(
        "RMSNorm",
        lambda **kwargs: {
            "epsilon": kwargs.get("epsilon", 1e-5),
        },
    )


def rms_norm(x, weight, epsilon=1e-5):
    """Apply RMS normalization to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    epsilon : float, optional, default=1e-5
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return RMSNormFn.apply(x, weight, epsilon)


class RMSNorm(nn.Module):
    """RMS normalization module."""

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        r"""Create a ``RMSNorm`` module.

        Parameters
        ----------
        hidden_size : int
            The size of last dimenstion.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        device : dragon.vm.torch.device, optional
            The initialized device.
        dtype : str, optional
            The initialized dtype.

        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        param_args = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(hidden_size, **param_args))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
