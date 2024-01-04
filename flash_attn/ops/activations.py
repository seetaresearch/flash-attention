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
"""Activation operator."""

from dragon.vm.torch import autograd


class SwiGLUFunction(object):
    """Apply SwiGLU."""

    @staticmethod
    def apply(x):
        return autograd.Function.apply("SwiGLU", x.device, [x])


swiglu = SwiGLUFunction.apply
