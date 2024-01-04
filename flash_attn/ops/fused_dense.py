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
"""Fused dense operator."""

from dragon.core import distributed
from dragon.vm import torch
from dragon.vm.torch import autograd
from dragon.vm.torch import nn


class FusedDenseFunc(object):
    """Apply fused dense."""

    @staticmethod
    def apply(
        x,
        weight,
        bias=None,
        epilogue="",
        pre_act=None,
        save_pre_act=False,
        process_group=None,
        col_parallel=False,
        row_parallel=False,
        seq_parallel=False,
    ):
        save_pre_act = save_pre_act if epilogue else False
        return autograd.Function.apply(
            "FusedDense",
            x.device,
            list(filter(lambda x: x is not None, [x, weight, bias, pre_act])),
            outputs=[None] + ([None] if save_pre_act else []),
            epilogue=epilogue.upper(),
            col_parallel=process_group is not None and col_parallel,
            row_parallel=process_group is not None and row_parallel,
            seq_parallel=process_group is not None and seq_parallel,
            **(process_group.arguments if process_group else {}),
        )

    autograd.Function.register(
        "FusedDense",
        lambda **kwargs: {
            "epilogue": kwargs.get("epilogue", ""),
            "col_parallel": kwargs.get("col_parallel", False),
            "row_parallel": kwargs.get("row_parallel", False),
            "seq_parallel": kwargs.get("seq_parallel", False),
            "comm": kwargs.get("comm", 0),
            "group": kwargs.get("group", 0),
            "ranks": kwargs.get("ranks", None),
        },
    )


def fused_dense_func(
    x,
    weight,
    bias=None,
    epilogue="",
    pre_act=None,
    save_pre_act=False,
    process_group=None,
    col_parallel=False,
    row_parallel=False,
    seq_parallel=False,
):
    """Compute fused dense function.

    Parameters
    ----------
    x : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    epilogue : str, optional, default=''
        The epilogue applied to output.
    pre_act : dragon.vm.torch.Tensor, optional
        The auxiliary tensor returned with epilogue.
    save_pre_act : bool, optional, default=False
        Save auxiliary tensor for epilogue or not.
    process_group : dragon.ProcessGroup, optional
        The group for communication.
    col_parallel : bool, optional, default=False
        Use column parallelism or not.
    row_parallel : bool, optional, default=False
        Use row parallelism or not.
    seq_parallel : bool, optional, default=False
        Use sequence parallelism or not.

    Returns
    -------
    Union[dragon.vm.torch.Tensor, Sequence[dragon.vm.torch.Tensor]]
        The output tensor and optional auxiliary tensor.

    """
    return FusedDenseFunc.apply(
        x,
        weight,
        bias,
        epilogue,
        pre_act,
        save_pre_act,
        process_group,
        col_parallel,
        row_parallel,
        seq_parallel,
    )


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear module."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        process_group=None,
        gather_output=False,
        sequence_parallel=False,
        device=None,
        dtype=None,
    ):
        """Create a ``ColumnParallelLinear`` module.

        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        process_group : ProcessGroup, optional
            The group for communication.
        gather_output : bool, optional, default=False
            Call all-gather on output or not.
        sequence_parallel : bool, optional, default=False
            Use sequence parallelism or not.
        device : dragon.vm.torch.device, optional
            The initialized device.
        dtype : str, optional
            The initialized dtype.

        """
        super(ColumnParallelLinear, self).__init__()
        self.process_group = process_group or distributed.get_group()
        self.parallel_size = self.process_group.size if self.process_group else 1
        self.in_features = in_features
        self.out_features = out_features // self.parallel_size
        param_args = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features, **param_args))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **param_args))
        else:
            self.bias = None
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, input):
        if input.device.type in ("cpu", "mps"):
            return nn.functional.linear(input, self.weight, self.bias)
        out = fused_dense_func(
            input,
            self.weight,
            bias=self.bias,
            process_group=self.process_group,
            col_parallel=True,
            seq_parallel=self.sequence_parallel,
        )
        if self.gather_output and self.process_group:
            out = torch.distributed.all_gather([], out, self.process_group)[0]
            out = out.permute(1, 2, 0, 3).flatten_(2)
        return out


class RowParallelLinear(nn.Module):
    """Row-parallel linear module."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        process_group=None,
        sequence_parallel=False,
        device=None,
        dtype=None,
    ):
        """Create a ``RowParallelLinear`` module.

        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        process_group : ProcessGroup, optional
            The group for communication.
        sequence_parallel : bool, optional, default=False
            Use sequence parallelism or not.
        device : dragon.vm.torch.device, optional
            The initialized device.
        dtype : str, optional
            The initialized dtype.

        """
        super(RowParallelLinear, self).__init__()
        self.process_group = process_group or distributed.get_group()
        self.parallel_size = self.process_group.size if self.process_group else 1
        self.in_features = in_features // self.parallel_size
        self.out_features = out_features
        param_args = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features, **param_args))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **param_args))
        else:
            self.bias = None
        self.sequence_parallel = sequence_parallel

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, input):
        if input.device.type in ("cpu", "mps"):
            return nn.functional.linear(input, self.weight, self.bias)
        return fused_dense_func(
            input,
            self.weight,
            bias=self.bias,
            process_group=self.process_group,
            row_parallel=True,
            seq_parallel=self.sequence_parallel,
        )
