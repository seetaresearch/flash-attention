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
"""Embedding modules."""

from dragon.core import distributed
from dragon.vm import torch
from dragon.vm.torch import nn


class ColumnParallelEmbedding(nn.Module):
    """Column-parallel lookup-embedding module."""

    def __init__(self, num_embeddings, embedding_dim, process_group=None, device=None, dtype=None):
        """Create an ``ColumnParallelEmbedding`` module.

        Parameters
        ----------
        num_embeddings : int
            The dictionary size.
        embedding_dim : int
            The embedding dimension.
        process_group : ProcessGroup, optional
            The group for communication.
        device : dragon.vm.torch.device, optional
            The initialized device.
        dtype : str, optional
            The initialized dtype.

        """
        super(ColumnParallelEmbedding, self).__init__()
        self.process_group = process_group or distributed.get_group()
        parallel_size = self.process_group.size if self.process_group else 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim // parallel_size
        param_args = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, self.embedding_dim, **param_args))

    def forward(self, input):
        out = nn.functional.embedding(input, self.weight)
        if self.process_group:
            out = torch.distributed.all_gather([], out, self.process_group)[0]
            out = out.permute(1, 2, 0, 3).flatten_(2)
        return out
