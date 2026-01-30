"""
Preconditioner base class.
"""
import torch
import numpy as np

from unocg.problems import Problem


class Preconditioner:
    """
    Generic Preconditioner
    """
    def __init__(self, problem: Problem):
        self.problem = problem

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply preconditioner

        :param r:
        :return:
        """
        return r.clone()

    def apply_field(self, res_field: torch.Tensor) -> torch.Tensor:
        r = self.problem.reshape_vec(res_field)
        s = self.apply(r)
        return self.problem.reshape_field(s)

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return self.apply(r)

    def get_apply_module(self, dtype, device):
        """
        Build PyTorch model for apply operation

        Input shape for model: [B, L, C, N, N] or [B, L, C, N, N, N]
        Output shape: same as input shape
        """
        return torch.nn.Identity()
