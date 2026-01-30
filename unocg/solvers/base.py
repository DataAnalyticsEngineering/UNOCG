"""
Solver base class.
"""
from typing import Optional

import torch
from unocg.preconditioners import Preconditioner
from unocg.problems import Problem


class Solver:
    """
    Abstract Solver
    """

    def __init__(
            self,
            problem: Problem,
            preconditioner: Optional[Preconditioner] = None,
            rtol: float = 1e-10,
            atol: float = 0.0,
            max_iter: int = 10000,
            verbose: int = 0,
    ):
        """

        :param problem:
        :param preconditioner:
        :param rtol:
        :param atol:
        :param max_iter:
        :param verbose:
        """
        self.problem = problem
        if preconditioner is None:
            self.preconditioner = Preconditioner(self.problem)
        else:
            self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, param_fields: torch.Tensor, loadings: Optional[torch.Tensor] = None,
              state_fields: Optional[torch.Tensor] = None, guess: Optional[torch.Tensor] = None,
              rtol: float = 1e-10, atol: float = 0.0, max_iter: int = 10000, verbose: int = 0):
        raise NotImplementedError("Subclasses must implement this method")

    def get_prec_model(self, *args, **kwargs):
        try:
            return self.preconditioner.get_apply_module(*args, **kwargs)
        except AttributeError:
            return torch.nn.Identity()

    def zero_guess(self, param_fields, loadings):
        vec_shape = self.problem.get_vec_shape(param_fields, loadings=loadings)
        u0 = torch.zeros(vec_shape, dtype=param_fields.dtype, device=param_fields.device)
        return u0
