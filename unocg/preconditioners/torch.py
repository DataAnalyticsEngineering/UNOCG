"""
Preconditioner implementations in PyTorch
"""
import warnings
from typing import Union

import numpy as np
import scipy as sp
import torch

from unocg.problems import Problem
from unocg.preconditioners import Preconditioner
from unocg.transforms import Transform
from unocg.modules.preconditioners import FFTApplyModule, UnoApplyModule, JacApplyModule


class MLPreconditioner(Preconditioner):
    """
    Machine-learned Preconditioner

    The ML is applied on the field quantities.
    """
    def __init__(self, problem: Problem, model):
        super().__init__(problem)
        self.model = model

    def apply(self, r):
        res = self.problem.reshape_field(r)
        while res.ndim < 4:
            res = res.unsqueeze(0)
        s = self.model(res)

        return self.problem.reshape_vec(s)

    def get_apply_module(self, dtype, device):
        return self.model.to(dtype=dtype, device=device)

    def apply_sp(self, r: np.ndarray, device="cpu", dtype=torch.float64):
        r_torch = torch.tensor(r).to(device=device, dtype=dtype)
        s_torch = self.apply(r_torch)
        return s_torch.detach().cpu().numpy()


class JacobiPreconditioner(Preconditioner):
    """
    Jacobi Preconditioner
    """
    def __init__(self, problem: Problem, weights: torch.Tensor):
        """

        :param problem:
        """
        super().__init__(problem)

        self.weights = weights
        self.weights_sp = weights.detach().cpu().numpy()

    def apply(self, r):
        """

        :param r:
        :return:
        """
        s = self.weights * r
        return s
    
    def apply_sp(self, r, device=None, dtype=None):
        s = self.weights_sp * r
        return s
    
    def get_apply_module(self, dtype, device):
        """
        Build PyTorch model for apply operation

        Input shape for model: [B, L, C, N, N] or [B, L, C, N, N, N]
        Output shape: same as input shape
        """
        weights = self.problem.reshape_field(self.weights.to(dtype=dtype, device=device))
        model = JacApplyModule(weights)
        return model


class FFTPreconditioner(Preconditioner):
    """
    FFT Preconditioner
    """
    def __init__(self, problem: Problem, weights: Union[torch.Tensor, np.ndarray], use_real_fft=True, sparse=False, use_real_numbers=False):
        """
        Constructor

        :param problem:
        :param weights:
        :param use_real_fft:
        :param sparse:
        :param use_real_numbers:
        """
        super().__init__(problem)
        if isinstance(weights, torch.Tensor):
            self.weights = weights
            self.weights_sp = weights.cpu().detach().numpy()
        else:
            self.weights_sp = weights
            self.weights = torch.tensor(weights)

        while self.weights.ndim < self.problem.n_dim + 2:
            self.weights = self.weights.unsqueeze(0)

        while self.weights_sp.ndim < self.problem.n_dim + 2:
            self.weights_sp = np.expand_dims(self.weights_sp, axis=0)

        self.use_real_fft = use_real_fft
        self.sparse = sparse
        self.use_real_numbers = use_real_numbers
        self.n_dims = self.weights.ndim - 2
        if self.n_dims == 1:
            self.fft_dims = [-1]
            self.einsum_dims = "x"
            self.use_real_numbers = False
        elif self.n_dims == 2:
            self.fft_dims = [-2, -1]
            self.einsum_dims = "yx"
        elif self.n_dims == 3:
            self.fft_dims = [-3, -2, -1]
            self.einsum_dims = "zyx"
            self.use_real_numbers = False
        else:
            raise NotImplementedError(f"Dimension {self.n_dims} is not supported")

        if self.use_real_fft:
            self.fftn = torch.fft.rfftn
            self.ifftn = torch.fft.irfftn
            self.fftn_sp = sp.fft.rfftn
            self.ifftn_sp = sp.fft.irfftn
        else:
            self.fftn = torch.fft.fftn
            self.ifftn = torch.fft.ifftn
            self.fftn_sp = sp.fft.fftn
            self.ifftn_sp = sp.fft.ifftn

        self.complex_type = Problem.get_complex_dtype(self.weights.dtype)

        # PyTorch UserWarning for sparse matrices can be ignored
        if self.sparse:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.prec_matrix = self.assemble_fourier_matrix(real_fft=self.use_real_fft)
                self.prec_matrix = self.prec_matrix.to(dtype=self.complex_type, device=self.weights.device).to_sparse_csr()
                self.prec_matrix_sp = self.problem.torch_sparse_to_sp(self.prec_matrix.cpu()).tocsr()

        # Assemble permutation matrix fft(x) -> ifft(x) only when needed
        self.fft_perm_matrix = None

    @classmethod
    def from_matrix(cls, problem: Problem, prec_matrix: torch.Tensor):
        prec = super().__init__(problem)
        prec.prec_matrix = prec_matrix
        return prec

    def apply(self, r):
        """
        Apply FFT preconditioner

        :param r:
        :return:
        """
        res = self.problem.reshape_field(r)
        res_hat = self.fftn(res, dim=self.fft_dims)

        if self.sparse:
            self.prec_matrix = self.prec_matrix.to(device=r.device)
            r_hat = res_hat.flatten(start_dim=self.problem.ch_dim)
            s_hat = torch.reshape((self.prec_matrix @ r_hat.flatten(end_dim=-2).T).T, res_hat.shape)
        else:
            s_hat = torch.einsum("ij" + self.einsum_dims + ",...j" + self.einsum_dims + "->...i" + self.einsum_dims,
                                 self.weights.to(dtype=res_hat.dtype, device=res_hat.device), res_hat)

        s = self.ifftn(s_hat, dim=self.fft_dims, s=res.shape[-self.problem.n_dim:])
        s_vec = self.problem.reshape_vec(s)
        return s_vec.real
    
    def get_apply_module(self, dtype, device):
        """
        Build PyTorch model for apply operation

        Input shape for model: [B, L, C, N, N] or [B, L, C, N, N, N]
        Output shape: same as input shape
        """
        complex_dtype = self.problem.get_complex_dtype(dtype)
        weights = self.weights.to(dtype=dtype, device=device)
        field_shape = (-1, self.problem.n_channels, *self.problem.dof_shape)
        model = FFTApplyModule(self.fft_dims, weights, field_shape, self.fftn, self.ifftn)
        return model

    def apply_sp(self, r: np.ndarray, device=None, dtype=None) -> np.ndarray:
        """
        Apply the FFT preconditioner to the residual field to obtain the preconditioned residual s

        :param r: residual
        :return: preconditioned residual s
        """
        res = self.problem.reshape_field(r)

        res_hat = self.fftn_sp(res, axes=self.fft_dims)
        r_hat = res_hat.reshape((*res_hat.shape[:self.problem.ch_dim], self.prec_matrix_sp.shape[0]))
        s_hat = np.reshape((self.prec_matrix_sp @ r_hat.reshape((-1, self.prec_matrix_sp.shape[0])).T).T, res_hat.shape)
        s = self.ifftn_sp(s_hat, axes=self.fft_dims).real
        s_vec = self.problem.reshape_vec(s)

        return s_vec

    def assemble_fourier_matrix_sp(self, real_fft=False):
        """
        Produces a sparse matrix containing the weights that can be applied in Fourier space

        :return:
        """
        out_ch = self.weights.shape[0]
        in_ch = self.weights.shape[1]
        N = self.weights.shape[2]
        D_blocks = []
        for i in range(out_ch):
            D_blocks_row = []
            for j in range(in_ch):
                D_block = sp.sparse.diags_array([self.weights_sp[i, j].ravel()], offsets=[0])
                D_blocks_row.append(D_block)
            D_blocks.append(D_blocks_row)
        D = sp.sparse.block_array(D_blocks)

        if real_fft and False:
            idx = torch.arange(self.problem.n_dof)
            idx_grid = self.problem.reshape_field(idx)
            idx_grid_half = idx_grid[..., :(N // 2 + 1)]
            idx_half = idx_grid_half.ravel().cpu().detach().numpy()
            D = (D.tocsr()[idx_half, :].tocsc()[:, idx_half]).tocsr()
        return D

    def assemble_fourier_matrix(self, real_fft=False):
        return Problem.sp_sparse_to_torch(self.assemble_fourier_matrix_sp(real_fft))


class FansPreconditioner(FFTPreconditioner):
    """
    Fourier-Accelerated Nodal Solvers (FANS) as Preconditioner
    """
    def __init__(self, problem: Problem, stiff_ref: torch.Tensor, *args, **kwargs):
        weights = problem.compute_fundamental_solution(stiff_ref)
        super().__init__(problem=problem, weights=weights, use_real_fft=True, *args, **kwargs)


class UnoPreconditioner(FFTPreconditioner):
    """
    Unitory Neural Operator as Preconditioner
    """
    def __init__(self, problem: Problem, transform: Transform, weights: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        super().__init__(problem=problem, weights=weights, sparse=False, use_real_fft=False, *args, **kwargs)
        self.transform = transform
        self.diag = (weights.shape[0] != weights.shape[1])
        self.einsum_dims = self.problem.einsum_dims

    def apply(self, r):
        res = self.problem.reshape_field(r)
        res_hat = self.transform.transform(res)

        if self.diag:
            s_hat = self.weights.to(dtype=res_hat.dtype, device=res_hat.device)[0] * res_hat
        else:
            s_hat = torch.einsum("ij" + self.einsum_dims + ",...j" + self.einsum_dims + "->...i" + self.einsum_dims,
                                 self.weights.to(dtype=res_hat.dtype, device=res_hat.device), res_hat)

        s = self.transform.inverse_transform(s_hat, s=res.shape)
        s_vec = self.problem.reshape_vec(s)
        return s_vec.real

    def get_apply_module(self, dtype, device):
        """
        Build PyTorch model for apply operation

        Input shape for model: [B, L, C, N, N] or [B, L, C, N, N, N]
        Output shape: same as input shape
        """
        model = UnoApplyModule(self.problem, self.transform, self.weights.to(dtype=dtype, device=device))
        return model
