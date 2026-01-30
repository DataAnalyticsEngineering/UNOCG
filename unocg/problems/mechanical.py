"""
Mechanical problem formulations
"""
import math
from typing import Optional, Union, Tuple, Sequence, Collection, Dict
import warnings
import torch
import torch.nn.functional as F
from unocg.problems import Problem, BC
from unocg.modules.operators import MatvecModule, ResidualModule, RhsModule, FieldModule
from unocg.materials import Material
from unocg.training.losses.mechanical import DispLoss, StressLoss
import math


class MechanicalProblem(Problem):
    """
    Mechanical problem for small strains
    """

    def __init__(
            self,
            n_grid: Union[torch.Size, Sequence[int]],
            material: Material,
            loadings: Optional[torch.Tensor] = None,
            quad_degree: int = 2,
            bc: Optional[BC] = None,
            lengths = None,
    ):
        """
        Initialize the mechanical problem.

        :param n_grid: dimensions of the grid
        :type n_grid: Union[torch.Size, Sequence[int]]
        :param material: material model
        :type material: Material
        :param loadings: load cases (macroscopic strain tensors in Mandel notation)
        :type loadings: torch.Tensor
        :param quad_degree: quadrature degree (1: reduced integration HEX8R, 2: full integration HEX8)
        :type quad_degree: int
        :param bc: boundary conditions (default: BC.PERIODIC)
        :type bc: BC
        """
        super().__init__(n_grid=n_grid, bc=bc, quad_degree=quad_degree, lengths=lengths, material=material)
        self._n_channels = self.n_dim

        if loadings is None:
            n_loadings = self.strain_dims
            self.loadings = torch.eye(n_loadings)
        else:
            assert loadings.ndim >= 2
            assert loadings.shape[1] == self.strain_dims
            self.loadings = loadings

        if self.bc == BC.PERIODIC:
            self.dof_shape = self.n_grid
        elif self.bc == BC.DIRICHLET:
            self.dof_shape = tuple([n_axis - 1 for n_axis in self.n_grid])
        elif self.bc == BC.DIRICHLET_LR:
            self.dof_shape = (self.n_grid[0] - 1, *self.n_grid[1:])
        elif self.bc == BC.DIRICHLET_TB:
            self.dof_shape = (*self.n_grid[:-2], self.n_grid[-2] - 1, self.n_grid[-1])
        elif self.bc == BC.DIRICHLET_FB:
            self.dof_shape = (*self.n_grid[:2], self.n_grid[2] - 1, *self.n_grid[3:])
        elif self.bc == BC.NEUMANN:
            self.dof_shape = tuple([n_axis + 1 for n_axis in self.n_grid])
        else:
            raise NotImplementedError()
        self._n_dof = self.n_channels * torch.prod(torch.tensor(self.dof_shape))

        # Precompute kernels for residual computation
        self.grad_operator = self.get_grad_operator()
        self.grad_kernels = self.get_grad_kernels()
        self.div_kernels = self.get_div_kernels()

        self.Idx = None
        self.Lf = None

    def compute_residual(self,
                         u: torch.Tensor,
                         param_fields: Optional[torch.Tensor] = None,
                         loadings: Optional[torch.Tensor] = None,
                         state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute residual

        :param u: vector containing the displacement DOFs (batches of vectors are possible)
        :param param_fields: parameter fields (batches of fields are possible)
        :param loadings: loading vector (batches of loading vectors are possible)
        :param state_fields: state fields (history variables corresponding to parameter fields)
        :return:
        """
        if param_fields is None:
            raise ValueError()

        # Move kernels to correct device
        args = {"device": u.device, "dtype": u.dtype}
        self.grad_kernels = self.grad_kernels.to(**args)
        self.div_kernels = self.div_kernels.to(**args)

        rhs = torch.zeros_like(self.reshape_field(u))
        for gp_i in range(self.n_gauss):
            strain = self.grad(self.reshape_field(u), gp_i=gp_i)
            strain = strain + loadings[(slice(None), None, slice(None)) + self.expand_dims].to(**args)
            stress, state_fields_new = self.material_law(strain, param_fields=param_fields, state_fields=state_fields)
            rhs -= self.div(stress, gp_i=gp_i)

        r = self.reshape_vec(rhs)

        return r, state_fields_new

    def get_residual_module(self, dtype, device, field=True) -> torch.nn.Module:
        grad_module = self.get_grad_module(dtype=dtype, device=device)
        div_module = self.get_div_module(dtype=dtype, device=device)

        model = ResidualModule(self.n_dim, grad_module, self.material, div_module)
        return model

    def matvec(self,
               d: torch.Tensor,
               param_fields: Optional[torch.Tensor] = None,
               state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Matrix-vector product (only meaningful for linear problems)

        :param d:
        :param param_fields:
        :param state_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()

        # Move kernels to correct device
        self.grad_kernels = self.grad_kernels.to(device=d.device, dtype=self.dtype)
        self.div_kernels = self.div_kernels.to(device=d.device, dtype=self.dtype)

        p = torch.zeros_like(self.reshape_field(d))
        for gp_i in range(self.n_gauss):
            strain = self.grad(self.reshape_field(d), gp_i=gp_i)
            stress, state_fields_new = self.material_law(strain, param_fields=param_fields, state_fields=state_fields)
            p += self.div(stress, gp_i=gp_i)

        return self.reshape_vec(p), state_fields_new

    def lin_matvec(self,
               u: torch.Tensor,
               d: torch.Tensor,
               param_fields: Optional[torch.Tensor] = None,
               loadings: Optional[torch.Tensor] = None,
               state_fields: Optional[torch.Tensor] = None):
        """
        Matrix-vector product with linearized stiffness matrix (for nonlinear problems)
        """
        matvec_model = self.get_lin_matvec_module(dtype=d.dtype, device=d.device, u=u, loadings=loadings)

        d_field = self.reshape_field(d)
        p_field = matvec_model(d_field, param_fields)
        return self.reshape_vec(p_field), None

    def get_lin_matvec_module(self, dtype, device, u, loadings) -> torch.nn.Module:
        grad_module = self.get_grad_module(dtype=dtype, device=device)
        div_module = self.get_div_module(dtype=dtype, device=device)

        u_field = self.reshape_field(u)
        model = LinMatvecModule(self.n_dim, grad_module, self.material, div_module,
                                x0=u_field, loadings=loadings)
        return model

    def get_matvec_module(self, dtype, device) -> torch.nn.Module:
        grad_module = self.get_grad_module(dtype=dtype, device=device)
        div_module = self.get_div_module(dtype=dtype, device=device)

        module = MatvecModule(self.n_dim, grad_module, self.material, div_module)
        return module

    def get_grad_module(self, dtype, device) -> torch.nn.Module:
        grad_kernels = self.grad_kernels.to(dtype=dtype, device=device)
        if self.n_dim == 2:
            conv_grad = torch.nn.Conv2d(self.n_channels, self.strain_dims * self.n_gauss, grad_kernels.shape[-self.n_dim:],
                                            bias=False, device=device, dtype=dtype)
            conv_grad.weight = torch.nn.Parameter(grad_kernels.flatten(end_dim=-(2 + self.n_dim)), requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad2d((0, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad2d((1, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad2d((0, 0, 0, 1)), torch.nn.ZeroPad2d((1, 1, 0, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad2d((0, 1, 0, 0)), torch.nn.ZeroPad2d((0, 0, 1, 1))]
            elif self.bc == BC.NEUMANN:
                padding_layers = []
            else:
                NotImplementedError()
            
        elif self.n_dim == 3:
            conv_grad = torch.nn.Conv3d(self.n_channels, self.strain_dims * self.n_gauss, grad_kernels.shape[-self.n_dim:],
                                            bias=False, device=device, dtype=dtype)
            conv_grad.weight = torch.nn.Parameter(grad_kernels.flatten(end_dim=-(2 + self.n_dim)), requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad3d((0, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad3d((1, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad3d((0, 0, 0, 1, 0, 1)), torch.nn.ZeroPad3d((1, 1, 0, 0, 0, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad3d((0, 1, 0, 0, 0, 1)), torch.nn.ZeroPad3d((0, 0, 1, 1, 0, 0))]
            elif self.bc == BC.DIRICHLET_FB:
                padding_layers = [torch.nn.CircularPad3d((0, 1, 0, 1, 0, 0)), torch.nn.ZeroPad3d((0, 0, 0, 0, 1, 1))]
            elif self.bc == BC.NEUMANN:
                padding_layers = []
            else:
                NotImplementedError()
            
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        module = torch.nn.Sequential(
            *padding_layers,
            conv_grad,
            torch.nn.Unflatten(1, (self.n_gauss, self.strain_dims)),
        )
        for param in module.parameters():
            param.requires_grad = False
        return module

    def get_div_module(self, dtype, device) -> torch.nn.Module:
        div_kernels = self.div_kernels.to(dtype=dtype, device=device)
        if self.n_dim == 2:
            Conv = torch.nn.Conv2d
            conv_div = Conv(self.strain_dims * self.n_gauss, self.n_channels, div_kernels.shape[-self.n_dim:],
                            bias=False, device=device, dtype=dtype)
            vol_e = 1.0 / torch.tensor(self.n_grid).prod()
            conv_div_weight = div_kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2) * vol_e / div_kernels.shape[0]
            conv_div.weight = torch.nn.Parameter(conv_div_weight, requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad2d((1, 0) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = []
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad2d((0, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad2d((1, 0, 0, 0))]
            elif self.bc == BC.NEUMANN:
                padding_layers = [torch.nn.ZeroPad2d((1, 1, 1, 1))]
            else:
                NotImplementedError()
        
        elif self.n_dim == 3:
            Conv = torch.nn.Conv3d
            conv_div = Conv(self.strain_dims * self.n_gauss, self.n_channels, div_kernels.shape[-self.n_dim:],
                            bias=False, device=device, dtype=dtype)
            vol_e = 1.0 / torch.tensor(self.n_grid).prod()
            conv_div_weight = div_kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2) * vol_e / div_kernels.shape[0]
            conv_div.weight = torch.nn.Parameter(conv_div_weight, requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad3d((1, 0) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = []
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad3d((1, 0, 0, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad3d((0, 0, 1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_FB:
                padding_layers = [torch.nn.CircularPad3d((1, 0, 1, 0, 0, 0))]
            elif self.bc == BC.NEUMANN:
                padding_layers = [torch.nn.ZeroPad3d((1, 1, 1, 1, 1, 1))]
            else:
                NotImplementedError()
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
            
        module = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=2),
            *padding_layers,
            conv_div,
        )
        for param in module.parameters():
            param.requires_grad = False
        return module

    def material_law(self,
                     strain: torch.Tensor,
                     param_fields: Optional[torch.Tensor] = None,
                     state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        :param strain:
        :param param_fields:
        :param state_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()
        return self.material(strain, param_fields, state_fields)

    def tangent(self, strain: torch.Tensor, param_fields: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the element-wise (local) tangent based on a given strain field and material parameter field

        :param strain:
        :param param_fields:
        :return:
        """
        args = {"device": strain.device, "dtype": strain.dtype}

        if param_fields is None:
            raise ValueError()
        return self.material.get_tangent(strain, param_fields, state_fields)

    def tangent_vp(self, strain0: torch.Tensor, strain: torch.Tensor, param_fields: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the element-wise (local) tangent based on a given strain field and material parameter field

        :param strain:
        :param param_fields:
        :return:
        """
        args = {"device": strain.device, "dtype": strain.dtype}

        if param_fields is None:
            raise ValueError()
        return self.material.get_tangent_vp(strain0, strain, param_fields, state_fields)

    def compute_field(
            self,
            u: torch.Tensor,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        args = {"dtype": u.dtype, "device": u.device}

        if param_fields is None:
            raise ValueError()
        param_fields = param_fields.to(**args)

        if loadings is None:
            loadings = self.loadings

        disp = self.reshape_field(u)
        strain_shape = (*disp.shape[:self.ch_dim], self.strain_dims, *self.n_grid)
        stress = torch.zeros(strain_shape, **args)

        for gp_i in range(self.n_gauss):
            strain = self.grad(disp, gp_i=gp_i) + loadings[(slice(None), None, slice(None)) + self.expand_dims].to(**args)
            stress = stress + self.material_law(strain, param_fields=param_fields, state_fields=state_fields)[0].squeeze(self.ch_dim - 1)
        stress = stress / self.n_gauss
        
        if self.n_dim == 2:
            if self.bc == BC.DIRICHLET:
                disp = F.pad(disp, [1, 0, 1, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_LR:
                disp = F.pad(disp, [1, 0, 0, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_TB:
                disp = F.pad(disp, [0, 0, 1, 0], mode="constant", value=0.0)
        elif self.n_dim == 3:
            if self.bc == BC.DIRICHLET:
                disp = F.pad(disp, [1, 0, 1, 0, 1, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_LR:
                disp = F.pad(disp, [1, 0, 0, 0, 0, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_TB:
                disp = F.pad(disp, [0, 0, 1, 0, 0, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_FB:
                disp = F.pad(disp, [0, 0, 0, 0, 1, 0], mode="constant", value=0.0)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

        field = torch.cat([disp, stress], dim=self.ch_dim)
        return field
    
    def get_field_module(self, dtype, device):
        padding_layers = []
        # TODO: generalize
        if self.n_dim == 2:
            if self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad2d((1,0,1,0))]
        elif self.n_dim == 3:
            if self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad3d((1,0,1,0,1,0))]
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} not implemented")
        return FieldModule(
            ch_dim=self.ch_dim,
            expand_dims=self.expand_dims,
            grad_module=self.get_grad_module(dtype=dtype, device=device),
            model=self.material,
            padding_layers=padding_layers,
        )
    
    def get_tangent_module(self, dtype, device):
        return TangentModule(
            ch_dim=self.ch_dim,
            expand_dims=self.expand_dims,
            grad_module=self.get_grad_module(dtype=dtype, device=device),
            model=self.material,
        )

    def get_tangent_rhs_model(self, dtype, device):
        return TangentRhsModule(
            ch_dim=self.ch_dim,
            tangent_module=self.get_tangent_module(dtype=dtype, device=device),
            div_module=self.get_div_module(dtype=dtype, device=device)
        )
    
    def get_tangent_problem(self, disp: torch.Tensor, param_fields: torch.Tensor, loadings: torch.Tensor):
        return MechanicalTangentProblem(self, disp, param_fields, loadings)
    
    def get_linearized_problem(self, u: torch.Tensor, loadings: torch.Tensor):
        return LinearizedMechanicalProblem(self, u, loadings)
    
    def get_hom_response(
            self,
            u: torch.Tensor,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute homogenized response based on DOF vector u

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        field = self.compute_field(u, param_fields, loadings, state_fields)
        hom_response = field[(Ellipsis, slice(self.n_dim, None)) + self.dims].mean(self.dims_list)
        return hom_response

    def mandel(self, eps: torch.Tensor) -> torch.Tensor:
        """
        Mandel notation for symmetric second-order tensors

        :param eps: symmetric 2x2 tensor field on a 2d grid or
            symmetric 3x3 tensor field on a 3d grid
        :return: vector field with 3 (2d) or 6 (3d) components
        """
        if eps.ndim == self.n_dim + 2:
            eps = torch.unsqueeze(eps, dim=0)
        assert eps.ndim >= self.n_dim + 3
        assert eps.shape[-(self.n_dim + 2)] == self.n_dim and eps.shape[-(self.n_dim + 1)] == self.n_dim

        diag_idx = torch.stack([torch.arange(self.n_dim), torch.arange(self.n_dim)])
        tril_idx = torch.tril_indices(self.n_dim, self.n_dim, -1)
        idx = torch.cat([diag_idx, tril_idx], dim=-1)

        eps_hat = eps[(Ellipsis, idx[0], idx[1]) + self.dims]
        eps_hat[(Ellipsis, slice(self.n_dim, None)) + self.dims] *= math.sqrt(2)
        return eps_hat

    def reverse_mandel(self, eps_hat: torch.Tensor) -> torch.Tensor:
        """

        :param eps_hat:
        :return:
        """
        # TODO: implement
        raise NotImplementedError()

    def stiffness(self, params: torch.Tensor) -> torch.Tensor:
        """
        Assemble isotropic stiffness tensor in mandel notation

        :rtype: object
        :param params: lame's constants
        :return:
        """
        lame_lambda, lame_mu = params[..., 0], params[..., 1]
        # TODO: asserts
        Id = self.Id.to(dtype=lame_mu.dtype, device=lame_mu.device)
        P = self.P.to(dtype=lame_lambda.dtype, device=lame_lambda.device)
        stiff = torch.einsum("...,ij->...ij", 2 * lame_mu, Id) + torch.einsum("...,ij->...ij", lame_lambda, P)
        return stiff

    def stiffnesses(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param params:
        :return:
        """
        return self.stiffness(params[..., 0, :]), self.stiffness(params[..., 1, :])

    def stiffness_field(self, param_fields: torch.Tensor) -> torch.Tensor:
        """
        Assemble isotropic stiffness tensor in mandel notation

        :param param_fields:
        :return:
        """
        args = {"dtype": param_fields.dtype, "device": param_fields.device}

        lame_lambda = param_fields[(Ellipsis, slice(1)) + self.dims]
        lame_mu = param_fields[(Ellipsis, slice(1, 2)) + self.dims]
        if lame_lambda.ndim == 0:
            lame_lambda = lame_lambda * torch.ones([1, 1, 1], **args)
        if lame_mu.ndim == 0:
            lame_mu = lame_mu * torch.ones([1, 1, 1], **args)
        if lame_lambda.ndim == 2:
            lame_lambda = torch.unsqueeze(lame_lambda, dim=0)
        if lame_mu.ndim == 2:
            lame_mu = torch.unsqueeze(lame_mu, dim=0)
        assert lame_lambda.shape == lame_mu.shape
        assert lame_lambda.ndim >= 3

        Id = self.Id.to(**args)
        P = self.P.to(**args)
        stiff = torch.einsum("...i" + self.einsum_dims + ",ij->...ij" + self.einsum_dims, 2 * lame_mu, Id) + \
                torch.einsum("...i" + self.einsum_dims + ",ij->...ij" + self.einsum_dims, lame_lambda, P)
        return stiff

    @property
    def P(self):
        if self.n_dim == 2:
            return torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        elif self.n_dim == 3:
            return torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    @property
    def I(self):
        if self.n_dim == 2:
            return torch.tensor([1, 1, 0])
        elif self.n_dim == 3:
            return torch.tensor([1, 1, 1, 0, 0, 0])
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    @property
    def Id(self):
        return torch.eye(self.strain_dims, self.strain_dims)

    @property
    def strain_dims(self):
        if self.n_dim == 2:
            return 3
        elif self.n_dim == 3:
            return 6
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def construct_sparsity(self):
        n = self.n_grid[-1]
        if self.n_dim == 2:
            # Generate node indices
            ny, nx = self.n_grid
            full_dof = 2 * ny * nx
            x = torch.arange(nx)
            y = torch.arange(ny)
            X, Y = torch.meshgrid(x, y, indexing="xy")
            idx = torch.arange(n * n).reshape(ny, nx)
            idx2 = torch.arange(full_dof).reshape(2, ny, nx)

            # Generate element connectivity
            Xe = torch.stack([X, X + 1, X, X + 1], dim=-1)  # x-coordinates of all nodes per element
            Ye = torch.stack([Y, Y, Y + 1, Y + 1], dim=-1)  # y-coordinates of all nodes per element
            idx_pad = F.pad(idx.unsqueeze(0), [0, 1, 0, 1], mode="circular").squeeze()
            conn = idx_pad[Ye, Xe]
            conn2 = torch.stack([conn, conn + ny * nx], dim=-1).flatten(-2)

            # Generate indices for global stiffness matrix
            idx_x = conn2.repeat(1, 1, 2 * 4)
            idx_y = conn2.repeat_interleave(2 * 4, dim=-1)
            self.Idx = torch.stack([idx_x.ravel(), idx_y.ravel()])

            # Determine indices belonging to free DOFs
            if self.bc == BC.PERIODIC:
                self.idx_free = idx2.ravel()
            elif self.bc == BC.DIRICHLET:
                self.idx_free = idx2[..., 1:, 1:].ravel()
            elif self.bc == BC.DIRICHLET_TB:
                self.idx_free = idx2[..., 1:, :].ravel()
            elif self.bc == BC.DIRICHLET_LR:
                self.idx_free = idx2[..., :, 1:].ravel()
            else:
                raise NotImplementedError()
            
            # Assemble scatter matrix for free DOFs
            self.Lf = self.get_scatter_matrix(self.idx_free, shape=(self.idx_free.shape[0], full_dof))
        
        elif self.n_dim == 3:
            # Generate node indices
            nz, ny, nx = self.n_grid
            full_dof = 3 * nz * ny * nx
            x = torch.arange(nx)
            y = torch.arange(ny)
            z = torch.arange(nz)
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")  # TODO: check indexing
            idx = torch.arange(nz * ny * nx).reshape(nz, ny, nx)
            idx3 = torch.arange(full_dof).reshape(3, nz, ny, nx)

            # Generate element connectivity
            Xe = torch.stack([X, X + 1, X, X + 1, X, X + 1, X, X + 1], dim=-1)  # x-coordinates of all nodes per element, TODO: change
            Ye = torch.stack([Y, Y, Y + 1, Y + 1, Y, Y, Y + 1, Y + 1], dim=-1)  # y-coordinates of all nodes per element, TODO: change
            Ze = torch.stack([Z, Z, Z, Z, Z + 1, Z + 1, Z + 1, Z + 1], dim=-1)  # z-coordinates of all nodes per element, TODO: change
            idx_pad = F.pad(idx.unsqueeze(0), [0, 1, 0, 1, 0, 1], mode="circular").squeeze()
            conn = idx_pad[Xe, Ye, Ze]
            conn3 = torch.stack([conn, conn + nz * ny * nx, conn + 2 * nz * ny * nx], dim=-1).flatten(-2)

            # Generate indices for global stiffness matrix
            idx_x = conn3.repeat(1, 1, 1, 3 * 8)
            idx_y = conn3.repeat_interleave(3 * 8, dim=-1)
            self.Idx = torch.stack([idx_x.ravel(), idx_y.ravel()])

            # Determine indices belonging to free DOFs
            if self.bc == BC.PERIODIC:
                self.idx_free = idx3.ravel()
            elif self.bc == BC.DIRICHLET:
                self.idx_free = idx3[..., 1:, 1:, 1:].ravel()
            elif self.bc == BC.DIRICHLET_LR:
                self.idx_free = idx3[..., 1:, :, :].ravel()
            elif self.bc == BC.DIRICHLET_TB:
                self.idx_free = idx3[..., :, 1:, :].ravel()
            else:
                raise NotImplementedError()
            
            # Assemble scatter matrix for free DOFs
            self.Lf = self.get_scatter_matrix(self.idx_free, shape=(self.idx_free.shape[0], full_dof))
            
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")



    def assemble_matrix(self, param_fields: torch.Tensor, coalesce=True) -> torch.Tensor:
        """
        Assemble global stiffness matrix

        :param param_fields:
        :return:
        """
        # TODO: Construct conn, Idx in constructor and cache
        if (self.Idx is None) or (self.Lf is None):
            self.construct_sparsity()
        self.Idx = self.Idx.to(dtype=param_fields.dtype, device=param_fields.device)
        self.Lf = self.Lf.to(dtype=param_fields.dtype, device=param_fields.device)
        
        if self.n_dim == 2:
            # Generate node indices
            ny, nx = self.n_grid
            n_dof = 2 * ny * nx

            # Assemble element stiffness matrices
            stiff_field = self.stiffness_field(param_fields)
            ke = torch.flatten(self.get_ke(stiff_field), start_dim=-2)

            # Assemble global stiffness matrix
            stiffness = torch.sparse_coo_tensor(self.Idx, ke.ravel(), (n_dof, n_dof))
        
        elif self.n_dim == 3:
            # Generate node indices
            nz, ny, nx = self.n_grid
            n_dof = 3 * nz * ny * nx

            # Assemble element stiffness matrices
            stiff_field = self.stiffness_field(param_fields)
            ke = torch.flatten(self.get_ke(stiff_field), start_dim=-2)

            # Assemble global stiffness matrix
            stiffness = torch.sparse_coo_tensor(self.Idx, ke.ravel(), (n_dof, n_dof))
            
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        
        if coalesce:
            stiffness = stiffness.coalesce()

        # Extract stiffness matrix acting on free DOFs
        if self.bc == BC.PERIODIC:
            stiffness_free = stiffness
        else:
            with warnings.catch_warnings():
                # PyTorch UserWarning for sparse matrices can be ignored
                warnings.filterwarnings("ignore")
                stiffness_free = self.Lf @ stiffness @ self.Lf.T

        return stiffness_free
    
    def assemble_matrix_diag(self, image, params):
        stiff0, stiff1 = self.stiffness(params[...,0,:]), self.stiffness(params[...,1,:])
        for _ in range(self.n_dim):
            stiff0, stiff1 = stiff0.unsqueeze(-1), stiff1.unsqueeze(-1)
        ke0, ke1 = self.get_ke(stiff0), self.get_ke(stiff1)
        A_diag0, A_diag1 = torch.diag(ke0.squeeze()).sum(), torch.diag(ke1.squeeze()).sum()
        A_diag_approx = A_diag0 * (1. - image.ravel()) + A_diag1 * image.ravel()
        return A_diag_approx
    
    def assemble_rhs(
            self,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Assemble right hand side vector of linear system for given loading

        Equivalent to residual computation with zero guess for temperature field

        :param param_fields:
        :param loadings:
        :return:
        """
        if loadings is None:
            loadings = self.loadings

        # Move kernels to correct device
        args = {"device": param_fields.device, "dtype": param_fields.dtype}
        kernels = self.div_kernels.to(**args)

        strain_shape = (*loadings.shape[:-1], 1, loadings.shape[-1], *self.n_grid)
        
        # strain and stress are the same for all gauss points
        strain = loadings[(slice(None), None, slice(None)) + self.expand_dims].expand(strain_shape).to(**args)
        stress, _ = self.material_law(strain, param_fields)

        rhs = torch.zeros(self.get_vec_shape(param_fields, loadings), **args)
        for gp_i in range(self.n_gauss):
            rhs = rhs - self.reshape_vec(self.div(stress, gp_i=gp_i))
        return rhs
    
    def get_rhs_module(self, dtype, device) -> torch.nn.Module:
        div_module = self.get_div_module(dtype=dtype, device=device)

        module = RhsModule(
            n_dim = self.n_dim,
            shape=(self.n_gauss, self.strain_dims, *self.n_grid),
            model=self.material,
            div_module=div_module,
        )
        return module

    @property
    def n_gauss(self) -> int:
        return self.div_kernels.shape[-(3 + self.n_dim)]

    def compute_strain(
            self,
            disp: torch.Tensor,
            loading: Optional[torch.Tensor] = None,
            reduce: bool = False,
    ) -> torch.Tensor:
        """
        Compute strain field based on the displacement field for a given

        :param disp:
        :param loading:
        :param reduce:
        :return:
        """
        if disp.ndim == self.n_dim + 1:
            disp = torch.unsqueeze(disp, dim=0)
        assert disp.ndim >= self.n_dim + 2
        assert disp.shape[self.ch_dim] == self.n_channels
        n_loadings = disp.shape[-(2 + self.n_dim)]
        if loading is None:
            loading = self.loadings
        else:
            n_loadings = disp.shape[-(2 + self.n_dim)] if disp.ndim >= 2 + self.n_dim else 1
        assert loading.shape[0] == n_loadings
        assert loading.shape[1] == self.strain_dims

        strain_fluctuations = self.grad(disp)
        strain = strain_fluctuations + loading[(slice(None), None, slice(None)) + self.expand_dims].to(dtype=disp.dtype, device=disp.device)

        if reduce:
            strain = strain.mean(dim=-(2 + self.n_dim))
        return strain

    def compute_stress(
            self,
            strain: torch.Tensor,
            param_fields: torch.Tensor,
            state_fields: Optional[torch.Tensor] = None,
            reduce: bool = False,
    ) -> torch.Tensor:
        """
        Compute stress field based on the strain field using the stiffness tensors stiff0 and stiff1 of the two phases

        :param strain:
        :param param_fields:
        :param state_fields:
        :param reduce: reduce stress by averaging over all gauss points of each element
        :return:
        """
        if strain.ndim == self.n_dim + 2:
            strain = torch.unsqueeze(strain, dim=0)
        assert strain.ndim >= self.n_dim + 3
        assert strain.shape[self.ch_dim] == self.strain_dims

        stress, _ = self.material_law(strain, param_fields, state_fields=state_fields)

        if reduce:
            stress = stress.mean(-(2 + self.n_dim))
        return stress

    def grad(self, disp: torch.Tensor, gp_i=None, padding=True) -> torch.Tensor:
        """
        Compute the symmetric gradient of a periodic 2d vector field disp

        :param disp:
        :return:
        """
        if disp.ndim == self.n_dim + 1:
            disp = torch.unsqueeze(disp, dim=0)
        assert disp.ndim >= self.n_dim + 2
        assert disp.shape[self.ch_dim] == self.n_channels

        if self.n_dim == 2:
            return self.grad_2d(disp, gp_i=gp_i, padding=padding)
        elif self.n_dim == 3:
            return self.grad_3d(disp, gp_i=gp_i, padding=padding)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def grad_2d(self, disp: torch.Tensor, gp_i=None, padding=True) -> torch.Tensor:
        """
        2D symmetric gradient

        :param disp:
        :return:
        """
        # Construct kernels if not given
        kernels = self.grad_kernels.to(dtype=disp.dtype, device=disp.device)
        if gp_i is not None:
            kernels = kernels[gp_i:gp_i+1]
        n_gauss = kernels.shape[0]

        # Determine appropriate padding
        if padding:
            if self.bc == BC.PERIODIC:
                periodic_padding = [0, 1, 0, 1]
                constant_padding = [0, 0, 0, 0]
            elif self.bc == BC.DIRICHLET:
                periodic_padding = [0, 0, 0, 0]
                constant_padding = [1, 1, 1, 1]
            elif self.bc == BC.DIRICHLET_TB:
                periodic_padding = [0, 1, 0, 0]
                constant_padding = [0, 0, 1, 1]
            elif self.bc == BC.DIRICHLET_LR:
                periodic_padding = [0, 0, 0, 1]
                constant_padding = [1, 1, 0, 0]
            elif self.bc == BC.NEUMANN:
                periodic_padding = [0, 0, 0, 0]
                constant_padding = [0, 0, 0, 0]
            else:
                raise NotImplementedError()

            # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
            original_shape = torch.tensor(disp.shape)
            output_shape = torch.cat(
                [original_shape[:self.ch_dim], torch.tensor([n_gauss]), original_shape[self.ch_dim:]])
            output_shape[self.ch_dim] = self.strain_dims

            # Periodic padding of the displacement field to treat boundaries correctly
            disp_pad = disp
            if sum(periodic_padding) > 0:
                disp_pad = F.pad(disp_pad.flatten(end_dim=-3), periodic_padding, mode="circular").reshape(
                    -1,
                    2,  # original_shape[-3],
                    disp_pad.shape[-2] + periodic_padding[2] + periodic_padding[3],
                    disp_pad.shape[-1] + periodic_padding[0] + periodic_padding[1],
                )
            # Constant padding of the displacement field to treat boundaries correctly
            if sum(constant_padding) > 0:
                disp_pad = F.pad(disp_pad.flatten(end_dim=-3), constant_padding, mode="constant", value=0.0).reshape(
                    -1,
                    2,  # original_shape[-3],
                    disp_pad.shape[-2] + constant_padding[2] + constant_padding[3],
                    disp_pad.shape[-1] + constant_padding[0] + constant_padding[1],
                )
        else:
            disp_pad = disp

        # Compute strain tensor in mandel notation using a 2d convolution
        eps = F.conv2d(disp_pad.flatten(end_dim=-4), kernels.flatten(end_dim=-4))
        return eps.reshape(*disp.shape[:-3], kernels.shape[0], 3, *eps.shape[-2:])

    def grad_3d(self, disp: torch.Tensor, gp_i=None, padding=True) -> torch.Tensor:
        """
        3D symmetric gradient

        :param disp:
        :return:
        """
        # Construct kernels if not given
        kernels = self.grad_kernels.to(dtype=disp.dtype, device=disp.device)
        if gp_i is not None:
            kernels = kernels[gp_i:gp_i+1]
        n_gauss = kernels.shape[0]

        # Determine appropriate padding
        if padding:
            if self.bc == BC.PERIODIC:
                periodic_padding = [0, 1, 0, 1, 0, 1]
                constant_padding = [0, 0, 0, 0, 0, 0]
            elif self.bc == BC.DIRICHLET:
                periodic_padding = [0, 0, 0, 0, 0, 0]
                constant_padding = [1, 1, 1, 1, 1, 1]
            elif self.bc == BC.DIRICHLET_LR:
                periodic_padding = [0, 0, 0, 1, 0, 1]
                constant_padding = [1, 1, 0, 0, 0, 0]
            elif self.bc == BC.DIRICHLET_TB:
                periodic_padding = [0, 1, 0, 0, 0, 1]
                constant_padding = [0, 0, 1, 1, 0, 0]
            elif self.bc == BC.DIRICHLET_FB:
                periodic_padding = [0, 1, 0, 1, 0, 0]
                constant_padding = [0, 0, 0, 0, 1, 1]
            elif self.bc == BC.NEUMANN:
                periodic_padding = [0, 0, 0, 0, 0, 0]
                constant_padding = [0, 0, 0, 0, 0, 0]
            else:
                raise NotImplementedError(f"BC not supported")

            # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
            original_shape = torch.tensor(disp.shape)
            output_shape = torch.cat(
                [original_shape[:self.ch_dim], torch.tensor([n_gauss]), original_shape[self.ch_dim:]])
            output_shape[self.ch_dim] = self.strain_dims

            # Periodic padding of the displacement field to treat boundaries correctly
            disp_pad = disp
            if sum(periodic_padding) > 0:
                disp_pad = F.pad(disp_pad.flatten(end_dim=-4), periodic_padding, mode="circular").reshape(
                    -1,
                    3,
                    disp_pad.shape[-3] + periodic_padding[4] + periodic_padding[5],
                    disp_pad.shape[-2] + periodic_padding[2] + periodic_padding[3],
                    disp_pad.shape[-1] + periodic_padding[0] + periodic_padding[1],
                )
            # Constant padding of the displacement field to treat boundaries correctly
            if sum(constant_padding) > 0:
                disp_pad = F.pad(disp_pad.flatten(end_dim=-4), constant_padding, mode="constant", value=0.0).reshape(
                    -1,
                    3,
                    disp_pad.shape[-3] + constant_padding[4] + constant_padding[5],
                    disp_pad.shape[-2] + constant_padding[2] + constant_padding[3],
                    disp_pad.shape[-1] + constant_padding[0] + constant_padding[1],
                )
        else:
            disp_pad = disp

        # Compute strain tensor in mandel notation using a 3d convolution
        eps = F.conv3d(disp_pad, kernels.flatten(end_dim=-5))
        return eps.reshape(*disp.shape[:-4], kernels.shape[0], self.strain_dims, *eps.shape[-3:])

    def div(self, sig: torch.Tensor, gp_i=None) -> torch.Tensor:
        """
        Compute the divergence of a periodic 2d vector field sig

        :param sig:
        :return:
        """
        if sig.ndim == self.n_dim + 1:
            sig = torch.unsqueeze(sig, dim=0)
        assert sig.ndim >= self.n_dim + 2
        assert sig.shape[self.ch_dim] == self.strain_dims

        if self.n_dim == 2:
            return self.div_2d(sig, gp_i=gp_i)
        elif self.n_dim == 3:
            return self.div_3d(sig, gp_i=gp_i)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def div_2d(self, sig: torch.Tensor, gp_i=None) -> torch.Tensor:
        """
        Compute the divergence of a 2d vector field sig

        :param sig:
        """
        # Construct kernels if not given
        kernels = self.div_kernels.to(dtype=sig.dtype, device=sig.device)
        if gp_i is not None:
            kernels = kernels[gp_i:gp_i+1]

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [1, 0, 0, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [0, 0, 1, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.NEUMANN:
            periodic_padding = [0, 0, 0, 0]
            constant_padding = [0, 0, 0, 0]
        else:
            raise NotImplementedError()

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        # Original shape of sig is (..., L, G, 3, X, Y)
        # Output shape of res will be (..., L, 2, X, Y)
        original_shape = torch.tensor(sig.shape)
        output_shape = torch.cat([original_shape[:(self.ch_dim - 1)], original_shape[self.ch_dim:]])
        output_shape[self.ch_dim] = 2

        # Padding of the stress field to treat boundaries correctly
        # Reshape sig before padding to (-1, (G * 3), X, Y)
        sig_reshaped = sig.flatten(end_dim=-5).flatten(start_dim=self.ch_dim - 1, end_dim=self.ch_dim)
        # Shape of sig_pad after padding is (-1, (G * 3), X + 1, Y + 1)
        # Periodic padding
        sig_pad = sig_reshaped
        if sum(periodic_padding) > 0:
            sig_pad = F.pad(sig_pad, periodic_padding, mode="circular")
        # Constant padding
        if sum(constant_padding) > 0:
            sig_pad = F.pad(sig_pad, constant_padding, mode="constant", value=0.0)
            #sig_pad = F.pad(sig_pad, constant_padding, mode="replicate")

        # Compute strain tensor in mandel notation using a 2d convolution
        vol_e = 1.0 / torch.tensor(sig.shape[-2:]).prod()
        res = F.conv2d(sig_pad, kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2)) * vol_e / kernels.shape[0]
        output_shape[-1] = sig_pad.shape[-1] - 1
        output_shape[-2] = sig_pad.shape[-2] - 1
        return res.reshape(*sig.shape[:-4], 2, *res.shape[-2:])

    def div_3d(self, sig: torch.Tensor, gp_i=None) -> torch.Tensor:
        """
        Compute the divergence of a 3d vector field sig

        :param sig:
        """
        # Construct kernels if not given
        kernels = self.div_kernels.to(dtype=sig.dtype, device=sig.device)
        if gp_i is not None:
            kernels = kernels[gp_i:gp_i+1]

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [1, 0, 1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0, 0, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [0, 0, 1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [1, 0, 0, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_FB:
            periodic_padding = [1, 0, 1, 0, 0, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.NEUMANN:
            periodic_padding = [0, 0, 0, 0, 0, 0]
            constant_padding = [1, 1, 1, 1, 1, 1]
        else:
            raise NotImplementedError(f"BC not supported")

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        # Original shape of sig is (..., L, G, 3, X, Y, Z)
        # Output shape of res will be (..., L, 2, X, Y, Z)
        original_shape = torch.tensor(sig.shape)
        output_shape = torch.cat([original_shape[:(self.ch_dim - 1)], original_shape[self.ch_dim:]])
        output_shape[self.ch_dim] = 3

        # Padding of the stress field to treat boundaries correctly
        # Reshape sig before padding to (-1, (G * 6), X, Y, Z)
        sig_reshaped = sig.flatten(end_dim=-6).flatten(start_dim=self.ch_dim - 1, end_dim=self.ch_dim)
        # Shape of sig_pad after padding is (-1, (G * 6), X + 1, Y + 1, Z + 1)
        # Periodic padding
        sig_pad = sig_reshaped
        if sum(periodic_padding) > 0:
            sig_pad = F.pad(sig_pad, periodic_padding, mode="circular")
        # Constant padding
        if sum(constant_padding) > 0:
            sig_pad = F.pad(sig_pad, constant_padding, mode="constant", value=0.0)

        # Compute strain tensor in mandel notation using a 2d convolution
        vol_e = 1.0 / torch.tensor(sig.shape[-3:]).prod()
        res = F.conv3d(sig_pad, kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2)) * vol_e / kernels.shape[0]
        output_shape[-1] = sig_pad.shape[-1] - 1
        output_shape[-2] = sig_pad.shape[-2] - 1
        output_shape[-3] = sig_pad.shape[-3] - 1
        return res.reshape(*sig.shape[:-5], 3, *res.shape[-3:])

    def get_ke(self, stiff_field: torch.Tensor) -> torch.Tensor:
        """
        Get element stiffness

        :param stiff_field:
        :return: element stiffness tensor with shape batch_size x *n_grid x 8 x 8
        """
        while stiff_field.ndim < 2 + self.n_dim:
            stiff_field = stiff_field.unsqueeze(-1)

        assert stiff_field.ndim >= 2 + self.n_dim

        args = {"dtype": stiff_field.dtype, "device": stiff_field.device}

        vol_e = 1. / torch.tensor(self.n_grid).prod()
        b = self.get_grad_operator().to(**args)
        bt = b.transpose(-1, -2)

        # Loop over Gauss points
        ke = torch.einsum("ij,...jk" + self.einsum_dims + ",kl->..." + self.einsum_dims + "il",
                               bt[0], stiff_field, b[0]) * vol_e / self.n_gauss
        for g in range(1, self.n_gauss):
            ke += torch.einsum("ij,...jk" + self.einsum_dims + ",kl->..." + self.einsum_dims + "il",
                               bt[g], stiff_field, b[g]) * vol_e / self.n_gauss
        return ke

    def get_grad_operator(self) -> torch.Tensor:
        """
        Get FEM gradient operator

        :return:
        """
        # TODO: more general
        s2 = math.sqrt(0.5)
        rule1d, rule1dw = self.get_quad_rule()

        n_gauss_dir = len(rule1d)
        n_gauss = n_gauss_dir ** self.n_dim
        n_nodes = 2 ** self.n_dim

        grad_operators = torch.zeros((n_gauss, self.strain_dims, self.n_channels * n_nodes), dtype=self.dtype)

        grads_bar = torch.zeros((self.n_channels, 2 ** self.n_dim), dtype=self.dtype)

        if self.n_dim == 2:
            qw = torch.outer(rule1dw, rule1dw).ravel()  # weights in 2D
            qx = rule1d.expand((n_gauss_dir, n_gauss_dir)).T.ravel()
            qy = rule1d.expand((n_gauss_dir, n_gauss_dir)).ravel()

            B = lambda xi, eta: torch.tensor([
                [eta - 1.0, -(eta - 1.0), -eta, eta],
                [xi - 1.0, -xi, 1.0 - xi, xi]
            ], dtype=self.dtype)

            for i in range(n_gauss):
                grads = B(qx[i], qy[i]) / self.h.flip([-1])[:, None]
                grads_bar = grads_bar + grads * qw[i]

                for j in range(n_nodes):
                    B_node = torch.tensor([
                        [grads[0, j], 0.0],
                        [0.0, grads[1, j]],
                        [s2 * grads[1, j], s2 * grads[0, j]]
                    ], dtype=self.dtype)

                    grad_operators[i, :, self.n_dim * j: self.n_dim * (j + 1)] = B_node

        elif self.n_dim == 3:
            qw = (rule1dw[:, None, None] * rule1dw[None, :, None] * rule1dw[None, None, :]).ravel()
            qx = rule1d.expand(len(rule1d), len(rule1d), len(rule1d)).permute(0, 1, 2).ravel()
            qy = rule1d.expand(len(rule1d), len(rule1d), len(rule1d)).permute(0, 2, 1).ravel()
            qz = rule1d.expand(len(rule1d), len(rule1d), len(rule1d)).permute(2, 1, 0).ravel()

            B = lambda xi, eta, zeta: torch.tensor([
                [-(eta - 1) * (zeta - 1), (eta - 1) * (zeta - 1), eta * (zeta - 1), -eta * (zeta - 1),
                    zeta * (eta - 1), -zeta * (eta - 1), -eta * zeta, eta * zeta],
                [-(xi - 1) * (zeta - 1), xi * (zeta - 1), (xi - 1) * (zeta - 1), -xi * (zeta - 1), zeta * (xi - 1),
                    -xi * zeta, -zeta * (xi - 1), xi * zeta],
                [-(eta - 1) * (xi - 1), xi * (eta - 1), eta * (xi - 1), -eta * xi, (eta - 1) * (xi - 1),
                    -xi * (eta - 1), -eta * (xi - 1), eta * xi]
            ], dtype=self.dtype)

            for i in range(n_gauss):
                grads = B(qx[i], qy[i], qz[i]) / self.h.flip([-1])[:, None]
                grads_bar = grads_bar + grads * qw[i]

                for j in range(n_nodes):
                    B_node = torch.tensor([
                        [grads[0, j], 0.0, 0.0],
                        [0.0, grads[1, j], 0.0],
                        [0.0, 0.0, grads[2, j]],
                        [s2 * grads[1, j], s2 * grads[0, j], 0.0],
                        [s2 * grads[2, j], 0.0, s2 * grads[0, j]],
                        [0.0, s2 * grads[2, j], s2 * grads[1, j]]
                    ], dtype=self.dtype)

                    grad_operators[i, :, self.n_dim * j: self.n_dim * (j + 1)] = B_node

        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

        return grad_operators

    def get_grad_kernels(self) -> torch.Tensor:
        """
        Get kernels for symmetric gradient computation

        :return:
        """
        if self.grad_operator is None:
            self.grad_operator = self.get_grad_operator()

        if self.n_dim == 2:
            sgo = torch.stack([
                self.grad_operator[..., 0::2],
                self.grad_operator[..., 1::2]
            ], dim=-2)
            grad_kernels = sgo.reshape((*sgo.shape[:-1], 2, 2))
        elif self.n_dim == 3:
            sgo = torch.stack([
                self.grad_operator[..., 0::3],
                self.grad_operator[..., 1::3],
                self.grad_operator[..., 2::3]
            ], dim=-2)
            grad_kernels = sgo.reshape((*sgo.shape[:-1], 2, 2, 2))
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

        return grad_kernels

    def get_div_kernels(self) -> torch.Tensor:
        """
        Get kernels for divergence computation

        :return:
        """
        if self.grad_kernels is None:
            self.grad_kernels = self.get_grad_kernels()

        if self.n_dim == 2:
            div_kernels = torch.roll(self.grad_kernels.transpose(-3, -4), shifts=(1, 1), dims=(-2, -1))
        elif self.n_dim == 3:
            div_kernels = torch.roll(self.grad_kernels.transpose(-4, -5), shifts=(1, 1, 1), dims=(-3, -2, -1))
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

        return div_kernels

    def get_grad_kernels_fourier(self):
        grad_kernels = self.get_grad_kernels()
        return self.kernels_to_fourier(grad_kernels)

    def get_div_kernels_fourier(self):
        # TODO: check
        div_kernels = self.get_div_kernels()
        return self.kernels_to_fourier(div_kernels)

    def kernels_to_fourier(self, kernels):
        if self.n_dim == 2:
            kernels_padded = torch.nn.functional.pad(torch.flip(kernels, dims=(-2, -1)),
                (0, self.n_grid[-2] - kernels.shape[-2],
                 0, self.n_grid[-1] - kernels.shape[-1]))
            kernels_padded = torch.roll(kernels_padded, shifts=(-1, -1), dims=(-2, -1))
            return torch.fft.fft2(kernels_padded)
        elif self.n_dim == 3:
            kernels_padded = torch.nn.functional.pad(torch.flip(kernels, dims=(-3, -2, -1)),
                (0, self.n_grid[-3] - kernels.shape[-3],
                 0, self.n_grid[-2] - kernels.shape[-2],
                 0, self.n_grid[-1] - kernels.shape[-1]))
            kernels_padded = torch.roll(kernels_padded, shifts=(-1, -1, -1), dims=(-3, -2, -1))
            return torch.fft.fft2(kernels_padded)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def compute_fundamental_solution(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """

        if self.n_dim == 2:
            return self.compute_fundamental_solution_2d(stiff, full=full)
        elif self.n_dim == 3:
            return self.compute_fundamental_solution_3d(stiff, full=full)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def compute_fundamental_solution_2d(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """
        fk = self.compute_fourier_kernel(stiff, full=full)

        if self.quad_degree == 1:
            # Compute 2x2 inverse using pinv for better stability
            epsilon = 100 * torch.finfo(fk.dtype).eps * fk.max().detach()
            phi_hat = torch.linalg.pinv(fk.permute([2, 3, 0, 1]), atol=epsilon).permute([2, 3, 0, 1])
        else:
            # Compute 2x2 inverse using hardcoded formula as this is more efficient than torch.linalg.pinv
            # Using torch.linalg.inv or torch.linalg.solve is numerically unstable in this case
            a, b = fk[0, 0], fk[0, 1]
            c, d = fk[1, 0], fk[1, 1]
            den = a * d - b * c
            phi_hat = torch.stack([
                torch.stack([d / den, -b / den], dim=0),
                torch.stack([-c / den, a / den], dim=0),
            ], dim=0)
        phi_hat_zeroed = torch.zeros_like(phi_hat)
        phi_hat[..., 0, 0] = phi_hat_zeroed[..., 0, 0]
        return phi_hat * torch.tensor(self.n_grid).prod()

    def compute_fundamental_solution_3d(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """
        fk = self.compute_fourier_kernel(stiff, full=full)

        if self.quad_degree == 1:
            # Compute 3x3 inverse using pinv for better stability
            epsilon = 100 * torch.finfo(fk.dtype).eps * fk.max().detach()
            phi_hat = torch.linalg.pinv(fk.permute([2, 3, 4, 0, 1]), atol=epsilon).permute([3, 4, 0, 1, 2])
        else:
            # Compute 3x3 inverse using hardcoded formula as this is more efficient than torch.linalg.pinv
            # Using torch.linalg.inv or torch.linalg.solve is numerically unstable in this case
            a, b, c = fk[0, 0], fk[0, 1], fk[0, 2]
            d, e, f = fk[1, 0], fk[1, 1], fk[1, 2]
            g, h, i = fk[2, 0], fk[2, 1], fk[2, 2]
            den = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g
            phi_hat = torch.stack([
                torch.stack([(e * i - f * h) / den, (c * h - b * i) / den, (b * f - c * e) / den], dim=0),
                torch.stack([(f * g - d * i) / den, (a * i - c * g) / den, (c * d - a * f) / den], dim=0),
                torch.stack([(d * h - e * g) / den, (b * g - a * h) / den, (a * e - b * d) / den], dim=0)
            ], dim=0)
        phi_hat[..., 0, 0, 0] = 0.0
        return phi_hat * torch.tensor(self.n_grid).prod()

    def compute_fourier_kernel(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """
        if self.n_dim == 2:
            return self.compute_fourier_kernel_2d(stiff, full=full)
        elif self.n_dim == 3:
            return self.compute_fourier_kernel_3d(stiff, full=full)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def compute_fourier_kernel_2d(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """
        ny, nx = self.n_grid
        nh = nx if full else nx // 2 + 1

        complex_type = self.get_complex_dtype(stiff.dtype)
        range_nx = torch.arange(nx, device=stiff.device).type(complex_type)
        range_ny = torch.arange(ny, device=stiff.device).type(complex_type)
        xi1 = torch.exp(2.0 * math.pi * 1j * range_nx / nx)[:nh]
        xi2 = torch.exp(2.0 * math.pi * 1j * range_ny / ny)
        ones_x = torch.ones(nh, dtype=stiff.dtype, device=stiff.device)
        ones_y = torch.ones(ny, dtype=stiff.dtype, device=stiff.device)

        # Assemble auxiliary vector field with 4 components on the 2D domain for the Fourier kernel
        # This vector field can be interpreted as Fourier representation of the FEM grad/B operator
        A = torch.zeros((4, ny, nh), dtype=complex_type, device=stiff.device)
        A[0] = ones_x[None, :] * ones_y[:, None]
        A[1] = xi1[None, :] * ones_y[:, None]
        A[2] = ones_x[None, :] * xi2[:, None]
        A[3] = xi1[None, :] * xi2[:, None]
        AA = torch.einsum("i" + self.einsum_dims + ",j" + self.einsum_dims +"->ij" + self.einsum_dims,
                          A, A.conj())

        ke = self.get_ke(stiff).squeeze()
        ke_mod = torch.stack(
            [torch.stack([ke[0::2, 0::2], ke[0::2, 1::2]], dim=0),
             torch.stack([ke[1::2, 0::2], ke[1::2, 1::2]], dim=0)],
            dim=0,
        )
        fk = torch.einsum("pqij,ij...->pq...", ke_mod.type(AA.dtype), AA).real
        return fk

    def compute_fourier_kernel_3d(self, stiff: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param stiff:
        :return:
        """
        nz, ny, nx = self.n_grid
        nh = nx if full else nx // 2 + 1

        # Construct frequency variables in all 3 dimensions
        complex_type = self.get_complex_dtype(stiff.dtype)
        range_nx = torch.arange(nx, device=stiff.device).type(complex_type)
        range_ny = torch.arange(ny, device=stiff.device).type(complex_type)
        range_nz = torch.arange(nz, device=stiff.device).type(complex_type)
        xi1 = torch.exp(2.0 * math.pi * 1j * range_nx / nx)[:nh]
        xi2 = torch.exp(2.0 * math.pi * 1j * range_ny / ny)
        xi3 = torch.exp(2.0 * math.pi * 1j * range_nz / nz)
        ones_x = torch.ones(nh, dtype=stiff.dtype, device=stiff.device)
        ones_y = torch.ones(ny, dtype=stiff.dtype, device=stiff.device)
        ones_z = torch.ones(nz, dtype=stiff.dtype, device=stiff.device)

        # Assemble auxiliary vector field with 8 components on the 3D domain for the Fourier kernel
        # This vector field can be interpreted as Fourier representation of the FEM grad/B operator
        A = torch.zeros((8, nz, ny, nh), dtype=complex_type, device=stiff.device)
        A[0] = ones_x[None, None, :] * ones_y[None, :, None] * ones_z[:, None, None]
        A[1] = xi1[None, None, :] * ones_y[None, :, None] * ones_z[:, None, None]
        A[2] = ones_x[None, None, :] * xi2[None, :, None] * ones_z[:, None, None]
        A[3] = xi1[None, None, :] * xi2[None, :, None] * ones_z[:, None, None]
        A[4] = ones_x[None, None, :] * ones_y[None, :, None] * xi3[:, None, None]
        A[5] = xi1[None, None, :] * ones_y[None, :, None] * xi3[:, None, None]
        A[6] = ones_x[None, None, :] * xi2[None, :, None] * xi3[:, None, None]
        A[7] = xi1[None, None, :] * xi2[None, :, None] * xi3[:, None, None]

        # Compute Fourier kernel based on the element stiffness matrix ke and the auxiliary vector field A
        ke = self.get_ke(stiff).squeeze()
        ke_mod = torch.stack([
            torch.stack([ke[0::3, 0::3], ke[0::3, 1::3], ke[0::3, 2::3]], dim=0),
            torch.stack([ke[1::3, 0::3], ke[1::3, 1::3], ke[1::3, 2::3]], dim=0),
            torch.stack([ke[2::3, 0::3], ke[2::3, 1::3], ke[2::3, 2::3]], dim=0)
        ], dim=0)
        fk = torch.zeros((3,3,nz,ny,nh), dtype=stiff.dtype, device=stiff.device)
        for p in range(3):
            for q in range(3):
                fk[p,q] = torch.einsum("ij,ixyz,jxyz->xyz", ke_mod[p,q].type(A.dtype), A, A.conj()).real
        return fk
    
    def get_disp_loss(self, reduction: str="sum"):
        return DispLoss(n_dim=self.n_dim, reduction=reduction)
    
    def get_stress_loss(self, reduction:str = "sum"):
        return StressLoss(n_dim=self.n_dim, reduction=reduction)
    
    def compute_losses(self, field_pred: torch.Tensor, field_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all relevant losses
        :param field_pred:
        :param field_ref:
        :return:
        """
        losses = {
            "disp": self.get_disp_loss(reduction="none")(field_pred, field_ref),
            "stress": self.get_stress_loss(reduction="none")(field_pred, field_ref).squeeze(-1),
        }
        return losses
