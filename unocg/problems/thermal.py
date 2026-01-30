"""
Thermal problem formulations
"""
import math
import warnings
from typing import Optional, Union, Tuple, Collection, Dict

import torch
import torch.nn.functional as F
from unocg.problems import Problem, BC
from unocg.modules.operators import MatvecModule, RhsModule, FieldModule
from unocg.materials import Material
from unocg.materials.thermal import LinearHeatConduction
from unocg.training.losses.thermal import TempLoss, GradLoss, FluxLoss, ThermalEnergyLoss, HeatCondResidualLoss


class ThermalProblem(Problem):
    """
    Thermal problem
    """

    def __init__(
            self,
            n_grid: Union[torch.Size, Collection[int]],
            material: Material,
            loadings: Optional[torch.Tensor] = None,
            quad_degree: int = 2,
            bc: Optional[BC] = None,
            lengths = None,
    ):
        """
        Initialise the thermal problem.

        :param n_grid:
        :type n_grid: Union[torch.Size, Collection[int]]
        :param material: material model
        :type material: Material
        :param loadings:
        :type loadings: torch.Tensor
        :param quad_degree:
        :type quad_degree: int
        :param bc:
        :type bc: BC
        """
        super().__init__(n_grid=n_grid, bc=bc, quad_degree=quad_degree, lengths=lengths, material=material)
        self._n_channels = 1

        if loadings is None:
            n_loadings = self.n_dim
            self.loadings = torch.eye(n_loadings)
        else:
            assert loadings.ndim == 2
            assert loadings.shape[1] == self.n_dim
            self.loadings = loadings

        if self.bc == BC.PERIODIC:
            self.dof_shape = self.n_grid
        elif self.bc == BC.DIRICHLET:
            self.dof_shape = tuple([n_axis - 1 for n_axis in self.n_grid])
        elif self.bc == BC.DIRICHLET_TB:
            self._n_dof = self.n_channels * (self.n_grid[-1] - 1) * self.n_grid[-2]
            self.dof_shape = (self.n_grid[0] - 1, *self.n_grid[1:])
        elif self.bc == BC.DIRICHLET_LR:
            self._n_dof = self.n_channels * self.n_grid[0] * (self.n_grid[1] - 1)
            self.dof_shape = (*self.n_grid[:1], self.n_grid[1] - 1, *self.n_grid[2:])
        elif self.bc == BC.DIRICHLET_FB:
            self._n_dof = self.n_channels * self.n_grid[0] * (self.n_grid[1] - 1)
            self.dof_shape = (*self.n_grid[:2], self.n_grid[2] - 1, *self.n_grid[3:])
        else:
            raise NotImplementedError()
        self._n_dof = self.n_channels * torch.prod(torch.tensor(self.dof_shape))

        # Precompute kernels for residual computation
        self.grad_operator = self.get_grad_operator()
        self.grad_kernels = self.get_grad_kernels()
        self.div_kernels = self.get_div_kernels()

        self.Idx = None
        self.Lf = None

    def compute_residual(
            self, u: torch.Tensor,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute residual for the thermal problem.

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()
        if loadings is None:
            loadings = self.loadings

        # Move kernels to correct device
        args = {"device": u.device, "dtype": u.dtype}
        self.grad_kernels = self.grad_kernels.to(**args)
        self.div_kernels = self.div_kernels.to(**args)

        temp_grad = self.grad(self.reshape_field(u)) + loadings[(slice(None), None, slice(None)) + self.expand_dims].to(**args)
        flux, _ = self.material_law(temp_grad, param_fields.to(**args))
        r = self.reshape_vec(self.div(flux))

        return r, None

    def matvec(self,
               d: torch.Tensor,
               param_fields: Optional[torch.Tensor] = None,
               state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Matvec

        :param d:
        :param param_fields:
        :param state_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()

        # Move kernels to correct device
        args = {"device": d.device, "dtype": d.dtype}
        self.grad_kernels = self.grad_kernels.to(**args)
        self.div_kernels = self.div_kernels.to(**args)

        temp_grad = self.grad(self.reshape_field(d))
        flux, _ = self.material_law(temp_grad, param_fields.to(**args))
        p = self.reshape_vec(self.div(flux))

        return -p, None

    def material_law(self, temp_grad: torch.Tensor, param_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Material law

        :param temp_grad:
        :param param_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()

        return self.material(temp_grad, param_fields)

    def get_matvec_module(self, dtype, device) -> torch.nn.Module:
        grad_module = self.get_grad_module(dtype=dtype, device=device)
        div_module = self.get_div_module(dtype=dtype, device=device)

        module = MatvecModule(self.n_dim, grad_module, self.material, div_module)
        return module

    def conductivity(self, params: torch.Tensor) -> torch.Tensor:
        """
        Assemble conductivity tensor for one phase

        :param params:
        :return:
        :rtype: torch.Tensor
        """
        if params.ndim == 0:
            params = params.unsqueeze(dim=0)
        assert params.ndim >= 1
        Id = self.Id.to(dtype=params.dtype, device=params.device)
        cond = params[..., None] * Id
        return cond

    def conductivities(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble conductivity tensor for both phases

        :param params:
        :return:
        :rtype: torch.Tensor
        """
        if params.ndim == 1:
            params = params.unsqueeze(dim=-1)
        return self.conductivity(params[..., 0, :]), self.conductivity(params[..., 1, :])

    def conductivity_field(self, param_fields: torch.Tensor) -> torch.Tensor:
        """
        Assemble isotropic stiffness tensor in mandel notation

        :param param_fields:
        :return:
        :rtype: torch.Tensor
        """
        if param_fields.ndim == 0:
            param_fields = param_fields * torch.ones([1, 1, 1], dtype=param_fields.dtype, device=param_fields.device)
        if param_fields.ndim == 2:
            param_fields = torch.unsqueeze(param_fields, dim=0)
        assert param_fields.ndim >= 3

        Id = self.Id.to(dtype=param_fields.dtype, device=param_fields.device)
        cond = torch.einsum("...i" + self.einsum_dims + ",ij->...ij" + self.einsum_dims, param_fields, Id)
        return cond

    @property
    def Id(self):
        return torch.eye(self.n_dim, self.n_dim)
    
    def construct_sparsity(self):
        n = self.n_grid[-1]
        if self.n_dim == 2:
            # Generate node indices
            x = torch.arange(n)
            y = torch.arange(n)
            X, Y = torch.meshgrid(x, y, indexing="xy")
            idx = torch.arange(n * n).reshape(n, n)

            # Generate element connectivity
            Xe = torch.stack([X, X + 1, X, X + 1], dim=-1)  # x-coordinates of all nodes per element
            Ye = torch.stack([Y, Y, Y + 1, Y + 1], dim=-1)  # y-coordinates of all nodes per element
            idx_pad = F.pad(idx.unsqueeze(0), [0, 1, 0, 1], mode="circular").squeeze()
            conn = idx_pad[Ye, Xe]

            # Generate indices for global stiffness matrix
            idx_x = conn.repeat(1, 1, 4)
            idx_y = conn.repeat_interleave(4, dim=-1)
            self.Idx = torch.stack([idx_x.ravel(), idx_y.ravel()])

            # Determine indices belonging to free DOFs
            if self.bc == BC.PERIODIC:
                self.idx_free = idx.ravel()
            elif self.bc == BC.DIRICHLET:
                self.idx_free = idx[1:, 1:].ravel()
            elif self.bc == BC.DIRICHLET_LR:
                self.idx_free = idx[1:, :].ravel()
            elif self.bc == BC.DIRICHLET_TB:
                self.idx_free = idx[:, 1:].ravel()
            else:
                raise NotImplementedError()
            
            # Assemble scatter matrix for free DOFs
            self.Lf = self.get_scatter_matrix(self.idx_free, shape=(self.idx_free.shape[0], n * n))
        
        elif self.n_dim == 3:
            # Generate node indices
            x = torch.arange(n)
            y = torch.arange(n)
            z = torch.arange(n)
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")  # TODO: check indexing
            idx = torch.arange(n * n * n).reshape(n, n, n)

            # Generate element connectivity
            Xe = torch.stack([X, X + 1, X, X + 1, X, X + 1, X, X + 1], dim=-1)  # x-coordinates of all nodes per element, TODO: change
            Ye = torch.stack([Y, Y, Y + 1, Y + 1, Y, Y, Y + 1, Y + 1], dim=-1)  # y-coordinates of all nodes per element, TODO: change
            Ze = torch.stack([Z, Z, Z, Z, Z + 1, Z + 1, Z + 1, Z + 1], dim=-1)  # z-coordinates of all nodes per element, TODO: change
            idx_pad = F.pad(idx.unsqueeze(0), [0, 1, 0, 1, 0, 1], mode="circular").squeeze()
            conn = idx_pad[Xe, Ye, Ze]

            # Generate indices for global stiffness matrix
            idx_x = conn.repeat(1, 1, 1, 8)
            idx_y = conn.repeat_interleave(8, dim=-1)
            self.Idx = torch.stack([idx_x.ravel(), idx_y.ravel()])

            # Determine indices belonging to free DOFs
            if self.bc == BC.PERIODIC:
                self.idx_free = idx.ravel()
            elif self.bc == BC.DIRICHLET:
                self.idx_free = idx[1:, 1:, 1:].ravel()
            elif self.bc == BC.DIRICHLET_LR:
                self.idx_free = idx[1:, :, :].ravel()
            elif self.bc == BC.DIRICHLET_TB:
                self.idx_free = idx[:, 1:, :].ravel()
            else:
                raise NotImplementedError()
            
            # Assemble scatter matrix for free DOFs
            self.Lf = self.get_scatter_matrix(self.idx_free, shape=(self.idx_free.shape[0], n * n * n))
            
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")


    def assemble_matrix(self, param_fields: torch.Tensor, coalesce=True) -> torch.Tensor:
        """
        Assemble global stiffness matrix

        :param param_fields:
        :return:
        """
        if (self.Idx is None) or (self.Lf is None):
            self.construct_sparsity()
        self.Idx = self.Idx.to(dtype=param_fields.dtype, device=param_fields.device)
        self.Lf = self.Lf.to(dtype=param_fields.dtype, device=param_fields.device)
        
        if self.n_dim == 2:
            # Generate node indices
            #param_field = param_fields.transpose(-1, -2)
            n = param_fields.shape[-1]

            # Assemble element stiffness matrices
            cond_field = self.conductivity_field(param_fields)
            ke = torch.flatten(self.get_ke(cond_field), start_dim=-3)

            # Assemble global stiffness matrix
            stiffness = torch.sparse_coo_tensor(self.Idx, ke.ravel(), (n * n, n * n))
        
        elif self.n_dim == 3:
            # Generate node indices
            # param_field = param_fields.transpose(-1, -2)  # TODO: change
            n = param_fields.shape[-1]

            # Assemble element stiffness matrices
            cond_field = self.conductivity_field(param_fields)
            ke = torch.flatten(self.get_ke(cond_field), start_dim=-3)

            # Assemble global stiffness matrix
            stiffness = torch.sparse_coo_tensor(self.Idx, ke.ravel(), (n * n * n, n * n * n))
            
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
        # TODO: correct interface contributions
        cond0, cond1 = self.conductivity(params[...,0,:]), self.conductivity(params[...,1,:])
        for _ in range(self.n_dim):
            cond0, cond1 = cond0.unsqueeze(-1), cond1.unsqueeze(-1)
        ke0, ke1 = self.get_ke(cond0), self.get_ke(cond1)
        A_diag0, A_diag1 = torch.diag(ke0.squeeze()).sum(), torch.diag(ke1.squeeze()).sum()
        A_diag_approx = A_diag0 * (1. - image.ravel()) + A_diag1 * image.ravel()
        return A_diag_approx

    def assemble_rhs(self, param_fields: torch.Tensor, loadings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Assemble right hand side vector of linear system for given loading

        Equivalent to residual computation with zero guess for temperature field

        :param param_fields:
        :param loadings:
        :return:
        """
        if param_fields is None:
            raise ValueError()
        if loadings is None:
            loadings = self.loadings

        # Move kernels to correct device
        args = {"device": param_fields.device, "dtype": param_fields.dtype}
        self.grad_kernels = self.grad_kernels.to(**args)
        self.div_kernels = self.div_kernels.to(**args)

        batch_shape = param_fields.shape[:self.ch_dim]
        grad_shape = (*batch_shape, loadings.shape[0], self.n_gauss, loadings.shape[-1], *self.n_grid)
        grad = loadings[(slice(None), None, slice(None)) + self.expand_dims].expand(grad_shape).to(**args)
        flux, _ = self.material_law(grad, param_fields)
        rhs = self.reshape_vec(self.div(flux))
        return rhs
    
    def get_rhs_module(self, dtype, device) -> torch.nn.Module:
        div_module = self.get_div_module(dtype=dtype, device=device)

        module = RhsModule(
            n_dim=self.n_dim,
            shape=(self.n_gauss, self.n_dim, *self.n_grid),
            model=self.material,
            div_module=div_module,
        )
        return module

    @property
    def n_gauss(self) -> int:
        return self.div_kernels.shape[-(3 + self.n_dim)]

    def compute_field(
            self,
            u: torch.Tensor,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        if param_fields is None:
            raise ValueError()

        temp = self.reshape_field(u)
        flux = self.compute_flux(temp, param_fields, loadings=loadings, reduce=True)

        if self.n_dim == 2:
            if self.bc == BC.DIRICHLET:
                temp = F.pad(temp, [1, 0, 1, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_TB:
                temp = F.pad(temp, [1, 0, 0, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_LR:
                temp = F.pad(temp, [0, 0, 1, 0], mode="constant", value=0.0)
        elif self.n_dim == 3:
            if self.bc == BC.DIRICHLET:
                temp = F.pad(temp, [1, 0, 1, 0, 1, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_TB:
                temp = F.pad(temp, [1, 0, 0, 0, 0, 0], mode="constant", value=0.0)
            elif self.bc == BC.DIRICHLET_LR:
                temp = F.pad(temp, [0, 0, 1, 0, 0, 0], mode="constant", value=0.0)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} not supported for this operation")

        field = torch.cat([temp, flux], dim=self.ch_dim)
        return field
    
    def get_field_module(self, dtype, device):
        padding_layers = []
        if self.n_dim == 2:
            if self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad2d((1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.ZeroPad2d((1, 0, 0, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.ZeroPad2d((0, 0, 1, 0))]
        elif self.n_dim == 3:
            if self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad3d((1, 0, 1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.ZeroPad3d((1, 0, 0, 0, 0, 0))]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.ZeroPad3d((0, 0, 1, 0, 0, 0))]
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} not supported for this operation")

        return FieldModule(
            ch_dim=self.ch_dim,
            expand_dims=self.expand_dims,
            grad_module=self.get_grad_module(dtype=dtype, device=device),
            model=self.material,
            padding_layers=padding_layers,
        )
    
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
        hom_response = field[(Ellipsis, slice(1, None)) + self.dims].mean(self.dims_list)
        return hom_response

    def compute_flux(
            self,
            temp: torch.Tensor,
            param_fields: torch.Tensor,
            loadings: Optional[torch.Tensor] = None,
            reduce: bool = False,
    ) -> torch.Tensor:
        """
        Compute temperature gradient field based on the temperature field for a given

        :param temp:
        :param param_fields:
        :param loadings:
        :param reduce: reduce flux at all gauss points to mean value for each element
        :return: flux
        """
        if temp.ndim == 3:
            temp = torch.unsqueeze(temp, dim=0)
        assert temp.ndim >= 4
        assert temp.shape[self.ch_dim] == 1
        n_loadings = temp.shape[self.ch_dim - 1]
        if loadings is None:
            loadings = torch.eye(n_loadings, dtype=temp.dtype, device=temp.device)
        else:
            n_loadings = temp.shape[self.ch_dim - 1] if temp.ndim >= 4 else 1
        assert loadings.shape[0] == n_loadings
        assert loadings.shape[1] == self.n_dim

        temp_grad_fluctuations = self.grad(temp)
        temp_grad = temp_grad_fluctuations + loadings[(slice(None), None, slice(None)) + self.expand_dims]
        flux, _ = self.material_law(temp_grad, param_fields)

        if reduce:
            flux = flux.mean(-(2 + self.n_dim))  # mean value over all gauss points of each element
        return flux

    def grad(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Compute the symmetric gradient of a periodic 2d vector field disp

        :param temp:
        :return:
        """
        if temp.ndim == self.n_dim + 1:
            temp = torch.unsqueeze(temp, dim=0)
        assert temp.ndim >= self.n_dim + 2
        # assert temp.shape[-1] == temp.shape[-2]
        assert temp.shape[self.ch_dim] == self.n_channels

        if self.n_dim == 2:
            return self.grad_2d(temp)
        elif self.n_dim == 3:
            return self.grad_3d(temp)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def grad_2d(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D gradient of a periodic vector field temp

        :param temp:
        :return:
        """
        # Construct kernels if not given
        kernels = self.grad_kernels.to(device=temp.device, dtype=temp.dtype)

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [0, 1, 0, 1]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0]
            constant_padding = [1, 1, 1, 1]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [0, 1, 0, 0]
            constant_padding = [0, 0, 1, 1]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [0, 0, 0, 1]
            constant_padding = [1, 1, 0, 0]
        else:
            raise NotImplementedError()

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        original_shape = torch.tensor(temp.shape)
        output_shape = torch.cat([original_shape[:self.ch_dim], torch.tensor(kernels.shape[:1]), original_shape[self.ch_dim:]])
        output_shape[self.ch_dim] = self.n_dim

        # Periodic padding of the temperature field to treat boundaries correctly
        temp_pad = temp
        if sum(periodic_padding) > 0:
            temp_pad = F.pad(temp_pad.flatten(end_dim=-3), periodic_padding, mode="circular").reshape(
                -1,
                original_shape[-3],
                temp_pad.shape[-2] + periodic_padding[2] + periodic_padding[3],
                temp_pad.shape[-1] + periodic_padding[0] + periodic_padding[1],
            )
        # Constant padding of the displacement field to treat boundaries correctly
        if sum(constant_padding) > 0:
            temp_pad = F.pad(temp_pad.flatten(end_dim=-3), constant_padding, mode="constant", value=0.0).reshape(
                -1,
                original_shape[-3],
                temp_pad.shape[-2] + constant_padding[2] + constant_padding[3],
                temp_pad.shape[-1] + constant_padding[0] + constant_padding[1],
            )

        # Compute temperature gradient tensor using a 2d convolution
        g = F.conv2d(temp_pad, kernels.flatten(end_dim=-4))
        return g.reshape(*temp.shape[:-3], kernels.shape[0], 2, *g.shape[-2:])

    def grad_3d(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Compute the 3D gradient of a periodic vector field temp

        :param temp:
        :return:
        """
        # Construct kernels if not given
        kernels = self.grad_kernels.to(device=temp.device, dtype=temp.dtype)

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [0, 1, 0, 1, 0, 1]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0, 0, 0]
            constant_padding = [1, 1, 1, 1, 1, 1]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [0, 1, 0, 0, 0, 1]
            constant_padding = [0, 0, 1, 1, 0, 0]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [0, 0, 0, 1, 0, 1]
            constant_padding = [1, 1, 0, 0, 0, 0]
        else:
            raise NotImplementedError()

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        original_shape = torch.tensor(temp.shape)
        output_shape = torch.cat([original_shape[:self.ch_dim], torch.tensor(kernels.shape[:1]), original_shape[self.ch_dim:]])
        output_shape[self.ch_dim] = self.n_dim

        # Periodic padding of the temperature field to treat boundaries correctly
        temp_pad = temp
        if sum(periodic_padding) > 0:
            temp_pad = F.pad(temp_pad.flatten(end_dim=-4), periodic_padding, mode="circular").reshape(
                -1,
                1,
                temp_pad.shape[-3] + periodic_padding[4] + periodic_padding[5],
                temp_pad.shape[-2] + periodic_padding[2] + periodic_padding[3],
                temp_pad.shape[-1] + periodic_padding[0] + periodic_padding[1],
            )
        # Constant padding of the displacement field to treat boundaries correctly
        if sum(constant_padding) > 0:
            temp_pad = F.pad(temp_pad.flatten(end_dim=-4), constant_padding, mode="constant", value=0.0).reshape(
                -1,
                1,
                temp_pad.shape[-3] + constant_padding[4] + constant_padding[5],
                temp_pad.shape[-2] + constant_padding[2] + constant_padding[3],
                temp_pad.shape[-1] + constant_padding[0] + constant_padding[1],
            )
        output_shape[-1] = temp_pad.shape[-1] - 1
        output_shape[-2] = temp_pad.shape[-2] - 1

        # Compute temperature gradient tensor using a 2d convolution
        g = F.conv3d(temp_pad, kernels.flatten(end_dim=-5))
        return g.reshape(*temp.shape[:-4], kernels.shape[0], self.n_dim, *g.shape[-3:])

    def get_grad_module(self, dtype, device, unflatten=True) -> torch.nn.Module:
        grad_kernels = self.grad_kernels.to(dtype=dtype, device=device)
        if self.n_dim == 2:
            conv_grad = torch.nn.Conv2d(self.n_channels, self.n_dim * self.n_gauss, grad_kernels.shape[-self.n_dim:],
                                        bias=False, device=device, dtype=dtype)
            conv_grad.weight = torch.nn.Parameter(grad_kernels.flatten(end_dim=-(2 + self.n_dim)), requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad2d((0, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad2d((1, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad2d((0, 0, 0, 1)), torch.nn.ZeroPad2d((1, 1, 0, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad2d((0, 1, 0, 0)), torch.nn.ZeroPad2d((0, 0, 1, 1))]
            else:
                raise NotImplementedError(f"Boundary condition {self.bc} is not supported")
            
        elif self.n_dim == 3:
            conv_grad = torch.nn.Conv3d(self.n_channels, self.n_dim * self.n_gauss, grad_kernels.shape[-self.n_dim:],
                                        bias=False, device=device, dtype=dtype)
            conv_grad.weight = torch.nn.Parameter(grad_kernels.flatten(end_dim=-(2 + self.n_dim)), requires_grad=False)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad3d((0, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET:
                padding_layers = [torch.nn.ZeroPad3d((1, 1) * self.n_dim)]
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad3d((0, 0, 0, 1, 0, 1)), torch.nn.ZeroPad3d((1, 1, 0, 0, 0, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad3d((0, 1, 0, 1, 0, 0)), torch.nn.ZeroPad3d((0, 0, 0, 0, 1, 1))]
            else:
                raise NotImplementedError(f"Boundary condition {self.bc} is not supported")
            
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

        model = torch.nn.Sequential(
            *padding_layers,
            conv_grad,
            torch.nn.Unflatten(1, (self.n_gauss, self.n_dim))
        )
        return model

    def div(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Compute the divergence of a periodic vector field flux

        :param flux:
        :return:
        """
        if flux.ndim == self.n_dim + 1:
            flux = torch.unsqueeze(flux, dim=0)
        assert flux.ndim >= self.n_dim + 2
        assert flux.shape[self.ch_dim] == self.n_dim

        if self.n_dim == 2:
            return self.div_2d(flux)
        elif self.n_dim == 3:
            return self.div_3d(flux)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def div_2d(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D divergence of a periodic vector field flux

        :param flux:
        :return:
        """
        # Construct kernels if not given
        kernels = self.div_kernels.to(device=flux.device, dtype=flux.dtype)

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [1, 0, 0, 0]
            constant_padding = [0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [0, 0, 1, 0]
            constant_padding = [0, 0, 0, 0]
        else:
            raise NotImplementedError()

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        # Original shape of flux is (..., L, G, 2, X, Y)
        # Output shape of res will be (..., L, 1, X, Y)
        original_shape = torch.tensor(flux.shape)
        output_shape = torch.cat([original_shape[:-4], original_shape[-3:]])
        output_shape[-3] = 1

        # Padding of the flux field to treat boundaries correctly
        # Reshape flux before padding to (-1, (G * 2), X, Y)
        flux_reshaped = flux.flatten(end_dim=-5).flatten(start_dim=-4, end_dim=-3)
        # Shape of flux_pad after padding is (-1, (G * 2), X + 1, Y + 1)
        # Periodic padding
        flux_pad = flux_reshaped
        if sum(periodic_padding) > 0:
            flux_pad = F.pad(flux_pad, periodic_padding, mode="circular")
        # Constant padding
        if sum(constant_padding) > 0:
            flux_pad = F.pad(flux_pad, constant_padding, mode="constant", value=0.0)

        # Compute residual field using a 2d convolution
        vol_e = 1.0 / torch.tensor(flux.shape[-2:]).prod()
        res = F.conv2d(flux_pad, kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2)) * vol_e / kernels.shape[0]
        output_shape[-1] = flux_pad.shape[-1] - 1
        output_shape[-2] = flux_pad.shape[-2] - 1
        return res.reshape(*flux.shape[:-4], 1, *res.shape[-2:])

    def div_3d(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Compute the 3D divergence of a periodic vector field flux

        :param flux:
        :return:
        """
        # Construct kernels if not given
        kernels = self.div_kernels.to(device=flux.device, dtype=flux.dtype)

        # Determine appropriate padding
        if self.bc == BC.PERIODIC:
            periodic_padding = [1, 0, 1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET:
            periodic_padding = [0, 0, 0, 0, 0, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_LR:
            periodic_padding = [1, 0, 0, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        elif self.bc == BC.DIRICHLET_TB:
            periodic_padding = [0, 0, 1, 0, 1, 0]
            constant_padding = [0, 0, 0, 0, 0, 0]
        else:
            raise NotImplementedError()

        # Prepare reshaping as F.pad and F.conv2d only accept 3D and 4D tensors
        # Original shape of flux is (..., L, G, 3, X, Y, Z)
        # Output shape of res will be (..., L, 1, X, Y, Z)
        original_shape = torch.tensor(flux.shape)
        output_shape = torch.cat([original_shape[:(self.ch_dim - 1)], original_shape[self.ch_dim:]])
        output_shape[self.ch_dim] = 1

        # Padding of the stress field to treat boundaries correctly
        # Reshape flux before padding to (-1, (G * 6), X, Y, Z)
        flux_reshaped = flux.flatten(end_dim=-6).flatten(start_dim=self.ch_dim - 1, end_dim=self.ch_dim)
        # Shape of flux_pad after padding is (-1, (G * 6), X + 1, Y + 1, Z + 1)
        # Periodic padding
        flux_pad = flux_reshaped
        if sum(periodic_padding) > 0:
            flux_pad = F.pad(flux_pad, periodic_padding, mode="circular")
        # Constant padding
        if sum(constant_padding) > 0:
            flux_pad = F.pad(flux_pad, constant_padding, mode="constant", value=0.0)

        # Compute residual field using a 3d convolution
        vol_e = 1.0 / torch.tensor(flux.shape[-3:]).prod()
        res = F.conv3d(flux_pad, kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2)) * vol_e / kernels.shape[0]
        output_shape[-1] = flux_pad.shape[-1] - 1
        output_shape[-2] = flux_pad.shape[-2] - 1
        return res.reshape(*flux.shape[:-5], 1, *res.shape[-3:])

    def get_div_module(self, dtype, device, reduced=False) -> torch.nn.Module:
        div_kernels = self.div_kernels.to(dtype=dtype, device=device)
        if reduced:
            div_kernels = div_kernels.mean(0).unsqueeze(0)

        if self.n_dim == 2:
            Conv = torch.nn.Conv2d
            conv_div = Conv(self.n_dim * self.n_gauss, 1, div_kernels.shape[-self.n_dim:],
                            bias=False, device=device, dtype=dtype)
            vol_e = 1.0 / torch.tensor(self.n_grid).prod()
            conv_div_weight = div_kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2) * vol_e / div_kernels.shape[0]
            conv_div.weight = torch.nn.Parameter(conv_div_weight, requires_grad=True)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad2d((1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET:
                padding_layers = []
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad2d((0, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad2d((1, 0, 0, 0))]
            else:
                raise NotImplementedError(f"Boundary condition {self.bc} is not supported")
        elif self.n_dim == 3:
            # Define div Conv3D
            conv_div = torch.nn.Conv3d(3 * self.n_gauss, 1, div_kernels.shape[-self.n_dim:],
                                       bias=False, device=device, dtype=dtype)
            vol_e = 1.0 / torch.tensor(self.n_grid).prod()
            conv_div_weight = div_kernels.transpose(0, 1).flatten(start_dim=1, end_dim=2) * vol_e / div_kernels.shape[0]
            conv_div.weight = torch.nn.Parameter(conv_div_weight, requires_grad=True)

            if self.bc == BC.PERIODIC:
                padding_layers = [torch.nn.CircularPad3d((1, 0, 1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET:
                padding_layers = []
            elif self.bc == BC.DIRICHLET_TB:
                padding_layers = [torch.nn.CircularPad3d((0, 0, 1, 0, 1, 0))]
            elif self.bc == BC.DIRICHLET_LR:
                padding_layers = [torch.nn.CircularPad3d((1, 0, 0, 0, 1, 0))]
            else:
                raise NotImplementedError(f"Boundary condition {self.bc} is not supported")
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        
        model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=2),
            *padding_layers,
            conv_div,
        )
        return model

    def get_ke(self, cond_field: torch.Tensor) -> torch.Tensor:
        """
        Get element stiffness

        :param cond_field:
        :return: element stiffness tensor with shape batch_size x *n_grid x 4 x 4
        """
        while cond_field.ndim < 2 + self.n_dim:
            cond_field = cond_field.unsqueeze(-1)

        assert cond_field.ndim >= 4
        assert cond_field.shape[-1] == cond_field.shape[-2]

        args = {"dtype": cond_field.dtype, "device": cond_field.device}
        vol_e = 1. / torch.tensor(self.n_grid).prod()
        b = self.get_grad_operator().to(**args)
        bt = b.transpose(-1, -2)

        # Loop over Gauss points
        ke = torch.einsum("ij,...jk" + self.einsum_dims + ",kl->..." + self.einsum_dims + "il",
                               bt[0], cond_field, b[0]) * vol_e / self.n_gauss
        for g in range(1, self.n_gauss):
            ke += torch.einsum("ij,...jk" + self.einsum_dims + ",kl->..." + self.einsum_dims + "il",
                               bt[g], cond_field, b[g]) * vol_e / self.n_gauss
        return ke

    def compute_fourier_kernel(self, cond: torch.Tensor, full=False) -> torch.Tensor:
        """

        :param cond:
        :return:
        """
        if self.n_dim == 2:
            return self.compute_fourier_kernel_2d(cond, full=full)
        elif self.n_dim == 3:
            return self.compute_fourier_kernel_3d(cond, full=full)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    def compute_fourier_kernel_2d(self, cond: Optional[torch.Tensor] = None, full=False) -> torch.Tensor:
        """

        :param cond:
        :return:
        """
        ny, nx = self.n_grid
        nh = nx if full else nx // 2 + 1

        complex_type = self.get_complex_dtype(cond.dtype)
        range_nx = torch.arange(nx, device=cond.device).type(complex_type)
        range_ny = torch.arange(ny, device=cond.device).type(complex_type)
        xi1 = torch.exp(2 * math.pi * 1j * 2 * range_nx / (2 * nx))[:nh]
        xi2 = torch.exp(2 * math.pi * 1j * 2 * range_ny / (2 * ny))
        ones_x = torch.ones(nh, dtype=cond.dtype, device=cond.device)
        ones_y = torch.ones(ny, dtype=cond.dtype, device=cond.device)

        A = torch.zeros((4, ny, nh), dtype=complex_type, device=cond.device)
        A[0] = ones_x[None, :] * ones_y[:, None]
        A[1] = xi1[None, :] * ones_y[:, None]
        A[2] = ones_x[None, :] * xi2[:, None]
        A[3] = xi1[None, :] * xi2[:, None]
        AA = torch.einsum("iyx,jyx->ijyx", A, A.conj())

        ke = self.get_ke(cond).squeeze().to(dtype=AA.dtype, device=AA.device)
        fk = torch.einsum("ij,ijyx->yx", ke, AA).real
        return fk

    def compute_fourier_kernel_3d(self, cond: Optional[torch.Tensor] = None, full=False) -> torch.Tensor:
        """
        Compute the Fourier kernel of the homogeneous stiffness matrix related to cond in 3D

        :param cond: conductivity tensor for the homogeneous stiffness matrix
        :return: Fourier kernel
        """
        nz, ny, nx = self.n_grid
        nh = nx if full else nx // 2 + 1

        # Construct frequency variables in all 3 dimensions
        complex_type = self.get_complex_dtype(cond.dtype)
        range_nx = torch.arange(nx, device=cond.device).type(complex_type)
        range_ny = torch.arange(ny, device=cond.device).type(complex_type)
        range_nz = torch.arange(nz, device=cond.device).type(complex_type)
        xi1 = torch.exp(2.0 * math.pi * 1j * range_nx / nx)[:nh]
        xi2 = torch.exp(2.0 * math.pi * 1j * range_ny / ny)
        xi3 = torch.exp(2.0 * math.pi * 1j * range_nz / nz)
        ones_x = torch.ones(nh, dtype=cond.dtype, device=cond.device)
        ones_y = torch.ones(ny, dtype=cond.dtype, device=cond.device)
        ones_z = torch.ones(nz, dtype=cond.dtype, device=cond.device)

        # Assemble auxiliary vector field with 8 components on the 3D domain for the Fourier kernel
        # This vector field can be interpreted as Fourier representation of the FEM grad/B operator
        A = torch.zeros((8, nz, ny, nh), dtype=complex_type, device=cond.device)
        A[0] = ones_x[None, None, :] * ones_y[None, :, None] * ones_z[:, None, None]
        A[1] = xi1[None, None, :] * ones_y[None, :, None] * ones_z[:, None, None]
        A[2] = ones_x[None, None, :] * xi2[None, :, None] * ones_z[:, None, None]
        A[3] = xi1[None, None, :] * xi2[None, :, None] * ones_z[:, None, None]
        A[4] = ones_x[None, None, :] * ones_y[None, :, None] * xi3[:, None, None]
        A[5] = xi1[None, None, :] * ones_y[None, :, None] * xi3[:, None, None]
        A[6] = ones_x[None, None, :] * xi2[None, :, None] * xi3[:, None, None]
        A[7] = xi1[None, None, :] * xi2[None, :, None] * xi3[:, None, None]

        # Compute Fourier kernel based on the element stiffness matrix ke and the auxiliary vector field A
        AA = torch.einsum("i" + self.einsum_dims + ",j" + self.einsum_dims + "->ij" + self.einsum_dims, A, A.conj())
        ke = self.get_ke(cond).squeeze().to(dtype=AA.dtype, device=AA.device)
        fk = torch.einsum("ij,ij" + self.einsum_dims + "->" + self.einsum_dims, ke, AA).real
        return fk

    def compute_fundamental_solution(self, cond: torch.Tensor, full=False) -> torch.Tensor:
        """
        Compute the FANS fundamental solution using a Fourier kernel for a given conductivity tensor cond

        :param cond: conductivity tensor for the homogeneous stiffness matrix
        :return: FANS fundamental solution
        """
        fk = self.compute_fourier_kernel(cond, full=full)

        tol = 1e-14
        phi_hat = torch.where(fk < tol, 0.0, 1.0 / fk)
        return phi_hat

    def get_grad_operator(self) -> torch.Tensor:
        """
        Get FEM gradient operator

        :return:
        """
        # TODO: more general, so far only 2D
        rule1d, rule1dw = self.get_quad_rule()

        n_gauss_dir = len(rule1d)
        n_gauss = n_gauss_dir ** self.n_dim
        n_nodes = 2 ** self.n_dim
        h = torch.ones(self.n_dim, dtype=self.dtype) / torch.tensor(self.n_grid, dtype=self.dtype)

        grad_operators = torch.zeros((n_gauss, self.n_dim, self.n_channels * n_nodes), dtype=self.dtype)

        if self.n_dim == 2:
            qw = torch.outer(rule1dw, rule1dw).ravel()  # weights in 2D
            qx = rule1d.expand((n_gauss_dir, n_gauss_dir)).T.ravel()
            qy = rule1d.expand((n_gauss_dir, n_gauss_dir)).ravel()

            B = lambda xi, eta: torch.stack(
                [
                    torch.stack([eta - 1.0, -(eta - 1.0), -eta, eta], dim=-1),
                    torch.stack([xi - 1.0, -xi, 1.0 - xi, xi], dim=-1),
                ],
                dim=-1,
            ).transpose(-1, -2)
            grad_operators = B(qx, qy) / h[None, :, None]
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
                grads = B(qx[i], qy[i], qz[i]) / h[:, None]
                grad_operators[i] = grads

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
            grad_kernels = self.grad_operator.reshape(-1, 2, 1, 2, 2).permute((0, 1, 2, 4, 3))
        elif self.n_dim == 3:
            grad_kernels = self.grad_operator.reshape(-1, 3, 1, 2, 2, 2).permute((0, 1, 2, 5, 4, 3))
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
        grad_kernels = self.grad_kernels

        if self.n_dim == 2:
            div_kernels = torch.roll(grad_kernels.transpose(-3, -4), shifts=(1, 1), dims=(-2, -1))
        elif self.n_dim == 3:
            div_kernels = torch.roll(grad_kernels.transpose(-4, -5), shifts=(1, 1, 1), dims=(-3, -2, -1))

        return div_kernels
    
    def get_temp_loss(self, reduction: str="sum"):
        """
        Temperature field loss
        """
        return TempLoss(n_dim=self.n_dim, reduction=reduction)
    
    def get_flux_loss(self, reduction: str="sum"):
        """
        Flux field loss
        """         
        return FluxLoss(n_dim=self.n_dim, reduction=reduction)
    
    def compute_losses(self, field_pred: torch.Tensor, field_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all relevant losses
        :param field_pred:
        :param field_ref:
        :param detach: detach grad
        :param cpu: transfer to cpu
        :return:
        """
        args = {"dtype": field_ref.dtype, "device": field_ref.device}
        field_pred = field_pred.to(**args)
        losses = {
            "temp": self.get_temp_loss(reduction="none")(field_pred, field_ref),
            "flux": self.get_flux_loss(reduction="none")(field_pred, field_ref),
        }
        return losses
