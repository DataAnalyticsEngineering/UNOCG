"""
General structure of a problem formulation
"""
from abc import ABC
import math
from enum import Enum
from typing import Optional, Union, Iterable, Tuple, Dict, Callable, Collection, List
from unocg.materials import Material
import numpy as np
import scipy as sp
import torch
from scipy import sparse
from tqdm import tqdm


class BC(Enum):
    PERIODIC = 0
    DIRICHLET_TB = 1
    DIRICHLET_LR = 2
    DIRICHLET = 3
    DIRICHLET_TB_X = 4
    DIRICHLET_TB_Y = 5
    DIRICHLET_LR_X = 6
    DIRICHLET_LR_Y = 7
    DIRICHLET_X = 8
    DIRICHLET_Y = 9
    DIRICHLET_FB = 10
    NEUMANN = 11

class FieldType(Enum):
    NODE = 0
    CELL = 1
    HOM = 2
    GAUSS = 3


class Problem:
    def __init__(self, n_grid: Union[torch.Size, Collection[int]], bc: Optional[BC] = None, material: Optional[Material] = None, quad_degree: int = 2, lengths = None):
        if bc is None:
            self.bc = BC.PERIODIC
        else:
            self.bc = bc
        self._n_dim = len(n_grid)
        self._n_channels = 1
        self.dtype = torch.float64
        self._n_grid = n_grid
        self.dof_shape = self._n_grid
        self._n_dof = None
        self._loadings = None
        self.quad_degree = quad_degree
        self.material = material
        if lengths is None:
            self.lengths = torch.ones(self.n_dim, dtype=self.dtype)
        else:
            self.lengths = torch.tensor(lengths, dtype=self.dtype)
        self.h = self.lengths / torch.tensor(self.n_grid, dtype=self.dtype)

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def dims(self):
        return self.n_dim * (slice(None),)

    @property
    def dims_list(self):
        return tuple(range(-1, -(1 + self.n_dim), -1))

    @property
    def expand_dims(self):
        return self.n_dim * (None,)

    @property
    def einsum_dims(self):
        if self.n_dim == 2:
            return "yx"
        elif self.n_dim == 3:
            return "zyx"
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def ch_dim(self):
        """
        Channel dimension

        :return:
        """
        return -(1 + self.n_dim)


    @property
    def n_grid(self) -> Union[torch.Size, Collection[int]]:
        return self._n_grid

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def loadings(self) -> torch.Tensor:
        return self._loadings

    @loadings.setter
    def loadings(self, loadings: torch.Tensor):
        self._loadings = loadings

    @property
    def n_loadings(self):
        return self.loadings.shape[0]

    def get_vec_shape(self, param_fields: torch.Tensor, loadings: Optional[torch.Tensor] = None) -> torch.Size:
        """
        Get shape for DOF vectors

        :param param_fields:
        :param loadings:
        :return:
        """
        if loadings is None:
            n_loadings = self.n_loadings
        else:
            n_loadings = loadings.shape[0]

        batch_shape = param_fields.shape[: -(self.n_dim + 1)]
        return torch.Size((*batch_shape, n_loadings, self.n_dof))

    def get_field_shape(self, param_fields, loadings: Optional[torch.Tensor] = None):
        """
        Get shape for fields

        :param param_fields:
        :param loadings:
        :return:
        """
        if loadings is None:
            n_loadings = self.n_loadings
        else:
            n_loadings = loadings.shape[0]

        batch_shape = param_fields.shape[:self.ch_dim]

        return torch.Size((*batch_shape, n_loadings, self.n_channels, *self.dof_shape))

    def reshape_vec(self, batch: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Reshape batch of fields to batch of DOF vectors

        :param batch:
        :return:
        """
        batch_shape = batch.shape[: -(self.n_dim + 1)]
        return batch.reshape((*batch_shape, self.n_dof))

    def reshape_field(self, batch: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Reshape batch of DOF vectors to batch of fields

        :param batch:
        :return:
        """
        batch_shape = batch.shape[:-1]
        return batch.reshape((*batch_shape, self.n_channels, *self.dof_shape))

    def compute_residual(self,
                         u: torch.Tensor,
                         param_fields: Optional[torch.Tensor] = None,
                         loadings: Optional[torch.Tensor] = None,
                         state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute residual

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        raise NotImplementedError()

    def matvec(self,
               d: torch.Tensor,
               param_fields: Optional[torch.Tensor] = None,
               state_fields: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Matrix-vector product with stiffness matrix

        :param d:
        :param param_fields:
        :param state_fields:
        :return:
        """
        raise NotImplementedError()

    def compute_field(
            self,
            u: torch.Tensor,
            param_fields: Optional[torch.Tensor] = None,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute field based on DOF vector u

        :param u:
        :param param_fields:
        :param loadings:
        :param state_fields:
        :return:
        """
        raise NotImplementedError()
    
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
        raise NotImplementedError()

    def split_fields(self, fields: torch.Tensor) -> Dict[str, Tuple[FieldType, torch.Tensor]]:
        """
        Split general field into problem-specific physically meaningful fields

        :param fields:
        :return:
        """
        raise NotImplementedError()

    def get_node_coords(self):
        """

        :return:
        """
        if self.n_dim == 2:
            ny, nx = self.n_grid
            ly, lx = self.lengths
            x = torch.linspace(0.0, lx, nx + 1)
            y = torch.linspace(0.0, ly, ny + 1)
            Y, X = torch.meshgrid(y, x, indexing="ij")
            vertices = torch.stack([X, Y], dim=-3)
        elif self.n_dim == 3:
            nz, ny, nx = self.n_grid
            lz, ly, lx = self.lengths
            x = torch.linspace(0.0, lx, nx + 1)
            y = torch.linspace(0.0, ly, ny + 1)
            z = torch.linspace(0.0, lz, nz + 1)
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
            vertices = torch.stack([X, Y, Z], dim=-4)
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        return vertices

    def get_dof_coords(self):
        """

        :return:
        """
        if self.n_dim == 2:
            ny, nx = self.n_grid
            vertices = self.get_node_coords()
            if self.bc == BC.PERIODIC:
                dof_vertices = vertices[...,1:,1:]
            elif self.bc == BC.DIRICHLET:
                dof_vertices = vertices[...,1:-1,1:-1]
            else:
                raise NotImplementedError(f"BC {self.bc} is not supported")
        elif self.n_dim == 3:
            nz, ny, nx = self.n_grid
            vertices = self.get_node_coords()
            if self.bc == BC.PERIODIC:
                dof_vertices = vertices[...,1:,1:,1:]
            elif self.bc == BC.DIRICHLET:
                dof_vertices = vertices[...,1:-1,1:-1,1:-1]
            else:
                raise NotImplementedError(f"BC {self.bc} is not supported")
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        return dof_vertices


    def get_cell_coords(self):
        """

        :return:
        """
        if self.n_dim == 2:
            ny, nx = self.n_grid
            ly, lx = self.lengths
            x_c = torch.linspace(0.5 / nx, lx - 0.5 / nx, nx)
            y_c = torch.linspace(0.5 / ny, ly - 0.5 / ny, ny)
            Y_c, X_c = torch.meshgrid(y_c, x_c, indexing="ij")
            vertices_c = torch.stack([X_c, Y_c], dim=-3).flatten(start_dim=1).T
        elif self.n_dim == 3:
            nz, ny, nx = self.n_grid
            lz, ly, lx = self.lengths
            x_c = torch.linspace(0.5 / nx, lx - 0.5 / nx, nx)
            y_c = torch.linspace(0.5 / ny, ly - 0.5 / ny, ny)
            z_c = torch.linspace(0.5 / nz, lz - 0.5 / nz, nz)
            Z_c, Y_c, X_c = torch.meshgrid(z_c, y_c, x_c, indexing="ij")
            vertices_c = torch.stack([X_c, Y_c, Z_c], dim=-4).flatten(start_dim=1).T
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported")
        return vertices_c

    @staticmethod
    def get_scatter_matrix_sp(dof: torch.Tensor, shape=None):
        """

        :param dof:
        :param shape:
        :return:
        """
        if shape is None:
            shape = (dof.shape[0], dof.shape[0])

        co = np.stack([np.arange(dof.shape[0]), dof])
        da = np.ones(dof.shape[0])
        L = sp.sparse.coo_array((da, co), shape=shape).tocsr()
        return L

    @staticmethod
    def get_scatter_matrix(dof: torch.Tensor, shape=None):
        """

        :param dof:
        :param shape:
        :return:
        """
        if shape is None:
            shape = (dof.shape[0], dof.shape[0])

        co = torch.stack([torch.arange(dof.shape[0], dtype=int, device=dof.device), dof])
        da = torch.ones(dof.shape[0], device=dof.device)
        L = torch.sparse_coo_tensor(co, da, size=shape).coalesce()
        return L

    def get_quad_rule(self):
        if self.quad_degree == 1:
            rule1d = torch.tensor([0.5], dtype=self.dtype)
            rule1dw = torch.tensor([1.0], dtype=self.dtype)
        elif self.quad_degree == 2:
            rule1d = torch.tensor([0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0], dtype=self.dtype)
            rule1dw = torch.tensor([0.5, 0.5], dtype=self.dtype)
        elif self.quad_degree == 3:
            rule1d = torch.tensor([0.5 - math.sqrt(0.6) / 2.0, 0.5 + math.sqrt(0.6) / 2.0], dtype=self.dtype)
            rule1dw = torch.tensor([5. / 18., 4. / 9., 5. / 18.], dtype=self.dtype)
        else:
            raise NotImplementedError(f"Quadrature degree {self.quad_degree} is not supported")
        return rule1d, rule1dw

    @staticmethod
    def extract_image(input_field: torch.Tensor) -> torch.Tensor:
        """
        Extract image

        :param input_field:
        :return:
        """
        return input_field[..., 0, :, :]

    @staticmethod
    def extract_param_fields(input_field: torch.Tensor) -> torch.Tensor:
        """
        Extract parameter fields

        :param input_field:
        :return:
        """
        return input_field[..., 1:, :, :]

    @staticmethod
    def extract_parameters(input_field: torch.Tensor) -> torch.Tensor:
        """
        Extract parameters

        :param input_field:
        :return: params
        """
        if input_field.ndim == 3:
            input_field = torch.unsqueeze(input_field, dim=0)

        params_view = input_field.view(*input_field.shape[:-2], -1)
        idx0 = params_view[..., 0, :].argmin(dim=-1)
        idx1 = params_view[..., 0, :].argmax(dim=-1)
        params0 = params_view[torch.arange(input_field.shape[0]), 1:, idx0]
        params1 = params_view[torch.arange(input_field.shape[0]), 1:, idx1]
        params = torch.stack([params0, params1], dim=-2)
        return params

    def extract_dof(self, field: torch.Tensor) -> torch.Tensor:
        """
        Extract DOF vector from field solution

        :param field:
        :return:
        """
        if self.bc == BC.PERIODIC:
            dof = field[..., :-1, :-1].ravel()
        elif self.bc == BC.DIRICHLET_LR:
            dof = field[..., :-1, 1:-1].ravel()
        elif self.bc == BC.DIRICHLET_TB:
            dof = field[..., 1:-1, :-1].ravel()
        elif self.bc == BC.DIRICHLET:
            dof = field[..., 1:-1, 1:-1].ravel()
        else:
            raise NotImplementedError()
        return dof

    def get_param_fields(self, image, params):
        """
        Construct parameter fields for a two-phase microstructure described by image based on params.

        :param image:
        :param params:
        :return:
        """
        param_fields = torch.einsum("..." + self.einsum_dims + ",i->...i" + self.einsum_dims, (image.to(dtype=int) == 0).to(dtype=params.dtype), params[0])
        for i in range(1, params.shape[0]):
            param_fields += torch.einsum("..." + self.einsum_dims + ",i->...i" + self.einsum_dims, (image.to(dtype=int) == i).to(dtype=params.dtype), params[i])
        return param_fields

    def homogenize(self, field: torch.Tensor) -> torch.Tensor:
        """
        Homogenization of a tensor field on a 2d grid by volume averaging

        :param field: tensor field with shape [..., n, n]
        :return:
        """
        return field.nanmean(self.dims_list)

    def field_norm(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute the point-wise norm of a tensor field

        :param field:
        :return: point-wise norm
        """
        return torch.sqrt(torch.square(field).nansum(self.ch_dim))

    def hom_norm(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute the homogenized norm of a tensor field

        :param field:
        :return: homogenized norm
        """
        return self.field_norm(field).nanmean(self.dims_list)

    def zero_mean(self, u: torch.Tensor) -> torch.Tensor:
        """
        Enforce zero-mean property for solution vector u
        """
        field = self.reshape_field(u)
        field_zero_mean = field - field.mean(self.dims_list)[..., *self.expand_dims]
        return self.reshape_vec(field_zero_mean)

    @classmethod
    def pad_circular_nd(cls, x: torch.Tensor, pad: int, dim) -> torch.Tensor:
        """
        Periodic padding for tensor fields on a 2d grid

        :param x: shape [H, W] of the grid
        :param pad: int >= 0
        :param dim: the dimension over which the tensors are padded
        :return:
        """
        if isinstance(dim, int):
            dim = [dim]

        for d in dim:
            if d >= len(x.shape):
                raise IndexError(f"dim {d} out of range")

            idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
            x = torch.cat([x, x[idx]], dim=d)

            idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
            x = torch.cat([x[idx], x], dim=d)
            pass

        return x

    @classmethod
    def pad_circular_2d(cls, x: torch.Tensor, pad_left: int, pad_right: int, mode: str = "circular") -> torch.Tensor:
        """
        Periodic padding for tensor fields on a 2d grid

        :param x: shape [H, W] of the grid
        :param pad_left: int >= 0
        :param pad_right: int >= 0
        :param mode: str {'circular', 'constant', 'reflect', 'replicate'}
        :return:
        """
        paddings = [pad_left, pad_right, pad_left, pad_right]
        original_shape = torch.tensor(x.shape)
        padded_shape = original_shape.clone()
        padded_shape[-2:] += torch.tensor([pad_left + pad_right, pad_left + pad_right])
        return torch.nn.functional.pad(x.reshape(-1, *original_shape[-3:]), paddings, mode=mode).reshape(*padded_shape)

    def get_image_masks(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get index masks for

        - interior of matrix (interior_matr),
        - interface of matrix (interface_matr),
        - interface of inclusion (interface_incl),
        - interior of inclusion (interior_incl).

        :param image:
        :type image: torch.Tensor
        """
        # TODO: generalize for 3D
        image = image.unsqueeze(dim=self.ch_dim)
        kernel_tensor = torch.ones((1, 1, 3, 3)).to(device=image.device, dtype=image.dtype)
        img_erosion = 1 - torch.clamp(torch.nn.functional.conv2d(1 - image, kernel_tensor, padding=(1, 1)), 0, 1)
        img_dilation = torch.clamp(torch.nn.functional.conv2d(image, kernel_tensor, padding=(1, 1)), 0, 1)
        interface_incl = (image - img_erosion).bool().squeeze(dim=self.ch_dim)
        interface_matr = (img_dilation - image).bool().squeeze(dim=self.ch_dim)
        interior_incl = torch.clamp(img_erosion - interface_incl.int(), 0, 1).bool().squeeze(dim=self.ch_dim)
        interior_matr = (1 - img_dilation).bool().squeeze(dim=self.ch_dim)
        return interior_matr, interface_matr, interface_incl, interior_incl

    @staticmethod
    def get_complex_dtype(dtype: torch.dtype) -> torch.dtype:
        """

        :param dtype:
        :return:
        """
        complex_dtypes = {
            torch.float16: torch.complex32,
            torch.half: torch.complex32,
            torch.complex32: torch.complex32,
            torch.float32: torch.complex64,
            torch.float: torch.complex64,
            torch.complex64: torch.complex64,
            torch.cfloat: torch.complex64,
            torch.float64: torch.complex128,
            torch.double: torch.complex128,
            torch.complex128: torch.complex128,
            torch.cdouble: torch.complex128,
        }

        try:
            for dtype_normal, dtype_complex in complex_dtypes.items():
                if dtype_normal == dtype:
                    return dtype_complex
        except:
            raise ValueError("Unknown type")
        raise ValueError("Unknown type")

    @staticmethod
    def get_complex_dtype_sp(dtype: torch.dtype) -> torch.dtype:
        """

        :param dtype:
        :return:
        """
        complex_dtypes = {
            np.float32: np.complex64,
            np.complex64: np.complex64,
            np.cfloat: np.complex128,
            "float64": np.complex128,
            np.float64: np.complex128,
            np.double: np.complex128,
            np.complex128: np.complex128,
            np.cdouble: np.complex128,
        }

        try:
            for dtype_normal, dtype_complex in complex_dtypes.items():
                if dtype_normal == dtype:
                    return dtype_complex
        except:
            raise ValueError("Unknown type")
        raise ValueError("Unknown type")

    def get_shapes(
            self, input_field: torch.Tensor, channels: Optional[int] = 1, n_loadings: Optional[int] = 2
    ) -> Tuple[Iterable, Iterable, Iterable]:
        """

        :param input_field:
        :param channels:
        :param n_loadings:
        :return:
        """
        batch_shape = input_field.shape[:self.ch_dim]
        field_shape = self.get_field_shape(input_field)
        vec_shape = self.get_vec_shape(input_field)
        return batch_shape, field_shape, vec_shape

    def extract_shape(self, input_field: torch.Tensor) -> torch.Size:
        """
        Extract shape of discretization from input

        :param input_field:
        :return: shape of discretization
        """
        return input_field.shape[-self.n_dim:]

    @classmethod
    def fft2_real(cls, field: torch.Tensor) -> torch.Tensor:
        """
        Computes FFT of real-valued field using rfft2(field) and reconstructs the negative frequencies
        to match the output of fft2(field)

        :param field:
        :return:
        """
        # TODO: generalize or remove
        assert field.shape[-1] == field.shape[-2]
        n = field.shape[-1]
        nh = n // 2

        rfft_field = torch.fft.rfft2(field)
        fft_field = torch.zeros_like(field, dtype=cls.get_complex_dtype(field.dtype))
        fft_field[..., :, : (nh + 1)] = rfft_field  # 0th and positive frequencies from real-valued fft

        # reconstruct negative frequencies
        fft_field[..., -nh:, -nh:] = torch.flip(rfft_field[..., 1: (nh + 1), 1: (nh + 1)].conj(), dims=[-1, -2])
        fft_field[..., 1: (nh + 1), -nh:] = torch.flip(rfft_field[..., -nh:, 1: (nh + 1)].conj(), dims=[-1, -2])
        fft_field[..., 0, -nh:] = torch.flip(rfft_field[..., 0, 1: (nh + 1)].conj(), dims=[-1])
        return fft_field

    @classmethod
    def dft_matrix(cls, shape):
        # TODO: generalize or remove
        return torch.fft.fft(torch.eye(shape[-1]))

    def get_coordinates(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get coordinates of all nodes in the image

        :param image: tensor with shape (..., H, W)
        :param lx: length of x direction
        :param ly: length of y direction
        :return: coordinates with shape (C, H, W); C=2
        """
        #return self.get_node_coords().to(dtype=image.dtype, device=image.device)
        if self.n_dim == 2:
            ny, nx = image.shape[-2:]
            ly, lx = self.lengths
            x = torch.linspace(-lx / 2.0, lx / 2.0 - 1.0 / nx, nx, dtype=image.dtype, device=image.device)
            y = torch.linspace(-ly / 2.0, ly / 2.0 - 1.0 / ny, ny, dtype=image.dtype, device=image.device)
            Y, X = torch.meshgrid(y, x, indexing="ij")
            return torch.stack([X, Y], dim=self.ch_dim)
        elif self.n_dim == 3:
            nz, ny, nx = image.shape[-3:]
            lz, ly, lx = self.lengths
            x = torch.linspace(-lx / 2.0, lx / 2.0 - 1.0 / nx, nx, dtype=image.dtype, device=image.device)
            y = torch.linspace(-ly / 2.0, ly / 2.0 - 1.0 / ny, ny, dtype=image.dtype, device=image.device)
            z = torch.linspace(-lz / 2.0, lz / 2.0 - 1.0 / nz, nz, dtype=image.dtype, device=image.device)
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
            return torch.stack([X, Y, Z], dim=self.ch_dim)

    def mandel_to_matrix(self, tensor_mandel: torch.Tensor) -> torch.Tensor:
        """

        :param tensor_mandel: strain/stress tensor in mandel notation with shape (..., 3)
        :return: tensor in matrix notation with shape (..., 2, 2)
        """
        if self.n_dim == 2:
            tensor_matrix = torch.stack(
                [
                    torch.stack([tensor_mandel[..., 0], tensor_mandel[..., 2] / math.sqrt(2.0)], dim=-1),
                    torch.stack([tensor_mandel[..., 2] / math.sqrt(2.0), tensor_mandel[..., 1]], dim=-1),
                ],
                dim=-1,
            )
        elif self.n_dim == 3:
            tensor_matrix = torch.stack(
                [
                    torch.stack([tensor_mandel[..., 0], tensor_mandel[..., 5] / math.sqrt(2.0), tensor_mandel[..., 4] / math.sqrt(2.0)], dim=-1),
                    torch.stack([tensor_mandel[..., 5] / math.sqrt(2.0), tensor_mandel[..., 1], tensor_mandel[..., 3] / math.sqrt(2.0)], dim=-1),
                    torch.stack([tensor_mandel[..., 4] / math.sqrt(2.0), tensor_mandel[..., 3] / math.sqrt(2.0), tensor_mandel[..., 2]], dim=-1),
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(f"Dimension {self.n_dim} is not supported for this operation")
        return tensor_matrix

    def get_boundary_idx(self, shape: Union[torch.Size, Iterable[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param shape:
        :return:
        """
        # TODO: generalize or remove
        x = torch.arange(shape[-1])
        y = torch.arange(shape[-2])
        Y, X = torch.meshgrid(y, x, indexing="ij")
        idx = torch.stack([X, Y], dim=0)
        boundary_idx = torch.cat(
            [idx[:, 0, :], idx[:, :, -1][..., 1:], idx[:, -1, :][..., :-1].flip(-1), idx[:, :, 0].flip(-1)], dim=-1
        )
        return boundary_idx[0].ravel(), boundary_idx[1].ravel()

    def generate_circle_ms(
            self,
            r: float = 0.4,
            cx: float = 0.0,
            cy: float = 0.0,
            cz: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate microstructure of given shape with spherical inclusion with center (cx, cy) and with radius r

        :param shape:
        :param r:
        :param cx:
        :param cy:
        :param lx:
        :param ly:
        :return: microstructure
        """
        # TODO: generalize or remove
        if self.n_dim == 2:
            ny, nx = self.n_grid[-2:]
            ly, lx = self.lengths[-2:]
            hx, hy = lx / nx, ly / ny
            x = torch.linspace(-lx / 2.0 + hx / 2.0, lx / 2.0 - hx / 2.0, nx)
            y = torch.linspace(-ly / 2.0 + hy / 2.0, ly / 2.0 - hy / 2.0, ny)
            Y, X = torch.meshgrid(y, x, indexing="ij")
            ms = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2).to(dtype=torch.int)
            return ms
        elif self.n_dim == 3:
            nz, ny, nx = self.n_grid[-3:]
            lz, ly, lx = self.lengths[-3:]
            hx, hy, hz = lx / nx, ly / ny, lz / nz
            x = torch.linspace(-lx / 2.0 + hx / 2.0, lx / 2.0 - hx / 2.0, nx)
            y = torch.linspace(-ly / 2.0 + hy / 2.0, ly / 2.0 - hy / 2.0, ny)
            z = torch.linspace(-lz / 2.0 + hz / 2.0, lz / 2.0 - hz / 2.0, nz)
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
            ms = ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2).to(dtype=torch.int)
            return ms
        else:
            raise NotImplementedError(f"Microstructure generation for dimension {self.n_dim} is not supported")

    def generate_ellipsoid_ms(self, ellipses):
        """
        Create an (nx, ny) image with ellipses.

        Args:
            nx (int): image height (rows)
            ny (int): image width (columns)
            ellipses (list of dict): list of ellipses,
                each dict must contain:
                - "center": (x0, y0) ellipse center
                - "axes": (a, b) semi-major and semi-minor axes
                - "angle": theta (rotation angle in radians)
                - "value": fill value (default=1.0)
            device: torch device ("cpu" or "cuda")
            dtype: tensor dtype

        Returns:
            torch.Tensor: (nx, ny) image with ellipses
        """
        if self.n_dim != 2:
            raise NotImplementedError()

        nx, ny = self.n_grid

        # Coordinate grid
        ys = torch.arange(nx).view(-1, 1).expand(nx, ny)
        xs = torch.arange(ny).view(1, -1).expand(nx, ny)
        
        # Empty image
        img = torch.zeros((nx, ny))
        
        for e in ellipses:
            x0, y0 = e["center"]
            a, b = e["axes"]
            theta = torch.tensor(e.get("angle", 0.0))
            val = e.get("value", 1.0)

            # Periodic coordinate differences
            dx = ((xs - x0 + ny/2) % ny) - ny/2
            dy = ((ys - y0 + nx/2) % nx) - nx/2

            # Rotate coordinates
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t

            # Ellipse mask
            mask = (x_rot**2) / (a**2) + (y_rot**2) / (b**2) <= 1.0

            # Overwrite pixels
            img = torch.where(mask, val * torch.ones_like(img), img)
    
        return img

    @staticmethod
    def torch_sparse_to_sp(tensor: torch.Tensor) -> sparse.coo_array:
        """
        Convert PyTorch sparse tensor to scipy sparse array

        :param tensor:
        :return:
        """
        tensor = tensor.to_sparse_coo().coalesce()
        sp_array = sparse.coo_array(
            (tensor.values().detach().cpu().numpy(), (tensor.indices()[0].detach().cpu().numpy(), tensor.indices()[1].detach().cpu().numpy())),
            shape=tensor.shape
        )
        return sp_array

    """
    def petsc_sparse_to_sp(mat: PETSc.Mat) -> sparse.coo_array:

        Convert PETSc sparse tensor to scipy sparse array

        :param tensor:
        :return:

        tensor = tensor.to_sparse_coo().coalesce()
        sp_array = sparse.coo_array(
            (tensor.values().numpy(), (tensor.indices()[0].numpy(), tensor.indices()[1].numpy())), shape=tensor.shape
        )
        return sp_array
    """

    @staticmethod
    def sp_sparse_to_torch(sp_array: sparse.coo_array) -> torch.Tensor:
        """
        Convert scipy sparse array to PyTorch sparse tensor

        :param sp_array:
        :return:
        """
        sp_array = sp_array.tocoo()
        values = sp_array.data
        indices = np.vstack((sp_array.row, sp_array.col))

        i = torch.LongTensor(indices)
        v = torch.tensor(values)
        shape = sp_array.shape

        sparse_torch = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        return sparse_torch

    @staticmethod
    def sparse_diag(sparse_tensor: torch.Tensor, return_sparse: bool = False) -> torch.Tensor:
        """
        Extract diagonal of sparse tensor as PyTorch has no native implementation for it so far

        :param sparse_tensor: PyTorch sparse tensor
        :param return_sparse: If a sparse diagonal matrix should be returned of a vector containing only the diagonal
        :return:
        """
        assert sparse_tensor.ndim == 2
        assert sparse_tensor.shape[-1] == sparse_tensor.shape[-2]

        sparse_tensor = sparse_tensor.to_sparse_coo().coalesce()
        idx = sparse_tensor.indices()
        val = sparse_tensor.values()
        diag_mask = idx[0, :] == idx[1, :]
        idx_diag = idx[:, diag_mask]
        val_diag = val[diag_mask]

        if return_sparse:
            diag = torch.sparse_coo_tensor(idx_diag, val_diag)
        else:
            diag = torch.zeros(sparse_tensor.shape[-1], device=sparse_tensor.device)
            diag[idx_diag[0, :]] = val_diag

        return diag

    @staticmethod
    def matvec_to_sparse(
            shape: Union[Tuple[int, int], torch.Size],
            matvec: Callable[[torch.Tensor], torch.Tensor],
            eps: float = 1e-16,
            progress: bool = False,
    ) -> torch.Tensor:
        """
        Converts a matrix-free matvec function to the corresponding sparse PyTorch matrix

        :param shape: shape of the matrix
        :param matvec: matrix-free matvec function
        :param eps: tolerance for non-zero entries
        :param progress: show progress bar
        :return:
        """
        assert shape[-2] == shape[-1]
        N = shape[-1]
        idx = torch.arange(N, dtype=torch.int)
        indices = torch.zeros((2, 0), dtype=torch.int)
        values = torch.tensor([])

        for i in tqdm(idx, disable=not progress):
            u = torch.zeros(N)
            u[i] = 1.0
            r = matvec(u)
            mask = torch.abs(r) > 1e-16
            idx_x = torch.repeat_interleave(torch.unsqueeze(i, 0), mask.sum())
            idx_y = idx[mask.ravel()]
            current_indices = torch.stack([idx_x, idx_y])
            indices = torch.cat([indices, current_indices], dim=-1)
            values = torch.cat([values, r[mask].ravel()], dim=-1)
        A_sparse = torch.sparse_coo_tensor(indices, values).coalesce()
        return A_sparse

    @staticmethod
    def setdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """

        :param a:
        :param b:
        :return:
        """
        combined = torch.cat((a.unique(), b.unique()))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]
        return difference
    
    def get_deformations(self, disp, loading, fluctuation_scaling: float = 1.0):
        coords = self.get_coordinates(disp)
        loading_matrix = self.mandel_to_matrix(loading).to(dtype=disp.dtype, device=disp.device)
        deformations = torch.einsum("...ij,j" + self.einsum_dims + "->...i" + self.einsum_dims, loading_matrix, coords) + fluctuation_scaling * disp
        return deformations

    def get_deformed_coordinates(
        self, disp: torch.Tensor, loading, lx=1.0, ly=1.0, fluctuation_scaling: float = 1.0, deformation_scaling: float = 1.0
    ) -> torch.Tensor:
        """
        Get coordinates in deformed configuration according to displacement fluctuations disp and loading
        :param disp: tensor with shape of (..., L, 2, H, W)
        :param loading: strain loading in mandel notation with shape (L, 3)
        :param lx: length along x-axis
        :param ly: length along y-axis
        :param fluctuation_scaling: scaling factor of displacement fluctuations, defaults to 1.0
        :param deformation_scaling: scaling factor of displacement fluctuations, defaults to 1.0
        :return: coordinates with shape (..., L, C, H, W); C=2
        """
        coords = self.get_coordinates(disp)
        loading_matrix = self.mandel_to_matrix(loading).to(dtype=disp.dtype, device=disp.device)
        deformations = torch.einsum("...ij,jyx->...iyx", loading_matrix, coords) + fluctuation_scaling * disp
        deformed_coords = coords + deformation_scaling * deformations
        return deformed_coords
