"""
PyTorch solver modules
"""
from typing import Optional, Union, Tuple

import torch


class CgModule(torch.nn.Module):
    def __init__(
        self,
        n_channels: int,
        dof_shape: Union[torch.Size, Tuple],
        rhs_model: torch.nn.Module,
        matvec_model: torch.nn.Module,
        prec_model: Optional[torch.nn.Module] = None,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        n_layers: int = None,
        alg_type: str = "cg",
        err_eps: float = 1e-10,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dof_shape = dof_shape
        self.n_dim = len(self.dof_shape)
        self.rhs_model = rhs_model
        self.matvec_model = matvec_model
        if prec_model is None:
            self.prec_model = torch.nn.Identity()
        else:
            self.prec_model = prec_model
        self.rtol = rtol
        self.atol = atol
        self.n_layers = n_layers
        self.alg_type = alg_type
        self.shape = (self.n_channels, *self.dof_shape)
        self.dims = self.n_dim * (slice(None),)
        self.dims_first = self.n_dim * (slice(1),)
        self.dims_list = list(range(-self.n_dim, 0))
        self.ch_dim = -(self.n_dim + 1)
        self.err_eps = err_eps

        self.init_layers = torch.nn.ModuleList([rhs_model])

        iteration_layer = IterationLayer(
            n_channels=self.n_channels,
            dof_shape=self.dof_shape,
            matvec_model=self.matvec_model,
            prec_model=self.prec_model,
            alg_type=self.alg_type,
        )
        self.iteration_layers = torch.nn.ModuleList([iteration_layer])

    def zero_guess(self, param_fields, loadings):
        batch_shape = (*param_fields.shape[:(-self.n_dim - 1)], loadings.shape[0])
        field_shape = (*batch_shape, *self.shape)
        u = torch.zeros_like(param_fields[..., :1, *self.dims_first].unsqueeze(0).expand(field_shape))
        return u

    def forward(self, u: torch.Tensor, param_fields: torch.Tensor, loadings: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        :param param_fields: parameter fields
        :param loadings: loadings
        :return: solution fields
        """
        if self.n_layers is None:
            return self.forward_dynamic(u, param_fields, loadings)
        else:
            return self.forward_fixed(u, param_fields, loadings)

    def forward_fixed(self, u: torch.Tensor, param_fields: torch.Tensor, loadings: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        :param param_fields: parameter fields
        :param loadings: loadings
        :return: solution fields
        """
        batch_shape = (*param_fields.shape[:(-self.n_dim - 1)], loadings.shape[0])
        field_shape = (*batch_shape, *self.shape)

        r = self.init_layers[0](u=u, param_fields=param_fields, loadings=loadings)
        iv_scalar, iv_fields = self.iteration_layers[0].init_internal_variables(batch_shape=batch_shape, init_residual=r)

        for _ in range(self.n_layers):
            u, r, iv_scalar, iv_fields = self.iteration_layers[0](u=u, r=r, param_fields=param_fields, loadings=loadings, iv_scalar=iv_scalar, iv_fields=iv_fields)

        return u

    def forward_dynamic(self, u: torch.Tensor, param_fields: torch.Tensor, loadings: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        :param param_fields: parameter fields
        :param loadings: loadings
        :return: solution fields
        """
        batch_shape = (*param_fields.shape[:(-self.n_dim - 1)], loadings.shape[0])
        field_shape = (*batch_shape, *self.shape)

        r = self.init_layers[0](u=u, param_fields=param_fields, loadings=loadings)
        err = torch.norm(r, torch.inf)
        err0 = err.clone()
        err_rel = torch.ones_like(err0)
        iv_scalar, iv_fields = self.iteration_layers[0].init_internal_variables(batch_shape=batch_shape, init_residual=r)

        while err_rel > self.rtol:
            u, r, iv_scalar, iv_fields = self.iteration_layers[0](u=u, r=r, param_fields=param_fields, loadings=loadings, iv_scalar=iv_scalar, iv_fields=iv_fields)
            err = torch.norm(r, torch.inf)
            err_rel = err / (err0 + self.err_eps)
        return u


class CgFieldModule(CgModule):
    def __init__(
        self,
        n_channels: int,
        dof_shape,
        rhs_model: torch.nn.Module,
        matvec_model: torch.nn.Module,
        field_module: torch.nn.Module,
        prec_model: Optional[torch.nn.Module] = None,
        rtol: float = 1e-6,
        n_layers = None,
        alg_type: str = "cg",
    ):
        super().__init__(
            n_channels=n_channels,
            dof_shape=dof_shape,
            rhs_model=rhs_model,
            matvec_model=matvec_model,
            prec_model=prec_model,
            rtol=rtol,
            n_layers=n_layers,
            alg_type=alg_type,
        )
        self.field_layers = torch.nn.ModuleList([field_module])

    def forward(self, u: torch.Tensor, param_fields: torch.Tensor, loadings: torch.Tensor) -> torch.Tensor:
        x = super().forward(u, param_fields, loadings)
        field = self.field_layers[0](x, param_fields, loadings)
        return field


class IterationLayer(torch.nn.Module):
    def __init__(
        self,
        n_channels,
        dof_shape,
        matvec_model: torch.nn.Module,
        prec_model: Union[torch.nn.Module],
        alg_type: str = "cg",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dof_shape = dof_shape
        self.alg_type = alg_type
        
        self.prec_layers = torch.nn.ModuleList([prec_model])

        if self.alg_type == "fp":
            update_layer = FpUpdateLayer(self.n_channels, self.dof_shape)
        elif self.alg_type == "cg":
            update_layer = CgUpdateLayer(self.n_channels, self.dof_shape, matvec_model)
        elif self.alg_type == "cgpr":
            update_layer = CgPrUpdateLayer(self.n_channels, self.dof_shape, matvec_model)
        else:
            raise ValueError("Unknown alg_type")
        self.update_layers = torch.nn.ModuleList([update_layer])

    def forward(self, u: torch.Tensor, r: torch.Tensor, param_fields: torch.Tensor, loadings: torch.Tensor, iv_scalar, iv_fields):
        s = self.prec_layers[0](r)
        u, r, iv_scalar, iv_fields = self.update_layers[0](
            u=u, s=s, param_fields=param_fields, loadings=loadings, iv_scalar=iv_scalar, iv_fields=iv_fields
        )
        return u, r, iv_scalar, iv_fields

    def init_internal_variables(self, batch_shape=None, init_residual=None):
        return self.update_layers[0].init_internal_variables(batch_shape, init_residual)

class FpUpdateLayer(torch.nn.Module):
    def __init__(self, dof_shape, residual_model: torch.nn.Module):
        super().__init__()
        self.dof_shape = dof_shape
        self.residual_layers = torch.nn.ModuleList([residual_model])

    def forward(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        param_fields: torch.Tensor,
        loadings: torch.Tensor,
        internal_variables: Optional[torch.Tensor] = None,
    ):
        new_u = u + s
        new_r = self.residual_layers[0](new_u, param_fields)
        return new_u, new_r, None, None

    def init_internal_variables(self, batch_shape=None, init_field=None):
        return None, None


class CgUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        n_channels,
        dof_shape,
        matvec_model: torch.nn.Module,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dof_shape = dof_shape
        self.n_dim = len(self.dof_shape)
        self.dims = (slice(None),) * self.n_dim
        self.expand_dims = (None,) * self.n_dim
        self.remove_dims = (0,) * (1 + self.n_dim)
        self.matvec_layer = torch.nn.ModuleList([matvec_model])
        self.dot_product_layer = DotProductLayer(self.n_dim)
        self.axpy_layer = AxpyLayer(self.n_dim)

    def forward(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        param_fields: torch.Tensor,
        loadings: torch.Tensor,
        iv_scalar: Optional[torch.Tensor] = None,
        iv_fields: Optional[torch.Tensor] = None,
    ):
        delta = iv_scalar[..., 0]
        r = iv_fields[..., 0:self.n_channels, *self.dims]
        d = iv_fields[..., self.n_channels:(2*self.n_channels), *self.dims]
        new_delta = self.dot_product_layer(r, s)
        beta = torch.nan_to_num(new_delta / delta, nan=0.0, posinf=0.0)
        new_d = self.axpy_layer(beta, d, s)
        new_p = self.matvec_layer[0](new_d, param_fields)
        alpha = torch.nan_to_num(new_delta / self.dot_product_layer(new_d, new_p), nan=0.0, posinf=0.0)
        new_r = self.axpy_layer(-alpha, new_p, r)
        new_u = self.axpy_layer(alpha, new_d, u)
        new_iv_scalar = torch.stack([new_delta], dim=-1)
        new_iv_fields = torch.cat([new_r, new_d], dim=-(self.n_dim + 1))
        return new_u, new_r, new_iv_scalar, new_iv_fields

    def init_internal_variables(self, batch_shape, init_r):
        delta = torch.ones_like(init_r[..., *self.remove_dims])
        d = torch.zeros_like(init_r)
        iv_scalar = torch.stack([delta], dim=-1)
        iv_fields = torch.cat([init_r, d], dim=-(self.n_dim + 1))
        return iv_scalar, iv_fields


class CgPrUpdateLayer(torch.nn.Module):
    def __init__(
        self,
        n_channels,
        dof_shape,
        matvec_model: torch.nn.Module,
    ):
        super().__init__()
        self.dot_product_layer = DotProductLayer()
        self.axpy_layer = AxpyLayer()
        self.matvec_layer = torch.nn.ModuleList([matvec_model])

    def forward(self, u, s, param_fields, iv_scalar, iv_fields):
        delta = iv_scalar[..., 0]
        r = iv_fields[..., 0:self.n_channels, *self.dims]
        d = iv_fields[..., self.n_channels:(2*self.n_channels), *self.dims]
        old_s = iv_fields[..., (2*self.n_channels):(3*self.n_channels), *self.dims]
        new_delta = self.dot_product_layer(r, s)
        beta = torch.maximum(
            (new_delta - self.dot_product_layer(r, old_s)) / delta,
            torch.zeros_like(delta),
        )
        new_d = self.axpy_layer(beta, d, s)
        new_p = self.matvec_model(new_d, param_fields)
        alpha = -new_delta / self.dot_product_layer(new_d, new_p)
        new_u = self.axpy_layer(alpha, new_d, u)
        new_r = self.axpy_layer(alpha, new_p, r)
        new_iv_scalar = torch.stack([delta], dim=-1)
        new_iv_fields = torch.cat([new_r, new_d, s], dim=-(self.n_dim + 1))
        return new_u, new_r, new_iv_scalar, new_iv_fields

    def init_internal_variables(self, field_shape, init_r):
        delta = torch.ones_like(init_r[..., *self.remove_dims])
        d = torch.zeros_like(init_r)
        old_s = torch.zeros_like(init_r)
        iv_scalar = torch.stack([delta], dim=-1)
        iv_fields = torch.cat([init_r, d, old_s], dim=-(self.n_dim + 1))
        return iv_scalar, iv_fields


class DotProductLayer(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.dims_list = tuple(range(-1, -(2 + self.n_dim), -1))

    def forward(self, a, b):
        field_c = (a * b).sum(self.dims_list)
        return field_c


class AxpyLayer(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.slices = (Ellipsis, None) + (None,) * self.n_dim

    def forward(self, alpha, x, y):
        return alpha[*self.slices] * x + y
