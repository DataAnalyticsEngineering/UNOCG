import torch
from typing import Optional, Union, Tuple, Sequence, Collection, Dict
from unocg.materials import Material


class LinearHeatConduction(Material):
    def __init__(self, n_dim, device, dtype):
        super().__init__(n_dim, device, dtype)

    def get_stress(self, temp_grad, param_fields, *args, **kwargs):
        # Isotropic Fourier law: q = -kappa * g
        cond_field = param_fields[(Ellipsis, 0) + self.dims].unsqueeze(self.ch_dim)
        flux = -cond_field[(Ellipsis, None, None, slice(None)) + self.dims] * temp_grad
        return flux, None

    def get_tangent(self, temp_grad, param_fields, *args, **kwargs):
        Id = torch.eye(self.n_dim, self.n_dim, **self.args)
        
        expand_scalar = (Ellipsis, None, None) + self.n_dim * (slice(None),)
        expand_tensor = (Ellipsis,) + self.n_dim * (None,)

        cond_field = param_fields[(Ellipsis, 0) + self.dims].unsqueeze(self.ch_dim)

        tangent = -cond_field[*expand_scalar] * Id[*expand_tensor]
        return torch.repeat_interleave(tangent, repeats=strain.shape[self.ch_dim - 1], dim=self.ch_dim - 2)

    def get_tangent_vp(self, strain0, strain, param_fields, *args, **kwargs):
        # Special case for linear heat conduction:
        return self.get_stress(strain, param_fields, *args, **kwargs)[0]

    def get_potential(self, strain, param_fields):
        raise NotImplementedError
