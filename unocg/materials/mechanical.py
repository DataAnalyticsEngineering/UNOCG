import torch
from typing import Optional, Union, Tuple, Sequence, Collection, Dict
from unocg.materials import Material


class LinearElasticity(Material):
    def __init__(self, n_dim, device, dtype):
        super().__init__(n_dim, device, dtype)

    def get_stress(self, strain, param_fields, *args, **kwargs):
        lame_lambda = param_fields.narrow(self.ch_dim, 0, 1).unsqueeze(self.ch_dim-1).unsqueeze(self.ch_dim-1)
        lame_mu = param_fields.narrow(self.ch_dim, 1, 1).unsqueeze(self.ch_dim-1).unsqueeze(self.ch_dim-1)

        strain_tr = strain.narrow(self.ch_dim, 0, self.n_dim).sum(dim=self.ch_dim).unsqueeze(self.ch_dim)

        stress = 2. * lame_mu * strain
        stress_diag = stress.narrow(self.ch_dim, 0, self.n_dim)
        stress_diag += lame_lambda * strain_tr
        return stress, None

    def get_tangent(self, strain, param_fields, *args, **kwargs):
        Id = torch.eye(self.strain_dims, self.strain_dims, **self.args)
        if self.n_dim == 2:
            P = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]], **self.args)
        elif self.n_dim == 3:
            P = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], **self.args)
        expand_scalar = (Ellipsis, None, None) + self.n_dim * (slice(None),)
        expand_tensor = (Ellipsis,) + self.n_dim * (None,)

        lame_lambda = param_fields[(Ellipsis, 0) + self.dims].unsqueeze(self.ch_dim)
        lame_mu = param_fields[(Ellipsis, 1) + self.dims].unsqueeze(self.ch_dim)

        tangent = 2. * lame_mu[*expand_scalar] * Id[*expand_tensor] \
            + lame_lambda[*expand_scalar] * P[*expand_tensor]
        return torch.repeat_interleave(tangent, repeats=strain.shape[self.ch_dim - 1], dim=self.ch_dim - 2)

    def get_tangent_vp(self, strain0, strain, param_fields, *args, **kwargs):
        # Special case for linear elasticity:
        return self.get_stress(strain, param_fields, *args, **kwargs)[0]

    def get_potential(self, strain, param_fields):
        raise NotImplementedError
