import torch


class Material(torch.nn.Module):
    def __init__(self, n_dim, device, dtype, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.strain_dims = int((self.n_dim + 1) * self.n_dim / 2.)
        self.device = device
        self.dtype = dtype
        self.args = {"dtype": self.dtype, "device": self.device}
        self.ch_dim = -(self.n_dim + 1)
        self.dims = self.n_dim * (slice(None),)
        self.dims_list = tuple(range(-self.n_dim, 0))
        if self.n_dim == 1:
            self.einsum_dims = "x"
        elif self.n_dim == 2:
            self.einsum_dims = "yx"
        elif self.n_dim == 3:
            self.einsum_dims = "zyx"
        else:
            raise NotImplementedError()

    def forward(self, strain, param_fields, *args, **kwargs):
        return self.get_stress(strain, param_fields, *args, **kwargs)

    def get_stress(self, strain, param_fields, *args, **kwargs):
        raise NotImplementedError()

    def get_tangent(self, strain0, param_fields, *args, **kwargs):
        raise NotImplementedError()

    def get_tangent_vp(self, strain0, strain, param_fields, *args, **kwargs):
        tangent = self.get_tangent(strain0, param_fields, *args, **kwargs)
        return torch.einsum(
            "...ij" + self.einsum_dims + ",...j" + self.einsum_dims + "->...i" + self.einsum_dims,
            tangent, strain)
