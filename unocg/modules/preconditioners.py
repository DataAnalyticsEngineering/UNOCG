import torch


class FFTRApplyModule(torch.nn.Module):
    def __init__(self, fft_dims, weights, field_shape, fftn, ifftn, n_loadings=None):
        super().__init__()
        self.fft_dims = fft_dims
        self.n_dim = len(self.fft_dims)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(weights, requires_grad=False)])
        self.field_shape = field_shape
        self.fftn = fftn
        self.ifftn = ifftn
        self.flatten_batch = torch.nn.Flatten(start_dim=0, end_dim=-(self.n_dim + 2))
        if self.n_dim == 2:
            self.einsum_dims = "ijyx,...jyx->...iyx"
        elif self.n_dim == 3:
            self.einsum_dims = "ijzyx,...jzyx->...izyx"
        else:
            raise ValueError()

    def forward(self, x):
        batch_dims = x.shape[:-(self.n_dim + 1)]
        x = self.flatten_batch(x)
        x = torch.view_as_real(self.fftn(x, dim=self.fft_dims))
        x = torch.einsum("ijyx,bjyxc->biyxc", self.weights[0].to(dtype=x.dtype), x)
        x = self.ifftn(torch.view_as_complex(x), dim=self.fft_dims).real
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x


class FFTApplyModule(torch.nn.Module):
    def __init__(self, fft_dims, weights, field_shape, fftn, ifftn):
        super().__init__()
        self.fft_dims = fft_dims
        self.n_dim = len(self.fft_dims)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(weights, requires_grad=False)])
        self.field_shape = field_shape
        self.fftn = fftn
        self.ifftn = ifftn
        self.flatten_batch = torch.nn.Flatten(start_dim=0, end_dim=-(self.n_dim + 2))
        if self.n_dim == 2:
            self.einsum_dims = "ijyx,...jyx->...iyx"
        elif self.n_dim == 3:
            self.einsum_dims = "ijzyx,...jzyx->...izyx"
        else:
            raise ValueError()

    def forward(self, x):
        batch_dims = x.shape[:-(self.n_dim + 1)]
        x = self.flatten_batch(x)
        x = self.fftn(x, dim=self.fft_dims)
        x = torch.einsum(self.einsum_dims, self.weights[0].to(dtype=x.dtype), x)
        x = self.ifftn(x, dim=self.fft_dims).real
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x


class JacApplyModule(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(weights, requires_grad=False)])

    def forward(self, x):
        return self.weights[0] * x


class UnoApplyModule(torch.nn.Module):
    def __init__(self, problem, transform, weights):
        super().__init__()
        self.problem = problem
        self.transform = transform
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(weights, requires_grad=False)])
        self.field_shape = (-1, self.problem.n_channels, *self.problem.dof_shape)
        self.transform = self.transform
        self.n_dim = self.problem.n_dim
        self.flatten_batch = torch.nn.Flatten(start_dim=0, end_dim=-(self.n_dim + 2))
        if weights.shape[0] == weights.shape[1]:
            if self.n_dim == 2:
                self.einsum_dims = "ijyx,...jyx->...iyx"
            elif self.n_dim == 3:
                self.einsum_dims = "ijzyx,...jzyx->...izyx"
            else:
                raise ValueError()
        else:
            if self.n_dim == 2:
                self.einsum_dims = "kiyx,...iyx->...iyx"
            elif self.n_dim == 3:
                self.einsum_dims = "kizyx,...izyx->...izyx"
            else:
                raise ValueError()

    def forward(self, x):
        batch_dims = x.shape[:-(self.n_dim + 1)]
        x = self.flatten_batch(x)
        x = self.transform.transform(x)
        x = torch.einsum(self.einsum_dims, self.weights[0].to(device=x.device, dtype=x.dtype), x)
        x = self.transform.inverse_transform(x).real
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x

    def get_weights(self):
        return self.weights[0]
