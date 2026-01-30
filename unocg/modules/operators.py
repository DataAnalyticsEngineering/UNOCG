import torch


class MatvecModule(torch.nn.Module):
    def __init__(self, n_dim, grad_module, model, div_module):
        super().__init__()
        self.n_dim = n_dim
        self.ch_dim = -(1 + self.n_dim)
        self.module_list = torch.nn.ModuleList([
            grad_module,
            model,
            div_module
        ])

    def forward(self, x, param_fields):
        batch_dims = x.shape[:self.ch_dim]
        x = torch.flatten(x, start_dim=0, end_dim=self.ch_dim - 1)
        x = self.module_list[0](x)  # grad
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        x = self.module_list[1](x, param_fields)[0]  # material law
        x = torch.flatten(x, start_dim=0, end_dim=self.ch_dim - 2)
        x = self.module_list[2](x)  # div
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x


class ResidualModule(torch.nn.Module):
    def __init__(self, n_dim, grad_module, model, div_module):
        super().__init__()
        self.n_dim = n_dim
        self.ch_dim = -(1 + self.n_dim)
        self.load_slice = (slice(None), None, slice(None)) + self.n_dim * (None,)
        self.module_list = torch.nn.ModuleList([
            grad_module,
            model,
            div_module
        ])

    def forward(self, x, param_fields, loadings):
        batch_dims = x.shape[:self.ch_dim]
        x = torch.flatten(x, start_dim=0, end_dim=self.ch_dim - 1)
        x = self.module_list[0](x)
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        x = x + loadings[self.load_slice].unsqueeze(0).expand(x.shape)
        x = self.module_list[1](x, param_fields)[0]
        x = torch.flatten(x, start_dim=0, end_dim=self.ch_dim - 2)
        x = -self.module_list[2](x)
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x


class RhsModule(torch.nn.Module):
    def __init__(self, n_dim, shape, model, div_module):
        super().__init__()
        self.n_dim = n_dim
        self.load_slice = (slice(None), None, slice(None)) + self.n_dim * (None,)
        self.shape = shape
        self.module_list = torch.nn.ModuleList([model, div_module])

    def forward(self, u, param_fields, loadings):
        batch_dims = (*param_fields.shape[:-(1 + self.n_dim)], loadings.shape[0])
        x = loadings[self.load_slice].expand(batch_dims + self.shape)
        x = self.module_list[0](x, param_fields)[0]
        x = torch.flatten(x, start_dim=0, end_dim=-(self.n_dim + 3))
        x = -self.module_list[1](x)
        x = torch.unflatten(x, dim=0, sizes=batch_dims)
        return x


class FieldModule(torch.nn.Module):
    def __init__(self, ch_dim, expand_dims, grad_module, model, padding_layers=[]):
        super().__init__()
        self.ch_dim = ch_dim
        self.expand_dims = expand_dims
        self.padding_layers = padding_layers
        self.module_list = torch.nn.ModuleList([grad_module, model])

    def forward(self, x, param_fields, loadings):
        batch_dims = x.shape[:self.ch_dim]
        y = torch.flatten(x, start_dim=0, end_dim=self.ch_dim - 1)
        y = self.module_list[0](y)  # gradient computation
        #print("after gradient", y.shape)
        y = torch.unflatten(y, dim=0, sizes=batch_dims)
        y = y + loadings[(slice(None), None, slice(None)) + self.expand_dims]  # construct strains
        y = self.module_list[1](y, param_fields)[0]  # material law
        y = y.mean(dim=self.ch_dim - 1)  # average over gauss points
        for padding_layer in self.padding_layers:
            x = padding_layer(x)
        return torch.cat([x, y], dim=self.ch_dim)
