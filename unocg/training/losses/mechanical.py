import torch
from .base import Loss


class DispLoss(Loss):
    """
    Displacement loss function
    """

    def __init__(self, n_dim=2, reduction="sum"):
        super().__init__(n_dim=n_dim, reduction=reduction)

    def abs(self, output, target):
        """
        Compute absolute error for temperature field

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_disp, target_disp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        loss = torch.sqrt(torch.nn.MSELoss(reduction="none")(target_disp, output_disp).nanmean(self.dims_list))
        loss = self.reduce(loss).squeeze(-1)
        return loss

    def rel(self, output, target):
        """
        Compute relative error for temperature field

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_disp, target_disp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        loss = torch.linalg.norm(target_disp - output_disp, dim=-(1 + self.n_dim)).nanmean(self.dims_list) \
            / torch.linalg.norm(target_disp, dim=-(1 + self.n_dim)).nanmean(self.dims_list)
        loss = self.reduce(loss).squeeze(-1)
        return loss

    def __call__(self, x, y):
        return self.rel(x, y)

    def __str__(self):
        return f"Temp({self.reduction})"


class StrainLoss(Loss):
    """
    Strain loss function
    """

    def __init__(self, grad_module, n_dim=2, reduction="sum"):
        super().__init__(n_dim=n_dim, reduction=reduction)
        self.grad_module = grad_module

    def abs(self, output, target):
        """
        Compute absolute error for temperature field

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_temp, target_temp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        loss = torch.linalg.norm(target_temp - output_temp, dim=self.ch_dim)
        loss = self.reduce(loss)
        return loss

    def rel(self, output, target):
        """
        Compute relative error for temperature field

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_temp, target_temp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        batch_dims = output_temp.shape[:self.ch_dim]
        output_temp = torch.flatten(output_temp, start_dim=0, end_dim=self.ch_dim - 1)
        target_temp = torch.flatten(target_temp, start_dim=0, end_dim=self.ch_dim - 1)
        output_grad = self.grad_module(output_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        target_grad = self.grad_module(target_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        output_temp = torch.unflatten(output_temp, dim=0, sizes=batch_dims)
        target_temp = torch.unflatten(target_temp, dim=0, sizes=batch_dims)
        loss = torch.linalg.norm(target_grad - output_grad, dim=-(1 + self.n_dim)).nanmean(self.dims_list)
        loss = self.reduce(loss)
        return loss

    def __call__(self, x, y):
        return self.rel(x, y)

    def __str__(self):
        return f"Grad({self.reduction})"


class StressLoss(Loss):
    """
    Flux loss function
    """

    def __init__(self, n_dim=2, reduction="sum"):
        super().__init__(n_dim=n_dim, reduction=reduction)

    def abs(self, output, target):
        """
        Compute absolute error in strain norm

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_flux, target_flux = output[..., self.n_dim:, *self.dims], target[..., self.n_dim:, *self.dims]
        loss = torch.linalg.norm(target_flux - output_flux, dim=-(1 + self.n_dim)).nanmean(self.dims_list)
        loss = self.reduce(loss)
        return loss

    def rel(self, output, target):
        """
        Compute absolute error in strain norm

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_flux, target_flux = output[..., self.n_dim:, *self.dims], target[..., self.n_dim:, *self.dims]
        loss = torch.linalg.norm(target_flux - output_flux, dim=-(1 + self.n_dim)).nanmean(self.dims_list) \
            / torch.linalg.norm(target_flux, dim=-(1 + self.n_dim)).nanmean(self.dims_list)
        loss = self.reduce(loss)
        return loss

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        return self.rel(output, target)

    def __str__(self):
        return f"Flux({self.reduction})"
    
    
class MechanicalEnergyLoss(Loss):
    """
    Thermal energy loss function
    """

    def __init__(self, grad_module, n_dim=2, reduction="sum"):
        super().__init__(n_dim=n_dim, reduction=reduction)
        self.grad_module = grad_module

    def abs(self, output, target):
        """
        Compute absolute error in strain norm

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_temp, target_temp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        output_flux, target_flux = output[..., self.n_dim:, *self.dims], target[..., self.n_dim:, *self.dims]
        batch_dims = output_temp.shape[:self.ch_dim]
        output_temp = torch.flatten(output_temp, start_dim=0, end_dim=self.ch_dim - 1)
        target_temp = torch.flatten(target_temp, start_dim=0, end_dim=self.ch_dim - 1)
        output_grad = self.grad_module(output_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        target_grad = self.grad_module(target_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        output_temp = torch.unflatten(output_temp, dim=0, sizes=batch_dims)
        target_temp = torch.unflatten(target_temp, dim=0, sizes=batch_dims)
        loss = torch.sqrt(
            torch.linalg.norm((output_grad - target_grad) * (output_flux - target_flux), dim=self.ch_dim).nanmean(self.dims_list)
        )
        loss = self.reduce(loss)
        return loss

    def rel(self, output, target):
        """
        Compute absolute error in strain norm

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_temp, target_temp = output[..., :self.n_dim, *self.dims], target[..., :self.n_dim, *self.dims]
        output_flux, target_flux = output[..., self.n_dim:, *self.dims], target[..., self.n_dim:, *self.dims]
        batch_dims = output_temp.shape[:self.ch_dim]
        output_temp = torch.flatten(output_temp, start_dim=0, end_dim=self.ch_dim - 1)
        target_temp = torch.flatten(target_temp, start_dim=0, end_dim=self.ch_dim - 1)
        output_grad = self.grad_module(output_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        target_grad = self.grad_module(target_temp).nanmean(-(2 + self.n_dim))  # average over gauss points
        output_temp = torch.unflatten(output_temp, dim=0, sizes=batch_dims)
        target_temp = torch.unflatten(target_temp, dim=0, sizes=batch_dims)
        loss = torch.sqrt(
            torch.linalg.norm((output_grad - target_grad) * (output_flux - target_flux), dim=self.ch_dim).nanmean(self.dims_list)
            / torch.linalg.norm(target_grad * target_flux, dim=self.ch_dim).nanmean(self.dims_list)
        )
        loss = self.reduce(loss)
        return loss

    def __call__(self, output_disp: torch.Tensor, target_disp: torch.Tensor, param_field=None):
        return self.abs(output_disp, target_disp)

    def __str__(self):
        return f"ThermEnergy({self.reduction})"


class LinearElasticResidualLoss(Loss):
    """
    Homogenized stress loss function
    """

    def __init__(self, div_module, n_dim=2, reduction="sum", residual_mode="mean"):
        super().__init__(n_dim=n_dim, reduction=reduction)
        self.div_module = div_module
        self.residual_mode = residual_mode

    def abs(self, output, target=None):
        """
        Compute absolute error in strain norm

        :param output:
        :param target:
        :return:
        """
        output, target = self.unsqueeze(output, target)
        output_flux, target_flux = output[..., 1:, *self.dims], target[..., 1:, *self.dims]
        batch_dims = target_flux.shape[:self.ch_dim]
        output_flux = torch.flatten(output_flux, start_dim=0, end_dim=self.ch_dim - 1).unsqueeze(self.ch_dim - 1)
        output_residual = self.div_module(output_flux)
        output_residual = torch.unflatten(output_residual, dim=0, sizes=batch_dims)
        if self.residual_mode == "sum":
            loss = torch.abs(output_residual).nansum(self.dims_list)
        elif self.residual_mode == "mean":
            loss = torch.abs(output_residual).nanmean(self.dims_list)
        else:
            raise ValueError("Unknown residual_mode")
        loss = self.reduce(loss).squeeze(-1)
        return loss

    def __call__(self, output: torch.Tensor, target):
        return self.abs(output, target)

    def __str__(self):
        return f"HeatCondResidual({self.reduction})"
