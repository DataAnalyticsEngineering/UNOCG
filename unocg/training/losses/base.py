"""
Definition of loss function that allow for a physics-informed training of machine learning models
"""
from abc import ABC
from typing import Union, Iterable, List, Optional

# third-party packages
import torch


class Loss(ABC):
    """
    Abstract loss function
    """

    def __init__(self, n_dim: int = 2, reduction: str = "sum"):
        """

        :param n_dim: dimension of the problem
        :param reduction: ('mean'|'sum'|'none'), defaults to 'sum'
        :type reduction: str, optional
        """
        super().__init__()
        self.n_dim = n_dim
        self.ch_dim = -(1 + self.n_dim)
        self.reduction = reduction

    def reduce(self, loss):
        """
        Perform a reduction step over all datasets to transform a loss function to a cost function.

        A loss function is evaluated element-wise for a dataset.
        However, a cost function should return a single value for the dataset.
        Typically, `mean` reduction is used.

        :param loss: Tensor that contains the element-wise loss for a dataset
        :type loss: :class:`torch.Tensor`
        :return: Reduced loss
        :rtype: float
        """
        if self.reduction == "mean":
            return torch.nanmean(loss)
        elif self.reduction == "sum":
            return torch.nansum(loss)
        else:
            return loss

    def unsqueeze(self, output, target):
        """
        Ensure that the tensors :code:`output` and :code:`target` have a shape of the form :code:`(N, features)`.

        When a loss function is called with a single data point, the tensor shape is :code:`(features)` and hence does not fit.
        This method expands the dimensions if needed.

        :param output: Model output
        :type output: :class:`torch.Tensor`
        :param target: Target data
        :type target: :class:`torch.Tensor`
        :return: Tuple (output, target)
        :rtype: tuple
        """
        while output.ndim < self.n_dim + 2:
            output = torch.unsqueeze(output, 0)
        while target.ndim < output.ndim:
            target = torch.unsqueeze(target, 0)
        while output.ndim < target.ndim:
            output = torch.unsqueeze(output, 0)
        assert output.ndim >= self.n_dim + 2 and output.ndim == target.ndim
        return output, target
        # return output.flatten(end_dim=-(self.n_dim + 2)), target.flatten(end_dim=-(self.n_dim + 2))

    @property
    def dims(self):
        return self.n_dim * (slice(None),)

    @property
    def dims_list(self):
        return tuple(range(-1, -(1 + self.n_dim), -1))

    @property
    def expand_dims(self):
        return self.n_dim * (None,)


class WeightedLoss(Loss):
    """
    Weighted loss function that represents a linear combination of several loss functions
    """

    def __init__(self, losses: Iterable[any], weights: Iterable[Union[float, int]], reduction="mean"):
        super().__init__(reduction=reduction)
        self.losses = losses
        self.weights = weights

    def __call__(self, output, target):
        total_loss = 0.0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(output, target)
        return total_loss

    def __str__(self):
        return f"Weighted(...)"
