"""
Implementation of various iterative solvers in PyTorch

- CgSolver: Linear conjugate gradient solver
"""
from typing import Optional, Callable, Dict, Tuple

import torch

from unocg.problems import Problem
from unocg.solvers import Solver
from unocg.preconditioners import Preconditioner
from unocg.modules.solvers import CgFieldModule, CgModule


class CgSolver(Solver):
    """
    Linear conjugate gradient (CG) Solver
    """

    def __init__(
            self,
            problem: Problem,
            preconditioner: Optional[Preconditioner] = None,
            rtol: float = 1e-10,
            atol: float = 0.0,
            max_iter: int = 100000,
            verbose: int = 0,
            cg_type: str = "cg",
            log_residual: bool = False,
            log_iterates: bool = False,
            loss_callback: Optional[Callable[[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ):
        """

        :param problem:
        :param preconditioner:
        :param rtol:
        :param atol:
        :param max_iter:
        :param verbose:
        :param cg_type:
        :param log_residual:
        :param log_iterates:
        :param loss_callback:
        """
        super().__init__(problem, preconditioner, rtol, atol, max_iter, verbose)
        self.cg_type = cg_type
        self.log_residual = log_residual
        self.log_iterates = log_iterates
        self.loss_callback = loss_callback

    def solve(
            self,
            param_fields: torch.Tensor,
            loadings: Optional[torch.Tensor] = None,
            state_fields: Optional[torch.Tensor] = None,
            guess: Optional[torch.Tensor] = None,
            rtol: Optional[float] = None,
            atol: Optional[float] = None,
            max_iter: Optional[int] = None,
            verbose: Optional[int] = None,
            cg_type: Optional[str] = None,
            log_residual: Optional[bool] = None,
            log_iterates: Optional[bool] = None,
            loss_callback: Optional[Callable[[torch.Tensor], Dict[str, torch.Tensor]]] = None,
            zero_mean: bool = False,
            err_eps: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        """
        Solve the problem for given parameter fields, loadings and state/history fields.

        :param param_fields: parameter fields
        :param loadings: loadings
        :param state_fields: initial state/history variables fields
        :param guess: initial guess for DOFs
        :param rtol: relative tolerance for residual norm
        :param atol: absolute tolerance for residual norm
        :param max_iter: maximum number of iterations
        :param cg_type: "cg" or "cgpr". More options will be implemented soon.
        :param log_residual: if residuals should be stored
        :param log_iterates: if iterates should be stored
        :param loss_callback: callback function for loss computation in every iteration
        :param zero_mean: if zero-mean property of iterates should be enforced
        :return: result data in form of a dictionary with keys
            "sol": DOF vectors u,
            "err_history": residual norm history,
            "n_iter": number of iterations,
            "state_fields": state/history variables after convergence
        """
        args = {"device": param_fields.device, "dtype": param_fields.dtype}

        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        if max_iter is None:
            max_iter = self.max_iter
        if verbose is None:
            verbose = self.verbose
        if cg_type is None:
            cg_type = self.cg_type
        if log_residual is None:
            log_residual = self.log_residual
        if log_iterates is None:
            log_iterates = self.log_iterates
        if loss_callback is None:
            loss_callback = self.loss_callback

        if loadings is None:
            loadings = self.problem.loadings.to(**args)
        else:
            loadings = loadings.to(**args)
        n_loadings = loadings.shape[0]

        vec_shape = self.problem.get_vec_shape(param_fields, loadings=loadings)

        # Initial guess for temperature
        if guess is None:
            u = torch.zeros(vec_shape, **args)
        else:
            u = guess.reshape(vec_shape).to(**args)

        # Compute initial residual
        r, _ = self.problem.compute_residual(u, param_fields, loadings=loadings, state_fields=state_fields)

        # Norm of initial residual for convergence control
        err = torch.norm(r, torch.inf, dim=-1)
        err0 = err.clone()
        err_rel = torch.ones_like(err0)
        err_history = torch.unsqueeze(err0, dim=0)

        # Log metrics
        if log_residual:
            res_history = torch.unsqueeze(r.clone(), dim=0)
        if log_iterates:
            u_history = torch.unsqueeze(u.clone(), dim=0)
        if loss_callback is not None:
            losses = loss_callback(u)

        n_iter = 0
        d = torch.zeros_like(u)
        s = torch.zeros_like(u)
        delta = torch.ones_like(err0)
        if cg_type == "cgpr":
            s_old = torch.empty_like(s)

        while (err_rel.max() > rtol) and (err.max() > atol) and (n_iter < max_iter):
            # Determine "not converged" flags
            nc_all = torch.logical_and(err_rel > rtol, err > atol)
            if not nc_all.max():
                break  # everything is converged
            nc_load = nc_all.reshape(-1, n_loadings).max(dim=0)[0]
            nc = (Ellipsis, nc_load)
            ncd = nc + (slice(None),)

            if cg_type == "cgpr":
                s_old.copy_(s)

            # Apply preconditioner
            s[ncd] = self.preconditioner.apply(r[ncd]).reshape(r[ncd].shape)

            delta0 = delta.clone()
            delta = (r * s).sum(dim=-1)
            if cg_type == "cgpr":
                beta = torch.maximum((delta[nc] - (r[ncd] * s_old[ncd]).sum(dim=-1)) / delta0[nc], torch.zeros_like(err0[nc]))
            else:
                beta = delta[nc] / delta0[nc]

            # Update search direction
            d[ncd] = s[ncd] + beta[..., None] * d[ncd]

            # Compute matrix-vector product: p = A @ d
            p, _ = self.problem.matvec(d[ncd], param_fields=param_fields, state_fields=state_fields)
            alpha = delta[nc] / (d[ncd] * p).sum(dim=-1)

            # Update residual
            r[ncd] = r[ncd] - alpha[..., None] * p
            #r[ncd].sub_(alpha[..., None] * p)

            # Update solution approximation
            u[ncd] = u[ncd] + alpha[..., None] * d[ncd]
            #u[ncd].add_(alpha[..., None] * d[ncd])

            if zero_mean:
                u = self.problem.zero_mean(u)

            # Logging
            err = torch.norm(r, torch.inf, dim=-1)
            err_history = torch.cat([err_history, torch.unsqueeze(err, dim=0)], dim=0)
            if log_residual:
                res_history = torch.cat([res_history, torch.unsqueeze(r, dim=0)], dim=0)
            if log_iterates:
                u_history = torch.cat([u_history, torch.unsqueeze(u, dim=0)], dim=0)
            if loss_callback is not None:
                losses_i = loss_callback(u)
                losses = {loss_key: torch.cat([losses[loss_key], losses_]) for (loss_key, losses_) in losses_i.items()}
            err_rel = err / (err0 + err_eps)
            n_iter += 1

            if verbose > 0:
                print(f"iter {n_iter}, err {err.max().item()}, err_rel {err_rel.max().item()}")

        r, state_fields_new = self.problem.compute_residual(u, param_fields=param_fields, loadings=loadings, state_fields=state_fields)

        result = {"sol": u, "err_history": err_history, "n_iter": n_iter, "state_fields": state_fields_new}
        if log_residual:
            result["res_history"] = res_history
        if log_iterates:
            result["u_history"] = u_history
        if loss_callback is not None:
            result["losses"] = losses
        return result
    
    def get_module(self, dtype, device, **kwargs):
        args = {"dtype": dtype, "device": device}
        model = CgModule(
            n_channels=self.problem.n_channels,
            dof_shape=self.problem.dof_shape,
            rhs_model=self.problem.get_rhs_module(**args),
            matvec_model=self.problem.get_matvec_module(**args),
            prec_model=self.preconditioner.get_apply_module(**args),
            **kwargs
        )
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def get_field_module(self, dtype, device, **kwargs):
        args = {"dtype": dtype, "device": device}
        model = CgFieldModule(
            n_channels=self.problem.n_channels,
            dof_shape=self.problem.dof_shape,
            rhs_model=self.problem.get_rhs_module(**args),
            matvec_model=self.problem.get_matvec_module(**args),
            field_module=self.problem.get_field_module(**args),
            prec_model=self.preconditioner.get_apply_module(**args),
            **kwargs
        )
        for param in model.parameters():
            param.requires_grad = False
        return model
