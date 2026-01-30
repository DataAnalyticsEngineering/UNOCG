from typing import Dict

import torch
from unocg.problems import Problem
from unocg.solvers.torch import CgSolver
from unocg.modules.solvers import CgModule
from unocg.preconditioners import Preconditioner
import timeit


def benchmark(model, inputs, device, n_runs=2, n_warmup=1, verbose=True):
    def run_model():
        with torch.inference_mode():
            model(*inputs)
            torch.cuda.synchronize(device)

    for _ in range(n_warmup):
        run_model()
    model_time = timeit.timeit(run_model, number=n_runs) / n_runs
    if verbose:
        print(f"Runtime per execution: {model_time*1000.:.4f}ms")
    return model_time


def benchmark_cg(cg_module, param_fields, loadings, device, n_runs=2, n_warmup=1, verbose=True):
    with torch.inference_mode():
        guess = cg_module.zero_guess(param_fields, loadings)

        inputs_module = (guess, param_fields, loadings)

        print(f"Overall solver:")
        u = cg_module(*inputs_module)
        time_overall = benchmark(cg_module, inputs_module, device, n_runs=n_runs)

        iter_layer = cg_module.iteration_layers[0]
        print(f"Solver iteration:")
        batch_shape = (*param_fields.shape[:(-cg_module.n_dim - 1)], loadings.shape[0])
        field_shape = (*batch_shape, *cg_module.shape)
        iv_fields, iv_scalars = iter_layer.init_internal_variables(batch_shape=batch_shape, init_residual=u)
        time_iter = benchmark(iter_layer, (u, u, param_fields, loadings, iv_fields, iv_scalars), device, n_runs=n_runs*10)

        print(f"Preconditioner application:")
        time_prec = benchmark(cg_module.prec_model, (u,), n_runs=n_runs*100, n_warmup=n_warmup, device=device);

        print(f"Residual computation:")
        time_matvec = benchmark(cg_module.matvec_model, (u, param_fields), n_runs=n_runs*10, n_warmup=n_warmup, device=device)

        return time_overall, time_iter, time_prec, time_matvec
