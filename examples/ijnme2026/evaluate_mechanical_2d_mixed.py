# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # UNO-CG performance evaluation: mechanical problem in 2D with mixed BC

# %% [markdown]
# ### Imports:

# %%
import os
import torch
from torch.utils.data import DataLoader
from unocg.utils.data import MicrostructureDataset
from unocg.problems.mechanical import MechanicalProblem
from unocg.materials.mechanical import LinearElasticity
from unocg.problems import BC
from unocg.solvers.torch import CgSolver
from unocg.preconditioners.torch import UnoPreconditioner, JacobiPreconditioner
from unocg.transforms.fourier import DiscreteMixedTransform
from unocg.utils.plotting import *
from unocg.utils.evaluation import *
from unocg import config
from matplotlib.transforms import Bbox
import time


# %% [markdown]
# ### Configuration:

# %%
show_plots = True
dtype = torch.float64
torch.set_float32_matmul_precision("high")
torch.set_default_dtype(dtype)
quad_degree = 2
bc = BC.DIRICHLET_TB
device = "cuda" if torch.cuda.is_available() else "cpu"
args = {'device': device, 'dtype': dtype}

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Create problem

# %% jupyter={"outputs_hidden": false}
shape = (400, 400)
material = LinearElasticity(n_dim=2, **args)
problem = MechanicalProblem(shape, material=material, quad_degree=quad_degree, bc=bc)

E = torch.tensor([1.0, 10.0], **args)
nu = torch.tensor([0.0, 0.3], **args)
lame_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lame_mu = E / (2.0 * (1.0 + nu))
params = torch.stack([lame_lambda, lame_mu]).T
loadings = torch.tensor([[0.05, -0.05, 0.0]], **args)

# %% [markdown]
# ### Generate artificial microstructure or load microstructure from file:

# %%
microstructures = MicrostructureDataset(
    file_name=os.path.join(config.data_path, "2d_microstructures.h5"),
    group_name="test",
    **args
)
data_loader = DataLoader(microstructures, batch_size=1, shuffle=False)
microstructure = microstructures[0]
param_fields = problem.get_param_fields(microstructure.unsqueeze(0), params)

if show_plots:
    plot_ms(microstructure, show_axis=False, show_cbar=True, file=os.path.join(config.results_path, "2d_microstructure_2.pdf"))

# %% [markdown] jupyter={"outputs_hidden": false}
# ### Load learned preconditioner

# %%
weights_uno = torch.load(os.path.join(config.data_path, "weights_uno_mechanical_2d_mixed.pt"), weights_only=True, map_location=device)

transform = DiscreteMixedTransform(dim_sine=[-2], dim_fourier=[-1])
uno_prec = UnoPreconditioner(problem, transform, weights_uno)

# %% [markdown]
# ### Compute reference solution

# %%
solver = CgSolver(problem)
print("Computing reference solution using unpreconditioned CG...")
solver_start = time.time()
result_ref = solver.solve(param_fields, loadings, rtol=1e-12)
solver_time = time.time() - solver_start
field_ref = problem.compute_field(result_ref["sol"], param_fields, loadings)
print(f"CG solver converged after {result_ref["n_iter"]} iterations and {solver_time:.4f} s")

# %%
if show_plots:
    disp = field_ref[..., :2, :, :].squeeze(0)
    stress = field_ref[..., 2:, :, :].squeeze(0)
    plot_deformed_rve_2d(
        problem,
        disp.cpu(),
        field=stress.norm(dim=-3).cpu(),
        loading=loadings,
        fluctuation_scaling = 1.0,
        deformation_scaling = 5.0,
        plot_loading = False,
        plot_boundary = False,
        vmin=[0., 0., 0.],
        vmax=[0.6, 0.6, 0.6],
        file = os.path.join(config.results_path, "mechanical_2d_mixed_deformed.png")
    )

# %% [markdown]
# ### Create Jacobi Preconditioner
#
# In contrast to FANS, this preconditioner depends on the stiffness matrix, i.e., changes together with the microstructure

# %%
A = problem.assemble_matrix(param_fields)
A_diag = problem.sparse_diag(A)
jac_weights = 1.0 / A_diag
jac_prec = JacobiPreconditioner(problem, jac_weights)

# %% [markdown]
# ### Run CG with different preconditioners

# %% jupyter={"is_executing": true}
rtol = 1e-11
max_iter = 10000
solver_args = {"rtol": rtol, "max_iter": max_iter}

def loss_callback(sol):
    field_pred = problem.compute_field(sol, param_fields, loadings.to(**args))
    return problem.compute_losses(field_pred, field_ref.to(**args))

unocg_solver = CgSolver(problem, uno_prec, loss_callback=loss_callback, **solver_args)
jac_solver = CgSolver(problem, jac_prec, loss_callback=loss_callback, **solver_args)
solver = CgSolver(problem, loss_callback=loss_callback, **solver_args)

unocg_result = unocg_solver.solve(param_fields, loadings=loadings.to(**args))
print(f"UNO-CG converged after {unocg_result['n_iter']} iterations")

jac_result = jac_solver.solve(param_fields, loadings=loadings.to(**args))
print(f"Jac-CG converged after {jac_result['n_iter']} iterations")

cg_result = solver.solve(param_fields, loadings=loadings.to(**args))
print(f"CG converged after {cg_result['n_iter']} iterations")

# %%
if show_plots:
    results = [None, unocg_result, None, jac_result, cg_result]
    labels = [None, "UNO-CG", None,  "Jac-CG", "CG"]
    metrics = ["disp"]
    metric_labels = [r"nRMSE $\boldsymbol{\tilde{u}} \, [-]$"]
    load_names = ["xx"]
    fig, ax = plt.subplots(1, 1, figsize=[plot_width * 0.5, 2.5], dpi=300, squeeze=False)
    plot_convergence(ax, results, labels, colors, metrics, metric_labels, load_names=load_names, rates=None,
                     xmin=0, xmax=7000, bounds=False, ymin=1e-10, ymax=1e0, zoom=True, zoom_it=60, zoom_tol=1e-6)
    ax[0,0].legend(loc="upper right")
    fig.tight_layout()
    bbox = fig.get_tightbbox()
    bbox = Bbox([[bbox.x0 - 0.05, bbox.y0 - 0.01], [bbox.x1 + 0.0, bbox.y1 + 0.05]])
    plt.savefig(os.path.join(config.results_path, "convergence_mechanical_2d_mixed_disp.pdf"), dpi=300, bbox_inches=bbox)
    plt.show()

# %% [markdown]
# ### Runtime measurements:

# %%
cg_solver = CgSolver(problem)
cg_module = cg_solver.get_module(**args, rtol=1e-6)
jac_module = jac_solver.get_module(**args, rtol=1e-6)
unocg_module = unocg_solver.get_module(**args, rtol=1e-6)

# %%
benchmark_cg(unocg_module, param_fields, loadings, device=device, n_runs=100);

# %%
benchmark_cg(jac_module, param_fields, loadings, device=device, n_runs=100);

# %%
benchmark_cg(cg_module, param_fields, loadings, device=device, n_runs=100);
