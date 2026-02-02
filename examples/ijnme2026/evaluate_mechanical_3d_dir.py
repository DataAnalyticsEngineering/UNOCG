# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # UNO-CG performance evaluation: mechanical problem in 3D with Dirichlet BC

# ### Imports:

# +
import os
import torch
from torch.utils.data import DataLoader
from unocg.utils.data import MicrostructureDataset
from unocg.problems.mechanical import MechanicalProblem
from unocg.materials.mechanical import LinearElasticity
from unocg.problems import BC
from unocg.solvers.torch import CgSolver
from unocg.preconditioners.torch import UnoPreconditioner, JacobiPreconditioner
from unocg.transforms.fourier import DiscreteSineTransform
from unocg.utils.plotting import *
from unocg.utils.evaluation import *
from matplotlib.transforms import Bbox
import time

try:
    import pyvista as pv
    pv.set_jupyter_backend("trame")
    pv.global_theme.trame.server_proxy_enabled = True
    use_pyvista = True
except:
    use_pyvista = False
    warnings.warn("pyvista not found")
# -

# ### Configuration:

show_plots = True
dtype = torch.float64
torch.set_float32_matmul_precision("high")
torch.set_default_dtype(dtype)
quad_degree = 2
bc = BC.DIRICHLET
device = "cuda" if torch.cuda.is_available() else "cpu"
args = {'device': device, 'dtype': dtype}
base_path = os.path.abspath(os.path.join(os.path.abspath(""), "..", ".."))
data_path = os.path.abspath(os.path.join(base_path, "data"))
results_path = os.path.abspath(os.path.join(data_path, "results"))
if not os.path.exists(results_path):
    os.makedirs(results_path)

# + [markdown] jupyter={"outputs_hidden": false}
# ### Define problem:

# + jupyter={"outputs_hidden": false}
shape = (192, 192, 192)
material = LinearElasticity(n_dim=3, **args)
problem = MechanicalProblem(shape, material=material, bc=bc, quad_degree=quad_degree)

E = torch.tensor([75.0, 400.0], **args)
nu = torch.tensor([0.3, 0.2], **args)
lame_lambda = E * nu / ((1.0 + nu) * (1.0 - 2. * nu))
lame_mu = E / (2.0 * (1.0 + nu))
params = torch.stack([lame_lambda, lame_mu]).T
loadings = 0.05 * torch.tensor([[0.5, -1.0, 0.5, 0.0, 0.0, 0.0]], **args)
# -

# ### Load microstructure from file:

# +
microstructures = MicrostructureDataset(
    file_name=os.path.join(data_path, "3d_microstructures.h5"),
    group_name="test",
    **args
)
data_loader = DataLoader(microstructures, batch_size=1, shuffle=False)
microstructure = microstructures[1]
param_fields = problem.get_param_fields(microstructure.unsqueeze(0), params)

if show_plots and use_pyvista:
    pl = pv.Plotter()
    pl.add_volume(microstructure.cpu().float().numpy(), cmap=["red"], shade=True, scalar_bar_args={"vertical": True, "height": 0.9, "n_labels": 2, "title": ""})
    pl.screenshot(filename=os.path.join(results_path, "3d_microstructure_2.png"), transparent_background=True, window_size=[600,600])
    pl.show()
# -

# ### Compute reference solution

solver = CgSolver(problem)
print("Computing reference solution using unpreconditioned CG...")
solver_start = time.time()
result_ref = solver.solve(param_fields, loadings, rtol=1e-12)
solver_time = time.time() - solver_start
sol_ref = result_ref['sol']
field_ref = problem.compute_field(result_ref['sol'], param_fields, loadings)
print(f"CG solver converged after {result_ref['n_iter']} iterations and {solver_time:.4f} s")

# +
disp = field_ref[0, ..., :3, :, :, :]
stress = field_ref[0, ..., 3:, :, :, :]
stress_norm = stress.norm(dim=problem.ch_dim).cpu().detach()

if show_plots and use_pyvista:
    plot_deformed_rve_3d(problem, disp, stress_norm, loadings, deformation_scaling=8,
                         vmin=0, vmax=26, file=os.path.join(results_path, "mechanical_3d_dir_deformed.png"), figsize=[600, 629])

# + [markdown] jupyter={"outputs_hidden": false}
# ### Load learned UNO preconditioner
# -

weights_uno = torch.load(os.path.join(data_path, "weights_uno_mechanical_3d_dir.pt"), weights_only=True, map_location=device)
trafo = DiscreteSineTransform(dim=[-3, -2, -1])
uno_prec = UnoPreconditioner(problem, trafo, weights_uno)

# ### Run CG method with different preconditioners:

# +
rtol = 1e-11
max_iter = 2000
solver_args = {'rtol': rtol, 'max_iter': max_iter, 'verbose': False}
torch.cuda.empty_cache()
disp_loss = problem.get_disp_loss(reduction="none")

def loss_callback(sol):
    return {"disp": disp_loss(sol, sol_ref).unsqueeze(0)}

unocg_solver = CgSolver(problem, uno_prec, loss_callback=loss_callback, **solver_args)
cg_solver = CgSolver(problem, loss_callback=loss_callback, **solver_args)

unocg_result = unocg_solver.solve(param_fields, loadings)
print(f"UNO-CG converged after {unocg_result['n_iter']} iterations")

cg_result = cg_solver.solve(param_fields, loadings)
print(f"CG converged after {cg_result['n_iter']} iterations")
# -

if show_plots:
    results = [None, unocg_result, None, None, cg_result]
    labels = [None, "UNO-CG", None, None, "CG"]
    metrics = ["disp"]
    metric_labels = [r"nRMSE $\boldsymbol{\tilde{u}} \, [-]$",]
    fig, ax = plt.subplots(1, 1, figsize=[plot_width * 0.5, 2.5], dpi=300, squeeze=False)
    plot_convergence(ax, results, labels, colors, metrics, metric_labels, load_names=["x"], rates=None,
                     xmin=0, xmax=max_iter, bounds=False, ymin=1e-10, ymax=1e0, zoom=True, zoom_it=40, zoom_tol=1e-6)
    ax[0,0].legend()
    fig.tight_layout()
    bbox = fig.get_tightbbox()
    bbox = Bbox([[bbox.x0 - 0.05, bbox.y0 - 0.01], [bbox.x1 + 0.0, bbox.y1 + 0.05]])
    plt.savefig(os.path.join(results_path, "convergence_mechanical_3d_dir.pdf"), dpi=300, bbox_inches=bbox)
    plt.savefig(os.path.join(results_path, "convergence_mechanical_3d_dir.png"), dpi=300, bbox_inches=bbox)
    plt.show()

# ### Runtime measurements:

cg_module = cg_solver.get_module(**args, rtol=1e-6)
unocg_module = unocg_solver.get_module(**args, rtol=1e-6)

benchmark_cg(unocg_module, param_fields, loadings, device=device, n_runs=10)

benchmark_cg(cg_module, param_fields.to(**args), loadings.to(**args), device=device, n_runs=10)


