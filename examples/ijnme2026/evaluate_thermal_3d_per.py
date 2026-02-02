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

# # UNO-CG performance evaluation: thermal problem in 3D with periodic BC

# ### Imports:

# +
import os
import torch
from torch.utils.data import DataLoader
from unocg.utils.data import MicrostructureDataset
from unocg.problems.thermal import ThermalProblem
from unocg.materials.thermal import LinearHeatConduction
from unocg.problems import BC
from unocg.solvers.torch import CgSolver
from unocg.preconditioners.torch import FansPreconditioner, UnoPreconditioner, JacobiPreconditioner
from unocg.transforms.fourier import DiscreteFourierTransform
from unocg.utils.plotting import *
from unocg.utils.evaluation import *
from unocg import config
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
bc = BC.PERIODIC
device = "cuda" if torch.cuda.is_available() else "cpu"
args = {'device': device, 'dtype': dtype}

# ### Define problem:

# + jupyter={"outputs_hidden": false}
shape = (192, 192, 192)
model = LinearHeatConductionModel(n_dim=3, **args)
problem = ThermalProblem(shape, model=model, bc=bc, quad_degree=quad_degree)

kappa0, kappa1 = 1.0, 0.2
params = torch.tensor([kappa0, kappa1], **args).reshape(2,1)
loadings = torch.eye(3).to(**args)[:1]
# -

# ### Load microstructure from file:

# +
microstructures = MicrostructureDataset(
    file_name=os.path.join(config.data_path, "3d_microstructures.h5"),
    group_name="test",
    **args
)
data_loader = DataLoader(microstructures, batch_size=1, shuffle=False)
microstructure = microstructures[334]
param_fields = problem.get_param_fields(microstructure.unsqueeze(0), params)

if show_plots and use_pyvista:
    pl = pv.Plotter()
    pl.add_volume(microstructure.cpu().float().numpy(), cmap=["red"], shade=True, scalar_bar_args={"vertical": True, "height": 0.9, "n_labels": 2, "title": ""})
    pl.screenshot(filename=os.path.join(config.results_path, "3d_microstructure_1.png"), transparent_background=True, window_size=[600,600])
    pl.show()

# + [markdown] jupyter={"outputs_hidden": false}
# ### Compute reference solution using unpreconditioned CG:

# + jupyter={"outputs_hidden": false}
cg_solver = CgSolver(problem)
print("Computing reference solution using unpreconditioned CG...")
solver_start = time.time()
result_ref = cg_solver.solve(param_fields, loadings, rtol=1e-12)
solver_time = time.time() - solver_start
sol_ref = result_ref["sol"]
field_ref = problem.compute_field(result_ref["sol"], param_fields, loadings)
print(f"CG solver converged after {result_ref["n_iter"]} iterations and {solver_time:.4f} s")

# + [markdown] jupyter={"outputs_hidden": false}
# ### Load learned UNO preconditioner:
# -

weights_uno = torch.load(os.path.join(config.data_path, "weights_uno_thermal_3d_per.pt"), weights_only=True, map_location=device)
trafo = DiscreteFourierTransform(dim=[-3, -2, -1])
uno_prec = UnoPreconditioner(problem, trafo, weights_uno)

# ### Create FANS preconditioner:

# + jupyter={"outputs_hidden": false}
params_ref = params.mean()
cond_ref = problem.conductivity(params_ref)
fans_prec = FansPreconditioner(problem, cond_ref)
# -

# ### Create Jacobi Preconditioner:
#
# In contrast to FANS, this preconditioner depends on the stiffness matrix, i.e., changes together with the microstructure

A = problem.assemble_matrix(param_fields.to(**args))
A_diag = problem.sparse_diag(A).to(device=device)
jac_weights = 1.0 / A_diag
jac_prec = JacobiPreconditioner(problem, jac_weights)

# ### Run CG method with different preconditioners:

# +
rtol = 1e-11
max_iter = 1000
solver_args = {'rtol': rtol, 'max_iter': max_iter}

def loss_callback(sol):
    field = problem.compute_field(sol, param_fields, loadings)
    return problem.compute_losses(field, field_ref)


fans_solver = CgSolver(problem, fans_prec, loss_callback=loss_callback, **solver_args)
unocg_solver = CgSolver(problem, uno_prec, loss_callback=loss_callback, **solver_args)
jac_solver = CgSolver(problem, jac_prec, loss_callback=loss_callback, **solver_args)
cg_solver = CgSolver(problem, loss_callback=loss_callback, **solver_args)

fans_result = fans_solver.solve(param_fields, loadings)
print(f"FANS converged after {fans_result['n_iter']} iterations")

unocg_result = unocg_solver.solve(param_fields, loadings)
print(f"UNO-CG converged after {unocg_result['n_iter']} iterations")

jac_result = jac_solver.solve(param_fields, loadings, zero_mean=True)
print(f"Jac-CG converged after {jac_result['n_iter']} iterations")

cg_result = cg_solver.solve(param_fields.to(**args), loadings=loadings.to(**args))
print(f"CG converged after {cg_result['n_iter']} iterations")
# -

results = [fans_result, unocg_result, None, jac_result, cg_result]
labels = ["FANS", "UNO-CG", None, "Jac-CG", "CG"]
metrics = ["temp", "flux"]
metric_labels = [r"nRMSE $\tilde{\vartheta} \, [-]$", r"nRMSE $\boldsymbol{q} \, [-]$"]
fig, ax = plt.subplots(1, len(metrics), figsize=[plot_width * 0.9, 2.5], dpi=300, squeeze=False)
plot_convergence(ax.T, results, labels, colors, metrics, metric_labels, load_names=["x"], rates=None,
                 xmin=0, xmax=680, bounds=False, ymin=1e-10, ymax=1e0, zoom=True, zoom_it=25, zoom_tol=1e-6)
handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,0].legend()
ax[0,1].legend()
plt.tight_layout()
bbox = fig.get_tightbbox()
bbox = Bbox([[bbox.x0 - 0.05, bbox.y0 - 0.01], [bbox.x1 + 0.02, bbox.y1 + 0.05]])
plt.savefig(os.path.join(config.results_path, "convergence_thermal_3d_per.pdf"), dpi=300, bbox_inches=bbox)
plt.show()

# ### Runtime measurements:

cg_module = cg_solver.get_module(**args, rtol=1e-6)
jac_module = jac_solver.get_module(**args, rtol=1e-6)
fans_module = fans_solver.get_module(**args, rtol=1e-6)
unocg_module = unocg_solver.get_module(**args, rtol=1e-6)

benchmark_cg(fans_module, param_fields, loadings, device=device, n_runs=10)

benchmark_cg(unocg_module, param_fields, loadings, device=device, n_runs=10)

benchmark_cg(jac_module, param_fields, loadings, device=device, n_runs=10)

benchmark_cg(cg_module, param_fields, loadings, device=device, n_runs=10)


