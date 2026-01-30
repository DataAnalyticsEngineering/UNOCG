"""
Plotting utilities
"""
from typing import Optional, Iterable, List, Tuple

import matplotlib
from matplotlib.tri import Triangulation
import numpy as np
import torch
from matplotlib import pyplot as plt, ticker
import math
import shutil
from unocg.problems import Problem, BC

try:
    import pyvista as pv
except:
    pass

plt.rcParams["text.usetex"] = True if shutil.which('latex') else False
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plot_width = 6.3

colors = [
    "g",
    "#00BEFF",
    "#004191",
    "r",
    "k"
]


def plot_ms(image: torch.Tensor, file: Optional[str] = None, show_axis: bool = False, show_cbar: bool = False):
    """
    Plot microstructure image

    :param image:
    :param file:
    :param show_axis:
    :param show_cbar:
    :return:
    """
    image = image.detach().cpu()
    fig, ax = plt.subplots(1, 1, figsize=[6.3 / 2, 2.65 if show_cbar else 6.3 / 2], dpi=300)
    ms_cmap = plt.get_cmap("viridis", 2)
    im = ax.imshow(image, origin="lower", interpolation="none", extent=(-0.5, 0.5, -0.5, 0.5), cmap=ms_cmap)

    if show_cbar:
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1])
        cb.ax.set_title(rf"$\chi_1$")
        cb.ax.set_yticklabels(["$0$", "$1$"])

    if show_axis:
        ax.axis("on")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.set_xlabel(r"$\frac{x_1}{l_1}$")
        ax.set_ylabel(r"$\frac{x_2}{l_2}$")
    else:
        ax.axis("off")

    plt.tight_layout()
    if file is not None:
        plt.savefig(file, dpi=300)
    plt.show()


def plot_prec_action(ax, field_ref, init_res, precs, labels, ch_idx=0, plot_res=False, centered=True, ch_label=None):
    idx = (0,0,ch_idx)
    ax_i = 0
    if centered:
        s_abs_max = torch.max(field_ref[idx].cpu().abs())
        s_max, s_min = s_abs_max, -s_abs_max
    else:
        s_max, s_min = torch.max(field_ref[idx].cpu()), torch.min(field_ref[idx].cpu())
    
    prec_actions = []
    with torch.inference_mode():
        for prec in precs:
            prec_action = prec.apply_field(init_res)
            prec_action = prec_action / prec_action[idx].max() * field_ref[idx].abs().max()
            prec_actions.append(prec_action)
            if centered:
                s_abs_max = torch.maximum(s_abs_max, torch.max(prec_action[idx].cpu().abs()))
                s_max, s_min = s_abs_max, -s_abs_max
            else:
                s_max = torch.maximum(s_max, torch.max(prec_action[idx].cpu()))
                s_min = torch.minimum(s_min, torch.min(prec_action[idx].cpu()))

    if plot_res:
        im = ax[0,ax_i].imshow(init_res[idx].cpu().detach(), origin="lower", cmap="seismic")
        if ch_label is None:
            ax[0,ax_i].set_title(r"$\boldsymbol{r}^{(0)}$")
        else:
            ax[0,ax_i].set_title(rf"$\left( \boldsymbol{{r}}^{{(0)}} \right)_{{{ch_label}}}$")
        plt.colorbar(im, ax=ax[ax_i])
        ax_i += 1

    im = ax[0,ax_i].imshow(field_ref[idx].cpu().detach(), origin="lower", cmap="jet", vmax=s_max, vmin=s_min)
    if ch_label is None:
        ax[0,ax_i].set_title(r"$\boldsymbol{u}_{\underline{\mu}}$")
    else:
        ax[0,ax_i].set_title(rf"$\left( \boldsymbol{{u}}_{{\underline{{a}}}} \right)_{{{ch_label}}}$")
    plt.colorbar(im, ax=ax[0,ax_i])
    ax_i += 1
    
    for prec_action, label in zip(prec_actions, labels):
        im = ax[0,ax_i].imshow(prec_action[idx].cpu().detach(), origin="lower", cmap="jet", vmax=s_max, vmin=s_min)
        if ch_label is None:
            ax[0,ax_i].set_title(rf"$\boldsymbol{{s}}_\text{{{label}}}$")
        else:
            ax[0,ax_i].set_title(rf"$\left( \boldsymbol{{s}}_\text{{{label}}} \right)_{{{ch_label}}}$")
        plt.colorbar(im, ax=ax[0,ax_i])
        ax_i += 1

    for ax_handle in ax.ravel():
        ax_handle.set_xticks([])
        ax_handle.set_yticks([])


def plot_convergence(ax, results, labels, colors, metrics, metric_labels, load_names=None, rates=None, show_load_labels=False,
                     xmin=0, xmax=None, bounds=False, ymin=1e-10, ymax=1e0, zoom=False, zoom_it = 10, zoom_tol = 1e-3):
    """
    
    """
    if load_names is None:
        load_names = ["x"]
    if rates is None:
        rates = (None,) * len(results)
    colors = colors[:len(results)]
    while len(labels) < len(results):
        labels.append("CG")

    for load_i, load_name in enumerate(load_names):
        if show_load_labels:
            ax[0, load_i].set_title(f"loading {load_name}")

        for metric_i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax[metric_i, load_i].set_xlabel(r"iterations $[-]$")
            ax[metric_i, load_i].set_ylabel(metric_label)
            ax[metric_i, load_i].set_ylim(bottom=ymin, top=ymax)
            if zoom:
                axins = ax[metric_i, load_i].inset_axes([0.2, 0.15, 0.1, 0.45],
                    xlim=(-1, zoom_it), ylim=(zoom_tol, 1e0), xticklabels=[], yticklabels=[])
                axins.grid()
                axins.set_xticks([0, zoom_it], ["$0$", f"${zoom_it}$"])

            for result, label, color, rate in zip(results, labels, colors, rates):
                if result is None:
                    continue
                iters = torch.arange(result["err_history"].shape[0])
                metric_losses = result["losses"][metric]
                if metric_losses.ndim == 1:
                    metric_losses = metric_losses.unsqueeze(-1)
                ax[metric_i, load_i].semilogy(iters, metric_losses[:, load_i].cpu() / metric_losses[0, load_i].cpu(), '-', c=color, label=label)

                if zoom:
                    axins.semilogy(iters, metric_losses[:, load_i].cpu() / metric_losses[0, load_i].cpu(), '-', c=color)

            if zoom:
                ax[metric_i, load_i].indicate_inset_zoom(axins, edgecolor="black")
    
    for ax_handle in ax.ravel():
        ax_handle.grid()
        ax_handle.set_xlim(left=xmin, right=xmax)


def plot_convergence_histogram(ax, results, labels, colors, rtol=1e-6, bins=None, xmin=0, xmax=None, log_scale=False, legend=True):
    """
    
    """
    for result, label, color in zip(results, labels, colors):
        if result is None:
            continue
        metric = get_rel_residual(result)
        iters = (metric > rtol).sum(dim=0)
        ax.hist(iters, edgecolor=color, facecolor=color, bins=bins, label=label)
    if log_scale:
        ax.set_xlim(max(xmin, 1), xmax)
        ax.set_xscale("log")
    else:
        ax.set_xlim(xmin, xmax)
    ax.set_axisbelow(True)
    ax.grid()
    if legend:
        ax.legend()

def get_rel_residual(result):
    """
    
    """
    return result['err_history'].cpu().flatten(start_dim=-2) / result['err_history'].cpu().flatten(start_dim=-2)[0]


def plot_deformed_rve_2d(
    problem,
    disp,
    field,
    loading=None,
    fluctuation_scaling: float = 1.0,
    deformation_scaling: float = 1.0,
    plot_loading: bool = False,
    plot_boundary: bool = False,
    shading: str = "gouraud",
    file: Optional[str] = None,
    vmin: Optional[List[float]] = None,
    vmax: Optional[List[float]] = None,
    figsize: Optional[List[float]] = None
):
    """

    :param disp:
    :param field:
    :param loading:
    :param fluctuation_scaling:
    :param deformation_scaling:
    :param plot_loading:
    :param plot_boundary:
    :param shading:
    :param file:
    :param vmin:
    :param vmax:
    :return:
    """
    def apply_mask(triang, alpha=0.4):
        # Mask triangles with sidelength greater than a threshold alpha
        triangles = triang.triangles
        # Mask off unwanted triangles.
        x = triang.x
        y = triang.y
        xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
        ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
        # apply masking
        triang.set_mask(maxi > alpha)
    
    if loading is None:
        loading = torch.eye(3, dtype=disp.dtype, device=disp.device)

    n_loadings = loading.shape[0]

    #if problem.n_dim == 2:
    #    disp = disp.transpose(-1, -2)

    deformed_coords = problem.get_deformed_coordinates(
        disp, loading, fluctuation_scaling=fluctuation_scaling, deformation_scaling=deformation_scaling
    )
    loading_coords = problem.get_deformed_coordinates(
        disp, loading, fluctuation_scaling=0.0, deformation_scaling=deformation_scaling
    )
    coords = problem.get_node_coords()
    #boundary_idx = problem.get_boundary_idx()
    #boundary = coords[..., boundary_idx[0], boundary_idx[1]]
    #deformed_boundary = deformed_coords[..., boundary_idx[0], boundary_idx[1]]
    #loading_boundary = loading_coords[..., boundary_idx[0], boundary_idx[1]]

    loadings = [
        rf"$\bar{{\boldsymbol{{\varepsilon}}}}=\bar{{\boldsymbol{{\varepsilon}}}}^{{({i + 1})}}$"
        for i in range(n_loadings)
    ]

    if figsize is None:
        figsize = [6.3, 2.0]

    if problem.n_dim == 3:
        N_cut = problem.n_grid[-1] // 2
        field = field[..., N_cut]
        deformed_coords = deformed_coords[..., N_cut]
            
    fig, ax = plt.subplots(1, n_loadings, figsize=figsize, dpi=300, squeeze=False)
    for load_idx, load_name in enumerate(loadings):
        ax[0, load_idx].axis("off")
        ax[0, load_idx].set_aspect("equal")
        ax[0, load_idx].set_title(load_name)
        tri = Triangulation(deformed_coords[load_idx, 0].ravel(), deformed_coords[load_idx, 1].ravel())
        apply_mask(tri, alpha=0.02)

        if vmin is None:
            vmin_idx = field[load_idx].min().item()
        else:
            vmin_idx = vmin[load_idx]
        if vmax is None:
            vmax_idx = 0.5 * field[load_idx].max().item()
        else:
            vmax_idx = vmax[load_idx]
        
        tpc = ax[0, load_idx].tripcolor(
            tri, field[load_idx].ravel(), cmap="jet", shading=shading, rasterized=True, vmin=vmin_idx, vmax=vmax_idx
        )

        if plot_loading:
            ax[0, load_idx].plot(
                loading_boundary[load_idx, 0].ravel(),
                loading_boundary[load_idx, 1].ravel(),
                "k--",
                lw=1,
            )
        if plot_boundary:
            ax[0, load_idx].plot(
                deformed_boundary[load_idx, 0].ravel(),
                deformed_boundary[load_idx, 1].ravel(),
                "r--",
                lw=1,
            )
        clb = fig.colorbar(tpc)
        clb.ax.set_title(r"$||\boldsymbol{\sigma}|| \,[\mathrm{GPa}]$")
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, dpi=300)
    plt.show()


def plot_deformed_rve_3d(
    problem,
    disp,
    field,
    loadings,
    fluctuation_scaling: float = 1.0,
    deformation_scaling: float = 1.0,
    file: Optional[str] = None,
    vmin: Optional[List[float]] = None,
    vmax: Optional[List[float]] = None,
    figsize: Optional[List[float]] = None
):
    disp = torch.nn.functional.pad(disp, pad=[0,1,0,1,0,1], mode="circular")
    coords = problem.get_node_coords().to(dtype=disp.dtype, device=disp.device)
    deformations = problem.get_deformations(disp.transpose(-1, -3), loadings, fluctuation_scaling=1.0)[0]
    
    x, y, z = coords[2].cpu().numpy(), coords[1].cpu().numpy(), coords[0].cpu().numpy()
    grid = pv.StructuredGrid(x, y, z)
    grid['vectors'] = deformations.flatten(start_dim=1).T.cpu().numpy() * deformation_scaling
    warped = grid.warp_by_vector()
    
    pl = pv.Plotter()
    pl.add_mesh(warped, scalars=field.numpy().ravel(), clim=[vmin, vmax], label="stress norm", cmap="jet", lighting=True, diffuse=0.2, specular=1.0, ambient=0.6, scalar_bar_args={"vertical": True})
    pl.screenshot(filename=file, window_size=figsize)
    pl.show()
