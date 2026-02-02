# UNO-CG: Accelerating Conjugate Gradient Solvers with Unitary Neural Operators

<!-- The badges we want to display -->
[![DOI][paper-shield]][paper-url]
[![Data][data-shield]][data-url]
[![PyPI](https://img.shields.io/pypi/v/unocg)](https://pypi.org/project/unocg/)
[![Python](https://img.shields.io/badge/python-3.11-purple.svg)](https://www.python.org/)
<!--[![Documentation Status][docs-shield]][docs-url]-->
<!--[![pytest](https://github.com/DataAnalyticsEngineering/UNOCG/actions/workflows/ci.yml/badge.svg)](https://github.com/DataAnalyticsEngineering/UNOCG/actions/workflows/ci.yml)
[![flake8](https://img.shields.io/badge/flake8-checked-blue.svg)](https://flake8.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->
<!--[![MIT License][license-shield]][license-url]-->
<!--[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)-->

This repository contains the software related to the article **Accelerating Conjugate Gradient Solvers for Homogenization Problems with Unitary Neural Operators**, accepted for publication in the *International Journal for Numerical Methods in Engineering (IJNME)* by Julius Herb and Felix Fritzen:

> Herb, J. and Fritzen, F. (2026), Accelerating Conjugate Gradient Solvers for Homogenization Problems with Unitary Neural Operators. Int J Numer Methods Eng. https://doi.org/10.1002/nme.70277

## Abstract

While classical numerical solvers offer reliable and accurate solutions supported by a solid theoretical foundation, their high computational costs and slow convergence remain limiting factors.
As a result, scientific machine learning is emerging as a promising alternative, aiming to rapidly approximate solutions using surrogate models.
However, such approaches often lack guaranteed accuracy and physical consistency.
This raises the question of whether it is possible to develop hybrid approaches that combine the advantages of both data-driven methods and classical solvers.
To address this, we introduce UNO-CG, a hybrid solver that accelerates conjugate gradient (CG) solvers using specially designed machine-learned preconditioners, while ensuring convergence by construction.
As a preconditioner, we propose Unitary Neural Operators (UNOs) as a modification of the established Fourier Neural Operators.
Our method can be interpreted as a data-driven discovery of Green's functions, which are then used much like expert knowledge to accelerate iterative solvers.

<!--
## Documentation

The documentation of this software, including examples on how to use **UNOCG**, can be found under [Documentation](https://DataAnalyticsEngineering.github.io/UNOCG/).
-->

## Features

UNO-CG (**unocg**) is a Python package that provides efficient implementations of solvers for homogenization problems, or parametric Partial Differential Equations (PDEs) in general.
It includes an implementation of the [Fourier-Accelerated Nodal Solvers (FANS)](https://doi.org/10.1007/s00466-017-1501-5) as a special case.

The focus is on the following features:
- support for common homogenization problems in computational mechanics, including thermal problems and mechanical problems
- support for common boundary conditions on the RVE (periodic, Dirichlet, Neumann, and combinations)
- completely based on PyTorch and hence supports GPU acceleration using NVIDIA CUDA or AMD ROCm, parallelization on CPUs using OpenMP, and automatic differentiation
- solvers are implemented as iterative solvers in a matrix-free way that are accelerated by machine-learned preconditioners

On the other hand, there are the following restrictions:
- restriction to a FEM discretization on regular grids, i.e., parameters are defined as voxelized data; for unstructured meshes, classical FEM solvers are the better choice
- optimized for solving small- and medium-sized homogenization problems in a many-query context, i.e., for many different microstructures, material parameters, and loadings; for large-sized homogenization problems, our [MPI-based implementation of FANS](https://github.com/DataAnalyticsEngineering/FANS) for CPUs is the better choice

Typical applications include:
- solving homogenization problems in $\text{FE}^2$ simulations
- data generation for reduced order models and machine-learned material models
- uncertainty quantification and parameter studies for homogenization problems
- solving inverse problems, e.g., design of architected materials

## Installation

### Using the PIP package

A PIP package is available on pypi and can be installed with:

```bash
pip install unocg
```

### From this repository

The most recent version of the PIP package can also be installed directly after cloning this repository.

```bash
git clone https://github.com/DataAnalyticsEngineering/UNOCG.git
cd UNOCG
pip install -e .
```

If you want to install optional dependencies for development:

```bash
git clone https://github.com/DataAnalyticsEngineering/UNOCG.git
cd UNOCG
pip install -e .[all]
```


### Requirements

- Python 3.11 or later
- `pip` packages listed in `pyproject.toml`
- Supplemental data: [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--5686-d45815.svg)](https://doi.org/10.18419/darus-5686)

All necessary data can be downloaded from [DaRUS](https://darus.uni-stuttgart.de/) using the script [`download_data.sh`](data/download_data.sh) in the directory [`data`](data).

## Acknowledgments

- Contributions by Felix Fritzen are partially funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under
  Germany’s Excellence Strategy - EXC 2075 – 390740016. Felix Fritzen is funded by Deutsche Forschungsgemeinschaft (DFG, German
  Research Foundation) within the Heisenberg program DFG-FR2702/8 - 406068690 and DFG-FR2702/10 - 517847245.

- Contributions of Julius Herb are partially funded by the Ministry of Science, Research and the Arts (MWK) Baden-Württemberg, Germany, within the Artificial Intelligence Software Academy (AISA).

- The authors acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

Affiliation: [Data Analytics in Engineering, University of Stuttgart](http://www.mib.uni-stuttgart.de/dae)

[license-shield]: https://img.shields.io/github/license/DataAnalyticsEngineering/UNOCG.svg
[license-url]: https://github.com/DataAnalyticsEngineering/UNOCG/blob/main/LICENSE
[data-shield]: https://img.shields.io/badge/doi-10.18419%2Fdarus--5686-d45815.svg
[data-url]: https://doi.org/10.18419/darus-5686
[paper-shield]: https://img.shields.io/badge/doi-10.1002%2Fnme.70277-d45815.svg
[paper-url]: https://doi.org/10.1002/nme.70277
[docs-url]: https://DataAnalyticsEngineering.github.io/UNOCG
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg
