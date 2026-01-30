# Supplemental data for "Accelerating Conjugate Gradient Solvers for Homogenization Problems with Unitary Neural Operators"

On the one hand, this DaRUS repository contains microstructure datasets that are used for training and testing of the proposed machine-learned preconditioners:

- `2d_microstructures.h5` contains bi-phasic two-dimensional microstructures with a resolution of 400 × 400 pixels that are a subset of the dataset published in
  ```
  Lißner, J. (2020). 2d microstructure data (Version V2) [dataset]. DaRUS. https://doi.org/doi:10.18419/DARUS-1151
  ```
- `3d_microstructures.h5` contains bi-phasic three-dimensional microstructures with a resolution of 192 × 192 × 192 voxels that are a subset of the dataset published in
  ```
  Prifling, B., Röding, M., Townsend, P., Neumann, M., and Schmidt, V. (2020). Large-scale statistical learning for mass transport prediction in porous materials using 90,000 artificially generated microstructures [dataset]. Zenodo. https://doi.org/10.5281/zenodo.4047774
  ```
- a `PyTorch` data loader for both datasets is available in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/unocg/utils/data.py>

On the other hand, this DaRUS repository contains the weights of the proposed machine-learned preconditioners trained for various problem formulations:

## Thermal homogenization problem in 2D with periodic BC
  - weights for the UNO preconditioner: `weights_uno_thermal_2d_per.pt`
  - weights for the UNO preconditioner (naive training): `weights_uno_naive_thermal_2d_per.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_thermal_2d_per.ipynb>

## Thermal homogenization problem in 3D with periodic BC
  - weights for the UNO preconditioner: `weights_uno_thermal_3d_per.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_thermal_3d_per.ipynb>

## Thermal homogenization problem in 3D with Dirichlet BC
  - weights for the UNO preconditioner: `weights_uno_thermal_3d_dir.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_thermal_3d_dir.ipynb>

## Mechanical homogenization problem in 2D with periodic BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_2d_per.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_2d_per.ipynb>

## Mechanical homogenization problem in 2D with Dirichlet BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_2d_dir.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_2d_dir.ipynb>

## Mechanical homogenization problem in 2D with mixed BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_2d_mixed.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_2d_mixed.ipynb>

## Mechanical homogenization problem in 3D with periodic BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_3d_per.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_3d_per.ipynb>

## Mechanical homogenization problem in 3D with Dirichlet BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_3d_dir.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_3d_dir.ipynb>

## Mechanical homogenization problem in 3D with mixed BC
  - weights for the UNO preconditioner: `weights_uno_mechanical_3d_mixed.pt`
  - example in the software repository: <https://github.com/DataAnalyticsEngineering/UNOCG/tree/main/examples/ijnme2026/evaluate_mechanical_3d_mixed.ipynb>
