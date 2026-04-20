# MITONet for 2D Shallow Water Equations

This repository consolidates the MITONet 2d shallow-water emulator workflows for both the **Shinnecock Inlet** and **Red River** studies that appear in the paper *"A Neural Operator Emulator for Coastal and Riverine Shallow Water Dynamics"*. Each case retains its own training scripts, configuration files, and notebooks, while the shared data directory and licensing live at the repo root.

## Repository Layout

```
MITONet/
├── cases/
│   ├── Shinnecock/
│   │   ├── scripts/         # Training + optimization entry points for every model flavor
│   │   ├── settings/        # Configuration for MITONet and the baseline operators
│   │   ├── notebooks/       # Visualization utilities for the Shinnecock study
│   │   └── src/             # Model components and data-processing utilities
│   └── RedRiver/
│       ├── scripts/         # MITONet training + optimization scripts for Red River
│       ├── settings/
│       ├── notebooks/
│       └── src/
├── data/                    # Placeholder for hydrodynamic simulation data
├── LICENSE
└── README.md
```

Each case directory is self-contained, so you can `cd cases/<Case>/scripts` and run the same commands you used previously. The shared `data/` directory lives alongside `cases/` and can store datasets that are reused across both scenarios. Add subfolders inside `data/` if you wish to keep case-specific raw files separated (e.g., `data/shinnecock/`, `data/redriver/`).

## Environment Setup

Create and activate the conda environment (from anywhere inside the repo):

```bash
conda create -n mitonet -c conda-forge tensorflow=2.14 matplotlib tqdm scikit-learn optuna optuna-integration netcdf4 pandas cmocean
conda activate mitonet
```

Any additional packages that are case-specific can be added within each case directory's README or scripts as needed.

## Data

Each case has its own dataset hosted on DesignSafe-CI. Download the appropriate archive, then store it under `data/<case>/...` so files do not mix:

```
Shinnecock: https://designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-5716
Red River:  https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-6207/#detail-5855ec2a-cd4d-423b-80a4-93769c023f29
```

Extract each archive directly under `data/PRJ-5716` and `data/PRJ-6207` and keep the publisher-provided folder names (for example, `Simulation--2d-adcirc-simulation-of-tidal-flow-in-shinnecock-inlet-ny-parameterized-by-bottom-friction-coefficient`). The settings modules already resolve the nested `Model--adcirc-model/data` and `Output--processed-2d-adh-output/data` paths, so no manual path editing is required as long as the extracted folders remain intact. Keeping the raw, processed, and intermediate files separated by case prevents accidental cross-contamination of meshes, checkpoints, or scalers.

## Running Training / Evaluation

1. `cd cases/Shinnecock/scripts` (or `cases/RedRiver/scripts`).
2. Run the desired script:
   - `python mito_net.py` for MITONet training.
   - `python mito_optuna.py` to perform hyperparameter search before training.
   - Use the additional Shinnecock-specific entry points (`mio_net.py`, `mdo_net.py`, etc.) as needed.
3. Update configuration parameters inside the corresponding `settings/` directory.
4. Visualization notebooks live in `cases/<Case>/notebooks/Visualization.ipynb` once training artifacts exist.

## Version Control

This folder is the canonical Git repository for both cases. Commit changes from the repo root so diffs clearly show whether you touched shared data or a particular case under `cases/`.

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{riveracasillas2026neuraloperatoremulatorcoastal,
      title={A Neural Operator Emulator for Coastal and Riverine Shallow Water Dynamics}, 
      author={Peter Rivera-Casillas and Sourav Dutta and Shukai Cai and Mark Loveland and Kamaljyoti Nath and Khemraj Shukla and Corey Trahan and Jonghyun Lee and Matthew Farthing and Clint Dawson},
      year={2026},
      eprint={2502.14782},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2502.14782}, 
}
```
