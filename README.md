# BRIDGE-GRN

Code and benchmark data for BRIDGE-GRN, a role-factorized graph-learning model for directed gene regulatory network prediction.

## Installation

Use Python 3.10. Download this repository, open a terminal in its root directory, and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

A dedicated Python environment is recommended. The script uses an NVIDIA GPU when CUDA is available to PyTorch and otherwise runs on the CPU.

## Run

```bash
python train_bridge_grn.py --cell_type mESC --sample sample1
```

This runs training and test evaluation on the Specific TFs+1000 setting. Use `--sample sample1` through `--sample sample5` to select a predefined split. Available cell types are `mDC`, `mESC`, `mHSC-E`, `mHSC-GM`, `mHSC-L`, `hESC`, and `hHEP`.

```bash
python train_bridge_grn.py --help
```

The default run uses 20 epochs and selects the checkpoint by validation AUROC. Test AUROC and AUPRC are printed to the terminal. The checkpoint is saved as `model/<cell_type>_bridgegrn_best.pkl`; a subsequent run for the same cell type overwrites that checkpoint.

## Data

- `Benchmark Datasets/`: expression matrices and gene/TF information.
- `Data/`: predefined training, validation, and test edge splits.

The training entry point reads the Specific TFs+1000 files directly from these directories.

## Archived version

BRIDGE-GRN v1.0.0: https://doi.org/10.5281/zenodo.24185862

## License

MIT License.
