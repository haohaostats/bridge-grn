# BRIDGE-GRN

BRIDGE-GRN is a role-aware graph learning framework for directed gene regulatory network inference from single-cell gene expression data and TF-target supervision.

This repository provides the BRIDGE-GRN model implementation, training and prediction workflows, transfer fine-tuning utilities, input-format documentation, benchmark data, and workflow examples.

## Features

- BRIDGE-GRN model implementation for directed TF-target prediction
- Role-specific TF and target embedding towers
- Cross-view contrastive regularization with edge perturbation
- Command-line training, prediction, and transfer fine-tuning
- Benchmark expression matrices, regulatory labels, and train/validation/test splits
- Workflow files derived from the mESC benchmark
- Documentation for expected input formats

## Installation

```bash
git clone https://github.com/haohaostats/bridge-grn.git
cd bridge-grn
pip install -r requirements.txt
pip install -e .
```

## Quick Start

The repository includes a workflow example under `examples/demo/`.

Train BRIDGE-GRN:

```bash
bridge-grn train \
  --expression examples/demo/expression.csv \
  --tf-list examples/demo/tf_list.txt \
  --train-edges examples/demo/train_edges.csv \
  --val-edges examples/demo/val_edges.csv \
  --test-edges examples/demo/test_edges.csv \
  --output-dir outputs/demo \
  --epochs 5 \
  --batch-size 32
```

Predict candidate TF-target edge scores:

```bash
bridge-grn predict \
  --checkpoint outputs/demo/bridge_grn_best.pt \
  --expression examples/demo/expression.csv \
  --tf-list examples/demo/tf_list.txt \
  --support-edges examples/demo/train_edges.csv \
  --query-edges examples/demo/query_edges.csv \
  --output outputs/demo/query_predictions.csv
```

Fine-tune from a source checkpoint:

```bash
bridge-grn transfer \
  --checkpoint outputs/source_run/bridge_grn_best.pt \
  --expression examples/demo/expression.csv \
  --tf-list examples/demo/tf_list.txt \
  --train-edges examples/demo/train_edges.csv \
  --val-edges examples/demo/val_edges.csv \
  --test-edges examples/demo/test_edges.csv \
  --output-dir outputs/transfer \
  --epochs 3 \
  --batch-size 32
```

## Input Format

BRIDGE-GRN expects:

- an expression matrix with genes as rows and cells/samples as columns;
- a TF list containing TF gene names;
- train, validation, and test edge files with `tf`, `target`, and `label` columns.

Both `tf` and `target` may be gene names or zero-based integer indices. Gene names are recommended for portability.

See `docs/input_format.md` for details.

## Data

The benchmark data used with BRIDGE-GRN are provided under `data/`:

- `data/benchmark_datasets/`: expression matrices, TF lists, target lists, regulatory labels, and prior-network files
- `data/splits/`: train, validation, and test edge splits
- `data/manifest.csv`: file-level manifest

CSV files are stored as `.csv.gz` files. They can be read directly with `pandas.read_csv`.

## Outputs

Training and transfer workflows write:

- `bridge_grn_best.pt`: model checkpoint selected by validation AUROC
- `metrics.json`: validation history and test metrics
- `test_predictions.csv`: test-edge scores

Prediction writes:

- a CSV containing query edges and BRIDGE-GRN scores
