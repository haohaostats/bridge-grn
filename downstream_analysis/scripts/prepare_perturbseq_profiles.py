                     


from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def mean_log_cpm(adata, rows, chunk_size=512):
    total = np.zeros(adata.n_vars, dtype=np.float64)
    for start in range(0, len(rows), chunk_size):
        matrix = adata.X[rows[start:start + chunk_size], :].tocsr().astype(np.float64)
        depth = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(1e4, depth, out=np.zeros_like(depth), where=depth > 0)
        matrix = matrix.multiply(scale[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
        total += np.asarray(matrix.sum(axis=0)).ravel()
    return total / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/perturbseq/response_profiles.npz"))
    parser.add_argument("--targets", nargs="+", default=["KDM6A", "PHF10", "PRMT5", "RCOR1", "SMC2", "SPI1", "STAG2"])
    parser.add_argument("--perturbation-column", default="perturbation")
    parser.add_argument("--control-label", default="CONTROL")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.h5ad, backed="r")
    labels = adata.obs[args.perturbation_column].astype(str).str.upper()
    genes = adata.var_names.astype(str).str.upper().to_numpy()
    if len(set(genes)) != len(genes):
        raise ValueError("Gene names must be unique")
    control_rows = np.flatnonzero(labels.eq(args.control_label.upper()).to_numpy())
    control_mean = mean_log_cpm(adata, control_rows)
    target_counts, deltas = [], []
    for target in map(str.upper, args.targets):
        rows = np.flatnonzero(labels.eq(target).to_numpy())
        if len(rows) == 0:
            raise ValueError(f"No cells found for perturbation {target}")
        target_counts.append(len(rows))
        deltas.append(np.abs(mean_log_cpm(adata, rows) - control_mean))
        print(target, len(rows), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        genes=genes,
        target_names=np.asarray([x.upper() for x in args.targets]),
        target_counts=np.asarray(target_counts, dtype=np.int32),
        control_count=np.asarray(len(control_rows), dtype=np.int32),
        deltas=np.stack(deltas),
    )


if __name__ == "__main__":
    main()
