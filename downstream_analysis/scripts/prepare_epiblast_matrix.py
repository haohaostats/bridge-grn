                     


from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import save_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-tar", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--network-expression", type=Path, required=True,
                        help="BL--ExpressionData.csv defining the model gene universe")
    parser.add_argument("--output-dir", type=Path, default=Path("data/epiblast"))
    args = parser.parse_args()

    stages = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75"]
    meta = pd.read_csv(args.cell_metadata, sep="\t", low_memory=False)
    selected = meta.loc[
        meta.celltype.eq("Epiblast") & ~meta.doublet & ~meta.stripped & meta.stage.isin(stages)
    ].copy()
    atlas_genes = pd.read_csv(args.gene_metadata, sep="\t", header=None, names=["ensembl", "symbol"])
    model_genes = pd.read_csv(args.network_expression, usecols=[0]).iloc[:, 0].astype(str).str.upper()
    atlas_symbols = set(atlas_genes.symbol.astype(str).str.upper())
    selected_genes = pd.Index([gene for gene in model_genes if gene in atlas_symbols]).drop_duplicates()
    gene_lookup = {gene: i for i, gene in enumerate(selected_genes)}
    gene_map = np.full(len(atlas_genes) + 1, -1, dtype=np.int32)
    for old, symbol in enumerate(atlas_genes.symbol.astype(str), 1):
        gene_map[old] = gene_lookup.get(symbol.upper(), -1)
    cell_map = np.full(len(meta) + 1, -1, dtype=np.int32)
    for new, old in enumerate(selected.index, 1):
        cell_map[old + 1] = new - 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "cells.csv", index=False)
    pd.DataFrame({"gene_symbol": selected_genes}).to_csv(args.output_dir / "genes.csv", index=False)
    library_totals = np.zeros(len(selected), dtype=np.int64)
    with tempfile.NamedTemporaryFile(dir=args.output_dir, delete=False) as temporary:
        body = Path(temporary.name)
        count = 0
        with tarfile.open(args.atlas_tar, "r|gz") as archive:
            for member in archive:
                if not member.name.endswith("/raw_counts.mtx"):
                    continue
                raw = archive.extractfile(member)
                header = raw.readline()
                if not header.startswith(b"%%MatrixMarket matrix coordinate integer general"):
                    raise ValueError("Unexpected Matrix Market header")
                raw.readline()              
                for line in raw:
                    first = line.find(b" "); second = line.find(b" ", first + 1)
                    cell = cell_map[int(line[first + 1:second])]
                    if cell < 0:
                        continue
                    value = int(line[second + 1:])
                    library_totals[cell] += value
                    gene = gene_map[int(line[:first])]
                    if gene < 0:
                        continue
                    temporary.write(f"{cell + 1} {gene + 1} ".encode() + line[second + 1:])
                    count += 1
                break
    matrix_market = args.output_dir / "filtered_counts.mtx"
    with matrix_market.open("wb") as output, body.open("rb") as input_file:
        output.write(b"%%MatrixMarket matrix coordinate integer general\n")
        output.write(f"{len(selected)} {len(selected_genes)} {count}\n".encode())
        for chunk in iter(lambda: input_file.read(4 * 1024 * 1024), b""):
            output.write(chunk)
    matrix = mmread(matrix_market).tocsr(); matrix.sum_duplicates()
    save_npz(args.output_dir / "raw_counts_cells_by_genes.npz", matrix, compressed=True)
    np.save(args.output_dir / "total_counts_per_cell_all_genes.npy", library_totals)
    body.unlink(); matrix_market.unlink()


if __name__ == "__main__":
    main()
