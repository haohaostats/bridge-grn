                     


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridgegrn_repro.splits import split_sampled_network, split_specific_hard_negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-edges", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--tfs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--mode", choices=["specific", "sampled"], default="specific")
    parser.add_argument("--density", type=float)
    args = parser.parse_args()
    positives = pd.read_csv(args.positive_edges)
    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).tolist()
    tfs = pd.read_csv(args.tfs).iloc[:, 0].astype(str).tolist()
    if args.mode == "specific":
        splits = split_specific_hard_negatives(positives, genes, tfs, args.seed, args.train_ratio)
    else:
        if args.density is None:
            raise ValueError("--density is required in sampled mode")
        splits = split_sampled_network(positives, genes, tfs, args.density, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        counts = frame.Label.value_counts().to_dict()
        print(name, counts, flush=True)


if __name__ == "__main__":
    main()
