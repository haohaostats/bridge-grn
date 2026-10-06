                     


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import ttest_rel
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridgegrn_repro.network import (              
    build_adjacency,
    degree_preserving_rewire,
    propagation_features,
    randomize_targets,
    reverse_edges,
    top_k_per_tf,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/epiblast")
    parser.add_argument("--network-dir", type=Path, default=ROOT / "data/networks")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/input_references")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/epiblast")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2025, 2031)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--training-cells-per-state", type=int, default=64)
    parser.add_argument("--selected-genes", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--normalization", choices=["raw", "logcpm"], default="logcpm")
    return parser.parse_args()


def load_expression(data_dir: Path, normalization: str):
    matrix = load_npz(data_dir / "raw_counts_cells_by_genes.npz").tocsr().astype(np.float64)
    if normalization == "logcpm":
        depth = np.load(data_dir / "total_counts_per_cell_all_genes.npy")
        scale = np.divide(1e4, depth, out=np.zeros_like(depth, dtype=float), where=depth > 0)
        matrix = matrix.multiply(scale[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
    expression = matrix.toarray()
    genes = pd.read_csv(data_dir / "genes.csv").gene_symbol.astype(str).str.upper().tolist()
    cells = pd.read_csv(data_dir / "cells.csv")
    stages = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75"]
    labels = cells.stage.map({stage: i for i, stage in enumerate(stages)}).to_numpy()
    if np.any(pd.isna(labels)):
        raise ValueError("Unexpected Epiblast stage label")
    return expression, genes, labels.astype(int), stages


def score_features(train, test, y_train, y_test, n_classes):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    model = RidgeClassifier(alpha=1.0, class_weight="balanced")
    model.fit(train, y_train)
    prediction = model.predict(test)
    per_class = f1_score(y_test, prediction, labels=range(n_classes), average=None, zero_division=0)
    return {
        "macro_f1": float(per_class.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "rare_state_f1": float(per_class.min()),
        "ordinal_mae": float(np.mean(np.abs(y_test - prediction))),
    }


def reference_edges(path, cap, seed):
    frame = pd.read_csv(path)[["TF", "Target"]].drop_duplicates()
    frame["TF"] = frame.TF.astype(str).str.upper()
    frame["Target"] = frame.Target.astype(str).str.upper()
    frame = frame.loc[frame.TF.ne(frame.Target)]
    if cap is not None:
        rng = np.random.default_rng(seed)
        groups = []
        for _, group in frame.groupby("TF", sort=True):
            if len(group) > cap:
                group = group.iloc[np.sort(rng.choice(len(group), cap, replace=False))]
            groups.append(group)
        frame = pd.concat(groups, ignore_index=True)
    frame["Score"] = 1.0
    return frame


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expression, genes, labels, stages = load_expression(args.data_dir, args.normalization)
    networks = {}
    references = {}
    for sample in range(1, 6):
        path = args.network_dir / f"mESC_sample{sample}_top128.csv.gz"
        networks[sample] = top_k_per_tf(pd.read_csv(path), args.top_k)
        reference_path = args.reference_dir / f"mESC_sample{sample}_train_positive_reference.csv.gz"
        references[sample] = {
            "input-reference-full": reference_edges(reference_path, None, 9000 + sample),
            "input-reference-cap128": reference_edges(reference_path, args.top_k, 9000 + sample),
        }

    rows = []
    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_pool, test_index) in enumerate(cv.split(expression, labels)):
            rng = np.random.default_rng(seed * 100 + fold)
            chosen = np.concatenate([
                rng.choice(train_pool[labels[train_pool] == state],
                           size=min(args.training_cells_per_state, np.sum(labels[train_pool] == state)),
                           replace=False)
                for state in range(len(stages))
            ])
            variance = expression[chosen].var(axis=0)
            selected = np.argsort(-variance, kind="stable")[: args.selected_genes]
            selected_genes = [genes[i] for i in selected]
            train_x = expression[chosen][:, selected]
            test_x = expression[test_index][:, selected]
            base = score_features(train_x, test_x, labels[chosen], labels[test_index], len(stages))
            rows.append({"seed": seed, "fold": fold, "sample": 0,
                         "condition": "expression-only", **base})

            for sample, inferred in networks.items():
                control_seed = seed * 10000 + fold * 100 + sample
                conditions = {
                    "BRIDGE-GRN": inferred,
                    "randomized-target": randomize_targets(inferred, genes, control_seed),
                    "degree-preserving": degree_preserving_rewire(inferred, control_seed),
                    "reversed": reverse_edges(inferred),
                    **references[sample],
                }
                for condition, edges in conditions.items():
                    adjacency = build_adjacency(edges, selected_genes)
                    train_features = propagation_features(train_x, adjacency)
                    test_features = propagation_features(test_x, adjacency)
                    metrics = score_features(
                        train_features, test_features, labels[chosen], labels[test_index], len(stages)
                    )
                    rows.append({"seed": seed, "fold": fold, "sample": sample,
                                 "condition": condition, **metrics})
            print(f"completed seed={seed} fold={fold}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "fold_metrics.csv", index=False)
                                                                             
    expr = detail.loc[detail.condition.eq("expression-only")].groupby(["seed", "fold"], as_index=False).mean(numeric_only=True)
    network = detail.loc[~detail.condition.eq("expression-only")]
    seed_means = network.groupby(["seed", "condition"], as_index=False).mean(numeric_only=True)
    expr_seed = expr.groupby("seed", as_index=False).mean(numeric_only=True).assign(condition="expression-only")
    seed_means = pd.concat([seed_means, expr_seed], ignore_index=True)
    seed_means.to_csv(args.output_dir / "seed_means.csv", index=False)
    summary = seed_means.groupby("condition")[["macro_f1", "balanced_accuracy", "rare_state_f1", "ordinal_mae"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "condition_summary.csv")

    bridge = seed_means.loc[seed_means.condition.eq("BRIDGE-GRN")].set_index("seed")
    comparisons = []
    for condition in sorted(set(seed_means.condition) - {"BRIDGE-GRN"}):
        comparator = seed_means.loc[seed_means.condition.eq(condition)].set_index("seed")
        common = bridge.index.intersection(comparator.index)
        for metric in ["macro_f1", "balanced_accuracy", "rare_state_f1", "ordinal_mae"]:
            difference = bridge.loc[common, metric] - comparator.loc[common, metric]
            comparisons.append({"comparator": condition, "metric": metric,
                                "n_seeds": len(common), "mean_difference": difference.mean(),
                                "paired_t_p": ttest_rel(bridge.loc[common, metric], comparator.loc[common, metric]).pvalue})
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
