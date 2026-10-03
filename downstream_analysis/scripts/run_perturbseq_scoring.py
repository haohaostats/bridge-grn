                     


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridgegrn_repro.network import (              
    degree_preserving_rewire,
    randomize_targets,
    reverse_edges,
    top_k_per_tf,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=ROOT / "data/perturbseq/response_profiles.npz")
    parser.add_argument("--network-dir", type=Path, default=ROOT / "data/networks")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/input_references")
    parser.add_argument("--universe-dir", type=Path, default=ROOT / "data/gene_universes")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/perturbseq")
    parser.add_argument("--top-k-per-tf", type=int, default=128)
    parser.add_argument("--response-set-size", type=int, default=100)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def evaluate(target, genes, delta, edges, response_set_size):
    universe = [gene for gene in genes if gene != target]
    gene_index = {gene: i for i, gene in enumerate(genes)}
    universe_delta = np.asarray([delta[gene_index[g]] for g in universe])
    response_genes = set(np.asarray(genes, dtype=object)[np.argsort(-delta, kind="stable")[:response_set_size]])
    response_genes.discard(target)
    response = np.asarray([gene in response_genes for gene in universe], dtype=bool)
    background = response.mean()
    outgoing = edges.loc[edges.TF.eq(target) & edges.Target.isin(universe), ["Target", "Score"]]
    outgoing = outgoing.sort_values(["Score", "Target"], ascending=[False, True]).drop_duplicates("Target")
    values = np.zeros(len(universe), dtype=float)
    location = {gene: i for i, gene in enumerate(universe)}
    for row in outgoing.itertuples(index=False):
        values[location[row.Target]] = float(row.Score)
    active = values != 0
    response_fraction = response[active].mean() if active.any() else 0.0
    enrichment = response_fraction / background if background > 0 else np.nan
    rho = float(spearmanr(values, universe_delta).statistic) if np.unique(values).size > 1 else 0.0
    return {
        "response_enrichment": enrichment,
        "spearman_abs_delta": rho,
        "nonzero_edges": int(active.sum()),
        "response_hits": int(response[active].sum()),
        "overlap_genes": len(universe),
    }


def load_reference(path):
    frame = pd.read_csv(path)
    if "OriginalReferenceScore" in frame.columns:
        frame = frame.rename(columns={"OriginalReferenceScore": "Score"})
    else:
        frame["Score"] = 1.0
    return frame[["TF", "Target", "Score"]]


def match_outdegree(reference, inferred):
    counts = inferred.groupby("TF").size().to_dict()
    groups = []
    ordered = reference.sort_values(["TF", "Score", "Target"], ascending=[True, False, True], kind="stable")
    for tf, group in ordered.groupby("TF", sort=False):
        number = int(counts.get(tf, 0))
        if number:
            groups.append(group.head(number))
    if not groups:
        return reference.iloc[:0][["TF", "Target", "Score"]]
    return pd.concat(groups, ignore_index=True)[["TF", "Target", "Score"]]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.profile) as bundle:
        all_genes = bundle["genes"].astype(str)
        target_names = bundle["target_names"].astype(str)
        target_counts = dict(zip(target_names, bundle["target_counts"].astype(int)))
        deltas = dict(zip(target_names, bundle["deltas"]))
        control_count = int(bundle["control_count"])
    all_index = {gene: i for i, gene in enumerate(all_genes)}

    rows = []
    for family in ("mHSC-E", "mHSC-GM"):
        requested = pd.read_csv(args.universe_dir / f"{family}.csv").gene.astype(str).str.upper()
        genes = [gene for gene in requested if gene in all_index]
        positions = np.asarray([all_index[gene] for gene in genes])
        for sample in range(1, 6):
            inferred = top_k_per_tf(
                pd.read_csv(args.network_dir / f"{family}_sample{sample}_top128.csv.gz"),
                args.top_k_per_tf,
            )
            reference = load_reference(
                args.reference_dir / f"{family}_sample{sample}_train_positive_reference.csv.gz"
            )
            matched_targets = [target for target in target_names if (inferred.TF == target).any()]
            seed = args.seed + sample + (100 if family == "mHSC-GM" else 0)
            conditions = {
                "BRIDGE-GRN": inferred,
                "randomized-target": randomize_targets(inferred, genes, seed),
                "degree-preserving": degree_preserving_rewire(inferred, seed),
                "reversed": reverse_edges(inferred),
                "input-reference-full": reference,
                "input-reference-degree-matched": match_outdegree(reference, inferred),
            }
            for target in matched_targets:
                delta = deltas[target][positions]
                for condition, edges in conditions.items():
                    metric = evaluate(target, genes, delta, edges, args.response_set_size)
                    rows.append({"family": family, "sample": sample, "target": target,
                                 "perturbed_cells": target_counts[target],
                                 "control_cells": control_count, "condition": condition, **metric})
            print(f"completed {family} sample={sample}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "per_target_metrics.csv", index=False)
    summaries, comparisons = [], []
    for threshold in args.thresholds:
        subset = detail.loc[detail.perturbed_cells.ge(threshold)]
        for condition, group in subset.groupby("condition"):
            summaries.append({"threshold": threshold, "condition": condition,
                              "matched_units": len(group),
                              "independent_family_target_groups": group[["family", "target"]].drop_duplicates().shape[0],
                              "response_enrichment_mean": group.response_enrichment.mean(),
                              "spearman_mean": group.spearman_abs_delta.mean()})
        bridge = subset.loc[subset.condition.eq("BRIDGE-GRN")]
        for condition in sorted(set(subset.condition) - {"BRIDGE-GRN"}):
            comparator = subset.loc[subset.condition.eq(condition)]
            joined = bridge.merge(comparator, on=["family", "sample", "target"], suffixes=("_model", "_control"), validate="one_to_one")
            for metric in ("response_enrichment", "spearman_abs_delta"):
                difference = joined[f"{metric}_model"] - joined[f"{metric}_control"]
                comparisons.append({"threshold": threshold, "control": condition,
                                    "metric": metric, "matched_units": len(joined),
                                    "mean_difference": difference.mean(),
                                    "paired_t_p": ttest_rel(joined[f"{metric}_model"], joined[f"{metric}_control"]).pvalue})
    pd.DataFrame(summaries).to_csv(args.output_dir / "condition_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
