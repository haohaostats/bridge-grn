import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, t, ttest_rel


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    margin = t.ppf(0.975, len(values) - 1) * sem(values)
    return values.mean() - margin, values.mean() + margin


def epiblast_table(output_dir):
    frame = pd.read_csv(output_dir / "epiblast" / "seed_means.csv")
    metrics = ["macro_f1", "balanced_accuracy", "rare_state_f1", "ordinal_mae"]
    rows = []
    bridge = frame.loc[frame.condition.eq("BRIDGE-GRN")].set_index("seed")
    for condition in frame.condition.drop_duplicates():
        group = frame.loc[frame.condition.eq(condition)].set_index("seed")
        common = bridge.index.intersection(group.index)
        row = {"condition": condition, "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            low, high = interval(values)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            if condition == "BRIDGE-GRN":
                row[f"{metric}_difference"] = np.nan
                row[f"{metric}_difference_ci_low"] = np.nan
                row[f"{metric}_difference_ci_high"] = np.nan
                row[f"{metric}_paired_p"] = np.nan
            else:
                differences = bridge.loc[common, metric].to_numpy(float) - group.loc[common, metric].to_numpy(float)
                diff_low, diff_high = interval(differences)
                row[f"{metric}_difference"] = differences.mean()
                row[f"{metric}_difference_ci_low"] = diff_low
                row[f"{metric}_difference_ci_high"] = diff_high
                row[f"{metric}_paired_p"] = ttest_rel(bridge.loc[common, metric], group.loc[common, metric]).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def perturbseq_table(output_dir):
    frame = pd.read_csv(output_dir / "perturbseq" / "per_target_metrics.csv")
    metrics = ["response_enrichment", "spearman_abs_delta"]
    rows = []
    for threshold in (100, 500):
        subset = frame.loc[frame.perturbed_cells.ge(threshold)]
        bridge = subset.loc[subset.condition.eq("BRIDGE-GRN")]
        for condition in subset.condition.drop_duplicates():
            group = subset.loc[subset.condition.eq(condition)]
            row = {
                "threshold": threshold,
                "condition": condition,
                "matched_units": len(group),
                "independent_family_target_groups": group[["family", "target"]].drop_duplicates().shape[0],
            }
            for metric in metrics:
                values = group[metric].to_numpy(float)
                low, high = interval(values)
                row[f"{metric}_mean"] = np.nanmean(values)
                row[f"{metric}_sd"] = np.nanstd(values, ddof=1)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                if condition == "BRIDGE-GRN":
                    row[f"{metric}_difference"] = np.nan
                    row[f"{metric}_difference_ci_low"] = np.nan
                    row[f"{metric}_difference_ci_high"] = np.nan
                    row[f"{metric}_paired_p"] = np.nan
                else:
                    joined = bridge.merge(group, on=["family", "sample", "target"], suffixes=("_model", "_control"), validate="one_to_one")
                    model = joined[f"{metric}_model"].to_numpy(float)
                    control = joined[f"{metric}_control"].to_numpy(float)
                    valid = np.isfinite(model) & np.isfinite(control)
                    differences = model[valid] - control[valid]
                    diff_low, diff_high = interval(differences)
                    row[f"{metric}_difference"] = differences.mean()
                    row[f"{metric}_difference_ci_low"] = diff_low
                    row[f"{metric}_difference_ci_high"] = diff_high
                    row[f"{metric}_paired_p"] = ttest_rel(model[valid], control[valid]).pvalue
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    epiblast_table(args.output_dir).to_csv(args.output_dir / "epiblast_summary.csv", index=False, float_format="%.10g")
    perturbseq_table(args.output_dir).to_csv(args.output_dir / "perturbseq_summary.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    main()
