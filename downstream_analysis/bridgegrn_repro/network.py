

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED = ("TF", "Target", "Score")


def clean_edges(edges: pd.DataFrame) -> pd.DataFrame:

    missing = set(REQUIRED) - set(edges.columns)
    if missing:
        raise ValueError(f"Missing edge columns: {sorted(missing)}")
    out = edges.loc[:, REQUIRED].copy()
    out["TF"] = out["TF"].astype(str).str.upper()
    out["Target"] = out["Target"].astype(str).str.upper()
    out["Score"] = pd.to_numeric(out["Score"], errors="raise")
    out = out.loc[out.TF.ne(out.Target)]
    return out.sort_values("Score", ascending=False, kind="stable").drop_duplicates(
        ["TF", "Target"], keep="first"
    ).reset_index(drop=True)


def top_k_per_tf(edges: pd.DataFrame, k: int) -> pd.DataFrame:

    edges = clean_edges(edges)
    return (
        edges.sort_values(["TF", "Score", "Target"], ascending=[True, False, True], kind="stable")
        .groupby("TF", sort=False, group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )


def build_adjacency(edges: pd.DataFrame, genes: list[str]) -> sparse.csr_matrix:

    edges = clean_edges(edges)
    genes = [str(x).upper() for x in genes]
    index = {gene: i for i, gene in enumerate(genes)}
    kept = edges.loc[edges.TF.isin(index) & edges.Target.isin(index)]
    row = kept.TF.map(index).to_numpy()
    col = kept.Target.map(index).to_numpy()
    matrix = sparse.csr_matrix(
        (kept.Score.to_numpy(float), (row, col)), shape=(len(genes), len(genes))
    )
    matrix.sum_duplicates()
    return matrix


def row_normalize(adjacency: sparse.spmatrix) -> sparse.csr_matrix:

    adjacency = adjacency.tocsr().astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree != 0)
    return sparse.diags(inverse) @ adjacency


def propagation_features(expression: np.ndarray, adjacency: sparse.spmatrix) -> np.ndarray:

    expression = np.asarray(expression, dtype=np.float64)
    adjacency = row_normalize(adjacency)
    if expression.ndim != 2 or expression.shape[1] != adjacency.shape[0]:
        raise ValueError("Expression columns and adjacency nodes must match")
    outgoing = expression @ adjacency.T
    incoming = expression @ adjacency
    return np.concatenate([np.asarray(outgoing), np.asarray(incoming)], axis=1)


def reverse_edges(edges: pd.DataFrame) -> pd.DataFrame:

    edges = clean_edges(edges)
    return clean_edges(edges.rename(columns={"TF": "Target", "Target": "TF"}))


def randomize_targets(
    edges: pd.DataFrame, genes: list[str], seed: int
) -> pd.DataFrame:

    edges = clean_edges(edges)
    genes = np.asarray(sorted({str(x).upper() for x in genes}), dtype=object)
    rng = np.random.default_rng(seed)
    result = []
    for tf, group in edges.groupby("TF", sort=True):
        candidates = genes[genes != tf]
        if len(group) > len(candidates):
            raise ValueError(f"Out-degree for {tf} exceeds target universe")
        targets = rng.choice(candidates, size=len(group), replace=False)
        randomized = group.copy()
        randomized["Target"] = targets
        result.append(randomized)
    return clean_edges(pd.concat(result, ignore_index=True))


def degree_preserving_rewire(
    edges: pd.DataFrame, seed: int, swaps_per_edge: int = 10
) -> pd.DataFrame:





    frame = clean_edges(edges).copy()
    pairs = list(map(tuple, frame[["TF", "Target"]].itertuples(index=False, name=None)))
    pair_set = set(pairs)
    rng = np.random.default_rng(seed)
    successes = 0
    attempts = 0
    target_successes = swaps_per_edge * len(pairs)
    max_attempts = max(100, target_successes * 30)
    while successes < target_successes and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(pairs), size=2, replace=False)
        a, b = pairs[i]
        c, d = pairs[j]
        new_i, new_j = (a, d), (c, b)
        if a == d or c == b or new_i == new_j:
            continue
        if new_i in pair_set or new_j in pair_set:
            continue
        pair_set.remove(pairs[i]); pair_set.remove(pairs[j])
        pairs[i], pairs[j] = new_i, new_j
        pair_set.add(new_i); pair_set.add(new_j)
        successes += 1
    frame[["TF", "Target"]] = pd.DataFrame(pairs, index=frame.index)
    frame.attrs["successful_swaps"] = successes
    frame.attrs["requested_swaps"] = target_successes
    return frame
