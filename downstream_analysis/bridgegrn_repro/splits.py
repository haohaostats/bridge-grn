

from __future__ import annotations

import numpy as np
import pandas as pd


def _positive_split(edges: pd.DataFrame, ratio: float, singleton_train_probability: float, rng):
    maps = {"train": [], "validation": [], "test": []}
    for tf, group in edges.groupby("TF", sort=True):
        targets = group.Target.drop_duplicates().to_numpy(object)
        rng.shuffle(targets)
        degree = len(targets)
        if degree == 1:
            key = "train" if rng.random() <= singleton_train_probability else "test"
            maps[key].append((tf, targets[0]))
        elif degree == 2:
            maps["train"].append((tf, targets[0]))
            maps["test"].append((tf, targets[1]))
        else:
            cut1 = int(degree * ratio)
            cut2 = int(degree * (ratio + 0.10))
            maps["train"].extend((tf, x) for x in targets[:cut1])
            maps["validation"].extend((tf, x) for x in targets[cut1:cut2])
            maps["test"].extend((tf, x) for x in targets[cut2:])
    return maps


def split_specific_hard_negatives(
    positives: pd.DataFrame,
    genes: list[str],
    tfs: list[str],
    seed: int,
    ratio: float = 0.67,
    singleton_train_probability: float = 0.5,
) -> dict[str, pd.DataFrame]:







    rng = np.random.default_rng(seed)
    positives = positives[["TF", "Target"]].drop_duplicates().copy()
    positives["TF"] = positives.TF.astype(str).str.upper()
    positives["Target"] = positives.Target.astype(str).str.upper()
    positives = positives.loc[positives.TF.ne(positives.Target)]
    genes = np.asarray(sorted({str(x).upper() for x in genes}), dtype=object)
    tfs = sorted({str(x).upper() for x in tfs})
    pos_maps = _positive_split(positives, ratio, singleton_train_probability, rng)
    neg_maps = {"train": [], "validation": [], "test": []}
    known = positives.groupby("TF").Target.apply(set).to_dict()
    for tf in tfs:
        negatives = np.asarray([g for g in genes if g != tf and g not in known.get(tf, set())], dtype=object)
        rng.shuffle(negatives)
        cut1 = int(len(negatives) * ratio)
        cut2 = int(len(negatives) * (ratio + 0.10))
        neg_maps["train"].extend((tf, x) for x in negatives[:cut1])
        neg_maps["validation"].extend((tf, x) for x in negatives[cut1:cut2])
        neg_maps["test"].extend((tf, x) for x in negatives[cut2:])
    output = {}
    for key in ("train", "validation", "test"):
        pos = pd.DataFrame(pos_maps[key], columns=["TF", "Target"]).assign(Label=1)
        neg = pd.DataFrame(neg_maps[key], columns=["TF", "Target"]).assign(Label=0)
        output[key] = pd.concat([pos, neg], ignore_index=True)
    return output


def sample_unlabeled_negatives(
    positives: pd.DataFrame,
    genes: list[str],
    tfs: list[str],
    number: int,
    seed: int,
    forbidden: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:

    positives = positives[["TF", "Target"]].astype(str).apply(lambda x: x.str.upper())
    excluded = set(map(tuple, positives.itertuples(index=False, name=None)))
    excluded |= forbidden or set()
    candidates = [(str(tf).upper(), str(g).upper()) for tf in tfs for g in genes
                  if str(tf).upper() != str(g).upper() and (str(tf).upper(), str(g).upper()) not in excluded]
    if number > len(candidates):
        raise ValueError("Requested more negatives than available unlabeled pairs")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(candidates), size=number, replace=False)
    return pd.DataFrame([candidates[i] for i in take], columns=["TF", "Target"]).assign(Label=0)
