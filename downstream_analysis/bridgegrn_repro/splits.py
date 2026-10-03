

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


def split_sampled_network(
    positives: pd.DataFrame,
    genes: list[str],
    tfs: list[str],
    density: float,
    seed: int,
    singleton_train_probability: float = 0.5,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    positives = positives[["TF", "Target"]].drop_duplicates().copy()
    positives["TF"] = positives.TF.astype(str).str.upper()
    positives["Target"] = positives.Target.astype(str).str.upper()
    positives = positives.loc[positives.TF.ne(positives.Target)]
    genes = np.asarray(sorted({str(x).upper() for x in genes}), dtype=object)
    tfs = np.asarray(sorted({str(x).upper() for x in tfs}), dtype=object)
    known = set(map(tuple, positives.itertuples(index=False, name=None)))
    positive_maps = {"train": [], "validation": [], "test": []}
    for tf, group in positives.groupby("TF", sort=True):
        targets = group.Target.to_numpy(object)
        rng.shuffle(targets)
        degree = len(targets)
        if degree == 1:
            key = "train" if rng.random() <= singleton_train_probability else "test"
            positive_maps[key].append((tf, targets[0]))
        elif degree == 2:
            positive_maps["train"].append((tf, targets[0]))
            positive_maps["test"].append((tf, targets[1]))
        else:
            train_end = degree * 2 // 3
            validation_end = train_end // 5
            positive_maps["validation"].extend((tf, x) for x in targets[:validation_end])
            positive_maps["train"].extend((tf, x) for x in targets[validation_end:train_end])
            positive_maps["test"].extend((tf, x) for x in targets[train_end:])
    negative_maps = {"train": [], "validation": [], "test": []}
    used = set()
    for split in ("train", "validation"):
        frame = pd.DataFrame(positive_maps[split], columns=["TF", "Target"])
        for tf, group in frame.groupby("TF", sort=True):
            candidates = [(tf, gene) for gene in genes if gene != tf and (tf, gene) not in known and (tf, gene) not in used]
            take = rng.choice(len(candidates), size=len(group), replace=False)
            selected = [candidates[index] for index in take]
            negative_maps[split].extend(selected)
            used.update(selected)
    test_positive_count = len(positive_maps["test"])
    test_negative_count = int(test_positive_count // density - test_positive_count)
    candidates = [(tf, gene) for tf in tfs for gene in genes
                  if tf != gene and (tf, gene) not in known and (tf, gene) not in used]
    take = rng.choice(len(candidates), size=test_negative_count, replace=False)
    negative_maps["test"] = [candidates[index] for index in take]
    output = {}
    for split in ("train", "validation", "test"):
        positive_frame = pd.DataFrame(positive_maps[split], columns=["TF", "Target"]).assign(Label=1)
        negative_frame = pd.DataFrame(negative_maps[split], columns=["TF", "Target"]).assign(Label=0)
        output[split] = pd.concat([positive_frame, negative_frame], ignore_index=True)
    return output
