import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgegrn_repro.network import (              
    build_adjacency,
    degree_preserving_rewire,
    propagation_features,
    randomize_targets,
    reverse_edges,
)
from bridgegrn_repro.splits import split_sampled_network, split_specific_hard_negatives


def example_edges():
    return pd.DataFrame({
        "TF": ["A", "A", "B", "B", "C"],
        "Target": ["B", "C", "C", "D", "D"],
        "Score": [0.9, 0.8, 0.7, 0.6, 0.5],
    })


def test_propagation_shape_and_direction():
    genes = ["A", "B", "C", "D"]
    adjacency = build_adjacency(example_edges(), genes)
    expression = np.eye(4)
    features = propagation_features(expression, adjacency)
    assert features.shape == (4, 8)
    assert np.isfinite(features).all()


def test_controls_preserve_required_properties():
    edges = example_edges()
    genes = ["A", "B", "C", "D", "E", "F"]
    randomized = randomize_targets(edges, genes, seed=7)
    assert edges.groupby("TF").size().to_dict() == randomized.groupby("TF").size().to_dict()
    rewired = degree_preserving_rewire(edges, seed=7, swaps_per_edge=2)
    assert edges.groupby("TF").size().to_dict() == rewired.groupby("TF").size().to_dict()
    assert edges.groupby("Target").size().to_dict() == rewired.groupby("Target").size().to_dict()
    reversed_edges = reverse_edges(edges)
    assert set(map(tuple, reversed_edges[["TF", "Target"]].to_numpy())) == {
        (target, tf) for tf, target in edges[["TF", "Target"]].itertuples(index=False, name=None)
    }


def test_split_is_deterministic_and_disjoint():
    positives = pd.DataFrame({
        "TF": ["A"] * 4 + ["B"] * 3,
        "Target": ["B", "C", "D", "E", "A", "C", "D"],
    })
    genes = ["A", "B", "C", "D", "E", "F"]
    first = split_specific_hard_negatives(positives, genes, ["A", "B"], seed=11)
    second = split_specific_hard_negatives(positives, genes, ["A", "B"], seed=11)
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    labeled = {}
    for key, frame in first.items():
        pairs = set(map(tuple, frame[["TF", "Target"]].itertuples(index=False, name=None)))
        for previous, previous_pairs in labeled.items():
            assert pairs.isdisjoint(previous_pairs), (key, previous)
        labeled[key] = pairs
        known = set(map(tuple, positives.itertuples(index=False, name=None)))
        negatives = set(map(tuple, frame.loc[frame.Label.eq(0), ["TF", "Target"]].itertuples(index=False, name=None)))
        assert negatives.isdisjoint(known)


def test_sampled_split_is_deterministic_and_disjoint():
    positives = pd.DataFrame({
        "TF": ["A"] * 4 + ["B"] * 3,
        "Target": ["B", "C", "D", "E", "A", "C", "D"],
    })
    genes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    first = split_sampled_network(positives, genes, ["A", "B"], 0.5, 13)
    second = split_sampled_network(positives, genes, ["A", "B"], 0.5, 13)
    known = set(map(tuple, positives.itertuples(index=False, name=None)))
    used = set()
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
        pairs = set(map(tuple, first[key][["TF", "Target"]].itertuples(index=False, name=None)))
        assert pairs.isdisjoint(used)
        used |= pairs
        negatives = set(map(tuple, first[key].loc[first[key].Label.eq(0), ["TF", "Target"]].itertuples(index=False, name=None)))
        assert negatives.isdisjoint(known)
