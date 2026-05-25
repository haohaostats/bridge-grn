from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.tsv', '.txt'}:
        try:
            return pd.read_csv(path, sep='\t', index_col=0)
        except Exception:
            return pd.read_csv(path, sep='\t')
    return pd.read_csv(path, index_col=0)


def load_expression(expression_path: str | Path, normalize: bool = True, svd_dim: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
    expr_df = _read_table(expression_path)
    values = expr_df.values.astype(np.float32)
    if normalize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values.T).T
    if svd_dim and svd_dim > 0:
        svd = TruncatedSVD(n_components=svd_dim, random_state=2025)
        values = svd.fit_transform(values).astype(np.float32)
    return expr_df, values.astype(np.float32)


def load_tf_list(tf_list_path: str | Path) -> List[str]:
    path = Path(tf_list_path)
    if path.suffix.lower() == '.txt':
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    df = pd.read_csv(path)
    lower_map = {str(c).lower(): c for c in df.columns}
    if 'tf' in lower_map:
        return df[lower_map['tf']].astype(str).tolist()
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def build_gene_index(expr_df: pd.DataFrame) -> Dict[str, int]:
    return {str(gene): idx for idx, gene in enumerate(expr_df.index.astype(str))}


def _resolve_node(value, gene_to_idx: Dict[str, int]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    value_str = str(value)
    if value_str in gene_to_idx:
        return gene_to_idx[value_str]
    try:
        return int(value_str)
    except ValueError as exc:
        raise KeyError(f"Gene or node '{value}' not found in expression index.") from exc


def load_edge_table(edge_path: str | Path, gene_to_idx: Dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(edge_path)
    lower_map = {c.lower(): c for c in df.columns}
    if not {'tf', 'target'}.issubset(lower_map):
        raise ValueError(f"Edge file must contain columns tf,target; got {list(df.columns)}")
    tf_col = lower_map['tf']
    target_col = lower_map['target']
    label_col = lower_map.get('label')
    out = pd.DataFrame()
    out['tf'] = df[tf_col].map(lambda x: _resolve_node(x, gene_to_idx)).astype(np.int64)
    out['target'] = df[target_col].map(lambda x: _resolve_node(x, gene_to_idx)).astype(np.int64)
    out['label'] = df[label_col].astype(np.float32) if label_col else 0.0
    return out


def adj_to_sparse_tensor(adj: sp.spmatrix) -> torch.Tensor:
    coo = adj.tocoo()
    idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(idx, values, coo.shape)


class EdgeDataset(Dataset):
    def __init__(self, edge_array: np.ndarray):
        super().__init__()
        self.edge_array = edge_array.astype(np.float32)

    def __getitem__(self, idx: int):
        pair = self.edge_array[idx, :2].astype(np.int64)
        label = np.float32(self.edge_array[idx, -1])
        return pair, label

    def __len__(self) -> int:
        return len(self.edge_array)


def build_support_graph(edge_df: pd.DataFrame, num_genes: int, tf_indices: Iterable[int], loop: bool = False, direction: bool = False) -> sp.dok_matrix:
    adj = sp.dok_matrix((num_genes, num_genes), dtype=np.float32)
    tf_set = set(int(x) for x in tf_indices)
    for tf, target, label in edge_df[['tf', 'target', 'label']].itertuples(index=False):
        if float(label) != 1.0:
            continue
        tf = int(tf)
        target = int(target)
        if direction:
            adj[tf, target] = 1.0
            if target in tf_set:
                adj[target, tf] = 1.0
        else:
            adj[tf, target] = 1.0
            adj[target, tf] = 1.0
    if loop:
        adj = adj + sp.identity(num_genes, dtype=np.float32)
    return adj.todok()


@dataclass
class PreparedData:
    expression_df: pd.DataFrame
    feature: torch.Tensor
    tf_indices: torch.Tensor
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    support_adj: torch.Tensor
    train_dataset: EdgeDataset
    gene_to_idx: Dict[str, int]
    idx_to_gene: Dict[int, str]


def prepare_training_data(expression_path: str | Path, tf_list_path: str | Path, train_edges_path: str | Path, val_edges_path: str | Path, test_edges_path: str | Path, device: torch.device, normalize: bool = True, svd_dim: int = 0, loop: bool = False) -> PreparedData:
    expr_df, feature_np = load_expression(expression_path, normalize=normalize, svd_dim=svd_dim)
    gene_to_idx = build_gene_index(expr_df)
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    tf_names = load_tf_list(tf_list_path)
    tf_indices_np = np.array([_resolve_node(tf, gene_to_idx) for tf in tf_names], dtype=np.int64)
    train_df = load_edge_table(train_edges_path, gene_to_idx)
    val_df = load_edge_table(val_edges_path, gene_to_idx)
    test_df = load_edge_table(test_edges_path, gene_to_idx)
    support_adj_sp = build_support_graph(train_df, feature_np.shape[0], tf_indices_np, loop=loop)
    return PreparedData(
        expression_df=expr_df,
        feature=torch.from_numpy(feature_np).to(device),
        tf_indices=torch.from_numpy(tf_indices_np).to(device),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        support_adj=adj_to_sparse_tensor(support_adj_sp).to(device),
        train_dataset=EdgeDataset(train_df[['tf', 'target', 'label']].values),
        gene_to_idx=gene_to_idx,
        idx_to_gene=idx_to_gene,
    )
