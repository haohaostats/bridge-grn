          

from typing import Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

class scRNADataset(Dataset):
    def __init__(self, train_set: np.ndarray, num_gene: int, flag: bool = False):
        super().__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx: int):
        pair = self.train_set[:, :2]
        label = self.train_set[:, -1]

        if self.flag:
            n = len(label)
            onehot = np.zeros([n, 2], dtype=np.float32)
            onehot[:, 0] = 1 - label
            onehot[:, 1] = label
            label = onehot

        data = pair[idx].astype(np.int64)
        y = label[idx].astype(np.float32)
        return data, y

    def __len__(self) -> int:
        return len(self.train_set)

    def Adj_Generate(self, TF_set: torch.Tensor, direction: bool = False, loop: bool = False) -> sp.dok_matrix:
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)
        tf_set_np = TF_set.cpu().numpy() if isinstance(TF_set, torch.Tensor) else np.array(TF_set)
        tf_hash = set(tf_set_np.tolist())

        for tf, target, label in self.train_set:
            if label != 1:
                continue
            if not direction:
                adj[tf, target] = 1.0
                adj[target, tf] = 1.0
            else:
                adj[tf, target] = 1.0
                if target in tf_hash:
                    adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene, dtype=np.float32)

        return adj.todok()


class load_data:
    def __init__(self, data: pd.DataFrame, normalize: bool = True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        epr = scaler.fit_transform(data.T) 
        return epr.T

    def exp_data(self) -> np.ndarray:
        data_feature = self.data.values.astype(np.float32)
        if self.normalize:
            data_feature = self.data_normalize(data_feature)
        return data_feature


def adj2saprse_tensor(adj: sp.spmatrix) -> torch.Tensor:
    coo = adj.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(i, v, coo.shape)


def Evaluation(y_true: torch.Tensor = None,
               y_pred: torch.Tensor = None,
               flag: bool = False) -> Tuple[float, float, float]:

    if flag:
        y_p = y_pred[:, -1].detach().cpu().numpy().reshape(-1)
    else:
        y_p = y_pred.detach().cpu().numpy().reshape(-1)

    y_t = y_true.detach().cpu().numpy().reshape(-1).astype(int)
    auc = roc_auc_score(y_true=y_t, y_score=y_p)
    aupr = average_precision_score(y_true=y_t, y_score=y_p)
    aupr_norm = aupr / np.mean(y_t) if np.mean(y_t) > 0 else 0.0
    return auc, aupr, aupr_norm


def normalize(expression: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(expression)


def Network_Statistic(data_type: str, net_scale: int, net_type: str) -> float:
    if net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}
        return dic[data_type + str(net_scale)]

    elif net_type == 'Non-Specific':
        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}
        return dic[data_type + str(net_scale)]

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165, 'hHEP500': 0.379, 'hHEP1000': 0.377, 'mDC500': 0.085,
               'mDC1000': 0.082, 'mESC500': 0.345, 'mESC1000': 0.347, 'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565, 'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}
        return dic[data_type + str(net_scale)]

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}
        return dic['mESC' + str(net_scale)]

    else:
        raise ValueError(f'Unknown net_type: {net_type}')
