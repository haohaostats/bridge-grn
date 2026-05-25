from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_scores(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:
    yt = y_true.detach().cpu().numpy().reshape(-1).astype(int)
    ys = y_score.detach().cpu().numpy().reshape(-1)
    auroc = roc_auc_score(yt, ys)
    auprc = average_precision_score(yt, ys)
    positive_rate = float(np.mean(yt)) if len(yt) else 0.0
    auprc_norm = auprc / positive_rate if positive_rate > 0 else 0.0
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'auprc_norm': float(auprc_norm),
        'positive_rate': positive_rate,
    }
