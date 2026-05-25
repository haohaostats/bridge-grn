# bridge_grn_cl.py

from __future__ import annotations
import types
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class _Identity:
    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return x, edge_index, None

class _EdgeRemoving:
    def __init__(self, pe: float = 0.2):
        assert 0.0 <= pe < 1.0, "pe must be in [0,1)"
        self.pe = pe

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if edge_index is None:
            return x, edge_index, None
        E = edge_index.size(1)
        if E == 0:
            return x, edge_index, None

        keep_prob = 1.0 - self.pe
        device = edge_index.device
        mask = torch.rand(E, device=device) < keep_prob

        if mask.sum() == 0:
            rand_idx = torch.randint(0, E, (1,), device=device)
            mask[rand_idx] = True

        new_edge_index = edge_index[:, mask]
        return x, new_edge_index, None

class _InfoNCE(nn.Module):
    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        assert z1.shape == z2.shape, f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}"
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        logits = (z1 @ z2.t()) / self.tau  # [N, N]
        N = logits.size(0)
        labels = torch.arange(N, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

class DualBranchContrast(nn.Module):
    def __init__(self, loss: nn.Module, mode: str = 'L2L', intraview_negs: bool = False):
        super().__init__()
        assert mode in ('L2L',), "This minimal implementation only supports mode='L2L'."
        self.loss_fn = loss
        self.mode = mode
        self.intraview_negs = intraview_negs  

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        loss12 = self.loss_fn(h1, h2)
        loss21 = self.loss_fn(h2, h1)
        return 0.5 * (loss12 + loss21)
    
    __call__ = forward

L = types.SimpleNamespace(InfoNCE=_InfoNCE)
A = types.SimpleNamespace(Identity=_Identity, EdgeRemoving=_EdgeRemoving)
