from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .contrast import A, L, DualBranchContrast
from .data import load_edge_table, prepare_training_data
from .metrics import evaluate_scores
from .model import BridgeGRN, BridgeGRNCore


@dataclass
class TrainingConfig:
    expression: str
    tf_list: str
    train_edges: str
    val_edges: str
    test_edges: str
    output_dir: str
    lr: float = 3e-3
    epochs: int = 20
    batch_size: int = 256
    seed: int = 2025
    hidden1: int = 128
    hidden2: int = 64
    hidden3: int = 32
    output_dim: int = 16
    num_head1: int = 3
    num_head2: int = 3
    alpha: float = 0.2
    decoder: str = 'dot'
    reduction: str = 'concate'
    lambda_ctr: float = 0.5
    loop: bool = False
    edge_drop: float = 0.2
    pre_epochs: int = 0
    share_tower: bool = False
    svd_dim: int = 0
    normalize_expression: bool = True
    device: str = 'cpu'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_model(cfg: TrainingConfig, input_dim: int, device: torch.device) -> BridgeGRN:
    encoder = BridgeGRNCore(
        input_dim=input_dim,
        hidden1_dim=cfg.hidden1,
        hidden2_dim=cfg.hidden2,
        hidden3_dim=cfg.hidden3,
        output_dim=cfg.output_dim,
        num_head1=cfg.num_head1,
        num_head2=cfg.num_head2,
        alpha=cfg.alpha,
        device=str(device),
        type=cfg.decoder,
        reduction=cfg.reduction,
        share_tower=cfg.share_tower,
    ).to(device)
    return BridgeGRN(encoder=encoder, aug_left=A.Identity(), aug_right=A.EdgeRemoving(pe=cfg.edge_drop)).to(device)


def _predict_scores(model: BridgeGRN, feature: torch.Tensor, support_adj: torch.Tensor, pair_tensor: torch.Tensor) -> torch.Tensor:
    (_, p1), (_, p2) = model(feature, support_adj, pair_tensor[:, :2].long())
    logits = 0.5 * (p1 + p2)
    return torch.sigmoid(logits)


def _save_predictions(df, scores: torch.Tensor, idx_to_gene: Dict[int, str], out_path: Path) -> None:
    out = df.copy()
    out['tf_name'] = out['tf'].map(idx_to_gene)
    out['target_name'] = out['target'].map(idx_to_gene)
    out['score'] = scores.detach().cpu().numpy().reshape(-1)
    out.to_csv(out_path, index=False)


def _save_checkpoint(model: BridgeGRN, cfg: TrainingConfig, idx_to_gene: Dict[int, str], out_path: Path) -> None:
    payload = {
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
        'idx_to_gene': idx_to_gene,
    }
    torch.save(payload, out_path)


def _run_training(cfg: TrainingConfig, initial_state_dict: Optional[dict] = None):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if cfg.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)
    data = prepare_training_data(
        expression_path=cfg.expression,
        tf_list_path=cfg.tf_list,
        train_edges_path=cfg.train_edges,
        val_edges_path=cfg.val_edges,
        test_edges_path=cfg.test_edges,
        device=device,
        normalize=cfg.normalize_expression,
        svd_dim=cfg.svd_dim,
        loop=cfg.loop,
    )
    model = _build_model(cfg, data.feature.size(1), device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=False)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    def run_epoch(with_contrast: bool) -> float:
        model.train()
        total_loss = 0.0
        loader = DataLoader(data.train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        for pair, lab in loader:
            optimizer.zero_grad()
            lab = lab.to(device).view(-1, 1)
            (h1, p1), (h2, p2) = model(data.feature, data.support_adj, pair.to(device))
            loss = F.binary_cross_entropy(torch.sigmoid(p1), lab) + F.binary_cross_entropy(torch.sigmoid(p2), lab)
            if with_contrast and cfg.lambda_ctr > 0:
                z1 = F.normalize(h1[data.tf_indices], p=2, dim=1)
                z2 = F.normalize(h2[data.tf_indices], p=2, dim=1)
                loss = loss + cfg.lambda_ctr * contrast_model(h1=z1, h2=z2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss

    best_state = None
    best_val = -1.0
    history = []

    for _ in range(cfg.pre_epochs):
        run_epoch(with_contrast=False)
        scheduler.step()

    val_tensor = torch.from_numpy(data.val_df[['tf', 'target', 'label']].values).to(device)
    test_tensor = torch.from_numpy(data.test_df[['tf', 'target', 'label']].values).to(device)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(with_contrast=True)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_scores = _predict_scores(model, data.feature, data.support_adj, val_tensor)
            val_metrics = evaluate_scores(val_tensor[:, -1], val_scores)
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})
        if val_metrics['auroc'] > best_val:
            best_val = val_metrics['auroc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        test_scores = _predict_scores(model, data.feature, data.support_adj, test_tensor)
        test_metrics = evaluate_scores(test_tensor[:, -1], test_scores)

    _save_checkpoint(model, cfg, data.idx_to_gene, out_dir / 'bridge_grn_best.pt')
    _save_predictions(data.test_df, test_scores, data.idx_to_gene, out_dir / 'test_predictions.csv')

    metrics = {
        'best_val_auroc': best_val,
        'test': test_metrics,
        'history': history,
        'config': asdict(cfg),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def train_model(cfg: TrainingConfig):
    return _run_training(cfg, initial_state_dict=None)


def transfer_train(cfg: TrainingConfig, checkpoint_path: str | Path):
    payload = torch.load(checkpoint_path, map_location='cpu')
    state_dict = payload['state_dict'] if isinstance(payload, dict) and 'state_dict' in payload else payload

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if cfg.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)
    data = prepare_training_data(
        expression_path=cfg.expression,
        tf_list_path=cfg.tf_list,
        train_edges_path=cfg.train_edges,
        val_edges_path=cfg.val_edges,
        test_edges_path=cfg.test_edges,
        device=device,
        normalize=cfg.normalize_expression,
        svd_dim=cfg.svd_dim,
        loop=cfg.loop,
    )
    model = _build_model(cfg, data.feature.size(1), device)
    model.load_state_dict(state_dict, strict=False)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    loader = DataLoader(data.train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for pair, lab in loader:
            optimizer.zero_grad()
            lab = lab.to(device).view(-1, 1)
            (h1, p1), (h2, p2) = model(data.feature, data.support_adj, pair.to(device))
            z1 = F.normalize(h1[data.tf_indices], p=2, dim=1)
            z2 = F.normalize(h2[data.tf_indices], p=2, dim=1)
            con_loss = contrast_model(h1=z1, h2=z2)
            loss = F.binary_cross_entropy(torch.sigmoid(p1), lab) + F.binary_cross_entropy(torch.sigmoid(p2), lab) + cfg.lambda_ctr * con_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': epoch_loss})

    test_tensor = torch.from_numpy(data.test_df[['tf', 'target', 'label']].values).to(device)
    model.eval()
    with torch.no_grad():
        test_scores = _predict_scores(model, data.feature, data.support_adj, test_tensor)
        test_metrics = evaluate_scores(test_tensor[:, -1], test_scores)

    _save_checkpoint(model, cfg, data.idx_to_gene, out_dir / 'bridge_grn_best.pt')
    _save_predictions(data.test_df, test_scores, data.idx_to_gene, out_dir / 'test_predictions.csv')

    metrics = {
        'test': test_metrics,
        'history': history,
        'config': asdict(cfg),
        'transfer_mode': 'final_epoch_finetune',
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def predict_edges(checkpoint_path: str | Path, expression_path: str | Path, tf_list_path: str | Path, support_edges_path: str | Path, query_edges_path: str | Path, output_path: str | Path, device_name: str = 'cpu') -> None:
    payload = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(payload, dict) or 'state_dict' not in payload or 'config' not in payload:
        raise ValueError('Checkpoint must be a BRIDGE-GRN checkpoint produced by this open implementation.')
    cfg_dict = payload['config']
    cfg = TrainingConfig(**{k: cfg_dict[k] for k in TrainingConfig.__dataclass_fields__.keys() if k in cfg_dict})
    device = torch.device(device_name if device_name == 'cpu' or torch.cuda.is_available() else 'cpu')
    data = prepare_training_data(
        expression_path=expression_path,
        tf_list_path=tf_list_path,
        train_edges_path=support_edges_path,
        val_edges_path=support_edges_path,
        test_edges_path=support_edges_path,
        device=device,
        normalize=cfg.normalize_expression,
        svd_dim=cfg.svd_dim,
        loop=cfg.loop,
    )
    model = _build_model(cfg, data.feature.size(1), device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    query_df = load_edge_table(query_edges_path, data.gene_to_idx)
    query_tensor = torch.from_numpy(query_df[['tf', 'target', 'label']].values).to(device)
    with torch.no_grad():
        scores = _predict_scores(model, data.feature, data.support_adj, query_tensor)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_predictions(query_df, scores, data.idx_to_gene, output_path)
