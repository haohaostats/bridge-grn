
                     
                                                              
            
                                                              
import os
from pathlib import Path
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from bridge_grn_model import BridgeGRNCore
from bridge_grn_cl import L, A, DualBranchContrast
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation

parser = argparse.ArgumentParser(description="BRIDGE-GRN Training")
parser.add_argument("--cell_type", type=str, default="mESC")
parser.add_argument("--sample", type=str, default="sample1")
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--hidden1", type=int, default=128)
parser.add_argument("--hidden2", type=int, default=64)
parser.add_argument("--hidden3", type=int, default=32)
parser.add_argument("--output_dim", type=int, default=16)
parser.add_argument("--num_head1", type=int, default=3)
parser.add_argument("--num_head2", type=int, default=3)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--Type", type=str, default="dot", choices=["dot", "cosine", "bilinear"])
parser.add_argument("--reduction", type=str, default="concate", choices=["concate", "mean"])
parser.add_argument("--lambda_ctr", type=float, default=0.5)
parser.add_argument("--loop", action="store_true")
parser.add_argument("--flag", action="store_true")
parser.add_argument("--calibrate", action="store_true")
parser.add_argument("--svd_dim", type=int, default=0)
parser.add_argument("--edge_drop", type=float, default=0.2)
parser.add_argument("--pre_epochs", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT: Path = Path(__file__).resolve().parent
expr_dir = ROOT / "Benchmark Datasets" / "Specific Dataset" / args.cell_type / "TFs+1000"
exp_file = expr_dir / "BL--ExpressionData.csv"
tf_file = expr_dir / "TF.csv"

split_dir = ROOT / "Data" / "Specific" / f"{args.cell_type} 1000" / args.sample
train_file = split_dir / "Train_set.csv"
val_file   = split_dir / "Validation_set.csv"
test_file  = split_dir / "Test_set.csv"

missing_expr = [p for p in (exp_file, tf_file) if not p.exists()]
if missing_expr:
    raise FileNotFoundError(
        "\n  - " + "\n  - ".join(str(p) for p in missing_expr)
    )
missing_split = [p for p in (train_file, val_file, test_file) if not p.exists()]
if missing_split:
    raise FileNotFoundError(
        "\n  - " + "\n  - ".join(str(p) for p in missing_split)
    )

expr_df = pd.read_csv(exp_file, index_col=0)
feat_loader = load_data(expr_df)
feature_np = feat_loader.exp_data().astype(np.float32)                         
print("feature shape:", feature_np.shape)

if args.svd_dim and args.svd_dim > 0:
    from sklearn.decomposition import TruncatedSVD
    print(f"[INFO] Apply TruncatedSVD to {args.svd_dim} dims ...")
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    feature_np = svd.fit_transform(feature_np)                                 
    print("svd feature shape:", feature_np.shape)

feature = torch.from_numpy(feature_np).to(device)

tf_idx = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
tf_idx = torch.from_numpy(tf_idx).to(device)

train_np = pd.read_csv(train_file, index_col=0).values
val_np   = pd.read_csv(val_file,   index_col=0).values
test_np  = pd.read_csv(test_file,  index_col=0).values

train_pairs = torch.from_numpy(train_np).to(device)
val_pairs   = torch.from_numpy(val_np).to(device)
test_pairs  = torch.from_numpy(test_np).to(device)

train_dataset = scRNADataset(train_np, feature.shape[0], flag=args.flag)
base_adj_sp = train_dataset.Adj_Generate(tf_idx, loop=args.loop)
base_adj = adj2saprse_tensor(base_adj_sp).to(device)
print("support edges (nnz):", int(base_adj._nnz()))

class BridgeGRN(nn.Module):
    def __init__(self, encoder, aug_left, aug_right):
        super().__init__()
        self.encoder = encoder
        self.aug_left = aug_left
        self.aug_right = aug_right

    def forward(self, node_feat: torch.Tensor, base_adj: torch.Tensor, pair_index: torch.LongTensor):
        idx = base_adj.coalesce().indices()
        size = base_adj.coalesce().size()

        _, eidx1, _ = self.aug_left(node_feat, idx)
        _, eidx2, _ = self.aug_right(node_feat, idx)

        v1 = torch.ones(eidx1.size(1), device=node_feat.device)
        v2 = torch.ones(eidx2.size(1), device=node_feat.device)
        adj1 = torch.sparse_coo_tensor(eidx1, v1, size).coalesce()
        adj2 = torch.sparse_coo_tensor(eidx2, v2, size).coalesce()

        h1, _, _, p1 = self.encoder(node_feat, adj1, pair_index)
        h2, _, _, p2 = self.encoder(node_feat, adj2, pair_index)
        return (h1, p1), (h2, p2)

encoder = BridgeGRNCore(
    input_dim=feature.size(1),
    hidden1_dim=args.hidden1,
    hidden2_dim=args.hidden2,
    hidden3_dim=args.hidden3,
    output_dim=args.output_dim,
    num_head1=args.num_head1,
    num_head2=args.num_head2,
    alpha=args.alpha,
    device=device,
    type=args.Type,
    reduction=args.reduction
).to(device)

aug_left  = A.Identity()
aug_right = A.EdgeRemoving(pe=args.edge_drop)

model = BridgeGRN(encoder=encoder, aug_left=aug_left, aug_right=aug_right).to(device)
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode="L2L", intraview_negs=False).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

def warmup_epoch():
    model.train()
    epoch_loss = 0.0
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    for pair, lab in loader:
        optimizer.zero_grad()
        lab = lab.to(device) if args.flag else lab.to(device).view(-1, 1)
        (_, p1), (_, p2) = model(feature, base_adj, pair.to(device))
        p1a = torch.softmax(p1, dim=1) if args.flag else torch.sigmoid(p1)
        p2a = torch.softmax(p2, dim=1) if args.flag else torch.sigmoid(p2)
        loss = F.binary_cross_entropy(p1a, lab) + F.binary_cross_entropy(p2a, lab)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
    return epoch_loss

if args.pre_epochs > 0:
    for ep in range(1, args.pre_epochs + 1):
        loss = warmup_epoch()
        scheduler.step()
        print(f"[Warmup] Epoch {ep:03d} | loss: {loss:.5f}")

def train_epoch():
    model.train()
    epoch_loss = 0.0
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    for pair, lab in loader:
        optimizer.zero_grad()
        lab = lab.to(device) if args.flag else lab.to(device).view(-1, 1)

        (h1, p1), (h2, p2) = model(feature, base_adj, pair.to(device))

        z1 = F.normalize(h1[tf_idx], p=2, dim=1)
        z2 = F.normalize(h2[tf_idx], p=2, dim=1)
        con_loss = contrast_model(h1=z1, h2=z2)

        p1a = torch.softmax(p1, dim=1) if args.flag else torch.sigmoid(p1)
        p2a = torch.softmax(p2, dim=1) if args.flag else torch.sigmoid(p2)
        loss = F.binary_cross_entropy(p1a, lab) + F.binary_cross_entropy(p2a, lab) + args.lambda_ctr * con_loss

        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
    return epoch_loss

def learn_temperature_on_val(val_pairs_ij, val_labels):
    with torch.no_grad():
        (_, p1), (_, p2) = model(feature, base_adj, val_pairs_ij)
        val_logits = 0.5 * (p1 + p2)

    T = torch.nn.Parameter(torch.tensor(1.0, device=device))
    optT = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    crit = torch.nn.BCEWithLogitsLoss()

    def closure():
        optT.zero_grad()
        loss = crit(val_logits / T.clamp_min(1e-3), val_labels.view(-1, 1))
        loss.backward()
        return loss

    optT.step(closure)
    return T.detach().clamp_min(1e-3)

model_dir = ROOT / "model"
model_dir.mkdir(parents=True, exist_ok=True)
best_auc = -1.0

for ep in range(1, args.epochs + 1):
    loss = train_epoch()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        (hv1, vp1), (hv2, vp2) = model(feature, base_adj, val_pairs[:, :2].long())
        val_logits = 0.5 * (vp1 + vp2)               
        val_score  = torch.softmax(val_logits, dim=1) if args.flag else torch.sigmoid(val_logits)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=val_score, y_true=val_pairs[:, -1], flag=args.flag)
    print(f"[Train] Epoch {ep:03d} | loss: {loss:.5f} | AUC: {AUC:.3f} | AUPR: {AUPR:.3f}")

    if AUC > best_auc:
        best_auc = AUC
        save_path = model_dir / f"{args.cell_type}_bridgegrn_best.pkl"
        torch.save(model.state_dict(), str(save_path))

model.load_state_dict(torch.load(model_dir / f"{args.cell_type}_bridgegrn_best.pkl", map_location=device))
model.eval()

T = None
if args.calibrate and not args.flag:
    T = learn_temperature_on_val(val_pairs[:, :2].long(), val_pairs[:, -1].float().to(device))

with torch.no_grad():
    (ht1, pt1), (ht2, pt2) = model(feature, base_adj, test_pairs[:, :2].long())
    test_logits = 0.5 * (pt1 + pt2)
    score = torch.softmax(test_logits, dim=1) if args.flag else torch.sigmoid(test_logits if T is None else test_logits / T)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_pairs[:, -1], flag=args.flag)
print(score)
print(f"[Test] AUROC: {AUC:.3f} | AUPRC: {AUPR:.3f}")

