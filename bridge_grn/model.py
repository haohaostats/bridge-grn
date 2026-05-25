import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))
        self.bias = nn.Parameter(torch.FloatTensor(self.output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        if self.bias is not None:
            self.bias.data.zero_()

    def _e_logits(self, h):
        wh1 = torch.matmul(h, self.a[:self.output_dim, :])
        wh2 = torch.matmul(h, self.a[self.output_dim:, :])
        return F.leaky_relu(wh1 + wh2.T, negative_slope=self.alpha)

    def forward(self, x, adj_sparse):
        h = torch.matmul(x, self.weight)
        e = self._e_logits(h)
        adj_mask = adj_sparse.to_dense() > 0
        has_neighbor = adj_mask.any(dim=1, keepdim=True)
        zero_vec = -9e15 * torch.ones_like(e)
        attn = torch.where(adj_mask, e, zero_vec)
        attn = F.softmax(attn, dim=1)
        attn = torch.where(has_neighbor, attn, torch.zeros_like(attn))
        attn = F.dropout(attn, p=0.0, training=self.training)
        h_pass = torch.matmul(attn, h)
        h_pass = torch.where(has_neighbor, h_pass, h)
        out = F.leaky_relu(h_pass, negative_slope=self.alpha)
        out = F.normalize(out, p=2, dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


class BridgeGRNCore(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1_dim,
        hidden2_dim,
        hidden3_dim,
        output_dim,
        num_head1,
        num_head2,
        alpha,
        device,
        type='dot',
        reduction='concate',
        share_tower=False,
    ):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.reduction = reduction
        self.type = (type or 'dot').lower()
        self.share_tower = bool(share_tower)

        if self.reduction == 'mean':
            h1_out = hidden1_dim
        elif self.reduction == 'concate':
            h1_out = num_head1 * hidden1_dim
        else:
            raise TypeError("reduction must be 'mean' or 'concate'")

        self.heads1 = nn.ModuleList([AttentionLayer(input_dim, hidden1_dim, alpha) for _ in range(num_head1)])
        self.heads2 = nn.ModuleList([AttentionLayer(h1_out, hidden2_dim, alpha) for _ in range(num_head2)])

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        if self.share_tower:
            self.target_linear1 = self.tf_linear1
            self.target_linear2 = self.tf_linear2
        else:
            self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
            self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        self.relation_R = nn.Parameter(torch.empty(output_dim, output_dim))
        self.reset_parameters()

        with torch.no_grad():
            self.relation_R.copy_(torch.eye(output_dim))
            self.relation_R.add_(1e-4 * torch.randn_like(self.relation_R))

    def reset_parameters(self):
        for m in list(self.heads1) + list(self.heads2):
            m.reset_parameters()
        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        if not self.share_tower:
            nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        if not self.share_tower:
            nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)
        if self.tf_linear1.bias is not None:
            nn.init.zeros_(self.tf_linear1.bias)
        if (not self.share_tower) and self.target_linear1.bias is not None:
            nn.init.zeros_(self.target_linear1.bias)
        if self.tf_linear2.bias is not None:
            nn.init.zeros_(self.tf_linear2.bias)
        if (not self.share_tower) and self.target_linear2.bias is not None:
            nn.init.zeros_(self.target_linear2.bias)
        nn.init.xavier_uniform_(self.relation_R, gain=1.414)

    def encode(self, x, adj_sparse):
        h1_list = [head(x, adj_sparse) for head in self.heads1]
        if self.reduction == 'concate':
            h1 = torch.cat(h1_list, dim=1)
        else:
            h1 = torch.mean(torch.stack(h1_list, dim=0), dim=0)
        h1 = F.elu(h1)

        h2_list = [head(h1, adj_sparse) for head in self.heads2]
        h2 = torch.mean(torch.stack(h2_list, dim=0), dim=0)
        return h2

    def _role_towers(self, h):
        tf = F.leaky_relu(self.tf_linear1(h))
        tf = F.dropout(tf, p=0.01, training=self.training)
        tf = F.leaky_relu(self.tf_linear2(tf))

        tg = F.leaky_relu(self.target_linear1(h))
        tg = F.dropout(tg, p=0.01, training=self.training)
        tg = F.leaky_relu(self.target_linear2(tg))
        return tf, tg

    def decode(self, tf_embed, target_embed):
        if self.type == 'bilinear':
            rt = torch.matmul(target_embed, self.relation_R)
            return (tf_embed * rt).sum(dim=1, keepdim=True)
        elif self.type == 'dot':
            return (tf_embed * target_embed).sum(dim=1, keepdim=True)
        elif self.type == 'cosine':
            return F.cosine_similarity(tf_embed, target_embed, dim=1).unsqueeze(1)
        else:
            raise TypeError(f'Unknown decode type: {self.type}')

    def forward(self, x, adj_sparse, pair_index):
        h = self.encode(x, adj_sparse)
        tf_all, tg_all = self._role_towers(h)
        tf_idx = pair_index[:, 0].long()
        tg_idx = pair_index[:, 1].long()
        tf_b = tf_all[tf_idx]
        tg_b = tg_all[tg_idx]
        pred = self.decode(tf_b, tg_b)
        return h, tf_all, tg_all, pred


class BridgeGRN(nn.Module):
    def __init__(self, encoder: nn.Module, aug_left, aug_right):
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
