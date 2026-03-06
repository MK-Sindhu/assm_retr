# train_missing_part_retrieval.py
# -------------------------------
# Train missing-part retrieval model using partial assembly graphs.
#
# Inputs:
# - missing_part_samples.jsonl from make_missing_part_samples.py
# - part_embeddings_64.pt (dict part_id -> FloatTensor[d])
#
# Model:
# - Edge-aware GNN encoder produces query embedding c(G_obs)
# - Score candidates by dot product with e(part)
# - Optimize InfoNCE loss with sampled negatives
#
# Output:
# - prints validation Top-K and MRR
# - saves checkpoint with encoder weights

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class MissingPartDataset(Dataset):
    def __init__(self, samples_jsonl: str, part_embeddings: Dict[str, torch.Tensor]):
        self.samples = []
        with open(samples_jsonl, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.part_embeddings = part_embeddings

        # Infer edge_attr_dim by scanning for the first non-empty edge_attr
        self.edge_attr_dim = None
        for r in self.samples[:2000]:
            ea = r.get("edge_attr", [])
            if isinstance(ea, list) and len(ea) > 0 and isinstance(ea[0], list):
                self.edge_attr_dim = len(ea[0])
                break
        if self.edge_attr_dim is None:
            # Fallback: allow edge_attr_dim=1, but this is unusual
            self.edge_attr_dim = 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        r = self.samples[idx]
        part_ids = r["observed_part_ids"]
        edges = r.get("edge_index", [])
        edge_attr = r.get("edge_attr", [])

        # node features from part embeddings
        xs = []
        for pid in part_ids:
            xs.append(self.part_embeddings[pid].float())
        x = torch.stack(xs, dim=0)

        # undirected edges -> directed edge_index
        src, dst = [], []
        attrs = []

        for e_i, (u, v) in enumerate(edges):
            a = edge_attr[e_i] if e_i < len(edge_attr) else []
            # pad/truncate edge attribute to fixed dim
            if not isinstance(a, list):
                a = []
            if len(a) < self.edge_attr_dim:
                a = a + [0.0] * (self.edge_attr_dim - len(a))
            elif len(a) > self.edge_attr_dim:
                a = a[:self.edge_attr_dim]

            src.append(u); dst.append(v); attrs.append(a)
            src.append(v); dst.append(u); attrs.append(a)

        if len(src) == 0:
            edge_index_t = torch.empty((2, 0), dtype=torch.long)
            edge_attr_t = torch.empty((0, self.edge_attr_dim), dtype=torch.float32)
        else:
            edge_index_t = torch.tensor([src, dst], dtype=torch.long)
            edge_attr_t = torch.tensor(attrs, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
        data.missing_part_id = r["missing_part_id"]
        return data


class AssemblyQueryEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        g = self.lin(g)
        g = F.normalize(g, p=2, dim=1)
        return g


def build_part_matrix(part_embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    part_ids = list(part_embeddings.keys())
    E = torch.stack([part_embeddings[pid].float() for pid in part_ids], dim=0)
    E = F.normalize(E, p=2, dim=1)
    id_to_idx = {pid: i for i, pid in enumerate(part_ids)}
    return E, part_ids, id_to_idx


def sample_negatives(rng: random.Random, num_parts: int, pos_idx: int, m: int) -> List[int]:
    negs = []
    while len(negs) < m:
        j = rng.randrange(num_parts)
        if j == pos_idx:
            continue
        negs.append(j)
    return negs


@torch.no_grad()
def evaluate(model, loader, part_mat, id_to_idx, device, topk=(1, 5, 10)) -> Dict[str, float]:
    model.eval()
    E = part_mat.to(device)

    hits = {k: 0 for k in topk}
    mrr_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        q = model(batch)  # [B, d]
        scores = q @ E.T   # [B, num_parts]

        # For each item in batch, compute ranks
        for i in range(batch.num_graphs):
            pid = batch.missing_part_id[i]
            pos = id_to_idx.get(pid, None)
            if pos is None:
                continue
            s = scores[i]
            # rank of true part (1 = best)
            rank = int((s > s[pos]).sum().item()) + 1
            mrr_sum += 1.0 / rank
            for k in topk:
                if rank <= k:
                    hits[k] += 1
            n += 1

    out = {f"top{k}": hits[k] / max(n, 1) for k in topk}
    out["mrr"] = mrr_sum / max(n, 1)
    out["n_eval"] = n
    return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_jsonl", type=str, required=True)
    ap.add_argument("--part_embeddings", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--negatives", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_ckpt", type=str, default="assembly_filter_out/missing_part_retriever_v1.pt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    part_embeddings = torch.load(args.part_embeddings, map_location="cpu", weights_only=False)
    part_mat, part_ids, id_to_idx = build_part_matrix(part_embeddings)

    ds = MissingPartDataset(args.samples_jsonl, part_embeddings)
    N = len(ds)
    perm = np.random.RandomState(args.seed).permutation(N)
    split = int(0.8 * N)

    train_idx = perm[:split]
    val_idx = perm[split:]

    train_set = torch.utils.data.Subset(ds, train_idx.tolist())
    val_set = torch.utils.data.Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    in_dim = part_mat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AssemblyQueryEncoder(in_dim=in_dim, hidden=128, out_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Edge attr dim:", ds.edge_attr_dim)
    print("Example edge_attr shapes:", ds[0].edge_attr.shape, ds[1].edge_attr.shape)

    E = part_mat.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            q = model(batch)  # [B, d]

            # Build sampled candidate set per item: [pos + negatives]
            B = batch.num_graphs
            logits = []
            targets = []

            for i in range(B):
                pid = batch.missing_part_id[i]
                pos_idx = id_to_idx.get(pid, None)
                if pos_idx is None:
                    continue
                negs = sample_negatives(rng, E.shape[0], pos_idx, args.negatives)
                cand = [pos_idx] + negs
                candE = E[cand]  # [1+M, d]
                s = torch.matmul(q[i:i+1], candE.T).squeeze(0)  # [1+M]
                logits.append(s)
                targets.append(0)

            if len(logits) == 0:
                continue

            logits = torch.stack(logits, dim=0)  # [B', 1+M]
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, targets)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            losses.append(loss.item())

        metrics = evaluate(model, val_loader, part_mat, id_to_idx, device)
        print(f"Epoch {epoch:02d} | loss {float(np.mean(losses)):.4f} | "
              f"top1 {metrics['top1']:.4f} top5 {metrics['top5']:.4f} top10 {metrics['top10']:.4f} mrr {metrics['mrr']:.4f} | n {metrics['n_eval']}")

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "in_dim": int(in_dim)}, args.out_ckpt)
    print("Saved:", args.out_ckpt)


if __name__ == "__main__":
    main()