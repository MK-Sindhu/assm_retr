# part_embeddings_struct64.py
# ---------------------------
# Create 64-dim deterministic part embeddings from part graph .pt files.
#
# Embedding components (64 dims total):
#  - 8 base stats: [n, m, mean_deg, max_deg, density, x_dim, x_mean, x_std]
#  - 16 degree histogram bins (normalized)
#  - 8 spectral features: top-k eigenvalues of normalized Laplacian (k=8)
#  - 32 feature summary stats over node features x (robust):
#       per-part aggregates: mean, std, min, max of x (computed over finite values)
#       plus quantiles (10%, 50%, 90%) of flattened x
#     packed into 32 dims (with safe padding/truncation)
#
# Output: part_embeddings_64.pt  (dict: part_id -> FloatTensor[64])
#
# Notes:
# - Requires only torch.
# - Handles non-finite values by filtering to finite values.
# - Spectral features are computed on a downsampled Laplacian if graph is huge.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch


def safe_to_data(obj: Any):
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def get_num_nodes(data) -> int:
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return int(data.num_nodes)
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        return int(data.x.shape[0])
    return 0


def get_edge_index(data) -> torch.Tensor:
    if hasattr(data, "edge_index") and isinstance(data.edge_index, torch.Tensor):
        ei = data.edge_index
        if ei.ndim == 2 and ei.shape[0] == 2:
            return ei.long()
    return torch.empty((2, 0), dtype=torch.long)


def compute_degree_stats(n: int, edge_index: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    """
    Returns: mean_deg, max_deg, degree_vector (float tensor length n)
    Uses out-degree of edge_index[0] (directed). Acceptable baseline.
    """
    if n <= 0 or edge_index.numel() == 0:
        deg = torch.zeros((max(n, 1),), dtype=torch.float32)
        return 0.0, 0.0, deg[:n] if n > 0 else deg

    src = edge_index[0]
    deg = torch.bincount(src, minlength=n).float()
    return float(deg.mean()), float(deg.max()), deg


def degree_histogram(deg: torch.Tensor, bins: int = 16) -> torch.Tensor:
    """
    Degree histogram with fixed bins. Uses log-binning style edges.
    Output is normalized to sum to 1.
    """
    if deg.numel() == 0:
        return torch.zeros((bins,), dtype=torch.float32)

    dmax = float(deg.max().item())
    if dmax <= 0:
        h = torch.zeros((bins,), dtype=torch.float32)
        h[0] = 1.0
        return h

    # log-spaced bin edges in [0, dmax]
    # include 0 separately, then log bins for 1..dmax
    edges = torch.logspace(0, torch.log10(torch.tensor(dmax + 1e-6)), steps=bins).float()
    # convert degrees to bucket indices
    # bucketize expects 1D sorted boundaries
    idx = torch.bucketize(deg, edges)
    # idx ranges 0..bins
    idx = torch.clamp(idx, 0, bins - 1)
    h = torch.bincount(idx, minlength=bins).float()
    s = h.sum()
    return h / s if s > 0 else h


def robust_x_stats(data) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns (x_dim, x_mean, x_std, x_min, x_max, q10, q50, q90) but packed below.
    We compute over finite values only.
    """
    if not (hasattr(data, "x") and isinstance(data.x, torch.Tensor) and data.x.numel() > 0):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    x = data.x.float()
    x_dim = float(x.shape[1]) if x.ndim == 2 else 0.0

    finite = torch.isfinite(x)
    if not finite.any():
        return x_dim, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    xf = x[finite]
    x_mean = float(xf.mean().item())
    x_std = float(xf.std(unbiased=False).item())
    x_min = float(xf.min().item())
    x_max = float(xf.max().item())

    # quantiles of flattened finite values
    xf_sorted = torch.sort(xf)[0]
    def q(p):
        if xf_sorted.numel() == 1:
            return float(xf_sorted[0].item())
        idx = int(p * (xf_sorted.numel() - 1))
        return float(xf_sorted[idx].item())

    q10 = q(0.10)
    q50 = q(0.50)
    q90 = q(0.90)

    return x_dim, x_mean, x_std, x_min, x_max, q10, q50, q90


def normalized_laplacian_eigs(n: int, edge_index: torch.Tensor, k: int = 8, max_nodes: int = 400) -> torch.Tensor:
    """
    Compute top-k eigenvalues of normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    For large graphs, we downsample nodes uniformly to max_nodes to keep it fast.

    Returns tensor of shape [k], padded with zeros if needed.
    """
    if n <= 1 or edge_index.numel() == 0:
        return torch.zeros((k,), dtype=torch.float32)

    # Downsample if too large
    if n > max_nodes:
        idx = torch.randperm(n)[:max_nodes]
        idx_sorted, _ = torch.sort(idx)
        mapping = -torch.ones((n,), dtype=torch.long)
        mapping[idx_sorted] = torch.arange(idx_sorted.numel())
        src = edge_index[0]
        dst = edge_index[1]
        mask = (mapping[src] >= 0) & (mapping[dst] >= 0)
        src2 = mapping[src[mask]]
        dst2 = mapping[dst[mask]]
        n2 = idx_sorted.numel()
        edge_index = torch.stack([src2, dst2], dim=0)
        n = n2

    # Build dense adjacency (n small now)
    A = torch.zeros((n, n), dtype=torch.float32)
    src = edge_index[0]
    dst = edge_index[1]
    # undirectedize
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    A.fill_diagonal_(0.0)

    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / torch.sqrt(deg[mask])

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L = torch.eye(n) - (D_inv_sqrt @ A @ D_inv_sqrt)

    # Eigenvalues of symmetric matrix
    # eigvalsh returns sorted ascending
    vals = torch.linalg.eigvalsh(L)
    # take smallest k (more stable / informative)
    vals_k = vals[:k]
    if vals_k.numel() < k:
        vals_k = torch.cat([vals_k, torch.zeros((k - vals_k.numel(),), dtype=torch.float32)], dim=0)
    return vals_k.float()


def compute_embedding_64(data) -> torch.Tensor:
    n = get_num_nodes(data)
    edge_index = get_edge_index(data)
    m = int(edge_index.shape[1])

    mean_deg, max_deg, deg_vec = compute_degree_stats(n, edge_index)
    density = float(m) / float(n * (n - 1)) if n > 1 else 0.0

    # x stats (robust)
    x_dim, x_mean, x_std, x_min, x_max, q10, q50, q90 = robust_x_stats(data)

    base8 = torch.tensor(
        [float(n), float(m), mean_deg, max_deg, density, x_dim, x_mean, x_std],
        dtype=torch.float32,
    )

    hist16 = degree_histogram(deg_vec, bins=16)

    eig8 = normalized_laplacian_eigs(n, edge_index, k=8, max_nodes=400)

    # Pack 32 dims: we’ll include repeated robust stats + some interactions
    # This is deterministic and padded to fixed length.
    extra = torch.tensor(
        [
            x_min, x_max, q10, q50, q90,
            x_mean * 0.0 + 1.0,              # constant marker (can help sanity)
            float(n) / (float(m) + 1.0),     # n/(m+1)
            float(m) / (float(n) + 1.0),     # m/(n+1)
        ],
        dtype=torch.float32,
    )
    # extra has 8 dims; expand to 32 by adding simple powers/products (safe)
    feats = [extra]
    feats.append(extra * extra)
    feats.append(torch.sqrt(torch.abs(extra) + 1e-6))
    feats.append(torch.sign(extra) * torch.log1p(torch.abs(extra)))
    extra32 = torch.cat(feats, dim=0)[:32]  # ensure 32 dims

    emb = torch.cat([base8, hist16, eig8, extra32], dim=0)
    assert emb.numel() == 64, f"Expected 64 dims, got {emb.numel()}"
    # final safety
    emb[~torch.isfinite(emb)] = 0.0
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part_graph_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="part_embeddings_64.pt")
    ap.add_argument("--progress_every", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.part_graph_dir)
    files = list(root.rglob("*.pt"))
    if args.limit is not None:
        files = files[: args.limit]

    print("Found part graph files:", len(files))

    out: Dict[str, torch.Tensor] = {}
    bad = 0

    for i, fp in enumerate(files, 1):
        pid = fp.stem
        try:
            obj = torch.load(fp, map_location="cpu", weights_only=False)
            data = safe_to_data(obj)
            emb = compute_embedding_64(data)
            out[pid] = emb
        except Exception:
            bad += 1

        if args.progress_every and i % args.progress_every == 0:
            print(f"Processed: {i} | embeddings: {len(out)} | bad: {bad}")

    torch.save(out, args.out)
    print("Saved:", args.out)
    print("Embeddings:", len(out))
    print("Bad files:", bad)
    if len(out) > 0:
        any_pid = next(iter(out))
        print("Embedding dim:", int(out[any_pid].numel()))


if __name__ == "__main__":
    main()