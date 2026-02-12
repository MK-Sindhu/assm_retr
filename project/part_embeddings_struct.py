# part_embeddings_struct.py
# ------------------------
# Build part embeddings from part graph statistics.
# FIXED to properly load PyTorch zip-format .pt files.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def compute_struct_embedding(data) -> torch.Tensor:
    """
    Returns fixed-length embedding from a PyG Data object.
    """
    # Node count
    n = int(getattr(data, "num_nodes", 0))
    if n == 0 and hasattr(data, "x"):
        n = data.x.shape[0]

    # Edge count
    m = 0
    if hasattr(data, "edge_index") and isinstance(data.edge_index, torch.Tensor):
        if data.edge_index.ndim == 2:
            m = data.edge_index.shape[1]

    # Degree stats
    mean_deg = 0.0
    max_deg = 0.0
    density = 0.0

    if n > 0 and m > 0:
        deg = torch.bincount(data.edge_index[0], minlength=n).float()
        mean_deg = float(deg.mean())
        max_deg = float(deg.max())
        if n > 1:
            density = float(m) / float(n * (n - 1))

    # Node feature stats
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        x = data.x.float()
        x_dim = float(x.shape[1]) if x.ndim == 2 else 0.0
        x_mean = float(x.mean())
        x_std = float(x.std(unbiased=False))
    else:
        x_dim = 0.0
        x_mean = 0.0
        x_std = 0.0

    emb = torch.tensor(
        [
            float(n),
            float(m),
            mean_deg,
            max_deg,
            density,
            x_dim,
            x_mean,
            x_std,
        ],
        dtype=torch.float32,
    )

    return emb  # dimension = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part_graph_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="part_embeddings.pt")
    ap.add_argument("--progress_every", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.part_graph_dir)
    files = list(root.rglob("*.pt"))

    print("Found part graph files:", len(files))

    part_embeddings: Dict[str, torch.Tensor] = {}
    bad = 0

    for i, fp in enumerate(files, 1):
        pid = fp.stem

        try:
            # 🔥 critical fix here:
            obj = torch.load(fp, map_location="cpu", weights_only=False)

            # If saved as dict with "data"
            if isinstance(obj, dict) and "data" in obj:
                data = obj["data"]
            else:
                data = obj

            emb = compute_struct_embedding(data)
            part_embeddings[pid] = emb

        except Exception as e:
            bad += 1

        if args.progress_every and i % args.progress_every == 0:
            print(f"Processed: {i} | embeddings: {len(part_embeddings)} | bad: {bad}")

    torch.save(part_embeddings, args.out)
    print("\n✅ Saved:", args.out)
    print("Embeddings:", len(part_embeddings))
    print("Bad files:", bad)

    if len(part_embeddings) > 0:
        any_pid = next(iter(part_embeddings))
        print("Embedding dim:", part_embeddings[any_pid].shape[0])


if __name__ == "__main__":
    main()
