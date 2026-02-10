# assembly_graph_pyg.py
# ---------------------
# Adapter: canonical assembly graphs (JSONL) -> PyTorch Geometric Data objects.
#
# Input:
#   assembly_filter_out/assembly_graphs_v1.jsonl
# Each line contains:
#   assembly_path, num_parts, num_edges, edge_index [[u,v]...], edge_multiplicity [...]
#
# This adapter loads the corresponding assembly JSON to fetch parts[i]["id"].
# Node features:
#   - dummy: random vectors (for correctness tests)
#   - embeddings: provide a dict {part_id: torch.Tensor([D])}
#
# Output:
#   torch_geometric.data.Data with:
#     x: [N, D]
#     edge_index: [2, 2E] (undirected)
#     edge_attr: [2E, 1] (multiplicity)
#     part_ids: list[str]
#     assembly_path: str
#     assembly_id: str (if present)
#
# Notebook usage:
#   from assembly_graph_pyg import AssemblyGraphDataset
#   ds = AssemblyGraphDataset("assembly_filter_out/assembly_graphs_v1.jsonl", feature_dim=128)
#   d0 = ds[0]
#
# CLI smoke test:
#   python assembly_graph_pyg.py --graphs_jsonl assembly_filter_out/assembly_graphs_v1.jsonl --num 3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _safe_list(obj: Dict[str, Any], key: str) -> List[Any]:
    v = obj.get(key)
    return v if isinstance(v, list) else []


def load_assembly_part_ids_and_id(assembly_path: str) -> Tuple[List[str], Optional[str]]:
    """
    Load assembly JSON and return:
      - part_ids: list of parts[i]["id"] in index order
      - assembly_id: assembly["assemblyId"] if present else None
    """
    with open(assembly_path, "r") as f:
        assembly = json.load(f)

    parts = _safe_list(assembly, "parts")
    part_ids: List[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("id"), str):
            part_ids.append(p["id"])
        else:
            part_ids.append("")  # keep index alignment even if missing

    assembly_id = assembly.get("assemblyId")
    if not isinstance(assembly_id, str):
        assembly_id = None

    return part_ids, assembly_id


def make_undirected_edges(
    edge_index: List[List[int]],
    edge_multiplicity: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert undirected edge list [[u,v],...] into PyG edge_index (both directions).
    edge_attr is multiplicity as a scalar feature.

    Returns:
      edge_index_t: LongTensor [2, 2E]
      edge_attr_t: LongTensor [2E, 1]
    """
    if len(edge_index) != len(edge_multiplicity):
        raise ValueError("edge_index and edge_multiplicity must have same length")

    if len(edge_index) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.long)

    src = []
    dst = []
    attr = []

    for (u, v), m in zip(edge_index, edge_multiplicity):
        # forward
        src.append(u); dst.append(v); attr.append(m)
        # backward
        src.append(v); dst.append(u); attr.append(m)

    edge_index_t = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_t = torch.tensor(attr, dtype=torch.long).unsqueeze(1)
    return edge_index_t, edge_attr_t


def build_node_features(
    part_ids: List[str],
    feature_dim: int,
    part_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 0,
) -> torch.Tensor:
    """
    Node feature matrix x: [N, D]
    - If part_embeddings provided: x[i] = part_embeddings[part_id]
    - Else: deterministic random features (seeded) for correctness testing
    """
    n = len(part_ids)

    if part_embeddings is not None:
        xs = []
        for pid in part_ids:
            if pid not in part_embeddings:
                raise KeyError(f"Missing embedding for part_id: {pid}")
            emb = part_embeddings[pid]
            if emb.ndim != 1:
                raise ValueError(f"Embedding must be 1D tensor. Got shape {tuple(emb.shape)} for {pid}")
            xs.append(emb)
        x = torch.stack(xs, dim=0)
        return x

    # Dummy deterministic random features (pipeline correctness)
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn((n, feature_dim), generator=gen)
    return x


@dataclass
class AssemblyGraphRecord:
    assembly_path: str
    num_parts: int
    num_edges: int
    edge_index: List[List[int]]
    edge_multiplicity: List[int]


def read_graph_records(graphs_jsonl: str) -> List[AssemblyGraphRecord]:
    records: List[AssemblyGraphRecord] = []
    with open(graphs_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            records.append(
                AssemblyGraphRecord(
                    assembly_path=obj["assembly_path"],
                    num_parts=int(obj["num_parts"]),
                    num_edges=int(obj["num_edges"]),
                    edge_index=obj.get("edge_index", []),
                    edge_multiplicity=obj.get("edge_multiplicity", []),
                )
            )
    return records


class AssemblyGraphDataset(Dataset):
    """
    Dataset that returns PyG Data objects from canonical assembly graph JSONL.

    Parameters
    ----------
    graphs_jsonl: path to assembly_graphs_v1.jsonl
    feature_dim: dimension of dummy features (ignored if part_embeddings provided)
    part_embeddings: optional dict {part_id: torch.Tensor([D])}
    seed: seed for dummy features
    """

    def __init__(
        self,
        graphs_jsonl: str,
        *,
        feature_dim: int = 128,
        part_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        seed: int = 0,
    ):
        self.graphs_jsonl = graphs_jsonl
        self.records = read_graph_records(graphs_jsonl)
        self.feature_dim = feature_dim
        self.part_embeddings = part_embeddings
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        r = self.records[idx]

        part_ids, assembly_id = load_assembly_part_ids_and_id(r.assembly_path)

        # Sanity alignment
        if len(part_ids) != r.num_parts:
            # If mismatch happens, keep index-based safety by clipping/padding
            if len(part_ids) > r.num_parts:
                part_ids = part_ids[: r.num_parts]
            else:
                part_ids = part_ids + [""] * (r.num_parts - len(part_ids))

        x = build_node_features(
            part_ids=part_ids,
            feature_dim=self.feature_dim,
            part_embeddings=self.part_embeddings,
            seed=self.seed + idx,  # vary per graph but deterministic overall
        )

        edge_index_t, edge_attr_t = make_undirected_edges(r.edge_index, r.edge_multiplicity)

        data = Data(
            x=x,
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
        )
        data.part_ids = part_ids
        data.assembly_path = r.assembly_path
        if assembly_id is not None:
            data.assembly_id = assembly_id
        return data


def _main() -> None:
    ap = argparse.ArgumentParser(description="Canonical assembly graphs JSONL -> PyG Data (smoke test).")
    ap.add_argument("--graphs_jsonl", type=str, required=True, help="Path to assembly_graphs_v1.jsonl")
    ap.add_argument("--num", type=int, default=3, help="Number of samples to print")
    ap.add_argument("--feature_dim", type=int, default=128, help="Dummy feature dim")
    args = ap.parse_args()

    ds = AssemblyGraphDataset(args.graphs_jsonl, feature_dim=args.feature_dim)

    n = min(args.num, len(ds))
    for i in range(n):
        d = ds[i]
        print(f"\nSample {i}")
        print("x:", tuple(d.x.shape))
        print("edge_index:", tuple(d.edge_index.shape))
        print("edge_attr:", tuple(d.edge_attr.shape))
        print("assembly_path:", d.assembly_path)
        print("assembly_id:", getattr(d, "assembly_id", None))


if __name__ == "__main__":
    _main()
