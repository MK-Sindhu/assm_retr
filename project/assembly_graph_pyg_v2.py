# assembly_graph_pyg_v2.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _safe_list(obj: Dict[str, Any], key: str) -> List[Any]:
    v = obj.get(key)
    return v if isinstance(v, list) else []


def load_assembly_part_ids_and_id(assembly_path: str) -> Tuple[List[str], Optional[str]]:
    with open(assembly_path, "r") as f:
        assembly = json.load(f)

    parts = _safe_list(assembly, "parts")
    part_ids = []
    for p in parts:
        part_ids.append(p["id"] if isinstance(p, dict) and isinstance(p.get("id"), str) else "")

    assembly_id = assembly.get("assemblyId")
    if not isinstance(assembly_id, str):
        assembly_id = None
    return part_ids, assembly_id


def build_node_features(
    part_ids: List[str],
    part_embeddings: Dict[str, torch.Tensor],
) -> torch.Tensor:
    xs = []
    for pid in part_ids:
        if pid not in part_embeddings:
            raise KeyError(f"Missing embedding for part_id: {pid}")
        emb = part_embeddings[pid]
        if emb.ndim != 1:
            raise ValueError(f"Embedding must be 1D tensor. Got shape {tuple(emb.shape)} for {pid}")
        xs.append(emb)
    return torch.stack(xs, dim=0).float()


def make_undirected_edges_with_attr(
    edge_index_und: List[List[int]],
    edge_multiplicity: List[int],
    edge_type_counts: List[Dict[str, int]],
    mate_vocab: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    edge_attr per undirected edge: [type_count_vector..., multiplicity]
    then duplicated for both directions.
    """
    if not (len(edge_index_und) == len(edge_multiplicity) == len(edge_type_counts)):
        raise ValueError("edge_index, edge_multiplicity, edge_mate_type_counts must align")

    T = len(mate_vocab)
    if len(edge_index_und) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, T + 1), dtype=torch.float32)

    src, dst = [], []
    attrs = []

    for (u, v), mult, tdict in zip(edge_index_und, edge_multiplicity, edge_type_counts):
        vec = torch.zeros((T + 1,), dtype=torch.float32)
        if isinstance(tdict, dict):
            for t, c in tdict.items():
                if t in mate_vocab:
                    vec[mate_vocab[t]] = float(c)
        vec[T] = float(mult)

        # forward
        src.append(u); dst.append(v); attrs.append(vec)
        # backward
        src.append(v); dst.append(u); attrs.append(vec)

    edge_index_t = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_t = torch.stack(attrs, dim=0)
    return edge_index_t, edge_attr_t


class AssemblyGraphDatasetV2(Dataset):
    """
    Reads assembly_graphs_v2.jsonl and returns PyG Data with:
      x: part embeddings
      edge_attr: mate type counts + multiplicity
    """

    def __init__(
        self,
        graphs_v2_jsonl: str,
        mate_vocab_json: str,
        part_embeddings: Dict[str, torch.Tensor],
    ):
        self.graphs_v2_jsonl = graphs_v2_jsonl
        self.part_embeddings = part_embeddings

        vocab_pack = json.loads(Path(mate_vocab_json).read_text())
        self.mate_vocab = vocab_pack["vocab"]
        self.T = len(self.mate_vocab)

        self.records = []
        with open(graphs_v2_jsonl, "r") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Data:
        r = self.records[idx]
        assembly_path = r["assembly_path"]

        part_ids, assembly_id = load_assembly_part_ids_and_id(assembly_path)

        # Align to num_parts
        n = int(r["num_parts"])
        if len(part_ids) != n:
            part_ids = part_ids[:n] + [""] * max(0, n - len(part_ids))

        x = build_node_features(part_ids, self.part_embeddings)

        edge_index_t, edge_attr_t = make_undirected_edges_with_attr(
            r.get("edge_index", []),
            r.get("edge_multiplicity", []),
            r.get("edge_mate_type_counts", []),
            self.mate_vocab,
        )

        data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
        data.part_ids = part_ids
        data.assembly_path = assembly_path
        if assembly_id is not None:
            data.assembly_id = assembly_id
        return data