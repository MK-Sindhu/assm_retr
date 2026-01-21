# assembly_graph_builder.py
# Build assembly-level graph (nodes = parts, edges = mates)

import json
import torch
from torch_geometric.data import Data


# -----------------------------
# Mate type encoding (simple)
# -----------------------------
MATE_TYPE_MAP = {
    "FASTENED": 0,
    "REVOLUTE": 1,
    "SLIDER": 2,
    "CYLINDRICAL": 3,
    "PLANAR": 4
}


# -----------------------------
# STEP 1: extract unique parts
# -----------------------------
def extract_unique_part_ids(assembly_json):
    """
    Returns a list of unique part IDs in the assembly.
    """
    parts = assembly_json["parts"]
    unique = {}
    for p in parts:
        unique[p["id"]] = True
    return list(unique.keys())


# ----------------------------------------
# STEP 2: map occurrence index -> part_id
# ----------------------------------------
def build_occurrence_map(assembly_json):
    """
    occurrence_index -> part_id
    """
    parts = assembly_json["parts"]
    occ_map = {}

    for i, occ in enumerate(assembly_json["occurrences"]):
        part_idx = occ["part"]
        part_id = parts[part_idx]["id"]
        occ_map[i] = part_id

    return occ_map


# ----------------------------------------
# STEP 3: extract part-level mates
# ----------------------------------------
def extract_part_mates(assembly_json, occ_map):
    """
    Returns list of edges:
    {p1, p2, mateType}
    """
    edges = []

    for m in assembly_json["mates"]:
        o1, o2 = m["occurrences"]
        p1 = occ_map[o1]
        p2 = occ_map[o2]

        edges.append({
            "p1": p1,
            "p2": p2,
            "mateType": m["mateType"]
        })

    return edges


# -----------------------------
# STEP 4: part_id -> node index
# -----------------------------
def build_part_index(part_ids):
    return {pid: i for i, pid in enumerate(part_ids)}


# -----------------------------
# STEP 5: build edge_index
# -----------------------------
def build_edge_index(edges, part_index):
    edge_list = []

    for e in edges:
        p1, p2 = e["p1"], e["p2"]

        if p1 == p2:
            continue  # skip self-loops for now

        i = part_index[p1]
        j = part_index[p2]

        edge_list.append([i, j])
        edge_list.append([j, i])  # undirected

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edge_list, dtype=torch.long).T


# -----------------------------
# STEP 6: build edge attributes
# -----------------------------
def build_edge_attr(edges):
    attrs = []

    for e in edges:
        t = MATE_TYPE_MAP.get(e["mateType"], -1)
        attrs.append(t)
        attrs.append(t)

    if len(attrs) == 0:
        return torch.empty((0, 1), dtype=torch.long)

    return torch.tensor(attrs, dtype=torch.long).unsqueeze(1)


# -----------------------------
# STEP 7: build node features
# -----------------------------
def build_node_features(part_ids, part_embeddings):
    """
    part_embeddings: dict {part_id: embedding_tensor}
    """
    return torch.stack([part_embeddings[pid] for pid in part_ids])


# -----------------------------
# STEP 8: build assembly graph
# -----------------------------
def build_assembly_graph(
    assembly_json_path,
    part_embeddings
):
    """
    assembly_json_path: path to one assembly JSON
    part_embeddings: dict {part_id: embedding_tensor}

    returns: torch_geometric.data.Data
    """

    with open(assembly_json_path, "r") as f:
        assembly = json.load(f)

    # parts
    part_ids = extract_unique_part_ids(assembly)

    # ensure embeddings exist
    for pid in part_ids:
        if pid not in part_embeddings:
            raise ValueError(f"Missing embedding for part {pid}")

    # occurrences -> parts
    occ_map = build_occurrence_map(assembly)

    # mates
    edges = extract_part_mates(assembly, occ_map)

    # indices
    part_index = build_part_index(part_ids)

    # tensors
    x = build_node_features(part_ids, part_embeddings)
    edge_index = build_edge_index(edges, part_index)
    edge_attr = build_edge_attr(edges)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        assembly_id=assembly["assemblyId"]
    )
