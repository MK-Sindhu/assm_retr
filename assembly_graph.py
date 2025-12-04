"""
assembly_graph.py

Build assembly-level graphs (occurrence-level) from AutoMate assembly JSON files.

Nodes = occurrences (each instance of a part in an assembly)
Edges = mates between occurrences
Node features (Option A - minimal):
    - has_step (1)
    - fixed (1)
    - transform (16 values)
Total node feature dim = 18

Edges have:
    - mateType_id (integer)
"""

import os
import json
import torch
from torch_geometric.data import Data

# -------------------------------------------
# Mate Type Encoding (minimal)
# -------------------------------------------
MATE_TYPE_MAP = {
    "REVOLUTE": 0,
    "SLIDER": 1,
    "CYLINDRICAL": 2,
    "PLANAR": 3,
    "BALL": 4,
    "FASTENED": 5,
    "GEAR": 6,
}

def encode_mate_type(mate_type: str) -> int:
    return MATE_TYPE_MAP.get(mate_type, len(MATE_TYPE_MAP))  # unknown last index


# -------------------------------------------
# Load a single assembly JSON
# -------------------------------------------
def load_assembly_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------
# Build Assembly Graph (Minimal Features)
# -------------------------------------------
def build_assembly_graph(json_path: str) -> Data:
    asm = load_assembly_json(json_path)

    parts = asm["parts"]
    occurrences = asm["occurrences"]
    mates = asm["mates"]

    num_occ = len(occurrences)

    # ----------------------
    # Node features
    # ----------------------
    node_features = []

    for occ in occurrences:
        part_idx = occ["part"]
        part_info = parts[part_idx]

        # minimal features:
        has_step = float(occ["has_step"])
        fixed_flag = float(occ["fixed"])
        transform = occ["transform"]  # list of 16 floats

        # Final node vector (18 dims)
        feat = [has_step, fixed_flag] + transform

        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)


    # ----------------------
    # Edges + attributes
    # ----------------------
    edge_list = []
    edge_attr_list = []

    for mate in mates:
        occ_a, occ_b = mate["occurrences"]
        mtype = mate["mateType"]
        mtype_id = encode_mate_type(mtype)

        # undirected graph
        edge_list.append([occ_a, occ_b])
        edge_list.append([occ_b, occ_a])

        edge_attr_list.append([mtype_id])
        edge_attr_list.append([mtype_id])

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

    # Build PyG object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    data.num_nodes = num_occ
    data.assembly_id = asm["assemblyId"]

    return data


# -------------------------------------------
# Convert all assemblies/*.json into .pt graphs
# -------------------------------------------
def convert_all_assemblies_in_dir(json_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    json_files = [
        f for f in os.listdir(json_dir)
        if f.lower().endswith(".json")
    ]

    print(f"Found {len(json_files)} assembly JSONs in {json_dir}")

    for f in json_files:
        json_path = os.path.join(json_dir, f)
        asm_id = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, asm_id + ".pt")

        try:
            data = build_assembly_graph(json_path)
            torch.save(data, out_path)
            print(f"[OK] {f}  ->  {out_path}")
        except Exception as e:
            print(f"[FAIL] {f}: {e}")


# Optional CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    convert_all_assemblies_in_dir(args.json_dir, args.out_dir)
