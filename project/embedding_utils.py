import os
import torch

def index_part_graphs(PART_GRAPH_DIR):
    """
    Scan PART_GRAPH_DIR once and return:
      part_id -> full_path_to_pt
    """
    idx = {}
    for root, _, files in os.walk(PART_GRAPH_DIR):
        for f in files:
            if f.endswith(".pt"):
                pid = f[:-3]  # remove .pt
                idx[pid] = os.path.join(root, f)
    return idx

def load_part_embedding_from_index(part_id, part_index, reduce="mean"):
    """
    Returns a 1D embedding tensor for a part_id, or None if not found.
    """
    path = part_index.get(part_id)
    if path is None:
        return None

    data = torch.load(path, map_location="cpu")

    # Common: data.x is [num_nodes, feat_dim]
    if not hasattr(data, "x") or data.x is None:
        return None

    x = data.x
    if reduce == "mean":
        return x.mean(dim=0)
    elif reduce == "sum":
        return x.sum(dim=0)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")
