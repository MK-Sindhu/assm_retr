# part_graph_index.py
# -------------------
# Index part graph files under a directory and map:
#   part_id (string) -> graph file path
#
# Assumption:
# - Part graph filenames contain the part_id as the stem (common in many pipelines),
#   OR the saved torch object contains part_id in attributes.
#
# This implementation supports BOTH:
# - If filename stem matches part_id: use it directly
# - Else: try reading the torch file and extracting data.part_id or data["part_id"]

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union
import torch


def _extract_part_id_from_obj(obj) -> Optional[str]:
    # PyTorch Geometric Data typically has attributes
    if hasattr(obj, "part_id"):
        pid = getattr(obj, "part_id")
        return pid if isinstance(pid, str) else None

    # Sometimes saved as dict
    if isinstance(obj, dict):
        for k in ["part_id", "id", "partId"]:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        # or nested
        if "data" in obj:
            return _extract_part_id_from_obj(obj["data"])

    return None


def index_part_graphs(
    root_dir: Union[str, Path],
    *,
    exts=(".pt", ".pth", ".pkl"),
    try_read_if_needed: bool = False,
) -> Dict[str, str]:
    """
    Returns dict: part_id -> file_path

    Strategy:
    1) Prefer filename stem as part_id.
    2) Optionally (if try_read_if_needed=True), read file and try to extract part_id.
    """
    root_dir = Path(root_dir)
    out: Dict[str, str] = {}

    files = []
    for ext in exts:
        files.extend(root_dir.rglob(f"*{ext}"))

    for fp in files:
        stem = fp.stem
        # Most common: filename is the part id
        if stem not in out:
            out[stem] = str(fp)

    if try_read_if_needed:
        # Second pass: validate and fix using actual saved content
        fixed = {}
        for pid_guess, fp in out.items():
            try:
                obj = torch.load(fp, map_location="cpu")
            except Exception:
                continue
            pid_true = _extract_part_id_from_obj(obj)
            if pid_true and pid_true != pid_guess:
                fixed[pid_true] = fp
        # merge (true id wins)
        out.update(fixed)

    return out
