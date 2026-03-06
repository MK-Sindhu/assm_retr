# assembly_graph_v2.py
# --------------------
# Build assembly graphs (nodes = parts, edges = mates) with mate-type counts per edge.
#
# Input:
# - assembly_filter_out/usable_assemblies.jsonl
# - Automate assembly JSON files referenced by assembly_path
#
# Output:
# - assembly_filter_out/assembly_graphs_v2.jsonl
#
# Each output line:
# {
#   "assembly_path": ...,
#   "num_parts": ...,
#   "num_occurrences": ...,
#   "num_mates": ...,
#   "num_edges": ...,
#   "edge_index": [[u,v], ...],                      # undirected (u<=v)
#   "edge_multiplicity": [m1, m2, ...],              # total mates per pair
#   "edge_mate_type_counts": [{"BALL":1}, {...}, ...],
#   "debug": {...}
# }

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

OccRef = Union[int, str]


def safe_list(obj: Dict[str, Any], key: str) -> List[Any]:
    v = obj.get(key)
    return v if isinstance(v, list) else []


def load_assembly_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def build_occ_maps(assembly_obj: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    occurrence index -> part index
    occurrence id (string) -> part index
    """
    occs = safe_list(assembly_obj, "occurrences")
    occ_index_to_part: Dict[int, int] = {}
    occ_id_to_part: Dict[str, int] = {}

    for i, occ in enumerate(occs):
        if not isinstance(occ, dict):
            continue
        part_idx = occ.get("part")
        occ_id = occ.get("id")
        if isinstance(part_idx, int):
            occ_index_to_part[i] = part_idx
            if isinstance(occ_id, str):
                occ_id_to_part[occ_id] = part_idx

    return occ_index_to_part, occ_id_to_part


def resolve_occ_ref(
    occ_ref: OccRef,
    occ_index_to_part: Dict[int, int],
    occ_id_to_part: Dict[str, int],
) -> Optional[int]:
    if isinstance(occ_ref, int):
        return occ_index_to_part.get(occ_ref)
    if isinstance(occ_ref, str):
        return occ_id_to_part.get(occ_ref)
    return None


@dataclass
class AssemblyGraphV2:
    assembly_path: str
    num_parts: int
    num_occurrences: int
    num_mates: int
    num_edges: int
    edge_index: List[List[int]]
    edge_multiplicity: List[int]
    edge_mate_type_counts: List[Dict[str, int]]
    debug: Dict[str, int]


def build_assembly_graph_v2(
    assembly_obj: Dict[str, Any],
    *,
    assembly_path: str,
    allow_self_edges: bool = False,
) -> AssemblyGraphV2:
    parts = safe_list(assembly_obj, "parts")
    occs = safe_list(assembly_obj, "occurrences")
    mates = safe_list(assembly_obj, "mates")

    occ_index_to_part, occ_id_to_part = build_occ_maps(assembly_obj)

    # For each undirected part pair, count:
    # - total mates (multiplicity)
    # - mate types (histogram)
    pair_total = Counter()  # (u,v) -> count
    pair_types = defaultdict(Counter)  # (u,v) -> Counter(mateType -> count)

    debug = Counter()

    for mate in mates:
        if not isinstance(mate, dict):
            debug["mate_not_dict"] += 1
            continue

        occ_pair = mate.get("occurrences")
        if not (isinstance(occ_pair, list) and len(occ_pair) == 2):
            debug["mate_occurrences_missing_or_not2"] += 1
            continue

        a, b = occ_pair[0], occ_pair[1]

        if isinstance(a, int) and isinstance(b, int):
            debug["used_int_occurrences"] += 1
        elif isinstance(a, str) and isinstance(b, str):
            debug["used_str_occurrences"] += 1
        else:
            debug["mixed_or_unknown_occurrence_types"] += 1
            continue

        pa = resolve_occ_ref(a, occ_index_to_part, occ_id_to_part)
        pb = resolve_occ_ref(b, occ_index_to_part, occ_id_to_part)
        if pa is None or pb is None:
            debug["occ_not_resolved"] += 1
            continue

        if (not allow_self_edges) and pa == pb:
            debug["self_edge_skipped"] += 1
            continue

        u, v = (pa, pb) if pa <= pb else (pb, pa)

        mate_type = mate.get("mateType")
        if not isinstance(mate_type, str):
            mate_type = "UNKNOWN"
            debug["missing_mate_type"] += 1

        pair_total[(u, v)] += 1
        pair_types[(u, v)][mate_type] += 1
        debug["edge_added"] += 1

    # deterministic ordering
    edges = sorted(pair_total.keys())
    multiplicity = [int(pair_total[e]) for e in edges]
    type_counts = [dict(pair_types[e]) for e in edges]

    # range check
    n_parts = len(parts)
    out_of_range = 0
    for u, v in edges:
        if not (0 <= u < n_parts and 0 <= v < n_parts):
            out_of_range += 1
    if out_of_range:
        debug["edge_out_of_range"] += out_of_range

    return AssemblyGraphV2(
        assembly_path=assembly_path,
        num_parts=len(parts),
        num_occurrences=len(occs),
        num_mates=len(mates),
        num_edges=len(edges),
        edge_index=[[u, v] for (u, v) in edges],
        edge_multiplicity=multiplicity,
        edge_mate_type_counts=type_counts,
        debug=dict(debug),
    )


def load_usable_paths(usable_jsonl: Union[str, Path], limit: Optional[int] = None) -> List[str]:
    usable_jsonl = Path(usable_jsonl)
    paths: List[str] = []
    with usable_jsonl.open("r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            p = rec.get("assembly_path")
            if isinstance(p, str):
                paths.append(p)
    return paths


def write_graphs_v2_jsonl(
    usable_jsonl: Union[str, Path],
    out_jsonl: Union[str, Path],
    *,
    limit: Optional[int] = None,
    allow_self_edges: bool = False,
    progress_every: int = 500,
) -> Dict[str, int]:
    paths = load_usable_paths(usable_jsonl, limit=limit)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    agg = Counter()
    written = 0

    with out_jsonl.open("w") as f_out:
        for p in paths:
            obj = load_assembly_json(p)
            g = build_assembly_graph_v2(obj, assembly_path=p, allow_self_edges=allow_self_edges)

            f_out.write(json.dumps(asdict(g)) + "\n")
            written += 1

            agg["assemblies"] += 1
            agg["total_parts"] += g.num_parts
            agg["total_mates"] += g.num_mates
            agg["total_edges"] += g.num_edges
            agg.update(g.debug)

            if progress_every > 0 and written % progress_every == 0:
                print(f"Written: {written}")

    return dict(agg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build assembly graphs v2 with mate type counts per edge.")
    ap.add_argument("--usable_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow_self_edges", action="store_true")
    ap.add_argument("--progress_every", type=int, default=500)
    args = ap.parse_args()

    agg = write_graphs_v2_jsonl(
        args.usable_jsonl,
        args.out_jsonl,
        limit=args.limit,
        allow_self_edges=args.allow_self_edges,
        progress_every=args.progress_every,
    )
    print("Aggregated counters:")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()