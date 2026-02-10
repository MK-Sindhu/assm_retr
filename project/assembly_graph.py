# assembly_graph.py
# -----------------
# Build part-level assembly graphs from Automate-style assembly JSON files.
#
# Graph definition:
# - Nodes: parts (index = position in assembly_obj["parts"])
# - Edges: mates mapped via occurrences -> parts
# - Supports mate["occurrences"] as:
#     1) [int, int]    (occurrence indices)
#     2) [str, str]    (occurrence IDs)
#
# Output:
# - edge_index: list of [u, v] (undirected by default with u<=v)
# - edge_multiplicity: list of int counts per edge (collapsed duplicates)
#
# Notebook usage:
#   from assembly_graph import load_assembly_json, build_assembly_graph
#   obj = load_assembly_json("/path/to/assembly.json")
#   g = build_assembly_graph(obj)
#
# Command line usage:
#   python assembly_graph.py --assembly_json /path/to/assembly.json
#   python assembly_graph.py --usable_jsonl assembly_filter_out/usable_assemblies.jsonl --out_jsonl assembly_graphs_v1.jsonl

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


OccRef = Union[int, str]


@dataclass
class AssemblyGraph:
    assembly_path: Optional[str]
    num_parts: int
    num_occurrences: int
    num_mates: int
    num_edges: int
    edge_index: List[List[int]]          # [[u,v], ...]
    edge_multiplicity: List[int]         # [m1, m2, ...]
    debug: Dict[str, int]                # parsing + mapping counters


def safe_list(obj: Dict[str, Any], key: str) -> List[Any]:
    v = obj.get(key)
    return v if isinstance(v, list) else []


def load_assembly_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def build_occ_maps(assembly_obj: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    Build both maps:
    - occurrence index -> part index
    - occurrence id (string) -> part index
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
    """
    Resolve an occurrence reference to a part index.
    occ_ref may be:
    - int: occurrence index
    - str: occurrence id
    """
    if isinstance(occ_ref, int):
        return occ_index_to_part.get(occ_ref)
    if isinstance(occ_ref, str):
        return occ_id_to_part.get(occ_ref)
    return None


def build_assembly_graph(
    assembly_obj: Dict[str, Any],
    *,
    assembly_path: Optional[str] = None,
    deduplicate: bool = True,
    allow_self_edges: bool = False,
) -> AssemblyGraph:
    """
    Construct a part-level assembly graph from an assembly JSON object.

    Parameters
    ----------
    deduplicate:
        If True, collapse multiple mates between the same unordered part pair.
        Edge multiplicity stores how many mates collapsed.
    allow_self_edges:
        If False, skip edges where both endpoints resolve to the same part index.

    Returns
    -------
    AssemblyGraph
    """
    parts = safe_list(assembly_obj, "parts")
    occs = safe_list(assembly_obj, "occurrences")
    mates = safe_list(assembly_obj, "mates")

    occ_index_to_part, occ_id_to_part = build_occ_maps(assembly_obj)

    pair_counts: Counter[Tuple[int, int]] = Counter()
    debug: Counter[str] = Counter()

    for mate in mates:
        if not isinstance(mate, dict):
            debug["mate_not_dict"] += 1
            continue

        occ_pair = mate.get("occurrences")
        if not (isinstance(occ_pair, list) and len(occ_pair) == 2):
            debug["mate_occurrences_missing_or_not2"] += 1
            continue

        a, b = occ_pair[0], occ_pair[1]

        # classify occurrence reference format
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

        if deduplicate:
            u, v = (pa, pb) if pa <= pb else (pb, pa)
        else:
            u, v = pa, pb

        pair_counts[(u, v)] += 1
        debug["edge_added"] += 1

    edges = list(pair_counts.keys())
    multiplicity = [pair_counts[e] for e in edges]

    # sanity: out of range edges
    n_parts = len(parts)
    out_of_range = 0
    for u, v in edges:
        if not (0 <= u < n_parts and 0 <= v < n_parts):
            out_of_range += 1
    if out_of_range:
        debug["edge_out_of_range"] += out_of_range

    return AssemblyGraph(
        assembly_path=assembly_path,
        num_parts=len(parts),
        num_occurrences=len(occs),
        num_mates=len(mates),
        num_edges=len(edges),
        edge_index=[[u, v] for (u, v) in edges],
        edge_multiplicity=multiplicity,
        debug=dict(debug),
    )


def load_usable_paths(usable_jsonl: Union[str, Path], limit: Optional[int] = None) -> List[str]:
    """
    Load 'assembly_path' entries from usable_assemblies.jsonl (one JSON per line).
    """
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


def write_graphs_jsonl(
    usable_jsonl: Union[str, Path],
    out_jsonl: Union[str, Path],
    *,
    limit: Optional[int] = None,
    deduplicate: bool = True,
    allow_self_edges: bool = False,
    progress_every: int = 1000,
) -> Dict[str, int]:
    """
    Read usable assemblies and write one AssemblyGraph record per line to out_jsonl.
    Returns aggregated debug counters.
    """
    paths = load_usable_paths(usable_jsonl, limit=limit)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    agg = Counter()
    written = 0

    with out_jsonl.open("w") as f_out:
        for p in paths:
            obj = load_assembly_json(p)
            g = build_assembly_graph(
                obj,
                assembly_path=p,
                deduplicate=deduplicate,
                allow_self_edges=allow_self_edges,
            )
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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build assembly graphs from Automate assembly JSON files.")
    parser.add_argument("--assembly_json", type=str, default=None, help="Path to a single assembly JSON file.")
    parser.add_argument("--usable_jsonl", type=str, default=None, help="Path to usable_assemblies.jsonl.")
    parser.add_argument("--out_jsonl", type=str, default=None, help="Output JSONL path for graphs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of assemblies processed.")
    parser.add_argument("--no_dedup", action="store_true", help="Disable edge deduplication.")
    parser.add_argument("--allow_self_edges", action="store_true", help="Allow edges where both endpoints are same part.")
    parser.add_argument("--progress_every", type=int, default=1000, help="Print progress every N graphs.")
    args = parser.parse_args()

    deduplicate = not args.no_dedup

    if args.assembly_json:
        obj = load_assembly_json(args.assembly_json)
        g = build_assembly_graph(
            obj,
            assembly_path=args.assembly_json,
            deduplicate=deduplicate,
            allow_self_edges=args.allow_self_edges,
        )
        print(json.dumps(asdict(g), indent=2))
        return

    if args.usable_jsonl and args.out_jsonl:
        agg = write_graphs_jsonl(
            args.usable_jsonl,
            args.out_jsonl,
            limit=args.limit,
            deduplicate=deduplicate,
            allow_self_edges=args.allow_self_edges,
            progress_every=args.progress_every,
        )
        print("Aggregated counters:")
        print(json.dumps(agg, indent=2))
        return

    parser.error("Provide either --assembly_json OR (--usable_jsonl and --out_jsonl).")


if __name__ == "__main__":
    _main()
