# make_missing_part_samples.py
# ----------------------------
# Build missing-part retrieval samples.
#
# For each assembly graph:
# - choose a removable node (prefer degree > 0)
# - remove the node and its incident edges => observed graph G_obs
# - label is removed part_id
#
# Output JSONL fields:
# {
#   "assembly_path": str,
#   "assembly_id": str|None,
#   "missing_part_id": str,
#   "missing_node_old_index": int,
#   "observed_part_ids": [str],                 # length N-1
#   "edge_index": [[u,v], ...],                 # undirected, indices into observed_part_ids
#   "edge_attr": [[...], ...],                  # aligned with edge_index, mate_type_counts + multiplicity
#   "missing_neighbors_old_indices": [int],     # neighbors in original graph (for analysis/hard negatives)
# }
#
# Notes:
# - Uses v2 graph file (which already contains edge mate-type counts).
# - Builds fixed-length edge_attr using mate_vocab.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch


def load_mate_vocab(path: str) -> Dict[str, int]:
    pack = json.loads(Path(path).read_text())
    return pack["vocab"]


def build_edge_attr_vec(tdict: Dict[str, int], mult: int, mate_vocab: Dict[str, int]) -> List[float]:
    T = len(mate_vocab)
    vec = [0.0] * (T + 1)
    if isinstance(tdict, dict):
        for t, c in tdict.items():
            if t in mate_vocab:
                vec[mate_vocab[t]] = float(c)
    vec[T] = float(mult)
    return vec


def undirected_degree(n: int, edges: List[List[int]]) -> List[int]:
    deg = [0] * n
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            deg[u] += 1
            deg[v] += 1
    return deg


def load_assembly_id(assembly_path: str) -> str | None:
    try:
        with open(assembly_path, "r") as f:
            a = json.load(f)
        aid = a.get("assemblyId")
        return aid if isinstance(aid, str) else None
    except Exception:
        return None


def choose_missing_node(n: int, edges: List[List[int]], rng: random.Random) -> int:
    deg = undirected_degree(n, edges)
    candidates = [i for i in range(n) if deg[i] > 0]
    if candidates:
        return rng.choice(candidates)
    return rng.randrange(n)


def neighbors_of(node: int, edges: List[List[int]]) -> List[int]:
    nbrs = set()
    for u, v in edges:
        if u == node:
            nbrs.add(v)
        elif v == node:
            nbrs.add(u)
    return sorted(nbrs)


def build_observed_graph(
    part_ids: List[str],
    edges: List[List[int]],
    edge_attr: List[List[float]],
    remove_idx: int,
) -> Tuple[List[str], List[List[int]], List[List[float]], Dict[int, int]]:
    """
    Remove node remove_idx and incident edges, then reindex remaining nodes.

    Returns:
      observed_part_ids
      observed_edges (undirected)
      observed_edge_attr
      old_to_new mapping for kept nodes
    """
    n = len(part_ids)

    kept = [i for i in range(n) if i != remove_idx]
    old_to_new = {old: new for new, old in enumerate(kept)}

    obs_part_ids = [part_ids[i] for i in kept]

    obs_edges = []
    obs_attr = []
    for (u, v), a in zip(edges, edge_attr):
        if u == remove_idx or v == remove_idx:
            continue
        if u not in old_to_new or v not in old_to_new:
            continue
        uu = old_to_new[u]
        vv = old_to_new[v]
        # keep undirected as (min,max) for determinism
        if uu <= vv:
            obs_edges.append([uu, vv])
        else:
            obs_edges.append([vv, uu])
        obs_attr.append(a)

    # optional: sort edges for stable output
    order = sorted(range(len(obs_edges)), key=lambda i: (obs_edges[i][0], obs_edges[i][1]))
    obs_edges = [obs_edges[i] for i in order]
    obs_attr = [obs_attr[i] for i in order]

    return obs_part_ids, obs_edges, obs_attr, old_to_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_v2_jsonl", type=str, required=True)
    ap.add_argument("--mate_vocab_json", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_n_leq", type=int, default=1, help="Skip assemblies with num_parts <= this")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mate_vocab = load_mate_vocab(args.mate_vocab_json)

    inp = Path(args.graphs_v2_jsonl)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with inp.open() as f_in, out.open("w") as f_out:
        for i, line in enumerate(f_in):
            if args.limit is not None and i >= args.limit:
                break

            r = json.loads(line)
            n = int(r["num_parts"])
            if n <= args.skip_n_leq:
                skipped += 1
                continue

            assembly_path = r["assembly_path"]
            assembly_id = load_assembly_id(assembly_path)

            # Load part IDs from assembly JSON to align indices
            with open(assembly_path, "r") as af:
                assembly = json.load(af)
            parts = assembly.get("parts", [])
            part_ids = [p.get("id", "") for p in parts]
            if len(part_ids) != n:
                part_ids = part_ids[:n] + [""] * max(0, n - len(part_ids))

            edges = r.get("edge_index", [])
            mult = r.get("edge_multiplicity", [])
            tcounts = r.get("edge_mate_type_counts", [])

            # Build edge_attr vectors aligned to edges
            edge_attr = [build_edge_attr_vec(td, m, mate_vocab) for td, m in zip(tcounts, mult)]

            remove_idx = choose_missing_node(n, edges, rng)
            missing_part_id = part_ids[remove_idx]
            if not isinstance(missing_part_id, str) or missing_part_id == "":
                skipped += 1
                continue

            missing_nbrs = neighbors_of(remove_idx, edges)

            obs_part_ids, obs_edges, obs_attr, old_to_new = build_observed_graph(
                part_ids, edges, edge_attr, remove_idx
            )

            rec = {
                "assembly_path": assembly_path,
                "assembly_id": assembly_id,
                "missing_part_id": missing_part_id,
                "missing_node_old_index": remove_idx,
                "missing_neighbors_old_indices": missing_nbrs,
                "observed_part_ids": obs_part_ids,
                "edge_index": obs_edges,
                "edge_attr": obs_attr,
            }

            f_out.write(json.dumps(rec) + "\n")
            written += 1

            if written % 1000 == 0:
                print("Written:", written)

    print("Done.")
    print("Written:", written)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()