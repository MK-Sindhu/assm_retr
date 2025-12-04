"""
part_graph.py

Utilities to convert CAD parts (STEP files) to face-level graphs
and then into PyTorch Geometric Data objects.

- Nodes  = faces
- Edges  = face adjacency (share an edge)
- Features per node:
    [area, centroid_x, centroid_y, centroid_z,
     normal_x, normal_y, normal_z,
     one-hot(surface_type: plane/cylinder/cone/sphere/torus/freeform/other)]
"""

import os
from typing import List, Dict

import numpy as np
import torch
from torch_geometric.data import Data

# OCC / pythonOCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TopExp import topexp, TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE

from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.TopTools import TopTools_IndexedMapOfShape

from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface
)

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Vec


# ------------------ STEP LOADING ------------------ #

def load_step_shape(path: str):
    """Load a STEP file and return an OpenCascade shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


# ------------------ FACE FEATURES ------------------ #

def face_area_centroid(face: TopoDS_Face):
    """Return (area, (cx, cy, cz)) for a face."""
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    area = props.Mass()
    c = props.CentreOfMass()
    centroid = (c.X(), c.Y(), c.Z())
    return area, centroid


def face_surface_type(face: TopoDS_Face) -> str:
    """Classify a face's surface type."""
    surf = BRepAdaptor_Surface(face)
    t = surf.GetType()
    if t == GeomAbs_Plane:
        return "plane"
    if t == GeomAbs_Cylinder:
        return "cylinder"
    if t == GeomAbs_Cone:
        return "cone"
    if t == GeomAbs_Sphere:
        return "sphere"
    if t == GeomAbs_Torus:
        return "torus"
    if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
        return "freeform"
    return "other"


SURF_TYPE_MAP: Dict[str, int] = {
    "plane": 0,
    "cylinder": 1,
    "cone": 2,
    "sphere": 3,
    "torus": 4,
    "freeform": 5,
    "other": 6,
}


def surf_type_onehot(s_type: str) -> np.ndarray:
    """Return one-hot encoding of surface type."""
    k = len(SURF_TYPE_MAP)
    vec = np.zeros(k, dtype=np.float32)
    idx = SURF_TYPE_MAP.get(s_type, SURF_TYPE_MAP["other"])
    vec[idx] = 1.0
    return vec


def build_face_normal_fn(shape):
    """
    Create a face_normal(face) function tied to this shape.
    Uses triangulation, robust to different pythonOCC versions.
    """
    # mesh once for the whole shape
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()

    def face_normal(face: TopoDS_Face):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            return (0.0, 0.0, 1.0)

        normal = gp_Vec(0, 0, 0)

        # --- robust node accessor ---
        if hasattr(tri, "Nodes"):
            try:
                nodes_arr = tri.Nodes()
                get_node = lambda idx: nodes_arr.Value(idx)
            except Exception:
                get_node = lambda idx: tri.Node(idx)
        else:
            get_node = lambda idx: tri.Node(idx)

        # --- robust triangle accessor ---
        if hasattr(tri, "Triangles"):
            try:
                tris_arr = tri.Triangles()
                get_tri = lambda i: tris_arr.Value(i)
            except Exception:
                get_tri = lambda i: tri.Triangle(i)
        else:
            get_tri = lambda i: tri.Triangle(i)

        for i in range(1, tri.NbTriangles() + 1):
            t = get_tri(i)
            try:
                n1, n2, n3 = t.Get()
            except Exception:
                n1, n2, n3 = t.Value(1), t.Value(2), t.Value(3)

            p1 = get_node(n1)
            p2 = get_node(n2)
            p3 = get_node(n3)

            v1 = gp_Vec(p1, p2)
            v2 = gp_Vec(p1, p3)
            nrm = v1.Crossed(v2)
            normal = normal.Added(nrm)

        if normal.Magnitude() > 1e-9:
            normal.Normalize()
            return (normal.X(), normal.Y(), normal.Z())
        return (0.0, 0.0, 1.0)

    return face_normal


# ------------------ MAIN: STEP -> PyG ------------------ #

def build_part_pyg_from_step(step_path: str) -> Data:
    """
    Load a STEP CAD file and return a PyTorch Geometric Data object.

    Nodes = faces
    Edges = adjacency between faces that share an edge
    """
    shape = load_step_shape(step_path)

    # map faces to indices: 1..N
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    num_faces = face_map.Size()
    if num_faces == 0:
        raise RuntimeError(f"No faces found in STEP file: {step_path}")

    # map edges to indices
    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

    # normals
    face_normal = build_face_normal_fn(shape)

    # build edge -> faces map
    edge_to_faces: Dict[int, set] = {}
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    while exp_face.More():
        f = exp_face.Current()
        f_idx = face_map.FindIndex(f)
        exp_edge = TopExp_Explorer(f, TopAbs_EDGE)
        while exp_edge.More():
            e = exp_edge.Current()
            e_idx = edge_map.FindIndex(e)
            edge_to_faces.setdefault(e_idx, set()).add(f_idx)
            exp_edge.Next()
        exp_face.Next()

    # -------- node features --------
    node_features: List[List[float]] = []
    for i in range(1, num_faces + 1):
        face = face_map.FindKey(i)
        area, centroid = face_area_centroid(face)
        nx, ny, nz = face_normal(face)
        s_type = face_surface_type(face)
        st_oh = surf_type_onehot(s_type)

        feats = [
            float(area),
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            float(nx),
            float(ny),
            float(nz),
        ] + st_oh.tolist()

        node_features.append(feats)

    x = torch.tensor(node_features, dtype=torch.float)

    # -------- edges (adjacency) --------
    edge_list: List[List[int]] = []
    for faces in edge_to_faces.values():
        faces = list(faces)
        if len(faces) > 1:
            for a in range(len(faces)):
                for b in range(a + 1, len(faces)):
                    u = faces[a] - 1  # 0-based index
                    v = faces[b] - 1
                    edge_list.append([u, v])
                    edge_list.append([v, u])  # undirected

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data.step_path = step_path  # store path as metadata (optional)
    return data


# ------------------ BATCH CONVERSION ------------------ #

def convert_all_steps_in_dir(step_dir: str, out_dir: str):
    """
    Convert all .step files in a directory to PyG graphs and save them.

    - step_dir: directory containing *.step files
    - out_dir : directory to write *.pt files (torch.save(Data, ...))
    """
    os.makedirs(out_dir, exist_ok=True)

    step_files = [
        f for f in os.listdir(step_dir)
        if f.lower().endswith(".step")
    ]

    print(f"Found {len(step_files)} STEP files in {step_dir}")

    for f in step_files:
        step_path = os.path.join(step_dir, f)
        base = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, base + ".pt")

        try:
            data = build_part_pyg_from_step(step_path)
            torch.save(data, out_path)
            print(f"[OK] {f} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {f}: {e}")


if __name__ == "__main__":
    # Example CLI usage (optional)
    import argparse

    parser = argparse.ArgumentParser(description="Convert STEP parts to PyG graphs.")
    parser.add_argument("--step_dir", type=str, required=True, help="Directory with .step files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for .pt graph files")
    args = parser.parse_args()

    convert_all_steps_in_dir(args.step_dir, args.out_dir)
