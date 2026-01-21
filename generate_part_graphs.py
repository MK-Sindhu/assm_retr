# """
# part_graph.py (SHARDED VERSION)

# Converts "Safe" STEP files into PyTorch Geometric Data objects.
# Output is organized into subfolders of 10,000 files each.
# """

# import os
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from typing import List, Dict
# from torch_geometric.data import Data

# # --- OCC IMPORTS ---
# from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Core.IFSelect import IFSelect_RetDone
# from OCC.Core.TopExp import topexp, TopExp_Explorer
# from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
# from OCC.Core.TopoDS import TopoDS_Face
# from OCC.Core.TopTools import TopTools_IndexedMapOfShape
# from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
# from OCC.Core.GProp import GProp_GProps
# from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
# from OCC.Core.GeomAbs import (
#     GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
#     GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
#     GeomAbs_BezierSurface
# )
# from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# from OCC.Core.BRep import BRep_Tool
# from OCC.Core.TopLoc import TopLoc_Location
# from OCC.Core.gp import gp_Vec

# # =============================================================================
# # 1. CORE LOGIC (Feature Extraction)
# # =============================================================================

# def load_step_shape(path: str):
#     reader = STEPControl_Reader()
#     status = reader.ReadFile(path)
#     if status != IFSelect_RetDone:
#         raise RuntimeError(f"Failed to read STEP file: {path}")
#     reader.TransferRoots()
#     return reader.OneShape()

# def face_area_centroid(face: TopoDS_Face):
#     props = GProp_GProps()
#     brepgprop_SurfaceProperties(face, props)
#     area = props.Mass()
#     c = props.CentreOfMass()
#     return area, (c.X(), c.Y(), c.Z())

# def face_surface_type(face: TopoDS_Face) -> str:
#     surf = BRepAdaptor_Surface(face)
#     t = surf.GetType()
#     if t == GeomAbs_Plane: return "plane"
#     if t == GeomAbs_Cylinder: return "cylinder"
#     if t == GeomAbs_Cone: return "cone"
#     if t == GeomAbs_Sphere: return "sphere"
#     if t == GeomAbs_Torus: return "torus"
#     if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface): return "freeform"
#     return "other"

# SURF_TYPE_MAP = {
#     "plane": 0, "cylinder": 1, "cone": 2, "sphere": 3,
#     "torus": 4, "freeform": 5, "other": 6,
# }

# def surf_type_onehot(s_type: str) -> np.ndarray:
#     vec = np.zeros(len(SURF_TYPE_MAP), dtype=np.float32)
#     vec[SURF_TYPE_MAP.get(s_type, 6)] = 1.0
#     return vec

# def build_face_normal_fn(shape):
#     mesh = BRepMesh_IncrementalMesh(shape, 0.1)
#     mesh.Perform()

#     def face_normal(face: TopoDS_Face):
#         loc = TopLoc_Location()
#         tri = BRep_Tool.Triangulation(face, loc)
#         if tri is None: return (0.0, 0.0, 1.0)

#         normal = gp_Vec(0, 0, 0)
        
#         # Robust accessor logic
#         if hasattr(tri, "Nodes"): get_node = lambda i: tri.Nodes().Value(i)
#         else: get_node = lambda i: tri.Node(i)
        
#         if hasattr(tri, "Triangles"): get_tri = lambda i: tri.Triangles().Value(i)
#         else: get_tri = lambda i: tri.Triangle(i)

#         for i in range(1, tri.NbTriangles() + 1):
#             t = get_tri(i)
#             try: n1, n2, n3 = t.Get()
#             except: n1, n2, n3 = t.Value(1), t.Value(2), t.Value(3)
            
#             p1, p2, p3 = get_node(n1), get_node(n2), get_node(n3)
#             v1, v2 = gp_Vec(p1, p2), gp_Vec(p1, p3)
#             normal = normal.Added(v1.Crossed(v2))

#         if normal.Magnitude() > 1e-9:
#             normal.Normalize()
#             return (normal.X(), normal.Y(), normal.Z())
#         return (0.0, 0.0, 1.0)

#     return face_normal

# def build_part_pyg_from_step(step_path: str) -> Data:
#     shape = load_step_shape(step_path)

#     # 1. Map Faces
#     face_map = TopTools_IndexedMapOfShape()
#     topexp.MapShapes(shape, TopAbs_FACE, face_map)
#     num_faces = face_map.Size()
#     if num_faces == 0: raise RuntimeError(f"No faces: {step_path}")

#     # 2. Map Edges
#     edge_map = TopTools_IndexedMapOfShape()
#     topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

#     # 3. Adjacency
#     edge_to_faces = {}
#     exp_face = TopExp_Explorer(shape, TopAbs_FACE)
#     while exp_face.More():
#         f = exp_face.Current()
#         f_idx = face_map.FindIndex(f)
#         exp_edge = TopExp_Explorer(f, TopAbs_EDGE)
#         while exp_edge.More():
#             e = exp_edge.Current()
#             e_idx = edge_map.FindIndex(e)
#             edge_to_faces.setdefault(e_idx, set()).add(f_idx)
#             exp_edge.Next()
#         exp_face.Next()

#     # 4. Features
#     face_normal = build_face_normal_fn(shape)
#     node_features = []
#     for i in range(1, num_faces + 1):
#         face = face_map.FindKey(i)
#         area, centroid = face_area_centroid(face)
#         nx, ny, nz = face_normal(face)
#         s_type = face_surface_type(face)
#         st_oh = surf_type_onehot(s_type)
        
#         feats = [float(area), float(centroid[0]), float(centroid[1]), float(centroid[2]),
#                  float(nx), float(ny), float(nz)] + st_oh.tolist()
#         node_features.append(feats)

#     x = torch.tensor(node_features, dtype=torch.float)

#     # 5. Edges
#     edge_list = []
#     for faces in edge_to_faces.values():
#         faces = list(faces)
#         if len(faces) > 1:
#             for a in range(len(faces)):
#                 for b in range(a + 1, len(faces)):
#                     u, v = faces[a] - 1, faces[b] - 1
#                     edge_list.append([u, v])
#                     edge_list.append([v, u])

#     if not edge_list: edge_index = torch.empty((2, 0), dtype=torch.long)
#     else: edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

#     return Data(x=x, edge_index=edge_index, step_path=step_path)


# # =============================================================================
# # 2. EXECUTION (SHARDED OUTPUT)
# # =============================================================================

# def main():
#     # --- CONFIGURATION ---
#     INPUT_PARQUET = "parts_trainable.parquet"  
#     OUTPUT_BASE = "graphs/parts"  # Will create graphs/parts/part_gr_00000/ etc.
#     CHUNK_SIZE = 10000
#     # ---------------------

#     if not os.path.exists(INPUT_PARQUET):
#         print(f"❌ Error: Missing '{INPUT_PARQUET}'")
#         return

#     print(f"1. Loading manifest: {INPUT_PARQUET}")
#     df = pd.read_parquet(INPUT_PARQUET)
    
#     print(f"2. Output Base: {os.path.abspath(OUTPUT_BASE)}")
#     print(f"3. Processing {len(df)} parts into folders of {CHUNK_SIZE}...")

#     success = 0
#     errors = 0
    
#     # Enumerate helps us calculate the folder index easily
#     for i, (idx, row) in tqdm(enumerate(df.iterrows()), total=len(df), unit="part"):
#         part_id = row['part_id']
#         step_path = row['absolute_path']

#         # --- SHARDING LOGIC ---
#         folder_index = i // CHUNK_SIZE
#         subfolder_name = f"part_gr_{folder_index:05d}" # e.g., part_gr_00000
#         full_out_dir = os.path.join(OUTPUT_BASE, subfolder_name)
        
#         # Create folder only if needed (check once per chunk ideally, but this is safe)
#         os.makedirs(full_out_dir, exist_ok=True)
#         # ----------------------
        
#         out_path = os.path.join(full_out_dir, f"{part_id}.pt")
        
#         if os.path.exists(out_path):
#             continue
            
#         try:
#             data = build_part_pyg_from_step(step_path)
#             torch.save(data, out_path)
#             success += 1
#         except Exception as e:
#             errors += 1

#     print("\n" + "="*30)
#     print("PROCESSING COMPLETE")
#     print(f"✅ Generated: {success}")
#     print(f"❌ Failed:    {errors}")
#     print(f"📂 Output organized in: {OUTPUT_BASE}")
#     print("="*30)

# if __name__ == "__main__":
#     main()

#################################################################################################################################

# """
# generate_part_graphs.py (CHUNKED EXECUTION VERSION)
# Usage: python generate_part_graphs.py --chunk_idx 0
# """

# import os
# import argparse
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from torch_geometric.data import Data

# # --- OCC IMPORTS ---
# from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Core.IFSelect import IFSelect_RetDone
# from OCC.Core.TopExp import topexp, TopExp_Explorer
# from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
# from OCC.Core.TopoDS import TopoDS_Face
# from OCC.Core.TopTools import TopTools_IndexedMapOfShape
# from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
# from OCC.Core.GProp import GProp_GProps
# from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
# from OCC.Core.GeomAbs import (
#     GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
#     GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
#     GeomAbs_BezierSurface
# )
# from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# from OCC.Core.BRep import BRep_Tool
# from OCC.Core.TopLoc import TopLoc_Location
# from OCC.Core.gp import gp_Vec

# # =============================================================================
# # 1. CORE LOGIC
# # =============================================================================

# def load_step_shape(path):
#     reader = STEPControl_Reader()
#     status = reader.ReadFile(path)
#     if status != IFSelect_RetDone:
#         raise RuntimeError(f"Failed to read STEP file")
#     reader.TransferRoots()
#     return reader.OneShape()

# def face_area_centroid(face):
#     props = GProp_GProps()
#     brepgprop_SurfaceProperties(face, props)
#     return props.Mass(), props.CentreOfMass()

# def face_surface_type(face):
#     surf = BRepAdaptor_Surface(face)
#     t = surf.GetType()
#     if t == GeomAbs_Plane: return "plane"
#     if t == GeomAbs_Cylinder: return "cylinder"
#     if t == GeomAbs_Cone: return "cone"
#     if t == GeomAbs_Sphere: return "sphere"
#     if t == GeomAbs_Torus: return "torus"
#     if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface): return "freeform"
#     return "other"

# SURF_MAP = {"plane":0, "cylinder":1, "cone":2, "sphere":3, "torus":4, "freeform":5, "other":6}

# def surf_type_onehot(s_type):
#     vec = np.zeros(7, dtype=np.float32)
#     vec[SURF_MAP.get(s_type, 6)] = 1.0
#     return vec

# def build_face_normal_fn(shape):
#     mesh = BRepMesh_IncrementalMesh(shape, 0.1)
#     mesh.Perform()
#     def get_normal(face):
#         loc = TopLoc_Location()
#         tri = BRep_Tool.Triangulation(face, loc)
#         if tri is None: return (0., 0., 1.)
        
#         # Robust Accessors
#         try: nodes = tri.Nodes()
#         except: nodes = None 
#         try: tris = tri.Triangles()
#         except: tris = None

#         norm = gp_Vec(0,0,0)
#         if tri.NbTriangles() > 0:
#             t = tris.Value(1) if tris else tri.Triangle(1)
#             try: n1, n2, n3 = t.Get()
#             except: n1, n2, n3 = t.Value(1), t.Value(2), t.Value(3)
            
#             p1 = nodes.Value(n1) if nodes else tri.Node(n1)
#             p2 = nodes.Value(n2) if nodes else tri.Node(n2)
#             p3 = nodes.Value(n3) if nodes else tri.Node(n3)
            
#             v1, v2 = gp_Vec(p1, p2), gp_Vec(p1, p3)
#             norm = v1.Crossed(v2)
#             if norm.Magnitude() > 1e-9: norm.Normalize()
#             else: return (0., 0., 1.)
#             return (norm.X(), norm.Y(), norm.Z())
#         return (0., 0., 1.)
#     return get_normal

# def build_graph(step_path):
#     shape = load_step_shape(step_path)
    
#     face_map = TopTools_IndexedMapOfShape()
#     topexp.MapShapes(shape, TopAbs_FACE, face_map)
#     num_faces = face_map.Size()
#     if num_faces == 0: raise RuntimeError("No faces")

#     edge_map = TopTools_IndexedMapOfShape()
#     topexp.MapShapes(shape, TopAbs_EDGE, edge_map)
    
#     edge_to_faces = {}
#     ex = TopExp_Explorer(shape, TopAbs_FACE)
#     while ex.More():
#         f = ex.Current()
#         f_idx = face_map.FindIndex(f)
#         ex_e = TopExp_Explorer(f, TopAbs_EDGE)
#         while ex_e.More():
#             e = ex_e.Current()
#             e_idx = edge_map.FindIndex(e)
#             edge_to_faces.setdefault(e_idx, set()).add(f_idx)
#             ex_e.Next()
#         ex.Next()
        
#     get_norm = build_face_normal_fn(shape)
#     feats = []
#     for i in range(1, num_faces+1):
#         f = face_map.FindKey(i)
#         area, c = face_area_centroid(f)
#         nx, ny, nz = get_norm(f)
#         st = surf_type_onehot(face_surface_type(f))
#         feats.append([area, c.X(), c.Y(), c.Z(), nx, ny, nz] + st.tolist())
        
#     x = torch.tensor(feats, dtype=torch.float)
    
#     src, dst = [], []
#     for faces in edge_to_faces.values():
#         fl = list(faces)
#         if len(fl) > 1:
#             for i in range(len(fl)):
#                 for j in range(i+1, len(fl)):
#                     u, v = fl[i]-1, fl[j]-1
#                     src.append(u); dst.append(v)
#                     src.append(v); dst.append(u)
                    
#     if not src: edge_index = torch.empty((2,0), dtype=torch.long)
#     else: edge_index = torch.tensor([src, dst], dtype=torch.long)
    
#     return Data(x=x, edge_index=edge_index)

# # =============================================================================
# # 2. ARGUMENT PARSER & MAIN
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--chunk_idx", type=int, default=0, help="Which chunk to process (0, 1, 2...)")
#     parser.add_argument("--chunk_size", type=int, default=10000, help="Files per chunk")
#     args = parser.parse_args()

#     # Configuration
#     INPUT_PARQUET = "parts_trainable.parquet"
#     OUTPUT_BASE = "graphs/parts"
    
#     # 1. Load & Slice
#     if not os.path.exists(INPUT_PARQUET):
#         print(f"❌ Error: {INPUT_PARQUET} not found.")
#         return

#     df = pd.read_parquet(INPUT_PARQUET)
    
#     # Calculate slice
#     start = args.chunk_idx * args.chunk_size
#     end = start + args.chunk_size
    
#     # Slice the dataframe
#     df_chunk = df.iloc[start:end]
    
#     if len(df_chunk) == 0:
#         print(f"⚠️ Chunk {args.chunk_idx} is empty (dataset ended).")
#         return

#     # Define output folder for this chunk
#     sub_dir_name = f"part_gr_{args.chunk_idx:05d}"
#     out_dir = os.path.join(OUTPUT_BASE, sub_dir_name)
#     os.makedirs(out_dir, exist_ok=True)
    
#     print(f"🚀 Processing Chunk {args.chunk_idx} (Files {start} to {end})")
#     print(f"📂 Saving to: {out_dir}")

#     success = 0
#     errors = 0

#     for idx, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), unit="part"):
#         part_id = row['part_id']
#         step_path = row['absolute_path']
#         out_path = os.path.join(out_dir, f"{part_id}.pt")
        
#         if os.path.exists(out_path):
#             continue
            
#         try:
#             data = build_graph(step_path)
#             torch.save(data, out_path)
#             success += 1
#         except Exception:
#             errors += 1

#     print(f"✅ Chunk {args.chunk_idx} Done. New: {success}, Failed: {errors}")

# if __name__ == "__main__":
#     main()


###################################################################################################################################################


"""
generate_part_graphs.py (PARALLEL VERSION)
Usage: python generate_part_graphs.py --chunk_idx 0 --workers 8
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import multiprocessing
from functools import partial

# --- OCC IMPORTS ---
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import topexp, TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
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

# =============================================================================
# 1. CORE LOGIC (Must be top-level for multiprocessing)
# =============================================================================

def load_step_shape(path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError("Failed to read STEP")
    reader.TransferRoots()
    return reader.OneShape()

def face_area_centroid(face):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return props.Mass(), props.CentreOfMass()

def face_surface_type(face):
    surf = BRepAdaptor_Surface(face)
    t = surf.GetType()
    if t == GeomAbs_Plane: return "plane"
    if t == GeomAbs_Cylinder: return "cylinder"
    if t == GeomAbs_Cone: return "cone"
    if t == GeomAbs_Sphere: return "sphere"
    if t == GeomAbs_Torus: return "torus"
    if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface): return "freeform"
    return "other"

SURF_MAP = {"plane":0, "cylinder":1, "cone":2, "sphere":3, "torus":4, "freeform":5, "other":6}

def surf_type_onehot(s_type):
    vec = np.zeros(7, dtype=np.float32)
    vec[SURF_MAP.get(s_type, 6)] = 1.0
    return vec

def build_face_normal_fn(shape):
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    def get_normal(face):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None: return (0., 0., 1.)
        
        try: nodes = tri.Nodes()
        except: nodes = None 
        try: tris = tri.Triangles()
        except: tris = None

        norm = gp_Vec(0,0,0)
        if tri.NbTriangles() > 0:
            t = tris.Value(1) if tris else tri.Triangle(1)
            try: n1, n2, n3 = t.Get()
            except: n1, n2, n3 = t.Value(1), t.Value(2), t.Value(3)
            
            p1 = nodes.Value(n1) if nodes else tri.Node(n1)
            p2 = nodes.Value(n2) if nodes else tri.Node(n2)
            p3 = nodes.Value(n3) if nodes else tri.Node(n3)
            
            v1, v2 = gp_Vec(p1, p2), gp_Vec(p1, p3)
            norm = v1.Crossed(v2)
            if norm.Magnitude() > 1e-9: norm.Normalize()
            else: return (0., 0., 1.)
            return (norm.X(), norm.Y(), norm.Z())
        return (0., 0., 1.)
    return get_normal

def build_graph(step_path):
    # This runs inside the worker process
    try:
        shape = load_step_shape(step_path)
        
        face_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_FACE, face_map)
        num_faces = face_map.Size()
        if num_faces == 0: raise RuntimeError("No faces")

        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)
        
        edge_to_faces = {}
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            f = ex.Current()
            f_idx = face_map.FindIndex(f)
            ex_e = TopExp_Explorer(f, TopAbs_EDGE)
            while ex_e.More():
                e = ex_e.Current()
                e_idx = edge_map.FindIndex(e)
                edge_to_faces.setdefault(e_idx, set()).add(f_idx)
                ex_e.Next()
            ex.Next()
            
        get_norm = build_face_normal_fn(shape)
        feats = []
        for i in range(1, num_faces+1):
            f = face_map.FindKey(i)
            area, c = face_area_centroid(f)
            nx, ny, nz = get_norm(f)
            st = surf_type_onehot(face_surface_type(f))
            feats.append([area, c.X(), c.Y(), c.Z(), nx, ny, nz] + st.tolist())
            
        x = torch.tensor(feats, dtype=torch.float)
        
        src, dst = [], []
        for faces in edge_to_faces.values():
            fl = list(faces)
            if len(fl) > 1:
                for i in range(len(fl)):
                    for j in range(i+1, len(fl)):
                        u, v = fl[i]-1, fl[j]-1
                        src.append(u); dst.append(v)
                        src.append(v); dst.append(u)
                        
        if not src: edge_index = torch.empty((2,0), dtype=torch.long)
        else: edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
        
    except Exception as e:
        return None # Signal failure

def process_one_part(args):
    # Unpack arguments for map
    part_id, step_path, out_dir = args
    out_path = os.path.join(out_dir, f"{part_id}.pt")
    
    if os.path.exists(out_path):
        return 0 # Skipped
    
    data = build_graph(step_path)
    if data is None:
        return -1 # Failed
        
    torch.save(data, out_path)
    return 1 # Success

# =============================================================================
# 2. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of CPU cores")
    args = parser.parse_args()

    INPUT_PARQUET = "parts_trainable.parquet"
    OUTPUT_BASE = "graphs/parts"
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"❌ Error: {INPUT_PARQUET} not found.")
        return

    df = pd.read_parquet(INPUT_PARQUET)
    start = args.chunk_idx * args.chunk_size
    end = start + args.chunk_size
    df_chunk = df.iloc[start:end]
    
    if len(df_chunk) == 0:
        print(f"⚠️ Chunk {args.chunk_idx} is empty.")
        return

    sub_dir_name = f"part_gr_{args.chunk_idx:05d}"
    out_dir = os.path.join(OUTPUT_BASE, sub_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 Processing Chunk {args.chunk_idx} with {args.workers} workers")
    
    # Prepare arguments for the workers
    tasks = []
    for idx, row in df_chunk.iterrows():
        tasks.append((row['part_id'], row['absolute_path'], out_dir))
        
    # Parallel Execution
    success = 0
    skipped = 0
    failed = 0
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        # imap_unordered is faster as it yields results as soon as they finish
        for res in tqdm(pool.imap_unordered(process_one_part, tasks), total=len(tasks)):
            if res == 1: success += 1
            elif res == 0: skipped += 1
            elif res == -1: failed += 1
            
    print(f"✅ Chunk {args.chunk_idx} Done. Success: {success}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main()