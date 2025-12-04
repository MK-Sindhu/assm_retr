# Automate Dataset

The Automate dataset contains 451967 unique CAD parts, 255211 CAD assemblies, and 1292016 unique mates scraped from public OnShape documents for [AutoMate: A Dataset and Learning Approach fon the Automatic Mating of CAD Assemblies](https://dl.acm.org/doi/10.1145/3478513.3480562).

Code associated with the paper can be found on the [project page](https://github.com/deGravity/automate).


## Data Format

The dataset is provided in both the original Parasolid format, as well as STEP files. Note that automatic conversion to STEP is not perfect, so some parts may be missing or have bad geometry. Missing parts have been annotated in the accompanying metadata.

### Assembly JSONS
Assembly information is stored as JSON files with the following schema:

```
{
    'assemblyId': str, // name of assembly in assemblies.zip; read as 'assemblies/{assemblyId}.json'
    'has_all_parasolid': bool, // if all parts in the assembly are in parasolid.zip
    'has_all_step':bool, // if all parts in the assembly are in step.zip
    parts:[
        {
            'id': string, // name of part in corresponding zip file; read as either 'step/{id}.step' or 'parasolid/{id}.x_t'
            'has_parasolid': bool, // if the part is present is parasolid.zip
            'has_step': bool, // if the part is present is step.zip 
        }, ...
    ],
    'occurrences': [
        {
            'part': int, // index into parts list
            'id': string, // unique id within assembly
            'transform': list[float], // flattened (row major) 4x4 homogenous transform matrix of part within assembly
            'fixed': bool, // if occurrence is constrained to be unmoving within assembly
            'hidden': bool, // if occurrence was hidden (invisible) in assembly
            'has_parasolid': bool, // if referenced part is in parasolid.zip
            'has_step': bool // if referenced part is in step.zip
        }, ...
    ],
    'mates': [
        {
            'name': str, // name of mate -- may not be unique
            'id': str, // unique id of mate within assembly
            'mateType': str, // type of constraint mate represents, see paper for descriptions
            'occurrences': list[int], // indices into occurrence list of constrained occurrences
            'mcfs': list[list[float]], // 2, 4x4 homogeneous frame matrices for mate connecting frames (see paper for description)
            'has_parasolid': bool, // if both referenced parts are in parasolid.zip
            'has_step': bool // if both referenced parts are in step.zip
        }, ...
    ],
    'mateRelations': [
        {
            'name': str, // name of mate relation -- may not be unique
            'relationType': str, // type of mate relation
            'reverseDirection': bool, // if relation is reversed from default direction
            'relationRatio': float, // (optional) relation parameter
            'relationLength': float, // (optional) relation parameter
            'mates': list[int] // indices into mate list of constrained mates
        }, ...
    ],
    'mateGroups': [
        {
            'name': str, // name of group -- may not be unique
            'id': str, // unique id of group in assembly
            'occurrences': list[int], // list of indices into occurrence list of grouped occurrences
            'has_parasolid': bool, // if all referenced parts are in parasolid.zip
            'has_step': bool // if all referenced parts are in in step.zip
        }, ...
    ],
    'subAssemblies': [
        {
            'id': str, // unique of subassembly relative to assembly
            'assemblyId': str // 
        }, ...
    ]

}
```

### Metadata Tables
Three parquet files contain metadata about parts, assemblies, and mates.

All distances (and derived units) are measured in meters. Masses are derived from assuming a unitless density of 1.

`assemblies.parquet` has the following columns:
 - `assemblyId`: unique assembly id, used to find file in zip 'assemblies/{assemblyId}.zip' and in subAssembly lists
 - `n_subassemblies`: number of unique subassemblies (excluding the root) flattened into this assembly. Does not count multiple instances of the same subassembly.
 - `n_parts`: Number of unique parts in the assembly
 - `n_parasolid`: Number of parts for which we have parasolid geometry
 - `n_parasolid_errors`: Number of parts for which parasolid geometry has some error (see `parts.parquet` for error details)
 - `n_step`: Number of parts for which we have step geometry
 - `n_occurrences`: Number of part occurrences in the assembly
 - `n_mates`: Number of mates in the assembly
 - `n_ps_mates`: Number of mates in the assembly for which we have parasolid geometry for both parts
 - `n_step_mates`: Number of mates in the assembly for which we have step geometry for both parts
 - `n_groups`: Number of mate groups in the assembly
 - `n_relations`: Number of mate relations in the assembly
 - `is_subassembly`: If this assembly is included in any other assembly of the dataset as a subassembly.

`mates.parquet` has the following columns:
- `mateType`: Type of constaint the mate forms 
- `mcfs`: List of flattened 4x4 mating coordinate frame matrices for mate (see paper for details)
- `has_step`: If we have step geometry for both parts in the mate
- `parts`: List of part_ids for the two parts of the mate
- `ps_has_errors`: If either of the parasolid forms of the parts has an error (see `parts.parquet` for error details)


`parts.parquet` has the following columns. Note that summary and error data is computed for the parasolid files and may not be exactly the same for step versions if they exist:
 - `part_id`: unique id of part -- used to locate part in zip files
 - `readable`: if the parasolid file was readable
 - `n_parts`: number of parts in the parasolid file. Should always be 1
 - `n_bodies`: number of topological bodies in the parasolid file, should always be 1
 - `has_corrupt_state`: if the parasolid geometry has any corrupt state
 - `has_invalid_state`: if the parasolid geometry has any invalid state
 - `has_missing_geometry`: if any topological entities in the parasolid file has no associated geometry
 - `error_checking_topology`: if an error occurred reading any of the topological entities
 - `error_finding_bounding_box`: if an error occurred asking parasolid for the part's bounding box
 - `error_finding_na_box`: if an error occrured asking parasolid for the part's non-axis-aligned bounding box
 - `error_computing_mass_properties`: if an error occurred asking parasolid for the part's mass properties
 - `n_faults`: number of faults found by the parasolid kernel when processing the part
 - `n_faces_no_geo`: number of topological faces with missing geometry
 - `n_edges_no_geo`: number of topological edges with missing geometry
 - `n_verts_no_geo`: number of topological vertices with missing geometry
 - `n_topols`: number of topological entities
 - `n_relations`: number of relationships between topological entities
 - `n_regions`: number of topological regions
 - `n_shells`: number of topological shells
 - `n_faces`: number of topological faces
 - `n_edges`: number of topological edges
 - `n_loops`: number of topological loops
 - `n_vertices`: number of topological vertices
 - `n_plane`: number of face topologies with planar geometry
 - `n_cyl`: number of face topologies with cylindrical geometry
 - `n_cone`: number of face topologies with conical geometry
 - `n_sphere`: number of face topologies with spherical geometry
 - `n_torus`: number of face topologies with toroidal geometry
 - `n_bsurf`: number of face topologies with b-spline surface geometry
 - `n_offset`: number of face topologies with offset surface geometry
 - `n_fsurf`: number of face topologies with foreign (imported) surface geometry
 - `n_swept`: number of face topologies with swept surface geometry
 - `n_spun`: number of face topologies with spun surface geometry
 - `n_blendsf`: number of face topologies with blend-surface geometry
 - `n_line`: number of edge topologies with line geometry
 - `n_circle`: number of edge topologies with circular geometry
 - `n_ellipse`: number of edge topologies with elliptical geometry
 - `n_bcurve`: number of edge topologies with b-spline geometry
 - `n_icurve`: number of edge topologies with intersection curve geometry
 - `n_fcurve`: number of edge topologies with foriegn (imported) geometry
 - `n_spcurve`: number of edge topologies with surface parameterized geometry
 - `n_trcurve`: number of edge topologies with trimmed curve geometry
 - `n_cpcurve`: number of edge topologies with cpcurve geometry
 - `bb_0`: axis-aligned bounding box min corner x
 - `bb_1`: axis-aligned bounding box min corner y
 - `bb_2`: axis-aligned bounding box min corner z
 - `bb_3`: axis-aligned bounding box max corner x
 - `bb_4`: axis-aligned bounding box max corner y
 - `bb_5`: axis-aligned bounding box max corner z
 - `nabb_axis_0`: non-axis-aligned bounding box coordinate system z-axis x-coordinate
 - `nabb_axis_1`: non-axis-aligned bounding box coordinate system z-axis y-coordinate
 - `nabb_axis_2`: non-axis-aligned bounding box coordinate system z-axis z-coordinate
 - `nabb_loc_0`: non-axis-aligned bounding box coordinate system center x-coordinate
 - `nabb_loc_1`: non-axis-aligned bounding box coordinate system center y-coordinate
 - `nabb_loc_2`: non-axis-aligned bounding box coordinate system center z-coordinate
 - `nabb_ref_0`: non-axis-aligned bounding box coordinate system x-axis x-coordinate
 - `nabb_ref_1`: non-axis-aligned bounding box coordinate system x-axis y-coordinate
 - `nabb_ref_2`: non-axis-aligned bounding box coordinate system x-axis z-coordinate
 - `nabb_box_0`: non-axis-aligned bounding box min corner x
 - `nabb_box_1`: non-axis-aligned bounding box min corner y
 - `nabb_box_2`: non-axis-aligned bounding box min corner z
 - `nabb_box_3`: non-axis-aligned bounding box max corner x
 - `nabb_box_4`: non-axis-aligned bounding box max corner y
 - `nabb_box_5`: non-axis-aligned bounding box max corner z
 - `mp_amount`: total volume
 - `mp_mass`: total mass
 - `c_of_g_0`: center of gravity x-coordinate
 - `c_of_g_1`: center of gravity y-coordinate
 - `c_of_g_2`: center of gravity z-coordinate
 - `m_of_i_0`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_1`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_2`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_3`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_4`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_5`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_6`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_7`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_8`: moment of inertia tensor components relative to center of mass (row-major)
 - `mp_periphery`: total surface area
 - `has_step`: if we have an associated step version
 - `uniqueid`: unique id used for deduplication (file should already be deduplicated)
 - `has_step_rep`: unused (leftover from deduplication)
 - `rep_part_id`: usused (leftover from deduplication)
 - `is_rep`: unused (leftover from deduplication)
 - `has_error`: if any of the error columns are true or non-zero.

 ### Associating with Onshape ids:

 All unique identifies are derived from their unique identifiers within Onshape. However, since Onshape query strings are case sensitive, contain non-path-friendly characters, and are too long for some file systems, we have canonicalized and shortened them. In general, the association is given by

 ```
 {documentId}_{documentMicroversion}_{elementId}_{encoded_configuration}_{[encoded_part_id if relevant]}
 ```

 associated files are named with this id plus the relevant file extension (.json, .x_t, or .step).

documentId, documentMicroversion, and elementId are unchanged from their onshape form except for being lower case only. The encoded configuration is the first 8 characters of the base32 encoded sha256 hash of the full configuration query string from Onshape. Because this is a destructive transform, the .json file `config_encodings.json` is provided to map back to the original, unencoded query strings. Part id is also encoded as a base32 encoding of the original Onshape part_id, but this transform is reversible. The file `file_encodings.py` contains helper functions for converting back-and-forth between Onshape identifiers and the identifiers used in the AutoMate dataset.