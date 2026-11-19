"""Mesh import helpers for the browser-based aircraft assembly editor.

STL is parsed without optional dependencies. STEP/IGES import requires the
optional CadQuery/OpenCascade extra. Mesh vertices are returned in metres,
recentered around their bounding-box centre, with triangle indices.
"""

from __future__ import annotations

import math
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

UNIT_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TRIANGLES = 12000


def _mesh_from_triangles(
    triangles: np.ndarray,
    *,
    unit_scale: float,
    source_units: str,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> dict[str, Any]:
    """Validate, cap, deduplicate and centre a triangle soup."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Mesh data must contain triangles with three 3D vertices")
    if not np.all(np.isfinite(triangles)):
        raise ValueError("Mesh contains non-finite coordinates")
    if not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("Unit scale must be a positive finite number")

    original_count = len(triangles)
    if original_count == 0:
        raise ValueError("The selected file does not contain any triangles")
    if original_count > max_triangles:
        sample = np.linspace(0, original_count - 1, max_triangles, dtype=np.int64)
        triangles = triangles[sample]

    triangles *= unit_scale
    flat = triangles.reshape(-1, 3)
    bounds_min = flat.min(axis=0)
    bounds_max = flat.max(axis=0)
    centre = (bounds_min + bounds_max) / 2.0
    centred = flat - centre

    # STL repeats all three vertices for every facet. Quantised unique vertices
    # significantly reduce the JSON payload sent to the browser.
    vertices, inverse = np.unique(
        np.round(centred, decimals=8), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1, 3)

    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    valid = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) > 1e-12
    faces = faces[valid]
    if len(faces) == 0:
        raise ValueError("The mesh contains only degenerate triangles")

    dimensions = bounds_max - bounds_min
    if np.any(dimensions <= 0):
        raise ValueError("The mesh has zero size on at least one axis")

    return {
        "vertices_m": vertices.astype(float).tolist(),
        "faces": faces.astype(int).tolist(),
        "bounds_m": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
            "dimensions": dimensions.astype(float).tolist(),
        },
        "triangle_count": int(len(faces)),
        "source_triangle_count": int(original_count),
        "source_units": source_units,
        "origin": "bounding_box_center",
    }


def parse_stl(
    file_bytes: bytes,
    *,
    units: str = "mm",
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> dict[str, Any]:
    """Parse binary or ASCII STL into a centred, metre-based mesh."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 20 MB upload limit")
    unit = units.lower()
    if unit not in UNIT_TO_METRES:
        raise ValueError(f"Unsupported STL unit '{units}'")

    triangles: np.ndarray | None = None

    # Binary STL: 80-byte header, uint32 facet count, then 50 bytes per facet.
    if len(file_bytes) >= 84:
        facet_count = struct.unpack_from("<I", file_bytes, 80)[0]
        if facet_count > 0 and 84 + facet_count * 50 == len(file_bytes):
            facet_dtype = np.dtype([
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ])
            facets = np.frombuffer(file_bytes, dtype=facet_dtype, count=facet_count, offset=84)
            triangles = facets["vertices"].astype(np.float64)

    if triangles is None:
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("The file is neither a valid binary nor ASCII STL") from exc
        matches = re.findall(
            r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not matches or len(matches) % 3:
            raise ValueError("Could not read triangle vertices from the STL file")
        coordinates = np.asarray(matches, dtype=np.float64)
        triangles = coordinates.reshape(-1, 3, 3)

    return _mesh_from_triangles(
        triangles,
        unit_scale=UNIT_TO_METRES[unit],
        source_units=unit,
        max_triangles=max_triangles,
    )


def parse_step_iges(
    file_bytes: bytes,
    filename: str,
    *,
    iges_units: str = "mm",
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> dict[str, Any]:
    """Import STEP or IGES through the optional CadQuery/OpenCascade runtime.

    STEP is explicitly converted to metres by CadQuery. IGES geometry is
    scaled using the unit selected by the user because IGES unit metadata is
    not consistently preserved by all exporters.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in {".step", ".stp", ".iges", ".igs"}:
        raise ValueError("Supported CAD formats are STEP (.step/.stp) and IGES (.iges/.igs)")
    if not file_bytes:
        raise ValueError("The uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 20 MB upload limit")
    if iges_units.lower() not in UNIT_TO_METRES:
        raise ValueError(f"Unsupported IGES unit '{iges_units}'")

    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "STEP/IGES import needs the optional CadQuery engine. "
            "Install the project with the 'cad-import' extra on the server."
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(file_bytes)
        temporary.flush()

        if suffix in {".step", ".stp"}:
            imported = cq.importers.importStep(temporary.name, unit="M")
            shapes = imported.vals()
            scale = 1.0
            source_units = "step-file-units-converted-to-m"
        else:
            try:
                from OCP.IGESControl import IGESControl_Reader
                from OCP.IFSelect import IFSelect_ReturnStatus
            except ImportError as exc:
                raise RuntimeError("The installed CadQuery build does not expose its OpenCascade IGES reader") from exc

            reader = IGESControl_Reader()
            status = reader.ReadFile(temporary.name)
            if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                raise ValueError("OpenCascade could not read this IGES file")
            transferred = reader.TransferRoots()
            if transferred == 0:
                raise ValueError("No transferable CAD roots were found in the IGES file")
            shape = cq.Shape.cast(reader.OneShape())
            shapes = [shape]
            scale = UNIT_TO_METRES[iges_units.lower()]
            source_units = f"iges-{iges_units.lower()}"

        triangles_list: list[np.ndarray] = []
        total = 0
        for shape in shapes:
            vertices, faces = shape.tessellate(0.001, 0.15)
            if not faces:
                continue
            vertex_array = np.asarray([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
            face_array = np.asarray(faces, dtype=np.int64)
            triangles_list.append(vertex_array[face_array])
            total += len(face_array)
            if total > max_triangles * 4:
                break

    if not triangles_list:
        raise ValueError("No solid or surface geometry could be tessellated from the CAD file")
    triangles = np.concatenate(triangles_list, axis=0)
    return _mesh_from_triangles(
        triangles,
        unit_scale=scale,
        source_units=source_units,
        max_triangles=max_triangles,
    )
