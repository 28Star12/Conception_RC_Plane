"""Tests for STL mesh parsing and unit conversion."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from rc_aircraft_design.aircraft.mesh_import import parse_stl


TETRA_FACES = [
    [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    [(0, 0, 0), (0, 0, 1), (1, 0, 0)],
    [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
    [(1, 0, 0), (0, 0, 1), (0, 1, 0)],
]


def _ascii_stl() -> bytes:
    lines = ["solid tetra"]
    for triangle in TETRA_FACES:
        lines.append(" facet normal 0 0 1")
        lines.append("  outer loop")
        for x, y, z in triangle:
            lines.append(f"   vertex {x} {y} {z}")
        lines.extend(["  endloop", " endfacet"])
    lines.append("endsolid tetra")
    return "\n".join(lines).encode("ascii")


def _binary_stl() -> bytes:
    data = bytearray(b"test tetra".ljust(80, b" "))
    data.extend(struct.pack("<I", len(TETRA_FACES)))
    for triangle in TETRA_FACES:
        flat = [float(value) for point in triangle for value in point]
        data.extend(struct.pack("<12fH", 0.0, 0.0, 1.0, *flat, 0))
    return bytes(data)


@pytest.mark.parametrize("factory", [_ascii_stl, _binary_stl], ids=["ascii", "binary"])
def test_parse_stl_centres_mesh_and_converts_millimetres(factory) -> None:
    result = parse_stl(factory(), units="mm")
    vertices = np.asarray(result["vertices_m"])
    faces = np.asarray(result["faces"])

    assert result["triangle_count"] == 4
    assert result["source_triangle_count"] == 4
    assert vertices.shape == (4, 3)
    assert faces.shape == (4, 3)
    assert result["bounds_m"]["dimensions"] == pytest.approx([0.001, 0.001, 0.001])
    assert (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0 == pytest.approx([0.0, 0.0, 0.0])
    assert result["source_units"] == "mm"


def test_parse_stl_supports_metre_units() -> None:
    result = parse_stl(_ascii_stl(), units="m")
    assert result["bounds_m"]["dimensions"] == pytest.approx([1.0, 1.0, 1.0])


def test_parse_stl_reduces_large_mesh_to_triangle_limit() -> None:
    repeated = TETRA_FACES * 5
    lines = ["solid repeated"]
    for triangle in repeated:
        lines.extend(["facet normal 0 0 1", "outer loop"])
        lines.extend(f"vertex {x} {y} {z}" for x, y, z in triangle)
        lines.extend(["endloop", "endfacet"])
    lines.append("endsolid repeated")
    result = parse_stl("\n".join(lines).encode(), units="m", max_triangles=3)
    assert result["source_triangle_count"] == 20
    assert result["triangle_count"] == 3


def test_parse_stl_rejects_invalid_units_and_invalid_file() -> None:
    with pytest.raises(ValueError, match="Unsupported STL unit"):
        parse_stl(_ascii_stl(), units="yards")
    with pytest.raises(ValueError, match="Could not read triangle vertices"):
        parse_stl(b"this is not an STL file")
    with pytest.raises(ValueError, match="empty"):
        parse_stl(b"")
