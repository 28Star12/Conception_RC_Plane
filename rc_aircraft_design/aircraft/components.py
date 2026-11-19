"""Aircraft component and assembly data models.

Coordinate convention
---------------------
The aircraft body frame is right-handed: +X forward, +Y starboard (right),
and +Z upward. Positions and dimensions are expressed in metres, masses in
kilograms, and angles in degrees. A component's position_m is its local
origin in the aircraft frame. center_of_mass_offset_m is measured from that
origin in the component's own coordinate frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Vector3 = tuple[float, float, float]
Tensor3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

_GEOMETRIES = {"point_mass", "box", "cylinder_x", "cylinder_y", "cylinder_z"}
_MASS_SOURCES = {"manufacturer", "measured", "calculated", "estimated"}


def _vector3(value: Any, field_name: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} values must be finite")
    return result  # type: ignore[return-value]


def _tensor3(value: Any) -> Tensor3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("inertia_tensor_kg_m2 must be a finite 3x3 matrix")
    if not np.allclose(array, array.T, atol=1e-10):
        raise ValueError("inertia_tensor_kg_m2 must be symmetric")
    if np.min(np.linalg.eigvalsh(array)) < -1e-10:
        raise ValueError("inertia_tensor_kg_m2 must be positive semi-definite")
    return tuple(tuple(float(v) for v in row) for row in array)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Component:
    """A physical item installed in an aircraft assembly.

    geometry selects the built-in inertia approximation:
    - point_mass: no intrinsic inertia; suitable only for a small item or
      when its dimensions are not yet known.
    - box: homogeneous solid rectangular cuboid.
    - cylinder_x/y/z: homogeneous solid cylinder aligned with the named
      local axis. dimensions_m stores its bounding-box dimensions.

    For a real irregular component, provide its manufacturer/measured inertia
    tensor about its own centre of mass using inertia_tensor_kg_m2.
    """

    component_id: str
    name: str
    category: str
    mass_kg: float
    position_m: Vector3 = (0.0, 0.0, 0.0)
    rotation_deg: Vector3 = (0.0, 0.0, 0.0)
    center_of_mass_offset_m: Vector3 = (0.0, 0.0, 0.0)
    geometry: str = "point_mass"
    dimensions_m: Vector3 | None = None
    inertia_tensor_kg_m2: Tensor3 | None = None
    manufacturer: str | None = None
    model: str | None = None
    mass_source: str = "estimated"
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id cannot be empty")
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.category.strip():
            raise ValueError("category cannot be empty")
        try:
            mass_kg = float(self.mass_kg)
        except (TypeError, ValueError) as exc:
            raise ValueError("mass_kg must be a finite, non-negative number") from exc
        if not math.isfinite(mass_kg) or mass_kg < 0:
            raise ValueError("mass_kg must be a finite, non-negative number")
        object.__setattr__(self, "mass_kg", mass_kg)

        if self.geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of: {', '.join(sorted(_GEOMETRIES))}")
        if self.mass_source not in _MASS_SOURCES:
            raise ValueError(f"mass_source must be one of: {', '.join(sorted(_MASS_SOURCES))}")

        object.__setattr__(self, "position_m", _vector3(self.position_m, "position_m"))
        object.__setattr__(self, "rotation_deg", _vector3(self.rotation_deg, "rotation_deg"))
        object.__setattr__(
            self,
            "center_of_mass_offset_m",
            _vector3(self.center_of_mass_offset_m, "center_of_mass_offset_m"),
        )

        if self.dimensions_m is not None:
            dimensions = _vector3(self.dimensions_m, "dimensions_m")
            if any(value <= 0 for value in dimensions):
                raise ValueError("dimensions_m values must be positive")
            object.__setattr__(self, "dimensions_m", dimensions)

        if self.geometry in {"box", "cylinder_x", "cylinder_y", "cylinder_z"}:
            if self.dimensions_m is None:
                raise ValueError(f"dimensions_m are required for geometry '{self.geometry}'")
            if self.geometry.startswith("cylinder_"):
                axis = self.geometry[-1]
                cross_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
                d1, d2 = (self.dimensions_m[index] for index in cross_axes)
                if not math.isclose(d1, d2, rel_tol=1e-5, abs_tol=1e-9):
                    raise ValueError("cylinder cross-section dimensions must be equal")

        if self.inertia_tensor_kg_m2 is not None:
            object.__setattr__(self, "inertia_tensor_kg_m2", _tensor3(self.inertia_tensor_kg_m2))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this component."""
        return {
            "component_id": self.component_id,
            "name": self.name,
            "category": self.category,
            "mass_kg": self.mass_kg,
            "position_m": list(self.position_m),
            "rotation_deg": list(self.rotation_deg),
            "center_of_mass_offset_m": list(self.center_of_mass_offset_m),
            "geometry": self.geometry,
            "dimensions_m": list(self.dimensions_m) if self.dimensions_m is not None else None,
            "inertia_tensor_kg_m2": [list(row) for row in self.inertia_tensor_kg_m2]
            if self.inertia_tensor_kg_m2 is not None
            else None,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "mass_source": self.mass_source,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Component":
        """Build a component from a JSON-decoded dictionary."""
        return cls(**data)


@dataclass(slots=True)
class AircraftAssembly:
    """A collection of physical components forming one aircraft."""

    aircraft_id: str
    name: str
    components: list[Component] = field(default_factory=list)
    description: str = ""
    vehicle_type: str = "fixed_wing"

    def __post_init__(self) -> None:
        if not self.aircraft_id.strip():
            raise ValueError("aircraft_id cannot be empty")
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.vehicle_type.strip():
            raise ValueError("vehicle_type cannot be empty")
        self.components = list(self.components)
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component_id values must be unique within an assembly")

    def add_component(self, component: Component) -> None:
        """Add a component, rejecting duplicate IDs."""
        if any(existing.component_id == component.component_id for existing in self.components):
            raise ValueError(f"component_id '{component.component_id}' already exists")
        self.components.append(component)

    def remove_component(self, component_id: str) -> Component:
        """Remove and return a component by ID."""
        for index, component in enumerate(self.components):
            if component.component_id == component_id:
                return self.components.pop(index)
        raise KeyError(f"component_id '{component_id}' was not found")

    def get_component(self, component_id: str) -> Component:
        """Return a component by ID."""
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(f"component_id '{component_id}' was not found")

    def mass_properties(self):
        """Compute total mass, centre of gravity and inertia tensor."""
        from .mass_properties import compute_mass_properties

        return compute_mass_properties(self.components)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible assembly, including all components."""
        return {
            "aircraft_id": self.aircraft_id,
            "name": self.name,
            "description": self.description,
            "vehicle_type": self.vehicle_type,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AircraftAssembly":
        """Build an assembly from a JSON-decoded dictionary."""
        values = dict(data)
        values["components"] = [Component.from_dict(item) for item in values.get("components", [])]
        return cls(**values)
