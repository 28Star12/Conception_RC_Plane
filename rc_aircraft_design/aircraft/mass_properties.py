"""Mass properties for aircraft component assemblies.

All returned inertias are about the aircraft centre of gravity and expressed
in the aircraft body frame (+X forward, +Y starboard, +Z upward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .components import Component, Tensor3, Vector3


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Aggregate rigid-body mass properties about the aircraft CG."""

    mass_kg: float
    center_of_gravity_m: Vector3 | None
    inertia_tensor_kg_m2: Tensor3
    principal_moments_kg_m2: Vector3
    component_count: int
    inertia_methods: dict[str, str]

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""
        return {
            "mass_kg": self.mass_kg,
            "center_of_gravity_m": list(self.center_of_gravity_m)
            if self.center_of_gravity_m is not None
            else None,
            "inertia_tensor_kg_m2": [list(row) for row in self.inertia_tensor_kg_m2],
            "principal_moments_kg_m2": list(self.principal_moments_kg_m2),
            "component_count": self.component_count,
            "inertia_methods": dict(self.inertia_methods),
        }


def rotation_matrix_xyz(rotation_deg: Vector3) -> np.ndarray:
    """Map local component axes into body axes using Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.radians(rotation_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def intrinsic_inertia_tensor(component: Component) -> tuple[np.ndarray, str]:
    """Return the component's inertia about its own CG and local axes."""
    if component.inertia_tensor_kg_m2 is not None:
        return np.asarray(component.inertia_tensor_kg_m2, dtype=float), "provided_tensor"

    mass = component.mass_kg
    if component.geometry == "point_mass":
        return np.zeros((3, 3), dtype=float), "point_mass_approximation"

    assert component.dimensions_m is not None  # validated by Component
    dx, dy, dz = component.dimensions_m

    if component.geometry == "box":
        ixx = mass * (dy**2 + dz**2) / 12.0
        iyy = mass * (dx**2 + dz**2) / 12.0
        izz = mass * (dx**2 + dy**2) / 12.0
        method = "solid_box"
    elif component.geometry == "cylinder_x":
        radius, length = dy / 2.0, dx
        axial = 0.5 * mass * radius**2
        transverse = mass * (3.0 * radius**2 + length**2) / 12.0
        ixx, iyy, izz = axial, transverse, transverse
        method = "solid_cylinder"
    elif component.geometry == "cylinder_y":
        radius, length = dx / 2.0, dy
        axial = 0.5 * mass * radius**2
        transverse = mass * (3.0 * radius**2 + length**2) / 12.0
        ixx, iyy, izz = transverse, axial, transverse
        method = "solid_cylinder"
    else:  # cylinder_z
        radius, length = dx / 2.0, dz
        axial = 0.5 * mass * radius**2
        transverse = mass * (3.0 * radius**2 + length**2) / 12.0
        ixx, iyy, izz = transverse, transverse, axial
        method = "solid_cylinder"

    return np.diag([ixx, iyy, izz]), method


def compute_mass_properties(components: Iterable[Component]) -> MassProperties:
    """Compute aggregate mass, CG and inertia for an iterable of components.

    The parallel-axis theorem is applied to each component after rotating its
    intrinsic inertia tensor into the aircraft frame. Components described as
    point masses contribute no intrinsic inertia; their offset from the total
    CG is still included. Results are engineering estimates unless component
    masses and inertia data have been measured or supplied by manufacturers.
    """
    items = tuple(components)
    if any(not isinstance(item, Component) for item in items):
        raise TypeError("components must contain only Component instances")

    total_mass = float(sum(item.mass_kg for item in items))
    zero_tensor = np.zeros((3, 3), dtype=float)
    if total_mass <= 0.0:
        return MassProperties(
            mass_kg=0.0,
            center_of_gravity_m=None,
            inertia_tensor_kg_m2=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            principal_moments_kg_m2=(0.0, 0.0, 0.0),
            component_count=len(items),
            inertia_methods={item.component_id: "zero_mass" for item in items},
        )

    component_data: list[tuple[Component, np.ndarray, np.ndarray, np.ndarray, str]] = []
    weighted_position = np.zeros(3, dtype=float)
    inertia_methods: dict[str, str] = {}

    for component in items:
        rotation = rotation_matrix_xyz(component.rotation_deg)
        origin = np.asarray(component.position_m, dtype=float)
        local_offset = np.asarray(component.center_of_mass_offset_m, dtype=float)
        center = origin + rotation @ local_offset
        local_inertia, method = intrinsic_inertia_tensor(component)
        body_inertia = rotation @ local_inertia @ rotation.T
        component_data.append((component, center, body_inertia, rotation, method))
        weighted_position += component.mass_kg * center
        inertia_methods[component.component_id] = method

    center_of_gravity = weighted_position / total_mass
    total_inertia = zero_tensor.copy()
    identity = np.eye(3)

    for component, center, body_inertia, _rotation, _method in component_data:
        displacement = center - center_of_gravity
        parallel_axis = component.mass_kg * (
            float(displacement @ displacement) * identity - np.outer(displacement, displacement)
        )
        total_inertia += body_inertia + parallel_axis

    # Remove harmless floating-point asymmetry before extracting principal values.
    total_inertia = 0.5 * (total_inertia + total_inertia.T)
    principal = np.linalg.eigvalsh(total_inertia)
    principal = np.maximum(principal, 0.0)

    return MassProperties(
        mass_kg=total_mass,
        center_of_gravity_m=tuple(float(value) for value in center_of_gravity),  # type: ignore[arg-type]
        inertia_tensor_kg_m2=tuple(
            tuple(float(value) for value in row) for row in total_inertia
        ),  # type: ignore[arg-type]
        principal_moments_kg_m2=tuple(float(value) for value in principal),  # type: ignore[arg-type]
        component_count=len(items),
        inertia_methods=inertia_methods,
    )
