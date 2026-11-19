"""Tests for the aircraft component and rigid-body mass-property core."""

from __future__ import annotations

import numpy as np
import pytest

from rc_aircraft_design.aircraft import AircraftAssembly, Component, compute_mass_properties


def test_empty_assembly_has_no_mass_properties_cg() -> None:
    assembly = AircraftAssembly(aircraft_id="empty", name="Empty aircraft")
    result = assembly.mass_properties()
    assert result.mass_kg == 0.0
    assert result.center_of_gravity_m is None
    assert result.component_count == 0
    assert np.allclose(result.inertia_tensor_kg_m2, np.zeros((3, 3)))


def test_point_masses_use_parallel_axis_theorem() -> None:
    components = [
        Component("a", "Front item", "payload", 2.0, position_m=(0, 0, 0)),
        Component("b", "Rear item", "payload", 1.0, position_m=(3, 0, 0)),
    ]
    result = compute_mass_properties(components)
    assert result.mass_kg == pytest.approx(3.0)
    assert result.center_of_gravity_m == pytest.approx((1.0, 0.0, 0.0))
    assert result.inertia_tensor_kg_m2[1][1] == pytest.approx(6.0)
    assert result.inertia_tensor_kg_m2[2][2] == pytest.approx(6.0)
    assert result.inertia_tensor_kg_m2[0][0] == pytest.approx(0.0)


def test_solid_box_intrinsic_inertia() -> None:
    item = Component(
        "battery", "Battery block", "electronics", 12.0,
        geometry="box", dimensions_m=(2.0, 4.0, 6.0),
    )
    result = compute_mass_properties([item])
    assert np.allclose(
        result.inertia_tensor_kg_m2,
        ((52.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 0.0, 20.0)),
    )
    assert result.inertia_methods["battery"] == "solid_box"


def test_cylinder_intrinsic_inertia_along_x() -> None:
    item = Component(
        "tank", "Cylindrical tank", "fuel", 10.0,
        geometry="cylinder_x", dimensions_m=(2.0, 1.0, 1.0),
    )
    result = compute_mass_properties([item])
    assert result.inertia_tensor_kg_m2[0][0] == pytest.approx(1.25)
    assert result.inertia_tensor_kg_m2[1][1] == pytest.approx(3.9583333333333335)
    assert result.inertia_tensor_kg_m2[2][2] == pytest.approx(3.9583333333333335)


def test_rotated_box_inertia_is_transformed_to_aircraft_axes() -> None:
    item = Component(
        "box", "Rectangular box", "structure", 12.0,
        rotation_deg=(0.0, 0.0, 90.0),
        geometry="box", dimensions_m=(2.0, 4.0, 6.0),
    )
    result = compute_mass_properties([item])
    assert result.inertia_tensor_kg_m2[0][0] == pytest.approx(40.0)
    assert result.inertia_tensor_kg_m2[1][1] == pytest.approx(52.0)
    assert result.inertia_tensor_kg_m2[2][2] == pytest.approx(20.0)


def test_component_center_of_mass_offset_is_rotated() -> None:
    item = Component(
        "servo", "Servo", "control", 1.0,
        position_m=(1.0, 2.0, 3.0),
        rotation_deg=(0.0, 0.0, 90.0),
        center_of_mass_offset_m=(1.0, 0.0, 0.0),
    )
    result = compute_mass_properties([item])
    assert result.center_of_gravity_m == pytest.approx((1.0, 3.0, 3.0))


def test_custom_inertia_tensor_is_accepted_and_used() -> None:
    item = Component(
        "engine", "Test engine", "propulsion", 5.0,
        inertia_tensor_kg_m2=((1.0, 0.1, 0.0), (0.1, 2.0, 0.0), (0.0, 0.0, 3.0)),
    )
    result = compute_mass_properties([item])
    assert result.inertia_tensor_kg_m2[0][1] == pytest.approx(0.1)
    assert result.inertia_methods["engine"] == "provided_tensor"


def test_assembly_rejects_duplicate_ids_and_can_remove_components() -> None:
    assembly = AircraftAssembly(aircraft_id="test", name="Test")
    item = Component("servo", "Servo", "control", 0.1)
    assembly.add_component(item)
    with pytest.raises(ValueError, match="already exists"):
        assembly.add_component(item)
    assert assembly.remove_component("servo") is item
    with pytest.raises(KeyError):
        assembly.remove_component("missing")


def test_assembly_serialization_round_trip() -> None:
    assembly = AircraftAssembly(
        aircraft_id="jet-01", name="RC jet", vehicle_type="jet",
        components=[
            Component(
                "turbine", "Micro turbine", "propulsion", 2.4,
                manufacturer="Example Manufacturer", model="T-01",
                mass_source="manufacturer", source_url="https://example.com/specs",
                geometry="cylinder_x", dimensions_m=(0.3, 0.1, 0.1),
            )
        ],
    )
    restored = AircraftAssembly.from_dict(assembly.to_dict())
    assert restored.to_dict() == assembly.to_dict()


def test_component_validation_rejects_invalid_mass_dimensions_and_tensor() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        Component("bad", "Bad", "other", -1.0)
    with pytest.raises(ValueError, match="dimensions_m are required"):
        Component("bad", "Bad", "other", 1.0, geometry="box")
    with pytest.raises(ValueError, match="symmetric"):
        Component(
            "bad", "Bad", "other", 1.0,
            inertia_tensor_kg_m2=((1, 1, 0), (0, 1, 0), (0, 0, 1)),
        )


def test_zero_mass_components_do_not_move_cg() -> None:
    result = compute_mass_properties(
        [
            Component("mass", "Mass", "payload", 2.0, position_m=(1, 0, 0)),
            Component("placeholder", "Unweighted placeholder", "other", 0.0, position_m=(100, 0, 0)),
        ]
    )
    assert result.mass_kg == pytest.approx(2.0)
    assert result.center_of_gravity_m == pytest.approx((1.0, 0.0, 0.0))


def test_component_ids_must_be_unique_when_assembly_is_created() -> None:
    item = Component("same", "One", "other", 1.0)
    with pytest.raises(ValueError, match="unique"):
        AircraftAssembly("bad", "Bad", components=[item, item])
