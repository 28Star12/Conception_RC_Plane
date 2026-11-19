"""Tests for the bundled real-component catalogue."""

from __future__ import annotations

import pytest

from rc_aircraft_design.aircraft import (
    AircraftAssembly,
    list_component_catalog_entries,
    load_catalog_component,
    load_component_catalog_entry,
)


def test_catalog_lists_kingtech_k260g5() -> None:
    entries = list_component_catalog_entries()
    assert any(entry["catalog_id"] == "kingtech-k260g5" for entry in entries)


def test_kingtech_k260g5_component_uses_manufacturer_mass_and_dimensions() -> None:
    component = load_catalog_component("kingtech-k260g5")
    assert component.mass_kg == pytest.approx(2.2)
    assert component.dimensions_m == pytest.approx((0.299, 0.12, 0.12))
    assert component.manufacturer == "KingTech Turbines Co., Ltd."
    assert component.mass_source == "manufacturer"
    assert component.source_url == "https://kingtechtw.com/product/k260g5/"


def test_kingtech_entry_labels_inertia_as_approximate() -> None:
    entry = load_component_catalog_entry("kingtech-k260g5")
    assert entry["data_quality"]["intrinsic_inertia"] == (
        "estimated_from_homogeneous_cylinder_envelope"
    )
    assert "unknown_assumed_at_origin" in entry["data_quality"]["component_cg_offset"]
    assert entry["manufacturer_specifications"]["maximum_thrust_n_calculated"] == pytest.approx(
        26.0 * 9.80665, abs=0.01
    )


def test_catalog_component_can_be_added_to_aircraft_and_recomputed() -> None:
    aircraft = AircraftAssembly("jet-01", "RC jet", vehicle_type="jet")
    turbine = load_catalog_component("kingtech-k260g5")
    aircraft.add_component(turbine)
    result = aircraft.mass_properties()
    assert result.mass_kg == pytest.approx(2.2)
    assert result.component_count == 1
    assert result.inertia_methods[turbine.component_id] == "solid_cylinder"


def test_missing_catalog_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="not found"):
        load_catalog_component("unknown-component")
