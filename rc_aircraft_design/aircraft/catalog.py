"""Offline component catalogue with manufacturer source traceability."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .components import Component


def _catalog_directory():
    return files("rc_aircraft_design.aircraft").joinpath("catalog_data")


def list_component_catalog_entries() -> list[dict[str, Any]]:
    """List all bundled catalogue entries, sorted by catalogue ID."""
    entries: list[dict[str, Any]] = []
    for resource in _catalog_directory().iterdir():
        if resource.is_file() and resource.name.endswith(".json"):
            with resource.open("r", encoding="utf-8") as stream:
                entries.append(json.load(stream))
    return sorted(entries, key=lambda entry: entry["catalog_id"])


def load_component_catalog_entry(catalog_id: str) -> dict[str, Any]:
    """Load one complete catalogue record by stable catalogue ID."""
    for entry in list_component_catalog_entries():
        if entry.get("catalog_id") == catalog_id:
            return entry
    raise KeyError(f"Catalogue component '{catalog_id}' was not found")


def load_catalog_component(catalog_id: str) -> Component:
    """Load the physical component model from a catalogue record."""
    entry = load_component_catalog_entry(catalog_id)
    return Component.from_dict(entry["component"])
