"""Interactive 3D assembly builder page.

The browser editor keeps the assembly as a JSON project. Geometry is rendered
with Plotly; component placement and orientation are edited numerically so
that every transformation is explicit and repeatable.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.cards import page_header
from rc_aircraft_design.aircraft import (
    AircraftAssembly,
    Component,
    list_component_catalog_entries,
    load_catalog_component,
    rotation_matrix_xyz,
)
from rc_aircraft_design.aircraft.mesh_import import (
    MAX_UPLOAD_BYTES,
    parse_step_iges,
    parse_stl,
)

dash.register_page(__name__, path="/builder", title="3D Builder", name="3D Builder")

PALETTE = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777"]
MAX_PROJECT_JSON_BYTES = 20 * 1024 * 1024


def _empty_project() -> dict[str, Any]:
    assembly = AircraftAssembly(
        aircraft_id="aircraft-01",
        name="New aircraft",
        vehicle_type="jet",
    )
    return {"schema_version": 1, "assembly": assembly.to_dict(), "mesh_assets": {}}


def _project_parts(project: dict[str, Any]) -> tuple[AircraftAssembly, dict[str, Any]]:
    """Validate a project store and return its domain assembly and mesh assets."""
    if not isinstance(project, dict):
        raise ValueError("Project data must be a JSON object")
    assembly_data = project.get("assembly", project)
    assembly = AircraftAssembly.from_dict(assembly_data)
    assets = project.get("mesh_assets", {})
    if not isinstance(assets, dict):
        raise ValueError("mesh_assets must be an object")

    valid_ids = {item.component_id for item in assembly.components}
    clean_assets: dict[str, Any] = {}
    for component_id, asset in assets.items():
        if component_id not in valid_ids:
            continue
        if not isinstance(asset, dict):
            raise ValueError(f"Mesh asset for '{component_id}' is invalid")
        vertices = np.asarray(asset.get("vertices_m", []), dtype=float)
        faces = np.asarray(asset.get("faces", []), dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) > 40000:
            raise ValueError(f"Mesh vertices for '{component_id}' are invalid or too large")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) > 12000:
            raise ValueError(f"Mesh triangles for '{component_id}' are invalid or too large")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(f"Mesh indices for '{component_id}' are out of range")
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f"Mesh vertices for '{component_id}' contain invalid values")
        clean_assets[component_id] = asset
    return assembly, clean_assets


def _payload(assembly: AircraftAssembly, mesh_assets: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assembly": assembly.to_dict(),
        "mesh_assets": mesh_assets,
    }


def _box_mesh(dimensions: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    dx, dy, dz = dimensions
    vertices = np.asarray([
        [-dx/2, -dy/2, -dz/2], [dx/2, -dy/2, -dz/2],
        [dx/2, dy/2, -dz/2], [-dx/2, dy/2, -dz/2],
        [-dx/2, -dy/2, dz/2], [dx/2, -dy/2, dz/2],
        [dx/2, dy/2, dz/2], [-dx/2, dy/2, dz/2],
    ], dtype=float)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    return vertices, faces


def _cylinder_mesh(
    dimensions: tuple[float, float, float], axis: str, segments: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    dims = np.asarray(dimensions, dtype=float)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    cross = [index for index in range(3) if index != axis_index]
    radius = dims[cross[0]] / 2.0
    length = dims[axis_index]
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices: list[list[float]] = []
    for axial in (-length / 2.0, length / 2.0):
        for angle in angles:
            point = np.zeros(3, dtype=float)
            point[axis_index] = axial
            point[cross[0]] = radius * np.cos(angle)
            point[cross[1]] = radius * np.sin(angle)
            vertices.append(point.tolist())
    bottom_center = len(vertices)
    point = np.zeros(3, dtype=float)
    point[axis_index] = -length / 2.0
    vertices.append(point.tolist())
    top_center = len(vertices)
    point = np.zeros(3, dtype=float)
    point[axis_index] = length / 2.0
    vertices.append(point.tolist())

    faces: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        a, b = index, nxt
        c, d = segments + index, segments + nxt
        faces.extend([[a, b, c], [b, d, c]])
        faces.append([bottom_center, b, a])
        faces.append([top_center, c, d])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _component_mesh(
    component: Component, mesh_assets: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray] | None:
    asset = mesh_assets.get(component.component_id)
    if asset:
        return (
            np.asarray(asset["vertices_m"], dtype=float),
            np.asarray(asset["faces"], dtype=int),
        )
    if component.geometry == "point_mass" or component.dimensions_m is None:
        return None
    if component.geometry == "box":
        return _box_mesh(component.dimensions_m)
    if component.geometry.startswith("cylinder_"):
        return _cylinder_mesh(component.dimensions_m, component.geometry[-1])
    return None


def _build_scene(
    project: dict[str, Any], selected_id: str | None = None
) -> go.Figure:
    assembly, mesh_assets = _project_parts(project)
    figure = go.Figure()
    for index, component in enumerate(assembly.components):
        geometry = _component_mesh(component, mesh_assets)
        position = np.asarray(component.position_m, dtype=float)
        rotation = rotation_matrix_xyz(component.rotation_deg)
        selected = component.component_id == selected_id
        color = "#e11d48" if selected else PALETTE[index % len(PALETTE)]

        if geometry is not None:
            vertices, faces = geometry
            transformed = vertices @ rotation.T + position
            figure.add_trace(go.Mesh3d(
                x=transformed[:, 0],
                y=transformed[:, 1],
                z=transformed[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.88 if selected else 0.68,
                name=component.name,
                hovertemplate=f"{component.name}<extra>{component.category}</extra>",
                flatshading=True,
                showlegend=True,
            ))

        figure.add_trace(go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode="markers+text",
            marker={
                "size": 9 if selected else 6,
                "color": color,
                "line": {"color": "#ffffff", "width": 1},
            },
            text=[component.name],
            textposition="top center",
            name=f"{component.name} origin",
            showlegend=False,
            hovertemplate=(
                f"<b>{component.name}</b><br>"
                f"x={position[0]:.3f} m · y={position[1]:.3f} m · z={position[2]:.3f} m"
                "<extra>Component origin</extra>"
            ),
        ))

    # Aircraft body axes.
    axis_length = 0.25
    for axis, end, colour in [
        ("X", (axis_length, 0, 0), "#dc2626"),
        ("Y", (0, axis_length, 0), "#16a34a"),
        ("Z", (0, 0, axis_length), "#2563eb"),
    ]:
        figure.add_trace(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line={"color": colour, "width": 5},
            text=["", axis],
            showlegend=False,
            hoverinfo="skip",
        ))

    figure.update_layout(
        template="plotly_white",
        height=650,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X — forward (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "yaxis": {"title": "Y — starboard (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "zaxis": {"title": "Z — up (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "camera": {"eye": {"x": 1.6, "y": 1.5, "z": 1.15}},
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
    )
    return figure


def _component_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:36] or "component"


def _unique_component_id(base: str, existing: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _uploaded_bytes(contents: str) -> bytes:
    if not contents or "," not in contents:
        raise ValueError("The uploaded file is empty or invalid")
    encoded = contents.split(",", 1)[1]
    if len(encoded) > (MAX_UPLOAD_BYTES * 4 // 3 + 8):
        raise ValueError("The file exceeds the 20 MB upload limit")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 20 MB upload limit")
    return raw


def layout(**kwargs):
    catalog_entries = list_component_catalog_entries()
    catalog_options = [
        {
            "label": f"{entry['name']} · {entry.get('manufacturer', 'Custom')}",
            "value": entry["catalog_id"],
        }
        for entry in catalog_entries
    ]
    default_catalog = "kingtech-k260g5" if any(
        option["value"] == "kingtech-k260g5" for option in catalog_options
    ) else (catalog_options[0]["value"] if catalog_options else None)

    return html.Div([
        dcc.Store(id="builder-project-store", storage_type="session", data=_empty_project()),
        dcc.Download(id="builder-download"),
        page_header(
            "3D Assembly Builder",
            "Compose a physical aircraft assembly, place its components and manage imported geometry.",
        ),
        dbc.Alert(
            [
                html.Strong("Coordinate system: "),
                "X forward · Y starboard · Z upward. Lengths in metres, angles in degrees. "
                "STL units must be selected during import. Imported mesh mass and centre of gravity "
                "are not inferred from geometry.",
            ],
            color="info",
            className="mb-3",
        ),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("3D viewport", className="fw-semibold"),
                        html.Small("  Select a component in the panel to highlight it", className="text-muted ms-2"),
                    ]),
                    dbc.CardBody(dcc.Graph(
                        id="builder-scene",
                        figure=_build_scene(_empty_project()),
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "scale": 2},
                        },
                        style={"height": "660px"},
                    )),
                ], className="mb-3"),
                html.Div(id="builder-mass-summary"),
                html.Div(id="builder-component-table", className="mt-3"),
            ], lg=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Aircraft project"),
                    dbc.CardBody([
                        dbc.Label("Aircraft name"),
                        dbc.Input(id="builder-aircraft-name", value="New aircraft", type="text", className="mb-3"),
                        dbc.Label("Add commercial component"),
                        dcc.Dropdown(
                            id="builder-catalog-select",
                            options=catalog_options,
                            value=default_catalog,
                            clearable=False,
                            placeholder="Choose a catalogue part",
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Add to assembly",
                            id="builder-add-catalog",
                            color="primary",
                            size="sm",
                            className="w-100 mb-3",
                            disabled=not bool(catalog_options),
                        ),
                        html.Hr(),
                        html.Div("Create a custom rectangular part", className="fw-semibold mb-2"),
                        dbc.Label("Part name"),
                        dbc.Input(id="builder-box-name", value="Custom component", type="text", className="mb-2"),
                        dbc.Label("Mass (kg)"),
                        dbc.Input(id="builder-box-mass", value=0.1, type="number", min=0, step=0.01, className="mb-2"),
                        dbc.Label("Dimensions X / Y / Z (m)"),
                        dbc.Row([
                            dbc.Col(dbc.Input(id="builder-box-dx", value=0.1, type="number", min=0.001, step=0.01), width=4),
                            dbc.Col(dbc.Input(id="builder-box-dy", value=0.1, type="number", min=0.001, step=0.01), width=4),
                            dbc.Col(dbc.Input(id="builder-box-dz", value=0.1, type="number", min=0.001, step=0.01), width=4),
                        ], className="g-1 mb-2"),
                        dbc.Button("Add rectangular part", id="builder-add-box", color="secondary", outline=True, size="sm", className="w-100"),
                    ]),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Import 3D component"),
                    dbc.CardBody([
                        dbc.Label("File units (STL / IGES)"),
                        dcc.Dropdown(
                            id="builder-import-units",
                            options=[
                                {"label": "Millimetres", "value": "mm"},
                                {"label": "Centimetres", "value": "cm"},
                                {"label": "Metres", "value": "m"},
                                {"label": "Inches", "value": "in"},
                                {"label": "Feet", "value": "ft"},
                            ],
                            value="mm",
                            clearable=False,
                            className="mb-2",
                        ),
                        dbc.Label("Component mass (kg) — enter known mass"),
                        dbc.Input(id="builder-import-mass", value=0.0, type="number", min=0, step=0.01, className="mb-2"),
                        dcc.Upload(
                            id="builder-import-upload",
                            children=dbc.Button("Choose STL / STEP / IGES", color="dark", outline=True, size="sm", className="w-100"),
                            accept=".stl,.step,.stp,.iges,.igs",
                            multiple=False,
                        ),
                        html.Small(
                            "STL works without extra packages. STEP / IGES require the optional CadQuery engine on the server. Maximum upload: 20 MB.",
                            className="text-muted d-block mt-2",
                        ),
                    ]),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Selected component"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="builder-selected-component",
                            options=[],
                            value=None,
                            placeholder="Choose a component",
                            clearable=True,
                            className="mb-3",
                        ),
                        html.Div(id="builder-selected-info", className="text-muted small mb-2"),
                        dbc.Label("Name"),
                        dbc.Input(id="builder-edit-name", value="", type="text", className="mb-2"),
                        dbc.Label("Mass (kg)"),
                        dbc.Input(id="builder-edit-mass", value=0, type="number", min=0, step=0.01, className="mb-2"),
                        dbc.Label("Position X / Y / Z (m)"),
                        dbc.Row([
                            dbc.Col(dbc.Input(id="builder-pos-x", value=0, type="number", step=0.01), width=4),
                            dbc.Col(dbc.Input(id="builder-pos-y", value=0, type="number", step=0.01), width=4),
                            dbc.Col(dbc.Input(id="builder-pos-z", value=0, type="number", step=0.01), width=4),
                        ], className="g-1 mb-2"),
                        dbc.Label("Rotation roll / pitch / yaw (°)"),
                        dbc.Row([
                            dbc.Col(dbc.Input(id="builder-rot-x", value=0, type="number", step=1), width=4),
                            dbc.Col(dbc.Input(id="builder-rot-y", value=0, type="number", step=1), width=4),
                            dbc.Col(dbc.Input(id="builder-rot-z", value=0, type="number", step=1), width=4),
                        ], className="g-1 mb-3"),
                        dbc.Button("Apply position & rotation", id="builder-apply-transform", color="primary", size="sm", className="w-100 mb-2"),
                        dbc.Button("Remove component", id="builder-remove-component", color="danger", outline=True, size="sm", className="w-100"),
                    ]),
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Save / load project"),
                    dbc.CardBody([
                        dbc.Button("Download assembly JSON", id="builder-save", color="success", outline=True, size="sm", className="w-100 mb-2"),
                        dcc.Upload(
                            id="builder-load-upload",
                            children=dbc.Button("Load assembly JSON", color="secondary", outline=True, size="sm", className="w-100"),
                            accept=".json",
                            multiple=False,
                        ),
                    ]),
                ], className="mb-3"),
                html.Div(id="builder-status"),
            ], lg=4),
        ], className="g-3"),
    ])


@callback(
    Output("builder-project-store", "data"),
    Output("builder-status", "children"),
    Output("builder-selected-component", "value"),
    Input("builder-add-catalog", "n_clicks"),
    Input("builder-add-box", "n_clicks"),
    Input("builder-apply-transform", "n_clicks"),
    Input("builder-remove-component", "n_clicks"),
    Input("builder-import-upload", "contents"),
    Input("builder-load-upload", "contents"),
    State("builder-project-store", "data"),
    State("builder-catalog-select", "value"),
    State("builder-box-name", "value"),
    State("builder-box-mass", "value"),
    State("builder-box-dx", "value"),
    State("builder-box-dy", "value"),
    State("builder-box-dz", "value"),
    State("builder-selected-component", "value"),
    State("builder-aircraft-name", "value"),
    State("builder-edit-name", "value"),
    State("builder-edit-mass", "value"),
    State("builder-pos-x", "value"),
    State("builder-pos-y", "value"),
    State("builder-pos-z", "value"),
    State("builder-rot-x", "value"),
    State("builder-rot-y", "value"),
    State("builder-rot-z", "value"),
    State("builder-import-upload", "filename"),
    State("builder-import-units", "value"),
    State("builder-import-mass", "value"),
    State("builder-load-upload", "filename"),
    prevent_initial_call=True,
)
def mutate_project(
    add_catalog_clicks, add_box_clicks, apply_clicks, remove_clicks,
    import_contents, load_contents, project_data, catalog_id,
    box_name, box_mass, box_dx, box_dy, box_dz, selected_id, aircraft_name,
    edit_name, edit_mass, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
    import_filename, import_units, import_mass, load_filename,
):
    trigger = ctx.triggered_id
    try:
        if trigger == "builder-load-upload":
            raw = _uploaded_bytes(load_contents)
            loaded = json.loads(raw.decode("utf-8"))
            assembly, assets = _project_parts(loaded)
            first_id = assembly.components[0].component_id if assembly.components else None
            return _payload(assembly, assets), dbc.Alert(
                f"Project loaded: {assembly.name}", color="success", duration=5000
            ), first_id

        assembly, assets = _project_parts(project_data or _empty_project())
        existing = {item.component_id for item in assembly.components}

        if aircraft_name and aircraft_name.strip():
            assembly.name = aircraft_name.strip()

        if trigger == "builder-add-catalog":
            if not catalog_id:
                raise ValueError("Choose a catalogue component first")
            source = load_catalog_component(catalog_id).to_dict()
            component_id = _unique_component_id(source["component_id"], existing)
            source["component_id"] = component_id
            source["position_m"] = [0.0, 0.0, 0.0]
            component = Component.from_dict(source)
            assembly.add_component(component)
            return _payload(assembly, assets), dbc.Alert(
                f"Added {component.name}. Set its installation position and verify its mass budget.",
                color="success", duration=6000,
            ), component_id

        if trigger == "builder-add-box":
            base = _component_slug(box_name or "custom-component")
            component_id = _unique_component_id(base, existing)
            component = Component(
                component_id=component_id,
                name=(box_name or "Custom component").strip(),
                category="custom",
                mass_kg=float(box_mass or 0.0),
                geometry="box",
                dimensions_m=(float(box_dx), float(box_dy), float(box_dz)),
                mass_source="estimated",
            )
            assembly.add_component(component)
            return _payload(assembly, assets), dbc.Alert(
                f"Added {component.name}. Its shape is a solid rectangular-box approximation.",
                color="success", duration=5000,
            ), component_id

        if trigger == "builder-apply-transform":
            if not selected_id:
                raise ValueError("Choose a component to edit")
            current = assembly.get_component(selected_id).to_dict()
            current["name"] = (edit_name or current["name"]).strip()
            updated_mass = float(edit_mass)
            if updated_mass != current["mass_kg"]:
                current["mass_source"] = "estimated"
            current["mass_kg"] = updated_mass
            current["position_m"] = [float(pos_x), float(pos_y), float(pos_z)]
            current["rotation_deg"] = [float(rot_x), float(rot_y), float(rot_z)]
            updated = Component.from_dict(current)
            assembly.components = [
                updated if item.component_id == selected_id else item
                for item in assembly.components
            ]
            return _payload(assembly, assets), dbc.Alert(
                f"Updated {updated.name}. Mass properties will be recomputed automatically.",
                color="success", duration=4000,
            ), selected_id

        if trigger == "builder-remove-component":
            if not selected_id:
                raise ValueError("Choose a component to remove")
            removed = assembly.remove_component(selected_id)
            assets.pop(selected_id, None)
            next_id = assembly.components[0].component_id if assembly.components else None
            return _payload(assembly, assets), dbc.Alert(
                f"Removed {removed.name}", color="warning", duration=4000
            ), next_id

        if trigger == "builder-import-upload":
            if not import_filename:
                raise ValueError("The uploaded file has no filename")
            extension = Path(import_filename).suffix.lower()
            raw = _uploaded_bytes(import_contents)
            if extension == ".stl":
                mesh = parse_stl(raw, units=import_units or "mm")
            elif extension in {".step", ".stp", ".iges", ".igs"}:
                mesh = parse_step_iges(
                    raw, import_filename, iges_units=import_units or "mm"
                )
            else:
                raise ValueError("Unsupported file type. Use STL, STEP or IGES.")

            stem = Path(import_filename).stem
            base = _component_slug(stem)
            component_id = _unique_component_id(base, existing)
            dimensions = tuple(float(value) for value in mesh["bounds_m"]["dimensions"])
            component = Component(
                component_id=component_id,
                name=stem,
                category="imported_geometry",
                mass_kg=float(import_mass or 0.0),
                geometry="box",
                dimensions_m=dimensions,
                mass_source="estimated",
            )
            assembly.add_component(component)
            assets[component_id] = mesh
            note = (
                f"Imported {import_filename}: {mesh['triangle_count']} triangles. "
                "The mesh is centred at its bounding-box centre. Inertia currently uses a "
                "bounding-box approximation; enter a known mass and replace the estimate "
                "when measured data is available."
            )
            return _payload(assembly, assets), dbc.Alert(
                note, color="success", duration=10000
            ), component_id

        return no_update, no_update, no_update

    except Exception as exc:
        return no_update, dbc.Alert(
            f"Could not update the assembly: {exc}", color="danger"
        ), no_update


@callback(
    Output("builder-edit-name", "value"),
    Output("builder-edit-mass", "value"),
    Output("builder-pos-x", "value"),
    Output("builder-pos-y", "value"),
    Output("builder-pos-z", "value"),
    Output("builder-rot-x", "value"),
    Output("builder-rot-y", "value"),
    Output("builder-rot-z", "value"),
    Output("builder-selected-info", "children"),
    Input("builder-selected-component", "value"),
    State("builder-project-store", "data"),
)
def populate_component_editor(selected_id, project_data):
    if not selected_id or not project_data:
        return "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No component selected"
    try:
        assembly, assets = _project_parts(project_data)
        component = assembly.get_component(selected_id)
    except (KeyError, ValueError):
        return "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Component not found"

    geometry_note = "Imported mesh · bounding-box inertia approximation" if selected_id in assets else component.geometry
    source_note = f"Mass source: {component.mass_source}"
    if component.manufacturer:
        source_note += f" · {component.manufacturer} {component.model or ''}".strip()
    return (
        component.name,
        component.mass_kg,
        component.position_m[0],
        component.position_m[1],
        component.position_m[2],
        component.rotation_deg[0],
        component.rotation_deg[1],
        component.rotation_deg[2],
        f"{component.category} · {geometry_note} · {source_note}",
    )


@callback(
    Output("builder-scene", "figure"),
    Output("builder-selected-component", "options"),
    Output("builder-mass-summary", "children"),
    Output("builder-component-table", "children"),
    Input("builder-project-store", "data"),
    Input("builder-selected-component", "value"),
)
def render_builder(project_data, selected_id):
    project = project_data or _empty_project()
    assembly, assets = _project_parts(project)
    figure = _build_scene(project, selected_id)
    options = [
        {"label": f"{item.name} · {item.mass_kg:.3f} kg", "value": item.component_id}
        for item in assembly.components
    ]
    props = assembly.mass_properties()
    cg = props.center_of_gravity_m
    cg_text = "—" if cg is None else f"X {cg[0]:.3f} · Y {cg[1]:.3f} · Z {cg[2]:.3f} m"

    summary = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Small("Total assembly mass", className="text-muted"), html.H4(f"{props.mass_kg:.3f} kg", className="mb-0")], md=3),
            dbc.Col([html.Small("Centre of gravity", className="text-muted"), html.Div(cg_text, className="fw-semibold")], md=5),
            dbc.Col([html.Small("Components", className="text-muted"), html.H4(str(props.component_count), className="mb-0")], md=2),
            dbc.Col([html.Small("Imported meshes", className="text-muted"), html.H4(str(len(assets)), className="mb-0")], md=2),
        ]),
        html.Hr(className="my-2"),
        html.Small(
            "Principal inertia (kg·m²): "
            + " · ".join(f"{value:.5g}" for value in props.principal_moments_kg_m2),
            className="text-muted",
        ),
    ]), className="mb-3")

    if not assembly.components:
        table = dbc.Alert(
            "The assembly is empty. Add a commercial part, create a box, or import a 3D file.",
            color="light",
        )
    else:
        rows = []
        for item in assembly.components:
            rows.append(html.Tr([
                html.Td(item.name),
                html.Td(item.category),
                html.Td(f"{item.mass_kg:.3f} kg"),
                html.Td(f"{item.position_m[0]:.2f}, {item.position_m[1]:.2f}, {item.position_m[2]:.2f}"),
                html.Td("Mesh" if item.component_id in assets else item.geometry),
            ]))
        table = dbc.Card([
            dbc.CardHeader("Assembly component list"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Component"), html.Th("Category"), html.Th("Mass"),
                    html.Th("Position X / Y / Z (m)"), html.Th("Geometry"),
                ])),
                html.Tbody(rows),
            ], bordered=False, hover=True, responsive=True, size="sm", className="mb-0"),
        ])
    return figure, options, summary, table


@callback(
    Output("builder-download", "data"),
    Input("builder-save", "n_clicks"),
    State("builder-project-store", "data"),
    prevent_initial_call=True,
)
def download_project(clicks, project_data):
    if not clicks or not project_data:
        return no_update
    assembly, assets = _project_parts(project_data)
    data = json.dumps(_payload(assembly, assets), indent=2)
    return dcc.send_string(data, "aircraft-assembly.json")
