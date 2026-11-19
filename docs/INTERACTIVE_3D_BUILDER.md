# Interactive 3D assembly builder

The `/builder` page introduces a first interactive component-assembly workspace.
It is intentionally separate from the legacy `/geometry` concept-view page.

## First-version features

- A 3D Plotly viewport using the aircraft body axes: X forward, Y starboard,
  Z upward.
- Add commercial components from the offline catalogue, including the
  KingTech K-260G5.
- Create custom rectangular components with explicit dimensions and mass.
- Select an installed component and edit its name, mass, position and
  roll/pitch/yaw orientation.
- Remove components from the assembly.
- Import binary or ASCII STL files without extra dependencies.
- Import STEP and IGES through the optional CadQuery/OpenCascade runtime.
- Save and reload the whole assembly and imported mesh geometry as JSON.
- Recalculate total mass, CG and principal inertia whenever the assembly
  changes.

## Running the web interface

Install the normal web UI and development extras:

```bash
uv sync --extra webui --extra dev
cd webui
uv run python app.py
```

The builder appears as **3D Builder** in the sidebar.

## Optional STEP / IGES support

STEP / IGES conversion requires the server to have CadQuery/OpenCascade
installed. Install the extra when building the server environment:

```bash
pip install -e ".[cad-import]"
```

CadQuery's STEP importer converts the imported model to metres. IGES files
must be assigned a source unit during upload because IGES exporters do not
consistently preserve unit metadata. The importer triangulates the model for
browser display; it does not preserve parametric CAD feature history.

Reference: [CadQuery importing/exporting documentation](https://cadquery.readthedocs.io/en/latest/importexport.html).

## Import safety and engineering assumptions

- Uploads are limited to 20 MB.
- Meshes are sampled to at most 12,000 triangles for responsive browser
  rendering.
- STL and IGES units are selected by the user; STL does not reliably encode
  units.
- Imported mesh origins are moved to the bounding-box centre.
- Imported components use their bounding box as a temporary solid-box inertia
  approximation. Replace it with measured or manufacturer mass properties
  when available.
- Imported mesh mass defaults to zero until the user enters a value.
- The catalogue K-260G5 geometry remains a simple envelope, not a vendor CAD
  file.

## Current limitations / next increments

This is a first assembly editor, not yet a full parametric CAD system. Position
and orientation are edited through numeric controls; viewport drag gizmos,
snapping, mating constraints, collision checks, persistent cloud project
storage and native CAD feature editing are future work. Plotly displays
triangulated surfaces rather than editable B-rep solids.
