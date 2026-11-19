# Aircraft core: components and mass properties

This first foundation adds a reusable, solver-independent representation of
an aircraft as a collection of physical components.

## Coordinate system and units

- Right-handed aircraft body axes: `+X` forward, `+Y` starboard, `+Z` upward.
- Lengths: metres (`m`); mass: kilograms (`kg`); angles: degrees.
- Inertia: `kg·m²`, expressed about the aircraft centre of gravity and in the
  aircraft body frame.
- Component rotation uses `Rz(yaw) · Ry(pitch) · Rx(roll)`.

## Component model

Each `Component` has a stable ID, name, category, mass, aircraft-frame origin,
local orientation, and local centre-of-mass offset. Optional manufacturer,
model, source URL and mass provenance fields preserve traceability.

Built-in inertia approximations are available for a point mass, a homogeneous
solid box, and a homogeneous solid cylinder aligned with a local axis. For an
irregular real-world part, supply a 3×3 inertia tensor about its own centre of
mass. Do not use the point-mass approximation as a substitute for measured
inertia when higher-fidelity dynamics are required.

## Assembly calculations

`AircraftAssembly.mass_properties()` recomputes:

- total mass;
- three-dimensional centre of gravity;
- the full symmetric inertia tensor about the total CG;
- principal moments of inertia;
- the method used for each component's intrinsic inertia.

The calculation rotates each component's inertia tensor into the aircraft
frame and applies the parallel-axis theorem. Adding, removing or replacing a
component is followed by a fresh calculation; values are not cached.

## Example

```python
from rc_aircraft_design.aircraft import AircraftAssembly, Component

plane = AircraftAssembly(aircraft_id="rc-jet-01", name="RC jet", vehicle_type="jet")
plane.add_component(Component(
    component_id="turbine-01",
    name="Main turbine",
    category="propulsion",
    mass_kg=2.4,
    position_m=(0.82, 0.0, 0.04),
    geometry="cylinder_x",
    dimensions_m=(0.30, 0.12, 0.12),
    mass_source="manufacturer",
    manufacturer="Manufacturer name",
    model="Model name",
    source_url="https://manufacturer.example/specification",
))

properties = plane.mass_properties()
print(properties.to_dict())
```

The example values are illustrative only, not a specification for a real
commercial turbine. Future work will connect this core to a sourced component
catalogue, the 3D assembly editor and analysis solvers.
