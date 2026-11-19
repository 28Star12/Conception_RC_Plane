# Commercial component catalogue

The catalogue stores source-traceable commercial component records as JSON.
It is intended to remain offline-capable: loading a record does not make a
network request. Each entry separates the reusable physical component
record from additional manufacturer specifications, geometry notes, data
quality and source URLs.

## Current entry: KingTech K-260G5

The first seeded record is `kingtech-k260g5`. Manufacturer-published values
currently stored include:

| Property | Value | Provenance |
|---|---:|---|
| Diameter | 120 mm | Manufacturer published |
| Length | 299 mm | Manufacturer published |
| Turbine mass | 2.2 kg | Manufacturer published |
| Maximum RPM | 112,000 rpm | Manufacturer published |
| Maximum thrust | 26 kgf at 15 °C | Manufacturer published |
| Maximum EGT | 750 °C | Manufacturer published |
| Fuel consumption | 760 g/min | Manufacturer published; operating point not specified on page |
| Fuels listed | Diesel, Jet A-1, kerosene | Manufacturer published |
| Lubrication | 5% | Manufacturer published |
| Pump | KP1000DP digital SBUS brushless pump | Manufacturer published |
| Battery | 11.1 V LiPo, at least 2200 mAh / 35C | Manufacturer published |
| Start time | 10–15 s | Manufacturer published |

The database also stores the equivalent maximum thrust of approximately
254.97 N, calculated from the published 26 kgf using standard gravity.

### Important modelling limitations

- The 2.2 kg value is the turbine mass, not the complete propulsion system.
  Pump, ECU/DRM, GSU, fuel plumbing, battery, mounts, accessories and fuel must
  be represented as separate components when their masses are known.
- The 120 mm × 299 mm cylinder is only a simple envelope for preliminary
  placement and visualization. It is not a manufacturer CAD model.
- The internal inertia is estimated from a homogeneous solid-cylinder
  approximation. It is not a measured turbine inertia tensor.
- The manufacturer page does not provide a centre-of-gravity location or an
  inertia tensor. The catalog record therefore marks the CG-at-origin default
  as an assumption that must be replaced when better data are available.
- Published fuel consumption is recorded as stated, without assuming a
  throttle setting or deriving flight endurance from it.

## Python usage

```python
from rc_aircraft_design.aircraft import (
    AircraftAssembly,
    list_component_catalog_entries,
    load_catalog_component,
)

print([entry["catalog_id"] for entry in list_component_catalog_entries()])

aircraft = AircraftAssembly("jet-01", "RC jet", vehicle_type="jet")
turbine = load_catalog_component("kingtech-k260g5")
aircraft.add_component(turbine)
print(aircraft.mass_properties().to_dict())
```

## Sources

- KingTech Turbines Co., Ltd., K260G5: https://kingtechtw.com/product/k260g5/
- KingTech Turbine European product listing: https://kingtechturbine.lu/K260G5_1

Values must be rechecked against manufacturer documentation when a record is
updated. Do not treat catalog data as certification or flight-release data.
