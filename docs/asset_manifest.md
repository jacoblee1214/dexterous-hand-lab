# Hand asset manifest and acquisition requirements

No source URL or complete redistribution license was present in the project at
the time of repository preparation. This manifest records the required package
layout without changing or deleting any asset.

All asset directories listed here are intentionally excluded from the first
private Git snapshot. They remain available in the original local workspace.

## Runtime source packages

| Package | Required entry point | SHA-256 | Approx. package size |
| --- | --- | --- | ---: |
| V6 right | `assets/V6_Force_R/urdf/V6_Force_R.urdf` | `b0fcc792f9905cf5c565a01eb867aded877a1c4e2616af344d2622f6646df292` | 28 MB |
| V6 left | `assets/V6_Force_L/urdf/V6_Force_L.urdf` | `f3378f6deb84770365d2e71f0c3369bc6453b1f560fc07e9bf2a1d5b2b76b4f2` | 27 MB |

Each package also requires its sibling `meshes/` directory. The model builder
resolves the URDF package mesh references against that directory. Both package
metadata files declare `BSD`, but neither package contains a complete license
text and the author/maintainer fields are placeholders. Obtain written
authorization and the applicable license/notice before repository upload.

## Legacy/duplicate candidates

- `assets/left_hand/` is a 117 MB legacy left-hand URDF/STL set. Its entry-point
  hash is `02d220c2235cb40424633f809c903622c218bc0a1d9ab647878734df41902da3`.
  It is not the current V6 left runtime source and has no included license.
- `assets/meshes/` and `assets/urdf/` are older right-hand copies. Current code
  resolves the V6 package instead. Their provenance and need in a first snapshot
  must be decided without deleting the local originals.

If assets are omitted from Git, obtain the exact authorized archives from the
owner and unpack them into the paths above, then run:

```bash
python -m simulation.model_builder --hand right
python -m simulation.model_builder --hand left
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_model_builder.py
```

Do not download similarly named geometry from an unverified source or replace it
with fabricated meshes; geometry changes invalidate calibration and experiments.
