"""Importing this package registers every invariant. Add a module here to add a measurement."""
from . import dimension, landmarks, adjacency, density, decodability, flow, sensitivity  # noqa: F401
# margin LAST: registration order breaks cost ties in build.py, and margin_typeb (cost "expensive", no ctx.rng
# draws) must not move any Stage 1 invariant's per-layer estimator draws (docs/plans/STAGE1.md amendment 2).
from . import margin  # noqa: F401,E402
