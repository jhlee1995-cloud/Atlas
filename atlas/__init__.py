"""
atlas -- Stage 0 of the "map the internal space first" pivot.

Two stages, mirroring the repo's extract/ (GPU) vs frame/ (CPU) split:

  STAGE A  atlas.extract_acts   GPU   data -> backbone -> multi-layer activation DUMP
                                      (+ input-side factors computed from raw pixels)
  STAGE B  atlas.build          CPU   dump -> per-layer invariants -> atlas.json / ATLAS.md
           atlas.compare        CPU   two dumps -> deformation.json (atlas vs atlas)
           atlas.critic         CPU   N atlases -> stability report (seed / scale / holdout)

Everything in Stage B is a plugin behind a registry (atlas.registry):
  @invariant("name")          per-layer   fn(ctx, cfg) -> dict
  @cross_layer("name")        all layers  fn(ctxs, per_layer, cfg) -> dict
  @factor("name", kind=...)   input side  fn(images_uint8) -> (N,) array

Adding a measurement = one new function + one line in the manifest.
Nothing here writes to memory or GitHub; results land under outputs.root of the manifest.
"""
__version__ = "0.2.0"
