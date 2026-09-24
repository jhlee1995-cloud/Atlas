"""
atlas/faults.py -- every batch-4 sensor fault, generated in code on uint8 CIFAR-size images (docs/plans/
B4_INTEGRATION.md D5). Owner: T3S. numpy + stdlib only. No file is read here: the caller passes the clean images and, for
`paste`, the paired CIFAR-10-C severity-5 images of the same rows. Every fault is a deterministic function of its key.

Kinds, named distinctly (D5):
  T1 frame faults (fixed position per (kind, level, salt): the SAME defect in every frame, as a sensor defect is):
    deadpix          a seeded set of 0.5 / 2 / 5 % of the pixel positions (l1 / l2 / l3) set to 0 or 255, all channels
    occlusion_disc   a seeded opaque disc of 10 / 20 / 30 % area, value 32, lying inside the frame (lens obstruction)
    exposure_global  HOLDOUT (confirmation only): gain 1.5 / 2.0 / 3.0, rounded, clipped at 255 (position-free)
    salt: 0 in the fit layout, 1 in the eval layout (T1 review E6: confirmation sees defect geometry it has not seen)
  Lane S local faults (position per (kind, area, dataset row); square side a06 = 8 px (6.25 %), a12 = 11 px (11.8 %),
  a25 = 16 px (25 %) on a 32 x 32 image):
    paste__<corr>    the paired CIFAR-10-C s5 pixels of the same row inside the square (X5(c))
    occluder_sq      the square filled with the clean image's global mean colour
    glare            additive Gaussian blob (sigma = side / 2, peak +1.5 of full scale) confined to the square, clipped
    dead_sq          50 % of the square's pixels set to 0 (a dead-pixel cluster)
    soiling          HOLDOUT (confirmation units only): soft dark blob x * (1 - 0.6 m), m = a disk of radius side / 2
                     blurred by a 5-tap binomial filter, plus a 3 x 3 box blur under m (lens dirt)
  Lane S global corruptions (whole-image CIFAR-10-C s3 rows; no generator): global__<corr>__s3 for S_GLOBALS.

INTERFACE (stable at P1)
  t1_fault(imgs, kind, level, salt=0) -> uint8 (N, H, W, 3)
  t1_fault_mask(kind, level, salt=0, H=32, W=32) -> bool (H, W) | None (exposure_global: no mask)
  s_fault(kind, clean, area, row, seed=SEED, paste_src=None) -> (uint8 (H, W, 3), bool mask (H, W))
  s_fault_batch(kind, area, clean_imgs, rows, seed=SEED, paste_imgs=None) -> (uint8 (N, H, W, 3), bool (N, H, W))
  parse_condition(cond) -> (kind, corruption | None, area | severity)
  s_conditions(confirmation) -> the frozen lane-S condition list (soiling only for confirmation units)
  T1_KINDS, T1_LEVELS, T1_HOLDOUT, S_KINDS, S_SIDES, S_CONDITIONS, S_GLOBALS, S_HOLDOUT, CONFIRMATION_ONLY_FAULTS
"""
import zlib

import numpy as np

SEED = 20260924
# ---- T1 frame faults ------------------------------------------------------------------------------------------------
T1_KINDS = ("deadpix", "occlusion_disc", "exposure_global")
T1_LEVELS = {"deadpix": {1: 0.005, 2: 0.02, 3: 0.05},
             "occlusion_disc": {1: 0.10, 2: 0.20, 3: 0.30},
             "exposure_global": {1: 1.5, 2: 2.0, 3: 3.0}}
T1_HOLDOUT = ("exposure_global",)
OCCLUSION_VALUE = 32
# ---- lane S ---------------------------------------------------------------------------------------------------------
S_KINDS = ("paste", "occluder_sq", "glare", "dead_sq", "soiling")
S_SIDES = {"a06": 8, "a12": 11, "a25": 16}
S_HOLDOUT = ("soiling",)
S_PASTE = ("gaussian_noise", "defocus_blur", "pixelate", "jpeg_compression")      # discovery corruptions only
S_CONDITIONS = tuple([f"paste__{c}__{a}" for c in S_PASTE for a in ("a06", "a12", "a25")]
                     + ["occluder_sq__a06", "occluder_sq__a12", "occluder_sq__a25", "glare__a12", "glare__a25",
                        "dead_sq__a06", "dead_sq__a12", "soiling__a12", "soiling__a25"])
S_GLOBALS = ("gaussian_noise", "defocus_blur", "pixelate", "jpeg_compression", "fog", "brightness", "contrast")
S_GLOBAL_SEVERITY = 3
CONFIRMATION_ONLY_FAULTS = T1_HOLDOUT + S_HOLDOUT
_BINOM5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _rng(*parts, seed=SEED):
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(p).encode("utf-8")) for p in parts])


# =====================================================================================================================
# T1 frame faults
# =====================================================================================================================
def t1_fault_mask(kind, level, salt=0, H=32, W=32):
    """The fixed defect region of (kind, level, salt); None for exposure_global (position-free)."""
    if kind not in T1_KINDS:
        raise ValueError(f"unknown T1 fault kind {kind!r}")
    lv = T1_LEVELS[kind][int(level)]
    rng = _rng("t1", kind, int(level), int(salt))
    if kind == "deadpix":
        n = int(round(lv * H * W))
        pos = rng.choice(H * W, size=n, replace=False)
        m = np.zeros(H * W, dtype=bool)
        m[pos] = True
        return m.reshape(H, W)
    if kind == "occlusion_disc":
        r = float(np.sqrt(lv * H * W / np.pi))
        cy, cx = rng.uniform(r, H - r), rng.uniform(r, W - r)                  # the disc lies inside the frame
        yy, xx = np.mgrid[0:H, 0:W]
        return (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= r * r
    return None


def t1_fault(imgs, kind, level, salt=0):
    """uint8 (N, H, W, 3) -> a faulted uint8 copy."""
    imgs = np.asarray(imgs, dtype=np.uint8)
    if kind not in T1_KINDS:
        raise ValueError(f"unknown T1 fault kind {kind!r}")
    N, H, W, _ = imgs.shape
    lv = T1_LEVELS[kind][int(level)]
    if kind == "exposure_global":
        return np.clip(np.rint(imgs.astype(np.float64) * lv), 0, 255).astype(np.uint8)
    x = imgs.copy()
    if kind == "deadpix":
        rng = _rng("t1", kind, int(level), int(salt))
        n = int(round(lv * H * W))
        pos = rng.choice(H * W, size=n, replace=False)                         # = t1_fault_mask's draw (same stream)
        val = np.where(rng.random(n) < 0.5, 0, 255).astype(np.uint8)
        flat = x.reshape(N, H * W, 3)
        flat[:, pos, :] = val[None, :, None]
        return flat.reshape(N, H, W, 3)
    m = t1_fault_mask(kind, level, salt, H, W)                                  # occlusion_disc
    x[:, m, :] = OCCLUSION_VALUE
    return x


# =====================================================================================================================
# lane S local faults
# =====================================================================================================================
def _square_origin(rng, side, H=32, W=32):
    return int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))


def _conv_sep(img2d, k):
    """Separable 1-D kernel on both axes, reflect padding."""
    r = len(k) // 2
    p = np.pad(img2d, r, mode="reflect")
    t = sum(k[i] * p[i:i + img2d.shape[0], :] for i in range(len(k)))
    return sum(k[j] * t[:, j:j + img2d.shape[1]] for j in range(len(k)))


def s_fault(kind, clean, area, row, seed=SEED, paste_src=None):
    """(faulted uint8 (H, W, 3), mask bool (H, W)) for one image; `clean` uint8 (H, W, 3); the square's position is
    keyed by (seed, kind, area, dataset row)."""
    if kind not in S_KINDS:
        raise ValueError(f"unknown lane-S fault kind {kind!r}")
    side = S_SIDES[area]
    H, W = clean.shape[:2]
    rng = _rng("fault", kind, area, int(row), seed=seed)
    i, j = _square_origin(rng, side, H, W)
    mask = np.zeros((H, W), dtype=bool)
    mask[i:i + side, j:j + side] = True
    x = clean.astype(np.float64) / 255.0
    if kind == "paste":
        if paste_src is None or paste_src.shape != clean.shape:
            raise ValueError("paste needs the paired corrupt image of the same shape")
        out = x.copy()
        out[mask] = paste_src.astype(np.float64)[mask] / 255.0
    elif kind == "occluder_sq":
        out = x.copy()
        out[mask] = x.reshape(-1, 3).mean(0)
    elif kind == "glare":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cy, cx, s = i + (side - 1) / 2.0, j + (side - 1) / 2.0, side / 2.0
        blob = 1.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s * s)) * mask
        out = x + blob[..., None]
    elif kind == "dead_sq":
        out = x.copy()
        kill = mask & (rng.random((H, W)) < 0.5)
        out[kill] = 0.0                                                         # mask stays the square (localisation)
    else:                                                                       # soiling
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cy, cx, rad = i + (side - 1) / 2.0, j + (side - 1) / 2.0, side / 2.0
        disk = (((yy - cy) ** 2 + (xx - cx) ** 2) <= rad * rad).astype(np.float64)
        m = np.clip(_conv_sep(disk, _BINOM5), 0.0, 1.0)
        box = np.ones(3) / 3.0
        blurred = np.stack([_conv_sep(x[..., c], box) for c in range(3)], -1)
        out = (1.0 - m[..., None]) * x + m[..., None] * blurred
        out = out * (1.0 - 0.6 * m[..., None])
        mask = m > 0.05
    out = np.clip(out, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8), mask


def s_fault_batch(kind, area, clean_imgs, rows, seed=SEED, paste_imgs=None):
    """One (kind, area) on a batch: clean_imgs (N, H, W, 3) uint8, rows (N,) dataset rows (the position key)."""
    clean_imgs = np.asarray(clean_imgs, dtype=np.uint8)
    imgs = np.empty_like(clean_imgs)
    masks = np.empty(clean_imgs.shape[:3], dtype=bool)
    for n in range(len(clean_imgs)):
        imgs[n], masks[n] = s_fault(kind, clean_imgs[n], area, int(rows[n]), seed,
                                    None if paste_imgs is None else paste_imgs[n])
    return imgs, masks


def parse_condition(cond):
    """'paste__gaussian_noise__a12' -> ('paste', 'gaussian_noise', 'a12'); 'occluder_sq__a12' -> ('occluder_sq', None,
    'a12'); 'global__fog__s3' -> ('global', 'fog', 3)."""
    p = cond.split("__")
    if p[0] == "paste" and len(p) == 3:
        return "paste", p[1], p[2]
    if p[0] == "global" and len(p) == 3:
        return "global", p[1], int(p[2][1:])
    if len(p) == 2 and p[0] in S_KINDS:
        return p[0], None, p[1]
    raise ValueError(f"bad lane-S condition {cond!r}")


def s_conditions(confirmation):
    """The frozen lane-S local-fault conditions of a unit: the held-out soiling family only on confirmation units."""
    return [c for c in S_CONDITIONS if confirmation or parse_condition(c)[0] not in S_HOLDOUT]
