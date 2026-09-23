"""
lowlevel.py -- input-side factors computed from raw uint8 images (N, H, W, 3).

Each factor is one (N,) continuous target for a per-layer linear probe (ridge, CV R^2).
The list was chosen against the data already in the project:

  CIFAR-10-C family     factor(s) that should move
  ---------------------------------------------------------------
  noise  (gaussian/shot/impulse)   noise_sigma, highfreq_ratio (up)
  blur   (defocus/glass/zoom)      highfreq_ratio (down), spectral_slope (steeper)
  blur   (motion)                  spectral_anisotropy (up)          <- row-6 "motion"
  weather (snow/frost/fog)         contrast_rms (down), colorfulness, edge_density
  digital brightness               luminance_mean                    <- row-6 "brightness"
  digital contrast                 contrast_rms
  digital elastic/pixelate/jpeg    edge_density, highfreq_ratio, blockiness

Everything is vectorized numpy; no torch. Images may be any square size (32 or 224).
"""
import numpy as np

from ..registry import factor


# ---- shared helpers ----------------------------------------------------------
def _luma(img):
    """(N,H,W,3) uint8 -> (N,H,W) float in [0,1]."""
    x = img.astype(np.float32) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _grad(Y):
    """Central-difference gradients (N,H,W) -> gx, gy."""
    gx = np.zeros_like(Y)
    gy = np.zeros_like(Y)
    gx[:, :, 1:-1] = (Y[:, :, 2:] - Y[:, :, :-2]) * 0.5
    gy[:, 1:-1, :] = (Y[:, 2:, :] - Y[:, :-2, :]) * 0.5
    return gx, gy


def _power_spectrum(Y):
    """Mean-removed 2D FFT power, fftshifted; plus radial-frequency grid in cycles/pixel."""
    Yc = Y - Y.mean(axis=(1, 2), keepdims=True)
    P = np.abs(np.fft.fftshift(np.fft.fft2(Yc), axes=(1, 2))) ** 2
    H, W = Y.shape[1:]
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)
    TH = np.arctan2(FY, FX) % np.pi
    return P, R, TH


# ---- factors -----------------------------------------------------------------
@factor("luminance_mean", kind="continuous", pool="mixed")
def luminance_mean(img):
    """Mean luminance in [0,1]. Target of the 'brightness' corruption (row 6)."""
    return _luma(img).mean(axis=(1, 2))


@factor("contrast_rms", kind="continuous", pool="mixed")
def contrast_rms(img):
    """RMS contrast = std of luminance. Drops under fog/contrast/frost."""
    return _luma(img).std(axis=(1, 2))


@factor("highfreq_ratio", kind="continuous", pool="mixed")
def highfreq_ratio(img):
    """Fraction of spectral power at radial frequency > 0.25 cycles/px.
    Down under any blur (defocus/glass/zoom), up under noise."""
    P, R, _ = _power_spectrum(_luma(img))
    tot = P.sum(axis=(1, 2)) + 1e-12
    hi = (P * (R > 0.25)[None]).sum(axis=(1, 2))
    return hi / tot


@factor("spectral_slope", kind="continuous", pool="mixed")
def spectral_slope(img):
    """Slope of log radial power vs log frequency (natural images ~ -2). More negative
    = blurrier; flatter = noisier. Fit over 0.05 < f < 0.45 cycles/px."""
    P, R, _ = _power_spectrum(_luma(img))
    mask = (R > 0.05) & (R < 0.45)
    bins = np.linspace(0.05, 0.45, 9)
    idx = np.digitize(R[mask], bins) - 1
    logf = np.log(0.5 * (bins[:-1] + bins[1:]))
    Pm = P[:, mask]
    # radial average per bin, vectorized over images
    sums = np.zeros((len(img), len(bins) - 1))
    counts = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        sel = idx == b
        counts[b] = sel.sum()
        if counts[b] > 0:
            sums[:, b] = Pm[:, sel].mean(axis=1)
    valid = counts > 0
    lp = np.log(sums[:, valid] + 1e-12)
    x = logf[valid]
    x = x - x.mean()
    out = (lp * x).sum(axis=1) / (x * x).sum()
    return out.astype(np.float32)


@factor("spectral_anisotropy", kind="continuous", pool="mixed")
def spectral_anisotropy(img):
    """Angular non-uniformity of spectral power over 8 orientation sectors (0..pi),
    restricted to mid/high frequencies (R > 0.1): max sector energy / mean sector
    energy. 1.0 = isotropic. Motion blur (a directional low-pass) raises it (row 6)."""
    P, R, TH = _power_spectrum(_luma(img))
    mask = R > 0.1
    sec = np.minimum((TH / np.pi * 8).astype(int), 7)
    E = np.zeros((len(img), 8))
    for s in range(8):
        sel = mask & (sec == s)
        E[:, s] = P[:, sel].sum(axis=1)
    return (E.max(axis=1) / (E.mean(axis=1) + 1e-12)).astype(np.float32)


@factor("noise_sigma", kind="continuous", pool="mixed")
def noise_sigma(img):
    """Immerkaer fast noise estimate: sigma ~ sqrt(pi/2) / (6 (H-2)(W-2)) * sum|Y * L|
    with L the 3x3 Laplacian mask. Rises under gaussian/shot/impulse noise."""
    Y = _luma(img)
    L = (Y[:, :-2, 1:-1] + Y[:, 2:, 1:-1] + Y[:, 1:-1, :-2] + Y[:, 1:-1, 2:]
         - 2 * (Y[:, :-2, :-2] + Y[:, :-2, 2:] + Y[:, 2:, :-2] + Y[:, 2:, 2:])
         + 4 * Y[:, 1:-1, 1:-1])
    H, W = Y.shape[1:]
    return (np.sqrt(np.pi / 2) / (6 * (H - 2) * (W - 2)) * np.abs(L).sum(axis=(1, 2))).astype(np.float32)


@factor("saturation_mean", kind="continuous", pool="mixed")
def saturation_mean(img):
    """Mean HSV saturation = (max - min) / max over RGB."""
    x = img.astype(np.float32) / 255.0
    mx = x.max(axis=3)
    mn = x.min(axis=3)
    return ((mx - mn) / (mx + 1e-6)).mean(axis=(1, 2))


def _hue_vec(img):
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=3)
    mn = x.min(axis=3)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6,
        np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * (np.pi / 3)
    w = (mx - mn)                     # saturation-like weight
    c = (np.cos(h) * w).sum(axis=(1, 2)) / (w.sum(axis=(1, 2)) + 1e-6)
    s = (np.sin(h) * w).sum(axis=(1, 2)) / (w.sum(axis=(1, 2)) + 1e-6)
    return c, s


@factor("hue_cos", kind="continuous", pool="mixed")
def hue_cos(img):
    """cos of the saturation-weighted circular mean hue (with hue_sin, encodes dominant hue)."""
    return _hue_vec(img)[0]


@factor("hue_sin", kind="continuous", pool="mixed")
def hue_sin(img):
    """sin of the saturation-weighted circular mean hue."""
    return _hue_vec(img)[1]


@factor("colorfulness", kind="continuous", pool="mixed")
def colorfulness(img):
    """Hasler-Suesstrunk colorfulness: sqrt(s_rg^2 + s_yb^2) + 0.3 sqrt(m_rg^2 + m_yb^2)."""
    x = img.astype(np.float32)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    s = np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2)
    m = np.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)
    return (s + 0.3 * m).astype(np.float32)


@factor("edge_density", kind="continuous", pool="mixed")
def edge_density(img):
    """Fraction of pixels whose gradient magnitude exceeds 0.1 (luminance units)."""
    gx, gy = _grad(_luma(img))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return (mag > 0.1).mean(axis=(1, 2)).astype(np.float32)


@factor("orientation_entropy", kind="continuous", pool="mixed")
def orientation_entropy(img):
    """Entropy (normalized to [0,1]) of the magnitude-weighted gradient-orientation
    histogram over 8 bins in [0, pi). Low = one dominant edge direction."""
    gx, gy = _grad(_luma(img))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    th = np.arctan2(gy, gx) % np.pi
    bins = np.minimum((th / np.pi * 8).astype(int), 7)
    H = np.zeros((len(img), 8))
    for b in range(8):
        H[:, b] = (mag * (bins == b)).sum(axis=(1, 2))
    p = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(8)
    return ent.astype(np.float32)


@factor("blockiness", kind="continuous", pool="mixed")
def blockiness(img):
    """8x8 block-boundary discontinuity ratio (JPEG / pixelate signature): mean |diff|
    across pixel columns/rows at multiples of 8 divided by mean |diff| elsewhere."""
    Y = _luma(img)
    dx = np.abs(np.diff(Y, axis=2))
    dy = np.abs(np.diff(Y, axis=1))
    W = dx.shape[2]
    Hh = dy.shape[1]
    bx = np.zeros(W, dtype=bool)
    bx[7::8] = True
    by = np.zeros(Hh, dtype=bool)
    by[7::8] = True
    on = (dx[:, :, bx].mean(axis=(1, 2)) + dy[:, by, :].mean(axis=(1, 2))) * 0.5
    off = (dx[:, :, ~bx].mean(axis=(1, 2)) + dy[:, ~by, :].mean(axis=(1, 2))) * 0.5
    return (on / (off + 1e-6)).astype(np.float32)


PIXEL_FACTOR_ORDER = [
    "luminance_mean", "contrast_rms", "highfreq_ratio", "spectral_slope",
    "spectral_anisotropy", "noise_sigma", "saturation_mean", "hue_cos", "hue_sin",
    "colorfulness", "edge_density", "orientation_entropy", "blockiness",
]


def compute_pixel_factors(img_uint8, names=None, chunk=2048):
    """Compute all (or the named) pixel factors on (N,H,W,3) uint8, chunked for memory."""
    from ..registry import FACTORS
    names = names or [n for n in PIXEL_FACTOR_ORDER if n in FACTORS]
    out = {n: [] for n in names}
    for i in range(0, len(img_uint8), chunk):
        blk = img_uint8[i:i + chunk]
        for n in names:
            out[n].append(np.asarray(FACTORS[n].fn(blk), dtype=np.float32))
    return {n: np.concatenate(v) for n, v in out.items()}
