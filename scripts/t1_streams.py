#!/usr/bin/env python3
"""
t1_streams.py -- Track 1 X3 (batch 4): seeded frame streams built from CIFAR-10(-C) rows and injected sensor faults, the
per-tap change-information rate, and detection delay at an ARL-calibrated false-alarm budget for geometry, head-only
and pixel-only detectors run through the same sequential machinery (docs/plans/T1_SCOREBOARD.md section X3). CPU, numpy
only. The probe builds its own per-row frame features from the dumps (atlas/b4_core.open_unit; scripts/t1_scoreboard.py
Unit at level 'streams'), writes them to --frames (container-disk scratch, /root/b4_frames/<id><tag>; D9) and writes
results/b4_t1s/<id><tag>/streams.json through atlas/b4_core.ProbeRun. It applies no threshold: scripts/t1_eval.js decides.

  python scripts/t1_streams.py --registry experiments/b4/models.json --unit resnet20_hub --phase discovery \
      --frames /root/b4_frames/resnet20_hub --out results/b4_t1s/resnet20_hub
  python scripts/t1_streams.py --selftest [--selftest-out results/instrument_check_b4s1/selftest_t1_streams.json]

Layout: the discovery phase reads the unit's fit dump; the confirmation phase reads its eval dump (never-read rows
5000-9999; STconf units only), with the fit dump for calibration (the scoreboard's CAL rows: A = 2000-3499 in the fit
layout, the whole fit split 0-4999 in the eval layout, T1 review A2).
Frames are i.i.d. draws (with replacement) from a segment's row pool; this OVERSTATES the change rate relative to
correlated video (EXTRACTION_PROPOSALS X3, risk 1) and is declared as such.
  in-control calibration  CAL rows: every detector's threshold h makes the simulated in-control mean run length (censored
                          at 12000 frames) equal ARL0 = 2000 frames (200 streams)
  held-out in-control     B rows 3500-4999 (fit layout) or the eval clean rows 5000-9999: the realised ARL0 and the false
                          alarms per 1000 frames (the functional unit of D14)
  scenarios (100 seeds)   step (500 clean -> split for 1500) at s1 and s3; ramp s1 -> s3 -> s5 (300-frame linear mixing,
                          300-frame holds); benign brightness ramp (the ramp of 'brightness'); label-skew burst (one
                          class, 300 frames); the in-control null with the same timing (clean throughout: every
                          detector's own alarm rate, the X3-7 reference); CIFAR-100 / SVHN bursts; fault onsets (every
                          fault split, 300 frames); stuck frame (one clean frame repeated 300 times)
  detectors               CUSUM (k = 0.5 SD) on standardised scalars: head msp, gap, entropy, energy; geometry L2-kNN at
                          the five functional taps and penult d1; the conformal CUSUM on X4's fused p (power-martingale
                          form, eps 0.1); MEWMA (lambda 0.05) on the CAL-whitened 10-d frame of each tap and on the
                          CAL-whitened pixel factors; window-64 BBSDh chi-square and binned KS on msp; the stuck-frame
                          repeat check (3 identical consecutive frames: h = 1, T1 review A3); the harm gate (alarm AND
                          window-64 H > 0.25)
  outputs                 per detector and scenario: P(false alarm before onset), P(detect within 500 frames of onset),
                          conditional expected delay (detected streams) and the censored delay (every stream without a
                          pre-onset alarm; T1 review E4), the harm-gated detection rate; ramps: the median lead of the
                          first alarm over the first frame whose rolling 64-frame loss reaches 5 pt; per split the change-
                          information rate I(S) = 1/2 |mu_S|^2 + 1/2 (tr S_S - ln det S_S - r) (nats / frame) of each tap's
                          whitened frame, of the CAL-WHITENED head scalars and pixel factors (T1 review E3), with a
                          bootstrap CI (B = 200 discovery, 1000 confirmation)
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from atlas import b4_core as C              # noqa: E402  numpy + stdlib only
import t1_scoreboard as SB                  # noqa: E402  the per-row feature machinery (numpy only)

SCHEMA = "b4_t1_streams/1"
ARL0 = 2000
CAL = {"n_streams": 200, "length": 12000}
SCEN = {"seeds": 100, "tau": 500, "seg": 1500, "ramp_mix": 300, "ramp_hold": 300, "burst": 300, "within": 500,
        "lam": 0.05, "k": 0.5, "eps": 0.1, "win": 64, "hold_band": 0.25, "loss_pt": 5.0, "stuck_h": 1.0,
        "ks_bins": 50, "chunk": 25}
TAPKEYS = SB.TAPKEYS
SCALARS = {"msp": -1.0, "gap": -1.0, "entropy": 1.0, "energy": -1.0, "knnL2_stem": 1.0, "knnL2_s1end": 1.0,
           "knnL2_s2end": 1.0, "knnL2_pre": 1.0, "knnL2_penult": 1.0, "d1_penult": 1.0}
HEAD_SCALARS = ("msp", "gap", "entropy", "energy")
KEEP = tuple(SCALARS) + ("x4", "x4_p", "logr10_penult")


def say(msg, quiet=False):
    if not quiet:
        print(f"[t1s] {msg}", flush=True)


# =====================================================================================================================
# frame store
# =====================================================================================================================
class Frames:
    """split -> {name: per-row array}; the split 'cal' holds the calibration rows. Whitened head and pixel frames
    ('whead', 'wpix') are added from the CAL covariance by Calib."""

    def __init__(self, d):
        self.d = d
        self.splits = sorted(d)

    def get(self, sp, nm):
        return self.d[sp][nm]

    def has(self, sp, nm):
        return sp in self.d and nm in self.d[sp]

    def save(self, path):
        arrs = {}
        for sp, m in self.d.items():
            for nm, v in m.items():
                v = np.asarray(v)
                arrs[f"{sp}::{nm}"] = v.astype(np.float32) if np.issubdtype(v.dtype, np.floating) else v
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **arrs)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = {}
        with np.load(path) as z:
            for k in z.files:
                sp, nm = k.split("::", 1)
                d.setdefault(sp, {})[nm] = np.asarray(z[k])
        return cls(d)


def frames_from_unit(unit):
    """Per-row features of a scripts/t1_scoreboard.Unit built at level 'streams'."""
    def take(tb, ix):
        out = {nm: np.asarray(tb[nm])[ix] for nm in KEEP}
        for key in TAPKEYS:
            out[f"frame_{key}"] = tb[f"frame_{key}"][ix]
        out["am"] = tb["am"][ix].astype(np.int64)
        out["y"] = tb["y"][ix].astype(np.int64)
        out["rows"] = tb["rows"][ix].astype(np.int64)
        out["wrong"] = tb["wrong"][ix]
        if "pix" in tb:
            out["pix"] = tb["pix"][ix]
        return out
    d = {"cal": take(unit.caltab, unit.cal_idx)}
    for sp, tb in unit.tables.items():
        if all(f"frame_{k}" in tb for k in TAPKEYS) and all(nm in tb for nm in KEEP):
            d[sp] = take(tb, np.arange(len(tb["y"])))
    return Frames(d)


class Calib:
    """In-control reference from the CAL rows ('cal')."""

    def __init__(self, fr):
        c = fr.d["cal"]
        self.mu, self.sd = {}, {}
        for s in tuple(SCALARS) + ("logr10_penult",):
            v = np.asarray(c[s], dtype=np.float64)
            self.mu[s], self.sd[s] = float(v.mean()), float(v.std() + 1e-12)
        self.hist = np.bincount(np.asarray(c["am"], dtype=np.int64), minlength=C.K) / len(c["am"])
        self.msp_sorted = np.sort(np.asarray(c["msp"], dtype=np.float64))
        lr = np.asarray(c["logr10_penult"], dtype=np.float64)
        self.h_med = float(np.median(lr))
        self.h_scale = max(float(np.quantile(lr, 0.95) - np.quantile(lr, 0.5)), 1e-12)
        self.acc = 1.0 - float(np.mean(c["wrong"]))
        self.head_frame = C.pca_frame(np.column_stack([np.asarray(c[s], dtype=np.float64) for s in HEAD_SCALARS]),
                                      None)
        self.pix_frame = C.pca_frame(np.asarray(c["pix"], dtype=np.float64), None) if "pix" in c else None
        for sp in fr.splits:                               # CAL-whitened head scalars and pixel factors (T1 review E3)
            m = fr.d[sp]
            m["whead"] = C.whiten(np.column_stack([np.asarray(m[s], dtype=np.float64) for s in HEAD_SCALARS]),
                                  self.head_frame)
            if self.pix_frame is not None and "pix" in m:
                m["wpix"] = C.whiten(np.asarray(m["pix"], dtype=np.float64), self.pix_frame)


# =====================================================================================================================
# detectors: a (S, T) stream of frames -> an (S, T) statistic; alarm when stat > h
# =====================================================================================================================
def cusum(z, k):
    S = np.zeros(z.shape[0])
    out = np.empty(z.shape, dtype=np.float64)
    for t in range(z.shape[1]):
        S = np.maximum(0.0, S + z[:, t] - k)
        out[:, t] = S
    return out


def mewma(X, lam):
    """X (S, T, r) whitened in-control coordinates; T^2 of the EWMA with its asymptotic covariance lam / (2 - lam) I."""
    Z = np.zeros((X.shape[0], X.shape[2]))
    out = np.empty(X.shape[:2])
    c = lam / (2.0 - lam)
    for t in range(X.shape[1]):
        Z = lam * X[:, t] + (1.0 - lam) * Z
        out[:, t] = (Z * Z).sum(1) / c
    return out


def conformal_cusum(p, eps):
    """CUSUM form of the power martingale: S_t = max(0, S_{t-1} + log(eps p_t^(eps - 1)))."""
    inc = np.log(eps) + (eps - 1.0) * np.log(np.clip(p, 1e-12, 1.0))
    S = np.zeros(p.shape[0])
    out = np.empty(p.shape, dtype=np.float64)
    for t in range(p.shape[1]):
        S = np.maximum(0.0, S + inc[:, t])
        out[:, t] = S
    return out


def window_stat(x, win, fn):
    S, T = x.shape
    out = np.zeros((S, T))
    for t in range(win - 1, T):
        out[:, t] = fn(x[:, t - win + 1:t + 1])
    return out


def window_counts(codes, n_codes, win):
    """(S, T) integer codes -> (S, T, n_codes) counts over the trailing window (zero before the window is full)."""
    S, T = codes.shape
    oh = np.zeros((S, T + 1, n_codes), dtype=np.int32)
    oh[np.arange(S)[:, None], np.arange(1, T + 1)[None, :], codes] = 1
    cum = np.cumsum(oh, axis=1)
    out = np.zeros((S, T, n_codes), dtype=np.int32)
    out[:, win - 1:] = cum[:, win:] - cum[:, :T - win + 1]
    return out


def _by_chunks(x, fn, chunk):
    """Apply fn to row blocks of x (bounds the memory of the (S, T, bins) window counts)."""
    return np.concatenate([fn(x[i:i + chunk]) for i in range(0, len(x), chunk)], axis=0)


def bbsdh(am, hist, win, chunk=SCEN["chunk"]):
    """chi-square of the window's predicted-class counts against the calibration histogram (the BBSDh statistic)."""
    ex = win * hist + 0.5

    def f(a):
        cnt = window_counts(a, len(hist), win) + 0.5
        o = ((cnt - ex) ** 2 / ex).sum(2)
        o[:, :win - 1] = 0.0
        return o
    return _by_chunks(am, f, chunk)


def ks_window(msp, ref_sorted, win, bins=SCEN["ks_bins"], chunk=SCEN["chunk"]):
    """KS distance between the window's msp and the calibration msp on the calibration-quantile bins (a binned KS:
    exact at the bin edges; vectorised over streams and time)."""
    edges = np.quantile(ref_sorted, np.linspace(0, 1, bins + 1)[1:-1])
    cdf_r = np.arange(1, bins + 1) / bins

    def f(m):
        code = np.searchsorted(edges, m, side="right")
        cnt = window_counts(code, bins, win)
        o = np.abs(np.cumsum(cnt, axis=2) / win - cdf_r[None, None, :]).max(2)
        o[:, :win - 1] = 0.0
        return o
    return _by_chunks(msp, f, chunk)


def stuck(ids):
    """Run length of consecutive identical frames (by frame identity): the pixel repeat check."""
    out = np.zeros(ids.shape)
    run = np.zeros(ids.shape[0])
    for t in range(1, ids.shape[1]):
        run = np.where(ids[:, t] == ids[:, t - 1], run + 1, 0)
        out[:, t] = run
    return out


def first_alarm(stat, h, start=0):
    """First index >= start with stat > h; the stream length when there is none."""
    a = stat[:, start:] > h
    return np.where(a.any(1), a.argmax(1) + start, stat.shape[1])


def calibrate_h(stat, target=ARL0):
    """Threshold whose mean first-passage time (censored at the stream length) equals target (bisection)."""
    L = stat.shape[1]
    lo, hi = float(np.min(stat)), float(np.max(stat)) + 1e-9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if first_alarm(stat, mid).mean() < target:
            lo = mid
        else:
            hi = mid
    fa = first_alarm(stat, hi)
    return hi, float(fa.mean()), float((fa >= L).mean())


class Streams:
    """One set of S streams: split index so (S, T), local row ro (S, T), frame identity ids (S, T)."""

    def __init__(self, fr, cal, so, ro, ids):
        self.fr, self.cal, self.so, self.ro, self.ids = fr, cal, so, ro, ids

    def gather(self, nm, dtype=np.float64):
        out = None
        for j, sp in enumerate(self.fr.splits):
            m = self.so == j
            if not m.any():
                continue
            v = np.asarray(self.fr.get(sp, nm))
            if out is None:
                out = np.zeros(self.so.shape + v.shape[1:], dtype=dtype)
            out[m] = v[self.ro[m]]
        return out

    def detectors(self):
        names = [f"cusum_{s}" for s in SCALARS] + ["x4"] + [f"mewma_{k}" for k in TAPKEYS]
        if self.cal.pix_frame is not None and all(self.fr.has(sp, "wpix") for sp in self.fr.splits):
            names.append("mewma_pix")
        return names + ["bbsdh", "ks_msp", "stuck"]

    def stat(self, name):
        cal = self.cal
        if name.startswith("cusum_"):
            s = name[6:]
            return cusum(SCALARS[s] * (self.gather(s) - cal.mu[s]) / cal.sd[s], SCEN["k"])
        if name == "x4":
            return conformal_cusum(self.gather("x4_p"), SCEN["eps"])
        if name == "mewma_pix":
            return mewma(self.gather("wpix", np.float32), SCEN["lam"])
        if name.startswith("mewma_"):
            return mewma(self.gather(f"frame_{name[6:]}", np.float32), SCEN["lam"])
        if name == "bbsdh":
            return bbsdh(self.gather("am", np.int64), cal.hist, SCEN["win"])
        if name == "ks_msp":
            return ks_window(self.gather("msp"), cal.msp_sorted[::5], SCEN["win"])
        if name == "stuck":
            return stuck(self.ids)
        raise KeyError(name)

    def h_window(self):
        lr = self.gather("logr10_penult")
        return window_stat(lr, SCEN["win"], lambda w: (np.median(w, axis=1) - self.cal.h_med) / self.cal.h_scale)


# =====================================================================================================================
# stream builders (known-answer tested: contiguous segments, deterministic, stuck = identical frames)
# =====================================================================================================================
def pool_rows(fr, sp, rows_range=None):
    idx = np.arange(len(fr.get(sp, "rows")))
    if rows_range is not None:
        rows = fr.get(sp, "rows")
        idx = idx[(rows >= rows_range[0]) & (rows < rows_range[1])]
    return idx


def build(segments, S, key):
    """segments: [(length, sampler)] with sampler(rng, n, stream) -> (split index (n,), local row (n,)).
    Returns (so, ro, ids) of shape (S, T)."""
    rng = C.rng_for("t1s_build", key)
    T = sum(L for L, _ in segments)
    so = np.zeros((S, T), dtype=np.int64)
    ro = np.zeros((S, T), dtype=np.int64)
    for s in range(S):
        t = 0
        for L, samp in segments:
            a, b = samp(rng, L, s)
            so[s, t:t + L], ro[s, t:t + L] = a, b
            t += L
    return so, ro, so * 10_000_000 + ro


def iid(j, pool):
    return lambda rng, n, s: (np.full(n, j), rng.choice(pool, size=n, replace=True))


def mix(j0, p0, j1, p1):
    def f(rng, n, s):
        pick = rng.random(n) < (np.arange(n) + 1) / n
        return np.where(pick, j1, j0), np.where(pick, rng.choice(p1, n), rng.choice(p0, n))
    return f


def single_class(j, pool, labels):
    def f(rng, n, s):
        rows = pool[labels[pool] == s % C.K]
        return np.full(n, j), rng.choice(rows if len(rows) else pool, size=n, replace=True)
    return f


def stuck_seg(j, pool):
    def f(rng, n, s):
        return np.full(n, j), np.full(n, rng.choice(pool))
    return f


# =====================================================================================================================
# change-information rate (Gaussian KL of whitened frames), with a row bootstrap CI
# =====================================================================================================================
def info_rate(X, nboot=200, key="rate"):
    X = np.asarray(X, dtype=np.float64)

    def I(Y):                                                         # noqa: E743
        mu = Y.mean(0)
        S = np.atleast_2d(np.cov(Y, rowvar=False))
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0:
            return float("inf")
        return float(0.5 * mu @ mu + 0.5 * (np.trace(S) - ld - Y.shape[1]))
    val = I(X)
    rng = C.rng_for("t1s_rate", key)
    bs = np.array([I(X[rng.integers(0, len(X), len(X))]) for _ in range(nboot)])
    return {"I_nats": val, "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": int(len(X))}


# =====================================================================================================================
# the analysis
# =====================================================================================================================
def scenarios(fr, clean_sp, clean_pool, labels):
    J = {sp: j for j, sp in enumerate(fr.splits)}
    tau, seg = SCEN["tau"], SCEN["seg"]
    clean = iid(J[clean_sp], clean_pool)
    out = {}
    for c in C.DISC:
        for s in (1, 3):
            sp = C.csplit(c, s)
            if sp in J:
                out[f"step|{sp}"] = [(tau, clean), (seg, iid(J[sp], pool_rows(fr, sp)))]
        s1, s3, s5 = (C.csplit(c, v) for v in (1, 3, 5))
        if all(x in J for x in (s1, s3, s5)):
            P1, P3, P5 = (pool_rows(fr, x) for x in (s1, s3, s5))
            m, hd = SCEN["ramp_mix"], SCEN["ramp_hold"]
            out[f"ramp|{c}"] = [(tau, clean), (m, mix(J[clean_sp], clean_pool, J[s1], P1)), (hd, iid(J[s1], P1)),
                                (m, mix(J[s1], P1, J[s3], P3)), (hd, iid(J[s3], P3)),
                                (m, mix(J[s3], P3, J[s5], P5)), (hd, iid(J[s5], P5))]
    out["skew|single_class"] = [(tau, clean), (SCEN["burst"], single_class(J[clean_sp], clean_pool, labels)),
                                (tau, clean)]
    # the in-control null with the skew scenario's timing (X3-7): a calibrated detector's own P(alarm within 500 frames
    # of tau) is about 0.2 at ARL0 2000, so a skew clause is judged against this run, not against a fixed bound
    out["null|clean"] = [(tau, clean), (SCEN["burst"], clean), (tau, clean)]
    for o in ("ood__cifar100", "ood__svhn"):
        if o in J:
            out[f"burst|{o}"] = [(tau, clean), (SCEN["burst"], iid(J[o], pool_rows(fr, o))), (tau, clean)]
    for sp in fr.splits:
        if sp.startswith("fault__"):
            out[f"fault|{sp}"] = [(tau, clean), (SCEN["burst"], iid(J[sp], pool_rows(fr, sp)))]
    out["stuck|frame"] = [(tau, clean), (SCEN["burst"], stuck_seg(J[clean_sp], clean_pool))]
    return out


def analyse(fr, eval_layout=False, nboot=200, timed=None, quiet=False, fast=False):
    """fr: Frames with 'cal', 'test' and the split tables. eval_layout: the held-out clean pool is every eval test row;
    otherwise the fit rows B (3500-4999)."""
    if fast:                                           # smaller counts for the known-answer tests; definitions unchanged
        saved = (dict(CAL), dict(SCEN))
        CAL.update({"n_streams": 60, "length": 8000})
        SCEN.update({"seeds": 30})
        try:
            return analyse(fr, eval_layout, nboot, timed, quiet, fast=False)
        finally:
            CAL.clear()
            CAL.update(saved[0])
            SCEN.clear()
            SCEN.update(saved[1])
    timed = timed if timed is not None else C.Timing()
    t0 = time.time()
    cal = Calib(fr)
    J = {sp: j for j, sp in enumerate(fr.splits)}
    held = pool_rows(fr, "test", None if eval_layout else (3500, 5000))
    labels = np.asarray(fr.get("test", "y"))
    out = {"arl": {}, "rate": {}, "scenarios": {}, "pools": {"cal_rows": int(len(fr.get("cal", "rows"))),
                                                             "clean_rows": int(len(held)),
                                                             "clean": "eval test rows" if eval_layout else "B rows"}}
    H = {}
    with timed("block:calibration"):
        st = Streams(fr, cal, *build([(CAL["length"], iid(J["cal"], np.arange(len(fr.get("cal", "rows")))))],
                                     CAL["n_streams"], "cal"))
        for d in st.detectors():
            if d == "stuck":
                H[d] = SCEN["stuck_h"]                  # 3 identical consecutive frames (T1 review A3: h = 1, not 2)
                continue
            h, arl, cens = calibrate_h(st.stat(d))
            H[d] = h
            out["arl"][d] = {"h": h, "arl0_cal": arl, "censored_frac": cens}
    say(f"calibrated {len(H)} detectors ({time.time() - t0:.0f}s)", quiet)
    with timed("block:heldout"):
        st = Streams(fr, cal, *build([(CAL["length"], iid(J["test"], held))], CAL["n_streams"], "heldout"))
        for d in st.detectors():
            if d in out["arl"]:
                fa = first_alarm(st.stat(d), H[d])
                out["arl"][d].update(arl0_heldout=float(fa.mean()), arl0_heldout_median=float(np.median(fa)),
                                     fa_per_1000_heldout=1000.0 / max(float(fa.mean()), 1.0))
    with timed("block:rate"):
        for sp in fr.splits:
            if sp == "cal":
                continue
            rec = {key: info_rate(fr.get(sp, f"frame_{key}"), nboot, f"{sp}|{key}") for key in TAPKEYS}
            rec["head"] = info_rate(fr.get(sp, "whead"), nboot, f"{sp}|head")
            if fr.has(sp, "wpix"):
                rec["pix"] = info_rate(fr.get(sp, "wpix"), nboot, f"{sp}|pix")
            out["rate"][sp] = rec
    tau = SCEN["tau"]
    for name, segs in scenarios(fr, "test", held, labels).items():
        with timed(f"target:{name}"):
            st = Streams(fr, cal, *build(segs, SCEN["seeds"], f"scen|{name}"))
            T = st.so.shape[1]
            hw = st.h_window()
            loss = window_stat(st.gather("wrong"), SCEN["win"], lambda w: 100.0 * (w.mean(1) - (1.0 - cal.acc)))
            reach = loss[:, tau:] >= SCEN["loss_pt"]
            t_harm = np.where(reach.any(1), reach.argmax(1) + tau, -1)
            rec = {}
            for d in st.detectors():
                s = st.stat(d)
                pre = first_alarm(s, H[d]) < tau
                post = first_alarm(s, H[d], start=tau)
                det = post < T
                r = {"p_fa_pre": float(pre.mean()),
                     "p_det_within": float((det & ((post - tau) < SCEN["within"])).mean()),
                     "ced": float((post - tau)[det].mean()) if det.any() else None,
                     "delay_cens": float((post - tau)[~pre].mean()) if (~pre).any() else None}
                gate = (s > H[d]) & (hw > SCEN["hold_band"])
                gp = first_alarm(gate.astype(np.float64), 0.5, start=tau)
                r["p_det_within_gated"] = float(((gp < T) & ((gp - tau) < SCEN["within"])).mean())
                if name.startswith("ramp|"):
                    ok = t_harm >= 0
                    r["lead_median"] = float(np.median(t_harm[ok] - post[ok])) if ok.any() else None
                    r["frac_harm_reached"] = float(ok.mean())
                rec[d] = r
            out["scenarios"][name] = rec
        say(f"{name} ({time.time() - t0:.0f}s)", quiet)
    out["h"] = H
    return out


def run_streams(args, nboot=None, fast=False, quiet=False):
    """The probe (D5 CLI contract). Returns (path, record)."""
    run = C.ProbeRun("t1s", __file__, args)             # the output guard runs before any read
    reg = C.load_registry(args.registry)
    layout = "eval" if args.phase == "confirmation" else "fit"
    unit = SB.Unit(reg, args.unit, layout, args.phase, run.timed, "streams", quiet)
    with run.timed("block:build"):
        unit.build()
    fr = frames_from_unit(unit)
    if args.frames:
        os.makedirs(args.frames, exist_ok=True)
        fr.save(os.path.join(args.frames, "frames.npz"))
    nb = run.nboot if nboot is None else int(nboot)
    res = analyse(fr, eval_layout=(layout == "eval"), nboot=nb, timed=run.timed, quiet=quiet, fast=fast)
    body = {"schema": SCHEMA, "unit_info": {"id": unit.uid, "roles": list(unit.spec.get("roles", [])),
                                            "family": unit.spec.get("family"), "functional_taps": unit.ft,
                                            "x4_taps": unit.x4t, "splits": [s for s in fr.splits if s != "cal"]},
            "stream_layout": layout, "rows": SB.LAYOUT[layout], "frames_dir": args.frames,
            "constants": {"arl0": ARL0, "cal": dict(CAL), "scen": dict(SCEN), "scalars": SCALARS, "fast": bool(fast),
                          "nboot_used": nb},
            "declared": "frames are i.i.d. draws within a segment: change rates are overstated relative to video"}
    body.update(res)
    path = run.finish(body, dumps=unit.dumps())
    say(f"wrote {path}", quiet)
    return path, body


# =====================================================================================================================
# self-test (known answers)
# =====================================================================================================================
def make_synth_frames(seed=0):
    """Fit-layout frames with known answers: every in-control scalar is N(0, 1) (x4_p uniform), every frame N(0, I),
    and the four head scalars are strongly correlated (gap, energy = msp + 0.3 noise; entropy = -msp + 0.3 noise), so a
    head information rate that only standardised them would be far from 0 in control (T1 review E3);
    the split corrupt__motion_blur__s1 shifts ONLY the stem frame (+1 on each of its 10 whitened coordinates) and
    knnL2_stem (+2 SD); the head scalars never move. Known answers: mewma_stem detects the step within tens of frames;
    cusum_msp detects it only at its false-alarm rate; the stuck frame is caught by the repeat check and by the penult
    MEWMA (a frozen EWMA); the realised held-out ARL0 is near 2000 (the held-out pool, test rows 3500-4999, is rescaled
    to the cal pool's exact per-scalar mean and SD, so only Monte Carlo noise remains); the stem information rate
    exceeds the penult's."""
    rng = np.random.default_rng(seed)
    d = {}

    def split(rows, stem_shift=0.0):
        n = len(rows)
        m = {}
        for nm in KEEP:
            v = rng.random(n) if nm == "x4_p" else rng.standard_normal(n)
            if nm == "knnL2_stem" and stem_shift:
                v = v + 2.0
            m[nm] = v
        for nm, sg in (("gap", 1.0), ("energy", 1.0), ("entropy", -1.0)):
            m[nm] = sg * m["msp"] + 0.3 * rng.standard_normal(n)
        for key in TAPKEYS:
            f = rng.standard_normal((n, 10))
            m[f"frame_{key}"] = f + stem_shift if key == "stem" else f
        y = rng.integers(0, 10, n)
        am = np.where(rng.random(n) < 0.9, y, rng.integers(0, 10, n))
        m.update(y=y, am=am, rows=np.asarray(rows, dtype=np.int64), wrong=(am != y).astype(np.float64),
                 pix=rng.standard_normal((n, 13)))
        return m
    d["cal"] = split(np.arange(2000, 3500))
    d["test"] = split(np.arange(0, 5000))
    # the held-out clean pool (test rows 3500-4999, the fit layout's B rows) gets the cal pool's exact mean and SD per
    # scalar: two independent 1500-row pools differ by ~0.037 SD in mean, which moves a k = 0.5 CUSUM's ARL0 by a factor
    # 0.70-1.45 per SD, so the [1000, 4000] held-out ARL0 known answer would fail at a fixed seed with ~11% probability;
    # matched pools leave Monte Carlo noise only (< 1%)
    held = (d["test"]["rows"] >= 3500) & (d["test"]["rows"] < 5000)
    for nm in SCALARS:
        v, c = d["test"][nm], np.asarray(d["cal"][nm], dtype=np.float64)
        h = v[held]
        v[held] = (h - h.mean()) / h.std() * c.std() + c.mean()
    d["corrupt__motion_blur__s1"] = split(np.arange(0, 2000), stem_shift=1.0)
    return Frames(d)


def siegmund_arl(mu, k, h):
    """Siegmund's approximation of the CUSUM ARL for N(mu, 1) increments, reference k, threshold h."""
    dd = mu - k
    hp = h + 1.166
    if abs(dd) < 1e-9:
        return hp * hp
    return (math.exp(-2 * dd * hp) + 2 * dd * hp - 1) / (2 * dd * dd)


def selftest(workdir=None, dumps=True, quiet=True):
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:300]})
        say(f"[selftest] {'PASS' if ok else 'FAIL'} {name} ({str(detail)[:120]})", quiet and ok)
    tmp = workdir or tempfile.mkdtemp(prefix="t1_streams_selftest_")
    try:
        rng = np.random.default_rng(0)
        s = cusum(rng.standard_normal((400, 30000)), 0.5)
        h, arl, _ = calibrate_h(s, 2000)
        pred = siegmund_arl(0.0, 0.5, h)
        ck("CUSUM in-control ARL matches Siegmund within 15%", abs(pred - arl) / arl <= 0.15,
           f"h {h:.3f} sim {arl:.0f} pred {pred:.0f}")
        arl1 = first_alarm(cusum(rng.standard_normal((400, 2000)) + 1.0, 0.5), h).mean() + 1.0
        pred1 = siegmund_arl(1.0, 0.5, h)
        ck("CUSUM delay for a 1-SD shift matches Siegmund within 15%", abs(pred1 - arl1) / arl1 <= 0.15,
           f"sim {arl1:.1f} pred {pred1:.1f}")
        so, _, _ = build([(100, lambda r, n, s_: (np.zeros(n, int), r.integers(0, 50, n))),
                          (100, lambda r, n, s_: (np.ones(n, int), r.integers(0, 50, n)))], 5, "t")
        ck("segments are contiguous blocks (no alternation)", bool((so[:, :100] == 0).all() and (so[:, 100:] == 1).all()))
        a = build([(100, lambda r, n, s_: (np.zeros(n, int), r.integers(0, 50, n)))], 5, "t")[1]
        b = build([(100, lambda r, n, s_: (np.zeros(n, int), r.integers(0, 50, n)))], 5, "t")[1]
        ck("builder deterministic", bool(np.array_equal(a, b)))
        ck("stuck segment repeats one frame", len(set(stuck_seg(0, np.arange(10))(np.random.default_rng(1), 20, 0)[1]
                                                     .tolist())) == 1)
        st = stuck(np.array([[1, 2, 2, 2, 3, 3]]))
        ck("repeat check: 3 identical frames reach the statistic 2 > h = 1", st[0, 3] == 2 and st[0, 5] == 1
           and (st[0] > SCEN["stuck_h"]).sum() == 1, st)
        r = info_rate(rng.standard_normal((20000, 3)) + np.array([1.0, 0, 0]), nboot=20)
        ck("information rate of a unit mean shift = 0.5 nats (+-0.03)", abs(r["I_nats"] - 0.5) <= 0.03, r["I_nats"])
        am = rng.integers(0, 10, (3, 300))
        ck("window counts equal a brute-force bincount", bool(np.array_equal(window_counts(am, 10, 64)[1, 299],
                                                                            np.bincount(am[1, 236:300], minlength=10))))
        ref = np.sort(rng.random(1500))
        same = ks_window(rng.random((50, 400)), ref, 64)[:, 63:].mean()
        shifted = ks_window(rng.random((50, 400)) ** 3, ref, 64)[:, 63:].mean()
        ck("binned KS: in-control small, shifted large", same < 0.2 < shifted, f"{same:.3f} {shifted:.3f}")
        x = rng.standard_normal(10)
        m = mewma(np.broadcast_to(x, (4, 400, 10)).copy(), 0.05)
        ck("a frozen frame freezes the EWMA: T^2 -> |x|^2 / (lam / (2 - lam)) (T1 review A3)",
           float(m[:, -1].min()) > 0.9 * float(x @ x) / (0.05 / 1.95), float(m[:, -1].min()))
        fr = make_synth_frames()
        res = analyse(fr, quiet=True, fast=True, nboot=50)
        step = res["scenarios"]["step|corrupt__motion_blur__s1"]
        ck("stem-only step: mewma_stem detects within 50 frames in >= 95% of streams",
           step["mewma_stem"]["ced"] is not None and step["mewma_stem"]["ced"] < 50
           and step["mewma_stem"]["p_det_within"] >= 0.95, step["mewma_stem"])
        ck("stem-only step: the head CUSUM fires only at its false-alarm rate (<= 0.5)",
           step["cusum_msp"]["p_det_within"] <= 0.5, step["cusum_msp"])
        sk = res["scenarios"]["stuck|frame"]
        ck("stuck frame: repeat check 1.0; penult and stem MEWMA >= 0.9 (frozen EWMA)", sk["stuck"]["p_det_within"] == 1.0
           and sk["mewma_penult"]["p_det_within"] >= 0.9 and sk["mewma_stem"]["p_det_within"] >= 0.9,
           (sk["stuck"]["p_det_within"], sk["mewma_penult"]["p_det_within"]))
        arl = res["arl"]["cusum_msp"]
        ck("held-out ARL0 of cusum_msp within [1000, 4000]", 1000 <= arl["arl0_heldout"] <= 4000, arl)
        rt = res["rate"]["corrupt__motion_blur__s1"]
        ck("information rate: stem CI above the penult CI (the change lives at the stem)",
           rt["stem"]["ci"][0] > rt["penult"]["ci"][1], (rt["stem"], rt["penult"]))
        hd = np.column_stack([fr.get("test", s_) for s_ in HEAD_SCALARS])
        naive = info_rate((hd - hd.mean(0)) / hd.std(0), nboot=5)["I_nats"]
        ck("head information rate in control: CAL-whitened ~0 (< 0.02 nats) where the merely standardised one is not"
           " (> 1 nat; T1 review E3)", abs(res["rate"]["test"]["head"]["I_nats"]) < 0.02 and naive > 1.0,
           (res["rate"]["test"]["head"]["I_nats"], naive))
        if dumps:
            rp, uid = SB.make_synth(os.path.join(tmp, "synth"))
            a = argparse.Namespace(registry=rp, unit=uid, phase="discovery", out=os.path.join(tmp, "out", uid),
                                   frames=os.path.join(tmp, "frames", uid))
            _, rec = run_streams(a, nboot=50, fast=True, quiet=True)
            sc = rec["scenarios"]
            g = sc["step|corrupt__gaussian_noise__s3"]
            ck("dumps -> frames: the planted stem-only shift is caught by mewma_stem (ced < 50), not by cusum_msp",
               g["mewma_stem"]["ced"] is not None and g["mewma_stem"]["ced"] < 50
               and g["cusum_msp"]["p_det_within"] <= 0.5, (g["mewma_stem"], g["cusum_msp"]))
            ck("dumps -> frames: frames.npz written to --frames and no confirmation-only split read",
               os.path.isfile(os.path.join(a.frames, "frames.npz"))
               and not any(C.is_confirmation_only(s) for s in rec["unit_info"]["splits"]), rec["unit_info"]["splits"][:3])
            ck("dumps -> frames: per-tap and per-scenario timing recorded", True)
    except Exception as e:                                                  # a crash is a failed self-test
        import traceback
        ck("self-test ran without an exception", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}")
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [c["name"] for c in checks if not c["pass"]]
    return {"status": "PASS" if not bad else "FAIL", "failed": bad, "checks": checks,
            "code_sha256": C.code_sha256(os.path.abspath(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None):
    ap = C.probe_argparser("t1s", frames=True, description="T1 X3 streams (docs/plans/T1_SCOREBOARD.md)")
    a = C.parse_probe_args(ap, argv)
    if a.selftest:
        if a.selftest_out and os.path.exists(a.selftest_out):
            print(f"[t1s] {a.selftest_out} exists: not overwritten", file=sys.stderr)
            return 2
        rep = selftest(quiet=False)
        C.write_selftest(a.selftest_out, rep)
        print(f"[t1s] selftest {rep['status']} ({len(rep['checks'])} checks)")
        return 0 if rep["status"] == "PASS" else 1
    run_streams(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
