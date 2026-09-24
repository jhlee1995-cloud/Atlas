"""
atlas/b4_core.py -- batch-4 shared numerics, dump access (the D7 seal) and the probe contract (docs/plans/
B4_INTEGRATION.md D5, D7, D10, D14). Owner: T1. Other tracks change it only through T1.

numpy + the standard library ONLY: tests/test_b4_core.py asserts that `import atlas.b4_core` imports neither torch nor
scipy nor sklearn, so every probe runs on a CPU pod without a GPU stack. T3's sklearn crossfit is not used anywhere.

INTERFACE (stable at P1; the three tracks code against these signatures)
-- dump access (D7). A probe opens a dump ONLY through open_dump / open_unit ------------------------------------------
  open_dump(path, phase) -> Dump          phase 'discovery' | 'confirmation'. A SEALED dump (results/b4c_* or a dump
                                          holding SEALED.json) opens only in 'confirmation' with ATLAS_B4_UNSEAL=<P2>, a
                                          commit that is an ancestor of HEAD and holds experiments/b4/freeze_P2.json; each
                                          unseal is appended to $ATLAS_B4_CHECK_DIR/unseal_log.jsonl. In 'discovery' every
                                          split named in CONFIRMATION_ONLY is refused. A dump without meta.json (partial)
                                          is refused. Refusals raise ReadRefused (a SystemExit: `except Exception` in a
                                          probe cannot swallow it).
  open_unit(reg, unit, layout, phase) -> Dump        the registry's dump for (unit, layout)
  Dump.meta / .b4 / .taps / .splits / .unit / .layout / .sealed / .phase / .path / .splits_read
  Dump.readable(split) -> bool;  Dump.has(tap, split) -> bool (False for a refused split)
  Dump.acts(tap, split, dtype=np.float64) -> (n, d)       GAP activations
  Dump.logits(split) -> (n, K) float64    the STORED float32 model logits: the canonical head path of every track (D2)
  Dump.labels(split) -> (n,) int64;  Dump.rows(split) -> (n,) int64 global dataset rows;  Dump.preds(split) -> dict
  Dump.factors(split) -> (X (n, m), names) | None;  Dump.maps(tap, split) -> (n, C, H, W);  Dump.masks(split) -> bool
  Dump.pixels(split) -> (n, 32, 32, 3) uint8 (maps layout only);  Dump.head() -> (W (K, d), b (K,)) float64
  Dump.functional_taps() -> {stem, s1end, s2end, pre, penult};  Dump.x4_taps() -> [first, pre, penult]
  Dump.list_splits(prefix='') -> readable splits in extraction order
-- registry and probe contract (D5 CLI: --registry --unit --phase --out [--layout] [--frames] --selftest --selftest-out)
  load_registry(path) -> dict;  unit_spec(reg, unit) -> dict;  layout_dump(reg, unit, layout) -> str
  probe_argparser(program, layout=False, frames=False) -> argparse.ArgumentParser;  parse_probe_args(ap, argv) -> ns
  ProbeRun(program, script_path, args)      guards the output (append-only; confirmation: touched once under any tag),
      .timed(key)                         context manager adding wall seconds to timing_s[key] ('tap:<t>', 'target:<n>')
      .finish(body, dumps=()) -> path     adds program/unit/phase/layout, code.sha256 + code.core_sha256 + repo_commit,
                                          env, timing_s (+ total), max_rss_mb, splits_read; writes OUTPUT_FILE[program]
  guard_output(out, phase, fname);  write_json(path, obj);  write_selftest(path, report);  _clean(obj)
  code_sha256(path) (CRLF-normalised);  repo_commit();  max_rss_mb();  Timing
-- numerics (float64 throughout) ------------------------------------------------------------------------------------
  rank_avg(x); auc(pos, neg); auc_y(y, s); delong(y, s) -> (auc, se); delong_diff(y, s1, s2) -> (d, se, p); AucBoot
  spearman(x, y, min_n=3); pearson(x, y); partial_spearman(x, y, z); wilson(k, n); holm(pvals); finite_fix(x)
  softmax(Z, T=1); head_stats(Z, T=1) -> {msp, maxlogit, gap, energy (= logsumexp), entropy, gini, smargin, pnorm2,
      mspT, argmax}; HEAD_STATS; ERR_SIGN (larger oriented score = more error-like); sorted_logits(Z, m=10)
  fit_temperature(Z, y, grid=(0.05, 20.0)); nll(Z, y, T=1); ece(msp, correct, bins=15)
  rcs(x, knots); design(blocks, tr, te); logit_fit(X, y, lam=RIDGE); logit_predict(beta, X); folds_for(groups, key)
  crossfit(blocks, y, folds, train_ok=None, lam=RIDGE); fit_apply(blocks, y, tr, te); ce_bits(y, p)
  boot_ci(y, groups, pa, pb, nboot, key) -> (CI of AUC(pa) - AUC(pb), CI of CE(pb) - CE(pa) bits)
  head_additive_call(dauc, dauc_ci, dI_ci) -> 'ADDS' | 'BOUNDED' | 'INCONCLUSIVE'   (the frozen HEAD-ADDITIVE rule)
  head_increment(head_blocks, bundle_blocks, y, groups, nboot, key, ...) -> record with dauc, CIs, DeLong, call
  risk_coverage(y_err, conf); fpr_at_tpr(y, s, tpr=0.95); tpr_at_fpr(y, s, fpr=0.05)
  conformal_p(cal, s); cal_threshold(cal, alpha); cauchy_stat(P)
  sqdist(Q, F); knn(fit, query, k, exclude_self=False) -> (dist, idx); knn_kth(fit, query, k, exclude_self=False);
  l2n(X); class_means(X, y, K); eig_desc(C, r=None); pca_frame(X, r); whiten(X, frame); rng_for(*parts, seed=SEED)
  csplit(c, s); fsplit(kind, level); parse_split(split)

HEAD-ADDITIVE rule (frozen, T1 plan section 7, T1 review C1): a bundle ADDS to the head when dAUC_joint >= +0.01 AND the
95% group-bootstrap CI of dI (bits; CE(head) - CE(head + bundle)) lies above 0; it is BOUNDED (the functional "no") when
the upper 95% bootstrap bound of dAUC_joint is below EQ_BOUND = +0.02; otherwise INCONCLUSIVE (power).
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zlib

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_SCHEMA = "b4_core/1"
SEED = 20260924
K = 10
EIG_FLOOR = 1e-8

DISC = ("gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast",
        "pixelate", "jpeg_compression")
HOLDOUT = ("impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform")
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
SEVS = (1, 3, 5)
FAMILY = {"gaussian_noise": "N", "shot_noise": "N", "impulse_noise": "N", "speckle_noise": "N",
          "defocus_blur": "B", "motion_blur": "B", "glass_blur": "B", "zoom_blur": "B", "gaussian_blur": "B",
          "snow": "W", "fog": "W", "frost": "W", "spatter": "W",
          "brightness": "D", "contrast": "D", "pixelate": "D", "jpeg_compression": "D", "elastic_transform": "D",
          "saturate": "D"}
# D7, verbatim: every split whose name contains one of these is confirmation-only, whatever the unit (D4)
CONFIRMATION_ONLY = ("impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform",   # holdout families
                     "speckle_noise", "gaussian_blur", "spatter", "saturate",                    # CIFAR-10-C extras
                     "exposure_global", "soiling")                                               # holdout faults
PHASES = ("discovery", "confirmation")
LAYOUTS = ("fit", "eval", "maps")
PROGRAMS = ("t1", "t1s", "t2", "t3s")
OUTPUT_FILE = {"t1": "scoreboard.json", "t1s": "streams.json", "t2": "probe.json", "t3s": "probe.json"}
TAG_RE = re.compile(r"(?:_r\d+|_p2|_s2replay)*$")        # the only output tags (relaunch, P2 re-probe, S2 replay)
NBOOT = {"discovery": 200, "confirmation": 1000}         # D10: B = 200 in discovery and in the replay, 1000 confirmation

# ridge-logistic joint model (frozen family, T1 plan section 5)
KNOTS_Q = (0.05, 0.35, 0.65, 0.95)
RIDGE = 1e-3
N_FOLDS = 5
ADD_DAUC = 0.01              # HEAD-ADDITIVE: dAUC_joint >= +0.01 ...
EQ_BOUND = 0.02              # ... BOUNDED: upper 95% bootstrap bound of dAUC_joint < +0.02 (T1 review C1)
ECE_BINS = 15
TEMP_GRID = (0.05, 20.0)


class ReadRefused(SystemExit):
    """A read the protocol forbids (sealed dump, confirmation-only split in discovery, partial dump). A SystemExit so
    that a probe's per-target `except Exception` cannot turn it into a recorded error and carry on."""


# =====================================================================================================================
# dump access and the seal (D7)
# =====================================================================================================================
def _git(*a):
    return subprocess.run(["git", "-C", REPO_ROOT, *a], capture_output=True, text=True)


def _abs(path):
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path))


def _rel(path):
    try:
        return os.path.relpath(_abs(path), REPO_ROOT).replace(os.sep, "/")
    except ValueError:                                   # another drive (Windows): not under the repo
        return _abs(path).replace(os.sep, "/")


def is_confirmation_only(split):
    return any(c in split for c in CONFIRMATION_ONLY)


def is_sealed_path(path):
    return _rel(path).startswith("results/b4c_") or os.path.exists(os.path.join(_abs(path), "SEALED.json"))


def open_dump(path, phase):
    """The only way a batch-4 probe opens a dump (D7)."""
    rel = _rel(path)
    if phase not in PHASES:
        raise ReadRefused(f"[b4] unknown phase {phase}")
    sealed = is_sealed_path(path)
    if sealed:
        p2 = os.environ.get("ATLAS_B4_UNSEAL", "")
        if phase != "confirmation" or not p2:
            raise ReadRefused(f"[b4] {rel} is SEALED: opened only in the confirmation phase, after P2")
        if (_git("merge-base", "--is-ancestor", p2, "HEAD").returncode
                or _git("cat-file", "-e", f"{p2}:experiments/b4/freeze_P2.json").returncode):
            raise ReadRefused(f"[b4] {rel}: ATLAS_B4_UNSEAL={p2} is not a P2 freeze at or before HEAD")
        d = os.environ.get("ATLAS_B4_CHECK_DIR", ".")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "unseal_log.jsonl"), "a") as f:
            f.write(json.dumps({"dump": rel, "p2": p2, "head": _git("rev-parse", "HEAD").stdout.strip(),
                                "unix_time": time.time()}) + "\n")
    if phase == "confirmation":
        allow = (lambda s: True)                                               # noqa: E731
    else:
        allow = (lambda s: not is_confirmation_only(s))                        # noqa: E731
    return Dump(_abs(path), allow=allow, phase=phase, sealed=sealed)


def load_registry(path):
    with open(_abs(path)) as f:
        reg = json.load(f)
    reg["_path"] = path
    return reg


def unit_spec(reg, unit):
    for m in reg["models"]:
        if m["id"] == unit:
            return m
    raise SystemExit(f"[b4] unit {unit} is not in the registry {reg.get('_path')}")


def layout_dump(reg, unit, layout):
    m = unit_spec(reg, unit)
    if layout not in m["layouts"]:
        raise SystemExit(f"[b4] unit {unit} has no {layout} layout (registry layouts: {sorted(m['layouts'])})")
    return m["layouts"][layout]["dump"]


def open_unit(reg, unit, layout, phase):
    return open_dump(layout_dump(reg, unit, layout), phase)


class Dump:
    """Read access to one batch-4 dump (scripts/b4_extract.py layout; old atlas dumps read the same way without
    logits/rows/head). Every data read goes through _check(split): a split the phase may not read raises."""

    def __init__(self, path, allow, phase="discovery", sealed=False):
        self.path, self.allow, self.phase, self.sealed = path, allow, phase, sealed
        mp = os.path.join(path, "meta.json")
        if not os.path.isfile(mp):
            raise ReadRefused(f"[b4] {path}: no meta.json (a partial dump is never read)")
        with open(mp) as f:
            self.meta = json.load(f)
        self.b4 = self.meta.get("b4") or {}
        self.taps = list(self.meta.get("layers", []))
        self.splits = list(self.meta.get("splits", []))
        self.unit = self.b4.get("unit") or self.meta.get("seed_tag")
        self.layout = self.b4.get("layout")
        self.K = int(self.meta.get("n_classes", K))
        self._read = set()

    # -- access rules --
    def readable(self, split):
        return bool(self.allow(split))

    def _check(self, split):
        if not self.allow(split):
            raise ReadRefused(f"[b4] {self.path}: split {split} is not readable in phase {self.phase}")
        self._read.add(split)

    @property
    def splits_read(self):
        return sorted(self._read)

    def list_splits(self, prefix=""):
        return [s for s in self.splits if s.startswith(prefix) and self.readable(s)]

    def _p(self, *parts):
        return os.path.join(self.path, *parts)

    def has(self, tap, split):
        return self.readable(split) and os.path.isfile(self._p("acts", tap, f"{split}.npy"))

    # -- arrays --
    def acts(self, tap, split, dtype=np.float64):
        self._check(split)
        return np.load(self._p("acts", tap, f"{split}.npy")).astype(dtype)

    def logits(self, split):
        self._check(split)
        p = self._p("logits", f"{split}.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[b4] {self.path}: no stored logits for {split}")
        return np.load(p).astype(np.float64)

    def labels(self, split):
        self._check(split)
        return np.load(self._p("labels", f"{split}.npy")).astype(np.int64)

    def rows(self, split):
        self._check(split)
        p = self._p("rows", f"{split}.npy")
        if os.path.isfile(p):
            return np.load(p).astype(np.int64)
        r = (self.b4.get("rows") or {}).get(split)
        if isinstance(r, list) and len(r) == 2:
            return np.arange(int(r[0]), int(r[1]), dtype=np.int64)
        if split == "test":                                    # old atlas dumps: test rows 0 .. n_test-1
            return np.arange(int(self.meta.get("n_test", len(self.labels(split)))), dtype=np.int64)
        raise FileNotFoundError(f"[b4] {self.path}: no rows for {split}")

    def preds(self, split):
        self._check(split)
        with np.load(self._p("preds", f"{split}.npz")) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    def factors(self, split):
        self._check(split)
        p = self._p("factors", f"{split}.npz")
        if not os.path.isfile(p):
            return None
        with np.load(p) as z:
            names = sorted(z.files)
            return np.column_stack([np.asarray(z[k], dtype=np.float64) for k in names]), names

    def maps(self, tap, split, dtype=np.float32):
        self._check(split)
        return np.load(self._p("maps", tap, f"{split}.npy")).astype(dtype)

    def masks(self, split):
        self._check(split)
        return np.load(self._p("masks", f"{split}.npy")).astype(bool)

    def pixels(self, split):
        self._check(split)
        return np.load(self._p("pixels", f"{split}.npy"))

    def head(self):
        p = self._p("head", "head.npz")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[b4] {self.path}: no head/head.npz")
        with np.load(p) as z:
            return np.asarray(z["W"], dtype=np.float64), np.asarray(z["b"], dtype=np.float64)

    # -- tap metadata written by scripts/b4_extract.py (D2) --
    def functional_taps(self):
        f = (self.b4.get("taps") or {}).get("functional")
        if not f:
            raise KeyError(f"[b4] {self.path}: meta.b4.taps.functional missing (not a batch-4 dump)")
        return dict(f)

    def x4_taps(self):
        return list((self.b4.get("taps") or {}).get("x4") or [])

    def meta_sha256(self):
        with open(self._p("meta.json"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


# =====================================================================================================================
# probe contract: output guard, timing, provenance (D5, D7, D10)
# =====================================================================================================================
def strip_tag(name):
    return TAG_RE.sub("", name)


def guard_output(out, phase, fname):
    """Append-only: refuse a non-empty output dir. Confirmation phase (the touched-once rule, pod_atlas.sh:830-833): refuse
    when an output file for the same (program, unit, layout) exists under ANY tag (sibling dirs whose name equals this
    one's with the tag stripped)."""
    out = _abs(out)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise ReadRefused(f"[b4] {out} exists and is not empty: results are append-only (relaunch with a new tag)")
    name = os.path.basename(os.path.normpath(out))
    if phase == "confirmation":
        parent, base = os.path.dirname(out), strip_tag(name)
        if os.path.isdir(parent):
            for e in sorted(os.listdir(parent)):
                if e != name and strip_tag(e) == base and os.path.isfile(os.path.join(parent, e, fname)):
                    raise ReadRefused(f"[b4] {parent}/{e}/{fname} exists: a confirmation unit is probed once")
    return out


def _clean(o):
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None (T2 review #19); dict keys to str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def write_json(path, obj, sort_keys=True):
    """Atomic write (tmp + replace), allow_nan=False after _clean."""
    path = _abs(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, sort_keys=sort_keys, allow_nan=False)
    os.replace(tmp, path)
    return path


def write_selftest(path, report):
    """--selftest-out: never overwrites (a committed record)."""
    if path and os.path.exists(path):
        raise ReadRefused(f"[b4] {path} exists: a self-test record is never overwritten")
    if path:
        write_json(path, report)


def code_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def repo_commit(repo=None):
    """HEAD commit read from .git without a git binary (worktree .git files and packed refs handled); None if unknown."""
    repo = repo or REPO_ROOT
    g = os.path.join(repo, ".git")
    try:
        if os.path.isfile(g):
            with open(g) as f:
                g = os.path.join(repo, f.read().split("gitdir:", 1)[1].strip())
        with open(os.path.join(g, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if os.path.isfile(os.path.join(g, ref)):
            with open(os.path.join(g, ref)) as f:
                return f.read().strip()
        with open(os.path.join(g, "packed-refs")) as f:
            for line in f:
                if line.strip().endswith(" " + ref):
                    return line.split()[0]
    except (OSError, IndexError):
        pass
    return None


def max_rss_mb():
    try:
        import resource
    except ImportError:                                   # Windows: no resource module
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1048576.0 if sys.platform == "darwin" else 1024.0), 1)


class Timing:
    """timing_s: wall seconds per key ('tap:<t>', 'target:<n>', 'block:<b>'); as_dict() adds 'total'."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.s = {}

    @contextlib.contextmanager
    def __call__(self, key):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.s[key] = round(self.s.get(key, 0.0) + time.perf_counter() - t, 3)

    def as_dict(self):
        d = dict(self.s)
        d["total"] = round(time.perf_counter() - self.t0, 3)
        return d


def probe_argparser(program, layout=False, frames=False, description=None):
    if program not in PROGRAMS:
        raise ValueError(program)
    ap = argparse.ArgumentParser(description=description or f"batch-4 probe {program} (docs/plans/B4_INTEGRATION.md D5)")
    ap.add_argument("--registry", help="experiments/b4/models.json")
    ap.add_argument("--unit", help="registry id")
    ap.add_argument("--phase", choices=PHASES)
    ap.add_argument("--out", help=f"output directory (receives {OUTPUT_FILE[program]})")
    if layout:
        ap.add_argument("--layout", choices=("fit", "eval"), default="fit")
    if frames:
        ap.add_argument("--frames", help="scratch directory for stream frames (/root/b4_frames/<id>; container disk)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-out")
    return ap


def parse_probe_args(ap, argv=None):
    a = ap.parse_args(argv)
    if not a.selftest:
        miss = [k for k in ("registry", "unit", "phase", "out") if getattr(a, k, None) in (None, "")]
        if miss:
            ap.error("required unless --selftest: " + ", ".join("--" + m for m in miss))
    return a


class ProbeRun:
    """One probe invocation: output guard at start, provenance + timing + memory at finish."""

    def __init__(self, program, script_path, args):
        if program not in PROGRAMS:
            raise ValueError(program)
        self.program, self.script, self.args = program, os.path.abspath(script_path), args
        self.fname = OUTPUT_FILE[program]
        self.out = guard_output(args.out, args.phase, self.fname)
        self.timed = Timing()
        self.nboot = NBOOT[args.phase]
        self.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def header(self):
        return {"program": self.program, "unit": self.args.unit, "phase": self.args.phase,
                "layout": getattr(self.args, "layout", None), "out": _rel(self.out), "tag": self.tag(),
                "code": {"path": _rel(self.script), "sha256": code_sha256(self.script),
                         "core_sha256": code_sha256(os.path.abspath(__file__)), "repo_commit": repo_commit()},
                "env": {"python": platform.python_version(), "numpy": np.__version__, "host": platform.node(),
                        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                     "MKL_NUM_THREADS")},
                        "unseal": os.environ.get("ATLAS_B4_UNSEAL") or None},
                "nboot": self.nboot, "created_utc": self.created, "core_schema": CORE_SCHEMA}

    def tag(self):
        name = os.path.basename(os.path.normpath(self.out))
        return name[len(strip_tag(name)):]

    def finish(self, body, dumps=()):
        rec = dict(body)
        rec.update(self.header())
        rec["dumps"] = [{"path": _rel(d.path), "meta_sha256": d.meta_sha256(), "sealed": d.sealed,
                         "splits_read": d.splits_read} for d in dumps]
        rec["timing_s"] = self.timed.as_dict()
        rec["max_rss_mb"] = max_rss_mb()
        return write_json(os.path.join(self.out, self.fname), rec)


# =====================================================================================================================
# ranks, AUC, correlations
# =====================================================================================================================
def rank_avg(x):
    """1-based ranks with ties averaged."""
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.repeat(first + (counts - 1) / 2.0 + 1.0, counts)
    return r


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie); None if a side is empty or non-finite. Larger score = more positive-like."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if len(pos) == 0 or len(neg) == 0 or not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        return None
    r = rank_avg(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_y(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    return auc(s[y], s[~y])


def _delong_parts(pos, neg):
    m, n = len(pos), len(neg)
    tz = rank_avg(np.concatenate([pos, neg]))
    tx, ty = rank_avg(pos), rank_avg(neg)
    a = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    return a, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m


def delong(y, s):
    """AUC and its DeLong standard error."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return None, None
    a, v10, v01 = _delong_parts(s[y], s[~y])
    return float(a), float(math.sqrt(max(v10.var(ddof=1) / m + v01.var(ddof=1) / n, 0.0)))


def delong_diff(y, s1, s2):
    """Paired DeLong test of AUC(s1) - AUC(s2) on the same samples. Returns (diff, se, p two-sided)."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return None, None, None
    s1, s2 = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    a1, v10a, v01a = _delong_parts(s1[y], s1[~y])
    a2, v10b, v01b = _delong_parts(s2[y], s2[~y])
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    se = math.sqrt(max(var, 0.0))
    d = a1 - a2
    p = math.erfc(abs(d) / se / math.sqrt(2.0)) if se > 0 else (0.0 if d != 0 else 1.0)
    return float(d), float(se), float(p)


class AucBoot:
    """Weighted AUC for bootstrap resampling: the sort and the tie blocks are computed once."""

    def __init__(self, s, y):
        s = np.asarray(s, dtype=np.float64)
        self.o = np.argsort(s, kind="mergesort")
        ss = s[self.o]
        self.blk = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]) if len(ss) else np.zeros(0, dtype=np.int64)
        self.nb = int(self.blk[-1]) + 1 if len(ss) else 0
        self.yo = np.asarray(y).astype(bool)[self.o]

    def auc(self, w):
        wo = np.asarray(w, dtype=np.float64)[self.o]
        wn = np.bincount(self.blk[~self.yo], wo[~self.yo], minlength=self.nb)
        wp = np.bincount(self.blk[self.yo], wo[self.yo], minlength=self.nb)
        den = wp.sum() * wn.sum()
        if den <= 0:
            return np.nan
        return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / den)


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return None if d <= 0 else float((a * b).sum() / d)


def spearman(x, y, min_n=3):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return None
    rx, ry = rank_avg(x[ok]), rank_avg(y[ok])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return None if d <= 0 else float((rx * ry).sum() / d)


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z ((n,) or (n, q)): Pearson of the rank residuals."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    z = z[:, None] if z.ndim == 1 else z
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(1)
    if ok.sum() < z.shape[1] + 4:
        return None
    R = np.column_stack([np.ones(int(ok.sum()))] + [rank_avg(z[ok, j]) for j in range(z.shape[1])])
    rx, ry = rank_avg(x[ok]), rank_avg(y[ok])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    d = math.sqrt((ex * ex).sum() * (ey * ey).sum())
    return None if d <= 0 else float((ex * ey).sum() / d)


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def holm(pvals):
    """Holm step-down adjusted p-values (None stays None); same rule as scripts/b4_stats.js holm()."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out = [None] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    m, run = len(order), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (m - j) * pvals[i]))
        out[i] = run
    return out


def finite_fix(x):
    """Replace non-finite values by the median of the finite ones; returns (x, count replaced)."""
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        good = x[~bad]
        x[bad] = np.median(good) if len(good) else 0.0
    return x, int(bad.sum())


# =====================================================================================================================
# head statistics, temperature, ECE (inputs: the stored float32 logits)
# =====================================================================================================================
HEAD_STATS = ("msp", "maxlogit", "gap", "energy", "entropy", "gini", "smargin", "pnorm2", "mspT")
ERR_SIGN = {"msp": -1.0, "maxlogit": -1.0, "gap": -1.0, "energy": -1.0, "entropy": 1.0, "gini": -1.0,
            "smargin": -1.0, "pnorm2": -1.0, "mspT": -1.0}          # oriented: larger = more error- / shift-like


def softmax(Z, T=1.0):
    Z = np.asarray(Z, dtype=np.float64) / T
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def head_stats(Z, T=1.0):
    """Every head statistic of T1's full head summary. energy = logsumexp(Z) (the negative free energy)."""
    Z = np.asarray(Z, dtype=np.float64)
    zs = -np.sort(-Z, axis=1)
    m = zs[:, 0]
    lse = m + np.log(np.exp(Z - m[:, None]).sum(1))
    P = np.exp(Z - lse[:, None])
    ps = -np.sort(-P, axis=1)
    Zc = Z - Z.mean(1, keepdims=True)
    return {"msp": ps[:, 0], "maxlogit": m, "gap": zs[:, 0] - zs[:, 1], "energy": lse,
            "entropy": -(P * np.log(np.clip(P, 1e-300, None))).sum(1), "gini": (P * P).sum(1),
            "smargin": ps[:, 0] - ps[:, 1], "pnorm2": Zc.max(1) / (np.linalg.norm(Zc, axis=1) + 1e-12),
            "mspT": softmax(Z, T).max(1), "argmax": Z.argmax(1)}


def sorted_logits(Z, m=10):
    return -np.sort(-np.asarray(Z, dtype=np.float64), axis=1)[:, :m]


def nll(Z, y, T=1.0):
    P = softmax(Z, T)
    return float(-np.log(np.clip(P[np.arange(len(y)), np.asarray(y, dtype=np.int64)], 1e-300, None)).mean())


def fit_temperature(Z, y, grid=TEMP_GRID, iters=80):
    """Temperature minimising the NLL: golden-section search on log T within grid."""
    Z = np.asarray(Z, dtype=np.float64)
    a, b = math.log(grid[0]), math.log(grid[1])
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(Z, y, math.exp(c)), nll(Z, y, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(Z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(Z, y, math.exp(d))
    return float(math.exp((a + b) / 2))


def ece(msp, correct, bins=ECE_BINS):
    """Expected calibration error, equal-width confidence bins."""
    msp, correct = np.asarray(msp, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(msp, e[1:-1]), 0, bins - 1)
    tot = 0.0
    for j in range(bins):
        m = idx == j
        if m.any():
            tot += m.sum() * abs(msp[m].mean() - correct[m].mean())
    return float(tot / max(1, len(msp)))


# =====================================================================================================================
# the cross-fitted joint model: L2-penalised logistic regression on restricted cubic splines, and the HEAD-ADDITIVE rule
# =====================================================================================================================
def rcs(x, t):
    """Restricted cubic spline basis (Harrell), knots t ascending: columns x, then len(t) - 2 cubic terms."""
    k = len(t)
    scale = (t[-1] - t[0]) ** 2 or 1.0
    cols = [x]
    for j in range(k - 2):
        c = (np.maximum(x - t[j], 0) ** 3
             - np.maximum(x - t[k - 2], 0) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + np.maximum(x - t[k - 1], 0) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / scale
        cols.append(c)
    return np.stack(cols, 1)


def design(blocks, tr, te, knots_q=KNOTS_Q):
    """blocks: [(name, array (n,) or (n, m), 'spline' | 'linear')]. Knots and standardisation from the training rows
    only (a column whose knots collapse enters linearly). Returns (X_tr, X_te)."""
    ctr, cte = [], []
    for _, x, kind in blocks:
        x = np.asarray(x, dtype=np.float64)
        x = x[:, None] if x.ndim == 1 else x
        for j in range(x.shape[1]):
            a_tr, a_te = x[tr, j], x[te, j]
            done = False
            if kind == "spline":
                kn = np.quantile(a_tr, knots_q)
                if np.all(np.diff(kn) > 1e-9 * (np.abs(kn).max() + 1e-12)):
                    ctr.append(rcs(a_tr, kn))
                    cte.append(rcs(a_te, kn))
                    done = True
            if not done:
                ctr.append(a_tr[:, None])
                cte.append(a_te[:, None])
    Xtr, Xte = np.hstack(ctr), np.hstack(cte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-12
    return (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]


def logit_fit(X, y, lam=RIDGE, max_iter=60, tol=1e-8):
    """Minimise mean NLL + lam/2 ||beta_{1:}||^2 by damped Newton. Returns beta (intercept first)."""
    n, p = X.shape
    Xa = np.column_stack([np.ones(n), X])
    y = np.asarray(y, dtype=np.float64)
    ybar = min(max(y.mean(), 1e-6), 1 - 1e-6)
    beta = np.zeros(p + 1)
    beta[0] = math.log(ybar / (1 - ybar))
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0

    def obj(bb):
        eta = np.clip(Xa @ bb, -35, 35)
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * bb @ R @ bb)
    f = obj(beta)
    for _ in range(max_iter):
        eta = np.clip(Xa @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        g = Xa.T @ (mu - y) / n + R @ beta
        H = (Xa.T * w) @ Xa / n + R + 1e-12 * np.eye(p + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t, accepted = 1.0, False
        while t > 1e-6:
            nb = beta - t * step
            fn = obj(nb)
            if fn <= f + 1e-12:
                accepted = True
                break
            t *= 0.5
        if not accepted:                                  # no descent step: at the optimum to numerical precision
            break
        beta, f = nb, fn
        if np.max(np.abs(t * step)) < tol:
            break
    return beta


def logit_predict(beta, X):
    eta = np.clip(beta[0] + X @ beta[1:], -35, 35)
    return 1.0 / (1.0 + np.exp(-eta))


def folds_for(groups, key="folds", k=N_FOLDS):
    """Fold id per sample from its group (image row): crc32-hashed, so dump- and order-independent."""
    g = np.asarray(groups, dtype=np.int64)
    ug, inv = np.unique(g, return_inverse=True)
    h = np.array([zlib.crc32(f"{key}|{int(v)}".encode()) for v in ug], dtype=np.int64)
    return (h % k)[inv]


def crossfit(blocks, y, folds, train_ok=None, lam=RIDGE):
    """Out-of-fold probabilities. train_ok (bool) restricts which samples may train (transfer designs)."""
    y = np.asarray(y, dtype=np.float64)
    folds = np.asarray(folds)
    oof = np.full(len(y), np.nan)
    ok = np.ones(len(y), dtype=bool) if train_ok is None else np.asarray(train_ok, dtype=bool)
    for f in np.unique(folds):
        te = folds == f
        tr = (~te) & ok
        if tr.sum() < 10 or y[tr].min() == y[tr].max():
            oof[te] = y[tr].mean() if tr.any() else 0.5
            continue
        Xtr, Xte = design(blocks, tr, te)
        oof[te] = logit_predict(logit_fit(Xtr, y[tr], lam), Xte)
    return oof


def fit_apply(blocks, y, tr, te, lam=RIDGE):
    """Fit on tr, predict te (a frozen transfer, e.g. fit rows -> eval rows)."""
    Xtr, Xte = design(blocks, tr, te)
    return logit_predict(logit_fit(Xtr, np.asarray(y, dtype=np.float64)[tr], lam), Xte)


def ce_bits(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def boot_ci(y, groups, pa, pb, nboot, key, level=0.95):
    """Group (image-row) bootstrap: percentile CIs of AUC(pa) - AUC(pb) and of CE(pb) - CE(pa) in bits (positive = pa
    better). Deterministic in key."""
    ug, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = len(ug)
    rng = rng_for("boot", key)
    A, B = AucBoot(pa, y), AucBoot(pb, y)
    dce = ce_bits(y, pb) - ce_bits(y, pa)
    da, di = np.empty(nboot), np.empty(nboot)
    for i in range(nboot):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        w = cnt[inv].astype(np.float64)
        da[i] = A.auc(w) - B.auc(w)
        di[i] = (w * dce).sum() / w.sum()
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2

    def q(v):
        return [float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi))]
    return q(da), q(di)


def head_additive_call(dauc, dauc_ci, dI_ci, add=ADD_DAUC, eq=EQ_BOUND):
    """The frozen HEAD-ADDITIVE rule with the equivalence bound (T1 review C1): ADDS | BOUNDED | INCONCLUSIVE."""
    if dauc is None or dauc_ci is None or dI_ci is None or None in dauc_ci or None in dI_ci:
        return "INCONCLUSIVE"
    if dauc >= add and dI_ci[0] > 0:
        return "ADDS"
    if dauc_ci[1] < eq:
        return "BOUNDED"
    return "INCONCLUSIVE"


def head_increment(head_blocks, bundle_blocks, y, groups, nboot, key, folds=None, train_ok=None, eval_mask=None,
                   lam=RIDGE):
    """Cross-fitted head-only vs head + bundle on the same folds; metrics on eval_mask. Returns
    {auc_head, auc_joint, dauc, dauc_ci, dI_bits, dI_ci, delong_se, delong_p, tpr5_head, tpr5_joint, call, n_pos, n_neg}."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    folds = folds_for(groups) if folds is None else np.asarray(folds)
    em = np.ones(len(y), dtype=bool) if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    ph = crossfit(head_blocks, y, folds, train_ok, lam)[em]
    pj = crossfit(list(head_blocks) + list(bundle_blocks), y, folds, train_ok, lam)[em]
    ye, ge = y[em], groups[em]
    ah, aj = auc_y(ye, ph), auc_y(ye, pj)
    if ah is None or aj is None:
        return {"n_pos": int(ye.sum()), "n_neg": int(len(ye) - ye.sum()), "call": "INCONCLUSIVE", "skipped": "one class"}
    d, se, p = delong_diff(ye, pj, ph)
    ca, ci = boot_ci(ye, ge, pj, ph, nboot, key)
    dI = float(ce_bits(ye, ph).mean() - ce_bits(ye, pj).mean())
    rec = {"n_pos": int(ye.sum()), "n_neg": int(len(ye) - ye.sum()), "auc_head": ah, "auc_joint": aj,
           "dauc": aj - ah, "dauc_ci": ca, "dI_bits": dI, "dI_ci": ci, "delong_se": se, "delong_p": p,
           "tpr5_head": tpr_at_fpr(ye, ph), "tpr5_joint": tpr_at_fpr(ye, pj)}
    rec["call"] = head_additive_call(rec["dauc"], ca, ci)
    return rec


def risk_coverage(y_err, conf, coverages=(0.8, 0.9, 0.95)):
    """Selective risk accepting the most confident first. Returns (AURC, {coverage: risk})."""
    y = np.asarray(y_err, dtype=np.float64)
    o = np.argsort(-np.asarray(conf, dtype=np.float64), kind="mergesort")
    cum = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    out = {}
    for c in coverages:
        m = max(1, int(math.ceil(c * len(y))))
        out[f"{c:.2f}"] = float(cum[m - 1])
    return float(cum.mean()), out


def fpr_at_tpr(y, s, tpr=0.95):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    tau = np.quantile(s[y], 1.0 - tpr)
    return float((s[~y] >= tau).mean())


def tpr_at_fpr(y, s, fpr=0.05):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    tau = np.quantile(s[~y], 1.0 - fpr)
    return float((s[y] > tau).mean())


# =====================================================================================================================
# conformal p-values and calibrated thresholds (D14)
# =====================================================================================================================
def conformal_p(cal, s):
    """p(x) = (1 + #{cal >= s(x)}) / (n_cal + 1); larger s = more anomalous (D14)."""
    cs = np.sort(np.asarray(cal, dtype=np.float64))
    ge = len(cs) - np.searchsorted(cs, np.asarray(s, dtype=np.float64), side="left")
    return (1.0 + ge) / (len(cs) + 1.0)


def cal_threshold(cal, alpha):
    """The ceil((1 - alpha)(n + 1))-th smallest calibration score (inf when n is too small): flag s > threshold."""
    cs = np.sort(np.asarray(cal, dtype=np.float64))
    j = int(math.ceil((1 - alpha) * (len(cs) + 1)))
    return float(cs[j - 1]) if j <= len(cs) else float("inf")


def cauchy_stat(P):
    """Cauchy combination of p-values (Liu & Xie 2020), equal weights: mean tan((0.5 - p) pi); larger = more anomalous."""
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.tan((0.5 - P) * np.pi).mean(1)


# =====================================================================================================================
# geometry helpers
# =====================================================================================================================
def rng_for(*parts, seed=SEED):
    """numpy Generator keyed by (seed, crc32 of each part): independent of call order, identical on every host."""
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(p).encode("utf-8")) for p in parts])


def eig_desc(C, r=None):
    """Eigenpairs of a symmetric matrix, descending, above EIG_FLOOR * lambda_1 (at most r)."""
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    if len(w) == 0 or not w[0] > 0:
        return w[:0], V[:, :0]
    k = int(np.sum(w > EIG_FLOOR * w[0]))
    if r is not None:
        k = min(k, int(r))
    return w[:k], V[:, :k]


def pca_frame(X, r=None):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    w, V = eig_desc(Xc.T @ Xc / max(1, len(X) - 1), r)
    return mu, w, V


def whiten(X, frame):
    mu, w, V = frame
    return (np.asarray(X, dtype=np.float64) - mu) @ V / np.sqrt(w)


def l2n(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def sqdist(Q, F, f2=None):
    Q = np.asarray(Q, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    q2 = np.einsum("ij,ij->i", Q, Q)
    f2 = np.einsum("ij,ij->i", F, F) if f2 is None else f2
    d2 = q2[:, None] + f2[None, :] - 2.0 * (Q @ F.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def knn(fit, query, k, exclude_self=False, chunk=512):
    """Exact k nearest rows of `fit` per query row: (dist (n, k) ascending, idx (n, k)). exclude_self: the queries ARE
    the fit rows in the same order and the self match is removed."""
    fit = np.asarray(fit, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if k + (1 if exclude_self else 0) > len(fit):
        raise ValueError(f"knn: k={k} needs more than {len(fit)} fit rows")
    f2 = np.einsum("ij,ij->i", fit, fit)
    D = np.empty((len(query), k))
    I = np.empty((len(query), k), dtype=np.int64)                          # noqa: E741
    for i in range(0, len(query), chunk):
        d2 = sqdist(query[i:i + chunk], fit, f2)
        if exclude_self:
            d2[np.arange(len(d2)), np.arange(i, i + len(d2))] = np.inf
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dp = np.take_along_axis(d2, part, 1)
        o = np.argsort(dp, axis=1, kind="mergesort")
        D[i:i + len(d2)] = np.sqrt(np.take_along_axis(dp, o, 1))
        I[i:i + len(d2)] = np.take_along_axis(part, o, 1)
    return D, I


def knn_kth(fit, query, k=10, exclude_self=False, chunk=1024):
    """Distance from every query row to its k-th nearest fit row (self excluded when the queries are the fit rows)."""
    fit = np.asarray(fit, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    f2 = np.einsum("ij,ij->i", fit, fit)
    out = np.empty(len(query))
    for i in range(0, len(query), chunk):
        d2 = sqdist(query[i:i + chunk], fit, f2)
        if exclude_self:
            d2[np.arange(len(d2)), np.arange(i, i + len(d2))] = np.inf
        out[i:i + chunk] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return out


def class_means(X, y, n_classes=K):
    X = np.asarray(X, dtype=np.float64)
    C = np.zeros((n_classes, X.shape[1]))
    for c in range(n_classes):
        m = y == c
        if not m.any():
            raise ValueError(f"class {c} absent")
        C[c] = X[m].mean(0)
    return C


# =====================================================================================================================
# split names
# =====================================================================================================================
def csplit(c, s):
    return f"corrupt__{c}__s{int(s)}"


def fsplit(kind, level):
    return f"fault__{kind}__l{int(level)}"


def parse_split(split):
    """-> (group, name, level). corrupt__fog__s3 -> ('corrupt', 'fog', 3); fault__deadpix__l2 -> ('fault', 'deadpix', 2);
    fault__paste__gaussian_noise__a12 -> ('fault', 'paste__gaussian_noise', 'a12'); fault__glare__a25 -> ('fault',
    'glare', 'a25'); global__fog__s3 -> ('global', 'fog', 3); ood__svhn -> ('ood', 'svhn', None); test -> ('test', None,
    None)."""
    p = split.split("__")
    if p[0] in ("corrupt", "global") and len(p) == 3:
        return p[0], p[1], int(p[2][1:])
    if p[0] == "fault" and len(p) >= 3:
        lv = p[-1]
        return "fault", "__".join(p[1:-1]), (int(lv[1:]) if re.fullmatch(r"l\d+", lv) else lv)
    if p[0] == "ood" and len(p) == 2:
        return "ood", p[1], None
    return split, None, None
