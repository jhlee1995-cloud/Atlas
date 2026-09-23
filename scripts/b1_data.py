"""
b1_data.py -- B1 data (docs/plans/B1_VIT_MARGIN.md). Never imports the `datasets` library (CLAUDE.md "Never").

  python scripts/b1_data.py provision /workspace                       # download what is missing (pinned), verify
  python scripts/b1_data.py verify /workspace --out results/instrument_check_b1/data.json

ImageNet val: the two parquet files of the non-gated HF mirror benjamin-paine/imagenet-1k-256x256 at a pinned revision
(sizes, LFS sha256: https://huggingface.co/api/datasets/benjamin-paine/imagenet-1k-256x256/tree/main/data) in
<volume>/datasets/imagenet_val (the atlas.extract_acts._resolve fallback; extract/populate_data.py has no imagenet_val).
The mirror card carries the ImageNet terms of access (non-commercial research and education); the owner approved this
use, and committed B1 results carry validation row indices and statistics only (no image, activation or per-image
prediction: dumps stay on the volume). Only the two validation files are downloaded (data/ also holds train and test
shards; test rows carry label 1000).
Hard: both files at size and sha256, 50,000 rows, labels 0..999 x 50 (exit 1). ReaL labels (E11, exploratory): real.json
at a pinned commit; missing or wrong is recorded and never fatal. When it verifies, `real.label_in_real_frac` (the share
of rows with a non-empty ReaL set whose mirror label is in that set) is gate G0's dataset-level label-order check
(>= 0.5; a misaligned order gives about 0.001).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

REPO = "benjamin-paine/imagenet-1k-256x256"
REVISION = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
FILES = {
    "data/validation-00000-of-00002.parquet": (351766666, "399ae54c309fa71f99b6f3c121ad4507ae3103f0205923e6695b6a6bac52c6eb"),
    "data/validation-00001-of-00002.parquet": (351313139, "3c665d47c5057e237be8c15348570466478e4a73b8d4ff913ed319cd8e3f16f7"),
}
REAL_URL = ("https://raw.githubusercontent.com/google-research/reassessed-imagenet/"
            "bcd006fe12e929f055e0ec7fe39dbf4d64a6881a/real.json")
REAL = (388478, "d83e9bff374c631aae8439eb064c7019acc56e1b3bc3f56b8380c2a710b0220b")


def root(volume):
    return os.path.join(volume, "datasets", "imagenet_val")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def verify(volume):
    import pyarrow.parquet as pq
    r, bad, labels, files = root(volume), [], [], {}
    for rel, (size, sha) in FILES.items():
        p = os.path.join(r, rel)
        rec = {"exists": os.path.isfile(p)}
        if rec["exists"]:
            rec["size"] = os.path.getsize(p)
            rec["sha256"] = sha256(p) if rec["size"] == size else None
            if rec["sha256"] == sha:
                labels.append(pq.read_table(p, columns=["label"]).column("label").to_numpy())
        files[rel] = rec
        if rec.get("sha256") != sha:
            bad.append(f"{rel}: {rec}")
    rep = {"repo": REPO, "revision": REVISION, "root": r, "files": files, "real": {"ok": False}}
    if not bad:
        y = np.concatenate(labels).astype(np.int64)
        c = np.bincount(y, minlength=1000)
        rep.update({"n_rows": int(len(y)), "labels": [int(y.min()), int(y.max())], "per_class": [int(c.min()), int(c.max())]})
        if len(y) != 50000 or y.min() < 0 or y.max() > 999 or (c != 50).any():
            bad.append(f"{len(y)} rows, labels {y.min()}..{y.max()}, {c.min()}..{c.max()} per class (want 50000, 0..999, 50)")
        rp = os.path.join(r, "real.json")
        if not bad and os.path.isfile(rp) and os.path.getsize(rp) == REAL[0] and sha256(rp) == REAL[1]:
            with open(rp) as f:
                sets = json.load(f)
            ne = [i for i, s in enumerate(sets) if s]
            rep["real"] = {"ok": len(sets) == 50000, "sha256": REAL[1], "n_nonempty": len(ne),
                           "label_in_real_frac": float(np.mean([int(y[i]) in sets[i] for i in ne])) if ne else None}
    rep.update({"status": "PASS" if not bad else "FAIL", "problems": bad})
    return rep


def provision(volume):
    from huggingface_hub import snapshot_download
    r = root(volume)
    os.makedirs(r, exist_ok=True)
    if verify(volume)["status"] != "PASS":
        print("[b1_data] snapshot_download", REPO, "@", REVISION, "->", snapshot_download(
            repo_id=REPO, repo_type="dataset", revision=REVISION, local_dir=r, allow_patterns=list(FILES)))
    if not verify(volume)["real"]["ok"]:
        tmp = os.path.join(r, "real.json.tmp")
        try:
            urllib.request.urlretrieve(REAL_URL, tmp)
            os.replace(tmp, os.path.join(r, "real.json"))
        except Exception as e:                       # E11 then not evaluable; never fatal
            print(f"[b1_data] real.json not downloaded: {e!r}")
    return verify(volume)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("provision", "verify"))
    ap.add_argument("volume")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = (provision if a.cmd == "provision" else verify)(a.volume)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    print(f"[b1_data] {a.cmd} {rep['status']}: rows {rep.get('n_rows')}, per class {rep.get('per_class')}, "
          f"ReaL {rep['real']} {rep['problems']}")
    sys.exit(0 if rep["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
