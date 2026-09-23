"""
tests/test_b1_imagenet.py -- B1 (ImageNet, ViT) plumbing checks, CPU only (docs/plans/B1_VIT_MARGIN.md, integration D22).
Run before the launch (python -m pytest -q tests/) and again inside pod_atlas.sh block_b1; a failure there stops B1
only (tests/test_atlas_smoke.py stays the session's hard gate and imports none of torch, timm, pyarrow,
huggingface_hub). Tests that need torch / torchvision / timm / pyarrow skip when the package is missing; block_b1 imports
all of them first, so on the pod nothing skips.

  1. the pre-registered split rule, the legacy-loop replay (plain batch-size lists), the manifests (validate_cfg on every
     ImageNet manifest and refusals of broken copies; the E9 manifests equal their A4b sources plus the b1 keys)
  2. parquet loader and the legacy row count against the literal Upgraded-Mod loop over real parquet files
  3. hooks: BlockHooks on torchvision resnet50; ViTHooks on tiny torchvision and timm ViTs (tap values recomputed by
     hand); the random-input self-test
  4. extract_imagenet end to end on a fake parquet volume (tiny models): the dump meta passes scripts/b1_gate.py P0
     items, preds carry logit_gap and real_ok, Stage B gives the B1 keys; the legacy_first path
  5. scripts/b1_gate.py known answers on tests/fixtures/b1/ (the same fixtures as tests/fixtures/b1/check_fixtures.js)
     and the gate record reuse rule
"""
import copy
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from atlas import extract_imagenet as ei  # noqa: E402
from atlas.build import build_atlas  # noqa: E402
from atlas.config import load_manifest  # noqa: E402
from atlas.extract_acts import file_sha256  # noqa: E402

Q = os.path.join(ROOT, "experiments", "queue")
FX = os.path.join(ROOT, "tests", "fixtures", "b1")
IMAGENET = ["margin_b1_resnet50_legacy10k", "margin_b1_resnet50", "margin_b1_resnet50_swap", "margin_b1_vitb16",
            "margin_b1_vitb16_swap", "margin_b1_deitb", "margin_b1_deitb_swap"]
E9 = ["resnet20_s0hub_st3", "resnet56_s0hub_st3", "resnet56_s1", "resnet56_s2"]


# ---- 1: split rule, legacy replay, manifests ------------------------------------------------------------------
def test_split_rule_prereg():
    labels = np.repeat(np.arange(4), 50)
    np.random.default_rng(0).shuffle(labels)
    A, B = ei.split_parts(labels, 0, 25, 4)
    assert len(A) == len(B) == 100 and not np.intersect1d(A, B).size
    assert np.array_equal(np.union1d(A, B), np.arange(200)) and np.array_equal(A, np.sort(A))
    assert all((labels[A] == c).sum() == 25 and (labels[B] == c).sum() == 25 for c in range(4))
    rng, lit = np.random.default_rng(0), []                  # the notes' rule, literally
    for c in range(4):
        rows = np.flatnonzero(labels == c)
        lit += list(rows[rng.permutation(50)[:25]])
    assert np.array_equal(A, np.sort(lit))
    assert np.array_equal(ei.split_parts(labels, 0, 25, 4)[0], A)
    assert not np.array_equal(ei.split_parts(labels, 1, 25, 4)[0], A)


def _legacy_loop_rows(batch_rows, n, loop=64):
    """Upgraded-Mod imagenet_extract.py:49-60 with every row decoding, counting rows instead of decoding them."""
    nd = bi = total = 0
    for b in batch_rows:
        for _ in range(b):
            bi += 1
            total += 1
            if bi >= loop:
                bi = 0
                nd += loop
        if nd >= n:
            break
    return total


def test_legacy_rows_replay():
    assert ei.legacy_rows_from_batches([64] * 200, 10000) == _legacy_loop_rows([64] * 200, 10000) == 10048
    rng = np.random.default_rng(1)
    for trial in range(50):
        sizes = [int(s) for s in rng.integers(1, 101, size=300)]
        for n in (1, 64, 100, 1000, 10000):
            assert ei.legacy_rows_from_batches(sizes, n) == _legacy_loop_rows(sizes, n), (trial, n)
    assert ei.legacy_rows_from_batches([64] * 156 + [40, 64], 10000) == _legacy_loop_rows([64] * 156 + [40, 64], 10000)


def test_validate_cfg_b1_manifests():
    for n in IMAGENET:
        ei.validate_cfg(load_manifest(os.path.join(Q, n + ".yaml")))
    base = load_manifest(os.path.join(Q, "margin_b1_vitb16.yaml"))
    breaks = [lambda c: c["extract"].update(dtype="float16"),
              lambda c: c["data"]["reference"].update(n=10000),
              lambda c: c["data"]["clean_test"].update(part="A"),
              lambda c: c.update(cross_layer="all"),
              lambda c: c.update(invariants=["margin_typeb", "class_centers"]),
              lambda c: c["data"]["imagenet_val"]["files"][0].update(sha256="0" * 64),
              lambda c: c["hooks"].update(extra=[]),
              lambda c: c["invariant_cfg"]["margin_typeb"].update(b1=False),
              lambda c: c["backbone"].update(transform="legacy_imagenet_extract"),
              lambda c: c["data"].update(panel={"dataset": "imagenet_val", "n": 64, "seed": 123})]
    for br in breaks:
        c = copy.deepcopy(base)
        br(c)
        with pytest.raises(SystemExit):
            ei.validate_cfg(c)
    leg =load_manifest(os.path.join(Q, "margin_b1_resnet50_legacy10k.yaml"))
    assert leg["data"]["reference"]["n"] is None and leg["data"]["clean_test"]["n"] is None
    c = copy.deepcopy(leg)
    c["data"]["clean_test"]["n"] = 5000                    # what config.DEFAULTS would have filled in
    with pytest.raises(SystemExit):
        ei.validate_cfg(c)
    deit = load_manifest(os.path.join(Q, "margin_b1_deitb.yaml"))
    assert deit["backbone"]["weights_sha256"].startswith("cd2da27b") and deit["backbone"]["hf_revision"].startswith("b78cc553")
    for n in IMAGENET[1:]:                                   # the six split runs share every instrument key
        m = load_manifest(os.path.join(Q, n + ".yaml"))
        for k in ("invariants", "cross_layer", "invariant_cfg", "extract", "holdout"):
            assert m[k] == base[k], (n, k)
        assert m["data"]["imagenet_val"] == base["data"]["imagenet_val"], n


def test_e9_manifests_are_the_a4b_rebuilds_plus_b1_keys():
    """Integration D9: every E9 manifest equals margin_v1_<same> (A4b's flag-off rebuild of the same dump) in every
    instrument key except the b1 opt-in keys, so every A3 key must come out identical."""
    for x in E9:
        a, b = load_manifest(os.path.join(Q, f"margin_v1_{x}.yaml")), load_manifest(os.path.join(Q, f"margin_b1_{x}.yaml"))
        for k in ("backbone", "data", "hooks", "extract", "invariants", "cross_layer", "probes"):
            assert a[k] == b[k], (x, k)
        assert b["exp_id"] == f"margin_b1_{x}" and b["outputs"]["root"] == f"results/margin_b1_{x}"
        ma, mb = a["invariant_cfg"]["margin_typeb"], b["invariant_cfg"]["margin_typeb"]
        assert mb == dict(ma, b1=True, strat_bins=10, strat_min=5), x
        assert a["holdout"]["discovery_seeds"] == b["holdout"]["discovery_seeds"]


# ---- 2: parquet loader and the legacy row count ---------------------------------------------------------------
def _jpeg(i, lab):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (8, 8), ((37 * i) % 256, (11 * i) % 256, (60 * int(lab)) % 256)).save(b, "JPEG")
    return b.getvalue()


def _fake_volume(tmp, K=4, per_class=25, sizes=(120, 80), row_group=50):
    """Two parquet files shaped like the mirror's validation split (image struct {bytes, path}, int64 label; every class
    2 x per_class rows spread over both files; row groups not a multiple of 64) and a real.json. Returns (volume,
    manifest-style file records, labels, raw image bytes, real sets)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    n = K * 2 * per_class
    assert sum(sizes) == n
    labels = np.random.default_rng(3).permutation(np.repeat(np.arange(K), 2 * per_class))
    raw = [_jpeg(i, labels[i]) for i in range(n)]
    stype = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    os.makedirs(os.path.join(tmp, "datasets", "imagenet_val", "data"))
    files, start = [], 0
    for fi, m in enumerate(sizes):
        t = pa.table({"image": pa.array([{"bytes": raw[i], "path": None} for i in range(start, start + m)], type=stype),
                      "label": pa.array(labels[start:start + m].astype(np.int64), type=pa.int64())})
        rel = f"data/validation-0000{fi}-of-00002.parquet"
        p = os.path.join(tmp, "datasets", "imagenet_val", rel)
        pq.write_table(t, p, row_group_size=row_group)
        files.append({"path": rel, "sha256": file_sha256(p, 64)})
        start += m
    real = [[int(labels[i])] if i % 5 else [] for i in range(n)]
    with open(os.path.join(tmp, "datasets", "imagenet_val", "real.json"), "w") as f:
        json.dump(real, f)
    return tmp, files, labels, raw, real


def _literal_legacy_rows(volume, files, n):
    """The literal legacy scan (every column read, pq.ParquetFile(f).iter_batches(batch_size=64)), counting rows."""
    import pyarrow.parquet as pq
    sizes = []
    for f in files:
        sizes += [b.num_rows for b in pq.ParquetFile(os.path.join(volume, "datasets", "imagenet_val", f["path"]))
                  .iter_batches(batch_size=64)]
    return _legacy_loop_rows(sizes, n)


def test_parquet_loader_and_legacy_rows(tmp_path):
    vol, files, labels, raw, _ = _fake_volume(str(tmp_path))
    blob, off, y, recs = ei.load_imagenet_val(vol, files)
    assert np.array_equal(y, labels) and [r["rows"] for r in recs] == [120, 80]
    assert all(r["sha256"] == r["sha256_expected"] for r in recs)
    assert all(blob[off[i]:off[i + 1]].tobytes() == raw[i] for i in (0, 119, 120, 199))
    for n in (1, 64, 100, 130, 1000):
        L, sizes = ei.legacy_first_rows(vol, files, n)
        assert L == _literal_legacy_rows(vol, files, n) and L == ei.legacy_rows_from_batches(sizes, n), n


# ---- 3: hooks and the self-test --------------------------------------------------------------------------------
def test_block_hooks_torchvision_resnet50():
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    from atlas.extract_acts import BlockHooks
    m = tvm.resnet50(weights=None).eval()
    h = BlockHooks(m, "blocks", 1, "gap")
    blocks = [f"layer{s}.{i}" for s, n in ((1, 3), (2, 4), (3, 6), (4, 3)) for i in range(n)]
    assert h.layer_names == ["stem"] + blocks + ["penult"]
    f, logits = h.forward(torch.randn(2, 3, 64, 64))
    width = {"1": 256, "2": 512, "3": 1024, "4": 2048}
    assert {k: v.shape[1] for k, v in f.items()} == {"stem": 64, **{b: width[b[5]] for b in blocks}, "penult": 2048}
    assert list(f) == h.layer_names
    assert torch.allclose(f["penult"], f["layer4.2"], atol=1e-5)             # avgpool == GAP of the last block
    with torch.no_grad():
        assert torch.allclose(m.fc(f["penult"]), logits, atol=1e-4)
    h.close()
    hs = BlockHooks(m, "stages", 1, "gap")
    assert hs.layer_names == ["stem", "layer1", "layer2", "layer3", "layer4", "penult"]
    hs.close()


def _tiny_tv_vit(num_classes=5, layers=3):
    import torch
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(0)
    m = VisionTransformer(image_size=32, patch_size=8, num_layers=layers, num_heads=2, hidden_dim=16, mlp_dim=32,
                          num_classes=num_classes).eval()
    torch.nn.init.normal_(m.heads.head.weight)              # torchvision zero-initialises the head
    return m


def test_vit_hooks_torchvision():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    m = _tiny_tv_vit()
    h = ei.ViTHooks(m, "torchvision")
    assert h.layer_names == ["block.0", "block.1", "block.2", "penult_mean", "penult"]
    x = torch.randn(4, 3, 32, 32)
    f, logits = h.forward(x)
    fc = h.final_cls
    with torch.no_grad():
        t = torch.cat([m.class_token.expand(4, -1, -1), m._process_input(x)], dim=1) + m.encoder.pos_embedding
        outs, z = [], t
        for blk in m.encoder.layers:
            z = blk(z)
            outs.append(z)
        fin = m.encoder.ln(z)
        head = m.heads.head(f["penult"])
    close = lambda a, b: torch.allclose(a, b, atol=1e-5)
    assert all(close(f[f"block.{i}"], outs[i][:, 0]) for i in range(3))
    assert close(f["penult_mean"], fin[:, 1:].mean(1)) and close(f["penult"], fin[:, 0])
    assert torch.equal(fc, f["penult"])            # the head input is the final-norm class token itself
    assert close(head, logits) and list(f) == h.layer_names and all(tuple(v.shape) == (4, 16) for v in f.values())
    h.close()
    h2 = ei.ViTHooks(m, "torchvision", block_stride=2)
    assert h2.layer_names == ["block.0", "block.2", "penult_mean", "penult"]
    h2.close()


def test_vit_hooks_timm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer
    torch.manual_seed(0)
    m = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, num_heads=2, num_classes=5).eval()
    h = ei.ViTHooks(m, "timm")
    assert h.layer_names == ["block.0", "block.1", "penult_mean", "penult"]
    x = torch.randn(3, 3, 32, 32)
    f, logits = h.forward(x)
    with torch.no_grad():
        z = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
        outs = []
        for blk in m.blocks:
            z = blk(z)
            outs.append(z)
        ff = m.forward_features(x)
        head = m.head(f["penult"])
    n0 = m.num_prefix_tokens
    close = lambda a, b: torch.allclose(a, b, atol=1e-5)
    assert all(close(f[f"block.{i}"], outs[i][:, 0]) for i in range(2))
    assert close(f["penult"], ff[:, 0]) and close(f["penult_mean"], ff[:, n0:].mean(1))
    assert close(head, logits)
    h.close()


def _vit_cfg(root, files, K=4, legacy=False, arch="tv_vit_b_16"):
    """A B1 manifest dict for the fake volume (validate_cfg is not applied: 4 classes, fake files)."""
    ref = ({"dataset": "imagenet_val", "part": "legacy_first", "n_target": 100, "loop_batch": 64, "n": None, "seed": 0}
           if legacy else {"dataset": "imagenet_val", "part": "A", "split_seed": 0, "per_class": 25, "n": 100, "seed": 0})
    ct = dict(ref, part=ref["part"] if legacy else "B")
    ct.pop("seed")
    return {"exp_id": "b1_e2e", "n_classes": K,
            "backbone": {"arch": arch, "weights": "IMAGENET1K_V1", "seed_tag": "t", "norm": "imagenet",
                         "transform": "legacy_imagenet_extract" if legacy else "official"},
            "data": {"imagenet_val": {"repo": "fake/fake", "revision": "0" * 40, "files": files},
                     "reference": ref, "clean_test": ct, "panel": None, "cifar10c": None, "ood": []},
            "hooks": ({"layers": "stages", "block_stride": 1, "pooling": "gap"} if legacy
                      else {"layers": "vit_blocks", "block_stride": 1, "pooling": "cls", "extra": ["penult_mean"]}),
            "extract": {"batch": 32, "dtype": "float32", "num_workers": 0, "store_logits": True},
            "invariants": ["margin_typeb"], "cross_layer": ["-layer_cka", "-commit_layer"],
            "invariant_cfg": {"margin_typeb": {"cut": 0.7, "cuts": [0.0, 0.5, 0.7], "min_n": 10, "b1": True,
                                               "strat_bins": 10, "strat_min": 5,
                                               "legacy_imagenet": {"cut": 0.5, "min_class_n": 3, "draws": 5,
                                                                   "seed": 0, "layers": ["penult"]}}},
            "outputs": {"root": os.path.join(root, "out"), "plots": False}}


def _tf32px():
    import torchvision.transforms as T
    return T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(), T.Normalize(*ei.IMAGENET_NORM)])


def test_selftest_random_with_a_tiny_vit():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    cfg = _vit_cfg("/nonexistent", [])
    rep = ei.selftest_random(cfg, "cpu", model_override=(_tiny_tv_vit(), None, {"weights_sha256": "0" * 64}), size=32)
    assert rep["status"] == "PASS", rep["checks"]
    assert rep["taps"] == ["block.0", "block.1", "block.2", "penult_mean", "penult"] and "n_taps_14" not in rep["checks"]
    assert rep["values"]["penult_vs_final_norm_cls_max_abs"] == 0.0


# ---- 4: extract_imagenet end to end ------------------------------------------------------------------------------
def test_extract_imagenet_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import b1_gate
    vol, files, labels, _, real = _fake_volume(str(tmp_path))
    rp = os.path.join(vol, "datasets", "imagenet_val", "real.json")
    monkeypatch.setattr(ei, "_real_spec", lambda: (os.path.getsize(rp), file_sha256(rp, 64)))
    monkeypatch.setenv("ATLAS_EXTRACT_WORKERS", "2")                  # the forked DataLoader path, order checked
    cfg = _vit_cfg(vol, files)
    dump = os.path.join(vol, "out", "dump")
    ei.extract_imagenet(cfg, vol, device="cpu", dump=dump, quiet=True,
                        model_override=(_tiny_tv_vit(num_classes=4), _tf32px(), {"weights_sha256": "0" * 64}))
    meta = json.load(open(os.path.join(dump, "meta.json")))
    A, B = ei.split_parts(labels, 0, 25, 4)
    assert meta["layers"] == ["block.0", "block.1", "block.2", "penult_mean", "penult"] and meta["splits"] == ["ref", "test"]
    assert meta["ref_indices"] == A.tolist() and meta["test_indices"] == B.tolist() and meta["n_test"] == 100
    assert meta["per_class_counts"] == {"ref": [25, 25], "test": [25, 25]} and meta["ref_test_disjoint"] is True
    assert meta["decode_failures"] == 0 and meta["parquet_sha256_ok"] is True and meta["loader_workers"] == 2
    assert meta["head_check_max_abs"] <= 1e-4 and meta["penult_vs_final_norm_cls_max_abs"] == 0.0
    assert meta["real_labels"]["used"] is True and meta["logits_stored"] is True and meta["norm"] == "imagenet"
    assert meta["dtype"] == "float32" and np.load(os.path.join(dump, "acts", "penult", "test.npy")).dtype == np.float32
    p = np.load(os.path.join(dump, "preds", "test.npz"))
    assert {"argmax", "maxprob", "logit_gap", "logit_top1", "logit_top2", "real_ok"} <= set(p.files)
    assert np.allclose(p["logit_gap"], p["logit_top1"] - p["logit_top2"]) and (p["logit_gap"] >= 0).all()
    want = np.array([-1 if not real[r] else int(int(a) in real[r]) for r, a in zip(B, p["argmax"])])
    assert np.array_equal(p["real_ok"], want) and p["real_ok"].dtype == np.int8
    lg = np.load(os.path.join(dump, "logits", "test.npy"))
    assert lg.shape == (100, 4) and lg.dtype == np.float16 and np.allclose(lg.max(1), p["logit_top1"], atol=2e-2)
    assert np.array_equal(np.load(os.path.join(dump, "labels", "test.npy")), labels[B])
    atlas = build_atlas(dump, os.path.join(vol, "out"), cfg, verbose=False)
    assert not atlas["skipped"]
    fails = []
    pm = atlas["per_layer"]["penult"]["margin_typeb"]
    b1_gate._p0("e2e", atlas, pm, False, fails, model="vitb16", n_classes=4, n_test=100, per_class=25)
    assert [f for f in fails if "acc band" not in f] == [], fails    # a random tiny model is outside the ImageNet band
    for l in atlas["layers"]:
        m = atlas["per_layer"][l]["margin_typeb"]
        assert "error" not in m, (l, m)
        assert "margin_minus_logitgap_typeb" in m and "spearman_margin_logitgap" in m and m["b1"]["real_ok"] is True
        assert all("n_typeb_real_ok" in r and "margin_minus_maxprob_confmatched" in r for r in m["sweep"])
        assert ("legacy_imagenet" in m) == (l == "penult")


def test_extract_imagenet_legacy_block(tmp_path):
    pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    vol, files, labels, _, _ = _fake_volume(str(tmp_path))
    cfg = _vit_cfg(vol, files, legacy=True, arch="tv_resnet50")
    dump = os.path.join(vol, "out", "dump")
    model = tvm.resnet18(weights=None, num_classes=4).eval()
    ei.extract_imagenet(cfg, vol, device="cpu", dump=dump, quiet=True, model_override=(model, _tf32px(), {}))
    meta = json.load(open(os.path.join(dump, "meta.json")))
    L = _literal_legacy_rows(vol, files, 100)
    assert meta["n_test"] == L == meta["legacy_loop"]["n_rows"] and meta["ref_indices"] == meta["test_indices"] == list(range(L))
    assert meta["layers"] == ["stem", "layer1", "layer2", "layer3", "layer4", "penult"]
    assert meta["ref_test_disjoint"] is False and meta["penult_vs_final_norm_cls_max_abs"] is None
    assert meta["real_labels"]["used"] is False                      # the fake real.json is not the pinned file
    assert "real_ok" not in np.load(os.path.join(dump, "preds", "test.npz")).files
    assert np.array_equal(np.load(os.path.join(dump, "labels", "ref.npy")), labels[:L])


# ---- 5: gate known answers (shared fixtures) -----------------------------------------------------------------------
def test_b1_gate_known_answers():
    import b1_gate
    exp = json.load(open(os.path.join(FX, "expected.json")))
    assert len(exp) >= 20
    for sc, e in exp.items():
        res = os.path.join(FX, sc, "results")
        rep = b1_gate.evaluate(res, os.path.join(res, "instrument_check_b1"))
        assert rep["gate"] == {k: e["gate"][k] for k in ("G0", "G1", "G2", "G3")}, (sc, rep["gate"], rep["plumbing_failures"])
        assert rep["open"] == e["gate"]["open"], sc
        pod = json.load(open(os.path.join(res, "b1_gate", "gate.json")))
        assert (pod["gate"] == rep["gate"]) == e.get("pod_record_agrees", True), sc


def test_b1_gate_record_reuse(tmp_path):
    """gate.json is written once; a relaunch reuses it (exit 0 iff open) while both ResNet50 atlas.json are unchanged."""
    res = os.path.join(FX, "A", "results")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "b1_gate.py"), "--out", str(tmp_path / "g"), "--root", res,
           "--check-dir", os.path.join(res, "instrument_check_b1")]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    rec = json.load(open(tmp_path / "g" / "gate.json"))
    assert rec["open"] is True and set(rec["sources"]) == {"margin_b1_resnet50_legacy10k", "margin_b1_resnet50"}
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "reusing" in r.stdout
    rec["sources"]["margin_b1_resnet50"] = "0" * 64                    # a rebuilt resnet50 atlas: no reuse
    json.dump(rec, open(tmp_path / "g" / "gate.json", "w"))
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 1
    res = os.path.join(FX, "CTRL", "results")                            # a closed gate exits 1
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "b1_gate.py"), "--out", str(tmp_path / "c"), "--root", res,
           "--check-dir", os.path.join(res, "instrument_check_b1")]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 1
