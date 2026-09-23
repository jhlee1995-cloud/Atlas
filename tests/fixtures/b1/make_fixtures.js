// make_fixtures.js -- writes the B1 decision-code fixtures: synthetic atlas.json files (no real run, no real number),
// one results tree per scenario, plus the pod-side records the decision code reads (b1_gate/gate.json,
// instrument_check_b1/{data.json, selftest_*.json, e9_identity.json}, critic_b1_vit_pair/critic.json) and A4b M56-b
// stand-ins (a4b/*.json). expected.json holds what each scenario must give.
//   node tests/fixtures/b1/make_fixtures.js          (rewrites every scenario; deterministic)
// Shared by tests/test_b1_imagenet.py (scripts/b1_gate.py known answers: the gate flags) and
// tests/fixtures/b1/check_fixtures.js (scripts/b1_verdicts.js: gate, outcome, labels, row 10, D13 joint reading).
"use strict";
const fs = require("fs"), path = require("path");
const HERE = __dirname;
const wr = (p, o) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, JSON.stringify(o) + "\n"); };

// ---- building blocks -------------------------------------------------------------------------------------------
const meta = (legacy, o = {}) => ({ source: "real", n_classes: 1000, decode_failures: 0, head_check_max_abs: 2e-5,
  parquet_sha256_ok: true, n_test: legacy ? 10048 : 25000,
  per_class_counts: legacy ? { ref: [2, 21], test: [2, 21] } : { ref: [25, 25], test: [25, 25] },
  ref_test_disjoint: !legacy, head_center_cos: 0.9, norm: "imagenet", git_commit: "fixture", created: "2026-09-23 00:00:00", ...o });
// V-state building blocks (SESOI 0.01, CI 0.02, P 0.05; median_margin_correct 4.0)
const PASS1 = { d: 0.05, se: 0.006, p: 1e-8 }, NULL1 = { d: 0.002, se: 0.003, p: 0.5 }, REV1 = { d: -0.02, se: 0.004, p: 1e-6 };
const WIDE1 = { d: 0.005, se: 0.009, p: 0.58 }, WEAK1 = { d: 0.005, se: 0.004, p: 0.2 };
const V2P = { mt: 1.0 }, V2F = { mt: 3.0 };
const EQ3 = { d3: 0.002, se3: 0.003, p3: 0.5 }, ADDS3 = { d3: 0.02, se3: 0.004, p3: 1e-6 }, EQ3B = { d3b: 0.001, se3b: 0.003, p3b: 0.7 };
const GOOD = { n7: 500, n5: 1000, ...PASS1, ...V2P, ...EQ3, ...EQ3B };
function srow(c, o, n) {
  const auc = o.auc ?? 0.85;
  return { cut: c, n_typeb: n, median_margin_typeb: o.mt ?? null, auc_margin_typeb: auc, auc_dist_typeb: auc - (o.d ?? 0),
    auc_maxprob_typeb: 0.84, auc_energy_typeb: o.en ?? 0.55, margin_minus_dist_typeb: o.d ?? null, margin_minus_dist_typeb_se: o.se ?? null,
    margin_minus_dist_typeb_p: o.p ?? null, margin_minus_maxprob_confmatched: o.d3 ?? null, margin_minus_maxprob_confmatched_se: o.se3 ?? null,
    margin_minus_maxprob_confmatched_p: o.p3 ?? null, margin_minus_logitgap_confmatched: o.d3b ?? null,
    margin_minus_logitgap_confmatched_se: o.se3b ?? null, margin_minus_logitgap_confmatched_p: o.p3b ?? null,
    margin_norm_minus_dist_typeb: 0.012, margin_norm_minus_dist_typeb_p: 0.001, median_margin_correct_confmatched: 3.0,
    strat_confmatched: { bins: 10, min_bin: 5, n_bins_used: 8, auc_margin: 0.70, auc_maxprob: 0.705 },
    auc_margin_typeb_realwrong: auc + 0.01, n_pos_typeb_realwrong: Math.round(n * 0.6), n_typeb_real_ok: Math.round(n * 0.3),
    median_margin_typeb_realwrong: o.mt ?? null };
}
const pm = (acc, o, model) => ({ acc, median_margin_correct: 4.0, spearman_margin_maxprob: 0.90, spearman_margin_logitgap: 0.93,
  d1_plus_d2_cv: model === "resnet50" ? 0.20 : 0.10, nearest_center_agrees_with_model: 0.80,
  centers_geometry: { n_centers: 1000, sep_ratio_ref: 1.3, sep_legacy_ref: 1.2 },
  margin_minus_maxprob_confmatched: o.d3 ?? null, sweep: [srow(0.7, o, o.n7), srow(0.5, o, o.n5)] });
const LEGACY_IMAGENET = { acc: 0.7712, valley_sep_legacy: 1.139, n_cw: 900, n_classes_centered: 1000, dir_auc_margin_full: 0.806,
  raw_auc_margin_full: 0.194, dir_auc_margin_sub_sd: 0.008, dir_auc_cluster_full: 0.64, dir_auc_cluster_sub_sd: 0.009,
  dir_auc_energy_full: 0.55 };
const R50_LAYERS = ["stem", "layer2.3", "penult"];
const VIT_LAYERS = ["block.0", "block.5", "penult_mean", "penult"];
function atlas(model, acc, o, mo = {}) {
  const layers = model === "resnet50" ? R50_LAYERS : VIT_LAYERS, per = {};
  for (const l of layers) {
    const early = l !== "penult" && l !== "penult_mean";
    per[l] = { margin_typeb: pm(acc, early ? { ...o, auc: 0.70, d: 0.0, p: 0.9 } : l === "penult_mean" ? { ...o, auc: 0.83 } : o, model) };
  }
  if (model === "resnet50") per.penult.margin_typeb.legacy_imagenet = { ...LEGACY_IMAGENET, raw_auc_margin_full: 0.12 };
  return { exp_id: null, source: "real", meta: meta(false, mo), layers, per_layer: per };
}
const legacy = (lo = {}, mo = {}) => ({ source: "real", meta: meta(true, mo), layers: ["stem", "layer1", "layer2", "layer3", "layer4", "penult"],
  per_layer: { penult: { margin_typeb: { acc: 0.771, legacy_imagenet: { ...LEGACY_IMAGENET, ...lo } } } } });
const critic = (runs, st = "PASS") => ({ runs, verdict: "fixture", checks: {
  synthetic_refusal: [{ name: "sources", status: "PASS", detail: "all runs real" }],
  holdout_hygiene: runs.map(r => ({ name: `holdout:${r}`, status: "PASS", detail: "declared" })),
  scalar_stability: ["auc_margin_typeb", "auc_dist_typeb", "auc_maxprob_typeb", "auc_margin_wrong", "median_margin_ratio_typeb"]
    .map(k => ({ name: `penult/margin_typeb.${k}`, status: st, detail: "fixture" })) } });

// ---- scenarios -----------------------------------------------------------------------------------------------------
// each: r50 (spec), vit {vitb16, vitb16_swap, deitb, deitb_swap: [acc, spec, meta override]}, legacy overrides, records
const VIT = (spec, acc = { vitb16: 0.80, deitb: 0.805 }) => Object.fromEntries(["vitb16", "vitb16_swap", "deitb", "deitb_swap"]
  .map(n => [n, [acc[n.replace(/_swap$/, "")], spec, {}]]));
const OPEN = { G0: true, G1: true, G2: true, G3: true, open: true };
const S = {
  A:        { vit: VIT(GOOD), e9: true, exp: { outcome: "A", cstar: 0.7, margin_vs_maxprob: "MARGIN~CONFIDENCE", margin_vs_logitgap: "MARGIN~CONFIDENCE",
              V4_replication: "PASS", accuracy_control: "ACC-MATCHED", row10: "ok", joint: { reject_notdepth: 2, promote: 2 } } },
  LEGDEC:   { vit: VIT(GOOD), legacyMeta: { decode_failures: 2 }, exp: { outcome: "A", cstar: 0.7, row10: "ok" } },
  B:        { vit: VIT({ ...GOOD, ...NULL1 }), exp: { outcome: "B (type-b still low-margin)", cstar: 0.7, row10: "x", b2_cancel: false,
              joint: { promote: 3, yellow: 3, reject_notdepth: 4, reject_depth: 5, reject_null: 5, noteval: null } } },
  BREV:     { vit: VIT({ ...GOOD, ...REV1, ...V2F }), exp: { outcome: "B (not low-margin)", cstar: 0.7, row10: "x", b2_cancel: true,
              joint: { promote: 3 } } },
  POW:      { r50: { ...GOOD, n7: 150, n5: 700 }, vit: VIT({ ...GOOD, n7: 150, n5: 700 }), exp: { outcome: "A", cstar: 0.5, row10: "ok" } },
  NOPOW:    { vit: VIT({ ...GOOD, n7: 150, n5: 250 }), exp: { outcome: "UNDECIDED-POWER", cstar: null, row10: "open" } },
  FRAG:     { vit: { ...VIT(GOOD), vitb16_swap: [0.80, { ...GOOD, ...NULL1 }, {}] }, exp: { outcome: "UNDECIDED", cstar: 0.7, row10: "y",
              states: { vitb16: { V1: "SPLIT-FRAGILE" } } } },
  UNTESTED: { vit: { ...VIT(GOOD), deitb_swap: [0.805, { ...GOOD, n7: 5, n5: 9, d: null, se: null, p: null }, {}] },
              exp: { outcome: "UNDECIDED", cstar: 0.7, states: { deitb: { V1: "SPLIT-UNTESTED" } } } },
  CTRL:     { r50: { ...GOOD, ...WEAK1 }, vit: null, gate: { G3: false, open: false }, exp: { outcome: "INVALID-CONTROL", row10: "open",
              joint: { reject_notdepth: 1 }, joint_cnn_sentence: true } },
  REPRO:    { legacy: { dir_auc_margin_full: 0.70, raw_auc_margin_full: 0.30 }, vit: null, gate: { G1: false, open: false },
              exp: { outcome: "INVALID-REPRO" } },
  G0SELF:   { selftest: { margin_b1_deitb: "FAIL" }, vit: null, gate: { G0: false, open: false }, exp: { outcome: "INVALID-PLUMBING (gate G0)" } },
  REALFRAC: { data: { status: "PASS", real: { ok: true, label_in_real_frac: 0.001 } }, vit: null, gate: { G0: false, open: false },
              exp: { outcome: "INVALID-PLUMBING (gate G0)" } },
  PP:       { vit: { ...VIT(GOOD), vitb16: [0.80, GOOD, { head_check_max_abs: 0.01 }] }, exp: { outcome: "INVALID-PLUMBING (post-gate run)" } },
  R2:       { vit: { ...VIT(GOOD), vitb16: [0.80, GOOD, { decode_failures: 3 }], vitb16_r2: [0.80, GOOD, {}] },
              exp: { outcome: "A", cstar: 0.7, rerun: { vitb16: "PASS" } } },
  ACCONLY:  { vit: { ...VIT(GOOD), vitb16: [0.70, GOOD, {}], vitb16_r2: [0.80, GOOD, {}] },
              exp: { outcome: "INVALID-PLUMBING (post-gate run)", rerun: { vitb16: "REFUSED" } } },
  BAND:     { vit: VIT(GOOD, { vitb16: 0.73, deitb: 0.805 }), exp: { outcome: "A", cstar: 0.7, accuracy_control: "ACC-CONFOUNDED" } },
  ADDSONLY: { vit: VIT({ ...GOOD, ...ADDS3 }), exp: { outcome: "A", margin_vs_maxprob: "ADDS-OVER-MAXPROB-ONLY", margin_vs_logitgap: "MARGIN~CONFIDENCE" } },
  MIXED:    { vit: { ...VIT(GOOD), deitb: [0.805, { ...GOOD, ...NULL1 }, {}], deitb_swap: [0.805, { ...GOOD, ...NULL1 }, {}] },
              exp: { outcome: "MIXED", cstar: 0.7, row10: "y", V4_replication: "FAIL" } },
  GATEREC:  { vit: VIT(GOOD), podGate: { G3: false, open: false }, exp: { outcome: "INVALID-PLUMBING (gate record)", pod_record_agrees: false } },
  R50SWAP:  { vit: VIT(GOOD), r50swapMeta: { head_check_max_abs: 0.01 }, exp: { outcome: "A", states: { resnet50: { V1: "SPLIT-UNTESTED" } },
              r50swap_info: true } },
  NULLWIDE: { vit: VIT({ ...GOOD, ...WIDE1 }), exp: { outcome: "UNDECIDED", states: { vitb16: { V1: "UNRESOLVED" }, deitb: { V1: "UNRESOLVED" } } } },
};

// ---- write -----------------------------------------------------------------------------------------------------------
const expected = {};
for (const [sc, s] of Object.entries(S)) {
  const root = path.join(HERE, sc);
  fs.rmSync(root, { recursive: true, force: true });
  const res = path.join(root, "results"), A = n => path.join(res, n, "atlas.json");
  wr(A("margin_b1_resnet50_legacy10k"), legacy(s.legacy || {}, s.legacyMeta || {}));
  wr(A("margin_b1_resnet50"), atlas("resnet50", 0.79, s.r50 || GOOD));
  wr(A("margin_b1_resnet50_swap"), atlas("resnet50", 0.79, s.r50 || GOOD, s.r50swapMeta || {}));
  for (const [n, [acc, spec, mo]] of Object.entries(s.vit || {})) wr(A(`margin_b1_${n}`), atlas("vit", acc, spec, mo));
  const I = path.join(res, "instrument_check_b1");
  wr(path.join(I, "data.json"), s.data || { status: "PASS", real: { ok: true, label_in_real_frac: 0.93, n_nonempty: 46837 } });
  for (const e of ["margin_b1_vitb16", "margin_b1_deitb"]) wr(path.join(I, `selftest_${e}.json`), { status: (s.selftest || {})[e] || "PASS" });
  const gate = { ...OPEN, ...(s.gate || {}) };
  const pod = { ...gate, ...(s.podGate || {}) };
  wr(path.join(res, "b1_gate", "gate.json"), { gate: { G0: pod.G0, G1: pod.G1, G2: pod.G2, G3: pod.G3 }, open: pod.open,
    check_dir: "results/instrument_check_b1", sources: {}, b1_acc_info: null });
  if (s.vit) wr(path.join(res, "critic_b1_vit_pair", "critic.json"), critic(["results/margin_b1_vitb16", "results/margin_b1_deitb"]));
  if (s.e9) {
    for (const x of ["resnet20_s0hub_st3", "resnet56_s0hub_st3", "resnet56_s1", "resnet56_s2"])
      wr(A(`margin_b1_${x}`), { source: "real", meta: { source: "real", n_classes: 10 }, layers: ["penult"],
        per_layer: { penult: { margin_typeb: { margin_minus_maxprob_confmatched: 0.004, margin_minus_maxprob_confmatched_se: 0.003,
          margin_minus_maxprob_confmatched_p: 0.18, margin_minus_maxprob_typeb: 0.001 } } } });
    wr(path.join(I, "e9_identity.json"), { resnet56_s1: { n_leaves: 100, n_exact: 100, mismatch_first20: [] } });
  }
  expected[sc] = { gate, ...s.exp };
}
const a4b = (tag, attr) => ({ M56: { b_row9_d56: { tag, m2_failure_attribution: attr, m2_failure_attribution_reason: "fixture" } } });
wr(path.join(HERE, "a4b", "promote.json"), a4b("d56 PROMOTE", null));
wr(path.join(HERE, "a4b", "yellow.json"), a4b("d56 YELLOW", null));
wr(path.join(HERE, "a4b", "reject_notdepth.json"), a4b("d56 REJECT", "NOT-DEPTH"));
wr(path.join(HERE, "a4b", "reject_depth.json"), a4b("d56 REJECT", "DEPTH"));
wr(path.join(HERE, "a4b", "reject_null.json"), a4b("d56 REJECT", null));
wr(path.join(HERE, "a4b", "noteval.json"), a4b("NOT_EVALUABLE (R0 FAIL)", null));
fs.writeFileSync(path.join(HERE, "expected.json"), JSON.stringify(expected, null, 1) + "\n");
console.log(`[b1 fixtures] ${Object.keys(S).length} scenarios -> ${HERE}`);
