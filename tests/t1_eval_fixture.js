#!/usr/bin/env node
// tests/t1_eval_fixture.js -- known-answer test of scripts/t1_eval.js (node only; runs on Windows, no pod, no Python).
// Builds synthetic results trees in a temp directory whose numbers are CONSTRUCTED to give known labels and checks
// (T1 review G; docs/plans/B4_INTEGRATION.md D8 step 2):
//   design   every claim gets its predicted label                      (confirmation, dry run)
//   flip     every claim gets the opposite definitive label            (confirmation, dry run)
//   strip    every claim is NOT_EVALUABLE when its fields are removed
//   power    every three-way claim is INCONCLUSIVE when the unit calls are
//   non-dry  in-process with an injected fake git: every gate PASSes, labels carry no prefix; a wrong code hash fails
//            provenance, a missing streams self-test fails the self-test gate (a .log in the check dir is ignored: T1
//            review A1), a replay FAIL tags [REPLAY-DRIFT] and nulls a near-threshold unit, a replay whose drift the
//            record cannot bound (a killed t1 unit, a non-numeric diff, more diffs than listed, no replay.json) tags
//            every label and nulls every decision, a withdrawn claim is INFO, a looser fraction in the rules is refused,
//            the anchor gate fails on a perturbed committed number (DO-3 sparse fraction: 0.03 passes, 0.04 fails)
//   discovery  labels 'INFO (discovery) ...', --emit-rules writes the rules, --amend refuses a looser fraction
//   CLI      a dry run writes the JSON and refuses to overwrite it; a non-dry CLI run on a non-git fixture repo fails its
//            provenance and rules gates (every label NOT_EVALUABLE (gate: ...)); the model_outcomes schema gate
// No commit is made anywhere: the non-dry PASS path uses a fake git adapter passed to evaluate().
//   node tests/t1_eval_fixture.js [--keep]
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const cp = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const E = require(path.join(ROOT, "scripts", "t1_eval.js"));
const S = require(path.join(ROOT, "scripts", "b4_stats.js"));
const EVAL_JS = path.join(ROOT, "scripts", "t1_eval.js");
const REG = JSON.parse(fs.readFileSync(path.join(ROOT, "experiments/b4/models.json"), "utf8"));
const OSCHEMA = JSON.parse(fs.readFileSync(path.join(ROOT, "experiments/b4/model_outcomes.schema.json"), "utf8"));
const DISC = ["gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast",
  "pixelate", "jpeg_compression"];
const HEAD = ["msp", "maxlogit", "gap", "energy", "entropy", "gini", "smargin", "pnorm2", "mspT"];
const KEEP = process.argv.includes("--keep");
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "t1_eval_fx_"));
let nchk = 0, nbad = 0;
const check = (name, ok, detail = "") => { nchk++; if (!ok) { nbad++; console.log(`FAIL ${name} ${detail}`); } };
const roles = id => REG.models.find(m => m.id === id).roles;
const ids = (...rs) => REG.models.filter(m => rs.some(r => m.roles.includes(r))).map(m => m.id);
const spec = id => REG.models.find(m => m.id === id);

// ---- constructed joint records ----------------------------------------------------------------------------------------------
const A = (x = {}) => ({ auc: 0.95, dauc: 0.05, dauc_ci: [0.03, 0.07], dI_bits: 0.02, dI_ci: [0.01, 0.03], dtpr5: 0.1, daurc: -0.01, ...x });
const Bd = (x = {}) => ({ auc: 0.9, dauc: 0.001, dauc_ci: [-0.005, 0.008], dI_bits: 0.0, dI_ci: [-0.002, 0.001], dtpr5: 0.0, daurc: 0.0, ...x });
const In = (x = {}) => ({ auc: 0.9, dauc: 0.005, dauc_ci: [-0.01, 0.03], dI_bits: 0.0005, dI_ci: [-0.001, 0.002], dtpr5: 0.01, daurc: 0.0, ...x });
const PA = { dauc_pix: 0.05, dauc_pix_ci: [0.03, 0.07], dI_pix_ci: [0.01, 0.03] };
const PB = { dauc_pix: 0.001, dauc_pix_ci: [-0.005, 0.008], dI_pix_ci: [-0.002, 0.001] };
const PI = { dauc_pix: 0.003, dauc_pix_ci: [-0.01, 0.03], dI_pix_ci: [-0.001, 0.002] };
const law = nc1 => 0.0224 - 0.108 * Math.log10(nc1);

function outcomes(id, phase) {
  return { schema: "b4_model_outcomes/1", unit: id, layout: "fit", phase,
    O1: { value: 0.001, ci95: [-0.004, 0.006], dI_bits: 0.0001, dI_ci95: [-0.001, 0.001], call: "BOUNDED", bundle: "penult",
      bundle_members: OSCHEMA.$defs.O1.properties.bundle_members.const, target: "err_clean", rows: [0, 5000], n_pos: 350,
      n_neg: 4650, nboot: phase === "discovery" ? 200 : 1000 },
    O5: { value: 0.1, x4_auroc_mean: 0.7, best_head: "gap", best_head_auroc_mean: 0.6, severity: 3, corruptions: DISC,
      negatives: [3500, 5000], per_corruption: Object.fromEntries(DISC.map(c => [c, { x4_auroc: 0.7,
        head_auroc: { msp: 0.6, maxlogit: 0.6, gap: 0.6, energy: 0.6, entropy: 0.6 } }])) } };
}

function board(id, layout, phase, mode, prov) {
  const f = mode === "flip", pw = mode === "power";
  const nboot = phase === "discovery" ? 200 : 1000;
  const sealed = !!(spec(id).layouts[layout] || {}).sealed;
  const hdr = { program: "t1", unit: id, layout, phase, tag: "", nboot, code: { sha256: prov.t1, core_sha256: prov.core,
    repo_commit: phase === "discovery" ? "S1HEAD000" : "S2HEAD000" }, env: { unseal: phase === "confirmation" ? "P2" : null },
  dumps: [{ path: spec(id).layouts[layout].dump, sealed, splits_read: ["test", "corrupt__fog__s3"] }] };
  if (layout === "eval") hdr.dumps.unshift({ path: spec(id).layouts.fit.dump, sealed: !!spec(id).layouts.fit.sealed, splits_read: ["ref", "test"] });
  const out = { ...hdr, schema: "b4_t1_scoreboard/1" };
  if (layout === "fit") out.model_outcomes = outcomes(id, phase);
  if (mode === "strip") return { ...out, persample: {}, head: {} };
  const nc1 = 0.05 + 0.1 * (id.length % 5) / 5;
  const j = (a, b, c) => (pw ? In(c) : f ? b : a);
  const uni = Object.fromEntries(HEAD.map(s => [s, { auc: 0.91 }]));
  uni.gap = { auc: f ? 0.90 : 0.92 };
  uni.margin_penult = { auc: 0.9 };
  const kinds = phase === "discovery" ? ["deadpix", "occlusion_disc"] : ["deadpix", "occlusion_disc", "exposure_global"];
  const faults = Object.fromEntries(kinds.map(k => [`fault_${k}`, { univariate: { ...Object.fromEntries(HEAD.map(s => [s, { auc: 0.55 }])),
    knnL2_stem: { auc: f ? 0.56 : 0.8 } }, joint: { early: A() } }]));
  const sp10 = [];
  for (const c of ["gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "pixelate"]) for (const s of [3, 5]) sp10.push(`corrupt__${c}__s${s}`);
  const cal = Object.fromEntries(HEAD.map(s => [s, { tpr: Object.fromEntries(sp10.map(sp => [sp, 0.6])) }]));
  const ps = {
    err_clean: { n_pos: 350, n_neg: 4650, univariate: uni, joint: { margin_pen: j(Bd(), A(), {}), all_geom: j(Bd(), A(), {}),
      local_pen: j(Bd(), A(), {}), traj: Bd(), penult: Bd() } },
    err_shift_s3: { joint: { pre: j(A(), Bd(), {}), early: j(A(PB), A(PA), PI), traj: j(Bd(), A(), {}) } },
    ood_svhn: { joint: { x4: j(A(), Bd(), {}) } },
    corrupt_holdout_s3_transfer: { joint: { x4: { dauc: f ? 0.0 : 0.05 } } },
    corrupt_extra_s3_transfer: { joint: { x4: { dauc: f ? 0.0 : 0.05 } } },
    ...faults,
  };
  for (const F of ["N", "B", "W", "D"]) {
    ps[`famid_${F}_s3`] = { joint: { early: j(A(PB), Bd(PA), PI) } };
    ps[`corrupt_${F}_s3`] = { joint: { null_pen: j(Bd(), A(), {}), ...(F === "N" || F === "B" ? { x4: j(A(), Bd(), {}) } : {}) } };
  }
  const b64 = {
    estimators: { ATC_MC: { mae_pp_loso: 2.0 }, H: { mae_pp_loso: f ? 1.5 : 3.0 } }, partial_H_given_ATC_mean: f ? 0.4 : 0.1,
    skew: { single_class: { R_pre: { fpr_max: f ? 0.5 : 0.12, fpr_mean: 0.06 }, T2_penult: { fpr_mean: 0.9 }, H: { fpr_mean: 0.05 } },
      "dirichlet_0.1": { bbsdh_chi2: 0.8 } },
    typing: { macro: { geometry_acc: f ? 0.6 : 0.9, head_acc: 0.7 } },
    by_split: Object.fromEntries(["motion_blur", "pixelate", "snow"].map(c => [`corrupt__${c}__s1`, { flag_rate: { R_pre: f ? 0.5 : 0.95 } }])),
  };
  return { ...out, head: { best_stat: "gap" }, persample: ps, calibration: cal, batch: { sizes: { 64: b64 } },
    x4: { fpr_B: f ? 0.09 : 0.05, n_cal_fusion: layout === "eval" ? 2500 : 750, n_B: 1500,
      tpr: Object.fromEntries(sp10.map(sp => [sp, f ? 0.6 : 0.9])) },
    do3: { sparse_frac_test: law(nc1) + (f ? 0.1 : 0.005), nc1_train: nc1 },
    cost: { ref_bytes: { knn_bank_penult: 2560000 }, timing_s: { ms_per_1000_queries: { knnL2_penult: 12.5 } } } };
}

function streams(id, phase, mode, prov) {
  const f = mode === "flip";
  const hdr = { program: "t1s", unit: id, layout: null, phase, tag: "", nboot: phase === "discovery" ? 200 : 1000,
    code: { sha256: prov.t1s, core_sha256: prov.core, repo_commit: phase === "discovery" ? "S1HEAD000" : "S2HEAD000" },
    env: { unseal: phase === "confirmation" ? "P2" : null }, dumps: [], schema: "b4_t1_streams/1" };
  if (mode === "strip") return hdr;
  const rate = {}, sc = {};
  for (const c of DISC) rate[`corrupt__${c}__s1`] = { s1end: { I_nats: 1.0, ci: f ? [0.05, 0.15] : [0.8, 1.2] }, penult: { I_nats: 0.1, ci: f ? [0.8, 1.2] : [0.05, 0.15] } };
  for (const c of ["motion_blur", "pixelate", "snow"]) sc[`step|corrupt__${c}__s1`] = { mewma_s1end: { delay_cens: f ? 50 : 5 }, cusum_msp: { delay_cens: 100 } };
  for (const c of ["motion_blur", "defocus_blur"]) sc[`ramp|${c}`] = { mewma_s1end: { lead_median: f ? 50 : 300 } };
  sc["stuck|frame"] = { mewma_penult: { p_det_within: f ? 0.5 : 0.99 }, mewma_stem: { p_det_within: 0.98 }, stuck: { p_det_within: 1.0 } };
  sc["ramp|brightness"] = { cusum_knnL2_stem: { p_det_within_gated: f ? 0.5 : 0.1, p_det_within: 0.95 }, mewma_stem: { p_det_within_gated: 0.05, p_det_within: 0.99 }, x4: { p_det_within_gated: 0.1 } };
  // X3-7: the calibrated flag's own in-control alarm rate within 500 frames is ~0.2 (ARL0 2000), in the null run and under
  // a skew it cannot see; the flip plants a skew rate 0.28 above the null
  sc["skew|single_class"] = { bbsdh: { p_det_within: 0.99 }, mewma_penult: { p_det_within: 0.97 }, x4: { p_det_within: f ? 0.5 : 0.22 } };
  sc["null|clean"] = { bbsdh: { p_det_within: 0.2 }, mewma_penult: { p_det_within: 0.21 }, x4: { p_det_within: 0.22 } };
  return { ...hdr, rate, scenarios: sc, arl: { cusum_msp: { arl0_heldout: f ? 500 : 2100, fa_per_1000_heldout: 0.48 }, mewma_stem: { arl0_heldout: f ? 9000 : 1900, fa_per_1000_heldout: 0.53 } } };
}

const w = (dir, name, obj) => { fs.mkdirSync(dir, { recursive: true }); fs.writeFileSync(path.join(dir, name), typeof obj === "string" ? obj : JSON.stringify(obj)); };

function makeRepo(name, mode, prov, opt = {}) {
  const repo = path.join(TMP, name), res = path.join(repo, "results");
  for (const f of E.T1_FILES) if (fs.existsSync(path.join(ROOT, f))) { fs.mkdirSync(path.dirname(path.join(repo, f)), { recursive: true }); fs.copyFileSync(path.join(ROOT, f), path.join(repo, f)); }
  for (const f of ["experiments/b4/models.json", "experiments/b4/model_outcomes.schema.json"]) { fs.mkdirSync(path.dirname(path.join(repo, f)), { recursive: true }); fs.copyFileSync(path.join(ROOT, f), path.join(repo, f)); }
  const put = (id, layout, phase) => w(path.join(res, "b4_t1", `${id}_${layout}`), "scoreboard.json", board(id, layout, phase, mode, prov));
  const disc = "design";                   // the anchors compare the discovery boards with the committed numbers
  for (const id of ids("D", "N", "Dnew")) w(path.join(res, "b4_t1", `${id}_fit`), "scoreboard.json", board(id, "fit", "discovery", opt.discMode || disc, prov));
  for (const id of ids("STdisc")) w(path.join(res, "b4_t1s", id), "streams.json", streams(id, "discovery", opt.discMode || disc, prov));
  if (!opt.discoveryOnly) {
    for (const id of ids("F", "F20")) { put(id, "fit", "confirmation"); put(id, "eval", "confirmation"); }
    for (const id of ids("C", "K")) put(id, "fit", "confirmation");
    for (const id of ids("R2")) put(id, "eval", "confirmation");
    for (const id of ids("STconf")) w(path.join(res, "b4_t1s", id), "streams.json", streams(id, "confirmation", mode, prov));
  }
  for (const [d, rep] of [["instrument_check_b4s1", null], ["instrument_check_b4s2", opt.replay || { status: "PASS", units: {
    "t1:resnet20_hub": { status: "PASS", n_diff: 0, first20: [] }, "t1:resnet56_hub": { status: "PASS", n_diff: 0, first20: [] },
    "t2:resnet20_hub": { status: "PASS", n_diff: 0, first20: [] } } }]]) {
    const dd = path.join(res, d);
    w(dd, "selftest_t1_scoreboard.json", { status: "PASS" });
    if (!opt.noStreamsSelftest) w(dd, "selftest_t1_streams.json", { status: "PASS" });
    w(dd, "selftest_b4_extract.json", { status: "PASS" });
    w(dd, "selftest_extract.log", "plain text, not JSON (T1 review A1)\n");
    if (rep) w(dd, "replay.json", rep);
  }
  // committed Atlas anchors (the T1 anchor gate reads them), consistent with the discovery boards
  for (const m of REG.models.filter(x => x.anchor && x.roles.includes("D"))) {
    const b = board(m.id, "fit", "discovery", "design", prov), rel = m.anchor.replace(/^results\//, "");
    const dsp = opt.perturbSparse && opt.perturbSparse.id === m.id ? opt.perturbSparse.delta : 0.005;
    w(path.join(res, rel), "atlas.json", { per_layer: { penult: { knn_density: { splits: { test: { sparse_frac: b.do3.sparse_frac_test + dsp } } },
      neural_collapse: { nc1: b.do3.nc1_train * 1.002 } } } });
    const auc = 0.9 + (opt.perturbMargin === m.id ? 0.01 : 0.001);
    w(path.join(res, rel.replace(/^atlas_v1_/, "margin_v1_")), "atlas.json", { per_layer: { penult: { margin_typeb: { auc_margin_wrong: auc } } } });
  }
  return repo;
}

// a fake git: P1 < P2 < HEAD; the snapshot of each rev is the fixture repo's files at 'commit' time (no commit is made)
function fakeGit(repo) {
  const snap = {}, order = { P1: 1, P2: 2, HEAD: 3 };
  const take = (rev, files) => { snap[rev] = {}; for (const f of files) { const p = path.join(repo, f); if (fs.existsSync(p)) snap[rev][f] = S.fileSha256(p); } };
  return { snap, take, head: () => "HEADSHA000", isAncestor: (a, b) => (order[a] || 99) <= (order[b] || 0),
    sha256At: (rev, f) => ((snap[rev] || {})[f] === undefined ? null : snap[rev][f]),
    exists: (rev, f) => (snap[rev] || {})[f] !== undefined,
    fileSha256: f => (fs.existsSync(path.join(repo, f)) ? S.fileSha256(path.join(repo, f)) : null) };
}
const provOf = () => ({ t1: S.fileSha256(path.join(ROOT, "scripts/t1_scoreboard.py")), t1s: S.fileSha256(path.join(ROOT, "scripts/t1_streams.py")),
  core: S.fileSha256(path.join(ROOT, "atlas/b4_core.py")) });
const strip = l => String(l).replace(/^INFO \((dry run|discovery)\) /, "");
const labels = rep => Object.fromEntries(rep.claims.map(c => [c.id, c.label]));
const OPP = { SUPPORTED: "REFUTED", REFUTED: "SUPPORTED" };

function main() {
  const prov = provOf();
  // ---- design / flip / strip / power, dry runs ------------------------------------------------------------------------
  const runs = {};
  for (const mode of ["design", "flip", "strip", "power"]) {
    const repo = makeRepo(`dry_${mode}`, mode, prov);
    runs[mode] = E.evaluate({ repo, phase: "confirmation", dry: true }, fakeGit(repo));
  }
  const pred = Object.fromEntries(runs.design.claims.map(c => [c.id, c.predicted]));
  const types = Object.fromEntries(runs.design.claims.map(c => [c.id, c.type]));
  check("the claim list is complete (31 claims)", runs.design.claims.length === 31, runs.design.claims.length);
  for (const [id, p] of Object.entries(pred)) {
    const d = strip(labels(runs.design)[id]), fl = strip(labels(runs.flip)[id]), st = strip(labels(runs.strip)[id]), pw = strip(labels(runs.power)[id]);
    check(`${id}: design gives the predicted ${p}`, d === p, d);
    check(`${id}: flip gives ${OPP[p]}`, fl === OPP[p], fl);
    check(`${id}: stripped fields give NOT_EVALUABLE`, st === "NOT_EVALUABLE", st);
    if (types[id] !== "bool") check(`${id}: inconclusive unit calls give INCONCLUSIVE`, pw === "INCONCLUSIVE", pw);
    check(`${id}: dry-run labels are INFO`, String(labels(runs.design)[id]).startsWith("INFO (dry run) "));
  }
  const x11 = runs.design.claims.find(c => c.id === "X1-2");
  check("secondary scopes are reported (R2, M) beside the primary R", x11.scopes.R2 && x11.scopes.M && x11.scopes.R2.secondary && !x11.scopes.R.secondary);
  check("R counts the 4 fresh-seed units, R2 the 8 spent seeds, M the 12 C units, S the 10 STconf streams",
    runs.design.planned.R === 4 && runs.design.planned.R2 === 8 && runs.design.planned.M === 12 && runs.design.planned.S === 10, JSON.stringify(runs.design.planned));
  check("functional block reports medians of dAUC / dTPR@5%FPR and CPU ms per 1000 queries",
    runs.design.functional.R && runs.design.functional.R.increments["err_clean|margin_pen"].dtpr5.median === 0
    && runs.design.functional.R.cpu_ms_per_1000_queries.knnL2_penult.median === 12.5);
  check("model_outcomes of every fit board validate against the schema", runs.design.gates.model_outcomes.status === "PASS", JSON.stringify(runs.design.gates.model_outcomes).slice(0, 200));
  check("anchor gate PASS on committed numbers consistent with the discovery boards", runs.design.gates.anchors.status === "PASS", JSON.stringify(runs.design.gates.anchors).slice(0, 200));

  // ---- non-dry, in-process, fake git: every gate PASSes ----------------------------------------------------------------
  const repo = makeRepo("nondry", "design", prov);
  const g = fakeGit(repo);
  g.take("P1", E.T1_FILES);
  const disc = E.evaluate({ repo, phase: "discovery", p1: "P1", pRun: ["S1HEAD000"] }, g);
  check("discovery: gates PASS (selftests, provenance, read guard)", ["selftests", "provenance", "read_guard"].every(k => disc.gates[k].status === "PASS"),
    JSON.stringify({ s: disc.gates.selftests.status, p: disc.gates.provenance.failures, r: disc.gates.read_guard.status }));
  check("discovery labels are 'INFO (discovery) <label>' and R2 is NOT_PLANNED",
    disc.claims.every(c => c.label.startsWith("INFO (discovery) ")) && disc.claims.find(c => c.id === "X1-2").scopes.R2.label === "NOT_PLANNED");
  check("discovery: R = the 16 D units, M = the 5 Dnew, S = the 3 STdisc", disc.planned.R === 16 && disc.planned.M === 5 && disc.planned.S === 3, JSON.stringify(disc.planned));
  check("discovery: X3-8 needs the two discovery fault kinds only", strip(labels(disc)["X3-8"]) === "SUPPORTED", labels(disc)["X3-8"]);
  const rules = E.emitRules(disc, { p1: "P1" }, g);
  check("emitted rules carry every claim ACTIVE with its P1 fractions", Object.keys(rules.claims).length === 31
    && Object.values(rules.claims).every(r => r.status === "ACTIVE") && rules.claims["X1-2"].frac.R === 0.875);
  let refused = false;
  const amendP = path.join(TMP, "amend_bad.json");
  fs.writeFileSync(amendP, JSON.stringify({ frac: { "X1-2": { R: 0.5 } } }));
  try { E.emitRules(disc, { p1: "P1", amend: amendP }, g); } catch (e) { refused = /not stricter/.test(e.message); }
  check("--amend refuses a looser fraction (D8 c)", refused);
  fs.writeFileSync(amendP, JSON.stringify({ withdraw: { "X6-2": "harm arm dropped for power" }, frac: { "X1-3": { R: 1.0 } } }));
  const rulesAm = E.emitRules(disc, { p1: "P1", amend: amendP }, g);
  check("--amend withdraws to INFO and tightens a fraction", rulesAm.claims["X6-2"].status === "INFO" && rulesAm.claims["X1-3"].frac.R === 1.0);
  w(path.join(repo, "experiments/b4"), "t1_rules.json", rules);
  w(path.join(repo, "experiments/b4"), "freeze_P2.json", { p1: "P1", cuts: [] });
  g.take("P2", [...E.T1_FILES, "experiments/b4/t1_rules.json", "experiments/b4/freeze_P2.json"]);
  const conf = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g);
  check("confirmation: every gate PASSes", ["selftests", "provenance", "read_guard", "rules", "replay"].every(k => conf.gates[k].status === "PASS"),
    JSON.stringify(Object.fromEntries(Object.entries(conf.gates).map(([k, v]) => [k, v.status]))) + JSON.stringify(conf.gates.provenance.failures || []));
  check("confirmation: every label is the predicted one, without a prefix", conf.claims.every(c => c.label === c.predicted),
    conf.claims.filter(c => c.label !== c.predicted).map(c => `${c.id}:${c.label}`).join(" "));
  check("the self-test gate ignored the .log file and the other track's self-test", conf.gates.selftests.records.every(r => /\.json$/.test(r.file)));

  // provenance failure: one board from other code
  const bp = path.join(repo, "results/b4_t1/resnet44_fit/scoreboard.json"), orig = fs.readFileSync(bp, "utf8");
  const bj = JSON.parse(orig); bj.code.sha256 = "0".repeat(64); fs.writeFileSync(bp, JSON.stringify(bj));
  const bad1 = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g);
  check("a board from other code fails provenance: every label NOT_EVALUABLE (gate: provenance)",
    bad1.gates.provenance.status === "FAIL" && bad1.claims.every(c => c.label === "NOT_EVALUABLE (gate: provenance)"), bad1.claims[0].label);
  fs.writeFileSync(bp, orig);
  // withdrawn claim and a looser fraction in the rules
  const r2 = JSON.parse(JSON.stringify(rules)); r2.claims["X6-2"] = { status: "INFO", reason: "dropped", frac: rules.claims["X6-2"].frac };
  r2.claims["X1-3"].frac.R = 0.5;
  const rp2 = path.join(TMP, "rules2.json"); fs.writeFileSync(rp2, JSON.stringify(r2));
  const cOther = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"], rules: rp2 }, g);
  check("a rules file other than the one committed at P2 fails the rules gate", cOther.gates.rules.status === "FAIL"
    && cOther.claims.every(c => /gate: .*rules/.test(c.label)), JSON.stringify(cOther.gates.rules));
  const rulesPath = path.join(repo, "experiments/b4/t1_rules.json"), rulesOrig = fs.readFileSync(rulesPath, "utf8");
  fs.writeFileSync(rulesPath, JSON.stringify(r2));                        // as if P2 had committed these rules
  const g2 = fakeGit(repo); g2.snap.P1 = g.snap.P1;
  g2.take("P2", [...E.T1_FILES, "experiments/b4/t1_rules.json", "experiments/b4/freeze_P2.json"]);
  const c2 = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g2);
  fs.writeFileSync(rulesPath, rulesOrig);
  check("a claim withdrawn in the rules is 'INFO (withdrawn at P2: ...)'", labels(c2)["X6-2"].startsWith("INFO (withdrawn at P2: dropped"), labels(c2)["X6-2"]);
  check("a fraction looser than P1 in the rules makes that claim NOT_EVALUABLE", labels(c2)["X1-3"].startsWith("NOT_EVALUABLE (rules"), labels(c2)["X1-3"]);
  // replay drift: a FAIL tags the labels and nulls a near-threshold unit decision
  const rj = path.join(repo, "results/instrument_check_b4s2/replay.json"), rorig = fs.readFileSync(rj, "utf8");
  fs.writeFileSync(rj, JSON.stringify({ status: "FAIL", units: { "t1:resnet20_hub": { status: "FAIL", n_diff: 1,
    first20: ["/persample/err_shift_s3/joint/pre/dauc: 0.0105 vs 0.0098"] }, "t1:resnet56_hub": { status: "PASS", first20: [] } } }));
  const np = path.join(repo, "results/b4_t1/resnet56_s31_eval/scoreboard.json"), norig = fs.readFileSync(np, "utf8");
  const nj = JSON.parse(norig); nj.persample.err_shift_s3.joint.pre.dauc = 0.0102; fs.writeFileSync(np, JSON.stringify(nj));
  const c3 = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g);
  const x14 = c3.claims.find(c => c.id === "X1-4");
  check("replay FAIL: labels carry [REPLAY-DRIFT] (drift 0.0007)", c3.gates.replay.status === "FAIL" && Math.abs(c3.gates.replay.drift - 0.0007) < 1e-12
    && c3.claims.every(c => c.label.endsWith("[REPLAY-DRIFT]")), labels(c3)["X1-2"]);
  check("replay FAIL: a unit within the drift of its threshold is not evaluable", x14.scopes.R.per_unit.resnet56_s31 === null && x14.scopes.R.n === 3,
    JSON.stringify(x14.scopes.R.per_unit));
  fs.writeFileSync(np, norig);
  // replay whose drift the record cannot bound (D15; verifier T1-2): every label tagged, every decision null
  const unb = [
    ["a killed t1 replay unit (NOT_EVALUABLE, missing output)", { status: "FAIL", units: {
      "t1:resnet20_hub": { status: "PASS", n_diff: 0, first20: [] }, "t1:resnet56_hub": { status: "NOT_EVALUABLE", why: "missing output" } } }],
    ["a FAIL with a non-numeric diff (missing key)", { status: "FAIL", units: {
      "t1:resnet20_hub": { status: "FAIL", n_diff: 1, first20: ["/persample/err_clean/joint/pre: missing on one side"] },
      "t1:resnet56_hub": { status: "PASS", n_diff: 0, first20: [] } } }],
    ["a FAIL with more diffs than first20 lists", { status: "FAIL", units: {
      "t1:resnet20_hub": { status: "FAIL", n_diff: 25, first20: Array.from({ length: 20 }, (_, i) => `/x/${i}: 0.5 vs 0.5000001`) },
      "t1:resnet56_hub": { status: "PASS", n_diff: 0, first20: [] } } }],
    ["no replay.json", null],
  ];
  for (const [what, rjson] of unb) {
    if (rjson) fs.writeFileSync(rj, JSON.stringify(rjson)); else fs.unlinkSync(rj);
    const cu = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g);
    check(`replay unbounded (${what}): drift_unbounded, every label NOT_EVALUABLE ... [REPLAY-DRIFT]`,
      cu.gates.replay.status !== "PASS" && cu.gates.replay.drift_unbounded === true && cu.gates.replay.drift === Infinity
      && cu.claims.every(c => c.label.startsWith("NOT_EVALUABLE") && c.label.endsWith("[REPLAY-DRIFT]")),
      JSON.stringify({ st: cu.gates.replay.status, u: cu.gates.replay.drift_unbounded, l: labels(cu)["X1-2"] }));
  }
  fs.writeFileSync(rj, rorig);
  const cOk = E.evaluate({ repo, phase: "confirmation", p1: "P1", p2: "P2", pRun: ["S1HEAD000", "S2HEAD000"] }, g);
  check("replay PASS restored: no [REPLAY-DRIFT] tag, drift 0", cOk.gates.replay.status === "PASS" && cOk.gates.replay.drift === 0
    && cOk.claims.every(c => !c.label.includes("REPLAY-DRIFT")));
  // self-test gate: the streams self-test missing
  const repo2 = makeRepo("nondry_noselftest", "design", prov, { noStreamsSelftest: true });
  const g3 = fakeGit(repo2); g3.take("P1", E.T1_FILES);
  const c4 = E.evaluate({ repo: repo2, phase: "discovery", p1: "P1", pRun: ["S1HEAD000"] }, g3);
  check("a missing streams self-test fails the self-test gate", c4.gates.selftests.status === "FAIL" && c4.claims.every(c => /gate: selftests/.test(c.label)), c4.claims[0].label);
  // anchor gate: a perturbed committed margin AUC
  const repo3 = makeRepo("anchor_fail", "design", prov, { perturbMargin: "resnet56_s1", discoveryOnly: true });
  const g4 = fakeGit(repo3); g4.take("P1", E.T1_FILES);
  const c5 = E.evaluate({ repo: repo3, phase: "discovery", p1: "P1", pRun: ["S1HEAD000"] }, g4);
  check("a committed margin AUC 0.01 away fails the anchor gate: X1-2 NOT_EVALUABLE (gate: anchors.margin)",
    c5.gates.anchors.margin === "FAIL" && labels(c5)["X1-2"] === "INFO (discovery) NOT_EVALUABLE (gate: anchors.margin)", labels(c5)["X1-2"]);
  check("... while the DO-3 anchor still PASSes and X4-2 is evaluated", c5.gates.anchors.do3 === "PASS" && !/gate/.test(labels(c5)["X4-2"]), labels(c5)["X4-2"]);
  // DO-3 anchor tolerance 0.035 (verifier T1-3: the committed number's subsampling noise, SD up to ~0.01)
  for (const [delta, pass] of [[0.03, true], [0.04, false]]) {
    const rp = makeRepo(`anchor_sparse_${delta}`, "design", prov, { perturbSparse: { id: "resnet56_s1", delta }, discoveryOnly: true });
    const gs = fakeGit(rp); gs.take("P1", E.T1_FILES);
    const cs = E.evaluate({ repo: rp, phase: "discovery", p1: "P1", pRun: ["S1HEAD000"] }, gs);
    check(`a committed DO-3 sparse fraction ${delta} away ${pass ? "passes" : "fails"} the anchor gate`,
      pass ? cs.gates.anchors.do3 === "PASS" && !/gate/.test(labels(cs)["X4-2"])
        : cs.gates.anchors.do3 === "FAIL" && labels(cs)["X4-2"] === "INFO (discovery) NOT_EVALUABLE (gate: anchors.do3)",
      `${cs.gates.anchors.do3} ${labels(cs)["X4-2"]}`);
  }
  // schema gate: an invalid model_outcomes record is reported
  const op = path.join(repo, "results/b4_t1/vgg13_bn_fit/scoreboard.json"), oorig = fs.readFileSync(op, "utf8");
  const oj = JSON.parse(oorig); oj.model_outcomes.O1.bundle_members = ["margin_penult"]; fs.writeFileSync(op, JSON.stringify(oj));
  const c6 = E.evaluate({ repo, phase: "confirmation", dry: true }, g);
  check("an invalid model_outcomes record fails the schema gate for that unit", c6.gates.model_outcomes.status === "FAIL"
    && c6.gates.model_outcomes.units.vgg13_bn.valid === false && c6.gates.model_outcomes.units.resnet44.valid === true);
  fs.writeFileSync(op, oorig);
  check("validateSchema: the constructed record is valid, a wrong const is not",
    E.validateSchema(outcomes("x", "discovery"), OSCHEMA).length === 0 && E.validateSchema({ ...outcomes("x", "discovery"), layout: "eval" }, OSCHEMA).length === 1);

  // ---- CLI --------------------------------------------------------------------------------------------------------------
  const outJ = path.join(TMP, "eval_dry.json");
  const run = args => { try { cp.execFileSync(process.execPath, [EVAL_JS, ...args], { stdio: "pipe" }); return 0; } catch (e) { return e.status; } };
  const repoD = path.join(TMP, "dry_design");
  check("CLI dry run writes the JSON", run(["--dry-run", "--repo", repoD, "--json", outJ]) === 0 && fs.existsSync(outJ));
  check("CLI refuses to overwrite an existing --json (exit 2)", run(["--dry-run", "--repo", repoD, "--json", outJ]) === 2);
  const outN = path.join(TMP, "eval_nondry.json");
  const rc = run(["--repo", repoD, "--p1", "deadbeef", "--p2", "deadbeef", "--p-run", "S1HEAD000,S2HEAD000", "--json", outN]);
  const rn = rc === 0 ? JSON.parse(fs.readFileSync(outN, "utf8")) : null;
  check("CLI non-dry run on a non-git fixture repo: provenance and rules FAIL, every label NOT_EVALUABLE (gate: ...)",
    rn && rn.gates.provenance.status === "FAIL" && rn.gates.rules.status === "FAIL" && rn.gates.selftests.status === "PASS"
    && rn.claims.every(c => /^NOT_EVALUABLE \(gate: .*provenance.*rules/.test(c.label)), rn ? rn.claims[0].label : `exit ${rc}`);
  const outR = path.join(TMP, "rules_cli.json");
  check("CLI --emit-rules only in the discovery phase", run(["--dry-run", "--repo", repoD, "--emit-rules", outR]) === 2);
  check("CLI --emit-rules writes, then refuses to overwrite", run(["--dry-run", "--phase", "discovery", "--repo", repoD, "--emit-rules", outR]) === 0
    && run(["--dry-run", "--phase", "discovery", "--repo", repoD, "--emit-rules", outR]) === 2);
  check("CLI refuses an unregistered cut", run(["--dry-run", "--repo", repoD, "--cut", "C"]) === 2);
  const outC = path.join(TMP, "eval_cut.json");
  run(["--dry-run", "--repo", repoD, "--cut", "R2 F20", "--json", outC]);
  const rcut = fs.existsSync(outC) ? JSON.parse(fs.readFileSync(outC, "utf8")) : null;
  check("a registered cut (R2, F20) shrinks the plan: R2 NOT_PLANNED, R = the 2 resnet56 fresh seeds",
    rcut && rcut.planned.R2 === 0 && rcut.planned.R === 2 && rcut.claims.find(c => c.id === "X1-2").scopes.R2.label === "NOT_PLANNED");

  console.log(nbad ? `t1_eval fixture: ${nbad} of ${nchk} checks FAIL` : `t1_eval fixture: PASS (${nchk} checks; 31 claims x design / flip / strip / power, non-dry gates, CLI)`);
  if (!KEEP) fs.rmSync(TMP, { recursive: true, force: true }); else console.log(`kept ${TMP}`);
  return nbad ? 1 : 0;
}
process.exit(main());
