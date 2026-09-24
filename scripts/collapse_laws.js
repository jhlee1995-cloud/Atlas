#!/usr/bin/env node
// collapse_laws.js -- T2, the collapse arm of batch 4: model-level laws from collapse coordinates, the single PRIMARY
// claim (the owner's hypothesis) and the T2 secondaries (docs/plans/T2_COLLAPSE.md; docs/plans/B4_INTEGRATION.md D12).
// Node standard library + scripts/b4_stats.js only. Once committed at P1 THIS FILE IS THE DECISION RULE; between P1 and
// P2 only evaluator bug fixes with updated fixtures are allowed (D8 e), never a rule change.
//
//   node scripts/collapse_laws.js --selftest
//   node scripts/collapse_laws.js --dry-run [--emit-p7 experiments/b4/prereg_p7.json]
//        INFO preview on the committed batch-1..3 atlas.json files; reproduces the AH-1 law (0.0224 - 0.1080 log10 nc1)
//        and X9's Spearman +0.937 (known answers); --emit-p7 writes the blind P7 registration (refuses an existing file).
//   node scripts/collapse_laws.js --fit --p1 <P1> --out experiments/b4/laws_frozen.json [--root results]
//        [--registry experiments/b4/models.json] [--tag <discovery tag filter>]
//        Fits every law on the 21 discovery nets (D16 + Dnew5) from results/b4_t2/<id><tag>/probe.json and the T1
//        model_outcomes in results/b4_t1/<id>_fit<tag>/scoreboard.json; freezes coefficients, chosen coordinates, the
//        direction flags, the P3b threshold and its score on D, and the code hashes and file hashes of every fitted
//        record. Committed in P2. Refuses an existing --out.
//   node scripts/collapse_laws.js --evaluate --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> --json <new file>
//        [--laws experiments/b4/laws_frozen.json] [--replay results/instrument_check_b4s2/replay.json]
//        [--refit-tag _p2] [--disc-tag <S1 tag filter>] [--tag <S2 tag filter>]
//        Applies the frozen laws (never refits, except the registered D7 failure path --refit-tag _p2) to the sealed
//        confirmation units read once in S2; PRIMARY, secondaries with Holm, MP-K, mechanism items, rule 8, INFO.
//        Without --p1, --p2 and --p-run every verdict is NOT_EVALUABLE (guard) and the run is printed UNOFFICIAL.
// OUTPUT SELECTION (verifier T2-1): every tag of a unit is scanned ('' | _r<k> | _p2; *_s2replay is never a unit), so a
// D10 relaunch that split the outputs between '' and _r2 loses nothing. A confirmation unit uses its one confirmation-
// phase record per program (two are a touched-once violation: that unit is not evaluable); a discovery unit uses its
// highest relaunch, and under --refit-tag its _p2 re-probe. The T2 probe and the T1 scoreboard are chosen independently;
// the chosen directories are recorded (info.units[].sources). --tag / --disc-tag only filter to one exact tag.
//
// THE PRIMARY (D12, the owner's hypothesis "collapse predicts how models differ"). Units: the 12 sealed C architectures.
// Per outcome o in O1..O5 four laws, fitted on D21 by frozen code and frozen at P2:
//   const    = mean of o over the full-recipe discovery nets
//   acc      = o ~ accuracy                  (err_te or log10 err_te, whichever has the lower leave-one-run-out MAE)
//   acc_head = o ~ accuracy + best head-only coordinate (sat999_te, log10 gap_mean_te, log10 msp_def_te; LORO)
//   acc_coll = o ~ accuracy + best LABEL-FREE collapse coordinate (log10 plnc1_te, log10 g_cv_te, topk_frac_te; LORO)
// WIN_o  = MAE(acc_coll) <= 0.8 min(MAE const, acc, acc_head) AND Spearman(pred, y) >= 0.6 AND family-demeaned
//          Spearman > 0 (b4_stats.famDemeanedSpearman, families of >= 2 members, >= 6 units)
// LOSS_o = MAE ratio > 0.9. Verdict: NOT_EVALUABLE if < 4 outcomes have >= 10 units; SUPPORTED if >= 3 WIN; REFUTED if
// <= 1 WIN and >= 3 LOSS; otherwise MIXED. Outcomes: O1 dAUC_joint (T1 model_outcomes.O1.value), O2 H10, O3 L2-kNN -
// best head (CIFAR-100), O4 mean MSP - accuracy under the 10 discovery corruptions (T2 probe outcomes), O5 X4 AUROC -
// best head at s3 (T1 model_outcomes.O5.value).
// SECONDARIES (Holm over every item with a p-value; one-sided permutation p unless stated): see SECONDARY below.
"use strict";
const fs = require("fs"), path = require("path"), os = require("os");
const S = require("./b4_stats.js");

const REPO_DEFAULT = path.resolve(__dirname, "..");
const argv = process.argv.slice(2);
const arg = (k, d = null) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const has = k => argv.includes(k);

// =====================================================================================================================
// registered constants (P1). Change none without a new P1.
// =====================================================================================================================
const P = {
  OUTCOMES: ["O1", "O2", "O3", "O4", "O5"], RIVALS: ["const", "acc", "acc_head"],
  RATIO_WIN: 0.8, RATIO_LOSS: 0.9, RHO_MIN: 0.6, FAM_MIN_SIZE: 2, FAM_MIN_N: 6, N_MIN: 10, EVAL_MIN: 4,
  WIN_SUPPORTED: 3, REFUTE_MAX_WIN: 1, REFUTE_MIN_LOSS: 3,
  FIT_MIN: 12,                 // fewer usable discovery nets for an outcome -> that outcome's laws are not frozen
  LORO_MIN_GROUPS: 4, EXTRAP: 0.25,
  ALPHA: 0.05,                 // Holm family-wise level over the T2 secondaries
  NPERM: 20000, SEED: 20260924,
  AH1: { a: 0.0224, b: -0.1080, band: 0.035, bands: [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05], frac: 0.75 },
  ICP2: { alpha: 0.05, level: 0.99, allow_out: 1 },
  P3B: { d_max: 0.005, lo_max: 0.0, min_units: 3 },
  P5C: { allow_viol_units: 2 },
  P7: { tau_pass: -0.66, min_members: 3, min_families: 4, pass_families: 4 },
  P8: { dex: 0.1, frac: 10 / 12, rho_max: -0.8 },
  MLF: { rho_min: 0.7 },
  FE: { bar: 0.6, outcomes_needed: 3 },
  MPK: { frac: 0.75, min_cells: 10, dacc_pt_max: 0.5 },
  LAW: { rho_min: 0.6, mae_ratio_const: 0.8 },
  RULE8: { fam_min_abs: 0.2, too_good: 0.95 },
  REPLAY: { rel_floor: 1e-3 },  // INFO only: |x - y| / max(|x|, 1e-3); decisions use absolute per-input drifts (T2-3)
};
// candidate coordinates: probe.coords.penult.<key> under the transform tf
const CAND = {
  acc: [{ name: "err", key: "err_te", tf: "id" }, { name: "log10_err", key: "err_te", tf: "log10" }],
  head: [{ name: "sat999", key: "sat999_te", tf: "id" }, { name: "log10_gap_mean", key: "gap_mean_te", tf: "log10" },
    { name: "log10_msp_def", key: "msp_def_te", tf: "log10" }],
  lf: [{ name: "log10_plnc1", key: "plnc1_te", tf: "log10" }, { name: "log10_g_cv", key: "g_cv_te", tf: "log10" },
    { name: "topk_frac", key: "topk_frac_te", tf: "id" }],
  depth: [{ name: "cdepth_pl", key: "cdepth_pl", tf: "id" }],
  nc1: [{ name: "log10_nc1_tr", key: "nc1_tr", tf: "log10" }],
};
const COLLAPSE_SIGN = { plnc1_te: -1, g_cv_te: +1, topk_frac_te: +1, cdepth_pl: +1, nc1_tr: -1 };  // + = rises with collapse
// single-coordinate LABEL-FREE law items (fitted on D21, frozen at P2): dir = registered sign of dy / d(collapse)
const LAW_ITEMS = [
  { id: "P3a", y: "targets.margin.lead_md", dir: -1, cands: "lf",
    what: "margin - d1 lead (all-errors AUROC, train centres, clean rows 0-4999) falls as collapse deepens" },
  { id: "P4", y: "targets.ood.svhn.knn_l2_minus_besthead", dir: +1, cands: "lf",
    what: "SVHN: L2-normalised 10-NN AUROC - best head AUROC rises with collapse (Harun et al. 2025)" },
  { id: "P5a", y: "targets.harm.slope", dir: -1, cands: "lf",
    what: "harm slope (pt of paired accuracy loss per unit H) falls with collapse" },
  { id: "P5b", y: "targets.harm.H10", dir: +1, cands: "lf", what: "H10 (the HOLD-band edge) rises with collapse" },
  { id: "P6b", y: "targets.taps.p6b", dir: +1, cands: "lf+depth",
    what: "best non-duplicate tap - best head statistic, per-sample corrupt vs clean (s1, s3), rises with collapse" },
  { id: "S7", y: "targets.overconf.shift_mean", dir: +1, cands: "lf",
    what: "overconfidence under shift (mean MSP - accuracy, 30 discovery splits) rises with collapse" },
];
// correlation items over the C units: Spearman(x, y) oriented by dir must reach bar
const CORR_ITEMS = [
  { id: "MP1", x: { key: "nc1_tr", tf: "log10" }, y: "targets.do3.fpr_trainref", dir: -1, bar: 0.6,
    what: "rho(log10 nc1_train, train-referenced density false alarm) <= -0.6 (T1 MP-1)" },
  { id: "MP3", x: { key: "nc1_tr", tf: "log10" }, y: "targets.margin.lead_md", dir: +1, bar: 0.6,
    what: "rho(log10 nc1_train, margin - d1 lead) >= +0.6 (T1 MP-3 on the discovered coordinate, T1 review D1)" },
  { id: "MP5", x: { key: "nc1_tr", tf: "log10" }, y: "targets.margin.gap_minus_msp", dir: -1, bar: 0.5,
    what: "rho(log10 nc1_train, AUROC(logit gap) - AUROC(MSP) on clean errors) <= -0.5 (MSP saturation, T1 MP-5)" },
  { id: "MP6", x: { key: "nc1_tr", tf: "log10" }, y: "targets.ood.svhn.auc_knn_l2", dir: -1, bar: 0.5,
    what: "rho(log10 nc1_train, SVHN AUROC of the penult L2 10-NN) <= -0.5 (T1 MP-6)" },
  { id: "MP7", x: { key: "te_tr_gap_log10", tf: "id" }, y: "coords.penult.ece_te", dir: +1, bar: 0.5,
    what: "rho(log10 nc1_test / nc1_train, clean ECE) >= +0.5 (T1 MP-7)" },
  { id: "P2b", x: { key: "tt_gap_log10", tf: "id" }, y: "targets.do3.fpr_trainref", dir: +1, bar: 0.6,
    what: "the train-test collapse gap log10(nc1_tetr / nc1_train) orders the over-alarm (Hui et al. 2022)" },
  { id: "M-NC4", x: { key: "ncc_disagree_te", tf: "id" }, y: "y.O1", dir: +1, bar: 0.5,
    what: "mechanism: rho(1 - held-out NC4 agreement, O1) >= 0.5 (T3 CZ-3 / CZ-7b)" },
];
const FE_ITEMS = [   // family fixed-effect twins (T1 review D4): family-demeaned Spearman, families of >= 2 members
  { id: "FE1", twin: "MP1", x: { key: "nc1_tr", tf: "log10" }, y: "targets.do3.fpr_trainref", dir: -1 },
  { id: "FE3", twin: "MP3", x: { key: "nc1_tr", tf: "log10" }, y: "targets.margin.lead_md", dir: +1 },
];
// PROBE_KEYS-BEGIN: every probe.json path this evaluator reads (tests/test_collapse_probe.py pins them against a real probe)
const PROBE_KEYS = [
  "unit", "phase", "code.sha256", "code.core_sha256", "code.collapse_sha256", "code.repo_commit", "dumps",
  "instrument.weights_status", "instrument.head_check_status",
  "coords.penult.plnc1_te", "coords.penult.g_cv_te", "coords.penult.topk_frac_te", "coords.penult.sat999_te",
  "coords.penult.gap_mean_te", "coords.penult.msp_def_te", "coords.penult.err_te", "coords.penult.acc_te",
  "coords.penult.nc1_tr", "coords.penult.nc1_tr_verbatim", "coords.penult.nc1_te", "coords.penult.nc1_tetr",
  "coords.penult.tt_gap_log10", "coords.penult.te_tr_gap_log10", "coords.penult.ncc_disagree_te", "coords.penult.ece_te",
  "coords.penult.cdepth_pl", "coords.penult.dim", "coords.pre_tap",
  "targets.do3.fpr_trainref", "targets.ic_p2.fpr", "targets.ic_p2.n_cal", "targets.ic_p2.n_test", "targets.ic_p2.curve",
  "targets.margin.lead_md", "targets.margin.gap_minus_msp", "targets.p3b.d", "targets.p3b.ci_lo", "targets.p3b.ci_hi",
  "targets.p3b.head_pick", "targets.ood.svhn.knn_l2_minus_besthead", "targets.ood.svhn.auc_knn_l2", "targets.harm.slope",
  "targets.harm.H10", "targets.harm_committed.hold_n", "targets.harm_committed.hold_viol",
  "targets.harm_committed.holdout_hold_viol", "targets.overconf.shift_mean", "targets.taps.p6b",
  "outcomes.O2", "outcomes.O3", "outcomes.O4",
  "targets_se.fpr_trainref", "targets_se.lead_md", "targets_se.O2", "targets_se.O3", "targets_se.O4", "targets_se.p6b",
  "targets_se.harm_slope",
];
// PROBE_KEYS-END

// =====================================================================================================================
// numerics
// =====================================================================================================================
const isNum = S.isNum;
const num = v => (isNum(v) ? v : null);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const getp = (o, p) => p.split(".").reduce((a, k) => (a == null ? null : a[k]), o);
const codeStr = v => (typeof v === "string" && v ? v : null);             // a code hash, or null when not recorded
const TF = { id: v => (isNum(v) ? v : null), log10: v => (isNum(v) && v > 0 ? Math.log10(v) : null) };
function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function strSeed(s) { let h = 2166136261; for (const ch of String(s)) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return (h ^ P.SEED) >>> 0; }
const hashU = s => mulberry32(strSeed(s))();                                  // a fixed uniform [0, 1) per string
function shuffle(a, rnd) { const b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; }
function shuffleWithin(a, fam, rnd) {
  const out = a.slice(), by = {};
  fam.forEach((f, i) => (by[f] = by[f] || []).push(i));
  for (const ix of Object.values(by)) { const v = shuffle(ix.map(i => a[i]), rnd); ix.forEach((i, k) => { out[i] = v[k]; }); }
  return out;
}
// one-sided permutation p = (1 + #{perm >= obs}) / (B + 1); obs and perm oriented so that larger = more support
function permP(obs, permStat, key, B = P.NPERM) {
  if (!isNum(obs)) return null;
  const rnd = mulberry32(strSeed(key)); let ge = 0;
  for (let b = 0; b < B; b++) { const v = permStat(rnd); if (isNum(v) && v >= obs - 1e-12) ge++; }
  return (1 + ge) / (B + 1);
}
function binomTailHalf(k, n) {   // P(X >= k), X ~ Bin(n, 1/2)
  if (n <= 0) return null; let p = 0;
  for (let j = Math.max(0, k); j <= n; j++) p += Math.exp(S.lgamma(n + 1) - S.lgamma(j + 1) - S.lgamma(n - j + 1) - n * Math.LN2);
  return Math.min(1, p);
}
function kendall(x, y) {         // tau-b
  let c = 0, d = 0, tx = 0, ty = 0;
  for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) {
    const a = Math.sign(x[i] - x[j]), b = Math.sign(y[i] - y[j]);
    if (a === 0 && b === 0) continue; if (a === 0) tx++; else if (b === 0) ty++; else if (a === b) c++; else d++;
  }
  const den = Math.sqrt((c + d + tx) * (c + d + ty)); return den > 0 ? (c - d) / den : null;
}
function partialSpearman(x, y, z) {   // Pearson of the rank residuals of x and y on [1, rank z] (= b4_core.partial_spearman)
  const ok = x.map((v, i) => isNum(v) && isNum(y[i]) && isNum(z[i]));
  const xs = x.filter((_, i) => ok[i]), ys = y.filter((_, i) => ok[i]), zs = z.filter((_, i) => ok[i]);
  if (xs.length < 5) return null;
  const rz = S.rankAvg(zs), mz = mean(rz), szz = rz.reduce((s, v) => s + (v - mz) ** 2, 0);
  const res = r => { const m = mean(r); const b = szz > 0 ? r.reduce((s, v, i) => s + (v - m) * (rz[i] - mz), 0) / szz : 0;
    return r.map((v, i) => v - m - b * (rz[i] - mz)); };
  const ex = res(S.rankAvg(xs)), ey = res(S.rankAvg(ys));
  return S.pearson(ex, ey);
}
function lsq(Z, y) {             // least squares by the normal equations, Gaussian elimination with partial pivoting
  const p = Z[0].length, A = [...Array(p)].map(() => new Array(p + 1).fill(0));
  for (let i = 0; i < Z.length; i++) for (let j = 0; j < p; j++) {
    A[j][p] += Z[i][j] * y[i]; for (let k = 0; k < p; k++) A[j][k] += Z[i][j] * Z[i][k]; }
  const scale = Math.max(...A.map(r => Math.max(...r.slice(0, p).map(Math.abs))));
  for (let c = 0; c < p; c++) {
    let piv = c; for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (!(Math.abs(A[piv][c]) > 1e-10 * scale)) return null;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r < p; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= p; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((r, i) => r[p] / r[i]);
}

// =====================================================================================================================
// units
// =====================================================================================================================
function kindOf(roles) {
  for (const k of ["C", "K", "F", "F20", "Dnew", "N", "D"]) if (roles.includes(k)) return k === "D" ? "D16" : k;
  return null;
}
const CONF_KINDS = ["C", "K", "F", "F20"];
function readJSON(p) { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } }
const covVal = (u, c) => (u.probe ? TF[c.tf](getp(u.probe, `coords.penult.${c.key}`)) : null);
function yVal(u, p) { if (p.startsWith("y.")) return num(u.y[p.slice(2)]); return u.probe ? num(getp(u.probe, p)) : null; }
function makeUnit(m, probe, mo, paths = {}) {
  const ok = probe && probe.unit === m.id;
  const pr = ok ? probe : null;
  const moOk = mo && mo.schema === "b4_model_outcomes/1" && mo.unit === m.id ? mo : null;
  const y = { O1: moOk ? num(getp(moOk, "O1.value")) : null, O5: moOk ? num(getp(moOk, "O5.value")) : null,
    O2: pr ? num(getp(pr, "outcomes.O2")) : null, O3: pr ? num(getp(pr, "outcomes.O3")) : null,
    O4: pr ? num(getp(pr, "outcomes.O4")) : null };
  const ws = pr ? getp(pr, "instrument.weights_status") : null, hc = pr ? getp(pr, "instrument.head_check_status") : null;
  return { id: m.id, kind: kindOf(m.roles || []), roles: m.roles || [], family: m.family, depth_family: m.depth_family,
    run_group: m.run_group, full_recipe: !!m.full_recipe, trained: m.trained !== false, penult: m.penult, knob: m.knob || null,
    train: m.train || null, arch: m.arch, anchor: m.anchor || null, probe: pr, t1: moOk, y, ...paths,
    probe_mismatch: !!probe && !ok, evaluable: !!pr && ws !== "NOT_EVALUABLE" && (hc == null || hc === "PASS") };
}
// ---- output selection (verifier T2-1): every tag of a unit, one record per unit and program -------------------------
const TAG_OK = /^(?:_r\d+|_p2)*$/;                      // '' | _r<k> (D10 relaunch) | _p2 (D7 re-probe); never _s2replay
const REPLAY_TAG = /^(?:_r\d+)*_s2replay$/;             // the S2 replay of the anchors
const relaunchOf = t => Math.max(0, ...[...String(t).matchAll(/_r(\d+)/g)].map(x => Number(x[1])));
const OUT_OF = { t2: ["b4_t2", "", "probe.json"], t1: ["b4_t1", "_fit", "scoreboard.json"] };
function listDir(dir, cache) {
  if (!cache.has(dir)) cache.set(dir, fs.existsSync(dir) ? fs.readdirSync(dir).sort() : []);
  return cache.get(dir);
}
function scanRecords(root, prog, id, cache, tagRe = TAG_OK) {   // results/b4_t2/<id><t>/probe.json, results/b4_t1/<id>_fit<t>/...
  const [sub, mid, file] = OUT_OF[prog], pre = `${id}${mid}`, out = [];
  for (const d of listDir(path.join(root, sub), cache)) {
    if (!d.startsWith(pre) || !tagRe.test(d.slice(pre.length))) continue;
    const p = path.join(root, sub, d, file);
    const j = fs.existsSync(p) ? readJSON(p) : null;
    if (!j) continue;
    if ((prog === "t2" ? j.unit : getp(j, "model_outcomes.unit")) !== id) continue;     // the record must be this unit's
    out.push({ dir: `${sub}/${d}`, rel: `${sub}/${d}/${file}`, tag: d.slice(pre.length), path: p, j,
      phase: prog === "t2" ? j.phase : (getp(j, "model_outcomes.phase") || j.phase || null) });
  }
  return out;
}
const highest = recs => (recs.length ? recs.reduce((a, b) => (relaunchOf(b.tag) > relaunchOf(a.tag) ? b : a)) : null);
function chooseRecord(recs, { conf, refit, filter }) {
  const r = filter === null || filter === undefined ? recs : recs.filter(x => x.tag === filter);
  if (conf) {                                            // one confirmation-phase record; two break the touched-once rule
    const on = r.filter(x => x.phase === "confirmation"), off = r.filter(x => x.phase !== "confirmation").map(x => x.dir);
    if (on.length > 1) return { rec: null, off, dup: on.map(x => x.dir),
      why: `probed ${on.length} times (${on.map(x => x.dir).join(", ")}): touched-once violation` };
    return { rec: on[0] || null, off };
  }
  if (filter !== null && filter !== undefined) return { rec: highest(r), off: [] };
  const p2 = r.filter(x => /_p2/.test(x.tag)), s1 = r.filter(x => !/_p2/.test(x.tag));
  return { rec: highest(refit && p2.length ? p2 : s1), off: [] };   // the P2 laws were fitted on S1; a refit on _p2
}
function loadUnits({ root, registry, discTag = null, confTag = null, readConf = true, refit = false }) {
  const reg = readJSON(registry);
  if (!reg || !Array.isArray(reg.models)) throw new Error(`registry ${registry} unreadable`);
  const cache = new Map();
  const units = reg.models.map(m => {
    const conf = CONF_KINDS.includes(kindOf(m.roles || []));
    if (conf && !readConf) return makeUnit(m, null, null, { sources: { note: "not read (--fit never reads C)" } });
    const filter = conf ? confTag : discTag;
    const a = chooseRecord(scanRecords(root, "t2", m.id, cache), { conf, refit, filter });
    const b = chooseRecord(scanRecords(root, "t1", m.id, cache), { conf, refit, filter });
    const probe = a.rec ? a.rec.j : null, sb = b.rec ? b.rec.j : null;
    const u = makeUnit(m, probe, sb && sb.model_outcomes, {
      probe_path: a.rec ? a.rec.path : null, probe_rel: a.rec ? a.rec.rel : null,
      probe_sha256: a.rec ? S.fileSha256(a.rec.path) : null,
      t1_path: b.rec ? b.rec.path : null, t1_rel: b.rec ? b.rec.rel : null, t1_sha256: b.rec ? S.fileSha256(b.rec.path) : null,
      t1_phase: b.rec ? b.rec.phase : null, t1_code: sb && sb.code ? sb.code : null,
      sources: { t2: a.rec ? a.rec.dir : null, t1: b.rec ? b.rec.dir : null, t2_dup: a.dup || null, t1_dup: b.dup || null,
        t2_off_phase: a.off.length ? a.off : null, t1_off_phase: b.off.length ? b.off : null } });
    if (a.dup) u.not_evaluable_why = a.why;
    if (b.dup) { u.t1 = null; u.y.O1 = null; u.y.O5 = null; u.t1_not_evaluable_why = b.why; }
    return u;
  });
  return { reg, units };
}
const D21 = units => units.filter(u => (u.kind === "D16" || u.kind === "Dnew") && u.trained && u.evaluable);

// =====================================================================================================================
// laws: OLS on standardised covariates, leave-one-run-group-out selection
// =====================================================================================================================
function fitLaw(rows, covs, yOf) {
  const n = rows.length, p = covs.length;
  if (n < p + 3) return null;
  const X = rows.map(u => covs.map(c => covVal(u, c))), y = rows.map(yOf);
  if (X.some(r => r.some(v => v === null)) || y.some(v => !isNum(v))) return null;
  const mu = [], sd = [], xmin = [], xmax = [];
  for (let j = 0; j < p; j++) {
    const v = X.map(r => r[j]), m = mean(v), s = Math.sqrt(mean(v.map(x => (x - m) ** 2)));
    if (!(s > 1e-12 * Math.max(1, Math.abs(m)))) return null;
    mu.push(m); sd.push(s); xmin.push(Math.min(...v)); xmax.push(Math.max(...v));
  }
  const Z = X.map(r => [1, ...r.map((x, j) => (x - mu[j]) / sd[j])]);
  const beta = lsq(Z, y); if (!beta || beta.some(b => !isNum(b))) return null;
  const fit = Z.map(z => z.reduce((s, v, j) => s + v * beta[j], 0)), my = mean(y);
  const rss = y.reduce((s, v, i) => s + (v - fit[i]) ** 2, 0), tss = y.reduce((s, v) => s + (v - my) ** 2, 0);
  return { covs: covs.map(c => ({ name: c.name, key: c.key, tf: c.tf })), mu, sd, beta, n, xmin, xmax,
    slope: beta.slice(1).map((b, j) => b / sd[j]), r2: tss > 0 ? 1 - rss / tss : null, resid_sd: Math.sqrt(rss / Math.max(1, n - p - 1)) };
}
function predictLaw(L, u) {
  if (!L) return null;
  if (L.type === "const") return num(L.value);
  const x = L.covs.map(c => covVal(u, c)); if (x.some(v => v === null)) return null;
  return L.beta[0] + x.reduce((s, v, j) => s + L.beta[j + 1] * (v - L.mu[j]) / L.sd[j], 0);
}
function extrapLaw(L, u) {
  if (!L || L.type === "const") return false;
  return L.covs.some((c, j) => { const v = covVal(u, c), r = L.xmax[j] - L.xmin[j];
    return v !== null && (v < L.xmin[j] - P.EXTRAP * r || v > L.xmax[j] + P.EXTRAP * r); });
}
function loroMAE(rows, covs, yOf) {
  const groups = [...new Set(rows.map(u => u.run_group))];
  if (groups.length < P.LORO_MIN_GROUPS) return null;
  const err = [];
  for (const g of groups) {
    const L = fitLaw(rows.filter(u => u.run_group !== g), covs, yOf); if (!L) return null;
    for (const u of rows.filter(v => v.run_group === g)) { const p = predictLaw(L, u); if (p === null) return null; err.push(Math.abs(yOf(u) - p)); }
  }
  return mean(err);
}
function pickByLoro(rows, covSets, yOf) {
  const table = covSets.map(cs => ({ covs: cs.map(c => c.name), loro_mae: loroMAE(rows, cs, yOf) }));
  let bi = -1;
  table.forEach((t, i) => { if (isNum(t.loro_mae) && (bi < 0 || t.loro_mae < table[bi].loro_mae - 1e-15)) bi = i; });
  if (bi < 0) return null;
  const law = fitLaw(rows, covSets[bi], yOf); if (!law) return null;
  law.loro_mae = table[bi].loro_mae;
  return { covs: covSets[bi], law, table };
}

// =====================================================================================================================
// --fit: every law on D21, frozen at P2
// =====================================================================================================================
function fitPrimaryOutcome(D, o) {
  const yOf = u => u.y[o], all = [...CAND.acc, ...CAND.head, ...CAND.lf, ...CAND.nc1];
  const rows = D.filter(u => isNum(u.y[o]) && all.every(c => covVal(u, c) !== null));
  const out = { outcome: o, n: rows.length, ids: rows.map(u => u.id), n_groups: new Set(rows.map(u => u.run_group)).size };
  if (rows.length < P.FIT_MIN) return { ...out, status: "NOT_FROZEN", why: `${rows.length} usable discovery nets < ${P.FIT_MIN}` };
  const acc = pickByLoro(rows, CAND.acc.map(a => [a]), yOf);
  if (!acc) return { ...out, status: "NOT_FROZEN", why: "accuracy law undefined" };
  const A = acc.covs[0];
  const head = pickByLoro(rows, CAND.head.map(h => [A, h]), yOf), coll = pickByLoro(rows, CAND.lf.map(z => [A, z]), yOf);
  const twin = pickByLoro(rows, [[A, CAND.nc1[0]]], yOf);
  const cr = D.filter(u => u.full_recipe && isNum(u.y[o]));
  if (!head || !coll || !cr.length) return { ...out, status: "NOT_FROZEN", why: "a rival or the collapse law is undefined" };
  return { ...out, status: "FROZEN", laws: { const: { type: "const", value: mean(cr.map(yOf)), n: cr.length, ids: cr.map(u => u.id) },
    acc: acc.law, acc_head: head.law, acc_coll: coll.law, acc_nc1: twin ? twin.law : null },
    selection: { acc: acc.table, acc_head: head.table, acc_coll: coll.table } };
}
function fitLawItem(D, it) {
  const yOf = u => yVal(u, it.y), cands = it.cands === "lf+depth" ? [...CAND.lf, ...CAND.depth] : CAND.lf;
  const all = [...cands, ...CAND.acc, ...CAND.head];
  const rows = D.filter(u => isNum(yOf(u)) && all.every(c => covVal(u, c) !== null));
  const out = { id: it.id, y: it.y, dir: it.dir, what: it.what, n: rows.length };
  if (rows.length < P.FIT_MIN) return { ...out, status: "NOT_FROZEN", why: `${rows.length} usable discovery nets` };
  const law = pickByLoro(rows, cands.map(c => [c]), yOf), acc = pickByLoro(rows, CAND.acc.map(c => [c]), yOf);
  const head = pickByLoro(rows, CAND.head.map(c => [c]), yOf), cr = D.filter(u => u.full_recipe && isNum(yOf(u)));
  if (!law || !cr.length) return { ...out, status: "NOT_FROZEN", why: "law undefined" };
  const key = law.covs[0].key, dirOk = Math.sign(law.law.slope[0]) * COLLAPSE_SIGN[key] === it.dir;
  return { ...out, status: "FROZEN", law: law.law, rival_acc: acc ? acc.law : null, rival_head: head ? head.law : null,
    const: { type: "const", value: mean(cr.map(yOf)), n: cr.length }, selection: law.table, direction_as_registered: dirOk,
    rho_D: S.spearman(rows.map(u => predictLaw(law.law, u)), rows.map(yOf)) };
}
function fitMLF(D) {   // the label-free coordinate that ranks the DO-3 over-alarm on D21, with its sign
  const y = "targets.do3.fpr_trainref", tab = CAND.lf.map(c => {
    const r = D.filter(u => covVal(u, c) !== null && isNum(yVal(u, y)));
    return { coord: c, n: r.length, rho: S.spearman(r.map(u => covVal(u, c)), r.map(u => yVal(u, y))) }; });
  const ok = tab.filter(t => isNum(t.rho)).sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
  return ok.length ? { coord: ok[0].coord, sign: Math.sign(ok[0].rho), rho_D: ok[0].rho, table: tab } : null;
}
function p3bRule(units, theta) {
  const rows = units.map(u => ({ id: u.id, plnc1: num(getp(u.probe, "coords.penult.plnc1_te")), d: yVal(u, "targets.p3b.d"),
    lo: yVal(u, "targets.p3b.ci_lo"), hi: yVal(u, "targets.p3b.ci_hi"), head: getp(u.probe, "targets.p3b.head_pick") }))
    .filter(r => isNum(r.plnc1) && isNum(r.d) && isNum(r.lo));
  const col = rows.filter(r => r.plnc1 <= theta), bad = col.filter(r => r.lo > P.P3B.lo_max), small = col.filter(r => r.d <= P.P3B.d_max);
  const verdict = !isNum(theta) || col.length < P.P3B.min_units ? "NOT_EVALUABLE" : bad.length ? "REFUTED"
    : small.length === col.length ? "SUPPORTED" : "PARTIAL";
  return { theta, n_collapsed: col.length, collapsed: col, above_ci: bad.map(r => r.id), within_d_max: small.length, verdict_rule: verdict,
    reading: bad.length ? "centre geometry adds beyond the picked head statistic at a collapsed penult on these units" : null };
}
function nc1Gate(D, repo) {   // D15: the probe re-derives the committed atlas nc1 of every old net within 1%
  const rows = D.filter(u => u.anchor).map(u => {
    const a = readJSON(path.join(repo, u.anchor, "atlas.json")), want = a ? num(getp(a, "per_layer.penult.neural_collapse.nc1")) : null;
    const got = num(getp(u.probe, "coords.penult.nc1_tr_verbatim"));
    return { id: u.id, committed: want, probe: got, rel: isNum(want) && isNum(got) ? Math.abs(got / want - 1) : null };
  });
  const ev = rows.filter(r => isNum(r.rel));
  return { rows, n: ev.length, status: !ev.length ? "NOT_EVALUABLE" : ev.every(r => r.rel <= 0.01) ? "PASS" : "FAIL",
    rule: "|nc1_probe / nc1_committed - 1| <= 0.01 (docs/plans/B4_INTEGRATION.md D15)" };
}
function fitAll(units, opt = {}) {
  const D = D21(units);
  if (D.length < P.FIT_MIN) throw new Error(`only ${D.length} evaluable discovery nets (D16 + Dnew5); refusing to freeze`);
  const primary = {}; for (const o of P.OUTCOMES) primary[o] = fitPrimaryOutcome(D, o);
  const items = {}; for (const it of LAW_ITEMS) items[it.id] = fitLawItem(D, it);
  const r56 = D.filter(u => u.kind === "D16" && u.depth_family === "resnet56" && u.full_recipe)
    .map(u => num(getp(u.probe, "coords.penult.plnc1_te"))).filter(isNum);
  const theta = r56.length ? Math.max(...r56) : null;
  const preview = {};
  for (const it of CORR_ITEMS) { const r = D.filter(u => covVal(u, it.x) !== null && isNum(yVal(u, it.y)));
    const rho = S.spearman(r.map(u => covVal(u, it.x)), r.map(u => yVal(u, it.y)));
    preview[it.id] = { n: r.length, rho_D21: rho, direction_on_D21_as_registered: isNum(rho) ? Math.sign(rho) === it.dir : null }; }
  // provenance of every fitted record (verifier T2-2): the probe instrument (collapse_probe.py, b4_core.py,
  // b4_collapse.py), the T1 scoreboard code behind O1 / O5, and the file hashes the evaluation re-checks
  const fitted_on = D.map(u => ({ id: u.id, kind: u.kind, run_group: u.run_group,
    probe_path: u.probe_rel || null, probe_sha256: u.probe_sha256, t1_path: u.t1_rel || null, t1_sha256: u.t1_sha256,
    probe_code: codeStr(getp(u.probe, "code.sha256")), core_sha256: codeStr(getp(u.probe, "code.core_sha256")),
    collapse_sha256: codeStr(getp(u.probe, "code.collapse_sha256")),
    t1_code: u.t1 ? codeStr(getp(u.t1_code, "sha256")) : null, t1_core: u.t1 ? codeStr(getp(u.t1_code, "core_sha256")) : null,
    sources: u.sources || null }));
  const setOf = (k, rows = fitted_on) => [...new Set(rows.map(f => f[k]))];
  const withT1 = fitted_on.filter(f => f.t1_sha256);
  return { primary, law_items: items, m_lf: fitMLF(D), p3b: { theta, theta_rule: "max plnc1_te over the D16 full-recipe resnet56 nets",
    on_D: p3bRule(D, theta) }, discovery_preview_INFO: preview, fitted_on,
    probe_code_sha256: setOf("probe_code"),
    code_sets: { probe_code: setOf("probe_code"), core_sha256: setOf("core_sha256"), collapse_sha256: setOf("collapse_sha256"),
      t1_code: setOf("t1_code", withT1), t1_core: setOf("t1_core", withT1) },
    nc1_gate: opt.repo ? nc1Gate(D, opt.repo) : null };
}

// =====================================================================================================================
// replay drift of each decision (D15; verifier T2-3). dz.d(input) = the largest absolute S1 - S2 difference of that
// input over the two anchors, in the input's own (transformed) units, counted only when replay.json has a non-PASS unit
// of the program that produces it ('t2:' for probe fields, 't1:' for O1 / O5); Infinity when it cannot be measured.
// dz.dPred(law) bounds the change of a law's prediction. A decision is NOT_EVALUABLE only when the drift of its own
// statistic reaches its margin |stat - bar| (as ownerPrimary does); every label is still tagged on any replay FAIL.
// =====================================================================================================================
const DZ0 = { on: false, d: () => 0, dPred: () => 0 };
function pairsOk(x, y) { const ok = x.map((v, i) => isNum(v) && isNum(y[i])); return [x.filter((_, i) => ok[i]), y.filter((_, i) => ok[i])]; }
function closePairs(v, d) {       // pairs (i < j) whose order can change when every value moves by at most d
  if (!(d > 0)) return 0;
  const s = v.filter(isNum).sort((a, b) => a - b), n = s.length;
  if (!Number.isFinite(d)) return n * (n - 1) / 2;
  let k = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n && s[j] - s[i] <= 2 * d; j++) k++;
  return k;
}
// Spearman: one adjacent transposition of ranks changes sum d^2 by at most 2(n - 1), i.e. rho by at most
// 12 / (n (n + 1)); 12 / (n (n - 1)) per possible swap is used (conservative, covers average ranks of ties)
function rhoDrift(x, y, dx, dy) {
  const [xs, ys] = pairsOk(x, y), n = xs.length, k = closePairs(xs, dx) + closePairs(ys, dy);
  return k && n > 1 ? 12 * k / (n * (n - 1)) : 0;
}
// family-demeaned Spearman, Kendall: exact when no order can change, unbounded otherwise
function rankDrift(x, y, dx, dy) { const [xs, ys] = pairsOk(x, y); return closePairs(xs, dx) + closePairs(ys, dy) > 0 ? Infinity : 0; }
const nearBar = (stat, bar, drift) => drift > 0 && isNum(stat) && Math.abs(stat - bar) <= drift;
function conjFlip(cl) { return !cl.some(c => !c.ok && !c.near) && cl.some(c => c.near); }   // AND of clauses {ok, near}
function countFlip(k, need, up, down) { return k >= need ? k - down < need : k + up >= need; }  // k passing of need
function driftOut(item, why, detail) {
  item.replay_drift = detail; item.verdict_rule = "NOT_EVALUABLE"; item.why = `REPLAY-DRIFT: ${why}`; item.p = null;
  return item;
}
function makeDZ(pairs, triggered) {
  const memo = new Map();
  const measure = (get, prog) => {
    if (!triggered[prog]) return 0;
    if (!pairs.length || pairs.some(p => !p[`${prog}ok`])) return Infinity;
    let m = 0;
    for (const p of pairs) {
      const x = get(p.a), y = get(p.b);
      if (x === null && y === null) continue;              // absent on both sides: nothing to move
      if (!isNum(x) || !isNum(y)) return Infinity;
      m = Math.max(m, Math.abs(x - y));
    }
    return m;
  };
  const d = spec => {
    const key = typeof spec === "string" ? spec : `${spec.key}|${spec.tf}`;
    if (!memo.has(key)) {
      let v;
      if (typeof spec !== "string") v = measure(u => covVal(u, spec), "t2");
      else if (/^y\.O[15]$/.test(spec)) v = measure(u => num(u.y[spec.slice(2)]), "t1");
      else v = measure(u => yVal(u, spec), "t2");
      memo.set(key, v);
    }
    return memo.get(key);
  };
  const dPred = L => {
    if (!L || L.type === "const") return 0;
    let s = 0;
    L.covs.forEach((c, j) => { const w = Math.abs(L.beta[j + 1] / L.sd[j]), dc = d(c); if (w > 0 && dc > 0) s += w * dc; });
    return s;
  };
  return { on: triggered.t2 || triggered.t1, d, dPred, triggered };
}

// =====================================================================================================================
// evaluation
// =====================================================================================================================
function predsFor(u, laws) {
  const pr = {};
  for (const o of P.OUTCOMES) { const f = laws.primary[o]; if (!f || f.status !== "FROZEN") continue;
    pr[o] = {}; for (const k of ["const", "acc", "acc_head", "acc_coll", "acc_nc1"]) pr[o][k] = predictLaw(f.laws[k], u);
    pr[o].extrap = extrapLaw(f.laws.acc_coll, u); }
  return pr;
}
function ownerPrimary(units, collKey = "acc_coll", drift = null) {   // the D12 rule (INTEGRATION/b4_primary.js, verbatim logic)
  const per = {}; let nEval = 0, win = 0, loss = 0;
  for (const o of P.OUTCOMES) {
    const rows = units.filter(u => u.kind === "C" && u.evaluable !== false && isNum(u.y && u.y[o]) && u.pred && u.pred[o]
      && [collKey, ...P.RIVALS].every(r => isNum(u.pred[o][r])))
      .map(u => ({ id: u.id, family: u.family, y: u.y[o], p: u.pred[o][collKey], pr: u.pred[o], extrap: !!u.pred[o].extrap,
        dim: u.penult, acc: u.probe ? covVal(u, CAND.acc[0]) : null }));
    if (rows.length < P.N_MIN) { per[o] = { n: rows.length, verdict: "NOT_EVALUABLE", why: `${rows.length} units < ${P.N_MIN}` }; continue; }
    const mae = r => mean(rows.map(x => Math.abs(x.y - x.pr[r])));
    const m = Object.fromEntries([collKey, ...P.RIVALS].map(r => [r, mae(r)]));
    const best = P.RIVALS.reduce((a, b) => (m[b] < m[a] ? b : a));
    const ratio = m[best] > 0 ? m[collKey] / m[best] : (m[collKey] > 0 ? Infinity : 1);
    const rho = S.spearman(rows.map(x => x.p), rows.map(x => x.y));
    const rhoFam = S.famDemeanedSpearman(rows.map(x => x.p), rows.map(x => x.y), rows.map(x => x.family), P.FAM_MIN_SIZE, P.FAM_MIN_N);
    const margin = Math.abs(m[collKey] - P.RATIO_WIN * m[best]);
    const dr = drift ? drift[o] : null;                               // Infinity: a drift the replay cannot bound
    if (dr === Infinity || (isNum(dr) && dr > 0 && dr >= margin)) {
      per[o] = { n: rows.length, mae: m, ratio, verdict: "NOT_EVALUABLE", why: `REPLAY-DRIFT ${drift[o]} >= margin ${margin}` }; continue; }
    nEval++;
    const W = ratio <= P.RATIO_WIN && rho !== null && rho >= P.RHO_MIN && rhoFam !== null && rhoFam > 0, L = ratio > P.RATIO_LOSS;
    win += W; loss += L;
    const nx = rows.filter(r => r.extrap).length;
    const r8 = rule8({ verdict_rule: "ANNOTATION" }, rows.map(x => x.p), rows.map(x => x.y), rows.map(x => x.family),
      rows.map(x => x.dim), rows.map(x => x.acc), P.RHO_MIN, +1).rule8;    // annotation only: D12 fixes the PRIMARY rule
    per[o] = { n: rows.length, ids: rows.map(r => r.id), mae: m, best_rival: best, ratio, rho, rho_family_demeaned: rhoFam,
      verdict: W ? "WIN" : L ? "LOSS" : "NEITHER", n_extrap: nx, rule8: r8,
      note: nx > rows.length / 2 ? "EXTRAP-dominated: reads 'extrapolation holds', not 'the law transfers'" : null,
      functional: { mae_acc_coll: m[collKey], mae_best_rival: m[best], units: "outcome units (see T2_COLLAPSE.md section 5)" } };
  }
  const verdict = nEval < P.EVAL_MIN ? "NOT_EVALUABLE" : win >= P.WIN_SUPPORTED ? "SUPPORTED"
    : (win <= P.REFUTE_MAX_WIN && loss >= P.REFUTE_MIN_LOSS) ? "REFUTED" : "MIXED";
  return { claim: "PRIMARY (owner hypothesis): a label-free collapse coordinate predicts model differences beyond accuracy, the head and the constant",
    law: collKey, outcomes_evaluable: nEval, wins: win, losses: loss, verdict, per_outcome: per };
}
function rule8(item, x, y, fam, dims, acc, bar, dir) {   // FAMILY-DRIVEN, DIMENSION, RULE8-CHECK (T3 review M3; CLAUDE.md rule 8)
  const rho = S.spearman(x, y), rf = S.famDemeanedSpearman(x, y, fam, P.FAM_MIN_SIZE, P.FAM_MIN_N);
  const pd = partialSpearman(x, y, dims.map(d => (isNum(d) && d > 0 ? Math.log10(d) : null)));
  const pa = partialSpearman(x, y, acc);
  const r = { rho, rho_family_demeaned: rf, partial_log_dim: pd, partial_accuracy: pa, flags: [] };
  if (isNum(rho) && (rf === null || Math.sign(rf) !== Math.sign(rho) || Math.abs(rf) < P.RULE8.fam_min_abs)) r.flags.push("FAMILY-DRIVEN");
  if (isNum(bar) && isNum(dir) && (pd === null || dir * pd < bar)) r.flags.push("DIMENSION");
  if (isNum(rho) && Math.abs(rho) >= P.RULE8.too_good) r.flags.push("RULE8-CHECK");
  item.rule8 = r;
  if (item.verdict_rule === "SUPPORTED") {
    if (r.flags.includes("FAMILY-DRIVEN")) item.verdict_rule = "PARTIAL (FAMILY-DRIVEN)";
    else if (r.flags.includes("DIMENSION")) item.verdict_rule = "PARTIAL (DIMENSION)";
  }
  if (r.flags.includes("RULE8-CHECK")) item.rule8.artifact_hypotheses = ["family offsets (see rho_family_demeaned)",
    "penult width (see partial_log_dim)", "accuracy (see partial_accuracy)", "a shared-row or shared-code artefact (see the replay)"];
  return item;
}
function ctx(units) { return { fam: units.map(u => u.family), dims: units.map(u => u.penult), acc: units.map(u => covVal(u, CAND.acc[0])) }; }
function evalCorr(C, it, dz = DZ0) {
  const r = C.filter(u => covVal(u, it.x) !== null && isNum(yVal(u, it.y)));
  const item = { id: it.id, what: it.what, n: r.length, dir: it.dir, bar: it.bar, p: null };
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const x = r.map(u => covVal(u, it.x)), y = r.map(u => yVal(u, it.y)), rho = S.spearman(x, y);
  item.rho = rho; item.verdict_rule = isNum(rho) && it.dir * rho >= it.bar ? "SUPPORTED" : "REFUTED";
  item.p = permP(isNum(rho) ? it.dir * rho : null, rnd => { const v = S.spearman(x, shuffle(y, rnd)); return isNum(v) ? it.dir * v : null; }, it.id);
  const dr = rhoDrift(x, y, dz.d(it.x), dz.d(it.y));
  if (nearBar(it.dir * rho, it.bar, dr))
    driftOut(item, `rho drift ${dr} >= margin ${Math.abs(it.dir * rho - it.bar)}`, { stat: it.dir * rho, bar: it.bar, drift: dr });
  const c = ctx(r); return rule8(item, x, y, c.fam, c.dims, c.acc, it.bar, it.dir);
}
function evalFE(C, it, dz = DZ0) {
  const r = C.filter(u => covVal(u, it.x) !== null && isNum(yVal(u, it.y)));
  const x = r.map(u => covVal(u, it.x)), y = r.map(u => yVal(u, it.y)), fam = r.map(u => u.family);
  const rf = S.famDemeanedSpearman(x, y, fam, P.FAM_MIN_SIZE, P.FAM_MIN_N);
  const item = { id: it.id, twin_of: it.twin, what: `family-demeaned twin of ${it.twin} (T1 review D4): dir * rho_fam >= ${P.FE.bar}`,
    n: r.length, rho_family_demeaned: rf, p: null };
  if (r.length < P.N_MIN || rf === null) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  item.verdict_rule = it.dir * rf >= P.FE.bar ? "SUPPORTED" : "REFUTED";
  item.p = permP(it.dir * rf, rnd => { const v = S.famDemeanedSpearman(x, shuffleWithin(y, fam, rnd), fam, P.FAM_MIN_SIZE, P.FAM_MIN_N);
    return isNum(v) ? it.dir * v : null; }, it.id);
  const dr = rankDrift(x, y, dz.d(it.x), dz.d(it.y));
  if (nearBar(it.dir * rf, P.FE.bar, dr))
    driftOut(item, "a rank of the family-demeaned Spearman can change within the drift", { stat: it.dir * rf, bar: P.FE.bar, drift: dr });
  return item;
}
function evalLawItem(C, it, f, dz = DZ0) {
  const item = { id: it.id, what: it.what, dir: it.dir, p: null };
  if (!f || f.status !== "FROZEN") return { ...item, verdict_rule: "NOT_EVALUABLE", why: f ? f.why : "not fitted" };
  const r = C.filter(u => isNum(yVal(u, f.y)) && predictLaw(f.law, u) !== null);
  item.n = r.length; item.coord = f.law.covs[0].name; item.direction_as_registered = f.direction_as_registered;
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const y = r.map(u => yVal(u, f.y)), pred = r.map(u => predictLaw(f.law, u));
  const mae = p => mean(p.map((v, i) => Math.abs(y[i] - v)));
  const m = { law: mae(pred), const: mae(r.map(() => f.const.value)),
    acc: f.rival_acc ? mae(r.map(u => predictLaw(f.rival_acc, u))) : null, head: f.rival_head ? mae(r.map(u => predictLaw(f.rival_head, u))) : null };
  const rho = S.spearman(pred, y), rf = S.famDemeanedSpearman(pred, y, r.map(u => u.family), P.FAM_MIN_SIZE, P.FAM_MIN_N);
  Object.assign(item, { mae: m, rho, rho_family_demeaned: rf, beyond_accuracy_ratio: isNum(m.acc) && m.acc > 0 ? m.law / m.acc : null,
    beyond_head_ratio: isNum(m.head) && m.head > 0 ? m.law / m.head : null, n_extrap: r.filter(u => extrapLaw(f.law, u)).length });
  item.verdict_rule = isNum(rho) && rho >= P.LAW.rho_min && isNum(rf) && rf > 0 && m.law <= P.LAW.mae_ratio_const * m.const
    && f.direction_as_registered === true ? "SUPPORTED" : "REFUTED";
  if (item.n_extrap > r.length / 2) item.note = "EXTRAP-dominated: reads 'extrapolation holds', not 'the law transfers'";
  const se = r.map(u => num(getp(u.probe, `targets_se.${{ P3a: "lead_md", P5a: "harm_slope", P5b: "O2", P6b: "p6b", S7: "O4" }[f.id] || "none"}`))).filter(isNum).sort((a, b) => a - b);
  const sdY = Math.sqrt(mean(y.map(v => (v - mean(y)) ** 2)));
  item.noise = { median_split_half_se: se.length ? se[Math.floor(se.length / 2)] : null, sd_between_units: sdY,
    noise_limited: se.length ? sdY < 2 * se[Math.floor(se.length / 2)] : null };
  item.p = permP(isNum(rho) ? rho : null, rnd => S.spearman(pred, shuffle(y, rnd)), f.id);
  if (dz.on) {                     // pred moves by <= dPred, y by <= dy: rank bounds for rho / rho_fam, triangle for the MAEs
    const dp = dz.dPred(f.law), dy = dz.d(f.y), drho = rhoDrift(pred, y, dp, dy), drf = rankDrift(pred, y, dp, dy);
    const dmae = dy + dp + P.LAW.mae_ratio_const * dy, gapMae = m.law - P.LAW.mae_ratio_const * m.const;
    if (conjFlip([{ ok: isNum(rho) && rho >= P.LAW.rho_min, near: nearBar(rho, P.LAW.rho_min, drho) },
      { ok: isNum(rf) && rf > 0, near: nearBar(rf, 0, drf) }, { ok: gapMae <= 0, near: nearBar(gapMae, 0, dmae) },
      { ok: f.direction_as_registered === true, near: false }]))
      driftOut(item, "a clause lies within its replay drift", { rho_drift: drho, rho_fam_drift: drf, mae_drift: dmae });
  }
  const c = ctx(r); return rule8(item, pred, y, c.fam, c.dims, c.acc, P.LAW.rho_min, +1);
}
function evalMDO3(C, dz = DZ0) {
  const item = mdo3Core(C);
  if (dz.on && item.residuals) {   // a residual moves by <= dy + |b| dx: units within that of the band edge may cross it
    const dres = dz.d("targets.do3.fpr_trainref") + Math.abs(P.AH1.b) * dz.d(CAND.nc1[0]), rr = Object.values(item.residuals);
    const up = rr.filter(v => Math.abs(v) > P.AH1.band && Math.abs(v) - P.AH1.band <= dres).length;
    const down = rr.filter(v => Math.abs(v) <= P.AH1.band && P.AH1.band - Math.abs(v) <= dres).length;
    if (dres > 0 && countFlip(item.inside, item.need, up, down))
      driftOut(item, `${up + down} units within the residual drift ${dres} of the band edge`, { drift: dres, up, down });
  }
  return item;
}
function mdo3Core(C) {
  const r = C.filter(u => covVal(u, CAND.nc1[0]) !== null && isNum(yVal(u, "targets.do3.fpr_trainref")));
  const x = r.map(u => covVal(u, CAND.nc1[0])), y = r.map(u => yVal(u, "targets.do3.fpr_trainref"));
  const res = xs => xs.map((v, i) => y[i] - (P.AH1.a + P.AH1.b * v)), inside = (xs, b) => res(xs).filter(v => Math.abs(v) <= b).length;
  const item = { id: "M-DO3", what: "mechanism: the committed AH-1 law |FPR - (0.0224 - 0.1080 log10 nc1_train)| <= 0.035 in >= 75% of C units (T1 X4-2M, T3 CZ-1, T2 P1 demoted)",
    n: r.length, p: null };
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const need = Math.ceil(P.AH1.frac * r.length), k = inside(x, P.AH1.band);
  Object.assign(item, { inside: k, need, residuals: Object.fromEntries(r.map((u, i) => [u.id, res(x)[i]])),
    curve_rule5: P.AH1.bands.map(b => ({ band: b, inside: inside(x, b) })), verdict_rule: k >= need ? "SUPPORTED" : "REFUTED",
    functional: { false_alarms_per_1000_frames_mae: 1000 * mean(res(x).map(Math.abs)) } });
  item.p = permP(k, rnd => inside(shuffle(x, rnd), P.AH1.band), "M-DO3");
  const te = r.map(u => num(getp(u.probe, "coords.penult.te_tr_gap_log10")));
  item.info_CZ7c = { rho_te_tr_gap_vs_residual: S.spearman(te, res(x)), note: "Hui et al.'s mechanism for DO-3 (INFO)" };
  return item;
}
function evalMLF(C, mlf, dz = DZ0) {
  const item = { id: "M-LF", what: "mechanism: the label-free coordinate chosen on D21 ranks the C units' train-referenced false alarm (oriented rho >= 0.7) better than every head coordinate and accuracy (T3 CZ-2 / CZ-7a, T1 MP-2)", p: null };
  if (!mlf) return { ...item, verdict_rule: "NOT_EVALUABLE", why: "not fitted" };
  const yk = "targets.do3.fpr_trainref", r = C.filter(u => covVal(u, mlf.coord) !== null && isNum(yVal(u, yk)));
  item.n = r.length; item.coord = mlf.coord.name; item.sign = mlf.sign;
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const x = r.map(u => covVal(u, mlf.coord)), y = r.map(u => yVal(u, yk)), rho = S.spearman(x, y);
  const riv = {}; for (const c of [...CAND.head, CAND.acc[0]]) riv[c.name] = S.spearman(r.map(u => covVal(u, c)), y);
  const bestRiv = Math.max(...Object.values(riv).filter(isNum).map(Math.abs), 0);
  Object.assign(item, { rho, rivals_rho: riv, best_rival_abs_rho: bestRiv,
    verdict_rule: isNum(rho) && mlf.sign * rho >= P.MLF.rho_min && Math.abs(rho) > bestRiv ? "SUPPORTED" : "REFUTED" });
  item.p = permP(mlf.sign * rho, rnd => { const v = S.spearman(shuffle(x, rnd), y); return isNum(v) ? mlf.sign * v : null; }, "M-LF");
  if (dz.on) {
    const dy = dz.d(yk), dr = rhoDrift(x, y, dz.d(mlf.coord), dy);
    const drRiv = Math.max(0, ...[...CAND.head, CAND.acc[0]].map(c => rhoDrift(r.map(u => covVal(u, c)), y, dz.d(c), dy)));
    if (conjFlip([{ ok: isNum(rho) && mlf.sign * rho >= P.MLF.rho_min, near: nearBar(mlf.sign * rho, P.MLF.rho_min, dr) },
      { ok: isNum(rho) && Math.abs(rho) > bestRiv, near: isNum(rho) && nearBar(Math.abs(rho) - bestRiv, 0, dr + drRiv) }]))
      driftOut(item, "a clause lies within its replay drift", { rho_drift: dr, rival_rho_drift: drRiv });
  }
  const c = ctx(r); return rule8(item, x, y, c.fam, c.dims, c.acc, P.MLF.rho_min, mlf.sign);
}
function evalICP2(C, dz = DZ0) {
  const r = C.filter(u => isNum(yVal(u, "targets.ic_p2.fpr")));
  const item = { id: "IC-P2", what: "instrument check: the held-out-calibrated conformal false alarm (cal rows A, test rows B, alpha 0.05) is inside the exact 99% beta-binomial band in >= n - 1 C units (no p-value: an instrument check)", n: r.length, p: null };
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const rows = r.map(u => { const b = S.betaBinomBand(yVal(u, "targets.ic_p2.n_cal"), yVal(u, "targets.ic_p2.n_test"), P.ICP2.alpha, P.ICP2.level);
    const f = yVal(u, "targets.ic_p2.fpr"); return { id: u.id, fpr: f, lo: b.lo, hi: b.hi, inside: f >= b.lo && f <= b.hi }; });
  const k = rows.filter(x => x.inside).length;
  const n0 = yVal(r[0], "targets.ic_p2.n_cal"), n1 = yVal(r[0], "targets.ic_p2.n_test");
  const out = { ...item, rows, inside: k, verdict_rule: k >= r.length - P.ICP2.allow_out ? "SUPPORTED" : "REFUTED",
    curve_rule5: isNum(n0) && isNum(n1) ? S.betaBinomCurve(n0, n1, [0.02, 0.03, 0.04, 0.05], P.ICP2.level) : null };
  const d = dz.d("targets.ic_p2.fpr");
  if (d > 0) {                     // a unit within d of a band edge may cross it
    const up = rows.filter(x => !x.inside && (x.fpr < x.lo ? x.lo - x.fpr : x.fpr - x.hi) <= d).length;
    const down = rows.filter(x => x.inside && Math.min(x.fpr - x.lo, x.hi - x.fpr) <= d).length;
    if (countFlip(k, r.length - P.ICP2.allow_out, up, down)) driftOut(out, `${up + down} units within ${d} of a band edge`, { drift: d, up, down });
  }
  return out;
}
function evalP5c(C, dz = DZ0) {
  const r = C.filter(u => isNum(yVal(u, "targets.harm_committed.hold_viol")));
  const item = { id: "P5c", what: "the committed HOLD band (AH-2(b): H <= 0.25 -> cost <= 8 pt) has no violation over the 30 discovery splits in >= n - 2 C units (a bound rule; no p-value)", n: r.length, p: null };
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const ok = r.filter(u => yVal(u, "targets.harm_committed.hold_viol") === 0).length;
  const out = { ...item, units_without_violation: ok, verdict_rule: ok >= r.length - P.P5C.allow_viol_units ? "SUPPORTED" : "REFUTED",
    info_holdout: Object.fromEntries(r.map(u => [u.id, yVal(u, "targets.harm_committed.holdout_hold_viol")])) };
  const d = dz.d("targets.harm_committed.hold_viol");
  if (d > 0) {                     // an integer count: 0 can become >= 1 when d >= 1; v >= 1 can become 0 when d >= v
    const v = r.map(u => yVal(u, "targets.harm_committed.hold_viol"));
    const down = v.filter(x => x === 0 && d >= 1).length, up = v.filter(x => x > 0 && x <= d).length;
    if (countFlip(ok, r.length - P.P5C.allow_viol_units, up, down)) driftOut(out, `violation counts within the drift ${d}`, { drift: d, up, down });
  }
  return out;
}
function evalP7(all, reg, dz = DZ0) {
  const item = { id: "P7", p: null, what: "blind collapse ordering registered at P1 (experiments/b4/prereg_p7.json): Kendall tau(order, nc1_train) <= -0.66 in >= 4 families" };
  if (!reg) return { ...item, verdict_rule: "NOT_EVALUABLE", why: "no P1 registration read" };
  const val = id => { const u = all.find(w => w.id === id); return u && u.evaluable ? covVal(u, { key: "nc1_tr", tf: "id" }) : null; };
  const fams = {}, idx = [], vals = [], famOf = [];
  for (const [f, order] of Object.entries(reg.families_monotone)) {
    const v = order.map((id, i) => [i, val(id)]).filter(p => isNum(p[1]));
    if (v.length < P.P7.min_members) { fams[f] = { n: v.length, tau: null }; continue; }
    const tau = kendall(v.map(p => p[0]), v.map(p => p[1])); fams[f] = { n: v.length, tau, pass: isNum(tau) && tau <= P.P7.tau_pass };
    v.forEach(p => { idx.push(p[0]); vals.push(p[1]); famOf.push(f); });
  }
  const nF = Object.values(fams).filter(f => isNum(f.tau)).length, pass = Object.values(fams).filter(f => f.pass).length;
  const points = {}; for (const [id, pt] of Object.entries(reg.points || {})) { const v = val(id);
    points[id] = { predicted: pt.nc1_tr, lo: pt.lo, hi: pt.hi, observed: v, inside: isNum(v) ? v >= pt.lo && v <= pt.hi : null }; }
  const out = { ...item, families: fams, families_evaluable: nF, families_pass: pass, points_INFO: points,
    verdict_rule: nF < P.P7.min_families ? "NOT_EVALUABLE" : pass >= P.P7.pass_families ? "SUPPORTED" : "REFUTED" };
  if (nF >= P.P7.min_families) out.p = permP(pass, rnd => { const s = shuffleWithin(vals, famOf, rnd); let k = 0;
    for (const f of new Set(famOf)) { const ix = famOf.map((g, i) => (g === f ? i : -1)).filter(i => i >= 0);
      const t = kendall(ix.map(i => idx[i]), ix.map(i => s[i])); if (isNum(t) && t <= P.P7.tau_pass) k++; } return k; }, "P7");
  const dn = dz.d({ key: "nc1_tr", tf: "id" });
  if (dn > 0 && nF >= P.P7.min_families) {    // a family whose members' order can change within the drift is uncertain
    const unc = f => { const v = vals.filter((_, i) => famOf[i] === f); return rankDrift(v, v, dn, 0) > 0; };
    const ev = Object.entries(fams).filter(([, v]) => isNum(v.tau)).map(([f, v]) => ({ f, pass: !!v.pass, unc: unc(f) }));
    const up = ev.filter(e => e.unc && !e.pass).length, down = ev.filter(e => e.unc && e.pass).length;
    if (countFlip(pass, P.P7.pass_families, up, down)) driftOut(out, `${up + down} families' order within the drift ${dn}`, { drift: dn, up, down });
  }
  return out;
}
function evalP8(C, dz = DZ0) {
  const item = { id: "P8", what: "label-free tracks labelled: |log10(plnc1 / nc1_test)| <= 0.1 in >= 10/12 C units AND rho(log10 g_cv, log10 nc1_test) <= -0.8 (T1 review D5 form of MP-8; p = max of the two, intersection-union)", p: null };
  const r = C.filter(u => [CAND.lf[0], CAND.lf[1], { key: "nc1_te", tf: "log10" }].every(c => covVal(u, c) !== null));
  item.n = r.length;
  if (r.length < P.N_MIN) return { ...item, verdict_rule: "NOT_EVALUABLE" };
  const lp = r.map(u => covVal(u, CAND.lf[0])), lt = r.map(u => covVal(u, { key: "nc1_te", tf: "log10" })), lg = r.map(u => covVal(u, CAND.lf[1]));
  const cnt = a => a.filter((v, i) => Math.abs(v - lt[i]) <= P.P8.dex).length, k = cnt(lp), need = Math.ceil(P.P8.frac * r.length);
  const rho = S.spearman(lg, lt);
  const pa = permP(k, rnd => cnt(shuffle(lp, rnd)), "P8a"), pb = permP(isNum(rho) ? -rho : null, rnd => { const v = S.spearman(lg, shuffle(lt, rnd)); return isNum(v) ? -v : null; }, "P8b");
  const out = { ...item, within_0_1_dex: k, need, rho_gcv_nc1te: rho, p_a: pa, p_b: pb, p: isNum(pa) && isNum(pb) ? Math.max(pa, pb) : null,
    verdict_rule: k >= need && isNum(rho) && rho <= P.P8.rho_max ? "SUPPORTED" : "REFUTED" };
  if (dz.on) {
    const dLt = dz.d({ key: "nc1_te", tf: "log10" }), ddex = dz.d(CAND.lf[0]) + dLt, gap = lp.map((v, i) => Math.abs(v - lt[i]));
    const up = gap.filter(g => g > P.P8.dex && g - P.P8.dex <= ddex).length, down = gap.filter(g => g <= P.P8.dex && P.P8.dex - g <= ddex).length;
    const dr = rhoDrift(lg, lt, dz.d(CAND.lf[1]), dLt);
    if (conjFlip([{ ok: k >= need, near: ddex > 0 && countFlip(k, need, up, down) },
      { ok: isNum(rho) && rho <= P.P8.rho_max, near: nearBar(rho, P.P8.rho_max, dr) }]))
      driftOut(out, "a clause lies within its replay drift", { dex_drift: ddex, up, down, rho_drift: dr });
  }
  return out;
}
function evalTwin(Cp, all, driftNc1 = null) {   // S-PL: the PRIMARY rule with the labelled log10 nc1_train in place of the label-free coordinate
  const obs = ownerPrimary(Cp, "acc_nc1", driftNc1);                  // per-outcome replay drift as in the PRIMARY (D15)
  const item = { id: "S-PL", what: "the PRIMARY rule with log10 nc1_train (labelled) as the collapse coordinate (T1 MP-9a)", ...obs, verdict_rule: obs.verdict, p: null };
  delete item.verdict;
  if (obs.verdict === "NOT_EVALUABLE") return item;
  // p: permute nc1_train across the C units (frozen coefficients, rival MAEs unchanged); statistic = - mean over the
  // evaluable outcomes of MAE(acc_nc1) / MAE(best rival), capped at 5 (larger = more support)
  const ev = P.OUTCOMES.filter(o => obs.per_outcome[o] && isNum(obs.per_outcome[o].ratio) && obs.per_outcome[o].ids);
  const byId = Object.fromEntries(Cp.map(u => [u.id, u])), ids = [...new Set(ev.flatMap(o => obs.per_outcome[o].ids))];
  const nc = ids.map(id => covVal(byId[id], CAND.nc1[0]));
  const stat = vals => { const v = Object.fromEntries(ids.map((id, i) => [id, vals[i]]));
    return -mean(ev.map(o => { const po = obs.per_outcome[o], L = all.laws.primary[o].laws.acc_nc1;
      const e = po.ids.map(id => { const u = byId[id], a = covVal(u, L.covs[0]), z = v[id];
        const p = L.beta[0] + L.beta[1] * (a - L.mu[0]) / L.sd[0] + L.beta[2] * (z - L.mu[1]) / L.sd[1];
        return Math.abs(u.y[o] - p); });
      const best = po.mae[po.best_rival]; return best > 0 ? Math.min(5, mean(e) / best) : 5; })); };
  item.p_statistic = "-mean MAE ratio (acc_nc1 / best rival) over the evaluable outcomes";
  item.p = nc.every(isNum) && ev.length ? permP(stat(nc), rnd => stat(shuffle(nc, rnd)), "S-PL") : null;
  return item;
}
function evalFE9(Cp, dz = DZ0, laws = null) {
  const item = { id: "FE9", what: "family-demeaned twin of the PRIMARY: rho_fam(pred acc_coll, y) >= 0.6 in >= 3 of the evaluable outcomes (T1 review D4)", p: null };
  const per = {}, data = {};
  for (const o of P.OUTCOMES) {
    const r = Cp.filter(u => u.kind === "C" && u.evaluable && isNum(u.y[o]) && u.pred[o] && isNum(u.pred[o].acc_coll));
    if (r.length < P.N_MIN) continue;
    data[o] = { p: r.map(u => u.pred[o].acc_coll), y: r.map(u => u.y[o]), f: r.map(u => u.family) };
    per[o] = S.famDemeanedSpearman(data[o].p, data[o].y, data[o].f, P.FAM_MIN_SIZE, P.FAM_MIN_N);
  }
  const ev = Object.keys(data).filter(o => per[o] !== null);
  if (ev.length < P.EVAL_MIN) return { ...item, per_outcome: per, verdict_rule: "NOT_EVALUABLE" };
  const k = ev.filter(o => per[o] >= P.FE.bar).length;
  item.p = permP(k, rnd => ev.filter(o => { const v = S.famDemeanedSpearman(data[o].p, shuffleWithin(data[o].y, data[o].f, rnd), data[o].f, P.FAM_MIN_SIZE, P.FAM_MIN_N);
    return isNum(v) && v >= P.FE.bar; }).length, "FE9");
  const out = { ...item, per_outcome: per, outcomes_pass: k, verdict_rule: k >= P.FE.outcomes_needed ? "SUPPORTED" : "REFUTED" };
  if (dz.on) {                     // an outcome whose rho_fam can change within the drift is uncertain
    const L = o => (laws && laws.primary && laws.primary[o] && laws.primary[o].laws ? laws.primary[o].laws.acc_coll : null);
    const unc = o => nearBar(per[o], P.FE.bar, rankDrift(data[o].p, data[o].y, L(o) ? dz.dPred(L(o)) : Infinity, dz.d(`y.${o}`)));
    const up = ev.filter(o => unc(o) && per[o] < P.FE.bar).length, down = ev.filter(o => unc(o) && per[o] >= P.FE.bar).length;
    if (countFlip(k, P.FE.outcomes_needed, up, down)) driftOut(out, `${up + down} outcomes within their replay drift`, { up, down });
  }
  return out;
}
function knobPairs(all) {       // (knob unit, its F baseline): same arch and seed (train.args), knob != null
  const seedOf = u => { const m = u.train && /--seed (\d+)/.exec(u.train.args || ""); return m ? m[1] : null; };
  return all.filter(u => u.kind === "K").map(k => ({ knob: k, base: all.find(b => b.kind === "F" && b.arch === k.arch && seedOf(b) === seedOf(k)) || null }));
}
function evalMPK(all, dz = DZ0, laws = null) {
  const item = { id: "MP-K", what: "knob pairs (label smoothing, weight decay) x 5 outcomes: the frozen acc + collapse law predicts the change of the outcome better than the accuracy-only law in >= 75% of the cells (15 of 20); a pair with |delta acc| > 0.5 pt is INFO", p: null };
  const cells = [], info = [];
  const dErr = dz.d(CAND.acc[0]);
  let pairUnc = 0;
  for (const { knob, base } of knobPairs(all)) {
    if (!base || !knob.evaluable || !base.evaluable) { info.push({ knob: knob.id, why: "baseline or knob not evaluable" }); continue; }
    const da = 100 * Math.abs(covVal(knob, CAND.acc[0]) - covVal(base, CAND.acc[0]));
    if (dErr > 0 && isNum(da) && Math.abs(da - P.MPK.dacc_pt_max) <= 200 * dErr) pairUnc++;   // the INFO cut may move
    if (!isNum(da) || da > P.MPK.dacc_pt_max) { info.push({ knob: knob.id, base: base.id, dacc_pt: da, why: "|delta acc| > 0.5 pt: INFO" }); continue; }
    for (const o of P.OUTCOMES) {
      const pk = knob.pred[o], pb = base.pred[o];
      if (!pk || !pb || ![knob.y[o], base.y[o], pk.acc_coll, pb.acc_coll, pk.acc, pb.acc].every(isNum)) continue;
      const dObs = knob.y[o] - base.y[o], eC = Math.abs(dObs - (pk.acc_coll - pb.acc_coll)), eA = Math.abs(dObs - (pk.acc - pb.acc));
      cells.push({ knob: knob.id, base: base.id, outcome: o, d_obs: dObs, err_coll: eC, err_acc: eA, win: eC < eA, dacc_pt: da });
    }
  }
  const n = cells.length, w = cells.filter(c => c.win).length;
  if (n < P.MPK.min_cells) return { ...item, n_cells: n, cells, info_pairs: info, verdict_rule: "NOT_EVALUABLE" };
  const out = { ...item, n_cells: n, wins: w, need: Math.ceil(P.MPK.frac * n), cells, info_pairs: info, p: binomTailHalf(w, n),
    p_note: "binomial tail under win probability 1/2 per cell (cells of one pair share its baseline)",
    verdict_rule: w >= Math.ceil(P.MPK.frac * n) ? "SUPPORTED" : "REFUTED" };
  if (dz.on) {                     // a cell's two errors each move by <= 2 dy + 2 dPred (knob and baseline both drift)
    const L = (o, k) => (laws && laws.primary && laws.primary[o] && laws.primary[o].laws ? laws.primary[o].laws[k] : null);
    const dCell = c => { const dy = dz.d(`y.${c.outcome}`), lc = L(c.outcome, "acc_coll"), la = L(c.outcome, "acc");
      return 4 * dy + 2 * (lc ? dz.dPred(lc) : Infinity) + 2 * (la ? dz.dPred(la) : Infinity); };
    const unc = cells.filter(c => nearBar(c.err_coll - c.err_acc, 0, dCell(c)));
    const up = unc.filter(c => !c.win).length, down = unc.filter(c => c.win).length;
    if (pairUnc || countFlip(w, out.need, up, down))
      driftOut(out, `${pairUnc} pairs at the 0.5 pt cut, ${up + down} cells within their replay drift`, { pairs_at_cut: pairUnc, up, down });
  }
  return out;
}
function relDriftProbes(a, b) {   // INFO: the largest relative difference over the PROBE_KEYS numeric leaves, |x-y| / max(|x|, 1e-3)
  let mx = 0; const walk = (x, y) => { if (isNum(x) && isNum(y)) mx = Math.max(mx, Math.abs(x - y) / Math.max(P.REPLAY.rel_floor, Math.abs(x)));
    else if (x && y && typeof x === "object" && typeof y === "object") for (const k of Object.keys(x)) walk(x[k], y[k]); };
  for (const k of PROBE_KEYS) walk(getp(a, k), getp(b, k));
  return mx;
}
function replayRecord(root, prog, id, confTag) {   // the S2 replay output of an anchor: <id>[_fit]<S2 tag>_s2replay
  const re = confTag === null || confTag === undefined ? REPLAY_TAG : new RegExp(`^${confTag.replace(/[^A-Za-z0-9_]/g, "")}_s2replay$`);
  const h = highest(scanRecords(root, prog, id, new Map(), re));
  return h ? h.j : null;
}
function replayDrift(opt, reg, laws, units) {   // D15 (verifier T2-3): per-program triggers, per-input absolute drifts
  const rp = opt.replay ? readJSON(opt.replay) : null;
  if (!rp) return { status: opt.replay ? "UNREADABLE" : "NOT_GIVEN" };
  if (rp.status !== "FAIL") return { status: rp.status };
  const notPass = pre => Object.entries(rp.units || {}).filter(([k, v]) => k.startsWith(pre) && (!v || v.status !== "PASS")).map(([k]) => k);
  const triggered = { t2: notPass("t2:").length > 0, t1: notPass("t1:").length > 0 };
  const out = { status: "FAIL", triggered, units_not_pass: [...notPass("t2:"), ...notPass("t1:"), ...notPass("t3s:")],
    outcome_drift: {}, outcome_drift_nc1: {}, rel_drift_probe_INFO: 0, anchors: [] };
  const pairs = [];
  for (const m of reg.models.filter(x => (x.roles || []).includes("ANCHOR"))) {
    const a = (units || []).find(u => u.id === m.id) || makeUnit(m, null, null);    // the S1 records the laws were fitted on
    const pb = replayRecord(opt.root, "t2", m.id, opt.confTag), tb = replayRecord(opt.root, "t1", m.id, opt.confTag);
    const b = makeUnit(m, pb, tb && tb.model_outcomes);
    const p = { id: m.id, a, b, t2ok: !!(a.probe && b.probe), t1ok: !!(a.t1 && b.t1) };
    pairs.push(p);
    out.anchors.push({ id: m.id, s1: a.sources || null, t2_replay: !!pb, t1_replay: !!tb });
    if (p.t2ok) out.rel_drift_probe_INFO = Math.max(out.rel_drift_probe_INFO, relDriftProbes(a.probe, b.probe));
  }
  const dz = makeDZ(pairs, triggered);
  for (const o of P.OUTCOMES) {    // the PRIMARY / S-PL drift: |dy| + the largest change of the rival and collapse predictions
    const t1o = o === "O1" || o === "O5";
    let mx = 0, mxn = 0;
    const dd = (x, y) => (isNum(x) && isNum(y) ? Math.abs(x - y) : x == null && y == null ? 0 : Infinity);
    for (const p of pairs) {
      const dy = !(t1o ? triggered.t1 : triggered.t2) ? 0 : !(t1o ? p.t1ok : p.t2ok) ? Infinity : dd(p.a.y[o], p.b.y[o]);
      let dp = 0, dpn = 0;
      if (triggered.t2) {
        if (!p.t2ok) { dp = Infinity; dpn = Infinity; } else {
          const pa = predsFor(p.a, laws)[o], pb = predsFor(p.b, laws)[o];
          const dk = keys => (pa && pb ? Math.max(0, ...keys.map(k => dd(pa[k], pb[k]))) : 0);
          dp = dk(["const", "acc", "acc_head", "acc_coll"]); dpn = dk(["const", "acc", "acc_head", "acc_nc1"]);
        }
      }
      mx = Math.max(mx, dy + dp); mxn = Math.max(mxn, dy + dpn);
    }
    if (!pairs.length && (triggered.t1 || triggered.t2)) { mx = Infinity; mxn = Infinity; }
    out.outcome_drift[o] = mx; out.outcome_drift_nc1[o] = mxn;
  }
  Object.defineProperty(out, "dz", { value: dz, enumerable: false });          // not written to the JSON
  return out;
}

// =====================================================================================================================
// the whole confirmation evaluation (pure: units + frozen laws in, record out)
// =====================================================================================================================
function evaluateAll(units, laws, opt = {}) {
  const Cu = units.filter(u => CONF_KINDS.includes(u.kind));
  for (const u of Cu) u.pred = predsFor(u, laws);
  const C = Cu.filter(u => u.kind === "C" && u.evaluable);
  // D15 (verifier T2-3): any replay FAIL tags every label; a decision is NOT_EVALUABLE only when the drift of its own
  // statistic (from the inputs it reads, measured on the anchors of the program that failed) reaches its margin
  const drift = opt.drift && opt.drift.status === "FAIL" ? opt.drift : null;
  const dz = drift && drift.dz ? drift.dz : DZ0;
  const primary = ownerPrimary(Cu, "acc_coll", drift ? drift.outcome_drift : null);
  const all = { laws };
  const sec = [];
  sec.push(evalTwin(Cu, all, drift ? drift.outcome_drift_nc1 : null));
  sec.push(evalMDO3(C, dz));
  sec.push(evalMLF(C, laws.m_lf ? { ...laws.m_lf } : null, dz));
  for (const it of CORR_ITEMS) sec.push(evalCorr(C, it, dz));
  for (const it of FE_ITEMS) sec.push(evalFE(C, it, dz));
  sec.push(evalFE9(Cu, dz, laws));
  for (const it of LAW_ITEMS) sec.push(evalLawItem(C, it, (laws.law_items || {})[it.id], dz));
  const theta = laws.p3b ? laws.p3b.theta : null, p3 = p3bRule(C, theta);
  const p3item = { id: "P3b", what: "at a penult as collapsed as a full-fit resnet56 (plnc1 <= theta frozen on D), the centre margin adds nothing beyond the head statistic picked on rows A: every such C unit d <= +0.005; REFUTED if any has its 99% CI above 0 (a bound rule; no p-value)", n: p3.n_collapsed, p: null, ...p3 };
  if (dz.on && isNum(theta)) {     // membership (plnc1 vs theta), ci_lo vs 0 and d vs +0.005 within their drifts
    const dpl = dz.d({ key: "plnc1_te", tf: "id" }), dd = dz.d("targets.p3b.d"), dlo = dz.d("targets.p3b.ci_lo");
    const rows = C.map(u => ({ id: u.id, pl: num(getp(u.probe, "coords.penult.plnc1_te")), d: yVal(u, "targets.p3b.d"), lo: yVal(u, "targets.p3b.ci_lo") }))
      .filter(r => isNum(r.pl) && isNum(r.d) && isNum(r.lo));
    const memb = rows.filter(r => nearBar(r.pl, theta, dpl)), cand = rows.filter(r => r.pl <= theta || nearBar(r.pl, theta, dpl));
    const unc = cand.filter(r => nearBar(r.lo, P.P3B.lo_max, dlo) || nearBar(r.d, P.P3B.d_max, dd));
    if (memb.length || unc.length) driftOut(p3item, `${memb.length} units at theta, ${unc.length} units at a bound within the drift`,
      { plnc1_drift: dpl, d_drift: dd, ci_lo_drift: dlo, at_theta: memb.map(r => r.id), at_bound: unc.map(r => r.id) });
  }
  sec.push(p3item);
  sec.push(evalP5c(C, dz));
  sec.push(evalICP2(C, dz));
  sec.push(evalP7(units, opt.p7, dz));
  sec.push(evalP8(C, dz));
  sec.push(evalMPK(Cu, dz, laws));
  const ps = sec.map(it => (isNum(it.p) ? it.p : null)), h = S.holm(ps, P.ALPHA);
  sec.forEach((it, i) => {
    it.p_holm = h.adjusted[i];
    it.verdict = it.verdict_rule === "SUPPORTED" && isNum(it.p) ? (h.reject[i] ? "SUPPORTED" : "RULE-MET, HOLM-NS") : it.verdict_rule;
    if (drift) it.tag = "REPLAY-DRIFT";
  });
  if (drift) primary.tag = "REPLAY-DRIFT";
  // INFO
  const info = {};
  const Cr = C.filter(u => covVal(u, CAND.nc1[0]) !== null && isNum(u.y.O1));
  info.MP4 = { rho_log10_nc1_train_vs_O1: S.spearman(Cr.map(u => covVal(u, CAND.nc1[0])), Cr.map(u => u.y.O1)), n: Cr.length,
    note: "T1 MP-4 is noise-limited (T1 review D5): INFO" };
  info.fresh_seeds_F_F20 = Cu.filter(u => (u.kind === "F" || u.kind === "F20") && u.evaluable).map(u => ({ id: u.id,
    y: u.y, pred: Object.fromEntries(P.OUTCOMES.map(o => [o, u.pred[o] ? { acc_coll: u.pred[o].acc_coll, acc: u.pred[o].acc } : null])) }));
  info.nulls_N = units.filter(u => u.kind === "N" && u.probe).map(u => ({ id: u.id, plnc1_te: getp(u.probe, "coords.penult.plnc1_te"),
    g_cv_te: getp(u.probe, "coords.penult.g_cv_te"), nc1_tr: getp(u.probe, "coords.penult.nc1_tr") }));
  info.units = Cu.map(u => ({ id: u.id, kind: u.kind, family: u.family, evaluable: u.evaluable, y: u.y, pred: u.pred,
    sources: u.sources || null, not_evaluable_why: u.not_evaluable_why || null, t1_not_evaluable_why: u.t1_not_evaluable_why || null,
    coords: u.probe ? Object.fromEntries([...CAND.lf, ...CAND.head, CAND.acc[0], CAND.nc1[0]].map(c => [c.name, covVal(u, c)])) : null }));
  const counts = {}; for (const it of sec) counts[it.verdict] = (counts[it.verdict] || 0) + 1;
  return { primary, secondary: { family: "T2 secondaries, Holm over the items with a p-value (alpha 0.05)", m: ps.filter(isNum).length,
    counts, items: Object.fromEntries(sec.map(it => [it.id, it])) }, info };
}

// =====================================================================================================================
// provenance and guards (evaluate)
// =====================================================================================================================
// the instrument behind each read (verifier T2-2): [record field, fitted_on field, file at P2]
const PROBE_CODE = [["sha256", "probe_code", "scripts/collapse_probe.py"], ["core_sha256", "core_sha256", "atlas/b4_core.py"],
  ["collapse_sha256", "collapse_sha256", "atlas/b4_collapse.py"]];
const T1_CODE = [["sha256", "t1_code", "scripts/t1_scoreboard.py"], ["core_sha256", "t1_core", "atlas/b4_core.py"]];
function guards(units, laws, opt) {
  const g = [], pRun = opt.pRun || [];
  const prov = { p1: opt.p1, p2: opt.p2, p_run: pRun, official: !!(opt.p1 && opt.p2 && pRun.length) };
  const CuAll = units.filter(u => CONF_KINDS.includes(u.kind)), Cu = CuAll.filter(u => u.probe), Ct = CuAll.filter(u => u.t1);
  const codes = [...new Set(Cu.map(u => getp(u.probe, "code.sha256")))];
  prov.confirmation_probe_code = codes;
  // one instrument for the fitted discovery records (the P2 laws, or the --refit-tag refit) and the confirmation records;
  // a P1 -> P2 change of any probe file needs the D7 refit on the _p2 re-probes
  const fitted = (opt.refit ? opt.refit.fitted_on : laws.fitted_on) || [], fitSrc = opt.refit ? "the refit discovery records" : "the fitted discovery records";
  prov.code = {};
  const compare = (name, confVals, fitVals, file) => {
    const cv = [...new Set(confVals)], lv = [...new Set(fitVals)];
    prov.code[name] = { confirmation: cv, fit: lv };
    if (cv.length > 1) g.push(`confirmation records ran different ${file}`);
    if (lv.length > 1) g.push(`${fitSrc} ran different ${file}`);
    if (cv.length === 1 && lv.length === 1 && cv[0] !== lv[0])
      g.push(opt.refit ? `${fitSrc} and the confirmation records ran different ${file}` : `probe code changed P1->P2 (${file}): rerun with --refit-tag _p2`);
    if (opt.p2 && cv.length) {
      const at = S.gitSha256At(opt.p2, file, opt.repo); prov.code[name].at_p2 = at;
      if (cv.some(v => v !== at)) g.push(`confirmation records' ${file} is not the one at P2`);
    }
  };
  for (const [k, fk, file] of PROBE_CODE) compare(`t2.${k}`, Cu.map(u => codeStr(getp(u.probe, `code.${k}`))), fitted.map(f => codeStr(f[fk])), file);
  for (const [k, fk, file] of T1_CODE) compare(`t1.${k}`, Ct.map(u => codeStr(getp(u.t1_code, k))),
    fitted.filter(f => f.t1_sha256).map(f => codeStr(f[fk])), file);
  const off = CuAll.filter(u => u.sources && (u.sources.t2_off_phase || u.sources.t1_off_phase)).map(u => u.id);
  if (off.length) g.push(`confirmation units probed outside the confirmation phase: ${off}`);
  prov.touched_twice = CuAll.filter(u => u.sources && (u.sources.t2_dup || u.sources.t1_dup)).map(u => ({ id: u.id, t2: u.sources.t2_dup, t1: u.sources.t1_dup }));
  const fitIds = new Set((laws.fitted_on || []).map(f => f.id)), leak = CuAll.filter(u => (u.probe || u.t1) && fitIds.has(u.id)).map(u => u.id);
  if (leak.length) g.push(`confirmation units in the fit: ${leak}`);
  if (!opt.refit) {                // the fitted records are still the files the laws were fitted on (T2 review #17)
    const bad = [];
    for (const f of fitted) {
      for (const [pk, sk] of [["probe_path", "probe_sha256"], ["t1_path", "t1_sha256"]]) {
        if (!f[pk]) { if (f[sk]) bad.push(`${f.id}: ${sk} without a path`); continue; }
        const p = path.join(opt.root, f[pk]), now = fs.existsSync(p) ? S.fileSha256(p) : null;
        if (now !== f[sk]) bad.push(`${f.id}: ${f[pk]} ${now ? "changed" : "missing"}`);
      }
      const u = units.find(v => v.id === f.id);
      if (u && f.probe_path && u.probe_rel !== f.probe_path) bad.push(`${f.id}: read ${u.probe_rel}, fitted on ${f.probe_path}`);
    }
    prov.fitted_files = { n: fitted.length, bad };
    if (bad.length) g.push(`fitted discovery files differ from the fit: ${bad.slice(0, 5).join("; ")}${bad.length > 5 ? " ..." : ""}`);
  }
  if (pRun.length) {               // prefix match, as in t1_eval.js and t3s_eval.js
    const inRun = rc => pRun.some(p => p && rc && (rc.startsWith(p) || p.startsWith(rc)));
    const offRun = Cu.filter(u => !inRun(getp(u.probe, "code.repo_commit"))).map(u => u.id);
    prov.probes_outside_p_run = offRun; if (offRun.length) g.push(`probes not run at --p-run: ${offRun}`);
  }
  if (opt.p2) {
    const lawsRel = path.relative(opt.repo, path.resolve(opt.laws)).replace(/\\/g, "/");
    prov.laws_at_p2 = S.gitSha256At(opt.p2, lawsRel, opt.repo); prov.laws_read = S.fileSha256(opt.laws);
    if (prov.laws_at_p2 !== prov.laws_read) g.push(`${lawsRel} differs from its P2 version`);
    const me = path.join(opt.repo, "scripts", "collapse_laws.js");
    prov.evaluator_at_p2 = S.gitSha256At(opt.p2, "scripts/collapse_laws.js", opt.repo); prov.evaluator_now = S.fileSha256(me);
    if (prov.evaluator_at_p2 !== prov.evaluator_now) g.push("scripts/collapse_laws.js changed after P2");
    if (opt.p1 && !S.gitIsAncestor(opt.p1, opt.p2, opt.repo)) g.push("P1 is not an ancestor of P2");
  }
  return { guard: g.length ? `NOT_EVALUABLE: ${g.join("; ")}` : null, reasons: g, provenance: prov };
}
function readP7(opt) {
  const rel = "experiments/b4/prereg_p7.json";
  if (opt.p1) { const b = S.gitShow(opt.p1, rel, opt.repo); return b ? { reg: JSON.parse(b.toString("utf8")), source: `${rel}@${opt.p1}` } : { reg: null, source: `${rel} absent at P1` }; }
  const r = readJSON(path.join(opt.repo, rel)); return { reg: r, source: r ? `${rel} (worktree; not official without --p1)` : "absent" };
}

// =====================================================================================================================
// dry run on the committed atlas.json files (INFO): known answers and the blind P7 registration
// =====================================================================================================================
const DRY_NETS = [   // [b4 id, committed atlas dir, depth_family, full_recipe]
  ["resnet20_hub", "atlas_v1_resnet20_s0hub_st3", "resnet20", true], ["resnet20_s1", "atlas_v1_resnet20_s1_st3", "resnet20", true],
  ["resnet20_s2", "atlas_v1_resnet20_s2_st3", "resnet20", true], ["resnet20_s3", "atlas_v1_resnet20_s3_st3", "resnet20", true],
  ["resnet20_s4", "atlas_v1_resnet20_s4_st3", "resnet20", true], ["resnet56_hub", "atlas_v1_resnet56_s0hub_st3", "resnet56", true],
  ["resnet56_s1", "atlas_v1_resnet56_s1", "resnet56", true], ["resnet56_s2", "atlas_v1_resnet56_s2", "resnet56", true],
  ["resnet56_e10", "atlas_v1_resnet56_e10", "resnet56", false], ["resnet56_e20", "atlas_v1_resnet56_e20", "resnet56", false],
  ["resnet56_e40", "atlas_v1_resnet56_e40_st3", "resnet56", false], ["resnet56_e50", "atlas_v1_resnet56_e50", "resnet56", false],
  ["resnet56_e60", "atlas_v1_resnet56_e60", "resnet56", false], ["resnet56_e70", "atlas_v1_resnet56_e70", "resnet56", false],
  ["resnet56_s12m", "atlas_v1_resnet56_s12m", "resnet56", false], ["resnet56_s13m", "atlas_v1_resnet56_s13m", "resnet56", false],
];
function dryUnits(repo) {
  const out = [];
  for (const [id, dir, df, full] of DRY_NETS) {
    const a = readJSON(path.join(repo, "results", dir, "atlas.json")); if (!a) continue;
    const pe = a.per_layer.penult, e = pe.pca_spectrum.eig_top20;
    out.push({ id, dir, depth_family: df, full_recipe: full, nc1: pe.neural_collapse.nc1, g_tr: e[8] / e[9],
      sparse: pe.knn_density.splits.test.sparse_frac, err: 1 - a.meta.accuracy.test });
  }
  return out;
}
function blindP7(U) {
  const f = id => U.find(u => u.id === id), h20 = f("resnet20_hub"), h56 = f("resnet56_hub");
  const out = { schema: "b4_t2_prereg_p7/1", registered_at: "P1 (batch 4 pre-registration), before any Dnew or C unit is extracted or read",
    source: "node scripts/collapse_laws.js --dry-run --emit-p7 on the committed atlas.json of the two chenyaofo hubs (train-reference nc1, landmarks.py:71): " +
      `resnet20 ${h20 ? h20.nc1 : null} (results/atlas_v1_resnet20_s0hub_st3), resnet56 ${h56 ? h56.nc1 : null} (results/atlas_v1_resnet56_s0hub_st3)`,
    coordinate: "probe.coords.penult.nc1_tr of scripts/collapse_probe.py (trimmed inverse; equals the landmarks.py:71 value whenever that pinv is stable; the D15 gate checks the committed nc1 within 1%)",
    rule_points: "log10 nc1_tr linear in network depth between the two hubs (same recipe, same best-val selection); half-width = 2 x the larger seed SD of log10 nc1 within depth 20 or 56 (full-recipe D nets) + 0.10 dex",
    families_monotone: { resnet: ["resnet20_hub", "resnet32", "resnet44", "resnet56_hub"], vgg: ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"],
      mobilenetv2: ["mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4"],
      shufflenetv2: ["shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0"], repvgg: ["repvgg_a0", "repvgg_a1", "repvgg_a2"] },
    ordering_claim: "within each family nc1_tr at the penult decreases along the listed order (deeper or wider = more collapsed)",
    verdict_rule: "SUPPORTED if Kendall tau(order index, nc1_tr) <= -0.66 in >= 4 families with >= 3 evaluable members (and >= 4 such families); REFUTED if >= 4 families are evaluable and < 4 pass; NOT_EVALUABLE otherwise. The two resnet points are INFO clauses (inside / outside their interval). Holm p: within-family permutation of nc1_tr.",
    thresholds: { tau_pass: P.P7.tau_pass, min_members: P.P7.min_members, min_families: P.P7.min_families, pass_families: P.P7.pass_families },
    points: {} };
  if (h20 && h56) {
    const la = Math.log10(h20.nc1), lb = Math.log10(h56.nc1), slope = (lb - la) / (56 - 20);
    const sd = df => { const v = U.filter(u => u.depth_family === df && u.full_recipe).map(u => Math.log10(u.nc1));
      if (v.length < 2) return 0; const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); };
    const hw = 2 * Math.max(sd("resnet20"), sd("resnet56")) + 0.10;
    for (const dpt of [32, 44]) { const l = la + slope * (dpt - 20);
      out.points[`resnet${dpt}`] = { nc1_tr: 10 ** l, lo: 10 ** (l - hw), hi: 10 ** (l + hw), log10: l, halfwidth_dex: hw }; }
  }
  return out;
}
function dryRun(repo) {
  const U = dryUnits(repo), lines = [], say = s => { lines.push(s); console.log(s); };
  say(`[dry-run] INFO only: ${U.length} discovery nets from the committed atlas.json (train-reference fields).`);
  const five = ["resnet20_hub", "resnet56_e10", "resnet56_e20", "resnet56_e40", "resnet56_hub"].map(id => U.find(u => u.id === id));
  let ok1 = false, ok2 = false;
  if (five.every(Boolean)) {
    const x = five.map(u => Math.log10(u.nc1)), y = five.map(u => u.sparse), mx = mean(x), my = mean(y);
    const b = x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / x.reduce((s, v) => s + (v - mx) ** 2, 0), a = my - b * mx;
    ok1 = Math.abs(a - P.AH1.a) < 6e-4 && Math.abs(b - P.AH1.b) < 6e-4;
    say(`[known answer] AH-1 refit on 5 weight sets: a ${a.toFixed(4)} b ${b.toFixed(4)} (committed 0.0224 / -0.1080) -> ${ok1 ? "REPRODUCED" : "NOT REPRODUCED"}`);
  }
  if (U.length === 16) {
    const x9 = S.spearman(U.map(u => u.g_tr), U.map(u => u.sparse)), n9 = S.spearman(U.map(u => u.nc1), U.map(u => u.sparse));
    ok2 = Math.abs(x9 - 0.937) < 0.002 && Math.abs(n9 + 0.942) < 0.002;
    say(`[known answer] X9 Spearman(lambda9/lambda10, sparse) ${x9.toFixed(3)} (committed +0.937); nc1 ${n9.toFixed(3)} (committed -0.942) -> ${ok2 ? "REPRODUCED" : "NOT REPRODUCED"}`);
    const pa = partialSpearman(U.map(u => u.nc1), U.map(u => u.sparse), U.map(u => u.err));
    const r56 = U.filter(u => u.depth_family === "resnet56");
    const pr = partialSpearman(r56.map(u => u.nc1), r56.map(u => u.sparse), r56.map(u => u.err));
    const f3 = v => (isNum(v) ? v.toFixed(3) : String(v));
    say(`[preview] partial Spearman(nc1, sparse | accuracy): all 16 ${f3(pa)}; resnet56 only (${r56.length}) ${f3(pr)} (T1 review D2: -0.891 / -0.247)`);
  }
  const p7 = blindP7(U);
  say(`[P7 blind] ${JSON.stringify(p7.points)}`);
  return { ok: ok1 && ok2, p7, units: U, lines };
}

// =====================================================================================================================
// synthetic results (self-test and tests/collapse_laws_fixture.js): probes and T1 outcomes with known relations
// =====================================================================================================================
const Z_FAMILY = { vgg11_bn: -0.9, vgg13_bn: -1.2, vgg16_bn: -1.4, vgg19_bn: -1.5, mobilenetv2_x0_5: -0.6,
  mobilenetv2_x0_75: -0.75, mobilenetv2_x1_0: -0.9, mobilenetv2_x1_4: -1.05, shufflenetv2_x0_5: -0.5, shufflenetv2_x1_0: -0.65,
  shufflenetv2_x1_5: -0.8, shufflenetv2_x2_0: -0.95, repvgg_a0: -1.1, repvgg_a1: -1.4, repvgg_a2: -1.5, resnet32: -1.0, resnet44: -1.15 };
function latentZ(m, flipOrder) {
  if ((m.roles || []).includes("N")) return 0.5;
  let z;
  const e = /_e(\d+)$/.exec(m.id);
  if (e) z = -0.16 - (Number(e[1]) - 10) * 0.0128;
  else if (/_s1[23]m$/.test(m.id)) z = -0.89;
  else if (Z_FAMILY[m.id] !== undefined) z = Z_FAMILY[m.id];
  else z = { resnet20: -0.77, resnet56: -1.3 }[m.depth_family] ?? -1.0;
  if (flipOrder && (m.roles || []).includes("C") && Z_FAMILY[m.id] !== undefined) z = -2.1 - z;       // reverses each family's order
  if (m.knob && /label_smoothing/.test(m.knob)) z -= 0.3;
  if (m.knob && /weight_decay/.test(m.knob)) z += 0.25;
  return z + 0.02 * (hashU(m.id + "z") - 0.5);
}
const F_OF_Z = { O1: z => 0.001 + 0.01 * (z + 1.4), O2: z => 0.07 - 0.29 * z, O3: z => -0.06 * z - 0.02, O4: z => 0.02 - 0.03 * z,
  O5: z => 0.02 + 0.03 * (z + 1.4) };
function synthUnit(m, scenario) {
  const kind = kindOf(m.roles || []), conf = CONF_KINDS.includes(kind), e = s => 2 * (hashU(m.id + s) - 0.5);
  const z = latentZ(m, scenario === "bad" && conf), g = 0.35 - 0.3 * (z + 0.8);
  const baseId = m.knob ? m.id.replace(/_(ls10|wd5e5)$/, "") : m.id;
  const err = kind === "N" ? 0.9 : 0.05 + 0.04 * hashU(baseId + "err") + (m.knob ? 0.002 : 0);
  const rnd = s => hashU(m.id + scenario + s);                     // confirmation targets in the "bad" scenario
  const bad = scenario === "bad" && conf && kind !== "F";          // F baselines stay lawful: the knob change is what flips
  const nc1 = 10 ** z, nc1te = 10 ** (z + g);
  const pen = { dim: m.penult, nc1_tr: nc1, nc1_tr_verbatim: nc1 * (1 + 1e-9), nc1_te: nc1te, nc1_tetr: 10 ** (z + g + 0.05),
    tt_gap_log10: g + 0.05, te_tr_gap_log10: g, plnc1_te: nc1te * 10 ** ((bad ? 0.4 : 0.02) * e("pl")),
    g_cv_te: bad ? 10 ** (1 + rnd("gcv")) : 10 ** (1.2 - 0.8 * (z + 0.8) + 0.01 * e("gcv")), topk_frac_te: 0.8 - 0.15 * (z + 0.8) + 0.002 * e("tk"),
    sat999_te: 0.5 + 0.3 * hashU(m.id + "sat"), gap_mean_te: 8 + 4 * hashU(m.id + "gap"), msp_def_te: 0.5 * err + 0.01 * hashU(m.id + "msp"),
    err_te: err, acc_te: 1 - err, ncc_disagree_te: 0.02 + 0.03 * (z + 1.4), ece_te: bad ? 0.02 + 0.05 * rnd("ece") : 0.02 + 0.05 * g,
    cdepth_pl: 0.5 - 0.3 * z };
  const lawful = { fpr: 0.0224 - 0.108 * z + 0.003 * e("fpr"), lead: 0.1 + 0.07 * z, gm: -0.02 * z, svhn: 0.9 - 0.05 * z,
    svhnm: -0.05 * z - 0.02, slope: 75 + 28 * z, p6b: -0.03 * z };
  const T = bad ? { fpr: 0.15 + 0.15 * rnd("fpr"), lead: 0.05 * rnd("lead"), gm: 0.02 * rnd("gm"), svhn: 0.9 + 0.05 * rnd("sv"),
    svhnm: 0.05 * rnd("svm"), slope: 50 + 20 * rnd("sl"), p6b: 0.03 * rnd("p6") } : lawful;
  const y = {};
  for (const o of P.OUTCOMES) {
    const f = F_OF_Z[o], amp = { O1: 0.01, O2: 0.3, O3: 0.06, O4: 0.03, O5: 0.03 }[o];
    const randomY = f(-0.9) + amp * (rnd(o) - 0.5);
    const mixedBad = scenario === "mixed" && kind === "C" && ["O3", "O4", "O5"].includes(o);
    y[o] = (bad && kind === "C") || mixedBad ? randomY : f(z) + 0.002 * amp * e(o);
  }
  if (bad && kind === "K") {                                          // knob change opposite to what collapse predicts
    const zb = latentZ({ ...m, id: baseId, knob: null }, false);
    for (const o of P.OUTCOMES) y[o] = F_OF_Z[o](zb) - (F_OF_Z[o](z) - F_OF_Z[o](zb));
  }
  const probe = { schema: "b4_t2_probe/1", program: "t2", unit: m.id, phase: conf ? "confirmation" : "discovery",
    code: { sha256: "0".repeat(64), repo_commit: null }, dumps: [{ sealed: conf }],
    instrument: { weights_status: "PASS", head_check_status: "PASS" },
    coords: { penult: pen, pre_tap: "pre" },
    targets: { do3: { fpr_trainref: T.fpr }, ic_p2: { fpr: bad ? 0.09 : 0.05 + 0.004 * e("ic"), n_cal: 1500, n_test: 1500, curve: {} },
      margin: { lead_md: T.lead, gap_minus_msp: T.gm },
      p3b: bad ? { d: 0.05, ci_lo: 0.02, ci_hi: 0.08, head_pick: "gap" } : { d: 0.001 + 0.002 * e("d"), ci_lo: -0.02, ci_hi: 0.02, head_pick: "gap" },
      ood: { svhn: { knn_l2_minus_besthead: T.svhnm, auc_knn_l2: T.svhn } }, harm: { slope: T.slope, H10: y.O2 },
      harm_committed: { hold_n: 5, hold_viol: bad ? 2 : 0, holdout_hold_viol: 0 }, overconf: { shift_mean: y.O4 }, taps: { p6b: T.p6b } },
    outcomes: { O2: y.O2, O3: y.O3, O4: y.O4 },
    targets_se: { fpr_trainref: 0.003, lead_md: 0.004, O2: 0.01, O3: 0.005, O4: 0.002, p6b: 0.004, harm_slope: 1.5 } };
  const mo = { schema: "b4_model_outcomes/1", unit: m.id, layout: "fit", phase: probe.phase, O1: { value: y.O1 }, O5: { value: y.O5 } };
  return { probe, mo };
}
function synthUnits(reg, scenario, { drop = [], noT1 = [] } = {}) {
  return reg.models.map(m => { const s = synthUnit(m, scenario);
    return makeUnit(m, drop.includes(m.id) ? null : s.probe, noT1.includes(m.id) || drop.includes(m.id) ? null : s.mo,
      { probe_sha256: "synthetic", t1_sha256: "synthetic" }); });
}

// =====================================================================================================================
// self-test
// =====================================================================================================================
function selftest(repo) {
  const res = [], chk = (name, c, info = "") => { res.push(!!c); console.log(c ? "PASS" : "FAIL", name, c ? "" : info); };
  chk("lsq recovers y = 1 + 2 x1 - 3 x2", (() => { const Z = [], y = []; for (let i = 0; i < 20; i++) { const a = Math.sin(i), b = Math.cos(3 * i);
    Z.push([1, a, b]); y.push(1 + 2 * a - 3 * b); } const bt = lsq(Z, y); return bt && Math.abs(bt[0] - 1) + Math.abs(bt[1] - 2) + Math.abs(bt[2] + 3) < 1e-9; })());
  chk("lsq refuses a collinear design", lsq([[1, 1, 2], [1, 2, 4], [1, 3, 6], [1, 4, 8]], [1, 2, 3, 4]) === null);
  chk("kendall reversed = -1, ties handled", kendall([0, 1, 2, 3], [4, 3, 2, 1]) === -1 && Math.abs(kendall([1, 2, 3], [1, 1, 2]) - 0.8164965809) < 1e-9);
  chk("partial Spearman given a common cause ~ 0", (() => { const z = [...Array(40)].map((_, i) => i), x = z.map(v => v + (v % 3)), y = z.map(v => v - (v % 2));
    const p = partialSpearman(x, y, z); return isNum(p) && Math.abs(p) < 0.35 && S.spearman(x, y) > 0.9; })());
  chk("binomial tail: P(X >= 15 | 20, 1/2) = 0.020695", Math.abs(binomTailHalf(15, 20) - 0.0206947327) < 1e-9 && binomTailHalf(0, 5) === 1);
  chk("permutation p of a perfect rank correlation at n 12 <= 1e-3", permP(1, rnd => S.spearman([...Array(12).keys()], shuffle([...Array(12).keys()], rnd)), "t", 5000) < 1e-3);
  chk("shuffleWithin keeps each family's multiset", (() => { const a = [1, 2, 3, 4, 5], f = ["a", "a", "b", "b", "b"], s = shuffleWithin(a, f, mulberry32(3));
    return s.slice(0, 2).sort().join() === "1,2" && s.slice(2).sort().join() === "3,4,5"; })());
  // replay-drift bounds of the secondaries (D15; verifier T2-3)
  chk("rhoDrift: 0 when no pair lies within 2d; otherwise it covers a brute-force perturbation of every value by <= d", (() => {
    const x = [0, 0.05, 1, 2, 2.03, 3, 4, 5, 5.01, 6, 7, 8], y = [3, 7, 1, 10, 0, 5, 11, 2, 8, 4, 9, 6], d = 0.03;
    if (rhoDrift(x, y, 0.004, 0.4) !== 0 || closePairs(x, d) !== 3) return false;
    const bound = rhoDrift(x, y, d, d), r0 = S.spearman(x, y), rnd = mulberry32(11);
    let mx = 0;
    for (let b = 0; b < 4000; b++) mx = Math.max(mx, Math.abs(S.spearman(x.map(v => v + d * (2 * rnd() - 1)), y.map(v => v + d * (2 * rnd() - 1))) - r0));
    return mx > 0 && mx <= bound && Math.abs(bound - 12 * 3 / (12 * 11)) < 1e-12 && rankDrift(x, y, d, 0) === Infinity;
  })());
  chk("countFlip / conjFlip truth tables", !countFlip(5, 3, 0, 2) && countFlip(5, 3, 0, 3) && countFlip(2, 3, 1, 0) && !countFlip(1, 3, 1, 0)
    && !conjFlip([{ ok: false, near: false }, { ok: true, near: true }]) && conjFlip([{ ok: true, near: true }, { ok: true, near: false }])
    && conjFlip([{ ok: false, near: true }]) && !conjFlip([{ ok: true, near: false }]));
  // the D12 rule on the b4_primary.js fixtures (4 checks, verbatim logic)
  const fam = ["resnet", "vgg", "vgg", "vgg", "mbv2", "mbv2", "mbv2", "shuf", "shuf", "shuf", "repvgg", "repvgg"];
  let s0 = 7; const r0 = () => ((s0 = (s0 * 16807) % 2147483647) / 2147483647 - 0.5);
  const make = (good, withinReversed = false) => fam.map((f, i) => { const c = i / 11, fo = { resnet: 0, vgg: 0.3, mbv2: -0.2, shuf: 0.1, repvgg: -0.1 }[f];
    const y = {}, pred = {}; for (const o of P.OUTCOMES) { y[o] = fo + c + 0.02 * r0(); const within = withinReversed ? fo + (1 - c) * 0.05 + 0.5 : c;
      pred[o] = { const: 0.5, acc: 0.5 + 0.3 * r0(), acc_head: 0.5 + 0.3 * r0(), acc_coll: good ? (withinReversed ? within : fo + c + 0.02 * r0()) : 0.5 + 0.6 * r0() }; }
    return { id: `u${i}`, family: f, kind: "C", y, pred }; });
  const a = ownerPrimary(make(true)), b = ownerPrimary(make(false)), c = ownerPrimary(make(true, true)), d = ownerPrimary(make(true).slice(0, 9));
  chk("PRIMARY: good law -> SUPPORTED", a.verdict === "SUPPORTED", a.verdict);
  chk("PRIMARY: noise law -> REFUTED", b.verdict === "REFUTED", `${b.verdict} ${b.wins} ${b.losses}`);
  chk("PRIMARY: family-offset-only law -> not SUPPORTED", c.verdict !== "SUPPORTED", c.verdict);
  chk("PRIMARY: 9 units -> NOT_EVALUABLE", d.verdict === "NOT_EVALUABLE", d.verdict);
  // end to end on synthetic units (the real registry's ids, families and run groups)
  const reg = readJSON(path.join(repo, "experiments", "b4", "models.json"));
  if (reg) {
    const good = synthUnits(reg, "good"), laws = fitAll(good);
    chk("fit: all five PRIMARY outcomes frozen on D21", P.OUTCOMES.every(o => laws.primary[o].status === "FROZEN" && laws.primary[o].n === 21),
      JSON.stringify(P.OUTCOMES.map(o => [laws.primary[o].status, laws.primary[o].n])));
    chk("fit: the collapse law picks a label-free coordinate", P.OUTCOMES.every(o => CAND.lf.some(z => z.name === laws.primary[o].laws.acc_coll.covs[1].name)));
    chk("fit: every law item frozen with its registered direction", LAW_ITEMS.every(it => laws.law_items[it.id].direction_as_registered === true),
      JSON.stringify(LAW_ITEMS.map(it => [it.id, laws.law_items[it.id].direction_as_registered])));
    const p7 = blindP7([{ id: "resnet20_hub", nc1: 0.17, depth_family: "resnet20", full_recipe: true },
      { id: "resnet56_hub", nc1: 0.05, depth_family: "resnet56", full_recipe: true }]);
    const eg = evaluateAll(good, laws, { p7 });
    chk("evaluate (lawful synthetic C): PRIMARY SUPPORTED", eg.primary.verdict === "SUPPORTED", JSON.stringify(eg.primary.per_outcome));
    const notSup = Object.values(eg.secondary.items).filter(it => it.verdict !== "SUPPORTED").map(it => `${it.id}:${it.verdict}`);
    chk("evaluate (lawful synthetic C): every secondary SUPPORTED after Holm", notSup.length === 0, notSup.join(" "));
    const eb = evaluateAll(synthUnits(reg, "bad"), laws, { p7 });
    const sup = Object.values(eb.secondary.items).filter(it => it.verdict === "SUPPORTED").map(it => it.id);
    chk("evaluate (flipped confirmation): PRIMARY REFUTED", eb.primary.verdict === "REFUTED", `${eb.primary.verdict} ${eb.primary.wins}/${eb.primary.losses}`);
    chk("evaluate (flipped confirmation): no secondary SUPPORTED", sup.length === 0, sup.join(" "));
  } else chk("registry experiments/b4/models.json readable", false);
  const ok = res.every(Boolean);
  console.log(ok ? `selftest PASS (${res.length} checks)` : "selftest FAIL");
  return ok ? 0 : 1;
}

// =====================================================================================================================
// main
// =====================================================================================================================
function refuseExisting(p) { if (p && fs.existsSync(p)) { console.error(`refused: ${p} exists (append-only)`); process.exit(2); } }
function writeJSON(p, obj) { fs.mkdirSync(path.dirname(path.resolve(p)), { recursive: true }); fs.writeFileSync(p, JSON.stringify(obj, null, 1) + "\n"); }
function options() {
  const repo = path.resolve(arg("--repo", REPO_DEFAULT));
  return { repo, root: path.resolve(arg("--root", path.join(repo, "results"))), registry: path.resolve(arg("--registry", path.join(repo, "experiments", "b4", "models.json"))),
    p1: arg("--p1"), p2: arg("--p2"), pRun: arg("--p-run") ? arg("--p-run").split(",").filter(Boolean) : [],
    laws: path.resolve(arg("--laws", path.join(repo, "experiments", "b4", "laws_frozen.json"))), replay: arg("--replay"),
    // optional exact-tag filters (verifier T2-1; default: every tag is scanned). --fit: --tag filters the S1 (discovery)
    // outputs. --evaluate: --tag filters the S2 (confirmation) outputs, --disc-tag the S1 outputs
    discTag: has("--fit") ? arg("--tag") : arg("--disc-tag"), confTag: has("--fit") ? null : arg("--tag"), refitTag: arg("--refit-tag") };
}
// --evaluate (in-process for tests/collapse_laws_fixture.js; requireFlags: false there only)
function runEvaluate(o, { requireFlags = true } = {}) {
  const laws0 = readJSON(o.laws); if (!laws0 || laws0.schema !== "b4_t2_laws/1") throw new Error(`${o.laws}: not a b4_t2_laws/1 file`);
  const { reg, units: u0 } = loadUnits({ root: o.root, registry: o.registry, discTag: o.discTag, confTag: o.confTag });
  let laws = laws0, refit = null, units = u0;
  if (o.refitTag) {                  // D7 failure path: the frozen procedure refitted on the _p2 re-probes; P2 laws become INFO
    const r = loadUnits({ root: o.root, registry: o.registry, discTag: o.discTag, readConf: false, refit: true });
    refit = fitAll(r.units, { repo: o.repo }); laws = { ...laws0, ...refit };
    const byId = new Map(r.units.map(u => [u.id, u]));      // discovery members (P7, the replay anchors) from the re-probes too
    units = u0.map(u => (CONF_KINDS.includes(u.kind) ? u : byId.get(u.id) || u));
  }
  const gd = guards(units, laws, { ...o, refit, refitTag: o.refitTag });
  if (requireFlags && (!o.p1 || !o.p2 || !(o.pRun && o.pRun.length))) {       // verifier T2-4: never a silent unofficial verdict
    gd.reasons.push("--p1, --p2 and --p-run <S1 HEAD>,<S2 HEAD> are required");
    gd.guard = `NOT_EVALUABLE: ${gd.reasons.join("; ")}`;
  }
  const p7 = readP7(o), drift = replayDrift(o, reg, laws, units);
  const res = { schema: "b4_t2_eval/1", created_utc: new Date().toISOString(), laws_file: path.relative(o.repo, o.laws).replace(/\\/g, "/"),
    laws_sha256: S.fileSha256(o.laws), laws_applied: refit ? `refit on ${o.refitTag} (D7 failure path; the P2 laws are INFO)` : "frozen at P2",
    guard: gd.guard, provenance: gd.provenance, p7_registration: p7.source, replay: drift, nc1_gate_at_fit: laws0.nc1_gate || null };
  if (gd.guard) { res.primary = { verdict: "NOT_EVALUABLE", why: gd.guard }; res.secondary = { items: {}, why: gd.guard }; }
  else Object.assign(res, evaluateAll(units, laws, { p7: p7.reg, drift }));
  return res;
}
function main() {
  if (has("--selftest")) return process.exit(selftest(path.resolve(arg("--repo", REPO_DEFAULT))));
  if (has("--dry-run")) {
    const o = options(), r = dryRun(o.repo), e = arg("--emit-p7");
    if (e) { refuseExisting(e); writeJSON(e, r.p7); console.log(`[dry-run] wrote ${e}`); }
    return process.exit(r.ok ? 0 : 1);
  }
  if (has("--fit")) {
    const o = options(), out = arg("--out"); if (!out) throw new Error("--out required"); refuseExisting(out);
    const { units } = loadUnits({ root: o.root, registry: o.registry, discTag: o.discTag, readConf: false });
    const D = D21(units), bad = D.filter(u => u.probe.phase !== "discovery").map(u => u.id);
    if (bad.length) throw new Error(`discovery units not probed in the discovery phase: ${bad}`);
    const laws = fitAll(units, { repo: o.repo });
    // one instrument for every fitted record (verifier T2-2): the probe files and the T1 scoreboard code behind O1 / O5
    const sets = laws.code_sets, want = [["probe_code", "scripts/collapse_probe.py"], ["core_sha256", "atlas/b4_core.py"],
      ["collapse_sha256", "atlas/b4_collapse.py"], ["t1_code", "scripts/t1_scoreboard.py"], ["t1_core", "atlas/b4_core.py"]];
    for (const [k, file] of want) {
      if (sets[k].length > 1 || (sets[k].length === 0 && !k.startsWith("t1_")))
        throw new Error(`the discovery records ran ${sets[k].length} versions of ${file} (${k}): ${sets[k]}`);
    }
    const prov = S.provenance({ repo: o.repo, p1: o.p1, files: ["scripts/collapse_probe.py", "scripts/collapse_laws.js", "atlas/b4_collapse.py",
      "atlas/b4_core.py", "scripts/t1_scoreboard.py"] });
    if (o.p1) for (const [k, file] of want) {
      if (sets[k].length === 1 && prov.files[file].at_p1 !== sets[k][0]) throw new Error(`the discovery records did not run ${file} as of P1 (${k})`);
    }
    const doc = { schema: "b4_t2_laws/1", created_utc: new Date().toISOString(), p1: o.p1, provenance: prov, constants: P, candidates: CAND,
      collapse_sign: COLLAPSE_SIGN, law_items_spec: LAW_ITEMS, corr_items_spec: CORR_ITEMS, disc_tag: o.discTag, ...laws };
    writeJSON(out, doc);
    for (const oc of P.OUTCOMES) { const f = laws.primary[oc];
      console.log(`[fit] ${oc} ${f.status} n ${f.n}${f.status === "FROZEN" ? `  acc ${f.laws.acc.covs.map(c => c.name)}  head ${f.laws.acc_head.covs[1].name}  coll ${f.laws.acc_coll.covs[1].name} (LORO ${f.laws.acc_coll.loro_mae.toExponential(3)} vs acc ${f.laws.acc.loro_mae.toExponential(3)})` : ` (${f.why})`}`); }
    for (const it of LAW_ITEMS) { const f = laws.law_items[it.id]; console.log(`[fit] ${it.id.padEnd(4)} ${f.status}${f.law ? ` ${f.law.covs[0].name} direction_as_registered ${f.direction_as_registered}` : ""}`); }
    console.log(`[fit] P3b theta ${laws.p3b.theta}; on D: ${laws.p3b.on_D.verdict_rule}; nc1 gate ${laws.nc1_gate && laws.nc1_gate.status}; wrote ${out}`);
    return;
  }
  if (has("--evaluate")) {
    const o = options(), out = arg("--json"); if (!out) throw new Error("--json required"); refuseExisting(out);
    const res = runEvaluate(o);
    writeJSON(out, res);
    const unoff = res.provenance && res.provenance.official === false ? " [UNOFFICIAL: provenance.official=false]" : "";
    console.log(`[eval] PRIMARY ${res.primary.verdict}${isNum(res.primary.wins) ? ` (wins ${res.primary.wins}, losses ${res.primary.losses}, outcomes ${res.primary.outcomes_evaluable})` : ""}${unoff}${res.guard ? `  ${res.guard}` : ""}`);
    for (const [k, v] of Object.entries(res.secondary.items || {})) console.log(`[eval] ${k.padEnd(6)} ${String(v.verdict).padEnd(24)} p ${isNum(v.p) ? v.p.toExponential(2) : "-"}  p_holm ${isNum(v.p_holm) ? v.p_holm.toExponential(2) : "-"}${v.rule8 && v.rule8.flags.length ? `  [${v.rule8.flags}]` : ""}`);
    console.log(`[eval] wrote ${out}`);
    return;
  }
  console.error("usage: --selftest | --dry-run [--emit-p7 F] | --fit --out F [--p1 P1] | --evaluate --json F --p1 P1 --p2 P2 (see header)");
  process.exit(1);
}
module.exports = { P, CAND, LAW_ITEMS, CORR_ITEMS, FE_ITEMS, PROBE_KEYS, loadUnits, makeUnit, fitAll, evaluateAll, ownerPrimary,
  guards, runEvaluate, replayDrift, blindP7, dryRun, synthUnit, synthUnits, predsFor, kendall, partialSpearman, binomTailHalf, lsq,
  rhoDrift, rankDrift, closePairs, countFlip, conjFlip };
if (require.main === module) main();
