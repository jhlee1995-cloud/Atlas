// A4b evaluator (docs/plans/STAGE2B.md). Committed with the pre-registration (commit P), before any A4b file exists, so
// the analysis is frozen with the rules: gates G0b-G0e and R0, the B20+ band, the matched rung (M11, Mc = M12/M13, the
// interpolation branch), the D-ID / D-COLL outcomes (ATLAS_STATUS row 11), the pair-metric levels (P20), the depth-56
// claims C1-C5a, the s1-s2 core items (critic statuses decide; recomputed as a cross-check), KILL-56, T56, the SHAPE
// labels, the d56 tags and the M56 margin items. node only; reads the pulled results/*/atlas.json,
// */compare_vs_*/deformation.json, critic_*/critic.json, train_*/train.json and the A4b check dir.
//   node scripts/a4b_eval.js --p <P> --p-run <P_run> [--check-dir instrument_check_a4b] [--json <out.json>]
//   node scripts/a4b_eval.js --dry-run     (Stage 2 stand-ins on committed files: s1 := r56 hub, s2 := e40, target :=
//                                           e40's accuracy; reproduces the Stage 2 numbers; T56, i2, check.json and
//                                           G0e do not exist there and are SKIPPED)
//   bash scripts/a4b_eval_fixture.sh       (committed atlases under A4b names; matched, interp and below scenarios)
// Run from the repo root (--root defaults to ./results; --root <dir> reads another results tree, as the fixture does).
// --p / --p-run are git revisions of this repo (G0e); --check-dir is relative to --root (a relaunch: ..._r2).
// Formulas follow atlas/critic.py (spearman with average ranks, alias layer dropped, decod MAD/rho, commit shift)
// and scripts/d1_distance_only.js (D1). Every threshold below is the STAGE2B.md one; change neither without an
// amendment. An evaluator bug found after the run is fixed only by a committed amendment listed in SESSION.md, and
// the outcomes are then reported under both versions.
"use strict";
const fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
const argv = process.argv.slice(2);
const arg = (k, d = null) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const DRY = argv.includes("--dry-run");
const JSON_OUT = arg("--json");
const ROOT = arg("--root", "results");
const CHECK = arg("--check-dir", "instrument_check_a4b");
const P_ARG = arg("--p"), PRUN_ARG = arg("--p-run");
const REPO = path.resolve(__dirname, "..");
const P = (...a) => path.join(ROOT, ...a);
const _cache = new Map();
const loadJ = p => { if (!fs.existsSync(p)) return null; try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } };
const load = d => { if (!_cache.has(d)) _cache.set(d, loadJ(P(d, "atlas.json"))); return _cache.get(d); };
const U = { ok: "\u2705", y: "\u{1F7E1}", x: "\u2717", open: "\u2B1C" };

// ---- runs --------------------------------------------------------------------------------------------------------
const R20 = DRY ? ["atlas_v1_resnet20_s0hub_st2", "atlas_v1_resnet20_s1_st2", "atlas_v1_resnet20_s2_st2",
                   "atlas_v1_resnet20_s3", "atlas_v1_resnet20_s4"]
                : ["s0hub", "s1", "s2", "s3", "s4"].map(s => `atlas_v1_resnet20_${s}_st3`);
const HUB = DRY ? "atlas_v1_resnet56_s0hub" : "atlas_v1_resnet56_s0hub_st3";
const TWIN = DRY ? null : "atlas_v1_resnet56_s0hub_ref1";
const SEEDS = DRY ? ["atlas_v1_resnet56_s0hub", "atlas_v1_resnet56_e40"] : ["atlas_v1_resnet56_s1", "atlas_v1_resnet56_s2"];
// e40 enters only as the same-session re-measure e40_st3 (review item 7); its accuracy is the Stage 2 train.json
const LADDER = DRY ? [10, 20, 40].map(E => ({ E, atlas: `atlas_v1_resnet56_e${E}` }))
                   : [40, 50, 60, 70, 90].map(E => ({ E, atlas: E === 40 ? "atlas_v1_resnet56_e40_st3" : `atlas_v1_resnet56_e${E}` }));
const REPL = DRY ? [] : [12, 13].map(s => ({ name: `M${s}`, seed: s, atlas: `atlas_v1_resnet56_s${s}m`, train: `train_resnet56_s${s}m` }));
const A4B_ATLASES = DRY ? [] : [...R20, HUB, TWIN, ...LADDER.map(r => r.atlas), ...REPL.map(r => r.atlas), ...SEEDS];
const MARGIN_OF = n => n.replace(/^atlas_/, "margin_");
const A4B_MARGINS = DRY ? [] : [...R20, HUB, ...SEEDS, ...LADDER.filter(r => r.E !== 40).map(r => r.atlas), ...REPL.map(r => r.atlas)].map(MARGIN_OF);
const CRIT = DRY ? { pair: "critic_v1_resnet20_st2_band", twin: null, margin: null }
                 : { pair: "critic_v1_resnet56_s1_s2", twin: "critic_v1_resnet56_hub_noise", margin: "critic_margin_v1_resnet56_s1_s2" };
const A4B_CRITICS = DRY ? [] : ["critic_v1_resnet56_s1_s2", "critic_v1_resnet56_s0_s1_s2", "critic_v1_resnet56_hub_noise",
  "critic_v1_resnet20_st3_band", "critic_v1_scale3_r20_r56seeds", "critic_v1_scale3_r20_r56matched", "critic_margin_v1_resnet56_s1_s2"];
const TARGET = DRY ? loadJ(P("train_resnet56_e40", "train.json")).final_test_acc_10k
                   : loadJ(P("norm_check_resnet20", "norm_check.json")).chenyaofo.acc_10k;
const WINDOW = 0.005;
const SHARP = ["luminance_mean", "spectral_slope", "saturation_mean", "hue_sin", "colorfulness", "orientation_entropy",
               "class", "corruption_family", "corruption_type", "severity"];
const SIDE = { id: "HIGH", sep_ratio: "HIGH", bridge: "HIGH", nc1: "LOW" };               // Stage 2 m* sides
const CORE_KEYS = ["S1", "S3", "S5", "S7", "S8", "S9"];
// G0e: files frozen between P and P_run (integration D17 item 3); every A4b and B1 manifest by name or pattern
const FROZEN = [
  ...["s0hub", "s1", "s2", "s3", "s4"].flatMap(s => [`atlas_v1_resnet20_${s}_st3`, `margin_v1_resnet20_${s}_st3`]),
  ...["s0hub_st3", "s0hub_ref1", "e40_st3", "e50", "e60", "e70", "e90", "s12m", "s13m", "s1", "s2"].map(s => `atlas_v1_resnet56_${s}`),
  ...["s0hub_st3", "s1", "s2", "e50", "e60", "e70", "e90", "s12m", "s13m"].map(s => `margin_v1_resnet56_${s}`),
].map(n => `experiments/queue/${n}.yaml`).concat(["experiments/queue/margin_b1_*.yaml", "docs/plans/STAGE2B.md",
  "docs/plans/STAGE2.md", "docs/plans/B1_VIT_MARGIN.md", "scripts/a4b_eval.js", "scripts/b1_verdicts.js", "scripts/b1_gate.py",
  "experiments/tolerances_default.yaml", "ATLAS_STATUS.md"]);

// ---- math (critic / d1 formulas) -------------------------------------------------------------------------------
const rank = a => { const o = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), r = new Array(a.length);
  for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
    for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
const mean = x => x.reduce((a, b) => a + b, 0) / x.length;
const pearson = (x, y) => { const mx = mean(x), my = mean(y); let sxy = 0, sx = 0, sy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sx += (x[i] - mx) ** 2; sy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sx * sy); };
const spearman = (x, y) => pearson(rank(x), rank(y));
const median = x => { const s = [...x].sort((a, b) => a - b), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const std0 = x => { const m = mean(x); return Math.sqrt(mean(x.map(v => (v - m) ** 2))); };
const d4 = x => Math.round(x * 1e4) / 1e4;
const num = v => typeof v === "number" && Number.isFinite(v);
const ge = (v, t) => (num(v) ? v >= t : null);                 // null when the value is missing (review item 11)
const le = (v, t) => (num(v) ? v <= t : null);
const lt = (v, t) => num(v) && v < t;                          // a missing value never counts as below a threshold
const f = (v, k = 4) => (v === null || v === undefined || Number.isNaN(v) ? "-" : typeof v === "number" ? v.toFixed(k) : String(v));
const safe = fn => { try { const v = fn(); return v === undefined ? null : v; } catch (e) { return null; } };

// ---- per-atlas quantities --------------------------------------------------------------------------------------
const is56 = a => /resnet56/.test(a.meta.arch);
const alias = a => (is56(a) ? "layer3.8" : "layer3.2");                   // == penult under gap pooling
const L31 = a => (is56(a) ? "layer3.5" : "layer3.1");                     // the aligned layer3.1 tap
const layers10 = a => a.layers.filter(l => l !== alias(a));              // the critic's alias-collapsed list
const pen = a => a.per_layer.penult;
const dists = a => { const C = pen(a).class_centers.centers, d = [];
  for (let i = 0; i < C.length; i++) for (let j = i + 1; j < C.length; j++) d.push(Math.sqrt(C[i].reduce((s, v, k) => s + (v - C[j][k]) ** 2, 0)));
  return d; };
const FIELD = {
  id: a => pen(a).twonn_id.id,
  sep_ratio: a => pen(a).class_centers.sep_ratio,
  bridge: a => { const d = dists(a); return mean(d) / mean(pen(a).class_centers.radius); },  // mean off-diag / mean radius
  nc1: a => pen(a).neural_collapse.nc1,
  class_excess: a => pen(a).linear_probes.factors.class.excess,
  nc_acc_test: a => pen(a).class_centers.nearest_center_acc_test,
  acc_test: a => a.meta.accuracy.test,
  ref_fit: a => a.meta.accuracy.ref,
  cv: a => { const d = dists(a); return std0(d) / mean(d); },
};
const valAt = (k, name) => { const a = name ? load(name) : null; return a ? safe(() => FIELD[k](a)) : null; };
function claims(a) {                                                      // C1-C5a, the atlas's own tap names
  const ids = a.layers.map(l => a.per_layer[l].twonn_id.id);
  const pk = a.layers[ids.indexOf(Math.max(...ids))], drop = 1 - FIELD.id(a) / Math.max(...ids);
  const pf = a.cross_layer.commit_layer.per_factor, w = k => pf[k].washout;
  const coh = (c, s) => pen(a).corruption_displacement.splits[`corrupt__${c}__s${s}`].coherence;
  const c4 = []; for (const [c, s] of [["defocus_blur", 5], ["motion_blur", 3], ["motion_blur", 5]])
    for (const n of ["gaussian_noise", "shot_noise"]) c4.push(coh(n, s) - coh(c, s));
  const sf = (c, s) => pen(a).knn_density.splits[`corrupt__${c}__s${s}`].sparse_frac;
  const c5 = ["defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast", "pixelate", "jpeg_compression"].map(c => sf(c, 5) - sf(c, 1));
  const peakSet = is56(a) ? ["layer3.0", "layer3.5"] : ["layer3.0", "layer3.1"];
  const commitSet = is56(a) ? ["layer3.0", "layer3.5", "layer3.8"] : ["layer3.0", "layer3.1", "layer3.2"];
  return {
    C1: peakSet.includes(pk) && drop >= 0.30, C1_peak: pk, C1_drop: drop,
    C2: w("luminance_mean") > 0.30 && w("highfreq_ratio") < 0.20 && w("spectral_anisotropy") < 0.20,
    C2_vals: [w("luminance_mean"), w("highfreq_ratio"), w("spectral_anisotropy")],
    C3: commitSet.includes(pf.class.commit_layer), C3_layer: pf.class.commit_layer,
    C4: c4.filter(v => v > 0).length, C4_min: Math.min(...c4), C5a: c5.filter(v => v > 0).length, C5a_min: Math.min(...c5),
  };
}
const claimsHold = c => !!c && c.C1 && c.C2 && c.C3 && c.C4 === 6 && c.C5a === 8;

// ---- pair quantities (critic formulas) -------------------------------------------------------------------------
const upperSep = (a, L) => { const M = a.per_layer[L].class_adjacency.sep_matrix, v = [];
  for (let i = 0; i < M.length; i++) for (let j = i + 1; j < M.length; j++) v.push(M[i][j]); return v; };
const sepRho = (a, la, b, lb) => { const ua = upperSep(a, la), ub = upperSep(b, lb), x = [], y = [];
  for (let i = 0; i < ua.length; i++) if (ua[i] > 0 && ub[i] > 0) { x.push(ua[i]); y.push(ub[i]); }
  return x.length > 3 ? spearman(x, y) : null; };
const prof = (a, fn) => layers10(a).map(l => fn(a.per_layer[l]));
function idPair(a, b) { const pa = prof(a, p => p.twonn_id.id), pb = prof(b, p => p.twonn_id.id);
  return { rho: spearman(pa, pb), shift: Math.abs(pa.indexOf(Math.max(...pa)) - pb.indexOf(Math.max(...pb))) }; }
function decodPair(a, b, fac) { const g = p => { const r = ((p.linear_probes || {}).factors || {})[fac] || {};
    const v = r.kind === "categorical" ? r.excess : r.score; return v === undefined || v === null ? NaN : v; };
  const pa = prof(a, g), pb = prof(b, g), x = [], y = [];
  for (let i = 0; i < pa.length; i++) if (Number.isFinite(pa[i]) && Number.isFinite(pb[i])) { x.push(pa[i]); y.push(pb[i]); }
  if (x.length < 3) return null;
  const rho = std0(x) > 1e-9 && std0(y) > 1e-9 ? spearman(x, y) : 1.0, mad = mean(x.map((v, i) => Math.abs(v - y[i])));
  return { rho, mad, pass: (rho >= 0.70 || Number.isNaN(rho)) && mad <= 0.10 }; }
const commitShift = (a, b) => { const pos = x => { const L = layers10(x); let c = x.cross_layer.commit_layer.per_factor.class.commit_layer;
  if (c === alias(x)) c = "penult"; return L.indexOf(c); }; return Math.abs(pos(a) - pos(b)); };
function dfm(na, nb) { for (const [x, y] of [[na, nb], [nb, na]]) { const d = loadJ(P(y, `compare_vs_${x}`, "deformation.json"));
    if (d) return d.per_layer.penult; } return null; }
function pair(na, nb) {
  const a = load(na), b = load(nb); if (!a || !b) return null;
  return safe(() => { const d = dfm(na, nb);
    const sepA = pen(a).class_centers.sep_ratio, sepB = pen(b).class_centers.sep_ratio;
    return { a: na, b: nb, D1: spearman(dists(a), dists(b)), adj: sepRho(a, "penult", b, "penult"), l31: sepRho(a, L31(a), b, L31(b)),
      cka: d && num(d.cka_test) ? d.cka_test : null,
      relrep: d && d.relrep_argmax_chance_ood_c100 ? d.relrep_argmax_agree_ood_c100 / d.relrep_argmax_chance_ood_c100 : null,
      ...idPair(a, b), commit_shift: commitShift(a, b),
      sep_rel_spread: Math.abs(sepA - sepB) / Math.abs((sepA + sepB) / 2), sep_abs_spread: Math.abs(sepA - sepB),
      decod: Object.fromEntries(Object.keys(pen(a).linear_probes.factors).map(k => [k, decodPair(a, b, k)])) }; });
}
const vals = (ps, k) => ps.filter(p => p && num(p[k])).map(p => p[k]);

// ---- critic statuses (they decide the core items; review item 6) ----------------------------------------------------
function critRead(dir, i = 0, j = 1) {
  const c = dir ? loadJ(P(dir, "critic.json")) : null; if (!c) return null;
  const items = Object.values(c.checks || {}).flat();
  const it = name => items.find(x => x.name === name);
  const tri = name => { const x = it(name); return !x ? null : x.status === "PASS" ? true : x.status === "FAIL" ? false : null; };
  const fam = k => (c.checks || {})[k] || [];
  const famPass = k => fam(k).length > 0 && fam(k).every(x => x.status === "PASS");
  const mads = SHARP.map(fc => { const x = it(`decod/${fc}[${i},${j}]`); const m = x && /mean_abs_delta=([-0-9.eE+]+)/.exec(String(x.detail));
    return m ? parseFloat(m[1]) : null; });
  const s7max = mads.some(v => v === null) ? null : Math.max(...mads);
  return { dir, S1: tri(`id_profile[${i},${j}]`), S3: tri("penult/class_centers.sep_ratio"), S5: tri(`penult/adjacency_spearman[${i},${j}]`),
    S7: s7max === null ? null : s7max <= 0.10, S7_max_mad: s7max, S8: tri("commit/class"), S9: tri(`penult/cka_test[${i},${j}]`),
    auc_margin_typeb: tri("penult/margin_typeb.auc_margin_typeb"),
    synthetic_refusal: famPass("synthetic_refusal"), holdout_hygiene: famPass("holdout_hygiene"),
    input_norm: (it("input_norm") || {}).status || null, verdict: c.verdict };
}
const core = p => (p ? { S1: p.rho !== null && num(p.rho) ? p.rho >= 0.90 && p.shift <= 1 : null,
  S3: num(p.sep_rel_spread) ? p.sep_rel_spread <= 0.15 || p.sep_abs_spread <= 0.05 : null, S5: ge(p.adj, 0.70),
  S7: SHARP.some(k => !p.decod[k]) ? null : SHARP.every(k => p.decod[k].mad <= 0.10), S8: le(p.commit_shift, 1), S9: ge(p.cka, 0.80),
  S7_max_mad: SHARP.some(k => !p.decod[k]) ? null : Math.max(...SHARP.map(k => p.decod[k].mad)) } : null);
const killHit = p => !!p && (lt(p.adj, 0.50) || lt(p.rho, 0.80) || p.shift >= 2 || SHARP.filter(k => p.decod[k] && p.decod[k].mad > 0.10).length >= 2 ||
  p.commit_shift >= 2 || lt(p.cka, 0.60));
const mismatch = (crit, rec) => (crit && rec ? CORE_KEYS.filter(k => crit[k] !== null && rec[k] !== null && crit[k] !== rec[k]) : []);

// ---- band B20+ (5 members) and B20 (3 members, INFO) ---------------------------------------------------------------
const A20 = R20.map(load);
if (A20.some(x => !x)) { console.log("missing resnet20 band atlases:", R20.filter((n, i) => !A20[i]), "under", ROOT); process.exit(1); }
const band = vs => { const mn = Math.min(...vs), mx = Math.max(...vs), w = mx - mn; return { vals: vs, min: mn, max: mx, w, lo: mn - w, hi: mx + w }; };
const BAND = {}, BAND3 = {};
for (const k of Object.keys(FIELD)) { BAND[k] = band(A20.map(FIELD[k])); BAND3[k] = band(A20.slice(0, 3).map(FIELD[k])); }
const side = (k, v, B = BAND) => (!num(v) ? null : v > B[k].hi ? "HIGH" : v < B[k].lo ? "LOW" : "IN");

// ---- ladder, M11, the interpolation branch, Mc --------------------------------------------------------------------
const rungs = LADDER.map(({ E, atlas }) => { const t = loadJ(P(`train_resnet56_e${E}`, "train.json")); if (!t) return null;
  const acc = t.final_test_acc_10k, delta = d4(Math.abs(acc - TARGET));
  return { E, acc, seed: t.recipe.seed, atlas, delta, inWindow: delta <= WINDOW, hasAtlas: !!load(atlas) }; }).filter(Boolean);
const inWindowAll = rungs.filter(r => r.inWindow);
const inW = inWindowAll.filter(r => r.hasAtlas).sort((x, y) => x.delta - y.delta || y.E - x.E);   // review item 10
const droppedW = inWindowAll.filter(r => !r.hasAtlas).map(r => r.E);
const ladderRec = DRY ? null : loadJ(P(CHECK, "ladder.json"));
let status, M11 = null, interp = null, statusWhy = null;
if (inW.length) { status = "matched"; M11 = inW[0]; }
else if (inWindowAll.length) { status = "not_evaluable"; statusWhy = `in-window rungs without an atlas: e${droppedW.join(", e")}`; }
else { const s = [...rungs].sort((x, y) => x.acc - y.acc); status = "not_evaluable"; statusWhy = "no in-window rung and no bracketing pair";
  for (let i = 0; i + 1 < s.length; i++) if (s[i].acc < TARGET && TARGET < s[i + 1].acc) {
    status = "interpolate"; statusWhy = null; interp = { lo: s[i], hi: s[i + 1], t: (TARGET - s[i].acc) / (s[i + 1].acc - s[i].acc) }; } }
const podEstar = ladderRec ? ladderRec.e_star : null;
const MC_ALL = REPL.map(r => { const t = loadJ(P(r.train, "train.json")); if (!t) return { ...r, trained: false, evaluable: false, why: ["not trained"] };
  const acc = t.final_test_acc_10k, E = t.recipe.epochs, delta = d4(Math.abs(acc - TARGET)), hasAtlas = !!load(r.atlas), why = [];
  if (t.recipe.seed !== r.seed) why.push(`seed ${t.recipe.seed}`);
  if (delta > WINDOW) why.push(`accuracy ${acc} outside the window`);
  if (status !== "matched") why.push("no matched rung");
  else if (E !== M11.E) why.push(`trained at E ${E}, M11 is e${M11.E}`);
  if (!hasAtlas) why.push("no atlas");
  return { ...r, trained: true, acc, E, delta, hasAtlas, evaluable: !why.length, why }; });
const MC = MC_ALL.filter(m => m.evaluable);
const mStar = k => (status === "matched" ? valAt(k, M11.atlas)
  : status === "interpolate" ? (() => { const a = valAt(k, interp.lo.atlas), b = valAt(k, interp.hi.atlas);
      return a === null || b === null ? null : a + interp.t * (b - a); })() : null);

// ---- gates: R0, G0b, instrument (G0c), G0e ---------------------------------------------------------------------------
const seedAcc = DRY ? [loadJ(P("norm_check_resnet56", "norm_check.json")).chenyaofo.acc_10k, loadJ(P("train_resnet56_e40", "train.json")).final_test_acc_10k]
                    : [1, 2].map(s => { const t = loadJ(P(`train_resnet56_s${s}`, "train.json")); return t && num(t.final_test_acc_10k) ? t.final_test_acc_10k : null; });
const R0ok = DRY ? true : seedAcc.every(v => v !== null && v >= 0.935) && Math.abs(seedAcc[0] - seedAcc[1]) <= 0.010;
const atlasProblems = n => { const a = load(n); if (!a) return null; const p = [];
  for (const [l, v] of Object.entries(a.per_layer || {})) for (const [k, x] of Object.entries(v || {})) if (x && typeof x === "object" && "error" in x) p.push(`${l}/${k} error`);
  for (const [k, x] of Object.entries(a.cross_layer || {})) if (x && typeof x === "object" && "error" in x) p.push(`cross/${k} error`);
  if (a.skipped && Object.keys(a.skipped).length) p.push("skipped non-empty");
  if (a.source !== "real") p.push(`source ${a.source}`);
  return p; };
const i2 = DRY ? null : loadJ(P(CHECK, "i2.json"));
const I2_RUNS = DRY ? [] : [...SEEDS, ...REPL.map(r => r.atlas)];
const G0B = {}, G0B_BAD = new Set();
for (const n of DRY ? [...R20, HUB, ...SEEDS, ...LADDER.map(r => r.atlas)] : [...A4B_ATLASES, ...A4B_MARGINS]) {
  const pr = atlasProblems(n); if (pr === null) continue;
  if (I2_RUNS.includes(n)) { const st = i2 && i2.runs && i2.runs[n] ? i2.runs[n].status : "MISSING"; if (st !== "PASS") pr.push(`i2 ${st}`); }
  G0B[n] = pr.length ? pr : "PASS"; if (pr.length) G0B_BAD.add(n);
}
const CRIT_NORM = Object.fromEntries(A4B_CRITICS.map(c => [c, (critRead(c) || {}).input_norm || "MISSING"]));
const checkRec = DRY ? null : loadJ(P(CHECK, "check.json"));
const CL = {};
for (const n of new Set([...SEEDS, ...(M11 ? [M11.atlas] : []), ...MC.map(m => m.atlas), ...inW.map(r => r.atlas), HUB, ...R20])) {
  const a = load(n); if (a) CL[n] = safe(() => claims(a)); }
const INSTR = { check_json: DRY ? "SKIPPED (dry run)" : checkRec ? checkRec.status : "MISSING",
  check_bitwise: checkRec ? checkRec.bitwise_identical : null, r20_claims_hold: Object.fromEntries(R20.map(n => [n, claimsHold(CL[n])])) };
INSTR.ok = (DRY || (!!checkRec && checkRec.status === "PASS")) && R20.every(n => claimsHold(CL[n]));
function g0e() {
  if (DRY) return { status: "SKIPPED (dry run)" };
  const rep = { p: P_ARG, p_run: PRUN_ARG, frozen: FROZEN, runs: {} }, probs = [];
  if (!P_ARG || !PRUN_ARG) return { ...rep, status: "NOT_EVALUABLE", problems: ["--p <P> and --p-run <P_run> are required"] };
  const git = (...a) => execFileSync("git", a, { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  const gitOk = (...a) => { try { execFileSync("git", a, { cwd: REPO, stdio: "ignore" }); return true; } catch (e) { return false; } };
  let p, pr;
  try { p = git("rev-parse", "--verify", `${P_ARG}^{commit}`); pr = git("rev-parse", "--verify", `${PRUN_ARG}^{commit}`); }
  catch (e) { return { ...rep, status: "FAIL", problems: [`cannot resolve --p ${P_ARG} / --p-run ${PRUN_ARG} in ${REPO}`] }; }
  Object.assign(rep, { p_full: p, p_run_full: pr });
  if (!gitOk("merge-base", "--is-ancestor", p, pr)) probs.push("P_run is neither P nor a descendant of P");
  if (!gitOk("diff", "--quiet", p, pr, "--", ...FROZEN)) probs.push(`frozen files differ between P and P_run: ${git("diff", "--name-only", p, pr, "--", ...FROZEN).split("\n").join(", ")}`);
  const pTime = Date.parse(git("show", "-s", "--format=%cI", p));
  rep.p_time_utc = new Date(pTime).toISOString();
  const sameCommit = c => typeof c === "string" && c.length >= 7 && pr.startsWith(c);
  for (const n of [...A4B_ATLASES, ...A4B_MARGINS]) {
    const a = load(n); if (!a) continue;
    const m = a.meta || {}, prov = loadJ(P(n, "provenance.json")) || {}, bad = [];
    const created = typeof m.created === "string" ? Date.parse(m.created.replace(" ", "T") + "Z") : NaN;
    if (!sameCommit(m.git_commit)) bad.push(`meta.git_commit ${m.git_commit}`);
    if (!sameCommit(prov.stage_b_git_commit)) bad.push(`stage_b_git_commit ${prov.stage_b_git_commit}`);
    if (!(created > pTime)) bad.push(`meta.created ${m.created} not after P (${rep.p_time_utc})`);
    rep.runs[n] = bad.length ? bad : "PASS"; if (bad.length) probs.push(`${n}: ${bad.join("; ")}`);
  }
  if (!Object.keys(rep.runs).length) probs.push("no A4b atlas found");
  return { ...rep, status: probs.length ? "FAIL" : "PASS", problems: probs };
}
const G0E = g0e();

// ---- pairs ---------------------------------------------------------------------------------------------------------
const PAIRK = ["D1", "adj", "cka", "l31"];
const W20 = []; for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) W20.push(pair(R20[i], R20[j]));
const X = []; for (const a of R20) for (const b of SEEDS) X.push(pair(a, b));
const mAtlases = status === "matched" ? [M11.atlas, ...MC.map(m => m.atlas)] : [];
let XM = [];
if (status === "matched") { for (const a of R20) for (const b of mAtlases) XM.push(pair(a, b)); }
else if (status === "interpolate") XM = R20.map(a => { const pl = pair(a, interp.lo.atlas), ph = pair(a, interp.hi.atlas); if (!pl || !ph) return null;
  const o = { a, b: `interp(e${interp.lo.E}, e${interp.hi.E}; t ${f(interp.t)})` };
  for (const k of PAIRK) o[k] = num(pl[k]) && num(ph[k]) ? pl[k] + interp.t * (ph[k] - pl[k]) : null; return o; });
const W56 = pair(SEEDS[0], SEEDS[1]);
const HUBP = SEEDS.map(s => pair(HUB, s));
const TW = TWIN ? pair(HUB, TWIN) : null;
const cvLo = BAND.cv.lo, seedCV = SEEDS.map(n => valAt("cv", n));
const LEVEL = {};
for (const k of PAIRK) {
  const w20 = vals(W20, k), e20 = w20.length ? Math.min(...w20) : null, xm = median(vals(X, k)), xmm = median(vals(XM, k)), w = W56 ? W56[k] : null;
  let lab = e20 === null || xm === null ? "NOT_EVALUABLE" : xm >= e20 ? "AGREE" : num(w) && w >= e20 ? "DEPTH-GAP" : num(w) ? "DEPTH-56-UNSTABLE" : "NOT_EVALUABLE (no W56)";
  if (k === "D1" && lab === "DEPTH-56-UNSTABLE" && seedCV.every(v => num(v) && v < cvLo)) lab += " (COLLAPSE)";
  LEVEL[k] = { e20, w20_n: w20.length, w20_max: w20.length ? Math.max(...w20) : null,
    x_median: xm, x_min: vals(X, k).length ? Math.min(...vals(X, k)) : null, x_n: vals(X, k).length, all_x_below: e20 !== null && vals(X, k).every(v => v < e20),
    w56: w, label: lab, xm_median: xmm, xm_n: vals(XM, k).length,
    matched: xmm === null || e20 === null ? "NOT_EVALUABLE" : lab !== "AGREE" && xmm >= e20 ? "NOT-DEPTH at matched accuracy" : lab !== "AGREE" ? "gap persists at matched accuracy" : "AGREE" };
}

// ---- core items, KILL-56, T56 --------------------------------------------------------------------------------------
const CRIT_PAIR = critRead(CRIT.pair), CRIT_TWIN = critRead(CRIT.twin), CRIT_MARGIN = critRead(CRIT.margin);
const CORE_REC = core(W56);
const pairCritOK = !!CRIT_PAIR && CRIT_PAIR.input_norm === "PASS";
const CORE = DRY || !pairCritOK ? CORE_REC : Object.fromEntries([...CORE_KEYS, "S7_max_mad"].map(k => [k, CRIT_PAIR[k]]));
const CORE_SOURCE = DRY ? "recomputed (dry run; the critic is a stand-in, cross-check only)"
  : pairCritOK ? `critic ${CRIT.pair}` : `recomputed (${CRIT.pair} ${CRIT_PAIR ? "input_norm " + CRIT_PAIR.input_norm : "missing"})`;
const CORE_MISMATCH = mismatch(CRIT_PAIR, CORE_REC);
const KILL56 = killHit(W56) && HUBP.some(killHit);
const T56_REC = core(TW);
const T56_CORE = DRY ? null : (CRIT_TWIN && CRIT_TWIN.input_norm === "PASS" ? Object.fromEntries([...CORE_KEYS, "S7_max_mad"].map(k => [k, CRIT_TWIN[k]])) : T56_REC);
const t56Fail = DRY ? false : !TW || !T56_CORE || CORE_KEYS.some(k => T56_CORE[k] !== true);
const T56 = DRY ? "SKIPPED (dry run: no twin)" : { core: T56_CORE, core_source: CRIT_TWIN ? `critic ${CRIT.twin}` : "recomputed (critic missing)",
  recomputed_mismatch: mismatch(CRIT_TWIN, T56_REC), fail: t56Fail,
  forecast: TW ? { adj: TW.adj, adj_ge_095: ge(TW.adj, 0.95), id_rho: TW.rho, id_rho_ge_095: ge(TW.rho, 0.95),
    decod_max_mad: Math.max(...Object.values(TW.decod).filter(Boolean).map(d => d.mad)), cka: TW.cka, D1: TW.D1,
    abs_delta_penult_id: Math.abs(valAt("id", HUB) - valAt("id", TWIN)) } : null };

// promotion blocks (review items 5 and 6; AGENT_LOOP.md promotion rule)
const PROMO_BLOCK = [];
if (!DRY) {
  if (G0E.status !== "PASS") PROMO_BLOCK.push(`G0e ${G0E.status}`);
  if (!CRIT_PAIR) PROMO_BLOCK.push(`${CRIT.pair} missing`);
  else {
    if (!CRIT_PAIR.synthetic_refusal) PROMO_BLOCK.push("synthetic_refusal not PASS");
    if (!CRIT_PAIR.holdout_hygiene) PROMO_BLOCK.push("holdout_hygiene not PASS");
    if (CRIT_PAIR.input_norm !== "PASS") PROMO_BLOCK.push(`input_norm ${CRIT_PAIR.input_norm}`);
  }
  if (CORE_MISMATCH.length) PROMO_BLOCK.push(`critic vs recomputation mismatch ${CORE_MISMATCH.join(",")} (explain in SESSION.md)`);
}
// no-verdict guard shared by the d56 tags of rows 1-7 and row 11; M56-b uses R0 and the instrument gate only (margin notes)
const GUARD = [];
if (!R0ok) GUARD.push("R0 FAIL");
if (KILL56) GUARD.push("KILL-56");
if (t56Fail) GUARD.push("T56 core FAIL or twin missing");
if (!INSTR.ok) GUARD.push("instrument changed");

// ---- D-ID / D-COLL outcome table (first match wins; review items 1, 2, 10) -----------------------------------------
const ORDER = ["NOT_EVALUABLE", "NOT-REPLICATED", "NOT-DEPTH", "SEED-SENSITIVE", "FIT-SENSITIVE", "CONFIRMED-1", "CONFIRMED"];
function outcome(k) {
  const sg = SIDE[k], ms = mStar(k), s11 = side(k, ms);
  const mc = MC.map(m => ({ name: m.name, E: m.E, side: side(k, valAt(k, m.atlas)) }));
  const sSeeds = SEEDS.map(n => side(k, valAt(k, n))), win = inW.map(r => ({ E: r.E, side: side(k, valAt(k, r.atlas)) }));
  const reads = [...R20, ...SEEDS, ...mAtlases, ...inW.map(r => r.atlas), ...(interp ? [interp.lo.atlas, interp.hi.atlas] : [])];
  const why = [];
  if (s11 === null) why.push(status === "not_evaluable" ? `no matched value (${statusWhy})` : "no matched value");
  const g0b = [...new Set(reads.filter(n => G0B_BAD.has(n)))]; if (g0b.length) why.push(`G0b: ${g0b.join(", ")}`);
  const missing = SEEDS.filter((n, i) => sSeeds[i] === null); if (missing.length) why.push(`missing: ${missing.join(", ")}`);
  why.push(...GUARD.map(g => (g === "instrument changed" ? "no verdict (instrument changed)" : g)));
  let o;
  if (why.length) o = "NOT_EVALUABLE";
  else if (sSeeds.some(s => s !== sg)) o = "NOT-REPLICATED";
  else if (s11 !== sg && !mc.some(m => m.side === sg)) o = "NOT-DEPTH";
  else if ([s11, ...mc.map(m => m.side)].some(s => s !== sg)) o = "SEED-SENSITIVE";
  else if (win.some(w => w.side !== sg)) o = "FIT-SENSITIVE";
  else if (status === "interpolate" || mc.length < 2) o = "CONFIRMED-1";
  else o = "CONFIRMED";
  const edge = num(ms) ? (sg === "HIGH" ? ms - BAND[k].hi : BAND[k].lo - ms) : null;
  return { field: k, side: sg, mstar: ms, m11_side: s11, mc, seeds: sSeeds, window: win, outcome: o, not_evaluable_because: why,
    info: { mstar_beyond_edge: edge, t56_abs_delta: TW ? safe(() => Math.abs(valAt(k, HUB) - valAt(k, TWIN))) : null },
    b20_3: { mstar_side: side(k, ms, BAND3), seeds: SEEDS.map(n => side(k, valAt(k, n), BAND3)) } };
}
// Stage-2-style label for the non-claim [ACC] fields, per full-recipe seed (INFO)
const R0NE = "NOT_EVALUABLE (R0 FAIL)";                  // R0 voids every item that reads s1 or s2 (notes R0)
const accLabel = k => SEEDS.map(n => { const sv = side(k, valAt(k, n)), sm = side(k, mStar(k));
  return !R0ok ? R0NE : sv === "IN" ? "IN-BAND" : sv === null || sm === null ? "NOT_EVALUABLE" : sm === sv ? "DEPTH" : "NOT-DEPTH"; });
const row11 = o => ({ "CONFIRMED": PROMO_BLOCK.length ? `${U.y} (CONFIRMED; promotion blocked: ${PROMO_BLOCK.join("; ")})` : U.ok,
  "CONFIRMED-1": U.y, "FIT-SENSITIVE": U.y, "SEED-SENSITIVE": U.y, "NOT-REPLICATED": `${U.y} hub-specific`,
  "NOT-DEPTH": `${U.x} (rows 1-2 DEPTH text replaced)`, "NOT_EVALUABLE": `${U.open} (no verdict)` })[o];

// ---- SHAPE labels over the 10 X pairs (null-safe) -------------------------------------------------------------------
const both = k => SEEDS.map(n => (CL[n] ? CL[n][k] === true : null));
const noneOf = (k, pred) => [...SEEDS, ...(M11 ? [M11.atlas] : [])].every(n => CL[n] && pred(CL[n][k]));
const medX = k => median(vals(X, k));
const maxX = k => (vals(X, k).length ? Math.max(...vals(X, k)) : null);
const sharpX = SHARP.map(fac => { const ds = X.map(p => p && p.decod[fac]).filter(Boolean);
  return { fac, n: ds.length, mad: median(ds.map(d => d.mad)), rho: median(ds.map(d => d.rho)) }; });
function shapeOf(row) {
  const NE = "NOT_EVALUABLE (missing)";
  if (SEEDS.some(n => !load(n))) return "NOT_EVALUABLE (s1 or s2 missing: X is not the 10 pairs)";
  switch (row) {
    case "1": { const r = medX("rho"), s = maxX("shift"); if (r === null || s === null || both("C1").includes(null)) return NE;
      return both("C1").every(Boolean) && r >= 0.90 && s <= 1 ? "SCALE-ROBUST" : noneOf("C1", v => v === false) || r < 0.80 || s >= 2 ? "SCALE-FRAGILE" : "UNDECIDED"; }
    case "3": { if (sharpX.some(s => !s.n) || both("C2").includes(null)) return NE;
      return both("C2").every(Boolean) && sharpX.filter(s => s.mad <= 0.10 && s.rho >= 0.70).length >= 9 ? "SCALE-ROBUST"
        : noneOf("C2", v => v === false) || sharpX.filter(s => s.mad > 0.10).length >= 2 ? "SCALE-FRAGILE" : "UNDECIDED"; }
    case "4": { const s = maxX("commit_shift"); if (s === null || both("C3").includes(null)) return NE;
      return both("C3").every(Boolean) && s <= 1 ? "SCALE-ROBUST" : both("C3").every(v => v === false) || s >= 2 ? "SCALE-FRAGILE" : "UNDECIDED"; }
    case "5": if (SEEDS.some(n => !CL[n])) return NE;
      return SEEDS.every(n => CL[n].C4 === 6) ? "SCALE-ROBUST" : noneOf("C4", v => v < 6) ? "SCALE-FRAGILE" : "UNDECIDED";
    case "6a": if (SEEDS.some(n => !CL[n])) return NE;
      return SEEDS.every(n => CL[n].C5a === 8) ? "SCALE-ROBUST" : noneOf("C5a", v => v < 8) ? "SCALE-FRAGILE" : "UNDECIDED";
    case "7": { const d = medX("D1"), a = medX("adj"), l = medX("l31"); if (d === null || a === null) return NE;
      if (a >= 0.70 && d >= 0.80) return "SCALE-ROBUST";
      if (l !== null && d < 0.70 && l < 0.70) return "SCALE-FRAGILE";
      if (l !== null && d < 0.80 && l >= 0.80 && seedCV.every(v => num(v) && v < cvLo)) return "COLLAPSE-ARTIFACT (diagnosis, not SCALE-ROBUST)";
      return "UNDECIDED"; }
    case "8": { const r = medX("relrep"); return r === null ? NE : r >= 3 ? "SCALE-ROBUST" : "SCALE-FRAGILE"; }
    case "cka (S9, INFO in row 8)": { const c = medX("cka"); return c === null ? NE : c >= 0.80 ? "SCALE-ROBUST" : c < 0.60 ? "SCALE-FRAGILE" : "UNDECIDED"; }
  }
  return NE;
}
const SHAPE = Object.fromEntries(["1", "3", "4", "5", "6a", "7", "8", "cka (S9, INFO in row 8)"].map(r => [r, shapeOf(r)]));
if (!R0ok) { for (const r of Object.keys(SHAPE)) SHAPE[r] = R0NE; for (const k of PAIRK) { LEVEL[k].label = R0NE; LEVEL[k].matched = R0NE; } }

// ---- depth-56 row tags (STAGE2B.md "Depth-56 scope"; review items 6, 11, 21) ---------------------------------------
const holds = k => SEEDS.map(n => (CL[n] ? (k === "C4" ? CL[n].C4 === 6 : k === "C5a" ? CL[n].C5a === 8 : CL[n][k] === true) : null));
const guardTag = () => (GUARD.length ? `no d56 verdict (${GUARD.join(", ")})` : null);
const promote = () => (PROMO_BLOCK.length ? `d56 PROMOTE blocked (${PROMO_BLOCK.join("; ")})` : "d56 PROMOTE (s1, s2)");
const tag = (h, coreV) => (h.includes(null) || coreV === null || coreV === undefined ? "NOT_EVALUABLE (missing)"
  : h.every(Boolean) && coreV ? promote() : h.every(Boolean) ? "d56 YELLOW (claim holds in s1 and s2; core item FAIL)"
  : h.some(Boolean) ? "d56 ONE-SEED (1 of 2)" : "d56 NOT-TRANSFERRED");
const d1Hub = HUBP.map(p => (p ? p.D1 : null));
const C = CORE || {};
const ROWS = {
  "1": () => tag(holds("C1"), C.S1), "3": () => tag(holds("C2"), C.S7), "4": () => tag(holds("C3"), C.S8),
  "5": () => tag(holds("C4"), true), "6a": () => tag(holds("C5a"), true),
  "7": () => (!W56 || C.S5 === null || C.S5 === undefined ? "NOT_EVALUABLE (missing)"
    : C.S5 && W56.D1 >= 0.80 ? promote()
    : d1Hub.some(v => num(v) && v >= 0.80) ? "d56 ONE-PAIR (S5 FAIL or D1 s1-s2 < 0.80; a hub pair >= 0.80)"
    : "d56 NOT-TRANSFERRED (D1 < 0.80 in all three depth-56 pairs)"),
};
const PROMO = CORE ? Object.fromEntries(Object.entries(ROWS).map(([k, fn]) => [k, guardTag() || fn()])) : { all: "NOT_EVALUABLE (no s1-s2 pair)" };
const ROW8_INFO = CORE ? `S9 penult cka_test s1-s2 ${!R0ok ? R0NE : C.S9 === null ? "missing" : C.S9 ? "PASS" : "FAIL"} (INFO in row 8's evidence; no cka row)` : null;

// ---- M56 margin (STAGE2B.md section M56; only if the margin_* rebuilds exist) ---------------------------------------
const mg = n => { const a = load(MARGIN_OF(n)); return a ? safe(() => a.per_layer.penult.margin_typeb) : null; };
let M56 = null;
const MSEEDS = DRY ? [HUB, HUB] : SEEDS;                  // dry: the hub margin twice
if (mg(MSEEDS[0]) && mg(MSEEDS[1])) {
  const r20m = R20.map(mg).filter(Boolean), sm = MSEEDS.map(mg), mm = mAtlases.map(mg).filter(Boolean);
  const mmp = m => (num(m.auc_margin_confmatched) && num(m.auc_maxprob_confmatched) ? m.auc_margin_confmatched - m.auc_maxprob_confmatched : null);
  const M2 = m => num(m.margin_minus_dist_typeb) && m.margin_minus_dist_typeb > 0 && num(m.margin_minus_dist_typeb_p) && m.margin_minus_dist_typeb_p < 0.05;
  const M3 = m => num(m.median_margin_typeb) && num(m.median_margin_correct) && m.median_margin_typeb < m.median_margin_correct &&
    le(m.median_margin_ratio_typeb, 0.5) === true && le(m.median_margin_ratio_wrong, 0.5) === true;
  const m4rec = num(sm[0].auc_margin_typeb) && num(sm[1].auc_margin_typeb) ? Math.abs(sm[0].auc_margin_typeb - sm[1].auc_margin_typeb) <= 0.05 : null;
  const m4crit = CRIT_MARGIN ? CRIT_MARGIN.auc_margin_typeb : null;
  const M4 = m4crit === null ? m4rec : m4crit;
  const aucB = band(r20m.map(m => m.auc_margin_typeb)), gapB = band(r20m.map(m => m.margin_minus_dist_typeb));
  const ties = (DRY ? null : loadJ(P(CHECK, "maxprob_ties.json"))) || {};
  const tieCall = (n, m) => { const t = ties[n], g = mmp(m); if (!t || !num(t.half_tie_confmatched) || g === null) return "NOT_EVALUABLE";
    return g >= 0.010 ? (t.half_tie_confmatched < 0.5 * g ? "TIES-EXCLUDED" : "TIE-ARTIFACT-POSSIBLE") : "NO-LEAD"; };
  const m2s = sm.map(M2), m3s = sm.map(M3);
  let grammar = m4rec === null && m4crit === null ? "NOT_EVALUABLE (M4 missing)"
    : m2s.every(Boolean) && m3s.every(Boolean) && M4 ? "d56 PROMOTE" : m2s.every(v => !v) ? "d56 REJECT" : "d56 YELLOW";
  let mtag = !R0ok ? "NOT_EVALUABLE (R0 FAIL)" : !INSTR.ok ? "no verdict (instrument changed)" : grammar;
  const block = [...PROMO_BLOCK.filter(b => b.startsWith("G0e")), ...(KILL56 ? ["KILL-56"] : []), ...(t56Fail ? ["T56"] : [])];
  if (!DRY) {
    if (!CRIT_MARGIN) block.push(`${CRIT.margin} missing`);
    else { if (!CRIT_MARGIN.synthetic_refusal) block.push("margin synthetic_refusal not PASS");
      if (!CRIT_MARGIN.holdout_hygiene) block.push("margin holdout_hygiene not PASS");
      if (m4crit !== null && m4rec !== null && m4crit !== m4rec) block.push("M4 critic vs recomputation mismatch"); }
  }
  let attr = null, attrWhy = null;
  if (!R0ok) attrWhy = R0NE;
  else if (m2s.some(v => !v)) {
    if (status === "interpolate") attrWhy = "NOT_EVALUABLE (interpolation branch)";
    else if (status !== "matched" || !mm.length) attrWhy = "NOT_EVALUABLE (no matched margin atlas)";
    else { attr = mm.every(M2) ? "NOT-DEPTH" : mm.every(m => !M2(m)) ? "DEPTH" : "SPLIT"; attrWhy = `M2 at ${mAtlases.filter(n => mg(n)).join(", ")}`; }
  } else attrWhy = "no M2 failure in s1/s2";
  const b1info = DRY ? null : [1, 2].map(s => { const a = load(`margin_b1_resnet56_s${s}`); const m = a ? safe(() => a.per_layer.penult.margin_typeb) : null;
    return m ? { margin_minus_maxprob_confmatched: m.margin_minus_maxprob_confmatched ?? null, p: m.margin_minus_maxprob_confmatched_p ?? null } : null; });
  M56 = {
    a_auc: { seeds: sm.map(m => m.auc_margin_typeb), matched: mm.map(m => m.auc_margin_typeb), r20_median: median(r20m.map(m => m.auc_margin_typeb)),
      all_ge_080: !R0ok ? R0NE : [...sm, ...mm].every(m => ge(m.auc_margin_typeb, 0.80) === true),
      scale_robust: !R0ok ? R0NE : Math.abs(mean(sm.map(m => m.auc_margin_typeb)) - median(r20m.map(m => m.auc_margin_typeb))) <= 0.05,
      improves_with_scale_at_matched_acc: status === "interpolate" ? "NOT_EVALUABLE (interpolation branch)"
        : mm.length ? mm.every(m => m.auc_margin_typeb > aucB.hi) : null, auc_band: [aucB.lo, aucB.hi] },
    b_row9_d56: { M2: m2s, M3: m3s, M4, M4_recomputed: m4rec, M4_critic: m4crit, gap: sm.map(m => m.margin_minus_dist_typeb),
      gap_p: sm.map(m => m.margin_minus_dist_typeb_p), gap_band: [gapB.lo, gapB.hi], grammar, tag: mtag,
      clauses: "M2, M3, M4 (row 9's d56 tag names them; M56-c beside it)",
      promotable: mtag === "d56 PROMOTE" && !block.length, promotion_blocked_by: mtag === "d56 PROMOTE" ? block : [],
      m2_failure_attribution: attr, m2_failure_attribution_reason: attrWhy, matched_M2: mm.map(M2) },
    c_confidence: { E1: sm.map(m => (!R0ok ? R0NE : ge(m.spearman_margin_maxprob, 0.7) === true && mmp(m) !== null && Math.abs(mmp(m)) <= 0.05)),
      gap_cm: sm.map(mmp), r20_gap_cm: r20m.map(mmp), tie_call: MSEEDS.map((n, i) => (!R0ok ? R0NE : tieCall(n, sm[i]))),
      note: "no paired test of margin - maxprob in margin_typeb; the lead is descriptive", b1_e9_info: b1info },
  };
}

// ---- G0c replay (recorded): st3 atlases vs the committed Stage 1/1b/2 atlases, leaf by leaf ----------------------
const SKIPLEAF = new Set(["/built", "/dump", "/timing_s", "/exp_id", "/meta/git_commit", "/meta/created", "/meta/exp_id"]);
function leafDiff(a, b, p = "", acc = { n: 0, bad: [] }) {
  if (SKIPLEAF.has(p)) return acc;
  if (a && b && typeof a === "object" && typeof b === "object") {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!(k in a) || !(k in b)) { acc.bad.push(`${p}/${k} on one side only`); continue; }
      leafDiff(a[k], b[k], `${p}/${k}`, acc); }
    return acc; }
  acc.n++; if (a !== b) acc.bad.push(`${p}: ${a} vs ${b}`); return acc;
}
const REPLAY_PAIRS = DRY ? [["atlas_v1_resnet20_s0hub", "atlas_v1_resnet20_s0hub_st2"]]
  : [["atlas_v1_resnet20_s0hub_st2", "atlas_v1_resnet20_s0hub_st3"], ["atlas_v1_resnet20_s1_st2", "atlas_v1_resnet20_s1_st3"],
     ["atlas_v1_resnet20_s2_st2", "atlas_v1_resnet20_s2_st3"], ["atlas_v1_resnet20_s3", "atlas_v1_resnet20_s3_st3"],
     ["atlas_v1_resnet20_s4", "atlas_v1_resnet20_s4_st3"], ["atlas_v1_resnet56_s0hub", "atlas_v1_resnet56_s0hub_st3"],
     ["atlas_v1_resnet56_e40", "atlas_v1_resnet56_e40_st3"],
     // M56 rebuilds vs the committed A3 margin atlases (a key B1 added to margin_typeb would show as "on one side only")
     ...["s0hub", "s1", "s2", "s3", "s4"].map(s => [`margin_v1_resnet20_${s}`, `margin_v1_resnet20_${s}_st3`]),
     ["margin_v1_resnet56_s0hub", "margin_v1_resnet56_s0hub_st3"]];
const G0C = { replay: REPLAY_PAIRS.map(([o, n]) => { const a = load(o), b = load(n); if (!a || !b) return { old: o, new: n, status: "MISSING" };
    const d = leafDiff(a, b); return { old: o, new: n, leaves: d.n, mismatches: d.bad.length, first: d.bad.slice(0, 5), status: d.bad.length ? "DIFFERS" : "IDENTICAL" }; }),
  same_space_cka: DRY ? null : [["atlas_v1_resnet20_s0hub_st2", "atlas_v1_resnet20_s0hub_st3"], ["atlas_v1_resnet20_s1_st2", "atlas_v1_resnet20_s1_st3"],
    ["atlas_v1_resnet20_s2_st2", "atlas_v1_resnet20_s2_st3"], ["atlas_v1_resnet20_s3", "atlas_v1_resnet20_s3_st3"], ["atlas_v1_resnet20_s4", "atlas_v1_resnet20_s4_st3"],
    ["atlas_v1_resnet56_s0hub", "atlas_v1_resnet56_s0hub_st3"], ["atlas_v1_resnet56_e40", "atlas_v1_resnet56_e40_st3"]].map(([o, n]) => {
    const d = loadJ(P(n, `compare_vs_${o}`, "deformation.json")); const c = d ? safe(() => d.per_layer.penult.cka_test) : null;
    return { old: o, new: n, cka_test: c, ge_0999: ge(c, 0.999) }; }),
  check_json: INSTR.check_json, check_bitwise: INSTR.check_bitwise,
  code_diff: DRY ? null : (fs.existsSync(P(CHECK, "code_diff.txt")) ? fs.readFileSync(P(CHECK, "code_diff.txt"), "utf8").trim().split(/\r?\n/) : "MISSING"),
  versions: DRY ? null : loadJ(P(CHECK, "versions.json")) };
G0C.status = G0C.replay.every(g => g.status === "IDENTICAL") && (DRY || (G0C.check_json === "PASS" && G0C.check_bitwise === true &&
  G0C.same_space_cka.every(s => s.ge_0999 === true))) ? "IDENTICAL" : "DIFFERS or incomplete (no Stage 1/1b/2 value is quoted beside an A4b value)";

// ---- report ----------------------------------------------------------------------------------------------------------
const D = Object.keys(SIDE).map(outcome);
const dcoll = ["sep_ratio", "bridge", "nc1"].map(k => D.find(d => d.field === k).outcome);
const D_ID = D.find(d => d.field === "id").outcome;
const D_COLL = dcoll.reduce((a, b) => (ORDER.indexOf(b) < ORDER.indexOf(a) ? b : a));
const out = { dry_run: DRY, root: ROOT, check_dir: CHECK, G0c: G0C, G0b: { atlases: G0B, critic_input_norm: CRIT_NORM, i2_status: i2 ? i2.status : (DRY ? "SKIPPED" : "MISSING") },
  G0e: G0E, instrument_gate: INSTR, target: TARGET, band: BAND, band3: BAND3,
  ladder: { rungs, status, status_reason: statusWhy, M11: M11 && M11.E, pod_e_star: podEstar, in_window: inW.map(r => r.E),
    in_window_without_atlas: droppedW, interp, Mc: MC_ALL.map(m => ({ name: m.name, trained: m.trained, E: m.E ?? null, acc: m.acc ?? null, evaluable: m.evaluable, why: m.why })) },
  seed_acc: seedAcc, R0_56: R0ok, guard: GUARD, promotion_blocked_by: PROMO_BLOCK, D, D_ID, D_COLL,
  row11: { "D-ID": row11(D_ID), "D-COLL": row11(D_COLL) },
  acc_fields: Object.fromEntries(["class_excess", "nc_acc_test", "acc_test"].map(k => [k, { mstar: mStar(k), seeds: SEEDS.map(n => valAt(k, n)), labels: accLabel(k) }])),
  ref_fit: { m11: M11 ? valAt("ref_fit", M11.atlas) : null, mc: MC.map(m => valAt("ref_fit", m.atlas)), window: inW.map(r => valAt("ref_fit", r.atlas)), seeds: SEEDS.map(n => valAt("ref_fit", n)) },
  LEVEL, seed_cv: seedCV, cv_band: BAND.cv, claims: CL, core_s1_s2: CORE, core_source: CORE_SOURCE, core_recomputed: CORE_REC,
  critic_pair: CRIT_PAIR, core_mismatch: CORE_MISMATCH, KILL56, T56, SHAPE, PROMO, row8_info: ROW8_INFO, M56,
  rule8: !R0ok ? R0NE : W56 && { adj_gt_max: num(W56.adj) && W56.adj > LEVEL.adj.w20_max, cka_gt_max: num(W56.cka) && W56.cka > LEVEL.cka.w20_max, D1_gt_max: W56.D1 > LEVEL.D1.w20_max },
  pairs: { W20, X, XM, W56, HUBP, TWIN: TW } };
const L = [];
L.push(`A4b evaluation${DRY ? " (DRY RUN on Stage 2 stand-ins)" : ""}; root ${ROOT}; target ${TARGET}, window ${WINDOW}`);
L.push(`G0e: ${G0E.status}${G0E.problems && G0E.problems.length ? ` (${G0E.problems.slice(0, 3).join(" | ")}${G0E.problems.length > 3 ? " | ..." : ""})` : ""}`);
L.push(`instrument gate: ${INSTR.ok ? "OK" : "FAIL"} (check.json ${INSTR.check_json}, bitwise ${INSTR.check_bitwise}; C1-C5a in the five band atlases ${R20.every(n => INSTR.r20_claims_hold[n])})`);
L.push(`ladder: ${rungs.map(r => `e${r.E} ${r.acc} (|d| ${r.delta}${r.inWindow ? ", in" : ""}${r.inWindow && !r.hasAtlas ? ", NO ATLAS" : ""})`).join("; ")} -> ${status}${M11 ? ` M11 = e${M11.E}` : ""}${interp ? ` e${interp.lo.E}..e${interp.hi.E} t ${f(interp.t)}` : ""}${statusWhy ? ` (${statusWhy})` : ""}${DRY ? "" : `; pod E* ${podEstar}`}`);
L.push(`G0c replay: ${G0C.replay.map(g => `${g.new} ${g.status}${g.leaves ? ` (${g.leaves} leaves, ${g.mismatches} differ)` : ""}`).join("; ")} -> ${G0C.status}`);
L.push(`Mc: ${MC_ALL.length ? MC_ALL.map(m => `${m.name} ${m.trained ? `E ${m.E} acc ${m.acc}` : "-"} ${m.evaluable ? "evaluable" : `not evaluable (${m.why.join("; ")})`}`).join("; ") : "none"}; R0-56 ${R0ok} (acc ${seedAcc.join(" / ")})`);
L.push(`guard: ${GUARD.length ? GUARD.join(", ") : "none"}; promotion blocked by: ${PROMO_BLOCK.length ? PROMO_BLOCK.join("; ") : "nothing"}`);
for (const d of D) L.push(`${d.field}: band [${f(BAND[d.field].lo)}, ${f(BAND[d.field].hi)}] m* ${f(d.mstar)} ${d.m11_side} | Mc ${d.mc.map(m => `${m.name}:${m.side}`).join(" ") || "-"} | seeds ${d.seeds.join("/")} | window ${d.window.map(w => `e${w.E}:${w.side}`).join(" ") || "-"} -> ${d.outcome}${d.not_evaluable_because.length ? ` [${d.not_evaluable_because.join("; ")}]` : ""} (B20-3: m* ${d.b20_3.mstar_side}, seeds ${d.b20_3.seeds.join("/")})`);
L.push(`D-ID ${D_ID}; D-COLL ${D_COLL} (${dcoll.join(", ")}) -> ATLAS_STATUS row 11: D-ID ${out.row11["D-ID"]}, D-COLL ${out.row11["D-COLL"]}`);
for (const [k, v] of Object.entries(out.acc_fields)) L.push(`[ACC] ${k}: m* ${f(v.mstar)} seeds ${v.seeds.map(x => f(x)).join("/")} -> ${v.labels.join("/")}`);
for (const [k, v] of Object.entries(LEVEL)) L.push(`pair ${k}: W20 min ${f(v.e20)} (n ${v.w20_n}, max ${f(v.w20_max)}) X median ${f(v.x_median)} (n ${v.x_n}, min ${f(v.x_min)}, all below ${v.all_x_below}) W56 ${f(v.w56)} -> ${v.label}; matched XM median ${f(v.xm_median)} (n ${v.xm_n}): ${v.matched}`);
L.push(`penult CV band lo ${f(cvLo)}; seeds ${seedCV.map(x => f(x)).join("/")}`);
for (const [n, c] of Object.entries(CL)) if (c) L.push(`claims ${n}: C1 ${c.C1} (${c.C1_peak}, drop ${f(c.C1_drop, 3)}) C2 ${c.C2} (${c.C2_vals.map(x => f(x, 3)).join("/")}) C3 ${c.C3} (${c.C3_layer}) C4 ${c.C4}/6 (min ${f(c.C4_min, 3)}) C5a ${c.C5a}/8 (min ${f(c.C5a_min, 3)})`);
L.push(`core s1-s2 [${CORE_SOURCE}]: ${CORE ? JSON.stringify(CORE) : "-"}; recomputed ${CORE_REC ? JSON.stringify(CORE_REC) : "-"}; mismatch ${JSON.stringify(CORE_MISMATCH)}; KILL-56 ${KILL56}`);
L.push(`T56: ${typeof T56 === "string" ? T56 : JSON.stringify(T56)}`);
L.push(`SHAPE: ${JSON.stringify(SHAPE)}`);
L.push(`d56 tags: ${JSON.stringify(PROMO)}; row 8: ${ROW8_INFO}`);
if (M56) L.push(`M56: ${JSON.stringify(M56)}`);
L.push(`rule 8: ${JSON.stringify(out.rule8)}`);
console.log(L.join("\n"));
if (JSON_OUT) fs.writeFileSync(JSON_OUT, JSON.stringify(out, null, 1));
