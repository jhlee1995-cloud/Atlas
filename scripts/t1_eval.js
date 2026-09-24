#!/usr/bin/env node
// scripts/t1_eval.js -- the frozen Track 1 evaluator (docs/plans/T1_SCOREBOARD.md section 7; docs/plans/B4_INTEGRATION.md
// D8, D14, D15). Node standard library + scripts/b4_stats.js only; read-only on results/; writes one new JSON (refuses an
// existing --json / --emit-rules). It reads
//   results/b4_t1/<id>_<fit|eval><tag>/scoreboard.json   (scripts/t1_scoreboard.py)
//   results/b4_t1s/<id><tag>/streams.json               (scripts/t1_streams.py)
//   results/instrument_check_b4s{1,2}[_r<k>]/            (selftest_*.json only; replay.json in S2)
//   results/<anchor>/atlas.json, results/margin_v1_*/atlas.json   (committed Atlas numbers: the T1 anchor gates)
//   experiments/b4/models.json (roles), experiments/b4/t1_rules.json (P2), experiments/b4/model_outcomes.schema.json
// and applies the pre-registered T1 claims. T1 owns no model-level (collapse) claim: those are T2's (D1).
//   node scripts/t1_eval.js --phase discovery --p1 <P1> [--p-run <S1 HEAD>] --json results/b4/eval_discovery.json \
//        --emit-rules experiments/b4/t1_rules.json [--amend <withdrawals / stricter fractions JSON>]
//   node scripts/t1_eval.js --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> --json results/b4/eval_t1.json
//   node scripts/t1_eval.js --dry-run --repo <fixture repo>   (gates skipped, labels 'INFO (dry run) ...')
// Other options: --repo, --results, --registry, --rules, --cut "<roles or ids>", --check-dirs "<dir,dir>".
// Labels: SUPPORTED / REFUTED / INCONCLUSIVE (power: T1 review C1) / NOT_EVALUABLE; prefixed 'INFO (discovery)' in the
// discovery phase, 'INFO (dry run)' in a dry run; 'NOT_EVALUABLE (gate: ...)' when a gate fails; 'INFO (withdrawn at P2:
// ...)' for a claim withdrawn in t1_rules.json; in the confirmation phase a replay that did not PASS (FAIL, a killed or
// missing replay) appends ' [REPLAY-DRIFT]' (D15) and every per-unit decision whose deciding number lies within the replay
// drift of its threshold becomes not evaluable; a drift the record cannot bound nulls every decision.
"use strict";
const fs = require("fs");
const path = require("path");
const S = require("./b4_stats.js");
const isNum = S.isNum;

const SCHEMA = "b4_t1_eval/1";
const RULES_SCHEMA = "b4_t1_rules/1";
// ---- frozen constants (docs/plans/T1_SCOREBOARD.md section 7) ---------------------------------------------------------
const ADD = 0.01;              // HEAD-ADDITIVE: dAUC_joint >= +0.01 AND the 95% CI of dI (bits) above 0 (atlas/b4_core)
const EQ = 0.02;               // BOUNDED: the upper 95% bootstrap bound of dAUC_joint below +0.02 (T1 review C1)
const ADD_PIX = 0.005;         // beyond head AND pixels: dAUC_pix >= +0.005 AND the CI of dI_pix above 0
const MIN_EVALUABLE = 0.75;    // a scope needs >= 75% of its planned units evaluable
const ALPHA = 0.05, BAND_LEVEL = 0.99;
const MSP_SHARE_MAX = 0.25;    // X1-1 is NOT_EVALUABLE when more than 25% of units choose MSP (T1 review A5)
const DO3 = { a: 0.0224, b: -0.108, tol: 0.035 };                      // AH-1 law, results/anomaly_h1/SESSION.md:149
const ANCHOR_TOL = { margin_auc: 0.002, sparse_frac: 0.035, nc1_rel: 0.01 };  // T1 review B6, D15; sparse_frac 0.035: the
// committed number subsamples 3000 of the 10k reference radii and 3000 of the 5000 test rows (atlas/invariants/density.py
// :23-41) while T1 uses all of them: per-unit SD 0.004-0.0097, so 0.02 would gate X4-2 out by noise alone in ~16% of runs
const CONF_ONLY = ["impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform", "speckle_noise",
  "gaussian_blur", "spatter", "saturate", "exposure_global", "soiling"];   // D7
const DISC = ["gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast",
  "pixelate", "jpeg_compression"];
const HEAD_STATS = ["msp", "maxlogit", "gap", "energy", "entropy", "gini", "smargin", "pnorm2", "mspT"];
const FAMS = ["N", "B", "W", "D"];
const FAULTS = ["deadpix", "occlusion_disc", "exposure_global"];
const EARLY_DET = ["mewma_stem", "mewma_s1end", "mewma_s2end", "cusum_knnL2_stem", "cusum_knnL2_s1end",
  "cusum_knnL2_s2end", "x4"];
const HEAD_DET = ["cusum_msp", "cusum_gap", "cusum_entropy", "cusum_energy", "bbsdh", "ks_msp"];
const T1_FILES = ["scripts/t1_scoreboard.py", "scripts/t1_streams.py", "scripts/t1_eval.js", "scripts/b4_stats.js",
  "atlas/b4_core.py", "atlas/b4_collapse.py", "experiments/b4/models.json", "experiments/b4/model_outcomes.schema.json",
  "tests/test_t1_scoreboard.py", "tests/test_t1_streams.py", "tests/t1_eval_fixture.js", "docs/plans/T1_SCOREBOARD.md"];
const SCRIPT_OF = { t1: "scripts/t1_scoreboard.py", t1s: "scripts/t1_streams.py" };
const KINDS = new Set(["D", "N", "Dnew", "C", "F", "F20", "K", "Kls", "Kwd"]);
const CUTTABLE = new Set(["R2", "Kwd", "F20", "Sdisc", "Sconf", "STconf"]);
const TAG_RE = /^(?:_r\d+|_p2|_s2replay)*$/;

const get = (o, ...ks) => ks.reduce((v, k) => (v === null || v === undefined ? undefined : v[k]), o);
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const readJSON = p => JSON.parse(fs.readFileSync(p, "utf8"));

// ---- git adapter (the fixture injects a fake one; the CLI uses b4_stats.js) ----------------------------------------------
function realGit(repo) {
  return {
    head: () => S.gitHead(repo),
    isAncestor: (a, b) => S.gitIsAncestor(a, b, repo),
    sha256At: (rev, p) => S.gitSha256At(rev, p, repo),
    exists: (rev, p) => S.gitShow(rev, p, repo) !== null,
    fileSha256: p => (fs.existsSync(path.join(repo, p)) ? S.fileSha256(path.join(repo, p)) : null),
  };
}

// ---- registry and outputs -----------------------------------------------------------------------------------------------
function loadRegistry(p, cuts) {
  const reg = readJSON(p);
  const ids = new Set(reg.models.map(m => m.id));
  const bad = cuts.filter(c => !CUTTABLE.has(c) && !ids.has(c));
  if (bad.length) throw new Error(`not a registered cut: ${bad.join(" ")}`);
  const cut = new Set(cuts);
  for (const m of reg.models) {
    const drop = cut.has(m.id) || m.roles.some(r => cut.has(r) && KINDS.has(r));
    m.eroles = drop ? [] : m.roles.filter(r => !cut.has(r));
  }
  return reg;
}
const idsWith = (reg, ...roles) => reg.models.filter(m => roles.some(r => m.eroles.includes(r))).map(m => m.id);

function loadOutputs(root, file, prog) {
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root).sort().filter(d => fs.existsSync(path.join(root, d, file))).map(d => {
    const j = readJSON(path.join(root, d, file));
    return { dir: d, prog, path: path.join(root, d, file), j, tag: typeof j.tag === "string" ? j.tag : "" };
  });
}

function select(outs, phase, keyOf) {
  // one output per key: a P2 re-probe (_p2) wins over the original; the S2 replay (_s2replay) is never a unit
  const by = new Map();
  for (const o of outs) {
    if (o.j.phase !== phase || /_s2replay/.test(o.tag)) continue;
    const k = keyOf(o.j);
    if (!by.has(k)) by.set(k, []);
    by.get(k).push(o);
  }
  const chosen = new Map(), dups = [];
  for (const [k, arr] of by) {
    const p2 = arr.filter(o => /_p2/.test(o.tag));
    const pick = p2.length ? p2 : arr;
    if (pick.length > 1) dups.push({ key: k, dirs: pick.map(o => o.dir) });
    chosen.set(k, pick[pick.length - 1]);
  }
  return { chosen, dups };
}

// ---- the JSON-schema subset of experiments/b4/model_outcomes.schema.json (= t1_scoreboard.validate_schema) --------------
function validateSchema(obj, schema, root = schema, p = "$") {
  if (schema.$ref) {
    let node = root;
    for (const part of schema.$ref.replace(/^#\//, "").split("/")) node = node[part];
    return validateSchema(obj, node, root, p);
  }
  const errs = [];
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    const is = t => (t === "null" ? obj === null : t === "boolean" ? typeof obj === "boolean"
      : t === "integer" ? Number.isInteger(obj) : t === "number" ? typeof obj === "number" && Number.isFinite(obj)
        : t === "string" ? typeof obj === "string" : t === "array" ? Array.isArray(obj)
          : t === "object" ? obj !== null && typeof obj === "object" && !Array.isArray(obj) : false);
    if (!types.some(is)) return [`${p}: type ${obj === null ? "null" : typeof obj} is not ${types}`];
  }
  if ("const" in schema && JSON.stringify(obj) !== JSON.stringify(schema.const)) errs.push(`${p}: != const`);
  if (schema.enum && !schema.enum.some(e => JSON.stringify(e) === JSON.stringify(obj))) errs.push(`${p}: not in enum`);
  if (typeof obj === "number" && isNum(schema.minimum) && obj < schema.minimum) errs.push(`${p}: < minimum`);
  if (obj !== null && typeof obj === "object" && !Array.isArray(obj)) {
    for (const k of schema.required || []) if (!(k in obj)) errs.push(`${p}: missing ${k}`);
    const props = schema.properties || {}, addl = "additionalProperties" in schema ? schema.additionalProperties : true;
    for (const [k, v] of Object.entries(obj)) {
      if (k in props) errs.push(...validateSchema(v, props[k], root, `${p}.${k}`));
      else if (addl === false) errs.push(`${p}: unexpected key ${k}`);
      else if (addl && typeof addl === "object") errs.push(...validateSchema(v, addl, root, `${p}.${k}`));
    }
  }
  if (Array.isArray(obj)) {
    if (isNum(schema.minItems) && obj.length < schema.minItems) errs.push(`${p}: too few items`);
    if (isNum(schema.maxItems) && obj.length > schema.maxItems) errs.push(`${p}: too many items`);
    if (schema.items) obj.forEach((v, i) => errs.push(...validateSchema(v, schema.items, root, `${p}[${i}]`)));
  }
  return errs;
}

// ---- decisions (every threshold comparison goes through here: under replay drift a near-threshold value is null) --------
function deciders(drift) {
  const near = (v, t) => drift > 0 && Math.abs(v - t) <= drift;
  const cmp = f => (v, t) => (!isNum(v) || !isNum(t) ? null : near(v, t) ? null : f(v, t));
  return { ge: cmp((v, t) => v >= t), gt: cmp((v, t) => v > t), le: cmp((v, t) => v <= t), lt: cmp((v, t) => v < t) };
}
function allOf(list) { if (list.some(v => v === null || v === undefined)) return null; return list.every(Boolean); }
const diff = (a, b) => (isNum(a) && isNum(b) ? a - b : NaN);       // JS: null - null is 0, never a number here

// the frozen HEAD-ADDITIVE rule (atlas/b4_core.head_additive_call), three-way
function call3(d, j, add = ADD) {
  if (!j || !isNum(j.dauc) || !Array.isArray(j.dauc_ci) || !Array.isArray(j.dI_ci) || !isNum(j.dauc_ci[1])
      || !isNum(j.dI_ci[0])) return null;
  const a = allOf([d.ge(j.dauc, add), d.gt(j.dI_ci[0], 0)]);
  if (a === null) return null;
  if (a) return "ADDS";
  const b = d.lt(j.dauc_ci[1], EQ);
  return b === null ? null : b ? "BOUNDED" : "INCONCLUSIVE";
}
// beyond head AND pixel statistics (t1_scoreboard.call_pix)
function call3pix(d, j) {
  if (!j || !isNum(j.dauc_pix) || !Array.isArray(j.dauc_pix_ci) || !Array.isArray(j.dI_pix_ci) || !isNum(j.dauc_pix_ci[1])
      || !isNum(j.dI_pix_ci[0])) return null;
  const a = allOf([d.ge(j.dauc_pix, ADD_PIX), d.gt(j.dI_pix_ci[0], 0)]);
  if (a === null) return null;
  if (a) return "ADDS";
  const b = d.lt(j.dauc_pix_ci[1], EQ);
  return b === null ? null : b ? "BOUNDED" : "INCONCLUSIVE";
}

// ---- the claims ------------------------------------------------------------------------------------------------------------
// type 'bool': per unit true / false; 'bounded': the claim asserts BOUNDED (not head-additive); 'adds': it asserts ADDS.
// scopes: {scope: fraction of evaluable units needed}; the first scope is the primary one. Scopes: R (confirmation: F and
// F20 eval layouts, the fresh seeds; discovery: the trained D fit layouts), R2 (the 8 spent seeds' never-read rows, eval
// layout; confirmation only), M (confirmation: the 12 sealed C architectures, fit layout; discovery: Dnew), S (stream
// units: STconf / STdisc).
function claimDefs(d, band, phase) {
  const J = (u, t, b) => get(u, "persample", t, "joint", b);
  const U = (u, t, s) => get(u, "persample", t, "univariate", s, "auc");
  const B64 = u => get(u, "batch", "sizes", "64");
  const scen = (s, name, det, k) => get(s, "scenarios", name, det, k);
  const famCalls = (u, t, b, f) => FAMS.map(F => f(J(u, t.replace("{F}", F), b)));
  const x4band = u => {
    const x = u.x4;
    if (!x || !isNum(x.fpr_B) || !isNum(x.n_cal_fusion) || !isNum(x.n_B)) return null;
    const bd = band(x.n_cal_fusion, x.n_B);
    return allOf([d.ge(x.fpr_B, bd.lo), d.le(x.fpr_B, bd.hi)]);
  };
  const bestDelay = (s, name, dets) => { const v = dets.map(x => scen(s, name, x, "delay_cens")).filter(isNum); return v.length ? Math.min(...v) : null; };
  const C = [];
  const add = (id, rows, type, scopes, predicted, text, unit, extra = {}) => C.push({ id, rows, type, scopes, predicted, text, unit, ...extra });
  add("X1-1", ["CM-4"], "bool", { R: 0.9, R2: 0.9 }, "SUPPORTED",
    "the best head statistic, chosen on fit rows 2000-3499 by AURC and frozen, keeps AUROC >= MSP - 0.002 on the clean errors of never-read rows",
    u => { const b = get(u, "head", "best_stat"); if (!b || b === "msp") return null; return d.ge(U(u, "err_clean", b), (U(u, "err_clean", "msp") ?? NaN) - 0.002); },
    { post: "msp_share" });
  add("X1-2", ["CM-1", "CM-2"], "bounded", { R: 0.875, R2: 0.875, M: 0.75 }, "SUPPORTED",
    "the penult centre margin is NOT head-additive on clean errors: BOUNDED against the full head (upper 95% bound of dAUC_joint < +0.02)",
    u => call3(d, J(u, "err_clean", "margin_pen")), { anchor: "margin" });
  add("X1-3", ["CM-1", "CM-11"], "bounded", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "no geometric bundle (all_geom, 19 signals at five taps) is head-additive on clean errors: BOUNDED",
    u => call3(d, J(u, "err_clean", "all_geom")));
  add("X1-4", ["CM-11"], "adds", { R: 0.5, R2: 0.5, M: 0.5 }, "SUPPORTED",
    "the pre-collapse bundle (margin, relative d1, NCC-vs-head disagreement, L2-kNN, kNN purity at the pre tap) IS head-additive on corrupt-split errors (s3, 10 discovery corruptions pooled)",
    u => call3(d, J(u, "err_shift_s3", "pre")));
  add("X1-5", ["CM-9"], "adds", { R: 0.5, R2: 0.5 }, "REFUTED",
    "the early-tap bundle adds on corrupt-split errors (s3) beyond the head AND the pixel statistics",
    u => call3pix(d, J(u, "err_shift_s3", "early")));
  const fam3 = f => u => {
    const v = famCalls(u, "famid_{F}_s3", "early", j => f(d, j));
    if (v.some(x => x === null)) return null;
    const kA = v.filter(x => x === "ADDS").length, kB = v.filter(x => x === "BOUNDED").length;
    return kA >= 3 ? "ADDS" : kB >= 2 ? "BOUNDED" : "INCONCLUSIVE";
  };
  add("PC-2a", ["PC-2", "ST-1"], "adds", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "per-sample corruption-family identification (one-vs-rest among corrupt rows, s3): the early-tap bundle is head-additive for >= 3 of the 4 families",
    fam3(call3));
  add("PC-2b", ["PC-2"], "adds", { R: 0.75, R2: 0.75 }, "REFUTED",
    "... and still adds beyond the head AND the pixel statistics for >= 3 of the 4 families", fam3(call3pix));
  add("X8-1", ["CM-11"], "bounded", { R: 0.875, R2: 0.875, M: 0.75 }, "SUPPORTED",
    "Trust Score, kNN purity (k 10, 50) and LID at the penult are NOT head-additive on clean errors: BOUNDED",
    u => call3(d, J(u, "err_clean", "local_pen")));
  add("X7-1", ["CM-9"], "adds", { R: 0.5, R2: 0.5 }, "REFUTED",
    "prediction depth / cross-tap agreement is head-additive on corrupt-split errors (s3) with dAUC >= 2 x its clean dAUC",
    u => {
      const c = call3(d, J(u, "err_shift_s3", "traj")), dc = get(J(u, "err_clean", "traj"), "dauc"), ds = get(J(u, "err_shift_s3", "traj"), "dauc");
      if (c === null) return null;
      if (c !== "ADDS") return c;
      const r = d.ge(ds, 2 * Math.max(isNum(dc) ? dc : NaN, 0));
      return r === null ? null : r ? "ADDS" : "INCONCLUSIVE";
    });
  add("X2-1", ["ST-4"], "adds", { R: 0.5, R2: 0.5 }, "REFUTED",
    "the whitened head-null residual adds >= +0.03 AUROC over the head (HEAD-ADDITIVE at +0.03) for corruption presence (s3) in >= 2 of the 4 families",
    u => {
      const v = famCalls(u, "corrupt_{F}_s3", "null_pen", j => call3(d, j, 0.03));
      if (v.some(x => x === null)) return null;
      const kA = v.filter(x => x === "ADDS").length, kB = v.filter(x => x === "BOUNDED").length;
      return kA >= 2 ? "ADDS" : kB >= 3 ? "BOUNDED" : "INCONCLUSIVE";
    });
  add("X4-1R", ["DO-6", "DO-3"], "bool", { R: 0.875, R2: 0.875 }, "SUPPORTED",
    "the fused X4 flag calibrated on the fit rows keeps its clean false-alarm rate on never-read rows inside the exact 99% beta-binomial band at alpha 0.05 (n_cal 2500, n_test 1500: [0.0327, 0.0693])",
    x4band);
  add("X4-1M", ["DO-6"], "bool", { M: 0.8 }, "SUPPORTED",
    "the same code with zero retuning on the 12 never-measured architectures: clean false-alarm rate inside the band (n_cal 750, n_test 1500: [0.0273, 0.0767])",
    x4band);
  add("X4-2", ["DO-3"], "bool", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "REPLICATION (AH-1a on never-read rows): |train-referenced 10-NN clean sparse fraction - (0.0224 - 0.108 log10 nc1_train)| <= 0.035",
    u => { const x = u.do3; if (!x || !isNum(x.sparse_frac_test) || !isNum(x.nc1_train) || x.nc1_train <= 0) return null; return d.le(Math.abs(x.sparse_frac_test - (DO3.a + DO3.b * Math.log10(x.nc1_train))), DO3.tol); },
    { anchor: "do3" });
  add("X4-3", ["DO-1"], "bool", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "at matched 5% clean false alarms, the X4 flag's TPR beats the best head statistic's TPR by >= 0.05 on >= 60% of the noise / blur / pixelate splits at s3 and s5 (10 splits)",
    u => {
      const xt = get(u, "x4", "tpr") || {}, cal = u.calibration || {};
      let win = 0, n = 0;
      for (const c of ["gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "pixelate"]) for (const s of [3, 5]) {
        const sp = `corrupt__${c}__s${s}`, hs = HEAD_STATS.map(k => get(cal, k, "tpr", sp)).filter(isNum);
        if (!isNum(xt[sp]) || !hs.length) continue;
        const w = d.ge(diff(xt[sp], Math.max(...hs)), 0.05);
        if (w === null) return null;
        n++; if (w) win++;
      }
      return n >= 8 ? win / n >= 0.6 : null;
    });
  add("X4-4", ["DO-1"], "adds", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "X4 is head-additive for corruption presence of the noise AND the blur families (s3)",
    u => { const a = call3(d, J(u, "corrupt_N_s3", "x4")), b = call3(d, J(u, "corrupt_B_s3", "x4")); if (a === null || b === null) return null; return a === "ADDS" && b === "ADDS" ? "ADDS" : a === "BOUNDED" || b === "BOUNDED" ? "BOUNDED" : "INCONCLUSIVE"; });
  add("X4-5", ["DO-1"], "bool", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "trained on the discovery families only, head + X4 beats the head by >= 0.02 AUROC on the 5 unseen holdout corruption families (s3)",
    u => d.ge(get(J(u, "corrupt_holdout_s3_transfer", "x4"), "dauc"), 0.02));
  add("X4-5x", ["DO-1"], "bool", { R: 0.75, M: 0.75 }, "SUPPORTED",
    "... and on the 4 CIFAR-10-C extras never used anywhere (s3; NOT_EVALUABLE when the volume lacks them)",
    u => d.ge(get(J(u, "corrupt_extra_s3_transfer", "x4"), "dauc"), 0.02));
  add("X4-6", ["DO-4"], "adds", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "X4 is head-additive for far-OOD detection (SVHN against clean rows)", u => call3(d, J(u, "ood_svhn", "x4")));
  add("X6-1", ["LH-1", "LH-2"], "bool", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "ATC (max confidence) estimates batch-64 accuracy at least as well as the geometric harm grade H (the same isotonic footing, leave-one-split-out MAE)",
    u => { const e = get(B64(u), "estimators"); return d.le(diff(get(e, "ATC_MC", "mae_pp_loso"), get(e, "H", "mae_pp_loso")), 0); });
  add("X6-2", ["LH-1"], "bool", { R: 0.5, R2: 0.5 }, "REFUTED",
    "H adds to ATC within splits: mean within-split partial Spearman(H, loss | ATC) >= 0.3 (batch 64)",
    u => d.ge(get(B64(u), "partial_H_given_ATC_mean"), 0.3));
  add("X6-3", ["ST-3"], "bool", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "label skew: the BBSE-explained residual R_pre and H stay quiet (single-class FPR max <= 0.15, mean <= 0.10; H mean <= 0.10) while the plain penult T2 (single-class mean >= 0.5) and BBSDh (Dirichlet 0.1, >= 0.5) fire",
    u => {
      const sc = get(B64(u), "skew", "single_class"), dir = get(B64(u), "skew", "dirichlet_0.1");
      if (!sc || !dir) return null;
      return allOf([d.le(get(sc, "R_pre", "fpr_max"), 0.15), d.le(get(sc, "R_pre", "fpr_mean"), 0.10),
        d.le(get(sc, "H", "fpr_mean"), 0.10), d.ge(get(sc, "T2_penult", "fpr_mean"), 0.5), d.ge(get(dir, "bbsdh_chi2"), 0.5)]);
    });
  add("X6-4", ["ST-1", "ST-5"], "bool", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "covariate / prior / novelty typing of batch-64 windows: geometry-rule macro accuracy >= 0.85 and >= the head-only rule + 0.05",
    u => { const t = get(B64(u), "typing", "macro"); if (!t) return null; return allOf([d.ge(t.geometry_acc, 0.85), d.ge(diff(t.geometry_acc, t.head_acc), 0.05)]); });
  add("X6-5", ["LH-2"], "bool", { R: 0.75, R2: 0.75 }, "SUPPORTED",
    "mild covariate shift at batch 64: R_pre flags >= 80% of the motion / pixelate / snow s1 batches at its clean 5% threshold",
    u => allOf(["motion_blur", "pixelate", "snow"].map(c => d.ge(get(B64(u), "by_split", `corrupt__${c}__s1`, "flag_rate", "R_pre"), 0.8))));
  add("X3-8", ["PC-7"], "bool", { R: 0.75, R2: 0.75, M: 0.75 }, "SUPPORTED",
    "injected sensor faults (dead pixels, occlusion disc, holdout exposure): the best early-tap per-frame AUROC beats the best head statistic by >= 0.10 for every fault kind",
    u => {
      const need = phase === "discovery" ? 2 : 3;
      let ok = 0, n = 0;
      for (const f of FAULTS) {
        const t = `fault_${f}`;
        if (!get(u, "persample", t)) continue;
        const e = Math.max(...["knnL2_stem", "wnorm_stem", "x4tap_stem", "knnL2_s1end", "wnorm_s1end"].map(s => U(u, t, s)).filter(isNum));
        const h = Math.max(...HEAD_STATS.map(s => U(u, t, s)).filter(isNum));
        if (!Number.isFinite(e) || !Number.isFinite(h)) continue;
        const w = d.ge(diff(e, h), 0.10);
        if (w === null) return null;
        n++; if (w) ok++;
      }
      return n >= need ? ok === n : null;
    });
  // X3 streams (unit = one streams.json)
  add("X3-1", ["ST-9"], "bool", { S: 2 / 3 }, "SUPPORTED",
    "change-information rate: I(end of stage 1) > I(penult) with disjoint 95% CIs on >= 8 of the 10 discovery corruptions at s1",
    s => {
      let k = 0, n = 0;
      for (const c of DISC) {
        const a = get(s, "rate", `corrupt__${c}__s1`, "s1end"), b = get(s, "rate", `corrupt__${c}__s1`, "penult");
        if (!a || !b || !Array.isArray(a.ci) || !Array.isArray(b.ci)) continue;
        const w = d.gt(diff(a.ci[0], b.ci[1]), 0);
        if (w === null) return null;
        n++; if (w) k++;
      }
      return n >= 8 ? k >= 8 : null;
    });
  add("X3-2", ["ST-9"], "bool", { S: 2 / 3 }, "SUPPORTED",
    "step onset at s1 (motion, pixelate, snow): the best early-tap censored delay <= 0.25 x the best head-only censored delay, at ARL0 2000",
    s => allOf(["motion_blur", "pixelate", "snow"].map(c => { const e = bestDelay(s, `step|corrupt__${c}__s1`, EARLY_DET), h = bestDelay(s, `step|corrupt__${c}__s1`, HEAD_DET); return isNum(e) && isNum(h) && h > 0 ? d.le(e / h, 0.25) : null; })));
  add("X3-3", ["ST-9"], "bool", { S: 2 / 3 }, "SUPPORTED",
    "ramps s1 -> s3 -> s5 (motion, defocus): the best early-tap detector alarms >= 100 frames (median) before the rolling 64-frame loss reaches 5 pt",
    s => allOf(["motion_blur", "defocus_blur"].map(c => { const l = EARLY_DET.map(x => scen(s, `ramp|${c}`, x, "lead_median")).filter(isNum); return l.length ? d.ge(Math.max(...l), 100) : null; })));
  add("X3-4", ["ST-9"], "bool", { S: 5 / 6 }, "SUPPORTED",
    "ARL transport: the realised in-control ARL0 on held-out clean rows lies in [1000, 4000] for >= 90% of the calibrated detectors",
    s => { const a = Object.values(s.arl || {}).map(r => r.arl0_heldout).filter(isNum); if (!a.length) return null; const v = a.map(x => allOf([d.ge(x, 1000), d.le(x, 4000)])); if (v.some(x => x === null)) return null; return v.filter(Boolean).length / v.length >= 0.9; });
  add("X3-5", ["PC-7"], "bool", { S: 5 / 6 }, "SUPPORTED",
    "a stuck (frozen) frame freezes the EWMA: the penult and stem MEWMA alarm within 500 frames in >= 90% of streams without pixel access (the pixel repeat check >= 0.99 is a sanity clause; T1 review A3)",
    s => allOf([d.ge(scen(s, "stuck|frame", "mewma_penult", "p_det_within"), 0.9), d.ge(scen(s, "stuck|frame", "mewma_stem", "p_det_within"), 0.9),
      d.ge(scen(s, "stuck|frame", "stuck", "p_det_within"), 0.99)]));
  add("X3-6", ["LH-4"], "bool", { S: 2 / 3 }, "SUPPORTED",
    "benign brightness ramp: the harm-gated early-tap alarms fire in <= 20% of streams while the ungated early-tap detector fires in >= 80%",
    s => {
      const g = ["cusum_knnL2_stem", "mewma_stem", "x4"].map(x => scen(s, "ramp|brightness", x, "p_det_within_gated")).filter(isNum);
      const u = ["cusum_knnL2_stem", "mewma_stem"].map(x => scen(s, "ramp|brightness", x, "p_det_within")).filter(isNum);
      return g.length && u.length ? allOf([d.le(Math.max(...g), 0.2), d.ge(Math.max(...u), 0.8)]) : null;
    });
  add("X3-7", ["ST-5"], "bool", { S: 2 / 3 }, "SUPPORTED",
    "label-skew burst: BBSDh and the penult MEWMA alarm (>= 0.9) while the per-frame conformal flag alarms no more than in its own in-control null run with the same timing (skew minus null p_det_within <= 0.10; a calibrated ARL0-2000 flag alarms within 500 frames in about 20% of clean streams)",
    s => allOf([d.ge(scen(s, "skew|single_class", "bbsdh", "p_det_within"), 0.9), d.ge(scen(s, "skew|single_class", "mewma_penult", "p_det_within"), 0.9),
      d.le(diff(scen(s, "skew|single_class", "x4", "p_det_within"), scen(s, "null|clean", "x4", "p_det_within")), 0.10)]));
  return C;
}

function scopeResult(claim, units, planned, frac) {
  const ev = units.map(u => { let v = null; try { v = claim.unit(u); } catch (e) { v = null; } return { unit: u.unit, v: v === undefined ? null : v }; });
  const ok = ev.filter(e => e.v !== null);
  const r = { planned, n: ok.length, frac_needed: frac, per_unit: Object.fromEntries(ev.map(e => [e.unit, e.v])) };
  if (planned === 0) return { ...r, label: "NOT_PLANNED" };
  if (ok.length < Math.ceil(MIN_EVALUABLE * planned - 1e-9) || ok.length === 0) return { ...r, label: "NOT_EVALUABLE" };
  const n = ok.length, eps = 1e-12;
  if (claim.type === "bool") {
    r.k = ok.filter(e => e.v === true).length;
    r.label = r.k / n >= frac - eps ? "SUPPORTED" : "REFUTED";
    return r;
  }
  r.k_adds = ok.filter(e => e.v === "ADDS").length;
  r.k_bounded = ok.filter(e => e.v === "BOUNDED").length;
  r.k_inconclusive = n - r.k_adds - r.k_bounded;
  const [kYes, kNo] = claim.type === "bounded" ? [r.k_bounded, r.k_adds] : [r.k_adds, r.k_bounded];
  r.label = kYes / n >= frac - eps ? "SUPPORTED" : kNo / n > 1 - frac + eps ? "REFUTED" : "INCONCLUSIVE";
  return r;
}

// ---- gates ---------------------------------------------------------------------------------------------------------------
function checkDirs(results, phase, override) {
  if (override && override.length) return override.map(d => (path.isAbsolute(d) ? d : path.join(results, d))).filter(fs.existsSync);
  if (!fs.existsSync(results)) return [];
  const re = phase === "discovery" ? /^instrument_check_b4s1(_r\d+)?$/ : /^instrument_check_b4s2(_r\d+)?$/;
  return fs.readdirSync(results).sort().filter(d => re.test(d)).map(d => path.join(results, d));
}

function selftestGate(dirs) {
  const recs = [];
  for (const d of dirs) for (const f of fs.readdirSync(d).filter(f => /^selftest_.*\.json$/.test(f)).sort()) {   // T1 review A1
    let st = null; try { st = readJSON(path.join(d, f)).status; } catch (e) { st = `unreadable: ${e.message}`; }
    recs.push({ dir: path.basename(d), file: f, status: st });
  }
  const need = ["selftest_t1_scoreboard.json", "selftest_t1_streams.json"];
  const ok = dirs.some(d => need.every(n => recs.some(r => r.dir === path.basename(d) && r.file === n && r.status === "PASS")));
  return { status: ok ? "PASS" : "FAIL", need, records: recs };
}

function replayGate(dirs) {
  // drift = the largest |S1 - S2| of the t1 anchor diffs; UNBOUNDED (Infinity: every threshold decision becomes null) when
  // the record cannot bound it: a t1 unit NOT_EVALUABLE (e.g. a killed replay), a FAIL unit with more diffs than first20
  // lists, a diff line without a numeric pair (missing key, length, string), no t1 unit at all, or no replay.json.
  // JSON writes Infinity as null, so drift_unbounded records it.
  for (const d of [...dirs].reverse()) {
    const p = path.join(d, "replay.json");
    if (!fs.existsSync(p)) continue;
    const rep = readJSON(p), units = Object.entries(rep.units || {}).filter(([k]) => k.startsWith("t1:"));
    let drift = 0;
    const unbounded = [];
    if (!units.length) unbounded.push("no t1 unit in replay.json");
    for (const [k, v] of units) {
      if (v.status === "PASS") continue;
      if (v.status !== "FAIL") { unbounded.push(`${k}: ${v.status}${v.why ? ` (${v.why})` : ""}`); continue; }
      const lines = Array.isArray(v.first20) ? v.first20 : [];
      if (!isNum(v.n_diff) || v.n_diff > lines.length) unbounded.push(`${k}: n_diff ${v.n_diff} > ${lines.length} listed`);
      for (const line of lines) {
        const m = /: (-?[0-9.eE+-]+) vs (-?[0-9.eE+-]+)$/.exec(line);
        if (m && isNum(+m[1]) && isNum(+m[2])) drift = Math.max(drift, Math.abs(+m[1] - +m[2]));
        else unbounded.push(`${k}: ${line}`);
      }
    }
    const st = !units.length ? "NOT_EVALUABLE" : units.every(([, v]) => v.status === "PASS") ? "PASS" : "FAIL";
    return { status: st, file: p, drift: st === "PASS" ? 0 : unbounded.length ? Infinity : drift,
      drift_unbounded: st !== "PASS" && unbounded.length > 0, unbounded_why: unbounded.slice(0, 20),
      units: Object.fromEntries(units.map(([k, v]) => [k, v.status])) };
  }
  return { status: "NOT_EVALUABLE", why: "replay.json missing", drift: Infinity, drift_unbounded: true,
    unbounded_why: ["replay.json missing"] };
}

function readGuard(discOuts, confSel, phase) {
  const bad = [];
  for (const o of discOuts) {
    if (o.j.phase !== "discovery") continue;
    for (const d of o.j.dumps || []) {
      if (d.sealed) bad.push(`${o.dir}: opened sealed ${d.path}`);
      for (const s of d.splits_read || []) if (CONF_ONLY.some(c => s.includes(c))) bad.push(`${o.dir}: read ${s}`);
    }
  }
  const dups = phase === "confirmation" ? confSel.flatMap(s => s.dups) : [];
  return { status: bad.length || dups.length ? "FAIL" : "PASS", discovery_read_confirmation: bad, scored_twice: dups };
}

function anchorGate(reg, res, discBoards) {
  // committed Atlas numbers reproduced by the T1 re-implementation on the b4 re-extractions (T1 review B6; D15)
  const out = { tol: ANCHOR_TOL, units: {}, margin: "NOT_EVALUABLE", do3: "NOT_EVALUABLE" };
  const mStat = [], dStat = [];
  for (const m of reg.models) {
    if (!m.anchor || !m.roles.includes("D")) continue;
    const o = discBoards.get(`${m.id}|fit`);
    if (!o) continue;
    const u = o.j, rec = {};
    const ap = path.join(res, m.anchor.replace(/^results\//, ""), "atlas.json");
    const mp = path.join(res, m.anchor.replace(/^results\//, "").replace(/^atlas_v1_/, "margin_v1_"), "atlas.json");
    if (fs.existsSync(ap)) {
      const pl = get(readJSON(ap), "per_layer", "penult");
      const sf = get(pl, "knn_density", "splits", "test", "sparse_frac"), nc = get(pl, "neural_collapse", "nc1");
      const my = get(u, "do3", "sparse_frac_test"), mn = get(u, "do3", "nc1_train");
      if (isNum(sf) && isNum(my) && isNum(nc) && isNum(mn)) {
        rec.do3 = { committed_sparse: sf, t1_sparse: my, committed_nc1: nc, t1_nc1: mn };
        rec.do3.pass = Math.abs(sf - my) <= ANCHOR_TOL.sparse_frac && Math.abs(nc - mn) / nc <= ANCHOR_TOL.nc1_rel;
        dStat.push(rec.do3.pass);
      }
    }
    if (fs.existsSync(mp)) {
      const ca = get(readJSON(mp), "per_layer", "penult", "margin_typeb", "auc_margin_wrong");
      const my = get(u, "persample", "err_clean", "univariate", "margin_penult", "auc");
      if (isNum(ca) && isNum(my)) {
        rec.margin = { committed_auc: ca, t1_auc: my, pass: Math.abs(ca - my) <= ANCHOR_TOL.margin_auc };
        mStat.push(rec.margin.pass);
      }
    }
    if (Object.keys(rec).length) out.units[m.id] = rec;
  }
  if (mStat.length) out.margin = mStat.every(Boolean) ? "PASS" : "FAIL";
  if (dStat.length) out.do3 = dStat.every(Boolean) ? "PASS" : "FAIL";
  out.status = [out.margin, out.do3].includes("FAIL") ? "FAIL" : [out.margin, out.do3].every(s => s === "PASS") ? "PASS" : "PARTIAL";
  return out;
}

function provenanceGate(o) {
  // o: {phase, git, p1, p2, pRun, outs: [{prog, j, dir}], repo}
  const { git } = o, checks = {}, bad = [];
  const conf = o.phase === "confirmation";
  const rev = conf ? o.p2 : o.p1;
  if (!o.p1 || (conf && !o.p2)) return { status: "FAIL", why: "--p1 (and --p2 in confirmation) required" };
  if (conf && !o.pRun.length) return { status: "FAIL", why: "--p-run <S1 HEAD>,<S2 HEAD> required in confirmation" };
  checks.head = git.head();
  checks.p1_ancestor_of_head = git.isAncestor(o.p1, "HEAD");
  if (o.phase === "confirmation") {
    checks.p1_ancestor_of_p2 = git.isAncestor(o.p1, o.p2);
    checks.p2_ancestor_of_head = git.isAncestor(o.p2, "HEAD");
    checks.freeze_P2 = git.exists(o.p2, "experiments/b4/freeze_P2.json");
    checks.rules_at_p2 = git.exists(o.p2, "experiments/b4/t1_rules.json");
  }
  checks.files = {};
  if (conf) {                              // frozen from P2 on (D8); between P1 and P2 evaluator fixes are allowed (D8 e)
    for (const f of [...T1_FILES, "experiments/b4/t1_rules.json"]) {
      const at = git.sha256At(rev, f), wt = git.fileSha256(f);
      checks.files[f] = { at_rev: at, worktree: wt, unchanged: at !== null && at === wt };
      if (!checks.files[f].unchanged) bad.push(`${f} differs from ${rev}`);
    }
  }
  const wantAt = r => ({ t1: git.sha256At(r, SCRIPT_OF.t1), t1s: git.sha256At(r, SCRIPT_OF.t1s), core: git.sha256At(r, "atlas/b4_core.py") });
  const wantRev = wantAt(rev), wantP2 = !conf && o.p2 ? wantAt(o.p2) : null;
  const nb = conf ? 1000 : 200;
  for (const x of o.outs) {
    const c = x.j.code || {}, rc = c.repo_commit || "";
    const reprobe = /_p2/.test(x.tag || "");
    if (reprobe && !wantP2) { bad.push(`${x.dir}: a P2 re-probe needs --p2`); continue; }
    const want = reprobe ? wantP2 : wantRev, at = reprobe ? o.p2 : rev;
    if (c.sha256 !== want[x.prog]) bad.push(`${x.dir}: code sha256 is not ${SCRIPT_OF[x.prog]} at ${at}`);
    if (c.core_sha256 !== want.core) bad.push(`${x.dir}: atlas/b4_core.py sha256 is not the one at ${at}`);
    if (o.pRun.length && !o.pRun.some(p => p && rc && (rc.startsWith(p) || p.startsWith(rc)))) bad.push(`${x.dir}: repo_commit ${rc} not in --p-run`);
    if (x.j.nboot !== nb) bad.push(`${x.dir}: nboot ${x.j.nboot} != ${nb}`);
    const sealed = (x.j.dumps || []).some(dd => dd.sealed);
    const un = get(x.j, "env", "unseal") || "";
    if (o.phase === "confirmation" && sealed && !(un && (un.startsWith(o.p2) || o.p2.startsWith(un)))) bad.push(`${x.dir}: sealed dump opened without ATLAS_B4_UNSEAL = P2`);
  }
  for (const k of ["p1_ancestor_of_head", "p1_ancestor_of_p2", "p2_ancestor_of_head", "freeze_P2", "rules_at_p2"]) {
    if (k in checks && checks[k] !== true) bad.push(`${k} false`);
  }
  return { status: bad.length ? "FAIL" : "PASS", rev, checks, failures: bad.slice(0, 50), n_failures: bad.length };
}

function outcomesGate(repo, boards) {
  const sp = path.join(repo, "experiments/b4/model_outcomes.schema.json");
  if (!fs.existsSync(sp)) return { status: "NOT_EVALUABLE", why: "schema missing" };
  const schema = readJSON(sp), units = {};
  for (const u of boards) if (u.layout === "fit") {
    const e = u.model_outcomes ? validateSchema(u.model_outcomes, schema) : ["model_outcomes missing"];
    units[u.unit] = e.length ? { valid: false, errors: e.slice(0, 5) } : { valid: true };
  }
  const v = Object.values(units);
  return { status: v.length ? (v.every(x => x.valid) ? "PASS" : "FAIL") : "NOT_EVALUABLE", units,
    note: "INFO for T1: T2's collapse_laws.js reads O1 / O5 from these records (D5)" };
}

// ---- rules (P2) -----------------------------------------------------------------------------------------------------------
function applyAmend(claims, amend) {
  const out = { withdraw: {}, frac: {} }, errs = [];
  const byId = Object.fromEntries(claims.map(c => [c.id, c]));
  for (const [id, why] of Object.entries(amend.withdraw || {})) { if (!byId[id]) errs.push(`unknown claim ${id}`); else out.withdraw[id] = String(why); }
  for (const [id, fr] of Object.entries(amend.frac || {})) {
    if (!byId[id]) { errs.push(`unknown claim ${id}`); continue; }
    for (const [sc, v] of Object.entries(fr)) {
      if (!(sc in byId[id].scopes)) errs.push(`${id}: no scope ${sc}`);
      else if (!isNum(v) || v < byId[id].scopes[sc] || v > 1) errs.push(`${id}.${sc}: ${v} is not stricter than ${byId[id].scopes[sc]} (D8 c)`);
      else (out.frac[id] = out.frac[id] || {})[sc] = v;
    }
  }
  return { out, errs };
}

function rulesFor(claims, rulesDoc) {
  const per = {}, errs = [];
  if (!rulesDoc) return { per, errs: ["rules missing"] };
  if (rulesDoc.schema !== RULES_SCHEMA) errs.push(`rules schema ${rulesDoc.schema}`);
  for (const c of claims) {
    const r = (rulesDoc.claims || {})[c.id];
    if (!r) { per[c.id] = { status: "ACTIVE", frac: { ...c.scopes } }; continue; }
    const fr = { ...c.scopes }, bad = [];
    for (const [sc, v] of Object.entries(r.frac || {})) {
      if (!(sc in c.scopes) || !isNum(v) || v < c.scopes[sc] || v > 1) bad.push(`${sc}: ${v}`);
      else fr[sc] = v;
    }
    per[c.id] = { status: r.status === "INFO" ? "INFO" : "ACTIVE", reason: r.reason || null, frac: fr, invalid: bad };
  }
  return { per, errs };
}

// ---- evaluation ---------------------------------------------------------------------------------------------------------------
function evaluate(opts, gitIn) {
  const repo = path.resolve(opts.repo), res = path.resolve(opts.results || path.join(repo, "results"));
  const phase = opts.phase || "confirmation", dry = !!opts.dry;
  if (!["discovery", "confirmation"].includes(phase)) throw new Error(`unknown phase ${phase}`);
  const git = gitIn || realGit(repo);
  let cuts = opts.cuts;
  const fz = path.join(repo, "experiments/b4/freeze_P2.json");
  if (!cuts && phase === "confirmation" && fs.existsSync(fz)) {
    const c = readJSON(fz).cuts;
    cuts = Array.isArray(c) ? c : typeof c === "string" ? c.split(/\s+/).filter(Boolean) : [];
  }
  const reg = loadRegistry(opts.registry || path.join(repo, "experiments/b4/models.json"), cuts || []);
  const allT1 = loadOutputs(path.join(res, "b4_t1"), "scoreboard.json", "t1");
  const allT1s = loadOutputs(path.join(res, "b4_t1s"), "streams.json", "t1s");
  const bsel = select(allT1, phase, j => `${j.unit}|${j.layout}`);
  const ssel = select(allT1s, phase, j => j.unit);
  const dsel = select(allT1, "discovery", j => `${j.unit}|${j.layout}`);           // discovery boards: the anchor gates
  const pick = (ids, layout) => ids.map(i => bsel.chosen.get(`${i}|${layout}`)).filter(Boolean).map(o => o.j);
  const pickS = ids => ids.map(i => ssel.chosen.get(i)).filter(Boolean).map(o => o.j);
  let sets, planned;
  if (phase === "discovery") {
    const D = idsWith(reg, "D"), M = idsWith(reg, "Dnew"), St = idsWith(reg, "STdisc");
    sets = { R: pick(D, "fit"), R2: [], M: pick(M, "fit"), S: pickS(St) };
    planned = { R: D.length, R2: 0, M: M.length, S: St.length };
  } else {
    const R = idsWith(reg, "F", "F20"), R2 = idsWith(reg, "R2"), M = idsWith(reg, "C"), St = idsWith(reg, "STconf");
    sets = { R: pick(R, "eval"), R2: pick(R2, "eval"), M: pick(M, "fit"), S: pickS(St) };
    planned = { R: R.length, R2: R2.length, M: M.length, S: St.length };
  }
  const dirs = checkDirs(res, phase, opts.checkDirs);
  const gates = {};
  gates.selftests = dry ? { status: "SKIPPED (dry run)" } : selftestGate(dirs);
  const used = [...bsel.chosen.values(), ...ssel.chosen.values()];
  gates.provenance = dry ? { status: "SKIPPED (dry run)" }
    : provenanceGate({ phase, git, p1: opts.p1, p2: opts.p2, pRun: opts.pRun || [], outs: used, repo });
  gates.read_guard = readGuard([...allT1, ...allT1s], [bsel, ssel], phase);
  gates.replay = phase === "confirmation" && !dry ? replayGate(checkDirs(res, "confirmation", opts.checkDirs)) : { status: "SKIPPED", drift: 0 };
  gates.anchors = anchorGate(reg, res, dsel.chosen);
  gates.model_outcomes = outcomesGate(repo, [...bsel.chosen.values()].map(o => o.j));
  let rules = null, rulesErr = null;
  if (phase === "confirmation" && !dry) {
    const rp = opts.rules || path.join(repo, "experiments/b4/t1_rules.json");
    rules = fs.existsSync(rp) ? readJSON(rp) : null;
    const atP2 = opts.p2 ? git.sha256At(opts.p2, "experiments/b4/t1_rules.json") : null;
    if (rules && atP2 !== S.fileSha256(rp)) rulesErr = "the rules file is not experiments/b4/t1_rules.json as committed at P2";
  }
  // D15: any replay result but PASS (FAIL, a killed replay, no replay.json) tags the labels; decisions within the drift
  // of their threshold (every decision, when the drift is unbounded) are not evaluable
  const replayBad = phase === "confirmation" && !dry && gates.replay.status !== "PASS";
  const drift = replayBad ? gates.replay.drift : 0;
  const d = deciders(drift);
  const bandCache = {};
  const band = (n, m) => (bandCache[`${n}|${m}`] = bandCache[`${n}|${m}`] || S.betaBinomBand(n, m, ALPHA, BAND_LEVEL));
  const claims = claimDefs(d, band, phase);
  const rr = phase === "confirmation" && !dry ? rulesFor(claims, rules) : { per: {}, errs: [] };
  if (rulesErr) rr.errs.push(rulesErr);
  gates.rules = phase === "confirmation" && !dry ? { status: rr.errs.length ? "FAIL" : "PASS", errors: rr.errs } : { status: "SKIPPED" };
  const hard = ["selftests", "provenance", "read_guard", "rules"].filter(g => gates[g].status === "FAIL");
  const out = [];
  for (const c of claims) {
    const rule = (gates.rules.status !== "FAIL" && rr.per[c.id]) || { status: "ACTIVE", frac: { ...c.scopes } };   // a failed rules gate applies nothing from that file
    const scopes = {};
    for (const [sc, fr0] of Object.entries(c.scopes)) {
      const units = sets[sc] || [];
      scopes[sc] = scopeResult(c, units, planned[sc] || 0, rule.frac[sc] ?? fr0);
    }
    const primary = Object.keys(c.scopes)[0];
    let label = scopes[primary].label, note = null;
    if (c.post === "msp_share") {
      const bs = (sets[primary] || []).map(u => get(u, "head", "best_stat")).filter(Boolean);
      const share = bs.length ? bs.filter(b => b === "msp").length / bs.length : 0;
      if (share > MSP_SHARE_MAX) { label = "NOT_EVALUABLE"; note = `MSP chosen as the best head statistic by ${(100 * share).toFixed(0)}% of units (reported for CM-4)`; }
    }
    if (c.anchor && gates.anchors[c.anchor] === "FAIL") label = `NOT_EVALUABLE (gate: anchors.${c.anchor})`;
    if (rule.invalid && rule.invalid.length) label = "NOT_EVALUABLE (rules: a fraction looser than P1)";
    if (hard.length) label = `NOT_EVALUABLE (gate: ${hard.join(", ")})`;
    if (rule.status === "INFO") label = `INFO (withdrawn at P2: ${rule.reason || "no reason given"})`;
    if (replayBad && !label.startsWith("INFO")) label += " [REPLAY-DRIFT]";
    if (dry) label = `INFO (dry run) ${label}`;
    else if (phase === "discovery") label = `INFO (discovery) ${label}`;
    out.push({ id: c.id, inventory_rows: c.rows, text: c.text, predicted: c.predicted, type: c.type, primary, label, note,
      scopes: Object.fromEntries(Object.entries(scopes).map(([k, v]) => [k, { ...v, secondary: k !== primary }])) });
  }
  const report = {
    schema: SCHEMA, phase, dry_run: dry, created: new Date().toISOString(), repo_head: dry ? null : git.head(),
    p1: opts.p1 || null, p2: opts.p2 || null, p_run: opts.pRun || [], cuts: cuts || [],
    constants: { ADD, EQ, ADD_PIX, MIN_EVALUABLE, ALPHA, BAND_LEVEL, MSP_SHARE_MAX, DO3, ANCHOR_TOL,
      bands: { R: band(2500, 1500), M: band(750, 1500) } },
    units: Object.fromEntries(Object.entries(sets).map(([k, v]) => [k, v.map(u => u.unit)])), planned,
    duplicates: { boards: bsel.dups, streams: ssel.dups }, gates, claims: out,
    functional: functional(sets), rules_applied: rr.per,
  };
  return report;
}

// ---- functional units (D14): medians and ranges over units, never a verdict --------------------------------------------------
function functional(sets) {
  const pairs = [["err_clean", "margin_pen"], ["err_clean", "penult"], ["err_clean", "all_geom"], ["err_clean", "local_pen"],
    ["err_shift_s3", "pre"], ["err_shift_s3", "early"], ["err_shift_s3", "all_geom"], ["corrupt_N_s3", "x4"],
    ["corrupt_B_s3", "x4"], ["corrupt_N_s3", "early"], ["ood_svhn", "x4"], ["ood_cifar100", "x4"],
    ["fault_deadpix", "early"], ["fault_occlusion_disc", "early"], ["fault_exposure_global", "early"]];
  const summ = v => { const a = v.filter(isNum); return a.length ? { median: median(a), min: Math.min(...a), max: Math.max(...a), n: a.length } : null; };
  const out = {};
  for (const sc of ["R", "R2", "M"]) {
    const units = sets[sc] || [];
    if (!units.length) continue;
    const t = {};
    for (const [tg, b] of pairs) {
      const js = units.map(u => get(u, "persample", tg, "joint", b)).filter(Boolean);
      if (js.length) t[`${tg}|${b}`] = { dauc: summ(js.map(j => j.dauc)), dtpr5: summ(js.map(j => j.dtpr5)), daurc: summ(js.map(j => j.daurc)) };
    }
    const ms = {}, rb = {};
    for (const u of units) {
      for (const [k, v] of Object.entries(get(u, "cost", "timing_s", "ms_per_1000_queries") || {})) (ms[k] = ms[k] || []).push(v);
      for (const [k, v] of Object.entries(get(u, "cost", "ref_bytes") || {})) (rb[k] = rb[k] || []).push(v);
    }
    out[sc] = { increments: t, cpu_ms_per_1000_queries: Object.fromEntries(Object.entries(ms).map(([k, v]) => [k, summ(v)])),
      reference_bytes: Object.fromEntries(Object.entries(rb).map(([k, v]) => [k, summ(v)])),
      x4_fpr_B: summ(units.map(u => get(u, "x4", "fpr_B"))),
      x4_fpr_curve: ["0.02", "0.03", "0.04", "0.05"].map(a => {                     // rule 5 (D14): the curve, not one cut
        const n = units.map(u => [get(u, "x4", "n_cal_fusion"), get(u, "x4", "n_B")]).find(v => v.every(isNum));
        const bd = n ? S.betaBinomBand(n[0], n[1], +a, BAND_LEVEL) : null;
        return { alpha: +a, fpr: summ(units.map(u => get(u, "x4", "fpr_curve", a))), band99: bd ? [bd.lo, bd.hi] : null };
      }) };
  }
  const st = sets.S || [];
  if (st.length) {
    const fa = {};
    for (const s of st) for (const [k, v] of Object.entries(s.arl || {})) (fa[k] = fa[k] || []).push(v.fa_per_1000_heldout);
    out.S = { false_alarms_per_1000_frames: Object.fromEntries(Object.entries(fa).map(([k, v]) => [k, summ(v)])) };
  }
  return out;
}

function emitRules(report, opts, git) {
  const claims = claimDefs(deciders(0), () => ({}), "discovery");
  let am = { out: { withdraw: {}, frac: {} }, errs: [] };
  if (opts.amend) am = applyAmend(claims, readJSON(opts.amend));
  if (am.errs.length) throw new Error(`--amend refused: ${am.errs.join("; ")}`);
  const disc = Object.fromEntries(report.claims.map(c => [c.id, c.label]));
  return {
    schema: RULES_SCHEMA, created: new Date().toISOString(), p1: opts.p1 || null, head: git ? git.head() : null,
    evaluator_sha256: S.fileSha256(__filename), rule: "D8: only withdrawals to INFO (b) and stricter fractions (c) differ from P1",
    claims: Object.fromEntries(claims.map(c => [c.id, { status: am.out.withdraw[c.id] ? "INFO" : "ACTIVE",
      reason: am.out.withdraw[c.id] || null, predicted: c.predicted, frac: { ...c.scopes, ...(am.out.frac[c.id] || {}) } }])),
    discovery_labels: disc, amendments: am.out,
  };
}

function writeNew(p, obj) {
  if (fs.existsSync(p)) { const e = new Error(`${p} exists: not overwritten (write a new file, e.g. *_v2.json)`); e.code = 2; throw e; }
  fs.mkdirSync(path.dirname(path.resolve(p)), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 1) + "\n");
}

function main(argv, gitIn) {
  const opt = (k, dflt) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt; };
  const has = k => argv.includes(k);
  const repo = path.resolve(opt("--repo", path.resolve(__dirname, "..")));
  const opts = {
    repo, results: opt("--results"), registry: opt("--registry"), phase: opt("--phase", "confirmation"), dry: has("--dry-run"),
    p1: opt("--p1"), p2: opt("--p2"), pRun: (opt("--p-run", "") || "").split(",").map(s => s.trim()).filter(Boolean),
    rules: opt("--rules"), cuts: has("--cut") ? opt("--cut", "").split(/\s+/).filter(Boolean) : null,
    checkDirs: has("--check-dirs") ? opt("--check-dirs", "").split(",").filter(Boolean) : null, amend: opt("--amend"),
  };
  const outJson = opt("--json"), outRules = opt("--emit-rules");
  if (outJson && fs.existsSync(outJson)) { console.error(`${outJson} exists: not overwritten (write a new file, e.g. *_v2.json)`); return 2; }
  if (outRules && fs.existsSync(outRules)) { console.error(`${outRules} exists: not overwritten`); return 2; }
  if (outRules && opts.phase !== "discovery") { console.error("--emit-rules is a discovery-phase output (P2)"); return 2; }
  const git = gitIn || realGit(repo);
  let report;
  try { report = evaluate(opts, git); } catch (e) { console.error(`t1_eval: ${e.message}`); return 2; }
  for (const c of report.claims) {
    const p = c.scopes[c.primary];
    const cnt = c.type === "bool" ? `${p.k ?? "-"}/${p.n}` : `A${p.k_adds ?? "-"} B${p.k_bounded ?? "-"} I${p.k_inconclusive ?? "-"} /${p.n}`;
    console.log(`${c.id.padEnd(6)} ${String(c.label).padEnd(40)} predicted ${c.predicted.padEnd(9)} ${c.primary} ${cnt}  ${c.text.slice(0, 80)}`);
  }
  console.log(`gates: ${Object.entries(report.gates).map(([k, v]) => `${k} ${v.status}`).join("; ")}`);
  try {
    if (outRules) { writeNew(outRules, emitRules(report, opts, report.dry_run ? null : git)); console.log(`wrote ${outRules}`); }
    if (outJson) { writeNew(outJson, report); console.log(`wrote ${outJson}`); }
  } catch (e) { console.error(`t1_eval: ${e.message}`); return e.code === 2 ? 2 : 1; }
  return 0;
}

module.exports = { evaluate, emitRules, main, claimDefs, scopeResult, call3, call3pix, deciders, validateSchema, realGit,
  applyAmend, rulesFor, T1_FILES, SCRIPT_OF, constants: { ADD, EQ, ADD_PIX, MIN_EVALUABLE, ALPHA, BAND_LEVEL, DO3, ANCHOR_TOL } };
if (require.main === module) process.exit(main(process.argv.slice(2)));
