// B1 verdicts (docs/plans/B1_VIT_MARGIN.md; pre-registration experiments/queue/margin_b1_vitb16.yaml notes).
// Committed with the pre-registration (commit P): it IS the decision rule, applied to the pulled atlas.json files, the
// pod's gate record (results/b1_gate/gate.json) and the B1 check dir. Node only.
//   node scripts/b1_verdicts.js --p <P> --p-run <P_run>[,<P_run2>] [--a4b results/atlas_v1_resnet56_s1/a4b_eval.json]
//                               [--root results] [--check-dir instrument_check_b1] [--json <out.json>] [--quiet]
//   node tests/fixtures/b1/check_fixtures.js    (every decision branch on the committed fixtures; run before P)
// --root is the results tree (the fixtures use their own); --check-dir is relative to it (default: the directory the
// pod gate recorded, else instrument_check_b1). --p / --p-run are git revisions of this repo (provenance, D17 item 4;
// a comma list when a committed amendment relaunched _r2 runs). --a4b adds the joint reading of A4b M56-b (D13).
// Decision order: gate record, G0 (plumbing), G1/G2 (legacy reproduction), G3 (positive control), post-gate P0, c*,
// then A / B / UNDECIDED / MIXED; margin_vs_maxprob and margin_vs_logitgap separately. Every threshold below is quoted
// in the manifest notes and in scripts/b1_gate.py; change neither without a new pre-registration. An evaluator bug
// found after the run is fixed only by a committed amendment listed in SESSION.md, and the outcomes are then reported
// under both versions.
"use strict";
const fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
const argv = process.argv.slice(2);
const arg = (k, d = null) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const ROOT = arg("--root", "results");
const P_ARG = arg("--p"), PRUN_ARG = arg("--p-run"), A4B_ARG = arg("--a4b"), JSON_OUT = arg("--json");
const QUIET = argv.includes("--quiet");
const REPO = path.resolve(__dirname, "..");
const R = (...a) => path.join(ROOT, ...a);
const loadJ = p => { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } };
const load = n => loadJ(R(n, "atlas.json"));
const PMOF = (a, layer = "penult") => a && a.per_layer && a.per_layer[layer] ? a.per_layer[layer].margin_typeb || null : null;
const num = v => typeof v === "number" && Number.isFinite(v);
const U = { ok: "\u2705", y: "\u{1F7E1}", x: "\u2717", open: "\u2B1C" };

const SESOI = 0.01, CI_EQ = 0.02, P = 0.05, NMIN = 300, CUTS = [0.7, 0.5];
const OFFICIAL = { resnet50: 0.80858, vitb16: 0.81072, deitb: 0.8198 };        // torchvision docs; timm results csv
const BAND = { resnet50: [-0.06, 0.01], vitb16: [-0.10, 0.01], deitb: [-0.10, 0.01] };   // P0 acc band (D14)
const LEGACY_N = [10048, 10111], LEGACY_ACC = [0.766, 0.776], LEGACY_SEP = [1.12, 1.16];
const G1_REF = 0.800, G1_FLOOR = 0.035, G2_REF = 0.636, G2_FLOOR = 0.04, PAIR_TOL = 0.05;
const MODELS = ["resnet50", "vitb16", "deitb"], VITS = ["vitb16", "deitb"];
const SELFTESTS = ["margin_b1_vitb16", "margin_b1_deitb"];
const E9 = ["resnet20_s0hub_st3", "resnet56_s0hub_st3", "resnet56_s1", "resnet56_s2"];
const B1_MANIFESTS = ["resnet50_legacy10k", "resnet50", "resnet50_swap", "vitb16", "vitb16_swap", "deitb", "deitb_swap", ...E9];
const gateRec = loadJ(R("b1_gate", "gate.json"));
const CHECK = arg("--check-dir", gateRec && gateRec.check_dir ? path.basename(gateRec.check_dir) : "instrument_check_b1");
const row = (pm, c) => (pm && pm.sweep || []).find(r => Math.abs(r.cut - c) < 1e-9) || null;
const base = s => String(s).replace(/\(.*\)$/, "");
const out = { args: { root: ROOT, check_dir: CHECK, p: P_ARG, p_run: PRUN_ARG, a4b: A4B_ARG },
  plumbing_pre: [], plumbing_post: [], gate: {}, cstar: null, models: {}, labels: {}, info: {}, exploratory: {} };

// ---- P0 plumbing (every B1 ImageNet dump; notes P0) --------------------------------------------------------------
function p0(name, legacy, sink) {
  const a = load(`margin_b1_${name}`), m = a && a.meta, pm = PMOF(a), f = msg => sink.push(`${name}: ${msg}`);
  if (!a || !m || !pm) { f("atlas.json / meta / penult margin_typeb missing"); return null; }
  if (pm.error) f(`penult margin_typeb error ${String(pm.error).slice(0, 80)}`);
  if (m.source !== "real") f(`source ${m.source}`);
  if (m.n_classes !== 1000) f(`n_classes ${m.n_classes}`);
  if (!(num(m.head_check_max_abs) && m.head_check_max_abs <= 1e-3)) f(`head_check_max_abs ${m.head_check_max_abs}`);
  if (m.parquet_sha256_ok !== true) f("parquet sha256 not verified against the pinned revision");
  if (!legacy) {                                     // the legacy code itself skipped undecodable rows (review item 17)
    if (m.decode_failures !== 0) f(`decode_failures ${m.decode_failures}`);
    if (m.n_test !== 25000) f(`n_test ${m.n_test}`);
    const c = m.per_class_counts || {};
    for (const s of ["ref", "test"]) if (!(c[s] && c[s][0] === 25 && c[s][1] === 25)) f(`per-class ${s} min/max ${JSON.stringify(c[s])}`);
    if (m.ref_test_disjoint !== true) f("ref and test indices not disjoint");
    const b = name.replace(/_r2$/, "").replace(/_swap$/, ""), lo = OFFICIAL[b] + BAND[b][0], hi = OFFICIAL[b] + BAND[b][1];
    if (!(num(pm.acc) && pm.acc >= lo && pm.acc <= hi)) f(`test acc ${pm.acc} outside [${lo.toFixed(4)}, ${hi.toFixed(4)}]`);
  }
  return { a, m, pm };
}

// ---- gate G (recomputed exactly as scripts/b1_gate.py, then checked against the pod's record) --------------------
const L = p0("resnet50_legacy10k", true, out.plumbing_pre), M50 = p0("resnet50", false, out.plumbing_pre);
{
  const g = out.gate, lg = (L && L.pm.legacy_imagenet) || {}, n = L ? L.m.n_test : null;
  const st = Object.fromEntries(SELFTESTS.map(e => [e, (loadJ(R(CHECK, `selftest_${e}.json`)) || {}).status || null]));
  const data = loadJ(R(CHECK, "data.json")) || {}, real = data.real || {};
  const dataOk = data.status === "PASS" && (!real.ok || (num(real.label_in_real_frac) && real.label_in_real_frac >= 0.5));
  g.G0 = !!(L && M50) && out.plumbing_pre.length === 0 && num(n) && n >= LEGACY_N[0] && n <= LEGACY_N[1]
    && num(lg.acc) && lg.acc >= LEGACY_ACC[0] && lg.acc <= LEGACY_ACC[1]
    && num(lg.valley_sep_legacy) && lg.valley_sep_legacy >= LEGACY_SEP[0] && lg.valley_sep_legacy <= LEGACY_SEP[1]
    && Object.values(st).every(s => s === "PASS") && dataOk;
  g.G0_detail = { n_legacy: n, acc: lg.acc, valley_sep_legacy: lg.valley_sep_legacy, n_cw: lg.n_cw, n_classes_centered: lg.n_classes_centered,
    legacy_decode_failures: L ? L.m.decode_failures : null, selftests: st, data_status: data.status || null,
    real: { ok: !!real.ok, label_in_real_frac: real.label_in_real_frac ?? null } };
  const w = Math.max(G1_FLOOR, 2.58 * (lg.dir_auc_margin_sub_sd || 0) + 0.005);
  const wc = Math.max(G2_FLOOR, 2.58 * (lg.dir_auc_cluster_sub_sd || 0) + 0.005);
  g.G1 = num(lg.dir_auc_margin_full) && num(lg.raw_auc_margin_full) && Math.abs(lg.dir_auc_margin_full - G1_REF) <= w && lg.raw_auc_margin_full < 0.5;
  g.G1_detail = { dir_auc_margin_full: lg.dir_auc_margin_full ?? null, raw: lg.raw_auc_margin_full ?? null, band_halfwidth: w,
    sub_mean: lg.dir_auc_margin_sub_mean ?? null, sub_q025: lg.dir_auc_margin_sub_q025 ?? null, sub_q975: lg.dir_auc_margin_sub_q975 ?? null };
  g.G2 = ["margin", "cluster", "energy"].every(s => num(lg[`dir_auc_${s}_full`]))
    && lg.dir_auc_margin_full > lg.dir_auc_cluster_full && lg.dir_auc_cluster_full > lg.dir_auc_energy_full
    && Math.abs(lg.dir_auc_cluster_full - G2_REF) <= wc;
  g.G2_detail = { cluster: lg.dir_auc_cluster_full ?? null, energy: lg.dir_auc_energy_full ?? null, band_halfwidth: wc };
  let ev = 0, ok = true; const det = {};
  for (const c of CUTS) { const r = M50 ? row(M50.pm, c) : null;
    if (!r || !(r.n_typeb >= NMIN)) { det[c] = { n_typeb: r ? r.n_typeb : null, evaluable: false }; continue; }
    ev++; const d = r.margin_minus_dist_typeb, p = r.margin_minus_dist_typeb_p;
    const pass = num(d) && num(p) && d >= SESOI && p < P; ok = ok && pass; det[c] = { n_typeb: r.n_typeb, d, p, pass }; }
  g.G3 = ev > 0 && ok; g.G3_detail = det;
  g.open = ["G0", "G1", "G2", "G3"].every(k => g[k] === true);
  g.pod_record = gateRec ? { gate: gateRec.gate, open: gateRec.open, check_dir: gateRec.check_dir } : "MISSING";
  g.pod_record_agrees = !!gateRec && !!gateRec.gate && ["G0", "G1", "G2", "G3"].every(k => gateRec.gate[k] === g[k]);
}
const gateOpen = out.gate.open;

// ---- post-gate runs: P0, the _r2 rule (review item 2), the discovery twin (review item 7) -------------------------
const runs = {}, used = {};
if (M50) { runs.resnet50 = M50; used.resnet50 = "margin_b1_resnet50"; }
{ const sink = [], r = p0("resnet50_swap", false, sink);   // discovery twin: a fault is INFO, never INVALID-PLUMBING
  if (sink.length) out.info.resnet50_swap_plumbing = sink; else if (r) { runs.resnet50_swap = r; used.resnet50_swap = "margin_b1_resnet50_swap"; } }
for (const m of VITS) for (const sfx of ["", "_swap"]) {
  const name = m + sfx;
  if (!gateOpen) { if (load(`margin_b1_${name}`)) out.info[`${name}_present_with_gate_closed`] = true; continue; }
  const sink = []; let r = p0(name, false, sink);
  if (sink.length && load(`margin_b1_${name}_r2`)) {
    const accOnly = sink.every(s => s.includes(": test acc "));
    if (accOnly) out.info[`rerun_${name}`] = { original: sink, rerun: "REFUSED: an acc-band failure alone is final (INVALID-PLUMBING, the model is spent)" };
    else { const s2 = [], r2 = p0(`${name}_r2`, false, s2); out.info[`rerun_${name}`] = { original: sink.slice(), rerun: s2.length ? s2 : "PASS" };
      if (!s2.length) { r = r2; sink.length = 0; used[name] = `margin_b1_${name}_r2`; } else sink.push(...s2); }
  }
  out.plumbing_post.push(...sink);
  if (r && !sink.length) { runs[name] = r; used[name] = used[name] || `margin_b1_${name}`; }
}
out.info.runs_used = used;

// ---- c*: the common primary cut (power, mechanical) ---------------------------------------------------------------
for (const c of CUTS) if (out.cstar == null && MODELS.every(m => runs[m] && (row(runs[m].pm, c) || {}).n_typeb >= NMIN)) out.cstar = c;

// ---- per-run states at c* -----------------------------------------------------------------------------------------
function v1(r) { const d = r.margin_minus_dist_typeb, se = r.margin_minus_dist_typeb_se, p = r.margin_minus_dist_typeb_p;
  if (!num(d)) return "NOT_EVALUABLE"; if (d >= SESOI && num(p) && p < P) return "PASS";
  if (d <= -SESOI && num(p) && p < P) return "NULL(REVERSED)";
  if (d < SESOI && num(se) && d + 1.96 * se <= CI_EQ) return "NULL"; return "UNRESOLVED"; }          // review item 6
function v2(r, pm) { const mt = r.median_margin_typeb, mc = pm.median_margin_correct;
  if (!num(mt) || !num(mc) || mc === 0) return "NOT_EVALUABLE"; return mt < mc && mt / mc <= 0.5 ? "PASS" : "FAIL"; }
function v3(r, key) { const d = r[key], se = r[key + "_se"], p = r[key + "_p"];
  if (!num(d)) return "NOT_EVALUABLE"; if (d >= SESOI && num(p) && p < P) return "ADDS";
  if (d <= -SESOI && num(p) && p < P) return "BELOW";
  if (Math.abs(d) < SESOI && num(se) && d - 1.96 * se >= -CI_EQ && d + 1.96 * se <= CI_EQ) return "EQUIVALENT"; return "UNRESOLVED"; }
const NE = { V1: "NOT_EVALUABLE", V2: "NOT_EVALUABLE", V3: "NOT_EVALUABLE", V3b: "NOT_EVALUABLE" };
if (out.cstar != null) for (const [name, r0] of Object.entries(runs)) {
  const r = row(r0.pm, out.cstar), need = name.endsWith("_swap") ? 1 : NMIN;   // twins are read at c* whatever their count
  if (!r || !(r.n_typeb >= need)) { out.models[name] = { acc: r0.pm.acc ?? null, n_typeb: r ? r.n_typeb : null, ...NE }; continue; }
  out.models[name] = { acc: r0.pm.acc ?? null, n_typeb: r.n_typeb, V1: v1(r), V2: v2(r, r0.pm),
    V3: v3(r, "margin_minus_maxprob_confmatched"), V3b: v3(r, "margin_minus_logitgap_confmatched"),
    values: { margin_minus_dist_typeb: r.margin_minus_dist_typeb ?? null, se: r.margin_minus_dist_typeb_se ?? null,
      p: r.margin_minus_dist_typeb_p ?? null, ratio: num(r.median_margin_typeb) && num(r0.pm.median_margin_correct)
        ? r.median_margin_typeb / r0.pm.median_margin_correct : null,
      margin_minus_maxprob_confmatched: r.margin_minus_maxprob_confmatched ?? null, se3: r.margin_minus_maxprob_confmatched_se ?? null,
      p3: r.margin_minus_maxprob_confmatched_p ?? null, margin_minus_logitgap_confmatched: r.margin_minus_logitgap_confmatched ?? null,
      se3b: r.margin_minus_logitgap_confmatched_se ?? null, p3b: r.margin_minus_logitgap_confmatched_p ?? null,
      auc_margin_typeb: r.auc_margin_typeb ?? null, auc_dist_typeb: r.auc_dist_typeb ?? null, auc_maxprob_typeb: r.auc_maxprob_typeb ?? null } };
}
// split robustness (review item 7): a state that differs from the swap twin's is SPLIT-FRAGILE; a twin state that is
// missing or not evaluable leaves the primary's state SPLIT-UNTESTED; both read as UNRESOLVED
for (const m of MODELS) { const a = out.models[m]; if (!a) continue; const b = out.models[m + "_swap"] || {};
  for (const k of ["V1", "V2", "V3", "V3b"]) { if (a[k] === "NOT_EVALUABLE") continue;
    if (!b[k] || b[k] === "NOT_EVALUABLE") { a[k + "_own"] = a[k]; a[k + "_swap"] = b[k] || null; a[k] = "SPLIT-UNTESTED"; }
    else if (base(a[k]) !== base(b[k])) { a[k + "_own"] = a[k]; a[k + "_swap"] = b[k]; a[k] = "SPLIT-FRAGILE"; } } }

// ---- labels, applied in order (notes: Decision) --------------------------------------------------------------------
const lab = out.labels, g = out.gate;
const s = VITS.map(m => out.models[m] || NE), V1 = s.map(x => x.V1), V2 = s.map(x => x.V2);
const split = x => x === "SPLIT-FRAGILE" || x === "SPLIT-UNTESTED";
if (!g.pod_record_agrees) lab.outcome = "INVALID-PLUMBING (gate record)";
else if (!g.G0) lab.outcome = "INVALID-PLUMBING (gate G0)";
else if (!(g.G1 && g.G2)) lab.outcome = "INVALID-REPRO";
else if (!g.G3) lab.outcome = "INVALID-CONTROL";
else if (out.plumbing_post.length) lab.outcome = "INVALID-PLUMBING (post-gate run)";
else if (out.cstar == null) lab.outcome = "UNDECIDED-POWER";
else {
  if (V1.every(x => x === "PASS") && V2.every(x => x === "PASS")) lab.outcome = "A";
  else if (V1.every(x => base(x) === "NULL"))
    lab.outcome = V2.every(x => x === "PASS") ? "B (type-b still low-margin)" : V2.every(x => x === "FAIL") ? "B (not low-margin)" : "B (ridge mixed)";
  else if (V1.some(x => x === "UNRESOLVED" || split(x) || x === "NOT_EVALUABLE") || V2.some(split)) lab.outcome = "UNDECIDED";
  else lab.outcome = "MIXED";
  const agg = v => v.every(x => x === "ADDS") ? "ADDS" : v.every(x => x === "EQUIVALENT") ? "MARGIN~CONFIDENCE"
    : v.every(x => x === "BELOW") ? "BELOW" : "MIXED/UNRESOLVED";
  lab.margin_vs_maxprob = agg(s.map(x => x.V3)); lab.margin_vs_logitgap = agg(s.map(x => x.V3b));
  if (lab.margin_vs_maxprob === "ADDS" && lab.margin_vs_logitgap !== "ADDS") lab.margin_vs_maxprob = "ADDS-OVER-MAXPROB-ONLY";   // H2b
  lab.V4_replication = ["V1", "V2", "V3"].every(k => base(s[0][k]) === base(s[1][k])) ? "PASS" : "FAIL";
  const r50 = out.models.resnet50 || NE; lab.resnet50_contrast = { V1: r50.V1, V2: r50.V2, V3: r50.V3, V3b: r50.V3b };
}
if (lab.outcome.startsWith("B")) {                   // review item 21, D13 row 3
  const cancel = V1.every(x => base(x) === "NULL" && x.includes("REVERSED")) || V2.every(x => x === "FAIL");
  lab.B_reading = { claim: "margin loses its advantage over distance on ViT", b2_cancel_and_cnn_scope: cancel,
    rule: "B2 is cancelled and Phase 3 is CNN-scoped only if V1 is NULL(REVERSED) on both ViTs or V2 FAILs on both ViTs; otherwise the roadmap decision goes to D-AB with row(c*).auc_margin_typeb of both ViTs beside resnet50's",
    auc_margin_typeb_at_cstar: Object.fromEntries(MODELS.map(m => [m, ((out.models[m] || {}).values || {}).auc_margin_typeb ?? null])) };
}
const accOf = m => runs[m] && num(runs[m].pm.acc) ? runs[m].pm.acc : null;
lab.accuracy_control = VITS.every(m => accOf(m) !== null && accOf("resnet50") !== null && Math.abs(accOf(m) - accOf("resnet50")) <= 0.02)
  ? "ACC-MATCHED" : "ACC-CONFOUNDED (level comparisons only; V1-V3 are within-model)";

// ---- promotion inputs (row 10; review item 5: the ViT pair at c*) --------------------------------------------------
if (out.cstar != null && runs.vitb16 && runs.deitb) {
  const a = row(runs.vitb16.pm, out.cstar) || {}, b = row(runs.deitb.pm, out.cstar) || {}, it = [];
  for (const k of ["auc_margin_typeb", "auc_dist_typeb", "auc_maxprob_typeb"]) {
    const sp = num(a[k]) && num(b[k]) ? Math.abs(a[k] - b[k]) : null;
    it.push({ key: k, spread: sp, status: sp !== null && sp <= PAIR_TOL ? "PASS" : "FAIL" }); }  // tolerances_default abs 0.05
  const ra = a.median_margin_typeb / runs.vitb16.pm.median_margin_correct, rb = b.median_margin_typeb / runs.deitb.pm.median_margin_correct;
  const sp = Math.abs(ra - rb);
  it.push({ key: "median_ratio", spread: num(sp) ? sp : null,
    status: num(sp) && (sp <= 0.05 || sp / (Math.abs((ra + rb) / 2) + 1e-9) <= 0.15) ? "PASS" : "FAIL" });   // critic default rule
  out.info.vit_pair_at_cstar = it;
}
const pairDir = used.vitb16 && used.vitb16.endsWith("_r2") || used.deitb && used.deitb.endsWith("_r2") ? "critic_b1_vit_pair_r2" : "critic_b1_vit_pair";
const cr = loadJ(R(pairDir, "critic.json"));
const critItems = cr ? ((cr.checks || {}).scalar_stability || []).filter(i => i.name.startsWith("penult/margin_typeb.")) : [];
out.info.critic_vit_pair = cr ? { dir: pairDir, penult: critItems.map(i => `${i.status} ${i.name} ${i.detail}`),
  synthetic_refusal: ((cr.checks || {}).synthetic_refusal || []).map(i => i.status), holdout_hygiene: ((cr.checks || {}).holdout_hygiene || []).map(i => i.status) } : `MISSING ${pairDir}`;
for (const m of ["resnet50", "vitb16", "deitb"]) { const c = loadJ(R(`critic_b1_${m}_swap`, "critic.json"));
  out.info[`critic_${m}_swap`] = c ? ((c.checks || {}).scalar_stability || []).filter(i => i.name.startsWith("penult/margin_typeb.")).map(i => `${i.status} ${i.name}`) : "MISSING"; }

// ---- provenance (D17 item 4; the A4b G0e pattern) ------------------------------------------------------------------
const FROZEN = [
  ...B1_MANIFESTS.map(n => `experiments/queue/margin_b1_${n}.yaml`),
  ...["s0hub", "s1", "s2", "s3", "s4"].flatMap(x => [`atlas_v1_resnet20_${x}_st3`, `margin_v1_resnet20_${x}_st3`]).map(n => `experiments/queue/${n}.yaml`),
  ...["s0hub_st3", "s0hub_ref1", "e40_st3", "e50", "e60", "e70", "e90", "s12m", "s13m", "s1", "s2"].map(x => `experiments/queue/atlas_v1_resnet56_${x}.yaml`),
  ...["s0hub_st3", "s1", "s2", "e50", "e60", "e70", "e90", "s12m", "s13m"].map(x => `experiments/queue/margin_v1_resnet56_${x}.yaml`),
  "docs/plans/STAGE2B.md", "docs/plans/STAGE2.md", "docs/plans/B1_VIT_MARGIN.md", "scripts/a4b_eval.js", "scripts/b1_verdicts.js",
  "scripts/b1_gate.py", "experiments/tolerances_default.yaml", "ATLAS_STATUS.md"];
function provenance() {
  const rep = { p: P_ARG, p_run: PRUN_ARG, runs: {} }, probs = [];
  if (!P_ARG || !PRUN_ARG) return { ...rep, status: "NOT_EVALUABLE", problems: ["--p <P> and --p-run <P_run> are required"] };
  const git = (...a) => execFileSync("git", a, { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  const gitOk = (...a) => { try { execFileSync("git", a, { cwd: REPO, stdio: "ignore" }); return true; } catch (e) { return false; } };
  let p, prs;
  try { p = git("rev-parse", "--verify", `${P_ARG}^{commit}`); prs = PRUN_ARG.split(",").map(x => git("rev-parse", "--verify", `${x.trim()}^{commit}`)); }
  catch (e) { return { ...rep, status: "FAIL", problems: [`cannot resolve --p ${P_ARG} / --p-run ${PRUN_ARG} in ${REPO}`] }; }
  for (const pr of prs) {
    if (!gitOk("merge-base", "--is-ancestor", p, pr)) probs.push(`${pr.slice(0, 7)} is neither P nor a descendant of P`);
    if (!gitOk("diff", "--quiet", p, pr, "--", ...FROZEN)) probs.push(`frozen files differ between P and ${pr.slice(0, 7)}: ${git("diff", "--name-only", p, pr, "--", ...FROZEN).split("\n").join(", ")}`);
  }
  const pTime = Date.parse(git("show", "-s", "--format=%cI", p));
  rep.p_time_utc = new Date(pTime).toISOString();
  const sameCommit = c => typeof c === "string" && c.length >= 7 && prs.some(pr => pr.startsWith(c));
  for (const n of B1_MANIFESTS.flatMap(x => [x, `${x}_r2`]).map(x => `margin_b1_${x}`)) {
    const a = load(n); if (!a) continue;
    const m = a.meta || {}, prov = loadJ(R(n, "provenance.json")) || {}, bad = [];
    const created = typeof m.created === "string" ? Date.parse(m.created.replace(" ", "T") + "Z") : NaN;
    if (!sameCommit(m.git_commit)) bad.push(`meta.git_commit ${m.git_commit}`);
    if (!sameCommit(prov.stage_b_git_commit)) bad.push(`stage_b_git_commit ${prov.stage_b_git_commit}`);
    if (!(created > pTime)) bad.push(`meta.created ${m.created} not after P (${rep.p_time_utc})`);
    rep.runs[n] = bad.length ? bad : "PASS"; if (bad.length) probs.push(`${n}: ${bad.join("; ")}`);
  }
  if (!Object.keys(rep.runs).length) probs.push("no B1 atlas found");
  return { ...rep, status: probs.length ? "FAIL" : "PASS", problems: probs };
}
out.provenance = provenance();

// ---- row 10 (ATLAS_STATUS; notes "Row 10") ---------------------------------------------------------------------------
{
  const o = lab.outcome, blocked = [];
  if (out.provenance.status !== "PASS") blocked.push(`provenance ${out.provenance.status}`);
  let status, why;
  if (o === "A") {
    const pair = out.info.vit_pair_at_cstar || [], st = n => (critItems.find(i => i.name === `penult/margin_typeb.${n}`) || {}).status;
    const pairOk = pair.length > 0 && pair.every(i => i.status === "PASS"), wrongOk = st("auc_margin_wrong") === "PASS";
    const restOk = out.cstar === 0.5 || (critItems.length > 0 && critItems.every(i => i.status === "PASS"));   // INFO at c* = 0.5
    const hyg = cr && ["synthetic_refusal", "holdout_hygiene"].every(k => ((cr.checks || {})[k] || []).length > 0 && cr.checks[k].every(i => i.status === "PASS"));
    const q = { "MARGIN~CONFIDENCE": "margin ~ maxprob (not an independent detector)", ADDS: "margin adds over maxprob (+d3)",
      "ADDS-OVER-MAXPROB-ONLY": "margin adds over maxprob but not over the logit gap (maxprob resolution, H2b)" }[lab.margin_vs_maxprob]
      || `margin vs maxprob ${lab.margin_vs_maxprob}`;
    if (pairOk && wrongOk && restOk && hyg) { status = `${U.ok} vitb16, deitb`; why = `A at c* ${out.cstar}; ${q}`; }
    else { status = `${U.y} states replicate, levels do not`; why = `A, pair items: ${JSON.stringify({ pairOk, wrongOk, restOk, hygiene: !!hyg })}; ${q}`; }
  } else if (o.startsWith("B")) { status = U.x; why = "margin does not beat distance on ImageNet ViT-B/16 (both); the same-session ResNet50 does (G3)"; }
  else if (o === "MIXED" || o === "UNDECIDED") { status = U.y; why = `${o}: V1 ${V1.join(" / ")}, V2 ${V2.join(" / ")}`; }
  else { status = U.open; why = `${o}: row 10 stays proposed`; }
  out.row10 = { status, why, promotion_blocked_by: blocked };
}

// ---- joint reading with A4b M56-b (D13; the table is in STAGE2B.md "M56" and in the notes) -------------------------
function joint() {
  if (!A4B_ARG) return { status: "NOT_EVALUABLE (no --a4b file)" };
  const a4 = loadJ(A4B_ARG); if (!a4) return { status: `NOT_EVALUABLE (${A4B_ARG} unreadable)` };
  const b = (a4.M56 || {}).b_row9_d56 || null, tag = b ? b.tag : null, attr = b ? b.m2_failure_attribution ?? null : null;
  const m56 = !tag ? "MISSING" : /PROMOTE/.test(tag) ? "PROMOTE" : /YELLOW/.test(tag) ? "YELLOW" : /REJECT/.test(tag) ? "REJECT" : "NOT_EVALUABLE";
  const o = lab.outcome, rep = { m56_tag: tag, m56: m56, m2_failure_attribution: attr, reason: b ? b.m2_failure_attribution_reason ?? null : null, b1_outcome: o };
  if (/^INVALID|^UNDECIDED-POWER/.test(o)) {
    rep.row = 1; rep.reading = "no A/B label";
    if (o === "INVALID-CONTROL" && m56 === "REJECT") rep.reading += "; recorded as \"the atlas-definition margin > distance does not hold in fully fit CNNs\"; MASTER exit clause 1 goes to D-AB";
  } else if (o === "A") { rep.row = 2; rep.reading = "outcome A as pre-registered; M56-b reported beside it"; }
  else if (o.startsWith("B")) {
    const cancel = lab.B_reading.b2_cancel_and_cnn_scope;
    if (m56 === "PROMOTE" || m56 === "YELLOW") { rep.row = 3; rep.reading = "margin loses its advantage over distance on ViT"; rep.b2_cancel_and_cnn_scope = cancel; }
    else if (m56 === "REJECT" && attr === "NOT-DEPTH") { rep.row = 4; rep.b2_cancel_and_cnn_scope = false;
      rep.reading = "not evidence for \"CNN-specific\": the clause also fails in a fully fit CNN; the decision goes to D-AB with V3 and row(c*).auc_margin_typeb of both ViTs and resnet50; B2 is not cancelled on this basis";
      rep.d_ab_inputs = { V3: Object.fromEntries(MODELS.map(m => [m, (out.models[m] || {}).V3 || null])), auc_margin_typeb_at_cstar: lab.B_reading.auc_margin_typeb_at_cstar }; }
    else if (m56 === "REJECT") { rep.row = 5; rep.reading = "recorded as \"depth/architecture-dependent\"; the B2 rule is as in row 3"; rep.b2_cancel_and_cnn_scope = cancel; }
    else { rep.row = null; rep.reading = `M56-b ${m56}: B is read alone, as pre-registered (the B2 rule of row 3)`; rep.b2_cancel_and_cnn_scope = cancel; }
  } else { rep.row = 6; rep.reading = "as pre-registered; M56-b beside it"; }
  return rep;
}
out.joint_reading = joint();

// ---- INFO and exploratory items (never promoted; notes E1-E11, B1-acc) ------------------------------------------------
out.info.b1_acc = { mirror_minus_published: Object.fromEntries(MODELS.map(m => [m, accOf(m) !== null ? accOf(m) - OFFICIAL[m] : null])),
  pod: gateRec ? gateRec.b1_acc_info ?? null : null };
out.info.E9 = Object.fromEntries(E9.map(x => { const pm = PMOF(load(`margin_b1_${x}`));
  return [x, pm ? { margin_minus_maxprob_confmatched: pm.margin_minus_maxprob_confmatched ?? null, se: pm.margin_minus_maxprob_confmatched_se ?? null,
    p: pm.margin_minus_maxprob_confmatched_p ?? null, margin_minus_maxprob_typeb: pm.margin_minus_maxprob_typeb ?? null } : null]; }));
out.info.E9_identity = loadJ(R(CHECK, "e9_identity.json")) || "MISSING";
if (out.cstar != null) {
  const c = out.cstar, X = out.exploratory, runOf = m => runs[m] ? runs[m].a : null;
  const tapRow = (m, layer) => row(PMOF(runOf(m), layer), c);
  const early = m => (runOf(m) ? runOf(m).layers || [] : []).filter(l => m === "resnet50" ? /^(stem|layer1\.\d+|layer2\.[0-3])$/.test(l) : /^block\.[0-5]$/.test(l));
  X.E1 = Object.fromEntries(MODELS.map(m => { const pen = (tapRow(m, "penult") || {}).auc_margin_typeb, e = early(m).map(l => (tapRow(m, l) || {}).auc_margin_typeb).filter(num);
    const first = (runOf(m) ? runOf(m).layers || [] : []).find(l => { const r = tapRow(m, l); return r && v1(r) === "PASS"; }) || null;
    return [m, { penult: pen ?? null, early_max: e.length ? Math.max(...e) : null, holds: num(pen) && e.length ? pen >= Math.max(...e) + 0.10 : null, first_tap_V1_PASS: first }]; }));
  X.E2 = Object.fromEntries(VITS.map(m => { const a = (tapRow(m, "penult") || {}).auc_margin_typeb, b = (tapRow(m, "penult_mean") || {}).auc_margin_typeb;
    return [m, { cls: a ?? null, mean_token: b ?? null, holds: num(a) && num(b) ? a >= b : null }]; }));
  X.E3 = Object.fromEntries(MODELS.map(m => { const pm = runs[m] ? runs[m].pm : {}; const cg = pm.centers_geometry || {};
    return [m, { sep_legacy_ref: cg.sep_legacy_ref ?? null, sep_ratio_ref: cg.sep_ratio_ref ?? null, valley_sep_legacy: (pm.legacy_imagenet || {}).valley_sep_legacy ?? null }]; }));
  const cv = m => runs[m] ? runs[m].pm.d1_plus_d2_cv : null;
  X.E4 = { spearman: Object.fromEntries(Object.entries(runs).map(([m, r]) => [m, { margin_logitgap: r.pm.spearman_margin_logitgap ?? null, margin_maxprob: r.pm.spearman_margin_maxprob ?? null,
      holds: num(r.pm.spearman_margin_logitgap) && num(r.pm.spearman_margin_maxprob) ? r.pm.spearman_margin_logitgap >= r.pm.spearman_margin_maxprob : null,
      nearest_center_agrees_with_model: r.pm.nearest_center_agrees_with_model ?? null, head_center_cos: r.m.head_center_cos ?? null }])),
    d1_plus_d2_cv: Object.fromEntries(MODELS.map(m => [m, cv(m) ?? null])),
    cv_lower_in_both_vits: VITS.every(m => num(cv(m)) && num(cv("resnet50")) && cv(m) < cv("resnet50")) };
  if (runs.resnet50) { const lg5 = runs.resnet50.pm.legacy_imagenet || {}, r5 = row(runs.resnet50.pm, 0.5) || {};   // review item 10
    X.E5 = { in_sample: num(lg5.raw_auc_margin_full) ? 1 - lg5.raw_auc_margin_full : null, reference_A: r5.auc_margin_typeb ?? null,
      holds: num(lg5.raw_auc_margin_full) && num(r5.auc_margin_typeb) ? 1 - lg5.raw_auc_margin_full >= r5.auc_margin_typeb : null }; }
  X.E6 = Object.fromEntries(Object.entries(out.models).filter(([, v]) => base(v.V1_own || v.V1) === "PASS").map(([m]) => {
    const r = row(runs[m].pm, c) || {}; return [m, { d: r.margin_norm_minus_dist_typeb ?? null, p: r.margin_norm_minus_dist_typeb_p ?? null,
      holds: num(r.margin_norm_minus_dist_typeb) && num(r.margin_norm_minus_dist_typeb_p) ? r.margin_norm_minus_dist_typeb > 0 && r.margin_norm_minus_dist_typeb_p < P : null }]; }));
  X.E7 = Object.fromEntries(Object.keys(runs).map(m => { const st = (row(runs[m].pm, c) || {}).strat_confmatched || {};
    const d = num(st.auc_margin) && num(st.auc_maxprob) ? st.auc_margin - st.auc_maxprob : null; return [m, { d, n_bins_used: st.n_bins_used ?? null, lt_002: d === null ? null : Math.abs(d) < 0.02 }]; }));
  const en = m => { const r = row(runs[m] ? runs[m].pm : null, c); return r && num(r.auc_energy_typeb) ? Math.abs(r.auc_energy_typeb - 0.5) : null; };
  X.E8 = { abs_energy_minus_half: Object.fromEntries(MODELS.map(m => [m, en(m)])), smaller_in_both_vits: VITS.every(m => num(en(m)) && num(en("resnet50")) && en(m) < en("resnet50")) };
  X.E10 = Object.fromEntries(Object.keys(runs).map(m => { const r = row(runs[m].pm, c) || {};
    return [m, num(r.median_margin_typeb) && num(r.median_margin_correct_confmatched) && r.median_margin_correct_confmatched ? r.median_margin_typeb / r.median_margin_correct_confmatched : null]; }));
  X.E11 = Object.fromEntries(Object.keys(runs).map(m => { const r = row(runs[m].pm, c) || {};
    return [m, { auc_margin_typeb: r.auc_margin_typeb ?? null, auc_margin_typeb_realwrong: r.auc_margin_typeb_realwrong ?? null,
      n_pos_realwrong: r.n_pos_typeb_realwrong ?? null, n_typeb_real_ok: r.n_typeb_real_ok ?? null, median_margin_typeb_realwrong: r.median_margin_typeb_realwrong ?? null }]; }));
}

// ---- report ----------------------------------------------------------------------------------------------------------
if (!QUIET) {
  const G = out.gate;
  console.log(`B1 verdicts; root ${ROOT}; check dir ${CHECK}`);
  console.log(`gate: G0 ${G.G0} G1 ${G.G1} G2 ${G.G2} G3 ${G.G3} open ${G.open}; pod record agrees ${G.pod_record_agrees}`);
  if (out.plumbing_pre.length) console.log(`plumbing (gate runs): ${out.plumbing_pre.join(" | ")}`);
  if (out.plumbing_post.length) console.log(`plumbing (post-gate runs): ${out.plumbing_post.join(" | ")}`);
  console.log(`c* ${out.cstar}; outcome ${lab.outcome}; margin_vs_maxprob ${lab.margin_vs_maxprob || "-"}; margin_vs_logitgap ${lab.margin_vs_logitgap || "-"}; V4 ${lab.V4_replication || "-"}; ${lab.accuracy_control}`);
  for (const [m, v] of Object.entries(out.models)) console.log(`  ${m}: V1 ${v.V1} V2 ${v.V2} V3 ${v.V3} V3b ${v.V3b} (n_typeb ${v.n_typeb}, acc ${v.acc})`);
  console.log(`row 10: ${out.row10.status} (${out.row10.why})${out.row10.promotion_blocked_by.length ? `; blocked by ${out.row10.promotion_blocked_by.join(", ")}` : ""}`);
  console.log(`joint reading (D13): ${JSON.stringify(out.joint_reading)}`);
  console.log(`provenance: ${out.provenance.status}${(out.provenance.problems || []).length ? ` (${out.provenance.problems.slice(0, 3).join(" | ")})` : ""}`);
}
if (JSON_OUT) fs.writeFileSync(JSON_OUT, JSON.stringify(out, null, 1));
