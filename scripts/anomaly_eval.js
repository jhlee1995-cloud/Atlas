// anomaly_eval.js -- ANOMALY_H1 evaluator (docs/plans/ANOMALY_H1.md). AH-1..AH-8 (and AH-8d) are read from the batch-3
// atlas.json files (A4b atlases, the margin_v1 rebuilds, the B1 margin_b1 atlases); AX-1, AX-2a, AX-2b, AX-3 and AX-4 from
// results/anomaly_probe_*/probe.json (scripts/anomaly_probe.py). Committed with the ANOMALY_H1 pre-registration (commit
// P_A, a child of the batch-3 commit P) before any batch-3 file exists: this file IS the decision rule. Node only.
//   node scripts/anomaly_eval.js --p <P_A> --p-run <P_run> --a4b results/atlas_v1_resnet56_s1/a4b_eval.json
//        --b1 results/margin_b1_vitb16/verdicts.json [--root results] [--cache <idgauss.json>] [--json <new out.json>]
//        [--no-idgauss]
//   node scripts/anomaly_eval.js --dry-run [--root <committed results>] [--b1 <b1_verdicts.js output on a fixture>
//        --b1-root tests/fixtures/b1/<case>/results] [--cache <idgauss.json>] [--no-idgauss] [--json <new out.json>]
// --dry-run reads committed discovery files only: s1 := resnet56 hub, s2 := e40, rung stand-ins e10/e20, twin :=
// resnet20 s1 vs s1_ref1, M11 := e40, M56new := the committed A3 margin atlases. It prints a calibration block that
// reproduces every discovery number quoted in ANOMALY_H1.md; AH-4 runs on B1's committed fixtures (--b1, --b1-root: the
// real B1 schema) with the E9 s1/s2 points taken from the committed A3 margin atlases; the AX code paths run on
// SYNTHETIC probe records. Every dry-run line is INFO: stand-ins are discovery material and never a verdict.
// Labels: per clause PASS / FAIL / NULL (missing); per item SUPPORTED / REFUTED / NOT_EVALUABLE (INFO when an item has no
// scored clause); per AX gate OPEN / CLOSED / DROPPED / NOT_EVALUABLE. Nothing is promoted by this script. An evaluator
// bug found after the run is fixed only by a committed amendment listed in SESSION.md; the outcomes are then reported
// under both versions (the amended run writes a NEW --json, e.g. results/anomaly_h1/eval_v2.json).
"use strict";
const fs = require("fs"), path = require("path"), crypto = require("crypto"), { execFileSync } = require("child_process");
const argv = process.argv.slice(2);
const arg = (k, d = null) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const DRY = argv.includes("--dry-run"), NO_IDG = argv.includes("--no-idgauss");
const ROOT = arg("--root", "results"), JSON_OUT = arg("--json"), CACHE = arg("--cache");
const P_ARG = arg("--p"), PRUN_ARG = arg("--p-run"), A4B_ARG = arg("--a4b");
const B1_ARG = arg("--b1", DRY ? null : path.join(ROOT, "margin_b1_vitb16", "verdicts.json"));
const B1_ROOT = arg("--b1-root", ROOT);
const REPO = path.resolve(__dirname, "..");
if (JSON_OUT && fs.existsSync(JSON_OUT)) {             // append-only: fail fast, before any work (review item A1)
  console.error(`[anomaly_eval] refused: ${JSON_OUT} exists (append-only; an amendment writes a new file, e.g. eval_v2.json)`);
  process.exit(2);
}
const writeNew = (p, s) => { fs.mkdirSync(path.dirname(path.resolve(p)), { recursive: true }); fs.writeFileSync(p, s); };
const R = (...a) => path.join(ROOT, ...a);
const loadJ = p => { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } };
// every atlas an item reads is recorded (READS), so that its provenance is checked item by item
let READS = null;
const _cache = new Map();
const loadAt = (root, n) => { if (!n) return null; const k = `${root}\u0000${n}`;
  if (!_cache.has(k)) _cache.set(k, loadJ(path.join(root, n, "atlas.json")));
  const a = _cache.get(k); if (a && READS) READS.set(k, { root, name: n }); return a; };
const load = n => loadAt(ROOT, n);

// ---- frozen thresholds (ANOMALY_H1.md; change none without an amendment) -------------------------------------------
const T = {
  AH1: { a0: 0.0224, b: -0.1080, band: 0.035, preMax: 0.070, medMax: 0.05, sfMin: 0.12, dq95: 0.01, twinMed: 0.02, twinSd: 0.015 },
  AH2: { rho: 0.93, hold: 0.25, holdCost: 8.0, bright: 0.30, nr: 0.95, nrCost: 10 },
  AH3: { preLead: 0.15, penLead: 0.02, dAuc: 0.25, dRatio: 1.8, spMpPen: 0.85, spMpPre: 0.60, ncaPen: 0.985, ncaPre: 0.90,
         spMdPen: -0.80, sePen: 1.25, sePre: 1.10, spSep: -0.6, sepLo: 3.5, leadLo: 0.02, sepHi: 4.5, leadHi: 0.02, minBuilds: 4 },
  AH4: { sep: 2.0, lead: 0.03, sp: -0.6, sepHi: 4.5, cuts: [0.7, 0.5], nmin: 300, nFresh: 5 },
  AH5: { stemPc1: 0.35, stemPc23: 0.85, csf: 0.33, csfBT: 0.70, gap: 0.08, btPen: 0.88, enrLo: 0.80, enrHi: 1.20, cost: 6.5,
         penEx: 0.10, stemEx: 0.25 },
  AH6: { cos15: 0.90, inc: 0.85, snowGap: 0.25, pathRatio: 1.25, gauss53: 1.10, shot53: 1.20, shotMed: 0.03, gaussMed: -0.02,
         c4: 0.15 },
  AH7: { hi: 1.90, lo: 1.75, rung: 1.85, spE: 0.8, cv: 0.056, nnCv: 0.035, refFit: 0.995, minRuns: 4 },
  AH8: { pear: 0.95, rhoPre: 0.80, rhoPen: 0.98, penSlack: 0.05, m11: 0.30, twin: 0.03, seeds: [100, 101, 102, 103, 104], n: 5000, discard: 0.1 },
  AX1: { auc: 0.90, aucBright: 0.75, fpr: 0.10, fprFull: 0.50, dAuc: 0.10, parAuc: 0.95, perpAuc: 0.65, hold: 0.90 },
  AX2: { gate: 0.90, conf: 0.85, tie: 0.005, sig: 0.65, impulse: 0.90, zoom: 0.80 },
  AX3: { rho: 0.85, hold: 0.25, loss: 10, bad: 0.05, bright: 0.30, brightFrac: 0.90, nr: 0.95, nrBad: 0.10, ood: 0.90 },
  AX4: { gain: 0.05, drop: 0.02, dens: 0.02 },
};
const DISC = ["gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "snow", "fog", "brightness", "contrast", "pixelate",
              "jpeg_compression"];
const SEV = [1, 3, 5];
const sp = (c, s) => `corrupt__${c}__s${s}`;
const DSPL = DISC.flatMap(c => SEV.map(s => sp(c, s)));
// frozen P_A -> P_run: AH items need the plan and this evaluator; AX items also need the probe and its tests
const FROZEN_AH = ["docs/plans/ANOMALY_H1.md", "scripts/anomaly_eval.js"];
const FROZEN_AX = ["scripts/anomaly_probe.py", "tests/test_anomaly_probe.py"];
const SELFTEST_DIRS = ["instrument_check_anomaly", "instrument_check_anomaly_r2"];

// ---- math ----------------------------------------------------------------------------------------------------------
const num = v => typeof v === "number" && Number.isFinite(v);
const mean = x => x.reduce((a, b) => a + b, 0) / x.length;
const std0 = x => { const m = mean(x); return Math.sqrt(mean(x.map(v => (v - m) ** 2))); };
const rank = a => { const o = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), r = new Array(a.length);
  for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
    for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
const pearson = (x, y) => { if (x.length < 2 || x.length !== y.length || !x.every(num) || !y.every(num)) return null;
  const mx = mean(x), my = mean(y); let sxy = 0, sx = 0, sy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sx += (x[i] - mx) ** 2; sy += (y[i] - my) ** 2; }
  return sx > 0 && sy > 0 ? sxy / Math.sqrt(sx * sy) : null; };
const spearman = (x, y) => (x.every(num) && y.every(num) ? pearson(rank(x), rank(y)) : null);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0), nrm = a => Math.sqrt(dot(a, a));
const cosv = (a, b) => dot(a, b) / (nrm(a) * nrm(b));
const ratio = (a, b) => (num(a) && num(b) && b !== 0 ? a / b : null);
// every threshold goes through these: a missing value is NULL, never a silent PASS or FAIL (review item A2)
const ge = (v, t) => (num(v) && num(t) ? v >= t : null), le = (v, t) => (num(v) && num(t) ? v <= t : null);
const lt = (v, t) => (num(v) && num(t) ? v < t : null), gt = (v, t) => (num(v) && num(t) ? v > t : null);
const all = xs => (xs.some(x => x === false) ? false : xs.some(x => x !== true) ? null : true);
const f = (v, k = 3) => (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v)) ? "-" : typeof v === "number" ? v.toFixed(k) : String(v));
const safe = fn => { try { const v = fn(); return v === undefined ? null : v; } catch (e) { return null; } };
const need = (v, what) => { if (!num(v)) throw new Error(`${what} missing`); return v; };
const short = n => String(n).replace(/^(atlas|margin)_v1_resnet(20|56)_/, "r$2:").replace(/^margin_b1_/, "b1:");
const NE = DRY ? "INFO (dry run) NOT_EVALUABLE" : "NOT_EVALUABLE";

// ---- clauses and verdicts ------------------------------------------------------------------------------------------
// scope: primary (s1/s2, required) | required (twin, graded-across-runs clauses) | secondary-refuting (a built run that fails
// refutes; an unbuilt one is ignored) | secondary (failure is recorded as an exception) | reading (own label) | info
const CL = (id, scope, value, pass, text = "") => ({ id, scope, value: value === undefined ? null : value, pass: pass === true ? true : pass === false ? false : null, text });
function decide(clauses, guard) {
  const req = c => c.scope === "primary" || c.scope === "required";
  const scored = clauses.some(req);
  const failed = clauses.filter(c => (req(c) || c.scope === "secondary-refuting") && c.pass === false).map(c => c.id);
  const missing = clauses.filter(c => req(c) && c.pass === null).map(c => c.id);
  const exceptions = clauses.filter(c => c.scope === "secondary" && c.pass === false).map(c => c.id);
  const readings = clauses.filter(c => c.scope === "reading").map(c => `${c.id}: ${c.text || c.pass}`);
  // no primary or required clause: nothing is scored, so nothing can be SUPPORTED (review item B16)
  let verdict = !scored ? "INFO (no scored clause)" : failed.length ? "REFUTED" : missing.length ? "NOT_EVALUABLE" : "SUPPORTED";
  const why = failed.length ? failed.map(x => `${x} FAIL`) : missing.map(x => `${x} missing`);
  if (guard && guard.length) { verdict = "NOT_EVALUABLE"; why.unshift(...guard.map(g => `guard: ${g}`)); }
  return { verdict: DRY ? `INFO (dry run) ${verdict}` : verdict, why, exceptions, readings };
}
const notEvaluable = (id, why, extra = {}) => ({ id, clauses: [], verdict: NE, why, exceptions: [], readings: [], ...extra });

// ---- atlas accessors -----------------------------------------------------------------------------------------------
const is56 = a => /resnet56/.test((a.meta || {}).arch || a.exp_id || "");
const PRE = a => (is56(a) ? "layer3.5" : "layer3.1");
const S2END = a => (is56(a) ? "layer2.8" : "layer2.2");
const ALIAS = a => (is56(a) ? "layer3.8" : "layer3.2");
const PL = (a, l) => a.per_layer[l];
const pen = a => a.per_layer.penult;
const kd = (a, l = "penult") => PL(a, l).knn_density;
const cdsp = (a, l = "penult") => PL(a, l).corruption_displacement.splits;
const acc = a => a.meta.accuracy;
const cost = (a, s) => 100 * (need(acc(a).test, "accuracy.test") - need(acc(a)[s], `accuracy.${s}`));
const centersDists = C => { const d = []; for (let i = 0; i < C.length; i++) for (let j = i + 1; j < C.length; j++)
  d.push(Math.sqrt(C[i].reduce((s, v, k) => s + (v - C[j][k]) ** 2, 0))); return d; };
const nearestOther = C => C.map((ci, i) => Math.min(...C.map((cj, j) => (i === j ? Infinity : Math.sqrt(ci.reduce((s, v, k) => s + (v - cj[k]) ** 2, 0))))));
function BT(a, l) { const L = PL(a, l), C = L.class_centers.centers, K = C.length, D = C[0].length, mu = new Array(D).fill(0);
  for (const c of C) for (let j = 0; j < D; j++) mu[j] += c[j] / K;
  let B = 0; for (const c of C) { let s = 0; for (let j = 0; j < D; j++) s += (c[j] - mu[j]) ** 2; B += s / K; }
  return need(B / (L.pca_spectrum.eig_top20[0] / L.pca_spectrum.top_eig_frac), `B/T ${l}`); }
function atlasProblems(n) { const a = load(n); if (!a) return null; const p = [];
  for (const [l, v] of Object.entries(a.per_layer || {})) for (const [k, x] of Object.entries(v || {})) if (x && typeof x === "object" && "error" in x) p.push(`${l}/${k} error`);
  for (const [k, x] of Object.entries(a.cross_layer || {})) if (x && typeof x === "object" && "error" in x) p.push(`cross/${k} error`);
  if (a.skipped && Object.keys(a.skipped).length) p.push("skipped non-empty");
  if (a.source !== "real") p.push(`source ${a.source}`);
  return p; }

// ---- runs ----------------------------------------------------------------------------------------------------------
const r56 = s => `atlas_v1_resnet56_${s}`, r20 = s => `atlas_v1_resnet20_${s}`, m56 = s => `margin_v1_resnet56_${s}`;
const RUN = DRY ? {
  s1: r56("s0hub"), s2: r56("e40"), newInst: [r56("e10"), r56("e20")], replays: [], sec: [r56("e10"), r56("e20")],
  twin: [r20("s1"), r20("s1_ref1")], band20: [r20("s0hub_st2"), r20("s1_st2"), r20("s2_st2"), r20("s3"), r20("s4")],
  mc: [r56("e10"), r56("e20")], rungs: [r56("e10"), r56("e20")],
  ladderE: [{ E: 10, n: r56("e10") }, { E: 20, n: r56("e20") }], seedE: 200,
  rungC: [r56("e10"), r56("e20")], cvRungs: [r56("e10"), r56("e20")],
  mSeeds: [m56("s0hub"), m56("s0hub")], mSeedAtlas: [r56("s0hub"), r56("s0hub")],
  m56new: ["margin_v1_resnet20_s0hub", "margin_v1_resnet20_s1", "margin_v1_resnet20_s2", "margin_v1_resnet20_s3",
           "margin_v1_resnet20_s4", m56("s0hub")],
} : {
  s1: r56("s1"), s2: r56("s2"), newInst: ["s12m", "s13m", "e50", "e60", "e70", "e90"].map(r56),
  replays: ["s0hub_st3", "e40_st3", "s0hub_ref1"].map(r56), sec: ["s12m", "s13m", "e50", "e60", "e70"].map(r56),
  twin: [r56("s0hub_st3"), r56("s0hub_ref1")], band20: ["s0hub", "s1", "s2", "s3", "s4"].map(s => r20(`${s}_st3`)),
  mc: ["s12m", "s13m"].map(r56), rungs: ["e50", "e60", "e70", "e90"].map(r56),
  // AH-7(b): the spent e40_st3 anchor is not used (review item B13); s1 and s2 are tied at E = 200
  ladderE: [{ E: 50, n: r56("e50") }, { E: 60, n: r56("e60") }, { E: 70, n: r56("e70") }, { E: 90, n: r56("e90") }], seedE: 200,
  rungC: ["e50", "e60", "e70", "s12m", "s13m"].map(r56), cvRungs: ["e50", "e60", "e70"].map(r56),
  mSeeds: [m56("s1"), m56("s2")], mSeedAtlas: [r56("s1"), r56("s2")],
  m56new: ["s1", "s2", "s12m", "s13m", "e50", "e60", "e70", "e90"].map(m56),
};
const SEEDS = [RUN.s1, RUN.s2];
const atlasOfMargin = n => n.replace(/^margin_/, "atlas_");

// ---- inputs from the other evaluators ------------------------------------------------------------------------------
const A4B = DRY ? null : (A4B_ARG ? loadJ(A4B_ARG) : null);
const B1V = B1_ARG ? loadJ(B1_ARG) : null;

// ---- provenance (scoped: review item A3) ---------------------------------------------------------------------------
const sha256 = buf => crypto.createHash("sha256").update(buf).digest("hex");
const lf = s => Buffer.from(String(s).replace(/\r\n/g, "\n"), "utf8");
function provenance(probeFiles) {
  if (DRY) return { status: "SKIPPED (dry run)", ax_status: "SKIPPED (dry run)", core: [], axCore: [], probes: {} };
  const rep = { p: P_ARG, p_run: PRUN_ARG, frozen_ah: FROZEN_AH, frozen_ax: FROZEN_AX, core: [], axCore: [], probes: {}, selftests: {} };
  const fin = () => ({ ...rep, status: rep.core.length ? "FAIL" : "PASS", ax_status: rep.core.length || rep.axCore.length ? "FAIL" : "PASS" });
  if (!P_ARG || !PRUN_ARG) { rep.core.push("--p <P_A> and --p-run <P_run> are required"); return fin(); }
  const git = (...a) => execFileSync("git", a, { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], maxBuffer: 1 << 26 });
  const gitOk = (...a) => { try { execFileSync("git", a, { cwd: REPO, stdio: "ignore" }); return true; } catch (e) { return false; } };
  let p, pr;
  try { p = git("rev-parse", "--verify", `${P_ARG}^{commit}`).trim(); pr = git("rev-parse", "--verify", `${PRUN_ARG}^{commit}`).trim(); }
  catch (e) { rep.core.push(`cannot resolve --p ${P_ARG} / --p-run ${PRUN_ARG}`); return fin(); }
  Object.assign(rep, { p_full: p, p_run_full: pr, pTime: Date.parse(git("show", "-s", "--format=%cI", p).trim()) });
  if (!gitOk("merge-base", "--is-ancestor", p, pr)) rep.core.push("P_run is neither P_A nor a descendant of P_A");
  const changed = files => (gitOk("diff", "--quiet", p, pr, "--", ...files) ? [] : git("diff", "--name-only", p, pr, "--", ...files).trim().split(/\r?\n/).filter(Boolean));
  const cAH = changed(FROZEN_AH), cAX = changed(FROZEN_AX);
  if (cAH.length) rep.core.push(`changed between P_A and P_run: ${cAH.join(", ")}`);
  if (cAX.length) rep.axCore.push(`changed between P_A and P_run: ${cAX.join(", ")}`);
  let probeSha = null;
  try { probeSha = sha256(lf(git("show", `${p}:scripts/anomaly_probe.py`))); } catch (e) { rep.axCore.push("scripts/anomaly_probe.py not in P_A"); }
  rep.probe_sha256_at_p = probeSha;
  // the probe self-test record(s): every one that exists must be PASS on the code of P_A; at least one must exist
  const st = SELFTEST_DIRS.map(d => [d, loadJ(R(d, "selftest.json"))]).filter(([, j]) => j);
  if (!st.length) rep.axCore.push(`no probe self-test record (${SELFTEST_DIRS.map(d => `${d}/selftest.json`).join(", ")})`);
  for (const [d, j] of st) { const ok = j.status === "PASS" && j.code_sha256 === probeSha;
    rep.selftests[d] = ok ? "PASS" : `${j.status}, code_sha256 ${j.code_sha256 === probeSha ? "= P_A" : "differs from P_A"}`;
    if (!ok) rep.axCore.push(`${d}/selftest.json: ${rep.selftests[d]}`); }
  for (const [file, pj] of Object.entries(probeFiles)) { const bad = [];
    if (!pj.code || pj.code.repo_commit !== pr) bad.push(`code.repo_commit ${pj.code && pj.code.repo_commit}`);
    if (!pj.code || pj.code.sha256 !== probeSha) bad.push("code.sha256 differs from scripts/anomaly_probe.py at P_A");
    if (!(Date.parse(pj.created_utc) > rep.pTime)) bad.push(`created_utc ${pj.created_utc} not after P_A`);
    rep.probes[file] = bad.map(b => `${file}: ${b}`); }
  return fin();
}
const _runProv = new Map();
function runProv(root, n) {                                // one atlas: made by this session's code at P_run, after P_A
  if (DRY || !PROV.p_run_full) return [];
  const k = `${root}\u0000${n}`; if (_runProv.has(k)) return _runProv.get(k);
  const a = loadJ(path.join(root, n, "atlas.json")), m = (a && a.meta) || {}, bad = [];
  const created = typeof m.created === "string" ? Date.parse(m.created.replace(" ", "T") + "Z") : NaN;
  if (!(typeof m.git_commit === "string" && m.git_commit.length >= 7 && PROV.p_run_full.startsWith(m.git_commit))) bad.push(`meta.git_commit ${m.git_commit}`);
  if (!(created > PROV.pTime)) bad.push(`meta.created ${m.created} not after P_A`);
  _runProv.set(k, bad); return bad;
}
function a4bGuard() {
  if (DRY) return [];
  if (!A4B) return ["A4b evaluation missing (--a4b results/atlas_v1_resnet56_s1/a4b_eval.json)"];
  const g = []; for (const x of A4B.guard || []) if (/R0 FAIL|instrument changed/.test(x)) g.push(`A4b ${x}`);
  return g;
}
// the s1/s2 dumps and margin rebuilds come from s1/s2's own weights only if A4b's i2 check passed (review item A4)
function i2Guard() { if (DRY || !A4B) return []; const s = A4B.G0b ? A4B.G0b.i2_status : null; return s === "PASS" ? [] : [`A4b G0b i2 ${s || "MISSING"}`]; }
function runGuard(names) { const g = [];
  for (const n of [...new Set(names)]) { const p = atlasProblems(n); if (p && p.length) g.push(`${short(n)}: ${p.join(", ")}`); } return g; }

// ---- AH-1: collapse level sets the train-reference density baseline --------------------------------------------------
const resid1 = a => need(pen(a).knn_density.splits.test.sparse_frac, "sparse_frac") - (T.AH1.a0 + T.AH1.b * Math.log10(need(pen(a).neural_collapse.nc1, "nc1")));
const maxPre = a => Math.max(...a.layers.filter(l => l !== "penult" && l !== ALIAS(a)).map(l => need(kd(a, l).splits.test.sparse_frac, `${l} sparse_frac`)));
function twinDelta(na, nb) { const a = load(na), b = load(nb); if (!a || !b) return null;
  const A = kd(a), B = kd(b), sps = ["test", ...DSPL];
  const dsf = sps.map(s => need(B.splits[s].sparse_frac - A.splits[s].sparse_frac, `${s} sparse_frac`));
  const dmed = sps.map(s => need(B.splits[s].median_log_radius_shift - A.splits[s].median_log_radius_shift, `${s} median shift`));
  return { dq95: B.ref_log_radius_quantiles.q95 - A.ref_log_radius_quantiles.q95, dq50: B.ref_log_radius_quantiles.q50 - A.ref_log_radius_quantiles.q50,
    dsf_mean: mean(dsf), dsf_sd: std0(dsf), dmed_mean: mean(dmed), dmed_sd: std0(dmed), n: sps.length }; }
function AH1() {
  const cl = [];
  const per = (id, runs, scope, fn, test) => { for (const n of runs) { const a = load(n); if (!a && scope !== "primary") continue;
    const v = a ? safe(() => fn(a)) : null; cl.push(CL(`${id}[${short(n)}]`, scope, v, num(v) ? test(v) : null)); } };
  per("a", SEEDS, "primary", resid1, v => Math.abs(v) <= T.AH1.band);
  per("a", RUN.newInst, "secondary-refuting", resid1, v => Math.abs(v) <= T.AH1.band);
  per("a", RUN.replays, "info", resid1, v => Math.abs(v) <= T.AH1.band);
  per("b", SEEDS, "primary", maxPre, v => v <= T.AH1.preMax);
  per("b", RUN.newInst, "secondary-refuting", maxPre, v => v <= T.AH1.preMax);
  per("b", RUN.replays, "info", maxPre, v => v <= T.AH1.preMax);
  for (const n of SEEDS) { const a = load(n), t = a ? safe(() => pen(a).knn_density.splits.test) : null;
    cl.push(CL(`c[${short(n)}]`, "primary", t ? { median_shift: t.median_log_radius_shift, sparse_frac: t.sparse_frac } : null,
      t ? all([le(t.median_log_radius_shift, T.AH1.medMax), ge(t.sparse_frac, T.AH1.sfMin)]) : null)); }
  const tw = safe(() => twinDelta(RUN.twin[0], RUN.twin[1]));
  let dpass = null;
  if (tw && [tw.dq95, tw.dsf_mean, tw.dmed_mean, tw.dmed_sd].every(num)) {
    const signOk = Math.abs(tw.dq95) < T.AH1.dq95 ? true : Math.sign(tw.dsf_mean) === -Math.sign(tw.dq95);
    dpass = all([signOk, le(Math.abs(tw.dmed_mean), T.AH1.twinMed), le(tw.dmed_sd, T.AH1.twinSd)]); }
  cl.push(CL(`d[twin ${RUN.twin.map(short).join(" vs ")}]`, "required", tw, dpass));
  return { id: "AH-1", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]),
    info: { nc1: Object.fromEntries([...SEEDS, ...RUN.newInst, ...RUN.replays].filter(n => load(n)).map(n => [short(n), safe(() => pen(load(n)).neural_collapse.nc1)])) } };
}

// ---- AH-2: normalized label-free harm grade H ------------------------------------------------------------------------
function harm(a) { const K = kd(a), q = K.ref_log_radius_quantiles, sc = need(q.q95 - q.q50, "q95 - q50"), t = need(K.splits.test.median_log_radius_shift, "test median shift");
  const H = DSPL.map(s => need((K.splits[s].median_log_radius_shift - t) / sc, `H ${s}`)), Cc = DSPL.map(s => cost(a, s));
  const inBand = H.map((h, i) => [h, Cc[i]]).filter(([h]) => h <= T.AH2.hold);
  const bright = SEV.map(s => H[DSPL.indexOf(sp("brightness", s))]);
  const nr = DSPL.filter((s, i) => need(cdsp(a)[s].norm_ratio, `norm_ratio ${s}`) >= T.AH2.nr && Cc[i] > T.AH2.nrCost);
  return { rho: spearman(H, Cc), maxHoldCost: inBand.length ? Math.max(...inBand.map(x => x[1])) : null, nHold: inBand.length,
    bright, energyFail: nr, scale: sc }; }
function AH2() {
  const cl = [], hs = {};
  for (const [runs, scope] of [[SEEDS, "primary"], [RUN.sec, "secondary"]]) for (const n of runs) {
    const a = load(n); if (!a && scope !== "primary") continue; const h = a ? safe(() => harm(a)) : null; hs[short(n)] = h;
    cl.push(CL(`a[${short(n)}]`, scope, h && h.rho, h ? ge(h.rho, T.AH2.rho) : null));
    // an empty HOLD band tests nothing: NULL, not a vacuous PASS (the review's item B14 rule, applied here too)
    cl.push(CL(`b[${short(n)}]`, scope, h && { max_cost: h.maxHoldCost, n: h.nHold }, h ? (h.nHold ? le(h.maxHoldCost, T.AH2.holdCost) : null) : null,
      h && !h.nHold ? "no split with H <= 0.25" : ""));
    cl.push(CL(`c[${short(n)}]`, scope, h && h.bright, h ? all(h.bright.map(v => le(v, T.AH2.bright))) : null)); }
  const e = SEEDS.map(n => (hs[short(n)] ? hs[short(n)].energyFail : null));
  const eRead = e.some(x => x === null) ? "NOT_EVALUABLE" : e.some(x => x.length) ? "energy HOLD rule fails again (as predicted)"
    : "energy is not refuted at depth 56 (no split with norm_ratio >= 0.95 and cost > 10 pt in s1 or s2)";
  cl.push(CL("d", "reading", e, e.every(x => x !== null) ? e.some(x => x.length > 0) : null, eRead));
  return { id: "AH-2", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]) };
}

// ---- AH-3: after the final collapse, margin equals distance ---------------------------------------------------------
const mt = (n, l) => { const a = load(n); return a && a.per_layer && a.per_layer[l] ? a.per_layer[l].margin_typeb : null; };
function AH3() {
  const cl = [], tie = A4B ? safe(() => A4B.M56.c_confidence.tie_call) : null;
  RUN.mSeeds.forEach((mn, i) => { const an = RUN.mSeedAtlas[i], a = load(an), tag = `${short(mn)}${DRY && i ? " (again)" : ""}`;
    const P = mt(mn, "penult"), Q = mt(mn, "layer3.5"), ok = !!(P && Q && a);
    cl.push(CL(`a[${tag}]`, "primary", ok ? [Q.margin_minus_dist_wrong, P.margin_minus_dist_wrong] : null,
      ok ? all([ge(Q.margin_minus_dist_wrong, T.AH3.preLead), le(P.margin_minus_dist_wrong, T.AH3.penLead)]) : null));
    const dD = ok && num(P.auc_dist_wrong) && num(Q.auc_dist_wrong) ? P.auc_dist_wrong - Q.auc_dist_wrong : null;
    const dM = ok && num(P.auc_margin_wrong) && num(Q.auc_margin_wrong) ? P.auc_margin_wrong - Q.auc_margin_wrong : null;
    cl.push(CL(`b[${tag}]`, "primary", ok ? [dD, dM] : null, ok ? all([ge(dD, T.AH3.dAuc), num(dD) && num(dM) ? dD >= T.AH3.dRatio * dM : null]) : null));
    const nca = l => (a ? safe(() => PL(a, l).class_centers.nearest_center_agrees_with_model) : null);
    cl.push(CL(`c[${tag}]`, "primary", ok ? { sp_mp: [P.spearman_margin_maxprob, Q.spearman_margin_maxprob], nca: [nca("penult"), nca("layer3.5")], sp_md_pen: P.spearman_margin_dist } : null,
      ok ? all([ge(P.spearman_margin_maxprob, T.AH3.spMpPen), le(Q.spearman_margin_maxprob, T.AH3.spMpPre), ge(nca("penult"), T.AH3.ncaPen),
        le(nca("layer3.5"), T.AH3.ncaPre), le(P.spearman_margin_dist, T.AH3.spMdPen)]) : null));
    const rp = ok ? ratio(P.se_maxprob_typeb, P.se_margin_typeb) : null, rq = ok ? ratio(Q.se_maxprob_typeb, Q.se_margin_typeb) : null;
    const tieNote = tie && tie[i] === "TIE-ARTIFACT-POSSIBLE" ? "read as float32 saturation (A4b M56-c TIE-ARTIFACT-POSSIBLE)" : "";
    cl.push(CL(`d[${tag}]`, "primary", ok ? [rp, rq] : null, ok ? all([ge(rp, T.AH3.sePen), le(rq, T.AH3.sePre)]) : null, tieNote)); });
  const pts = RUN.m56new.map(n => safe(() => { const P = mt(n, "penult"), a = load(atlasOfMargin(n)); if (!P || !a) return null;
    const sep = pen(a).class_centers.sep_ratio, lead = P.margin_minus_dist_typeb;
    return num(sep) && num(lead) ? { n: short(n), sep, lead } : null; })).filter(Boolean);
  const enough = pts.length >= T.AH3.minBuilds;
  const rho = enough ? spearman(pts.map(p => p.sep), pts.map(p => p.lead)) : null;
  const lo = pts.filter(p => p.sep <= T.AH3.sepLo), hi = pts.filter(p => p.sep >= T.AH3.sepHi);
  const few = `only ${pts.length} builds (>= ${T.AH3.minBuilds} needed)`;
  cl.push(CL("e.spearman", "required", { rho, n: pts.length }, rho === null ? null : rho <= T.AH3.spSep, enough ? "" : few));
  // an empty subset tests nothing: NULL "no build in range" (review item B14)
  cl.push(CL("e.low-sep", "required", lo.map(p => [p.n, p.sep, p.lead]), !enough || !lo.length ? null : lo.every(p => p.lead >= T.AH3.leadLo),
    !enough ? few : !lo.length ? `no build in range (sep <= ${T.AH3.sepLo})` : ""));
  cl.push(CL("e.high-sep", "required", hi.map(p => [p.n, p.sep, p.lead]), !enough || !hi.length ? null : hi.every(p => Math.abs(p.lead) <= T.AH3.leadHi),
    !enough ? few : !hi.length ? `no build in range (sep >= ${T.AH3.sepHi})` : ""));
  return { id: "AH-3", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard([...RUN.mSeedAtlas, ...RUN.mSeeds])]), info: { m56new: pts, tie_call: tie } };
}

// ---- AH-4: the collapse regime decides margin vs distance (B1 keys only, after B1's own verdicts) ---------------------
// review item A5: only b1_verdicts.js output is read (no recomputed c*); B1's info.runs_used decides which atlas is a
// primary (a run B1 dropped is NOT_EVALUABLE); an INVALID-PLUMBING outcome voids the item; V1 is read with its own label
function AH4(src) {
  const V = src.verdicts, B = src.load;
  if (!V) return notEvaluable("AH-4", [`B1 verdicts unreadable (${B1_ARG || "no --b1"}): AH-4 is read only after scripts/b1_verdicts.js`]);
  const outcome = (V.labels && V.labels.outcome) || "";
  if (/^INVALID-PLUMBING/.test(outcome)) return notEvaluable("AH-4", [`B1 outcome ${outcome}`]);
  const gateOpen = V.gate && typeof V.gate.open === "boolean" ? V.gate.open : null;
  const used = (V.info && V.info.runs_used) || {};
  const pmOf = n => { const a = n ? B(n) : null; return a && a.per_layer && a.per_layer.penult ? a.per_layer.penult.margin_typeb : null; };
  const row = (pm, c) => (pm && pm.sweep || []).find(r => Math.abs(r.cut - c) < 1e-9) || null;
  const IMN = ["resnet50", "vitb16", "deitb"], VIT = ["vitb16", "deitb"];
  const cstar = num(V.cstar) ? V.cstar : null;
  let rn50cut = cstar;                                     // gate closed or no c*: resnet50 at its own first powered cut
  if (rn50cut === null) for (const c of T.AH4.cuts) if (rn50cut === null && ge((row(pmOf(used.resnet50), c) || {}).n_typeb, T.AH4.nmin)) rn50cut = c;
  const cl = [], pts = [];
  for (const m of IMN) {
    const open = VIT.includes(m) ? gateOpen === true : true, run = used[m] || null, pm = open && run ? pmOf(run) : null;
    const cut = m === "resnet50" ? rn50cut : cstar;
    const sep = pm ? safe(() => pm.centers_geometry.sep_ratio_ref) : null, r = pm && cut !== null ? row(pm, cut) : null;
    const lead = r && num(r.margin_minus_dist_typeb) ? r.margin_minus_dist_typeb : null;
    const why = !open ? `gate G not open (${gateOpen})` : !run ? "not in B1 info.runs_used (dropped by B1)" : !pm ? "no penult margin_typeb" : cut === null ? "no powered cut (B1 c* null)" : "";
    cl.push(CL(`a[${m}]`, "primary", sep, open && run ? le(sep, T.AH4.sep) : null, [why, "premise check (1000 classes, 25 per class): not evidence"].filter(Boolean).join("; ")));
    cl.push(CL(`b[${m} @ ${cut}]`, "primary", lead, open && run && cut !== null ? ge(lead, T.AH4.lead) : null, why));
    if (num(sep) && num(lead)) pts.push({ m, run, sep, lead });
    const sw = used[`${m}_swap`], q = open && sw ? pmOf(sw) : null, rq = q && cut !== null ? row(q, cut) : null;
    if (q) cl.push(CL(`replicate[${m}_swap]`, "info", { sep: safe(() => q.centers_geometry.sep_ratio_ref), lead: rq ? rq.margin_minus_dist_typeb ?? null : null }, null));
  }
  // the fresh CIFAR primaries: E9 margin_b1 rebuilds of s1 and s2 (their dumps come from their own weights only if A4b's
  // guards and i2 pass); the two hub replays are discovery replays and INFO only (review item B12)
  const e9g = [...a4bGuard(), ...i2Guard()];
  for (const x of ["resnet56_s1", "resnet56_s2"]) { const n = `margin_b1_${x}`, pm = e9g.length ? null : pmOf(n);
    const sep = pm ? safe(() => pm.centers_geometry.sep_ratio_ref) : null, lead = pm && num(pm.margin_minus_dist_typeb) ? pm.margin_minus_dist_typeb : null;
    pts.push({ m: x, run: n, sep, lead, why: e9g.length ? `guard: ${e9g.join(", ")}` : !pm ? "no penult margin_typeb" : "" }); }
  const hubs = ["resnet20_s0hub_st3", "resnet56_s0hub_st3"].map(x => { const pm = pmOf(`margin_b1_${x}`);
    return { m: x, sep: pm ? safe(() => pm.centers_geometry.sep_ratio_ref) : null, lead: pm ? pm.margin_minus_dist_typeb ?? null : null }; });
  const full = pts.filter(p => num(p.sep) && num(p.lead));
  const rho = full.length === T.AH4.nFresh ? spearman(full.map(p => p.sep), full.map(p => p.lead)) : null;
  const miss = ["resnet50", "vitb16", "deitb", "resnet56_s1", "resnet56_s2"].filter(m => !full.some(p => p.m === m));
  cl.push(CL("c", "required", { rho, n: full.length, points: pts }, rho === null ? null : rho <= T.AH4.sp,
    full.length === T.AH4.nFresh ? "rule 8: sep_ratio_ref uses 1000 reference rows per class on CIFAR and 25 on ImageNet" : `only ${full.length} of 5 fresh primaries (missing ${miss.join(", ")})`));
  if (full.length >= 3 && full.length < T.AH4.nFresh) cl.push(CL("c-partial", "info", spearman(full.map(p => p.sep), full.map(p => p.lead)), null, `${full.length} fresh primaries`));
  cl.push(CL("c-hubs", "info", hubs, null, "discovery replays (r20 hub sep 3.0 +0.07; r56 hub 5.33 -0.001): never in (c)"));
  for (const m of VIT) { const run = used[m] || null, pm = gateOpen && run ? pmOf(run) : null, sep = pm ? safe(() => pm.centers_geometry.sep_ratio_ref) : null;
    const mv = V.models && V.models[m] ? V.models[m] : null, v1 = mv ? mv.V1 : null, own = mv ? (mv.V1_own ?? mv.V1) : null;
    const read = gateOpen !== true ? `NOT_EVALUABLE (gate G not open: ${gateOpen})` : !run ? "NOT_EVALUABLE (run dropped by B1)"
      : !v1 || v1 === "NOT_EVALUABLE" ? "NOT_EVALUABLE (no B1 V1 label)"
      : /^SPLIT-/.test(v1) || v1 === "UNRESOLVED" ? `UNDECIDED-SPLIT (V1 ${v1}${own !== v1 ? `, own label ${own}` : ""})`
      : v1 === "PASS" ? "n/a (V1 PASS)"
      : /^NULL/.test(v1) ? (le(sep, T.AH4.sep) ? "evidence against margin being representation-general (no regime excuse)"
        : ge(sep, T.AH4.sepHi) ? "REGIME-CONFOUNDED (no statement about architecture)" : num(sep) ? "UNDECIDED-REGIME" : "NOT_EVALUABLE (no sep_ratio_ref)")
      : `NOT_EVALUABLE (unknown V1 label ${v1})`;
    cl.push(CL(`d[${m}]`, "reading", { V1: v1, V1_own: own, sep }, null, read)); }
  return { id: "AH-4", clauses: cl, ...decide(cl, []), info: { b1_outcome: outcome, gate_open: gateOpen, cstar, resnet50_cut: rn50cut, runs_used: used, points: pts } };
}

// ---- AH-5: brightness takes a learned class-neutral path and is absorbed ---------------------------------------------
function bright(a) { const pre = PRE(a), d = s => cdsp(a, "stem")[sp("brightness", s)].direction_pca20;
  const stem = [1, 5].map(s => ({ pc1: Math.abs(need(d(s)[0], "dir[0]")), pc23: Math.hypot(need(d(s)[1], "dir[1]"), need(d(s)[2], "dir[2]")) }));
  const btp = BT(a, pre), btq = BT(a, "penult"), c = cdsp(a, pre);
  const csf = [1, 3].map(s => need(c[sp("brightness", s)].class_sub_frac, "class_sub_frac"));
  const s1 = DISC.map(x => [x, need(c[sp(x, 1)].class_sub_frac, `class_sub_frac ${x}`)]).sort((x, y) => x[1] - y[1]);
  const enr = DSPL.map(s => need(cdsp(a)[s].class_sub_frac / btq, `enrichment ${s}`));
  const ex = l => need(kd(a, l).splits[sp("brightness", 5)].sparse_frac - kd(a, l).splits.test.sparse_frac, `${l} excess`);
  return { stem, btp, csf, lowest: s1[0][0], gap: s1[1][1] - s1[0][1], btq, enrMin: Math.min(...enr), enrMax: Math.max(...enr),
    cost5: cost(a, sp("brightness", 5)), penEx: ex("penult"), stemEx: ex("stem") }; }
function AH5() {
  const cl = [];
  const aPass = b => all(b.stem.flatMap(x => [le(x.pc1, T.AH5.stemPc1), ge(x.pc23, T.AH5.stemPc23)]));
  for (const n of [...SEEDS, ...RUN.sec]) { const a = load(n), prim = SEEDS.includes(n); if (!a && !prim) continue;
    const b = a ? safe(() => bright(a)) : null, t = short(n);
    cl.push(CL(`a[${t}]`, prim ? "primary" : "secondary-refuting", b && b.stem, b ? aPass(b) : null));
    if (!prim) continue;
    cl.push(CL(`b[${t}]`, "primary", b && { csf: b.csf, bt_pre: b.btp, lowest: b.lowest, gap: b.gap },
      b ? all([...b.csf.flatMap(v => [le(v, T.AH5.csf), le(v, T.AH5.csfBT * b.btp)]), b.lowest === "brightness", ge(b.gap, T.AH5.gap)]) : null,
      "depth-56 specific: fails in the spent r20 s1, s2, s4 and in e10 (discovery)"));
    cl.push(CL(`c[${t}]`, "primary", b && { bt_pen: b.btq, enrichment: [b.enrMin, b.enrMax] },
      b ? all([ge(b.btq, T.AH5.btPen), ge(b.enrMin, T.AH5.enrLo), le(b.enrMax, T.AH5.enrHi)]) : null));
    cl.push(CL(`d[${t}]`, "primary", b && { cost_s5: b.cost5, penult_excess: b.penEx }, b ? all([le(b.cost5, T.AH5.cost), le(b.penEx, T.AH5.penEx)]) : null));
    // review item B9: the random nulls pass the stem part (0.32-0.36), so it is INFO, never a clause
    cl.push(CL(`d.stem[${t}]`, "info", b && b.stemEx, b ? ge(b.stemEx, T.AH5.stemEx) : null, "input statistics: random nulls give 0.32-0.36")); }
  return { id: "AH-5", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]) };
}

// ---- AH-6: motion is directional, noise keeps moving, C4 tracks collapse ---------------------------------------------
function c4min(a) { const coh = (c, s) => need(cdsp(a)[sp(c, s)].coherence, `coherence ${c} s${s}`), v = [];
  for (const [c, s] of [["defocus_blur", 5], ["motion_blur", 3], ["motion_blur", 5]]) for (const x of ["gaussian_noise", "shot_noise"]) v.push(coh(x, s) - coh(c, s));
  return Math.min(...v); }
function motion(a) { const out = {};
  for (const l of ["penult", PRE(a)]) { const S = cdsp(a, l), K = kd(a, l).splits;
    const v = (c, s) => S[sp(c, s)].direction_pca20.map(x => x * S[sp(c, s)].magnitude);
    const cos15 = c => cosv(S[sp(c, 1)].direction_pca20, S[sp(c, 5)].direction_pca20);
    const inc = c => { const i1 = v(c, 3).map((x, i) => x - v(c, 1)[i]), i2 = v(c, 5).map((x, i) => x - v(c, 3)[i]); return cosv(i1, i2); };
    out[l === "penult" ? "pen" : "pre"] = { cos15m: cos15("motion_blur"), incm: inc("motion_blur"), cos15s: cos15("snow"),
      g53: ratio(S[sp("gaussian_noise", 5)].magnitude, S[sp("gaussian_noise", 3)].magnitude), s53: ratio(S[sp("shot_noise", 5)].magnitude, S[sp("shot_noise", 3)].magnitude),
      gmed: K[sp("gaussian_noise", 5)].median_log_radius_shift - K[sp("gaussian_noise", 3)].median_log_radius_shift,
      smed: K[sp("shot_noise", 5)].median_log_radius_shift - K[sp("shot_noise", 3)].median_log_radius_shift,
      path: ratio(S[sp("motion_blur", 5)].magnitude, S[sp("gaussian_noise", 5)].magnitude) }; }
  out.pathRatio = ratio(out.pre.path, out.pen.path); out.c4 = c4min(a); return out; }
function AH6() {
  const cl = [];
  for (const n of SEEDS) { const a = load(n), m = a ? safe(() => motion(a)) : null, t = short(n);
    cl.push(CL(`a[${t}]`, "primary", m && { cos15_pen: m.pen.cos15m, cos15_pre: m.pre.cos15m, inc_pen: m.pen.incm, snow_cos15_pen: m.pen.cos15s },
      m ? all([ge(m.pen.cos15m, T.AH6.cos15), ge(m.pre.cos15m, T.AH6.cos15), ge(m.pen.incm, T.AH6.inc), ge(m.pen.cos15m - m.pen.cos15s, T.AH6.snowGap)]) : null));
    cl.push(CL(`b[${t}]`, "primary", m && m.pathRatio, m ? ge(m.pathRatio, T.AH6.pathRatio) : null));
    cl.push(CL(`c[${t}]`, "primary", m && { g53: m.pen.g53, s53: m.pen.s53, shot_dmed: m.pen.smed, gauss_dmed: m.pen.gmed },
      m ? all([ge(m.pen.g53, T.AH6.gauss53), ge(m.pen.s53, T.AH6.shot53), ge(m.pen.smed, T.AH6.shotMed), ge(m.pen.gmed, T.AH6.gaussMed)]) : null,
      "rule 8: the random nulls pass this clause more strongly; it tests 'no saturation at depth 56', not learned structure"));
    cl.push(CL(`d.c4[${t}]`, "primary", m && m.c4, m ? ge(m.c4, T.AH6.c4) : null)); }
  // review item B11 (the verifier's H-T1 correction): collapse ordering on independent seeds only, s1/s2 (200 epochs)
  // against s12m/s13m (matched length); the seed-11 rungs are one training run and INFO only
  const c4of = n => { const a = load(n); return a ? { n, built: true, v: safe(() => ({ c4: c4min(a), sep: need(pen(a).class_centers.sep_ratio, "sep_ratio") })) } : null; };
  const S = SEEDS.map(c4of), M = RUN.mc.map(c4of).filter(Boolean);
  if (!M.length) cl.push(CL("d.order", "secondary-refuting", null, null, "neither s12m nor s13m built: ignored"));
  else { const ok = S.every(x => x && x.v) && M.every(x => x.v);
    const premise = ok ? Math.min(...S.map(x => x.v.sep)) > Math.max(...M.map(x => x.v.sep)) : null;
    const val = ok ? { seeds: S.map(x => [short(x.n), x.v.sep, x.v.c4]), matched: M.map(x => [short(x.n), x.v.sep, x.v.c4]) } : null;
    cl.push(CL("d.order", "secondary-refuting", val, ok && premise ? Math.min(...S.map(x => x.v.c4)) > Math.max(...M.map(x => x.v.c4)) : null,
      !ok ? "a value is missing" : !premise ? "premise failed: the 200-epoch seeds are not more separated than s12m/s13m (NULL)" : "min C4(s1, s2) > max C4(s12m, s13m)")); }
  const sc = xs => { const p = xs.filter(x => x && x.v); return p.length >= 3 ? { rho: spearman(p.map(x => x.v.sep), p.map(x => x.v.c4)), n: p.length } : { rho: null, n: p.length }; };
  cl.push(CL("d.spearman-seeds", "info", sc([...S, ...M]), null, "Spearman(sep, C4) over s1, s2, s12m, s13m (built)"));
  cl.push(CL("d.spearman-rungs", "info", sc(RUN.rungs.map(c4of)), null, "seed-11 rungs: one training run"));
  return { id: "AH-6", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]) };
}

// ---- AH-7: how deep the collapse goes depends on training length ------------------------------------------------------
const comp = a => Math.log10(need(PL(a, S2END(a)).neural_collapse.nc1, "nc1 s2end") / need(pen(a).neural_collapse.nc1, "nc1 penult"));
const cvAll = a => { const d = centersDists(pen(a).class_centers.centers); return need(std0(d) / mean(d), "cv_all"); };
const cvNN = a => { const d = nearestOther(pen(a).class_centers.centers); return need(std0(d) / mean(d), "cv_nn"); };
function AH7() {
  const cl = [], c = SEEDS.map(n => (load(n) ? safe(() => comp(load(n))) : null));
  const read = c.some(v => v === null) ? null : c.every(v => v >= T.AH7.hi) ? "PREDICTED (depth x long training)"
    : c.every(v => v <= T.AH7.lo) ? "HUB-SPECIFIC" : "UNDECIDED";
  cl.push(CL("a", "primary", c, read === null ? null : read.startsWith("PREDICTED") ? true : read === "HUB-SPECIFIC" ? false : null, read || "missing"));
  for (const n of RUN.rungC) { const a = load(n); if (!a) continue; const v = safe(() => comp(a));
    cl.push(CL(`b.rung[${short(n)}]`, "required", v, le(v, T.AH7.rung))); }
  const ptsE = [...RUN.ladderE.map(r => ({ n: r.n, E: r.E })), ...SEEDS.map(n => ({ n, E: RUN.seedE }))].map(p => ({ ...p, c: load(p.n) ? safe(() => comp(load(p.n))) : null })).filter(p => num(p.c));
  const rhoE = ptsE.length >= T.AH7.minRuns ? spearman(ptsE.map(p => p.E), ptsE.map(p => p.c)) : null;
  cl.push(CL("b.spearman(E, c)", "required", { rho: rhoE, n: ptsE.length, pts: ptsE.map(p => [short(p.n), p.E, p.c]) }, rhoE === null ? null : rhoE >= T.AH7.spE,
    `rungs e50, e60, e70 (e90 if built) and s1, s2 at E = 200 (tied); >= ${T.AH7.minRuns} built; e40_st3 not used`));
  for (const n of SEEDS) { const a = load(n), v = a ? safe(() => ({ cv_all: cvAll(a), cv_nn: cvNN(a) })) : null;
    cl.push(CL(`c[${short(n)}]`, "primary", v, v ? all([lt(v.cv_all, T.AH7.cv), le(v.cv_nn, T.AH7.nnCv)]) : null)); }
  for (const n of RUN.cvRungs) { const a = load(n); if (!a) continue; const v = safe(() => cvAll(a)); cl.push(CL(`c.rung[${short(n)}]`, "required", v, ge(v, T.AH7.cv))); }
  const fit = Object.fromEntries([...RUN.ladderE.map(r => r.n), ...RUN.rungC, ...SEEDS].filter(n => load(n)).map(n => { const v = safe(() => acc(load(n)).ref); return [short(n), { acc_ref: v, misfit_caveat: num(v) ? v < T.AH7.refFit : null }]; }));
  const bcv = safe(() => { const band = RUN.band20.map(load); if (!band.every(Boolean)) return null; const v = band.map(cvAll), mn = Math.min(...v), mx = Math.max(...v); return { min: mn, max: mx, lo: mn - (mx - mn) }; });
  const r = decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]);
  if (read === "UNDECIDED" && r.verdict.endsWith("NOT_EVALUABLE")) r.why.unshift("a UNDECIDED (1.75 < c < 1.90 in a seed, or seeds on opposite sides)");
  return { id: "AH-7", clauses: cl, ...r, info: { ref_fit: fit, live_band_cv_edge: bcv } };
}

// ---- AH-8: TwoNN ID follows the spectrum except at a fully trained penult ---------------------------------------------
// port of scratchpad/anom/D/{twonn.js, rho_mc.js}: TwoNN as atlas/invariants/dimension.py (mu = r2/r1, F = i/n, keep 90%,
// least squares through the origin) on N(0, diag(lambda)), lambda = eig_top20 + a geometric tail summing to the rest.
function mulberry(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gaussSampler(rng) { let spare = null; return () => { if (spare !== null) { const s = spare; spare = null; return s; }
  let u, v, s; do { u = rng() * 2 - 1; v = rng() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  const m = Math.sqrt(-2 * Math.log(s) / s); spare = v * m; return u * m; }; }
function sampleGauss(n, sds, seed) { const D = sds.length, g = gaussSampler(mulberry(seed)), X = new Float64Array(n * D);
  for (let i = 0; i < n; i++) for (let t = 0; t < D; t++) X[i * D + t] = g() * sds[t]; return X; }
function twonn(X, n, D, discard) { const r1 = new Float64Array(n).fill(Infinity), r2 = new Float64Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) { const oi = i * D; for (let j = i + 1; j < n; j++) { const oj = j * D; let s = 0;
    for (let t = 0; t < D; t++) { const d = X[oi + t] - X[oj + t]; s += d * d; }
    if (s < r2[i]) { if (s < r1[i]) { r2[i] = r1[i]; r1[i] = s; } else r2[i] = s; }
    if (s < r2[j]) { if (s < r1[j]) { r2[j] = r1[j]; r1[j] = s; } else r2[j] = s; } } }
  const mu = []; for (let i = 0; i < n; i++) if (r1[i] > 0) mu.push(Math.sqrt(r2[i] / r1[i])); mu.sort((a, b) => a - b);
  const m = mu.length, keep = Math.floor(m * (1 - discard)); let sxy = 0, sxx = 0;
  for (let i = 0; i < keep; i++) { const F = (i + 1) / m, x = Math.log(mu[i]), y = -Math.log(1 - F); sxy += x * y; sxx += x * x; }
  return sxy / (sxx + 1e-12); }
function spectrum(p) { const w = p.eig_top20.slice(), D = p.dim, tot = w[0] / p.top_eig_frac, Tl = tot - w.reduce((s, v) => s + v, 0), full = w.slice();
  if (D > w.length) { const m = D - w.length, last = w[w.length - 1]; let lo = 1e-6, hi = 1.0;
    const S = q => { let s = 0, v = last; for (let k = 0; k < m; k++) { v *= q; s += v; } return s; };
    if (S(hi) < Tl) { for (let k = 0; k < m; k++) full.push(Tl / m); }
    else { for (let it = 0; it < 100; it++) { const mid = (lo + hi) / 2; if (S(mid) < Tl) lo = mid; else hi = mid; }
      let v = last; for (let k = 0; k < m; k++) { v *= lo; full.push(v); } } }
  return full; }
const IDG0 = CACHE ? (loadJ(CACHE) || {}) : {}, IDG = { ...IDG0 };
function idGauss(p) { if (!p || !Array.isArray(p.eig_top20) || !num(p.dim) || !num(p.top_eig_frac)) throw new Error("pca_spectrum missing");
  const key = `${p.dim}|${p.top_eig_frac}|${p.eig_top20.join(",")}`;
  if (!(key in IDG)) { const sd = spectrum(p).map(Math.sqrt);
    IDG[key] = mean(T.AH8.seeds.map(s => twonn(sampleGauss(T.AH8.n, sd, s), T.AH8.n, p.dim, T.AH8.discard))); }
  return IDG[key]; }
const taps10 = a => a.layers.filter(l => l !== ALIAS(a));
const rhoAt = (a, l) => { const L = PL(a, l); const g = idGauss(L.pca_spectrum); return { id: need(L.twonn_id.id, `twonn ${l}`), gauss: g, rho: L.twonn_id.id / g }; };
const r20maxPen = () => { const g = RUN.band20.map(n => { const a = load(n); return a ? safe(() => rhoAt(a, "penult").gauss) : null; }); return g.every(num) ? Math.max(...g) : null; };
function m11Runs() {
  if (DRY) return [{ name: "M11", atlas: r56("e40") }];
  if (!A4B || !A4B.ladder) return null;
  const out = []; const L = A4B.ladder;
  if (L.status === "matched" && num(L.M11)) out.push({ name: "M11", atlas: L.M11 === 40 ? r56("e40_st3") : r56(`e${L.M11}`) });
  for (const m of L.Mc || []) if (m.evaluable) out.push({ name: m.name, atlas: r56(`s${m.name.slice(1)}m`) });
  return out; }
function AH8() {
  if (NO_IDG) return notEvaluable("AH-8", ["--no-idgauss"]);
  const cl = [];
  for (const n of [...SEEDS, ...RUN.newInst, ...RUN.replays]) { const a = load(n), prim = SEEDS.includes(n); if (!a && !prim) continue;
    const v = a ? safe(() => { const L = taps10(a), r = L.map(l => rhoAt(a, l)); return { pearson: pearson(r.map(x => x.id), r.map(x => x.gauss)), taps: L.length }; }) : null;
    cl.push(CL(`a[${short(n)}]`, prim ? "primary" : RUN.replays.includes(n) ? "info" : "secondary-refuting", v, v ? ge(v.pearson, T.AH8.pear) : null)); }
  const r20max = r20maxPen();
  for (const n of SEEDS) { const a = load(n), t = short(n);
    const q = a ? safe(() => rhoAt(a, "layer3.5")) : null, p = a ? safe(() => rhoAt(a, "penult")) : null;
    cl.push(CL(`b[${t}]`, "primary", q && q.rho, q ? le(q.rho, T.AH8.rhoPre) : null));
    cl.push(CL(`c[${t}]`, "primary", p && { rho: p.rho, id_gauss: p.gauss, r20_max: r20max },
      p && num(r20max) ? all([ge(p.rho, T.AH8.rhoPen), le(p.gauss, r20max + T.AH8.penSlack)]) : null)); }
  const [ta, tb] = RUN.twin.map(load); const dr = ta && tb ? safe(() => Math.abs(rhoAt(tb, "penult").rho - rhoAt(ta, "penult").rho)) : null;
  cl.push(CL(`e[twin ${RUN.twin.map(short).join(" vs ")}]`, "required", dr, le(dr, T.AH8.twin)));
  return { id: "AH-8", clauses: cl, ...decide(cl, [...a4bGuard(), ...runGuard(SEEDS)]), info: { r20_max_penult_id_gauss: r20max } };
}
// AH-8(d) has its own verdict: it needs A4b's matched ladder, which (a)-(c) and (e) do not (review item B15)
function AH8d() {
  if (NO_IDG) return notEvaluable("AH-8d", ["--no-idgauss"]);
  const cl = [], r20max = r20maxPen(), mm = m11Runs();
  if (mm === null) cl.push(CL("d", "required", null, null, "no A4b ladder record (--a4b)"));
  else if (!mm.length) cl.push(CL("d", "required", null, null, "no matched rung (A4b ladder not 'matched')"));
  else for (const m of mm) { const a = load(m.atlas), g = a ? safe(() => rhoAt(a, "penult").gauss) : null;
    cl.push(CL(`d[${m.name} ${short(m.atlas)}]`, "required", { id_gauss: g, r20_max: r20max }, num(g) && num(r20max) ? g >= r20max + T.AH8.m11 : null)); }
  return { id: "AH-8d", clauses: cl, ...decide(cl, [...a4bGuard()]), info: { r20_max_penult_id_gauss: r20max, ladder: A4B && A4B.ladder ? A4B.ladder.status : null } };
}

// ---- AX: probe records -------------------------------------------------------------------------------------------------
function probeFiles() { const out = {};
  if (!fs.existsSync(ROOT)) return out;
  for (const d of fs.readdirSync(ROOT)) if (d.startsWith("anomaly_probe_")) { const j = loadJ(path.join(ROOT, d, "probe.json")); if (j) out[path.join(d, "probe.json")] = j; }
  return out; }
function indexProbes(files) { const idx = {}, dup = [];
  const order = Object.entries(files).sort((x, y) => String(x[1].created_utc).localeCompare(String(y[1].created_utc)));
  for (const [file, j] of order) for (const [name, rec] of Object.entries(j.dumps || {})) {
    if (idx[name]) { dup.push(`${name}: ${file} ignored (first touch ${idx[name].file} counts)`); continue; }
    idx[name] = { ...rec, file }; }
  return { idx, dup }; }
// a record counts for axis <ax> when its status is OK or PARTIAL and that axis ran without error (one failing AX never
// voids the others); discovery instances fall back to the Stage 1/2 dump of the same weights when the _st3 one is missing
const usable = (rec, ax) => !!rec && (rec.status === "OK" || rec.status === "PARTIAL") && !!rec[ax] && !rec[ax].error;
const firstOK = (idx, names, ax) => { for (const n of names) if (usable(idx[n], ax)) return { name: n, rec: idx[n] }; return null; };
function gateInstances(idx, ax) { return {
  r20hub: firstOK(idx, [r20("s0hub_st3"), r20("s0hub_st2")], ax), r56hub: firstOK(idx, [r56("s0hub_st3"), r56("s0hub")], ax),
  e40: firstOK(idx, [r56("e40_st3"), r56("e40")], ax) }; }
function confInstances(idx, ax) { return [r56("s1"), r56("s2")].map(n => (usable(idx[n], ax) && idx[n].role === "confirmation" ? { name: n, rec: idx[n] } : { name: n, rec: null })); }
function gateStatus(r20pass, others) { if (r20pass === false) return "DROPPED"; if (r20pass === null || others.some(x => x === null)) return "NOT_EVALUABLE";
  return others.every(Boolean) ? "OPEN" : "CLOSED"; }
// gateOf(inst, tag, fn): fn(rec) returns the list of gate clause results; any evaluator error or missing value is NULL
function gateRunner(gcl, scope = "required") { return (inst, tag, fn) => {
  if (!inst) { gcl.push(CL(`gate[${tag}]`, scope, null, null, "no probe record")); return null; }
  const r = safe(() => fn(inst.rec)); const p = r ? all(r.pass) : null;
  gcl.push(CL(`gate[${tag} ${short(inst.name)}]`, scope, r ? r.value : null, p, r ? (r.text || "") : "evaluator could not read the record")); return p; }; }
function axResult(id, gate, gateCl, conf, hold, info, insts) {
  const g = [...a4bGuard(), ...i2Guard()];                 // s1/s2 dumps from their own weights (review item A4)
  const cr = decide(conf, g), hr = hold ? decide(hold, g) : null;
  const open = gate === "OPEN";
  const v = open ? cr : { verdict: NE, why: [`gate ${gate}`], exceptions: [], readings: [] };
  const files = [...new Set(insts.filter(x => x && x.rec && x.rec.file).map(x => x.rec.file))];
  return { id, gate, gate_clauses: gateCl, confirmation_clauses: conf, ...v, confirmation_values_are_info: !open,
    holdout: hr ? (open ? hr : { verdict: NE, why: [`gate ${gate}`] }) : null, holdout_clauses: hold, info, probe_files: files }; }
const s1Taps = ["s1end", "s2end", "pre"];

function AX1(idx) {
  const g = gateInstances(idx, "AX1"), gcl = [], gateOf = gateRunner(gcl);
  const c1 = A => [["motion_blur", 1], ["defocus_blur", 3], ["fog", 3], ["contrast", 3]].map(([x, s]) => ge(A[sp(x, s)], T.AX1.auc)).concat([ge(A[sp("brightness", 5)], T.AX1.aucBright)]);
  const c3 = (A, Pn) => [sp("motion_blur", 1), sp("brightness", 3)].map(s => (num(A[s]) && num(Pn[s]) ? A[s] - Pn[s] : null));
  // the gate: c1, the two single-class FPRs, and c3 (review item B8: c3 was never computed on discovery data)
  const gfn = rec => { const a = rec.AX1, A = a.auc_perp_pre, Pn = a.auc_perp_penult, sc = a.single_class, d = c3(A, Pn);
    return { pass: [...c1(A), le(sc.perp_pre.fpr_mean, T.AX1.fpr), ge(sc.full_penult.fpr_mean, T.AX1.fprFull), ...d.map(v => ge(v, T.AX1.dAuc))],
      value: { auc: A, fpr_perp_pre: sc.perp_pre.fpr_mean, fpr_full_penult: sc.full_penult.fpr_mean, c3: d } }; };
  const gate = gateStatus(gateOf(g.r20hub, "r20 hub", gfn), [gateOf(g.r56hub, "r56 hub", gfn)]);
  const conf = [], hold = [], ci = confInstances(idx, "AX1");
  for (const c of ci) { const t = short(c.name), a = c.rec ? c.rec.AX1 : null;
    const A = (a && a.auc_perp_pre) || {}, Pn = (a && a.auc_perp_penult) || {};
    conf.push(CL(`c1[${t}]`, "primary", a && A, a ? all(c1(A)) : null));
    const fp = a ? safe(() => a.single_class.perp_pre.fpr_mean) : null;
    conf.push(CL(`c2[${t}]`, "primary", fp, a ? le(fp, T.AX1.fpr) : null));
    const d = a ? c3(A, Pn) : null;
    conf.push(CL(`c3[${t}]`, "primary", d, a ? all(d.map(v => ge(v, T.AX1.dAuc))) : null));
    const sv = a ? a.single_vs_mixed_auc || null : null;
    conf.push(CL(`c4[${t}]`, "primary", sv && sv.perp_pre, sv ? le(sv.perp_pre, T.AX1.perpAuc) : null, "T_perp does not separate single-class from mixed clean batches"));
    // review item B8: T_par >> chi2(9) on a single-class batch by construction; a positive control, never a clause
    conf.push(CL(`c4.par-control[${t}]`, "info", sv && sv.par_pre, sv ? ge(sv.par_pre, T.AX1.parAuc) : null, "positive control (true by construction)"));
    hold.push(CL(`c5[${t}]`, "primary", a && [A[sp("zoom_blur", 3)], A[sp("glass_blur", 3)]], a ? all([ge(A[sp("zoom_blur", 3)], T.AX1.hold), ge(A[sp("glass_blur", 3)], T.AX1.hold)]) : null)); }
  const info = ci.map(c => (c.rec ? safe(() => ({ run: short(c.name), W1_cov_auc: { defocus_s3: c.rec.AX1.auc_cov_pre[sp("defocus_blur", 3)], motion_s3: c.rec.AX1.auc_cov_pre[sp("motion_blur", 3)] },
    W1_cov_single_fpr: c.rec.AX1.single_class.cov_pre.fpr_mean,
    W1_prediction_holds: all([le(c.rec.AX1.auc_cov_pre[sp("defocus_blur", 3)], c.rec.AX1.auc_perp_pre[sp("defocus_blur", 3)]),
      le(c.rec.AX1.auc_cov_pre[sp("motion_blur", 3)], c.rec.AX1.auc_perp_pre[sp("motion_blur", 3)]), ge(c.rec.AX1.single_class.cov_pre.fpr_mean, 0.30)]) })) : null));
  return axResult("AX-1", gate, gcl, conf, hold, { H_W1_competing_arm: info }, [g.r20hub, g.r56hub, ...ci]);
}
// AX-2 is split (review item B7): AX-2a the tap profile, AX-2b the family router; each has its own gate and verdict
const argmaxSet = aucs => { const v = Object.values(aucs || {}).filter(num); if (!v.length) return null; const m = Math.max(...v);
  return Object.keys(aucs).filter(k => num(aucs[k]) && aucs[k] >= m - T.AX2.tie); };
function profileClauses(a2) {                             // (a), (b), (c) of AX-2a on one record's AUC table
  const set = s => (a2 && a2.auc && a2.auc[s] ? argmaxSet(a2.auc[s]) : null);
  const sub = (s, ok) => { const x = set(s); return x ? x.every(k => ok.includes(k)) : null; };
  const nn = DISC.filter(x => !/noise/.test(x)).map(x => { const au = a2.auc[sp(x, 1)] || {}, v = Object.values(au).filter(num), mx = v.length ? Math.max(...v) : null;
    return { c: x, max_auc: mx, set: set(sp(x, 1)), scored: ge(mx, T.AX2.sig) === true }; });
  const scored = nn.filter(x => x.scored);
  return {
    a: { value: set(sp("brightness", 1)), pass: sub(sp("brightness", 1), ["stem", "l10"]) },
    b: { value: [set(sp("defocus_blur", 1)), set(sp("motion_blur", 1))], pass: all([sub(sp("defocus_blur", 1), s1Taps), sub(sp("motion_blur", 1), s1Taps)]) },
    // (c) only where some tap sees the corruption (max AUC >= 0.65); the others are recorded, and none scored is NULL
    c: { value: Object.fromEntries(nn.map(x => [x.c, x.scored ? x.set : `no signal (max AUC ${f(x.max_auc)})`])),
      pass: scored.length ? all(scored.map(x => (x.set ? !x.set.includes("penult") : null))) : null,
      text: scored.length ? `${scored.length} of 8 corruptions with max AUC >= ${T.AX2.sig}` : "no corruption with signal" } }; }
function AX2a(idx) {
  const g = gateInstances(idx, "AX2"), gcl = [], gateOf = gateRunner(gcl);
  const gfn = rec => { const p = profileClauses(rec.AX2); return { pass: [p.a.pass, p.b.pass, p.c.pass], value: { a: p.a.value, b: p.b.value, c: p.c.value }, text: p.c.text }; };
  const gate = gateStatus(gateOf(g.r20hub, "r20 hub", gfn), [gateOf(g.r56hub, "r56 hub", gfn)]);
  const conf = [], ci = confInstances(idx, "AX2");
  for (const c of ci) { const t = short(c.name), p = c.rec ? safe(() => profileClauses(c.rec.AX2)) : null;
    conf.push(CL(`a[${t}]`, "primary", p && p.a.value, p ? p.a.pass : null));
    conf.push(CL(`b[${t}]`, "primary", p && p.b.value, p ? p.b.pass : null));
    conf.push(CL(`c[${t}]`, "primary", p && p.c.value, p ? p.c.pass : null, p ? p.c.text : "")); }
  return axResult("AX-2a", gate, gcl, conf, null, null, [g.r20hub, g.r56hub, ...ci]);
}
function AX2b(idx) {
  const g = gateInstances(idx, "AX2"), gcl = [], gateOf = gateRunner(gcl);
  const fam = ["N", "B", "L", "P"];
  const gfn = rec => { const a = rec.AX2.router.accuracy_by_family; return { pass: fam.map(k => ge(a[k], T.AX2.gate)), value: a }; };
  const gate = gateStatus(gateOf(g.r20hub, "r20 hub", gfn), [gateOf(g.r56hub, "r56 hub", gfn)]);
  const conf = [], hold = [], ci = confInstances(idx, "AX2");
  for (const c of ci) { const t = short(c.name), r = c.rec ? safe(() => c.rec.AX2.router) : null;
    const a = r ? r.accuracy_by_family || {} : null;
    conf.push(CL(`d[${t}]`, "primary", a, a ? all(fam.map(k => ge(a[k], T.AX2.conf))) : null));
    const H = r && r.holdout ? r.holdout : null;
    const frac = (x, fm) => { const v = [3, 5].map(s => (H && H[sp(x, s)] && H[sp(x, s)].assigned ? H[sp(x, s)].assigned[fm] : null)); return v.every(num) ? mean(v) : null; };
    hold.push(CL(`e[${t}]`, "primary", r && { impulse_to_N: frac("impulse_noise", "N"), zoom_to_B: frac("zoom_blur", "B") },
      r ? all([ge(frac("impulse_noise", "N"), T.AX2.impulse), ge(frac("zoom_blur", "B"), T.AX2.zoom)]) : null)); }
  return axResult("AX-2b", gate, gcl, conf, hold, null, [g.r20hub, g.r56hub, ...ci]);
}
function ax3Stats(rec) { const S = rec.AX3.splits, H = [], L = []; let inb = 0, bad = 0, nrIn = 0, nrBad = 0;
  for (const s of DSPL) { const x = S[s]; if (!x) throw new Error(`AX3 ${s} missing`);
    for (let i = 0; i < x.h.length; i++) { if (!num(x.h[i]) || !num(x.loss[i])) continue; H.push(x.h[i]); L.push(x.loss[i]);
      if (x.h[i] <= T.AX3.hold) { inb++; if (x.loss[i] > T.AX3.loss) bad++; }
      if (num(x.e[i]) && x.e[i] >= T.AX3.nr) { nrIn++; if (x.loss[i] > T.AX3.loss) nrBad++; } } }
  const bh = SEV.flatMap(s => S[sp("brightness", s)].h).filter(num);
  const ood = rec.AX3.ood && rec.AX3.ood.ood__cifar100 ? rec.AX3.ood.ood__cifar100.h.filter(num) : null;
  return { rho: spearman(H, L), frac_bad_in_band: inb ? bad / inb : null, n_in_band: inb, bright_frac: bh.length ? bh.filter(v => v <= T.AX3.bright).length / bh.length : null,
    energy_frac_bad: nrIn ? nrBad / nrIn : null, n_energy_band: nrIn, ood_frac: ood && ood.length ? ood.filter(v => v > T.AX3.hold).length / ood.length : null }; }
function AX3(idx) {
  const g = gateInstances(idx, "AX3"), gcl = [], gateOf = gateRunner(gcl);
  const gfn = rec => { const s = ax3Stats(rec); return { pass: [ge(s.rho, T.AX3.rho), le(s.frac_bad_in_band, T.AX3.bad)], value: { rho: s.rho, frac_bad_in_band: s.frac_bad_in_band, n_in_band: s.n_in_band } }; };
  const gate = gateStatus(gateOf(g.r20hub, "r20 hub", gfn), [gateOf(g.r56hub, "r56 hub", gfn), gateOf(g.e40, "e40", gfn)]);
  const conf = [], ci = confInstances(idx, "AX3");
  for (const c of ci) { const t = short(c.name), s = c.rec ? safe(() => ax3Stats(c.rec)) : null;
    conf.push(CL(`a[${t}]`, "primary", s && s.rho, s ? ge(s.rho, T.AX3.rho) : null));
    conf.push(CL(`b[${t}]`, "primary", s && { frac_bad_in_band: s.frac_bad_in_band, n: s.n_in_band }, s ? le(s.frac_bad_in_band, T.AX3.bad) : null));
    conf.push(CL(`c.brightness[${t}]`, "primary", s && s.bright_frac, s ? ge(s.bright_frac, T.AX3.brightFrac) : null));
    const e = s ? ge(s.energy_frac_bad, T.AX3.nrBad) : null;
    conf.push(CL(`d.energy[${t}]`, "reading", s && { frac_bad: s.energy_frac_bad, n: s.n_energy_band }, e,
      e === true ? "energy grade fails (as predicted)" : e === false ? "energy grade not refuted here" : "missing"));
    conf.push(CL(`ood[${t}]`, "info", s && s.ood_frac, s ? ge(s.ood_frac, T.AX3.ood) : null)); }
  return axResult("AX-3", gate, gcl, conf, null, null, [g.r20hub, g.r56hub, g.e40, ...ci]);
}
function AX4(idx) {
  const g = gateInstances(idx, "AX4"), gcl = [], gains = [];
  const gains2 = A => [sp("defocus_blur", 3), sp("motion_blur", 3)].map(s => (A[s] && num(A[s].e_perp) && num(A[s].d1) ? A[s].e_perp - A[s].d1 : null));
  const gOf = (inst, tag, scope) => gateRunner(gcl, scope)(inst, tag, rec => { const v = gains2(rec.AX4.auroc); gains.push(...v); return { pass: v.map(x => ge(x, T.AX4.gain)), value: v }; });
  // review item B17: confirmation is at depth 56, so OPEN needs the resnet56 hub; the resnet20 hub is INFO (DROPPED
  // still needs all four gains < +0.02)
  const a = gOf(g.r20hub, "r20 hub", "info"), b = gOf(g.r56hub, "r56 hub", "required");
  const gate = b === true ? "OPEN" : gains.length === 4 && gains.every(num) && gains.every(x => x < T.AX4.drop) ? "DROPPED" : b === null ? "NOT_EVALUABLE" : "CLOSED";
  const conf = [], hold = [], ci = confInstances(idx, "AX4");
  for (const c of ci) { const t = short(c.name), A = c.rec ? safe(() => c.rec.AX4.auroc) : null;
    const gx = A ? gains2(A) : null;
    conf.push(CL(`gain[${t}]`, "primary", gx, gx ? all(gx.map(v => ge(v, T.AX4.gain))) : null));
    const dv = A ? [sp("defocus_blur", 3), sp("motion_blur", 3)].map(s => (A[s] && num(A[s].e_perp) && num(A[s].dens2) ? A[s].e_perp - A[s].dens2 : null)) : null;
    conf.push(CL(`vs-stage2-density[${t}]`, "primary", dv, dv ? all(dv.map(v => ge(v, -T.AX4.dens))) : null));
    hold.push(CL(`zoom_s3[${t}]`, "info", (A && A[sp("zoom_blur", 3)]) || null, null)); }
  return axResult("AX-4", gate, gcl, conf, hold, { r20_hub_gate: a }, [g.r20hub, g.r56hub, ...ci]);
}

// ---- dry-run synthetic probe records (code-path exercise only; never data) --------------------------------------------
function synthProbe() { const rng = mulberry(20260923), gs = gaussSampler(rng), idx = {};
  const mk = (name, role, lvl) => { const aucp = {}, aucq = {}, aucc = {}; for (const s of DSPL) { aucp[s] = Math.min(1, 0.80 + 0.2 * lvl + 0.02 * gs()); aucq[s] = aucp[s] - 0.15; aucc[s] = aucp[s] - 0.05; }
    aucp[sp("brightness", 3)] = 0.95; aucq[sp("brightness", 3)] = 0.70;
    if (role === "confirmation") { aucp[sp("zoom_blur", 3)] = 0.97; aucp[sp("glass_blur", 3)] = 0.93; }
    const taps = ["stem", "l10", "s1end", "s2end", "pre", "penult"], a2 = {};
    for (const s of DSPL) { a2[s] = Object.fromEntries(taps.map((k, i) => [k, 0.6 + 0.05 * i])); }
    a2[sp("brightness", 1)] = { stem: 0.97, l10: 0.95, s1end: 0.8, s2end: 0.7, pre: 0.65, penult: 0.6 };
    for (const c of ["defocus_blur", "motion_blur", "fog", "contrast", "snow", "pixelate"]) a2[sp(c, 1)] = { stem: 0.6, l10: 0.7, s1end: 0.85, s2end: 0.93, pre: 0.95, penult: 0.8 };
    if (name === r56("s2")) a2[sp("pixelate", 1)] = { stem: 0.6, l10: 0.7, s1end: 0.85, s2end: 0.9, pre: 0.91, penult: 0.93 };   // a (c) failure path
    a2[sp("jpeg_compression", 1)] = { stem: 0.5, l10: 0.52, s1end: 0.55, s2end: 0.58, pre: 0.6, penult: 0.62 };   // no signal: recorded, not scored
    const splits = {}; for (const c of DISC) for (const s of SEV) { const base = (DISC.indexOf(c) % 5) * 0.2 * s / 3, h = [], loss = [], e = [];
      for (let i = 0; i < 200; i++) { const hi = base + 0.05 * gs(); h.push(hi); loss.push(Math.max(-5, 20 * hi + 1.5 * gs())); e.push(1 - 0.03 * hi + 0.005 * gs()); }
      splits[sp(c, s)] = { h, loss, e, e2: e.map(x => x * x) }; }
    const hold = role === "confirmation" ? { [sp("impulse_noise", 3)]: { assigned: { N: 0.97, B: 0.02, L: 0, P: 0.01 } }, [sp("impulse_noise", 5)]: { assigned: { N: 0.99, B: 0.01, L: 0, P: 0 } },
      [sp("zoom_blur", 3)]: { assigned: { N: 0, B: 0.9, L: 0.02, P: 0.08 } }, [sp("zoom_blur", 5)]: { assigned: { N: 0, B: 0.94, L: 0, P: 0.06 } } } : null;
    const au4 = {}; for (const s of DSPL) au4[s] = { e_perp: role === "confirmation" ? 0.7 + 0.1 * lvl : 0.69, d1: 0.68, d1_over_r: 0.69, dens2: 0.75 };
    if (name === r56("s0hub_st3")) for (const s of [sp("defocus_blur", 3), sp("motion_blur", 3)]) au4[s].e_perp = 0.75;
    return { status: "OK", role, file: "SYNTHETIC", AX1: { auc_perp_pre: aucp, auc_perp_penult: aucq, auc_cov_pre: aucc,
      single_class: { perp_pre: { fpr_mean: 0.07 }, perp_penult: { fpr_mean: 0.3 }, full_penult: { fpr_mean: 0.98 }, cov_pre: { fpr_mean: 0.45 } },
      single_vs_mixed_auc: { par_pre: 0.99, perp_pre: 0.58 } },
      AX2: { auc: a2, router: { accuracy_by_family: { N: 0.99, B: 0.93, L: 0.97, P: 0.91 }, holdout: hold } },
      AX3: { splits, ood: { ood__cifar100: { h: Array.from({ length: 200 }, () => 1 + 0.1 * gs()) } } }, AX4: { auroc: au4 } }; };
  idx[r20("s0hub_st3")] = mk(r20("s0hub_st3"), "discovery", 1.0); idx[r56("s0hub_st3")] = mk(r56("s0hub_st3"), "discovery", 1.0);
  idx[r56("e40_st3")] = mk(r56("e40_st3"), "discovery", 0.9);
  idx[r56("s1")] = mk(r56("s1"), "confirmation", 1.0); idx[r56("s2")] = mk(r56("s2"), "confirmation", 0.8);
  return idx; }
// dry run, AH-4: B1's committed fixture atlases (real schema) under --b1-root; the fixtures carry no E9 geometry, so the
// E9 s1/s2 points are the committed A3 margin atlases of the r56 and r20 hubs with sep_ratio_ref := the atlas
// class_centers.sep_ratio (identical by definition on CIFAR, atlas/invariants/margin.py _b1 centers_geometry)
function dryB1Load(n) {
  const stand = { margin_b1_resnet56_s1: ["margin_v1_resnet56_s0hub", r56("s0hub")], margin_b1_resnet56_s2: ["margin_v1_resnet20_s0hub", r20("s0hub")] }[n];
  if (!stand) return loadAt(B1_ROOT, n);
  const m = load(stand[0]), a = load(stand[1]), pm = m && m.per_layer && m.per_layer.penult ? m.per_layer.penult.margin_typeb : null;
  if (!pm || !a) return null;
  return { source: "real", per_layer: { penult: { margin_typeb: { ...pm, centers_geometry: { sep_ratio_ref: safe(() => pen(a).class_centers.sep_ratio) } } } } }; }

// ---- calibration block (dry run): the discovery numbers quoted in ANOMALY_H1.md ---------------------------------------
function calibration() {
  const L = [], R20D = [r20("s0hub"), r20("s1"), r20("s2"), r20("s3"), r20("s4")], R56D = [r56("e10"), r56("e20"), r56("e40"), r56("s0hub")];
  const fitRuns = [r20("s0hub"), ...R56D].filter(load);
  const xy = fitRuns.map(n => [Math.log10(pen(load(n)).neural_collapse.nc1), pen(load(n)).knn_density.splits.test.sparse_frac]);
  const mx = mean(xy.map(p => p[0])), my = mean(xy.map(p => p[1])), b = xy.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / xy.reduce((s, p) => s + (p[0] - mx) ** 2, 0);
  L.push(`[calib AH-1] refit on ${fitRuns.map(short).join(", ")}: sparse = ${(my - b * mx).toFixed(4)} + ${b.toFixed(4)} log10(nc1) (frozen: ${T.AH1.a0} + ${T.AH1.b})`);
  const twelve = [...R20D, r20("s1_ref1"), r20("rand"), ...R56D, r56("rand")].filter(load);
  L.push(`[calib AH-1] residual to the frozen curve: ${[...R20D, r20("s1_ref1"), ...R56D].filter(load).map(n => `${short(n)} ${f(resid1(load(n)))}`).join(", ")}`);
  L.push(`[calib AH-1] max pre-penult test sparse_frac over ${twelve.length} committed atlases (nulls incl.): ${f(Math.max(...twelve.map(n => maxPre(load(n)))))}; penult median shift ${R20D.concat(R56D).map(n => f(pen(load(n)).knn_density.splits.test.median_log_radius_shift)).join("/")}`);
  const tw = twinDelta(r20("s1"), r20("s1_ref1"));
  L.push(`[calib AH-1] r20 twin (test + 30 splits): dq95 ${f(tw.dq95)} dsparse ${f(tw.dsf_mean)} +- ${f(tw.dsf_sd)} dmedian ${f(tw.dmed_mean)} +- ${f(tw.dmed_sd)}`);
  for (const n of [...R20D, ...R56D]) { const h = harm(load(n)); L.push(`[calib AH-2] ${short(n)}: spearman(H,cost) ${f(h.rho)}, max cost at H<=0.25 ${f(h.maxHoldCost, 1)}, brightness max H ${f(Math.max(...h.bright))}, #(nr>=0.95 & cost>10) ${h.energyFail.length}`); }
  for (const [m, a, pre] of [["margin_v1_resnet20_s0hub", r20("s0hub"), "layer3.1"], [m56("s0hub"), r56("s0hub"), "layer3.5"]]) for (const l of [pre, "penult"]) {
    const p = mt(m, l), A = load(a); if (!p || !A) continue;
    L.push(`[calib AH-3] ${short(m)} ${l}: m-d_wrong ${f(p.margin_minus_dist_wrong)} m-d_typeb ${f(p.margin_minus_dist_typeb)} aucD_w ${f(p.auc_dist_wrong)} aucM_w ${f(p.auc_margin_wrong)} sp(m,maxprob) ${f(p.spearman_margin_maxprob)} sp(m,dist) ${f(p.spearman_margin_dist)} seP/seM ${f(p.se_maxprob_typeb / p.se_margin_typeb)} ncAgree ${f(PL(A, l).class_centers.nearest_center_agrees_with_model)} sep ${f(PL(A, l).class_centers.sep_ratio)}`); }
  for (const n of [...R20D, ...R56D, r20("rand"), r56("rand")]) { const b = bright(load(n));
    L.push(`[calib AH-5] ${short(n)}: stem s1 |pc1| ${f(b.stem[0].pc1)} |pc2,3| ${f(b.stem[0].pc23)} s5 ${f(b.stem[1].pc1)}/${f(b.stem[1].pc23)} | pre B/T ${f(b.btp)} csf s1/s3 ${f(b.csf[0])}/${f(b.csf[1])} lowest ${b.lowest} gap ${f(b.gap)} | penult B/T ${f(b.btq)} enr ${f(b.enrMin)}-${f(b.enrMax)} | excess stem/penult ${f(b.stemEx)}/${f(b.penEx)} cost s5 ${f(b.cost5, 1)}`); }
  for (const n of [r20("s0hub"), ...R56D, r20("rand"), r56("rand")]) { const m = motion(load(n));
    L.push(`[calib AH-6] ${short(n)}: pen motion cos15 ${f(m.pen.cos15m)} inc ${f(m.pen.incm)} snow cos15 ${f(m.pen.cos15s)} | pre cos15 ${f(m.pre.cos15m)} | g53 ${f(m.pen.g53)} s53 ${f(m.pen.s53)} dmed g/s ${f(m.pen.gmed)}/${f(m.pen.smed)} | path ratio pre/pen ${f(m.pathRatio)} | C4 min ${f(m.c4)} sep ${f(pen(load(n)).class_centers.sep_ratio)}`); }
  for (const n of [...R20D, ...R56D]) { const a = load(n);
    L.push(`[calib AH-7] ${short(n)}: c ${f(comp(a), 2)} cv_all ${f(cvAll(a))} cv_nn ${f(cvNN(a))} acc_ref ${f(acc(a).ref, 4)}`); }
  const bnd = RUN.band20.map(load).filter(Boolean).map(cvAll); if (bnd.length) L.push(`[calib AH-7] B20 (Stage 2 band stand-in) cv min ${f(Math.min(...bnd))} max ${f(Math.max(...bnd))} edge min - w ${f(2 * Math.min(...bnd) - Math.max(...bnd))} (frozen threshold ${T.AH7.cv})`);
  if (!NO_IDG) for (const [n, l] of [[r56("s0hub"), "layer3.5"], [r56("s0hub"), "penult"], [r56("e40"), "penult"], [r20("s0hub"), "penult"], [r20("s1"), "penult"], [r20("s0hub"), "layer3.1"], [r56("rand"), "layer3.5"]]) {
    const a = load(n); if (!a) continue; const r = rhoAt(a, l); L.push(`[calib AH-8] ${short(n)} ${l}: id ${f(r.id, 2)} ID_gauss ${f(r.gauss, 2)} rho ${f(r.rho)}`); }
  return L;
}

// ---- main ----------------------------------------------------------------------------------------------------------
// every item runs isolated: an evaluator error is that item's NOT_EVALUABLE, never a crash of the others (review item A2)
function runItem(id, fn, isAx) {
  READS = new Map(); let r;
  try { r = fn(); } catch (e) {
    r = notEvaluable(id, [`evaluator error: ${e.message}`], isAx ? { gate: "NOT_EVALUABLE", gate_clauses: [], confirmation_clauses: [], holdout: null, holdout_clauses: null, probe_files: [] } : {}); }
  r.reads = [...READS.values()]; READS = null; return r; }
const PF = DRY ? {} : probeFiles();
const { idx: PIDX, dup: PDUP } = DRY ? { idx: synthProbe(), dup: [] } : indexProbes(PF);
const PROV = provenance(PF);
const b1src = { load: DRY ? dryB1Load : n => loadAt(B1_ROOT, n), verdicts: B1V };
const ITEMS = [runItem("AH-1", AH1), runItem("AH-2", AH2), runItem("AH-3", AH3), runItem("AH-4", () => AH4(b1src)), runItem("AH-5", AH5),
  runItem("AH-6", AH6), runItem("AH-7", AH7), runItem("AH-8", AH8), runItem("AH-8d", AH8d)];
const AXS = [runItem("AX-1", () => AX1(PIDX), true), runItem("AX-2a", () => AX2a(PIDX), true), runItem("AX-2b", () => AX2b(PIDX), true),
  runItem("AX-3", () => AX3(PIDX), true), runItem("AX-4", () => AX4(PIDX), true)];
// scoped provenance (review item A3): AH items need P_A -> P_run with the plan and this file unchanged, and the provenance
// of every atlas they read; AX items also need the probe and its tests unchanged, the self-test record, and the
// provenance of every probe.json they read
if (!DRY) for (const x of [...ITEMS, ...AXS]) { const isAx = /^AX/.test(x.id);
  const probs = [...PROV.core, ...(isAx ? PROV.axCore : [])];
  for (const r of x.reads || []) probs.push(...runProv(r.root, r.name).map(p => `${short(r.name)}: ${p}`));
  for (const fl of x.probe_files || []) probs.push(...(PROV.probes[fl] || []));
  x.provenance = probs.length ? probs : "PASS";
  if (probs.length) { x.verdict = "NOT_EVALUABLE"; x.why = [`provenance: ${probs.slice(0, 3).join("; ")}${probs.length > 3 ? ` (+${probs.length - 3} more)` : ""}`, ...(x.why || [])];
    if (x.holdout) x.holdout = { ...x.holdout, verdict: "NOT_EVALUABLE", why: ["provenance"] }; } }
const out = { dry_run: DRY, root: ROOT, b1: { verdicts: B1_ARG, root: B1_ROOT }, thresholds: T,
  provenance: { ...PROV, pTime: undefined }, probe_duplicates: PDUP,
  probe_dumps: Object.fromEntries(Object.entries(PIDX).map(([k, v]) => [k, { status: v.status, role: v.role, file: v.file || null }])),
  items: ITEMS, axes: AXS };
const Lines = [];
Lines.push(`ANOMALY_H1 evaluation${DRY ? " -- DRY RUN: committed discovery stand-ins, B1 fixture records and synthetic AX records; every line is INFO, nothing is a verdict" : ""}; root ${ROOT}`);
Lines.push(`provenance: AH ${PROV.status}, AX ${PROV.ax_status}${[...PROV.core, ...PROV.axCore].length ? ` (${[...PROV.core, ...PROV.axCore].slice(0, 4).join(" | ")})` : ""}`);
if (DRY) { try { Lines.push(...calibration()); } catch (e) { Lines.push(`[calib] error: ${e.message}`); } }
const cl2s = c => `${c.id}${c.scope === "primary" || c.scope === "required" ? "" : `(${c.scope})`}=${c.pass === null ? "NULL" : c.pass ? "PASS" : "FAIL"}${c.text ? ` [${c.text}]` : ""}`;
const jv = v => JSON.stringify(v, (k, x) => (typeof x === "number" ? +x.toFixed(4) : x));
for (const x of ITEMS) {
  Lines.push(`${x.id}: ${x.verdict}${x.why && x.why.length ? ` (${x.why.slice(0, 6).join("; ")})` : ""}${x.exceptions && x.exceptions.length ? ` exceptions: ${x.exceptions.join(", ")}` : ""}${DRY && x.id === "AH-4" && B1_ARG ? ` [B1 fixture ${B1_ROOT}]` : ""}`);
  Lines.push(`   ${x.clauses.map(cl2s).join(" | ")}`);
  for (const c of x.clauses) if (c.value !== null && typeof c.value !== "object") Lines.push(`     ${c.id}: ${f(c.value)}`);
  for (const c of x.clauses) if (c.value && typeof c.value === "object") Lines.push(`     ${c.id}: ${jv(c.value).slice(0, 240)}`);
}
for (const x of AXS) {
  Lines.push(`${x.id}: gate ${x.gate} -> ${x.verdict}${x.why && x.why.length ? ` (${x.why.slice(0, 6).join("; ")})` : ""}${x.holdout ? `; double holdout ${x.holdout.verdict}` : ""}${DRY ? " [SYNTHETIC probe record]" : ""}`);
  Lines.push(`   gate: ${(x.gate_clauses || []).map(cl2s).join(" | ")}`);
  Lines.push(`   confirmation${x.confirmation_values_are_info ? " (INFO: gate not OPEN)" : ""}: ${(x.confirmation_clauses || []).map(cl2s).join(" | ")}`);
  if (x.holdout_clauses) Lines.push(`   holdout: ${x.holdout_clauses.map(cl2s).join(" | ")}`);
}
if (PDUP.length) Lines.push(`probe duplicates: ${PDUP.join("; ")}`);
console.log(Lines.join("\n"));                             // printed before any file is written (review item A1)
if (CACHE && Object.keys(IDG).length > Object.keys(IDG0).length) {   // deterministic key -> value: only adds keys
  try { writeNew(CACHE, JSON.stringify(IDG)); } catch (e) { console.error(`[anomaly_eval] cache not written: ${e.message}`); } }
if (JSON_OUT) writeNew(JSON_OUT, JSON.stringify(out, null, 1));
