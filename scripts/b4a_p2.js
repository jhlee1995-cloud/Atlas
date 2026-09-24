#!/usr/bin/env node
// scripts/b4a_p2.js -- the two B4a steps on Windows after the pulls (docs/plans/B4A_AMENDMENT.md). Node standard library
// only; it reads the pulled records, never a dump. Committed at P_A with the amendment, before any timing or seal of the
// three B4a units exists, so both rules below are fixed before their inputs.
//
//   node scripts/b4a_p2.js timing --timing <newest check dir>/timing.json --out experiments/b4/timing_b4a.json \
//        [--registry experiments/b4/models.json]                                  (P2, before b4_timeouts.js)
//     shufflenetv2_x0_5's T1 and T2 were timed in the supplement: two jobs on an otherwise idle pod. Every other family
//     was timed in the loaded S1 lane (about P probes at nice 19 beside the training pairs and GPU-D). Rule: per program
//     p in {t1, t2}, wall_s(shufflenetv2_x0_5) := max(measured, max over REFS of wall_s(ref) * (1024 / penult(ref))^2),
//     REFS = the Dnew units of the nearest width (mobilenetv2_x0_5, repvgg_a0: 1280; fit rows without extras, as
//     shufflenetv2_x0_5), timed in the S1 lane. It never lowers a value; every other entry is copied unchanged. Then
//       node scripts/b4_timeouts.js --registry experiments/b4/models.json --timing experiments/b4/timing_b4a.json \
//            --out experiments/b4/timeouts.json
//     Refuses (exit 1, no file) when the timing holds no T1 or T2 run of shufflenetv2_x0_5; refuses an existing --out.
//
//   node scripts/b4a_p2.js seals [--b4a-dir <dir>] [--root .] [--json results/b4/b4a_seals.json]   (step E, after R2)
//     The D7 seal guarantee for the two B4a sealed fit dumps (mobilenetv2_x0_75, shufflenetv2_x1_0). S1's own
//     sealed.json lists them ABSENT and verify_seals skips ABSENT entries, so S2's guard covers them only when S2 ran
//     with ATLAS_B4S1_CHECK_DIR at a complete manifest. --b4a-dir defaults to experiments/b4/freeze_P2.json
//     "b4a".check_dir (the newest supplement check dir). A unit PASSES when
//       (1) <b4a-dir>/seal_diff.json is PASS, names <b4a-dir>/sealed.json with its current sha256 and lists the unit as
//           added, and <b4a-dir>/sealed.json lists the unit's fit dump SEALED;
//       (2) some results/instrument_check_b4s2[_r<k>]/verify_seals*.json has a PASS entry for <unit>:fit, and the
//           manifest it verified lists that dump SEALED with the same meta_sha256 (S2's guard verify_seals.json, or a
//           later --verify-seals run against a complete manifest; the record says which);
//       (3) every T1 fit and T2 probe record of the unit (results/b4_t1/<unit>_fit<tag>, results/b4_t2/<unit><tag>,
//           every tag) lists that dump with the same meta_sha256.
//     A unit that FAILS is NOT_EVALUABLE for every claim, PRIMARY included, until the check passes. Exit 0 PASS, 1 FAIL,
//     2 usage. Refuses an existing --json.
//
//   node scripts/b4a_p2.js --selftest                    fixtures: every branch of both steps, a CLI run of each
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");
const { spawnSync } = require("child_process");

const REPO = path.resolve(__dirname, "..");
const DOC = "docs/plans/B4A_AMENDMENT.md";
const SHUF = "shufflenetv2_x0_5";                              // the B4a Dnew unit (open fit dump)
const REFS = ["mobilenetv2_x0_5", "repvgg_a0"];                 // Dnew, penult 1280, timed in the loaded S1 lane
const TIMED = ["t1", "t2"];                                     // the programs S1 discovery runs on a Dnew unit
const SEALED_UNITS = ["mobilenetv2_x0_75", "shufflenetv2_x1_0"];   // the B4a C units (sealed fit dumps)
const TAG = "(?:_r\\d+|_p2|_s2replay)*";                        // = atlas/b4_core.py TAG_RE
const S2_DIR = /^instrument_check_b4s2(_r\d+)?$/;               // = scripts/t1_eval.js (confirmation check dirs)
const RULE_T = "per program p in {t1, t2}: wall_s(shufflenetv2_x0_5) = max(measured, max over REFS of wall_s(ref) * "
  + "(penult(shufflenetv2_x0_5) / penult(ref))^2); REFS = mobilenetv2_x0_5, repvgg_a0 (Dnew, 1280, timed in the S1 lane)";

const loadJ = p => { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return null; } };
// sha256 of a pod-written JSON file with CRLF read as LF: a Windows checkout (core.autocrlf) hashes as on the pod
const sha256 = p => crypto.createHash("sha256").update(fs.readFileSync(p, "utf8").replace(/\r\n/g, "\n")).digest("hex");
const norm = p => String(p || "").replace(/\\/g, "/").replace(/^(\.\/)+/, "").replace(/\/+$/, "");
const num = v => typeof v === "number" && Number.isFinite(v);
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const writeNew = (p, obj) => {
  if (fs.existsSync(p)) return `${p} exists: never overwritten`;
  fs.mkdirSync(path.dirname(path.resolve(p)), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 1) + "\n");
  return null;
};

// ---------------------------------------------------------------- timing (P2)
function floorTiming(reg, timing) {
  const byId = Object.fromEntries(reg.models.map(m => [m.id, m]));
  const out = JSON.parse(JSON.stringify(timing)), adjusted = {}, problems = [];
  for (const p of TIMED) {
    const m = (timing[p] || {})[SHUF];
    if (!m || !num(m.wall_s)) { problems.push(`${p}: no run of ${SHUF} in the timing`); continue; }
    const refs = REFS.filter(r => num(((timing[p] || {})[r] || {}).wall_s)).map(r => {
      const e = timing[p][r], scale = (byId[SHUF].penult / byId[r].penult) ** 2;
      return { id: r, wall_s: e.wall_s, dir: e.dir || null, scale, floor_s: e.wall_s * scale };
    });
    const floor = refs.length ? Math.max(...refs.map(r => r.floor_s)) : null;
    const used = floor === null ? m.wall_s : Math.max(m.wall_s, floor);
    const b4a = { refs, floor_s: floor, raised: used > m.wall_s,
                  note: refs.length ? null : "no REF unit timed: the measured value is kept" };
    out[p][SHUF] = { ...m, wall_s: used, wall_s_measured: m.wall_s, b4a_floor: b4a };
    adjusted[p] = { measured: m.wall_s, dir: m.dir || null, floor_s: floor, used, raised: b4a.raised };
  }
  return { out, adjusted, problems };
}

function cmdTiming(a) {
  if (!a.timing || !a.out) { console.error("usage: node scripts/b4a_p2.js timing --timing <timing.json> --out <file>"); return 2; }
  const reg = loadJ(a.registry), timing = loadJ(a.timing);
  if (!reg || !timing) { console.error(`[b4a_p2] cannot read ${!reg ? a.registry : a.timing}`); return 2; }
  if (fs.existsSync(a.out)) { console.error(`[b4a_p2] REFUSED: ${a.out} exists: never overwritten`); return 1; }
  const { out, adjusted, problems } = floorTiming(reg, timing);
  if (problems.length) {
    console.error(`[b4a_p2] REFUSED: ${problems.join("; ")}. b4_timeouts.js would take the ShuffleNet t_ref from the`
      + " widest measured unit: relaunch the supplement's discovery jobs, or run b4_timeouts.js on the timing directly"
      + " and record that in freeze_P2.json \"b4a\" (docs/plans/B4A_AMENDMENT.md).");
    return 1;
  }
  out.b4a = { schema: "b4a_timing/1", doc: DOC, rule: RULE_T, source: { path: norm(a.timing), sha256: sha256(a.timing) },
              adjusted, created_utc: new Date().toISOString().replace(/\.\d+Z$/, "Z") };
  const err = writeNew(a.out, out);
  if (err) { console.error(`[b4a_p2] REFUSED: ${err}`); return 1; }
  for (const [p, x] of Object.entries(adjusted))
    console.log(`[b4a_p2] ${p} ${SHUF}: measured ${x.measured} s, floor ${x.floor_s === null ? "none" : x.floor_s.toFixed(1)}`
      + ` s -> ${x.used.toFixed(1)} s${x.raised ? " (raised)" : ""}`);
  console.log(`[b4a_p2] -> ${a.out}`);
  return 0;
}

// ---------------------------------------------------------------- seals (step E)
function readManifest(root, p) {                 // a manifest path as a verify record gives it (relative to the repo)
  const cands = [path.resolve(root, norm(p))];
  const i = norm(p).lastIndexOf("results/");
  if (i > 0) cands.push(path.resolve(root, norm(p).slice(i)));   // an absolute pod path: its repo-relative part
  for (const c of cands) { const j = loadJ(c); if (j) return j; }
  return null;
}
const entryOf = (man, u) => ((man && man.entries) || []).find(e => e.unit === u && e.layout === "fit") || null;

function sealCheck(root, b4aDir) {
  const dir = norm(b4aDir), manRel = `${dir}/sealed.json`, manAbs = path.resolve(root, manRel);
  const man = loadJ(manAbs), diff = loadJ(path.resolve(root, dir, "seal_diff.json"));
  const rep = { schema: "b4a_seals/1", doc: DOC, b4a_dir: dir, manifest: manRel,
                manifest_sha256: man ? sha256(manAbs) : null, seal_diff: null, verify_files: [], units: {} };
  const dprob = [];
  if (!man) dprob.push(`${manRel} missing or unreadable`);
  if (!diff) dprob.push(`${dir}/seal_diff.json missing or unreadable`);
  else {
    if (diff.status !== "PASS") dprob.push(`seal_diff.json status ${diff.status}`);
    if (norm(diff.manifest) !== manRel) dprob.push(`seal_diff.json names ${diff.manifest}, not ${manRel}`);
    if (man && diff.sha256 !== rep.manifest_sha256) dprob.push("sealed.json changed since seal_diff.json");
  }
  rep.seal_diff = { status: diff ? diff.status : null, problems: dprob };
  const res = path.resolve(root, "results");
  const s2dirs = fs.existsSync(res) ? fs.readdirSync(res).filter(d => S2_DIR.test(d)).sort() : [];
  const verifies = [];
  for (const d of s2dirs)
    for (const f of fs.readdirSync(path.join(res, d)).filter(f => /^verify_seals.*\.json$/.test(f)).sort()) {
      const rel = `results/${d}/${f}`, v = loadJ(path.join(res, d, f));
      rep.verify_files.push(rel);
      if (v) verifies.push({ rel, v, man: readManifest(root, v.manifest), guard: f === "verify_seals.json" });
    }
  for (const u of SEALED_UNITS) {
    const e = entryOf(man, u), probs = [...dprob];
    const r = { dump: e ? e.dump : null, meta_sha256: e ? e.meta_sha256 || null : null, covered_by: [], records: [] };
    if (!e || e.status !== "SEALED" || !r.meta_sha256) probs.push(`${manRel} does not list ${u}:fit SEALED`);
    if (diff && !(diff.added || []).includes(`${u}:fit`)) probs.push(`seal_diff.json does not list ${u}:fit as added`);
    if (r.meta_sha256) {
      for (const x of verifies) {                                                        // (2) S2's seal check
        const ve = ((x.v && x.v.entries) || []).find(y => y.unit === u && y.layout === "fit");
        const me = entryOf(x.man, u);
        if (ve && ve.status === "PASS" && me && me.status === "SEALED" && me.meta_sha256 === r.meta_sha256)
          r.covered_by.push({ file: x.rel, manifest: x.v.manifest, s2_guard: x.guard });
      }
      if (!r.covered_by.length) probs.push("no S2 verify_seals*.json checked this dump against a manifest listing it SEALED"
        + " (S2 ran without ATLAS_B4S1_CHECK_DIR at a complete manifest?)");
      for (const [sub, re, fn] of [["b4_t1", new RegExp(`^${reEsc(u)}_fit${TAG}$`), "scoreboard.json"],
                                   ["b4_t2", new RegExp(`^${reEsc(u)}${TAG}$`), "probe.json"]]) {   // (3) records
        const base = path.join(res, sub);
        if (!fs.existsSync(base)) continue;
        for (const d of fs.readdirSync(base).filter(d => re.test(d)).sort()) {
          const p = path.join(base, d, fn);
          if (!fs.existsSync(p)) continue;                   // an incomplete output: no evaluator reads it
          const rec = loadJ(p), rel = `results/${sub}/${d}/${fn}`;
          const got = ((rec && rec.dumps) || []).find(y => norm(y.path) === norm(r.dump));
          const ok = !!got && got.meta_sha256 === r.meta_sha256;
          r.records.push({ path: rel, phase: rec ? rec.phase || null : null, meta_sha256_equal: ok });
          if (!ok) probs.push(`${rel}: ${got ? `meta sha ${String(got.meta_sha256).slice(0, 12)} != sealed` : `lists no ${r.dump}`}`);
        }
      }
    }
    r.problems = probs;
    r.status = probs.length ? "FAIL" : "PASS";
    rep.units[u] = r;
  }
  rep.status = Object.values(rep.units).every(r => r.status === "PASS") ? "PASS" : "FAIL";
  return rep;
}

function cmdSeals(a) {
  let dir = a.b4aDir;
  if (!dir) {
    const fz = loadJ(path.resolve(a.root, "experiments/b4/freeze_P2.json"));
    dir = fz && fz.b4a && fz.b4a.check_dir;
    if (!dir) { console.error("[b4a_p2] give --b4a-dir (experiments/b4/freeze_P2.json has no \"b4a\".check_dir)"); return 2; }
  }
  const rep = sealCheck(a.root, dir);
  rep.created_utc = new Date().toISOString().replace(/\.\d+Z$/, "Z");
  for (const [u, r] of Object.entries(rep.units)) {
    console.log(`[b4a_p2] ${u}:fit ${r.status}: ${r.covered_by.length} S2 seal check(s)`
      + `${r.covered_by.some(c => c.s2_guard) ? " (S2 guard)" : ""}, ${r.records.length} probe record(s)`);
    for (const m of r.problems) console.log(`[b4a_p2]    ${m}`);
  }
  console.log(`[b4a_p2] seals ${rep.status} (B4a manifest ${rep.manifest})`
    + (rep.status === "PASS" ? "" : ": the failing units are NOT_EVALUABLE until the check passes"));
  if (a.json) {
    const err = writeNew(a.json, rep);
    if (err) { console.error(`[b4a_p2] REFUSED: ${err}`); return 2; }
    console.log(`[b4a_p2] -> ${a.json}`);
  }
  return rep.status === "PASS" ? 0 : 1;
}

// ---------------------------------------------------------------- self-test
function selftest() {
  const reg = loadJ(path.join(REPO, "experiments/b4/models.json"));
  const TO = require("./b4_timeouts.js");
  const res = [], chk = (name, c) => { res.push(!!c); console.log(c ? "PASS" : "FAIL", name); };
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "b4a_p2_"));
  const w = (rel, o) => { const p = path.join(tmp, rel); fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, typeof o === "string" ? o : JSON.stringify(o, null, 1)); return p; };
  try {
    // --- timing
    const t = { t1: { [SHUF]: { wall_s: 900, max_rss_mb: 3000, layout: "fit", dir: `${SHUF}_fit_r2` },
                      mobilenetv2_x0_5: { wall_s: 2000, layout: "fit" }, repvgg_a0: { wall_s: 1000, layout: "fit" } },
                t2: { [SHUF]: { wall_s: 500, max_rss_mb: 2000, dir: `${SHUF}_r2` }, mobilenetv2_x0_5: { wall_s: 700 } },
                t1s: {}, t3s: { resnet20_hub: { wall_s: 100 } } };
    const f = floorTiming(reg, t);
    chk("t1: raised to the loaded-lane floor 2000 * (1024/1280)^2 = 1280", Math.abs(f.out.t1[SHUF].wall_s - 1280) < 1e-9
      && f.out.t1[SHUF].wall_s_measured === 900 && f.adjusted.t1.raised);
    chk("t2: measured 500 > floor 700 * 0.64 = 448: kept", f.out.t2[SHUF].wall_s === 500 && !f.adjusted.t2.raised);
    chk("t2: a missing REF (repvgg_a0) is skipped", f.out.t2[SHUF].b4a_floor.refs.length === 1);
    chk("every other entry unchanged; the input not mutated", f.out.t1.mobilenetv2_x0_5.wall_s === 2000
      && f.out.t3s.resnet20_hub.wall_s === 100 && t.t1[SHUF].wall_s === 900 && !f.problems.length);
    const t2 = JSON.parse(JSON.stringify(t)); delete t2.t1.mobilenetv2_x0_5; delete t2.t1.repvgg_a0;
    const f2 = floorTiming(reg, t2);
    chk("no REF timed: measured kept, noted", f2.out.t1[SHUF].wall_s === 900 && f2.adjusted.t1.floor_s === null);
    const t3 = JSON.parse(JSON.stringify(t)); delete t3.t2[SHUF];
    chk("no ShuffleNet T2 run: a problem (refused)", floorTiming(reg, t3).problems.length === 1);
    const o0 = TO.compute(reg, t), o1 = TO.compute(reg, f.out);
    chk("b4_timeouts.js reads the floored t_ref: t1 shufflenetv2_x1_0 = ceil(3 * 1280 * 138500/122500)",
      o1.t1.shufflenetv2_x1_0 === Math.ceil(3 * 1280 * TO.rowsOf(reg.models.find(m => m.id === "shufflenetv2_x1_0"), "fit")
        / TO.rowsOf(reg.models.find(m => m.id === SHUF), "fit")) && o1.t1.shufflenetv2_x1_0 > o0.t1.shufflenetv2_x1_0);
    chk("b4_timeouts.js: other families unchanged", o1.t1.mobilenetv2_x1_0 === o0.t1.mobilenetv2_x1_0);
    const tin = w("timing.json", t), tout = path.join(tmp, "timing_b4a.json");
    const cli = (...args) => spawnSync(process.execPath, [__filename, ...args], { encoding: "utf8" });
    let r = cli("timing", "--timing", tin, "--out", tout, "--registry", path.join(REPO, "experiments/b4/models.json"));
    const tj = loadJ(tout);
    chk("CLI timing: written, with the rule and the source sha256", r.status === 0 && tj && tj.b4a.rule === RULE_T
      && tj.b4a.source.sha256 === sha256(tin) && Math.abs(tj.t1[SHUF].wall_s - 1280) < 1e-9);
    r = cli("timing", "--timing", tin, "--out", tout, "--registry", path.join(REPO, "experiments/b4/models.json"));
    chk("CLI timing: an existing --out is refused", r.status === 1);
    r = cli("timing", "--timing", w("t3.json", t3), "--out", path.join(tmp, "x.json"),
      "--registry", path.join(REPO, "experiments/b4/models.json"));
    chk("CLI timing: no ShuffleNet run -> refused, no file", r.status === 1 && !fs.existsSync(path.join(tmp, "x.json")));

    // --- seals
    const D = "results/instrument_check_b4s1_r2", S1 = "results/instrument_check_b4s1";
    const dumps = { mobilenetv2_x0_75: "results/b4c_mobilenetv2_x0_75/dump", shufflenetv2_x1_0: "results/b4c_shufflenetv2_x1_0/dump" };
    const man = (st, sha = u => `meta_${u}`) => ({ entries: [
      { unit: "resnet44", layout: "fit", dump: "results/b4c_resnet44/dump", status: "SEALED", meta_sha256: "meta_resnet44" },
      ...SEALED_UNITS.map(u => st === "SEALED" ? { unit: u, layout: "fit", dump: dumps[u], status: "SEALED", meta_sha256: sha(u) }
        : { unit: u, layout: "fit", dump: dumps[u], status: "ABSENT" })] });
    const verify = (m, units) => ({ manifest: m, status: "PASS", entries: units.map(u => ({ unit: u, layout: "fit", status: "PASS", why: [] })) });
    const probe = (u, sha = `meta_${u}`) => ({ unit: u, phase: "confirmation", dumps: [{ path: dumps[u], meta_sha256: sha, sealed: true }] });
    const setup = name => {
      const root = path.join(tmp, name), W = (rel, o) => { const p = path.join(root, rel);
        fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, JSON.stringify(o, null, 1)); return p; };
      W(`${S1}/sealed.json`, man("ABSENT"));
      const mp = W(`${D}/sealed.json`, man("SEALED"));
      W(`${D}/seal_diff.json`, { status: "PASS", manifest: `${D}/sealed.json`, sha256: sha256(mp),
                                 added: SEALED_UNITS.map(u => `${u}:fit`), problems: [] });
      W("results/instrument_check_b4s2/verify_seals.json", verify(`${D}/sealed.json`, ["resnet44", ...SEALED_UNITS]));
      for (const u of SEALED_UNITS) {
        W(`results/b4_t2/${u}/probe.json`, probe(u)); W(`results/b4_t1/${u}_fit/scoreboard.json`, probe(u));
      }
      return { root, W };
    };
    let s = setup("ok");
    let rep = sealCheck(s.root, D);
    chk("seals: S2 guard at the B4a manifest, records equal -> PASS", rep.status === "PASS"
      && rep.units.mobilenetv2_x0_75.covered_by[0].s2_guard && rep.units.shufflenetv2_x1_0.records.length === 2);
    s = setup("s1_manifest");
    s.W("results/instrument_check_b4s2/verify_seals.json", verify(`${S1}/sealed.json`, ["resnet44"]));
    rep = sealCheck(s.root, D);
    chk("seals: S2 ran at S1's manifest (the B4a dumps ABSENT, skipped) -> FAIL both", rep.status === "FAIL"
      && SEALED_UNITS.every(u => rep.units[u].status === "FAIL" && !rep.units[u].covered_by.length));
    s.W("results/instrument_check_b4s2/verify_seals_b4a.json", verify(`/workspace/Atlas/${D}/sealed.json`, SEALED_UNITS));
    rep = sealCheck(s.root, D);
    chk("seals: a later --verify-seals at the B4a manifest (absolute pod path) repairs it; not the guard",
      rep.status === "PASS" && !rep.units.shufflenetv2_x1_0.covered_by[0].s2_guard);
    s = setup("relaunch_manifest");
    s.W("results/instrument_check_b4s1_r3/sealed.json", man("SEALED"));
    s.W("results/instrument_check_b4s2/verify_seals.json", verify("results/instrument_check_b4s1_r3/sealed.json", SEALED_UNITS));
    chk("seals: S2 at a later complete manifest with the same seals -> PASS", sealCheck(s.root, D).status === "PASS");
    s.W("results/instrument_check_b4s1_r3/sealed.json", man("SEALED", u => (u === "shufflenetv2_x1_0" ? "other" : `meta_${u}`)));
    rep = sealCheck(s.root, D);
    chk("seals: ... with a different meta sha -> that unit FAILS", rep.units.shufflenetv2_x1_0.status === "FAIL"
      && rep.units.mobilenetv2_x0_75.status === "PASS");
    s = setup("probe_sha");
    s.W("results/b4_t2/shufflenetv2_x1_0_r2/probe.json", probe("shufflenetv2_x1_0", "tampered"));
    rep = sealCheck(s.root, D);
    chk("seals: a probe record (tag _r2) with another meta sha -> FAIL", rep.units.shufflenetv2_x1_0.status === "FAIL"
      && rep.units.shufflenetv2_x1_0.records.length === 3 && rep.units.mobilenetv2_x0_75.status === "PASS");
    s = setup("other_units");
    s.W("results/b4_t1/shufflenetv2_x1_5_fit/scoreboard.json", { dumps: [{ path: "results/b4c_shufflenetv2_x1_5/dump" }] });
    fs.mkdirSync(path.join(s.root, "results/b4_t1/shufflenetv2_x1_0_fit_r3"), { recursive: true });   // incomplete
    rep = sealCheck(s.root, D);
    chk("seals: other units and incomplete outputs are not read", rep.status === "PASS"
      && rep.units.shufflenetv2_x1_0.records.length === 2);
    s = setup("diff_fail");
    s.W(`${D}/seal_diff.json`, { status: "FAIL", manifest: `${D}/sealed.json`, added: [], problems: ["x"] });
    chk("seals: seal_diff.json FAIL -> FAIL", sealCheck(s.root, D).status === "FAIL");
    s = setup("manifest_changed");
    s.W(`${D}/sealed.json`, { ...man("SEALED"), edited: true });
    chk("seals: sealed.json changed after seal_diff.json -> FAIL", sealCheck(s.root, D).status === "FAIL");
    s = setup("crlf");
    const mpath = path.join(s.root, D, "sealed.json");
    fs.writeFileSync(mpath, fs.readFileSync(mpath, "utf8").replace(/\n/g, "\r\n"));
    chk("seals: a CRLF checkout of the same sealed.json still matches seal_diff.json", sealCheck(s.root, D).status === "PASS");
    s = setup("cli");
    s.W("experiments/b4/freeze_P2.json", { b4a: { check_dir: D } });
    const js = path.join(s.root, "results/b4/b4a_seals.json");
    r = spawnSync(process.execPath, [__filename, "seals", "--root", s.root, "--json", js], { encoding: "utf8" });
    chk("CLI seals: --b4a-dir from freeze_P2.json, PASS, record written", r.status === 0 && loadJ(js).status === "PASS");
    r = spawnSync(process.execPath, [__filename, "seals", "--root", s.root, "--json", js], { encoding: "utf8" });
    chk("CLI seals: an existing --json is refused", r.status === 2);
    s.W("results/instrument_check_b4s2/verify_seals.json", verify(`${S1}/sealed.json`, ["resnet44"]));
    r = spawnSync(process.execPath, [__filename, "seals", "--root", s.root, "--b4a-dir", D], { encoding: "utf8" });
    chk("CLI seals: FAIL exits 1", r.status === 1 && /NOT_EVALUABLE/.test(r.stdout));
    r = spawnSync(process.execPath, [__filename, "seals", "--root", path.join(tmp, "empty")], { encoding: "utf8" });
    chk("CLI seals: no --b4a-dir and no freeze_P2.json -> usage (2)", r.status === 2);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  const ok = res.every(Boolean);
  console.log(`b4a_p2 selftest ${ok ? "PASS" : "FAIL"} (${res.filter(Boolean).length}/${res.length})`);
  return ok ? 0 : 1;
}

function main(argv) {
  if (argv.includes("--selftest")) return selftest();
  const arg = (k, d = null) => { const i = argv.indexOf(k); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
  const a = { timing: arg("--timing"), out: arg("--out"), json: arg("--json"), b4aDir: arg("--b4a-dir"),
              root: path.resolve(arg("--root", REPO)), registry: arg("--registry", path.join(REPO, "experiments/b4/models.json")) };
  if (argv[0] === "timing") return cmdTiming(a);
  if (argv[0] === "seals") return cmdSeals(a);
  console.error("usage: node scripts/b4a_p2.js timing|seals ... | --selftest (see the header)");
  return 2;
}

module.exports = { floorTiming, sealCheck, SHUF, REFS, SEALED_UNITS };
if (require.main === module) process.exit(main(process.argv.slice(2)));
