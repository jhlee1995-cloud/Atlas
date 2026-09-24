#!/usr/bin/env node
// tests/collapse_laws_fixture.js -- the node fixture of scripts/collapse_laws.js (docs/plans/T2_COLLAPSE.md section 8;
// T1 review G applied to T2). It writes synthetic probe.json / scoreboard.json trees with KNOWN relations into temporary
// directories, fits with the real CLI and evaluates in-process (collapse_laws.runEvaluate; the CLI itself refuses
// verdicts without --p1 / --p2 / --p-run, verifier T2-4, and is exercised by the git run and the refusal checks):
//   * lawful confirmation units -> the PRIMARY and every secondary SUPPORTED; the same discovery with the confirmation
//     flipped -> PRIMARY REFUTED and every secondary not SUPPORTED (every claim flips, each for its own reason);
//     O1/O2 lawful but O3-O5 not -> MIXED;
//   * NOT_EVALUABLE runs: too few confirmation units; T1 outcomes missing;
//   * output selection (verifier T2-1): C outputs split between '' and _r2 (one T1 scoreboard under _r2 only) give the
//     same PRIMARY as the unsplit run; a C unit probed twice is not evaluable;
//   * guards: different probe code, a leak of a C unit into the fit, append-only refusals, the CLI without --p1 / --p2 /
//     --p-run (UNOFFICIAL), a b4_core.py-only or T1-scoreboard code change P1 -> P2 without --refit-tag, a fitted
//     discovery file changed after the fit (verifier T2-2);
//   * the D7 re-probe path (--refit-tag _p2) and the D15 replay drift per decision (verifier T2-3): a FAIL with zero drift
//     only tags; a t2 drift in O2 / H10 nulls exactly the decisions that read them; a t1-only FAIL moves no t2 decision;
//     a missing replay output nulls every decision; a 1e-9 wobble at a near-zero leaf moves nothing;
//   * a non-dry run in a temporary git repository: --fit --p1 (every instrument hash as of P1), commit P2, --evaluate --p1
//     --p2 --p-run (official; --p-run prefixes accepted), then an evaluator edit after P2 -> NOT_EVALUABLE;
//   * --selftest and --dry-run (the committed known answers).
//   node tests/collapse_laws_fixture.js            (exit 0 = every check PASS)
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), { spawnSync } = require("child_process");
const REPO = path.resolve(__dirname, "..");
const LAWS_JS = path.join(REPO, "scripts", "collapse_laws.js");
const L = require(LAWS_JS), S = require(path.join(REPO, "scripts", "b4_stats.js"));
const REG_PATH = path.join(REPO, "experiments", "b4", "models.json");
const reg = JSON.parse(fs.readFileSync(REG_PATH, "utf8"));
const CONF = ["C", "K", "F", "F20"];
const isConf = m => m.roles.some(r => CONF.includes(r));
const ok = [];
function chk(name, c, info = "") { ok.push(!!c); console.log(c ? "PASS" : "FAIL", name, c ? "" : String(info).slice(0, 600)); }
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "t2_laws_fixture_"));

// tagOf(m, prog, defaultTag) -> the output tag of that unit's T2 probe ("t2") or T1 scoreboard ("t1")
function writeTree(root, scenario, { drop = [], noT1 = [], mutate = null, discTag = "", confTag = "", tagOf = null } = {}) {
  for (const m of reg.models) {
    if (drop.includes(m.id)) continue;
    const { probe, mo } = L.synthUnit(m, scenario);
    const sb = { model_outcomes: mo };
    if (mutate) mutate(m, probe, mo, sb);
    const tag = isConf(m) ? confTag : discTag, t2 = tagOf ? tagOf(m, "t2", tag) : tag, t1 = tagOf ? tagOf(m, "t1", tag) : tag;
    fs.mkdirSync(path.join(root, "b4_t2", `${m.id}${t2}`), { recursive: true });
    fs.writeFileSync(path.join(root, "b4_t2", `${m.id}${t2}`, "probe.json"), JSON.stringify(probe));
    if (!noT1.includes(m.id)) {
      fs.mkdirSync(path.join(root, "b4_t1", `${m.id}_fit${t1}`), { recursive: true });
      fs.writeFileSync(path.join(root, "b4_t1", `${m.id}_fit${t1}`, "scoreboard.json"), JSON.stringify(sb));
    }
  }
}
function run(args, script = LAWS_JS) {
  const r = spawnSync(process.execPath, [script, ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
  return { code: r.status, out: (r.stdout || "") + (r.stderr || "") };
}
const readJ = p => JSON.parse(fs.readFileSync(p, "utf8"));
function evalIn(root, laws, extra = {}) {   // in-process --evaluate, as written to --json (JSON round trip)
  const o = { repo: REPO, root, registry: REG_PATH, laws, replay: null, discTag: null, confTag: null, refitTag: null,
    p1: null, p2: null, pRun: [], ...extra };
  return JSON.parse(JSON.stringify(L.runEvaluate(o, { requireFlags: false })));
}
function fitEval(name, scenario, opts = {}, evalExtra = {}) {
  const root = path.join(TMP, name, "results"), laws = path.join(TMP, name, "laws.json"), json = path.join(TMP, name, "eval.json");
  writeTree(root, scenario, opts);
  const f = run(["--fit", "--root", root, "--registry", REG_PATH, "--out", laws]);
  if (f.code !== 0) return { fit: f, laws, json, root };
  let res = null, err = null;
  try { res = evalIn(root, laws, evalExtra); } catch (e) { err = e; }
  return { fit: f, eval: { code: err ? 1 : 0, out: err ? String(err.stack) : "" }, laws, json, root, res };
}
const ALL_SEC = ["S-PL", "M-DO3", "M-LF", "MP1", "MP3", "MP5", "MP6", "MP7", "P2b", "M-NC4", "FE1", "FE3", "FE9", "P3a", "P4", "P5a",
  "P5b", "P6b", "S7", "P3b", "P5c", "IC-P2", "P7", "P8", "MP-K"];
const primaryKey = p => JSON.stringify({ v: p.verdict, w: p.wins, l: p.losses, n: p.outcomes_evaluable,
  o: Object.fromEntries(Object.entries(p.per_outcome || {}).map(([k, v]) => [k, [v.n, v.mae, v.ratio, v.rho, v.verdict]])) });
function writeReplay(root, m, mutateProbe = null) {   // the S2 replay of one anchor: the lawful S1 records, then mutated
  const { probe, mo } = L.synthUnit(m, "good");
  if (mutateProbe) mutateProbe(probe);
  fs.mkdirSync(path.join(root, "b4_t2", `${m.id}_s2replay`), { recursive: true });
  fs.writeFileSync(path.join(root, "b4_t2", `${m.id}_s2replay`, "probe.json"), JSON.stringify(probe));
  fs.mkdirSync(path.join(root, "b4_t1", `${m.id}_fit_s2replay`), { recursive: true });
  fs.writeFileSync(path.join(root, "b4_t1", `${m.id}_fit_s2replay`, "scoreboard.json"), JSON.stringify({ model_outcomes: mo }));
}
const replayJSON = (p, st) => fs.writeFileSync(p, JSON.stringify({ status: "FAIL", units: {
  "t1:resnet20_hub": { status: st.t1 || "PASS" }, "t1:resnet56_hub": { status: "PASS" },
  "t2:resnet20_hub": { status: st.t2 || "PASS" }, "t2:resnet56_hub": { status: "PASS" }, "t3s:resnet20_hub": { status: st.t3s || "PASS" } } }));

try {
  // ---- 1. flip every claim --------------------------------------------------------------------------------------------
  const g = fitEval("good", "good"), b = fitEval("bad", "bad"), mx = fitEval("mixed", "mixed");
  chk("good: --fit and --evaluate run (exit 0)", g.fit.code === 0 && g.eval && g.eval.code === 0, g.fit.out + (g.eval ? g.eval.out : ""));
  chk("good: PRIMARY SUPPORTED (5 WIN)", g.res && g.res.primary.verdict === "SUPPORTED" && g.res.primary.wins === 5, g.res && JSON.stringify(g.res.primary));
  chk("the secondary family is exactly the registered list", g.res && JSON.stringify(Object.keys(g.res.secondary.items).sort()) === JSON.stringify([...ALL_SEC].sort()),
    g.res && Object.keys(g.res.secondary.items));
  chk("bad (same discovery, confirmation flipped): PRIMARY REFUTED", b.res && b.res.primary.verdict === "REFUTED", b.res && JSON.stringify(b.res.primary.per_outcome));
  for (const id of ALL_SEC) {
    const vg = g.res && g.res.secondary.items[id], vb = b.res && b.res.secondary.items[id];
    chk(`flip ${id}: SUPPORTED when lawful, not SUPPORTED when flipped (and evaluable both times)`,
      vg && vb && vg.verdict === "SUPPORTED" && vb.verdict !== "SUPPORTED" && vb.verdict_rule !== "NOT_EVALUABLE",
      `${vg && vg.verdict} / ${vb && vb.verdict} ${vb && vb.why ? vb.why : ""}`);
  }
  chk("mixed (O1, O2 lawful; O3-O5 not): PRIMARY MIXED", mx.res && mx.res.primary.verdict === "MIXED" && mx.res.primary.wins === 2,
    mx.res && `${mx.res.primary.verdict} ${mx.res.primary.wins}`);
  chk("laws_frozen.json is ASCII and records coordinates, directions, the P3b threshold, its score on D and the fitted files", (() => {
    const s = fs.readFileSync(g.laws, "utf8"), j = JSON.parse(s);
    return ![...s].some(c => c.charCodeAt(0) > 127) && j.schema === "b4_t2_laws/1" && j.primary.O1.status === "FROZEN"
      && typeof j.law_items.P3a.direction_as_registered === "boolean" && isFinite(j.p3b.theta) && j.p3b.on_D.verdict_rule
      && j.fitted_on.length === 21 && j.fitted_on.every(f => !CONF.some(k => (reg.models.find(m => m.id === f.id).roles).includes(k)))
      && j.fitted_on.every(f => /^b4_t2\/.+\/probe\.json$/.test(f.probe_path) && f.probe_sha256 && "core_sha256" in f && "collapse_sha256" in f)
      && ["probe_code", "core_sha256", "collapse_sha256", "t1_code", "t1_core"].every(k => Array.isArray(j.code_sets[k]));
  })());
  chk("the fit never reads a confirmation unit (C probes present in the tree, 21 fitted)", readJ(g.laws).fitted_on.length === 21);
  chk("functional units are reported for the PRIMARY", g.res && g.res.primary.per_outcome.O1.functional && isFinite(g.res.primary.per_outcome.O1.functional.mae_acc_coll));
  chk("rule 8 is annotated on the correlation items (RULE8-CHECK at |rho| >= 0.95)", g.res && g.res.secondary.items.MP1.rule8.flags.includes("RULE8-CHECK"));
  chk("rule 8 is annotated (not applied) on every PRIMARY outcome: partial on log penult width, RULE8-CHECK",
    g.res && Object.values(g.res.primary.per_outcome).every(v => v.rule8 && "partial_log_dim" in v.rule8 && v.rule8.flags.includes("RULE8-CHECK"))
    && g.res.primary.verdict === "SUPPORTED");

  // ---- 2. NOT_EVALUABLE ----------------------------------------------------------------------------------------------
  const dropC = ["vgg13_bn", "vgg16_bn", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "repvgg_a1",
    ...reg.models.filter(m => m.roles.includes("K")).map(m => m.id)];
  const ne = fitEval("ne", "good", { drop: dropC });
  chk("too few C units: PRIMARY NOT_EVALUABLE", ne.res && ne.res.primary.verdict === "NOT_EVALUABLE", ne.res && ne.res.primary.verdict);
  const neBad = ne.res ? ALL_SEC.filter(id => ne.res.secondary.items[id].verdict !== "NOT_EVALUABLE") : ["no result"];
  chk("too few C units: every secondary NOT_EVALUABLE", neBad.length === 0, neBad.map(id => `${id}:${ne.res && ne.res.secondary.items[id].verdict}`));
  const noT1 = fitEval("not1", "good", { noT1: reg.models.filter(m => m.roles.includes("C")).map(m => m.id) });
  chk("T1 outcomes missing for C: PRIMARY NOT_EVALUABLE (3 outcomes < 4), M-NC4 NOT_EVALUABLE",
    noT1.res && noT1.res.primary.verdict === "NOT_EVALUABLE" && noT1.res.primary.outcomes_evaluable === 3
    && noT1.res.secondary.items["M-NC4"].verdict === "NOT_EVALUABLE", noT1.res && JSON.stringify(noT1.res.primary));
  const fewD = fitEval("fewd", "good", { drop: reg.models.filter(m => m.roles.includes("D")).slice(0, 12).map(m => m.id) });
  chk("too few discovery nets: --fit refuses (exit != 0), nothing frozen", fewD.fit.code !== 0 && !fs.existsSync(fewD.laws), fewD.fit.out);

  // ---- 2b. output selection across tags (verifier T2-1) --------------------------------------------------------------
  const Cids = reg.models.filter(m => m.roles.includes("C")).map(m => m.id);
  const sp = fitEval("split", "good", { tagOf: (m, prog, tag) => {
    if (!m.roles.includes("C")) return tag;
    if (m.id === "resnet44") return prog === "t1" ? "_r2" : "";            // T2 probe under '', T1 scoreboard under _r2 only
    return Cids.indexOf(m.id) % 2 ? "_r2" : ""; } });
  chk("C outputs split between '' and _r2 (one T1 scoreboard under _r2 only): the same PRIMARY as the unsplit run",
    sp.res && g.res && primaryKey(sp.res.primary) === primaryKey(g.res.primary), sp.res && `${primaryKey(sp.res.primary)}\n${primaryKey(g.res.primary)}`);
  const s44 = sp.res && sp.res.info.units.find(u => u.id === "resnet44");
  chk("... the chosen directories are recorded per unit (resnet44: probe b4_t2/resnet44, scoreboard b4_t1/resnet44_fit_r2)",
    s44 && s44.sources.t2 === "b4_t2/resnet44" && s44.sources.t1 === "b4_t1/resnet44_fit_r2", s44 && JSON.stringify(s44.sources));
  {
    const m = reg.models.find(x => x.id === "vgg19_bn"), { probe } = L.synthUnit(m, "good");
    fs.mkdirSync(path.join(sp.root, "b4_t2", "vgg19_bn_r3"), { recursive: true });
    fs.writeFileSync(path.join(sp.root, "b4_t2", "vgg19_bn_r3", "probe.json"), JSON.stringify(probe));
    const dup = evalIn(sp.root, sp.laws), u = dup.info && dup.info.units.find(x => x.id === "vgg19_bn");
    chk("a C unit with two confirmation probes (touched twice) is not evaluable; the PRIMARY runs on the other 11",
      u && u.evaluable === false && /probed 2 times/.test(u.not_evaluable_why || "") && dup.provenance.touched_twice.some(t => t.id === "vgg19_bn")
      && dup.primary.per_outcome.O2.n === 11, u && JSON.stringify([u.evaluable, u.not_evaluable_why, dup.primary.per_outcome.O2.n]));
  }

  // ---- 3. guards and append-only -------------------------------------------------------------------------------------
  const gc = fitEval("guard_code", "good", { mutate: (m, p) => { if (m.id === "vgg16_bn") p.code.sha256 = "f".repeat(64); } });
  chk("guard: a C unit probed by other code -> NOT_EVALUABLE", gc.res && gc.res.guard && gc.res.primary.verdict === "NOT_EVALUABLE", gc.res && gc.res.guard);
  const gp = fitEval("guard_phase", "good", { mutate: (m, p) => { if (m.id === "resnet44") p.phase = "discovery"; } });
  chk("guard: a C unit probed in the discovery phase -> NOT_EVALUABLE", gp.res && /confirmation phase/.test(gp.res.guard || ""), gp.res && gp.res.guard);
  const lk = JSON.parse(fs.readFileSync(g.laws, "utf8")); lk.fitted_on.push({ id: "vgg13_bn" });
  const lkp = path.join(TMP, "good", "laws_leak.json"); fs.writeFileSync(lkp, JSON.stringify(lk));
  const el = run(["--evaluate", "--root", g.root, "--registry", REG_PATH, "--laws", lkp, "--json", path.join(TMP, "good", "eval_leak.json")]);
  chk("guard: a C unit inside the fit (leak) -> NOT_EVALUABLE", el.code === 0 && /in the fit/.test(readJ(path.join(TMP, "good", "eval_leak.json")).guard || ""), el.out);
  fs.writeFileSync(g.json, "{}\n");
  const again = run(["--evaluate", "--root", g.root, "--registry", REG_PATH, "--laws", g.laws, "--json", g.json]);
  chk("append-only: --evaluate refuses an existing --json (exit 2)", again.code === 2, again.out);
  const fitAgain = run(["--fit", "--root", g.root, "--registry", REG_PATH, "--out", g.laws]);
  chk("append-only: --fit refuses an existing --out (exit 2)", fitAgain.code === 2, fitAgain.out);
  chk("an evaluation without --p1 / --p2 / --p-run is marked non-official", g.res && g.res.provenance.official === false);
  const cj = path.join(TMP, "good", "eval_cli_noflags.json");
  const cli = run(["--evaluate", "--root", g.root, "--registry", REG_PATH, "--laws", g.laws, "--json", cj]);
  const cjr = cli.code === 0 ? readJ(cj) : null;
  chk("the CLI without --p1 / --p2 / --p-run emits no verdict: guard NOT_EVALUABLE naming the flags, printed UNOFFICIAL (T2-4)",
    cjr && /--p1, --p2 and --p-run/.test(cjr.guard || "") && cjr.primary.verdict === "NOT_EVALUABLE" && /\[UNOFFICIAL/.test(cli.out), cli.out);
  const cc = fitEval("core", "good", { mutate: (m, p) => { p.code.core_sha256 = (isConf(m) ? "c" : "b").repeat(64); } });
  chk("guard: atlas/b4_core.py changed P1 -> P2 (C probes only) without --refit-tag -> NOT_EVALUABLE, rerun with --refit-tag _p2 (T2-2)",
    cc.res && /probe code changed P1->P2 \(atlas\/b4_core\.py\): rerun with --refit-tag _p2/.test(cc.res.guard || "") && cc.res.primary.verdict === "NOT_EVALUABLE",
    cc.res && cc.res.guard);
  writeTree(cc.root, "good", { discTag: "_p2", confTag: "_unused", mutate: (m, p) => { p.code.core_sha256 = "c".repeat(64); } });
  const ccr = evalIn(cc.root, cc.laws, { refitTag: "_p2" });
  chk("... with --refit-tag _p2 on discovery re-probes run with the P2 core the run is evaluable (PRIMARY SUPPORTED)",
    !ccr.guard && ccr.primary.verdict === "SUPPORTED" && /refit on _p2/.test(ccr.laws_applied), ccr.guard);
  const ct = fitEval("t1code", "good", { mutate: (m, p, mo, sb) => { sb.code = { sha256: (isConf(m) ? "e" : "d").repeat(64), core_sha256: "a".repeat(64) }; } });
  chk("guard: the T1 scoreboard code behind O1 / O5 differs between the fit and the C units -> NOT_EVALUABLE (T2-2)",
    ct.res && /scripts\/t1_scoreboard\.py/.test(ct.res.guard || "") && ct.res.primary.verdict === "NOT_EVALUABLE", ct.res && ct.res.guard);
  const fx = fitEval("fitted_file", "good");
  fs.appendFileSync(path.join(fx.root, "b4_t2", "resnet56_e40", "probe.json"), "\n");
  const fxr = evalIn(fx.root, fx.laws);
  chk("guard: a fitted discovery probe changed after the fit -> NOT_EVALUABLE (T2 review #17)",
    /fitted discovery files differ from the fit: resnet56_e40/.test(fxr.guard || ""), fxr.guard);

  // ---- 4. D7 re-probe path and D15 replay drift ----------------------------------------------------------------------
  writeTree(g.root, "good", { discTag: "_p2", confTag: "_unused" });
  const er = evalIn(g.root, g.laws, { refitTag: "_p2" });
  chk("--refit-tag _p2: the frozen procedure refitted on the re-probes; P2 laws marked INFO",
    /refit on _p2/.test(er.laws_applied) && er.primary.verdict === "SUPPORTED" && !er.guard, er.guard);
  const anchors = reg.models.filter(m => m.roles.includes("ANCHOR"));
  for (const m of anchors) writeReplay(g.root, m);
  const rp = path.join(TMP, "good", "replay.json");
  fs.writeFileSync(rp, JSON.stringify({ status: "FAIL", units: {} }));
  const j0 = evalIn(g.root, g.laws, { replay: rp });
  chk("replay FAIL without a failing t1 / t2 unit: verdicts kept, tagged REPLAY-DRIFT", j0.primary.verdict === "SUPPORTED" && j0.primary.tag === "REPLAY-DRIFT"
    && j0.secondary.items.MP1.verdict === "SUPPORTED" && j0.secondary.items.MP1.tag === "REPLAY-DRIFT", JSON.stringify(j0.replay));
  replayJSON(rp, { t2: "FAIL" });
  const j1 = evalIn(g.root, g.laws, { replay: rp });
  chk("replay t2 FAIL with zero measured drift: verdicts kept, every label tagged", j1.replay.triggered.t2 === true && j1.primary.verdict === "SUPPORTED"
    && ALL_SEC.every(id => j1.secondary.items[id].verdict === "SUPPORTED" && j1.secondary.items[id].tag === "REPLAY-DRIFT"),
    ALL_SEC.filter(id => j1.secondary.items[id].verdict !== "SUPPORTED").join(" "));
  for (const m of anchors) writeReplay(g.root, m, p => { p.outcomes.O2 += 1.0; p.targets.harm.H10 += 1.0; });
  const j2 = evalIn(g.root, g.laws, { replay: rp });
  const ne2 = ALL_SEC.filter(id => j2.secondary.items[id].verdict === "NOT_EVALUABLE");
  chk("t2 drift of 1.0 in O2 / H10: PRIMARY O2 NOT_EVALUABLE (REPLAY-DRIFT), PRIMARY on 4 outcomes",
    j2.primary.per_outcome.O2.verdict === "NOT_EVALUABLE" && /REPLAY-DRIFT/.test(j2.primary.per_outcome.O2.why) && j2.primary.outcomes_evaluable === 4,
    JSON.stringify(j2.primary.per_outcome.O2));
  chk("... P5b (reads H10) NOT_EVALUABLE; MP1 (reads neither) keeps SUPPORTED, tagged; no blanket rule (T2-3)",
    j2.secondary.items.P5b.verdict === "NOT_EVALUABLE" && /REPLAY-DRIFT/.test(j2.secondary.items.P5b.why || "")
    && j2.secondary.items.MP1.verdict === "SUPPORTED" && j2.secondary.items.MP1.tag === "REPLAY-DRIFT" && ne2.length < ALL_SEC.length / 2, ne2.join(" "));
  replayJSON(rp, { t1: "FAIL", t3s: "FAIL" });
  const j3 = evalIn(g.root, g.laws, { replay: rp });
  chk("a t1 / t3s-only replay FAIL moves no decision that reads t2 fields (same drifted t2 files): PRIMARY on 5 outcomes, P5b evaluable, tagged",
    j3.replay.triggered.t2 === false && j3.replay.triggered.t1 === true && j3.primary.outcomes_evaluable === 5 && j3.primary.verdict === "SUPPORTED"
    && j3.secondary.items.P5b.verdict === "SUPPORTED" && j3.secondary.items.P5b.tag === "REPLAY-DRIFT", JSON.stringify(j3.replay.triggered));
  replayJSON(rp, { t2: "FAIL" });
  fs.rmSync(path.join(g.root, "b4_t2", "resnet56_hub_s2replay"), { recursive: true, force: true });
  const j4 = evalIn(g.root, g.laws, { replay: rp });
  chk("a t2 FAIL whose replay output is missing (drift cannot be bounded): PRIMARY and every secondary NOT_EVALUABLE",
    j4.primary.verdict === "NOT_EVALUABLE" && ALL_SEC.every(id => j4.secondary.items[id].verdict === "NOT_EVALUABLE"),
    ALL_SEC.filter(id => j4.secondary.items[id].verdict !== "NOT_EVALUABLE").join(" "));
  const isAnchor = m => m.roles.includes("ANCHOR");
  const wb = fitEval("wobble", "good", { mutate: (m, p) => { if (isAnchor(m)) p.targets.p3b.ci_lo = 1e-7; } });
  for (const m of anchors) writeReplay(wb.root, m, p => { p.targets.p3b.ci_lo = 1e-7 + 1e-9; });
  const rpw = path.join(TMP, "wobble", "replay.json"); replayJSON(rpw, { t2: "FAIL" });
  const jw = evalIn(wb.root, wb.laws, { replay: rpw });
  chk("a 1e-9 wobble at a 1e-7 leaf (relative 1e-2, the old blanket rule's trigger) moves no decision",
    jw.primary.verdict === "SUPPORTED" && ALL_SEC.every(id => jw.secondary.items[id].verdict === "SUPPORTED") && jw.replay.rel_drift_probe_INFO < 1e-3,
    `${ALL_SEC.filter(id => jw.secondary.items[id].verdict !== "SUPPORTED").join(" ")} ${jw.replay.rel_drift_probe_INFO}`);

  // ---- 5. non-dry run in a temporary git repository ------------------------------------------------------------------
  const git = (cwd, ...a) => spawnSync("git", ["-c", "user.name=t2fixture", "-c", "user.email=t2fixture@invalid", "-c", "core.autocrlf=false", ...a],
    { cwd, encoding: "utf8" });
  if (git(TMP, "--version").status === 0) {
    const R = path.join(TMP, "gitrepo");
    for (const f of ["scripts/collapse_laws.js", "scripts/b4_stats.js", "scripts/collapse_probe.py", "scripts/t1_scoreboard.py", "atlas/b4_core.py",
      "atlas/b4_collapse.py", "experiments/b4/models.json", "experiments/b4/prereg_p7.json"]) {
      fs.mkdirSync(path.dirname(path.join(R, f)), { recursive: true }); fs.copyFileSync(path.join(REPO, f), path.join(R, f)); }
    git(R, "init", "-q"); git(R, "add", "-A"); git(R, "commit", "-q", "-m", "P1");
    const P1 = git(R, "rev-parse", "HEAD").stdout.trim(), sha = f => S.fileSha256(path.join(R, f));
    const code = sha("scripts/collapse_probe.py"), core = sha("atlas/b4_core.py"), coll = sha("atlas/b4_collapse.py"), t1c = sha("scripts/t1_scoreboard.py");
    writeTree(path.join(R, "results"), "good", { mutate: (m, p, mo, sb) => {
      Object.assign(p.code, { sha256: code, core_sha256: core, collapse_sha256: coll, repo_commit: isConf(m) ? "s2head" : "s1head" });
      sb.code = { sha256: t1c, core_sha256: core, repo_commit: isConf(m) ? "s2head" : "s1head" }; } });
    const script = path.join(R, "scripts", "collapse_laws.js"), lawsP = path.join(R, "experiments", "b4", "laws_frozen.json");
    const f = run(["--fit", "--p1", P1, "--out", lawsP], script);
    chk("git: --fit --p1 checks every instrument hash (probe, core, collapse, T1 scoreboard) against P1 and freezes", f.code === 0 && fs.existsSync(lawsP), f.out);
    git(R, "add", "experiments/b4/laws_frozen.json"); git(R, "commit", "-q", "-m", "P2");
    const P2 = git(R, "rev-parse", "HEAD").stdout.trim(), ej = path.join(R, "results", "b4", "t2_eval.json");
    const e = run(["--evaluate", "--p1", P1, "--p2", P2, "--p-run", "s1head,s2head", "--json", ej], script);
    const je = e.code === 0 ? readJ(ej) : null;
    chk("git: official evaluation (P1 <= P2, laws and evaluator as at P2, P7 read at P1, instrument hashes at P2) -> PRIMARY SUPPORTED",
      je && !je.guard && je.provenance.official && je.primary.verdict === "SUPPORTED" && /@/.test(je.p7_registration) && !/UNOFFICIAL/.test(e.out),
      e.out + (je ? je.guard : ""));
    const e1 = run(["--evaluate", "--p1", P1, "--p2", P2, "--p-run", "s1he,s2he", "--json", ej + ".1"], script);
    chk("git: --p-run matches by prefix (as t1_eval.js / t3s_eval.js)", e1.code === 0 && !readJ(ej + ".1").guard, e1.out);
    const e2 = run(["--evaluate", "--p1", P1, "--p2", P2, "--p-run", "s1head", "--json", ej + ".2"], script);
    chk("git: probes outside --p-run -> NOT_EVALUABLE", e2.code === 0 && /p-run/.test(readJ(ej + ".2").guard || ""), e2.out);
    fs.appendFileSync(script, "\n// an edit after P2\n");
    const e3 = run(["--evaluate", "--p1", P1, "--p2", P2, "--p-run", "s1head,s2head", "--json", ej + ".3"], script);
    chk("git: an evaluator edit after P2 -> NOT_EVALUABLE", e3.code === 0 && /changed after P2/.test(readJ(ej + ".3").guard || ""), e3.out);
    const fb = run(["--fit", "--p1", P2, "--out", lawsP + ".x"], script);
    let rz = null;
    chk("git: --fit refuses discovery records whose b4_collapse.py is not the one at P1 (T2-2)", (() => {
      fs.appendFileSync(path.join(R, "atlas", "b4_collapse.py"), "\n# changed\n"); git(R, "commit", "-qam", "collapse change");
      const P3 = git(R, "rev-parse", "HEAD").stdout.trim();
      rz = run(["--fit", "--p1", P3, "--out", lawsP + ".z"], script);
      return fb.code === 0 && rz.code !== 0 && /b4_collapse\.py/.test(rz.out) && !fs.existsSync(lawsP + ".z"); })(), rz && rz.out);
    chk("git: --fit refuses discovery probes that did not run the P1 probe code", (() => {
      fs.appendFileSync(path.join(R, "scripts", "collapse_probe.py"), "\n# changed\n"); git(R, "commit", "-qam", "probe change");
      const P4 = git(R, "rev-parse", "HEAD").stdout.trim();
      const r = run(["--fit", "--p1", P4, "--out", lawsP + ".y"], script);
      return r.code !== 0 && /collapse_probe\.py/.test(r.out) && !fs.existsSync(lawsP + ".y"); })(), fb.out);
  } else chk("git available for the non-dry run", false, "git --version failed");

  // ---- 6. the evaluator's own self-test and the committed known answers ----------------------------------------------
  const st = run(["--selftest"]);
  chk("collapse_laws.js --selftest exits 0", st.code === 0, st.out.split("\n").filter(l => l.startsWith("FAIL")).join(" | "));
  const dr = run(["--dry-run"]);
  chk("--dry-run reproduces AH-1 and X9 (committed known answers)", dr.code === 0 && (dr.out.match(/REPRODUCED/g) || []).length === 2 && !/NOT REPRODUCED/.test(dr.out), dr.out);
  chk("the P7 registration equals a fresh --dry-run", (() => { const reg7 = readJ(path.join(REPO, "experiments", "b4", "prereg_p7.json"));
    const fresh = L.dryRun(REPO).p7; return JSON.stringify(reg7) === JSON.stringify(JSON.parse(JSON.stringify(fresh))); })());
} catch (e) {
  chk(`fixture crashed: ${e.stack}`, false);
} finally {
  fs.rmSync(TMP, { recursive: true, force: true });
}
const all = ok.every(Boolean);
console.log(all ? `collapse_laws fixture PASS (${ok.length} checks)` : `collapse_laws fixture FAIL (${ok.filter(v => !v).length} of ${ok.length})`);
process.exit(all ? 0 : 1);
