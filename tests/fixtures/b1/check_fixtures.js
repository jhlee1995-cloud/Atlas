// check_fixtures.js -- runs the frozen B1 decision code (scripts/b1_verdicts.js) on every committed fixture scenario
// and checks gate, outcome, labels, states, row 10 and the D13 joint reading against expected.json (integration D17
// item 1: the D14 band, ADDS-OVER-MAXPROB-ONLY, SPLIT-UNTESTED / SPLIT-FRAGILE, _r2, pod_record_agrees, the joint
// reading). Node + git only; writes nothing in the repo (temporary JSON in the OS temp dir).
//   node tests/fixtures/b1/check_fixtures.js          (exit 1 on the first failed expectation)
// tests/test_b1_imagenet.py checks scripts/b1_gate.py on the same scenarios (gate flags).
"use strict";
const fs = require("fs"), os = require("os"), path = require("path"), { execFileSync } = require("child_process");
const HERE = __dirname, REPO = path.resolve(HERE, "..", "..", ".."), SCRIPT = path.join(REPO, "scripts", "b1_verdicts.js");
const EXP = JSON.parse(fs.readFileSync(path.join(HERE, "expected.json"), "utf8"));
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "b1fx_"));
const ROW10 = { ok: "\u2705", y: "\u{1F7E1}", x: "\u2717", open: "\u2B1C" };
let nFail = 0;
function run(sc, a4b) {
  const out = path.join(TMP, `${sc}_${a4b || "none"}.json`);
  const args = [SCRIPT, "--root", path.join(HERE, sc, "results"), "--json", out, "--quiet"];
  if (a4b) args.push("--a4b", path.join(HERE, "a4b", `${a4b}.json`));
  execFileSync(process.execPath, args, { cwd: REPO, stdio: ["ignore", "inherit", "inherit"] });
  return JSON.parse(fs.readFileSync(out, "utf8"));
}
function check(sc, what, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) { nFail++; console.log(`  FAIL ${sc} ${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); }
  return ok;
}
for (const [sc, e] of Object.entries(EXP)) {
  const o = run(sc, null), before = nFail;
  for (const k of ["G0", "G1", "G2", "G3", "open"]) check(sc, `gate ${k}`, o.gate[k], e.gate[k]);
  check(sc, "outcome", o.labels.outcome, e.outcome);
  if ("cstar" in e) check(sc, "c*", o.cstar, e.cstar);
  for (const k of ["margin_vs_maxprob", "margin_vs_logitgap", "V4_replication"]) if (k in e) check(sc, k, o.labels[k], e[k]);
  if ("accuracy_control" in e) check(sc, "accuracy_control", o.labels.accuracy_control.split(" ")[0], e.accuracy_control);
  if ("row10" in e) check(sc, "row10", o.row10.status.startsWith(ROW10[e.row10]), true);
  if ("b2_cancel" in e) check(sc, "B2 rule", o.labels.B_reading && o.labels.B_reading.b2_cancel_and_cnn_scope, e.b2_cancel);
  if ("pod_record_agrees" in e) check(sc, "pod_record_agrees", o.gate.pod_record_agrees, e.pod_record_agrees);
  for (const [m, st] of Object.entries(e.states || {})) for (const [k, v] of Object.entries(st)) check(sc, `${m}.${k}`, (o.models[m] || {})[k], v);
  for (const [m, v] of Object.entries(e.rerun || {})) {
    const r = (o.info[`rerun_${m}`] || {}).rerun; check(sc, `rerun ${m}`, typeof r === "string" ? r.split(":")[0] : r, v); }
  if (e.r50swap_info) check(sc, "resnet50_swap plumbing is INFO", Array.isArray(o.info.resnet50_swap_plumbing) && o.plumbing_post.length === 0, true);
  if (sc === "A") {                                   // INFO paths: E9, exploratory items, provenance without --p
    check(sc, "E9 info", o.info.E9.resnet56_s1.margin_minus_maxprob_confmatched, 0.004);
    check(sc, "E5 (in-sample vs reference A)", o.exploratory.E5.holds, true);
    check(sc, "provenance without --p", o.provenance.status, "NOT_EVALUABLE");
  }
  for (const [a4b, want] of Object.entries(e.joint || {})) {
    const j = run(sc, a4b).joint_reading;
    check(sc, `joint(${a4b}).row`, j.row, want);
    if (e.joint_cnn_sentence) check(sc, `joint(${a4b}) CNN sentence`, /fully fit CNNs/.test(j.reading), true);
    if (want === 3 && "b2_cancel" in e) check(sc, `joint(${a4b}) B2 rule`, j.b2_cancel_and_cnn_scope, e.b2_cancel);
    if (want === 4) check(sc, `joint(${a4b}) B2 not cancelled`, j.b2_cancel_and_cnn_scope, false);
  }
  console.log(`${nFail === before ? "PASS" : "FAIL"} ${sc}: ${o.labels.outcome}${o.cstar != null ? ` (c* ${o.cstar})` : ""}`);
}
fs.rmSync(TMP, { recursive: true, force: true });
console.log(nFail ? `${nFail} expectation(s) failed` : `all ${Object.keys(EXP).length} scenarios PASS`);
process.exit(nFail ? 1 : 0);
