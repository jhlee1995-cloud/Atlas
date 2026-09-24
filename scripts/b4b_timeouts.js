#!/usr/bin/env node
// scripts/b4b_timeouts.js -- amendment B4b (docs/plans/B4B_AMENDMENT.md; owner-approved 2026-09-24). Every session-2
// time limit becomes twice the limit the frozen scripts/b4_timeouts.js emitted, capped at that script's 14400 s
// ceiling, and is never lowered. Only limits change: a limit decides whether a job may finish, never what it computes.
// 'rule' is rewritten to state the B4b rule and quote the emitted one; peak_gb, split and unmeasured are copied
// unchanged. The source sha256 is taken over the CRLF-normalised bytes (the batch-4 convention, scripts/b4_stats.js
// fileSha256), so a Windows checkout (core.autocrlf) records the same value and writes the same output.
//   node scripts/b4b_timeouts.js --in experiments/b4/timeouts_b4_emitted.json --out experiments/b4/timeouts.json
//        (Windows, before the P2 commit; refuses to overwrite)
//   node scripts/b4b_timeouts.js --selftest
"use strict";
const fs = require("fs");
const crypto = require("crypto");
const os = require("os");
const path = require("path");
const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const FACTOR = 2, MAX_S = 14400;                                  // B4b; MAX_S = scripts/b4_timeouts.js MAX_S
const PROGS = ["t1", "t1s", "t2", "t3s"];                         // the keyed limits scripts/b4_reg.py timeout() reads
const RULE = `min(${MAX_S}, ${FACTOR} x emitted), never lower`;
const sha256Lf = text => crypto.createHash("sha256").update(text.replace(/\r\n/g, "\n"), "utf8").digest("hex");

function amend(emitted, sourcePath, sourceSha) {
  const o = JSON.parse(JSON.stringify(emitted));
  let n = 0;
  for (const p of PROGS) {
    for (const [k, v] of Object.entries(o[p] || {})) {
      if (!Number.isFinite(v) || v <= 0) throw new Error(`${p}.${k}: not a positive number (${v})`);
      o[p][k] = Math.max(v, Math.min(MAX_S, Math.ceil(FACTOR * v)));
      n += 1;
    }
  }
  if (typeof emitted.rule !== "string") throw new Error("the emitted file has no 'rule'");
  o.rule = `B4b: ${RULE} (see 'amendment'); emitted limit (scripts/b4_timeouts.js): ${emitted.rule}`;
  o.amendment = { id: "B4b", doc: "docs/plans/B4B_AMENDMENT.md", rule: RULE,
                  factor: FACTOR, cap_s: MAX_S, limits: n, source: sourcePath, source_sha256: sourceSha };
  return o;
}

function run(inp, outp) {
  if (fs.existsSync(outp)) { console.error(`[b4b_timeouts] ${outp} exists: never overwritten`); return 2; }
  const text = fs.readFileSync(inp, "utf8");
  const o = amend(JSON.parse(text), inp.replace(/\\/g, "/"), sha256Lf(text));
  fs.writeFileSync(outp, JSON.stringify(o, null, 1) + "\n");
  console.log(`[b4b_timeouts] ${o.amendment.limits} limits x${FACTOR} (cap ${MAX_S} s) -> ${outp}`);
  return 0;
}

function selftest() {
  const ck = [];
  const t = (name, ok) => ck.push([name, !!ok]);
  const src = { rule: "r", t1: { a: 1800, b_eval: 2126, c: 9000 }, t1s: { d: 1800 }, t2: { e: 1931 }, t3s: {},
                peak_gb: { t1: 2.1 }, split: [], unmeasured: [] };
  const o = amend(src, "in.json", "00");
  t("doubles every limit", o.t1.a === 3600 && o.t1.b_eval === 4252 && o.t1s.d === 3600 && o.t2.e === 3862);
  t("caps at 14400 s", o.t1.c === 14400);
  t("never lowers", PROGS.every(p => Object.keys(src[p]).every(k => o[p][k] >= src[p][k])));
  t("copies peak_gb, split and unmeasured", o.peak_gb.t1 === 2.1 && Array.isArray(o.split) && o.split.length === 0
    && Array.isArray(o.unmeasured) && o.unmeasured.length === 0);
  t("'rule' states the B4b rule and quotes the emitted one",
    o.rule === "B4b: min(14400, 2 x emitted), never lower (see 'amendment'); emitted limit (scripts/b4_timeouts.js): r"
    && Object.keys(o)[0] === "rule");
  t("records the amendment", o.amendment.id === "B4b" && o.amendment.limits === 5 && o.amendment.source_sha256 === "00");
  t("leaves the input unchanged", src.t1.a === 1800 && src.rule === "r");
  let threw = false;
  try { amend({ rule: "r", t1: { x: -1 } }, "i", "0"); } catch (e) { threw = /not a positive number/.test(e.message); }
  t("refuses a non-positive limit", threw);
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "b4b_"));
  const inp = path.join(d, "in.json"), outp = path.join(d, "out.json");
  fs.writeFileSync(inp, JSON.stringify(src));
  const r1 = run(inp, outp), r2 = run(inp, outp);
  t("CLI writes once, then refuses to overwrite", r1 === 0 && r2 === 2 && JSON.parse(fs.readFileSync(outp)).t1.a === 3600);
  // a CRLF checkout of the same committed bytes records the same source sha256 and writes the same output
  const lf = JSON.stringify(src, null, 1) + "\n", outs = {}, cwd = process.cwd();
  for (const [k, text] of [["lf", lf], ["crlf", lf.replace(/\n/g, "\r\n")]]) {
    fs.mkdirSync(path.join(d, k));
    fs.writeFileSync(path.join(d, k, "in.json"), text);
    try { process.chdir(path.join(d, k)); run("in.json", "out.json"); } finally { process.chdir(cwd); }
    outs[k] = fs.readFileSync(path.join(d, k, "out.json"));
  }
  const want = crypto.createHash("sha256").update(lf).digest("hex");
  t("a CRLF input gives the LF input's sha256 and the same output bytes", outs.lf.equals(outs.crlf)
    && JSON.parse(outs.crlf).amendment.source_sha256 === want);
  fs.rmSync(d, { recursive: true, force: true });
  const bad = ck.filter(c => !c[1]);
  for (const [n, ok] of ck) console.log(`  ${ok ? "PASS" : "FAIL"} ${n}`);
  console.log(`b4b_timeouts selftest ${bad.length ? "FAIL" : "PASS"} (${ck.length - bad.length}/${ck.length})`);
  return bad.length ? 1 : 0;
}

if (require.main === module) {
  if (process.argv.includes("--selftest")) process.exit(selftest());
  const inp = arg("--in"), outp = arg("--out");
  if (!inp || !outp) { console.error("usage: b4b_timeouts.js --in <emitted timeouts.json> --out <timeouts.json>"); process.exit(2); }
  process.exit(run(inp, outp));
}
module.exports = { amend, FACTOR, MAX_S };
