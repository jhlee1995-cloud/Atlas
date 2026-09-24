#!/usr/bin/env node
// scripts/b4_timeouts.js -- session-2 time limits from session-1 measurements (docs/plans/B4_INTEGRATION.md D10).
// Batch 3's lesson: a fixed 1800 s limit killed the ResNet50 Stage B at its 18th tap (results/margin_b1_vitb16/
// RELAUNCH_r2.md). Here every limit is 3x a measured wall time, scaled to the unit it guards:
//   timeout(prog, u) = clamp(ceil(3 * t_ref * max(1, (d_u / d_ref)^2) * max(1, rows_u / rows_ref)), 1800, 14400)
// t_ref = the slowest session-1 run of the same program on the same family (else the widest measured unit), d = penult
// width, rows = rows the job reads (from the unit's registry layout; fit 122500 (+16000 extras), eval 228500 (+40000
// extras), maps 33000 (+2000 held-out faults)). A raw limit above 14400 s is listed under SPLIT (the lead splits that job
// into per-target chunks before P2) and capped. peak_gb(prog) = the max over units of the measured max_rss scaled by
// width and rows (sets the parallelism P in scripts/pod_b4.sh). Jobs mirror scripts/b4_reg.py `jobs` (replay +
// confirmation).
//   node scripts/b4_timeouts.js --registry experiments/b4/models.json --timing results/instrument_check_b4s1/timing.json \
//        --out experiments/b4/timeouts.json            (Windows, before the P2 commit; refuses to overwrite)
//   node scripts/b4_timeouts.js --selftest
"use strict";
const fs = require("fs");
const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const ROWS = { fit: 138500, eval: 228500, maps: 35000 };           // fallback when a unit carries no layouts
const MIN_S = 1800, MAX_S = 14400;

function rowsOf(u, lay) {
  const L = u.layouts && u.layouts[lay];
  if (!L) return ROWS[lay];
  if (lay === "fit") return 122500 + (L.extras ? 16000 : 0);
  if (lay === "eval") return 228500 + (L.extras ? 40000 : 0);
  return 33000 + (L.holdout_faults ? 2000 : 0);
}

function jobsFor(reg) {                  // (program, unit, layout) of session 2, as scripts/b4_reg.py `jobs`
  const has = (m, r) => m.roles.includes(r), J = [];
  for (const m of reg.models) {
    if (["C", "K", "F", "F20"].some(r => has(m, r))) J.push(["t2", m.id, "fit"]);
    if (["C", "K", "F", "F20"].some(r => has(m, r))) J.push(["t1", m.id, "fit"]);
    if (["F", "F20", "R2"].some(r => has(m, r))) J.push(["t1", m.id, "eval"]);
    if (has(m, "STconf") && ["F", "F20", "R2"].some(r => has(m, r))) J.push(["t1s", m.id, "eval"]);
    if (has(m, "Sconf")) J.push(["t3s", m.id, "maps"]);
    if (has(m, "ANCHOR")) {                                          // the S2 replay
      if (!J.some(j => j[0] === "t1" && j[1] === m.id && j[2] === "fit")) J.push(["t1", m.id, "fit"]);
      J.push(["t2", m.id, "fit"]);
      if (has(m, "Sdisc")) J.push(["t3s", m.id, "maps"]);
    }
  }
  return J;
}

function compute(reg, timing) {
  const byId = Object.fromEntries(reg.models.map(m => [m.id, m]));
  const out = { rule: "clamp(ceil(3 * t_ref * max(1,(d/d_ref)^2) * max(1, rows/rows_ref)), 1800, 14400); t_ref = slowest session-1 run of the program on the same family (else the widest measured unit)",
                t1: {}, t1s: {}, t2: {}, t3s: {}, peak_gb: {}, split: [], unmeasured: [] };
  for (const [prog, id, lay] of jobsFor(reg)) {
    const u = byId[id];
    const meas = Object.entries(timing[prog] || {}).map(([k, v]) => ({ m: byId[k], ...v }))
      .filter(x => x.m && Number.isFinite(x.wall_s));
    if (!meas.length) { out.unmeasured.push(`${prog}:${id}`); continue; }
    const fam = meas.filter(x => x.m.family === u.family);
    const pool = fam.length ? fam : [meas.reduce((a, b) => (b.m.penult > a.m.penult ? b : a))];
    const ref = pool.reduce((a, b) => (b.wall_s > a.wall_s ? b : a));
    const dScale = Math.max(1, (u.penult / ref.m.penult) ** 2);
    const rScale = Math.max(1, rowsOf(u, lay) / rowsOf(ref.m, ref.layout || "fit"));
    const raw = Math.ceil(3 * ref.wall_s * dScale * rScale);
    const key = lay === "eval" && prog === "t1" ? `${id}_eval` : id;
    if (raw > MAX_S) out.split.push(`${prog}:${key} (${raw} s)`);
    out[prog][key] = Math.min(MAX_S, Math.max(MIN_S, raw));
    if (Number.isFinite(ref.max_rss_mb)) {
      const g = ref.max_rss_mb / 1024 * Math.max(1, u.penult / ref.m.penult) * rScale;
      out.peak_gb[prog] = Math.max(out.peak_gb[prog] || 0, Math.ceil(g * 10) / 10);
    }
  }
  return out;
}

function selftest() {
  const reg = { models: [
    { id: "resnet20_hub", family: "resnet", penult: 64, roles: ["D", "ANCHOR"] },
    { id: "vgg11_bn", family: "vgg", penult: 512, roles: ["Dnew"] },
    { id: "vgg16_bn", family: "vgg", penult: 512, roles: ["C"] },
    { id: "shufflenetv2_x0_5", family: "shufflenetv2", penult: 1024, roles: ["Dnew"] },
    { id: "shufflenetv2_x2_0", family: "shufflenetv2", penult: 2048, roles: ["C"] },
    { id: "resnet56_s31", family: "resnet", penult: 64, roles: ["F", "STconf"] },
    { id: "resnet44", family: "resnet", penult: 64, roles: ["C"] },
    { id: "resnet56_s31_ls10", family: "resnet", penult: 64, roles: ["K", "Kls"] },
    { id: "resnet20_s31", family: "resnet", penult: 64, roles: ["F20", "STconf"],
      layouts: { fit: { extras: true }, eval: { extras: true } } }] };
  const timing = { t1: { resnet20_hub: { wall_s: 700, max_rss_mb: 1500, layout: "fit" }, vgg11_bn: { wall_s: 1200, max_rss_mb: 3000, layout: "fit" },
                         shufflenetv2_x0_5: { wall_s: 1500, max_rss_mb: 3500, layout: "fit" } },
                   t2: { resnet20_hub: { wall_s: 150, max_rss_mb: 900 }, shufflenetv2_x0_5: { wall_s: 400, max_rss_mb: 2500 } } };
  const o = compute(reg, timing), ok = [];
  const chk = (name, c) => { ok.push(c); console.log(c ? "PASS" : "FAIL", name); };
  chk("same family, same width: resnet44 = 3*700 = 2100", o.t1.resnet44 === 2100);
  chk("floor 1800 s for a fast unit: t2 resnet44 = 3*150 = 450 -> 1800", o.t2.resnet44 === 1800);
  chk("width scaling within family: shufflenet x2.0 = 3*1500*4 = 18000 -> SPLIT, capped", o.split.some(s => s.startsWith("t1:shufflenetv2_x2_0")) && o.t1.shufflenetv2_x2_0 === MAX_S);
  chk("same width, same family: vgg16 = 3*1200 = 3600", o.t1.vgg16_bn === 3600);
  chk("eval layout scaled by rows: resnet56_s31_eval = ceil(3*700*228500/138500) = 3465", o.t1.resnet56_s31_eval === Math.ceil(3 * 700 * 228500 / 138500));
  chk("rows from the registry layout: resnet20_s31_eval (extras) = ceil(3*700*268500/138500) = 4072", o.t1.resnet20_s31_eval === Math.ceil(3 * 700 * 268500 / 138500));
  chk("the knob units get a T1 limit (K is probed by T1 in S2)", o.t1.resnet56_s31_ls10 === 2100);
  chk("family missing -> widest measured unit (t2 vgg16 from shufflenet x0.5: 3*400 = 1200 -> floor 1800)", o.t2.vgg16_bn === 1800);
  chk("unmeasured program listed (t1s)", o.unmeasured.some(s => s.startsWith("t1s:")));
  chk("peak_gb scales with width (t1 >= 3.5*2 GB)", o.peak_gb.t1 >= 6.8);
  if (ok.every(Boolean)) { console.log(`selftest PASS (${ok.length} checks)`); return 0; }
  console.log("selftest FAIL"); return 1;
}

module.exports = { compute, jobsFor, rowsOf };
if (require.main === module) {
  if (process.argv.includes("--selftest")) process.exit(selftest());
  const reg = JSON.parse(fs.readFileSync(arg("--registry"), "utf8"));
  const timing = JSON.parse(fs.readFileSync(arg("--timing"), "utf8"));
  const outp = arg("--out");
  if (fs.existsSync(outp)) { console.error(`refusing to overwrite ${outp}`); process.exit(1); }
  const o = compute(reg, timing);
  fs.writeFileSync(outp, JSON.stringify(o, null, 1) + "\n");
  console.log(`wrote ${outp}; SPLIT: ${o.split.length ? o.split.join(", ") : "none"}; unmeasured: ${o.unmeasured.length}`);
  process.exit(o.split.length ? 2 : 0);
}
