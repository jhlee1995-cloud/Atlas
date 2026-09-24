#!/usr/bin/env node
// scripts/t3s_eval.js -- the frozen evaluator of batch-4 lane S (X5, "what the pooled head cannot see"; docs/plans/
// T3S_SPATIAL.md section 5 is the text, THIS FILE is the binding rule; docs/plans/B4_INTEGRATION.md D7, D8, D14, D15).
// Owner: T3S. Node standard library + scripts/b4_stats.js only. It reads the probe records only
// (results/b4_t3s/<id><tag>/probe.json written by scripts/t3s_spatial_probe.py), never a dump.
//
//   node scripts/t3s_eval.js --phase discovery    --p1 <P1> --p-run <S1 HEAD> [--json results/b4/eval_t3s_discovery.json]
//   node scripts/t3s_eval.js --phase confirmation --p1 <P1> --p2 <P2> --p-run <S1 HEAD>,<S2 HEAD> \
//        [--json results/b4/eval_t3s.json] [--sealed results/instrument_check_b4s1/sealed.json]
//        [--replay results/instrument_check_b4s2/replay.json] [--root results/b4_t3s] [--registry experiments/b4/models.json]
//   node scripts/t3s_eval.js --check-record <probe.json>     the fields this evaluator reads (tests/test_t3s_spatial.py)
//   node scripts/t3s_eval.js --selftest                      fixtures: every claim flipped, NOT_EVALUABLE runs, a CLI run
//
// Claims (confirmation units = registry role Sconf: resnet20_s3, resnet20_s4, resnet56_s12m, resnet56_s13m; every rule
// needs >= 3 evaluable units, else NOT_EVALUABLE). F5 = paste x {gaussian_noise, defocus_blur, pixelate, jpeg_compression}
// + occluder_sq, all at a12 (11 x 11 px, 11.8% of the image); AUROC = fault vs its clean twin (same rows 8500-9499).
//   SP-1 X5's prediction: F5-mean AUROC(map_max) >= 0.90 AND every pooled score (d1, knn_l2, gap_maha, msp, maxlogit,
//        gap, energy, entropy; F5-mean of the two-sided AUROC max(a, 1 - a)) <= 0.70. SUPPORTED >= 3 units; REFUTED
//        (MAP-BLIND) if map_max <= 0.80 on >= 2 units, REFUTED (POOLED-SEES) if a pooled score >= 0.80 on >= 2 units;
//        PARTIAL 2 units; else INCONCLUSIVE.
//   SP-2 the map bundle ADDS over head + pooled geometry (frozen HEAD-ADDITIVE rule, F5 vs clean): SUPPORTED if ADDS on
//        >= 3 units, REFUTED if BOUNDED on >= 3, PARTIAL if ADDS on 2, else INCONCLUSIVE.
//   SP-3 the map bundle over head + pooled + pixel scores, 4 paste families: GEOMETRY-ADDS-OVER-PIXELS (ADDS >= 3),
//        PIXELS-SUFFICE (BOUNDED >= 3; a legitimate outcome: the controller then needs a pixel check), else MIXED.
//   SP-4 pointing: F5-mean hit >= 0.60 AND hit - chance >= 0.30. SUPPORTED >= 3, PARTIAL 2, else REFUTED.
//   SP-5 held-out soiling (confirmation-only, never seen in discovery): AUROC(map_max, soiling__a12) >= 0.85.
//        SUPPORTED >= 3, PARTIAL 2, else REFUTED.
//   SP-6 the hold case (fault argmax = clean argmax = label): F5-mean map_max >= 0.85 AND every head score (two-sided)
//        <= 0.65, over the F5 conditions with >= 50 such rows (>= 3 needed). SUPPORTED >= 3, PARTIAL 2, else REFUTED.
//   SP-7 X5 arm (a): std_half ADDS over head + pooled (F5 vs clean): as SP-2.
//   SP-8 local-vs-global typing: the map bundle ADDS over head + pooled for local a12 faults vs global s3 corruptions:
//        as SP-2.
// INFO: SP-9 learned vs random-init (discovery pairs), SP-10 collapse link, SP-11 stage-2 map (hubs), area curves,
// functional units (false alarms per 1000 clean frames, TPR at the calibrated 5%, reference bytes, CPU ms / 1000),
// Holm over the increment DeLong p-values, and every per-unit predicate on the discovery hubs.
// Known answers (D15): per unit KA-1 map-GAP identity, KA-2 pairing, KA-3 pixel hash (+ equal across units), KA-5 mask
// areas, KA-6 accuracy (trained >= 0.85), the probe's call = the frozen HEAD-ADDITIVE rule; lane: KA-4 the clean false-
// alarm rate of the calibrated map flag inside the beta-binomial 99% band (b4_stats.betaBinomBand; more than one Sconf
// failure -> lane NOT_EVALUABLE). Provenance: probe and core code sha = P1 (discovery) / P2 (confirmation, _p2); repo
// commit in --p-run; confirmation unsealed with P2; sealed meta sha = sealed.json; extraction code and registry unchanged
// P1 -> P2; this file = its frozen version. Replay (rule 6): drift of the claim statistics between the S1 and the
// _s2replay records of the anchors tags labels REPLAY-DRIFT; a claim whose margin is below the drift is NOT_EVALUABLE.
// P2 amendments (D8 b, c; freeze_P2.json "amendments" entries with "track": "T3S"): withdraw a claim to INFO, or a
// STRICTER threshold of SP-1 / SP-4 / SP-5 / SP-6; anything else is refused and the claim is NOT_EVALUABLE.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const S = require("./b4_stats.js");

const REPO = path.resolve(__dirname, "..");
const isNum = S.isNum;
const OUT_SCHEMA = "t3s_eval/1";
const PROBE = "scripts/t3s_spatial_probe.py";
const CORE = "atlas/b4_core.py";
const SELF = "scripts/t3s_eval.js";
const EXTRACTION = ["atlas/faults.py", "scripts/b4_extract.py", "experiments/b4/models.json"];
const FILES = [PROBE, SELF, CORE, "atlas/b4_collapse.py", ...EXTRACTION, "docs/plans/T3S_SPATIAL.md"];
const DEFAULTS = { root: "results/b4_t3s", registry: "experiments/b4/models.json",
  sealed: "results/instrument_check_b4s1/sealed.json", replay: "results/instrument_check_b4s2/replay.json" };
const TAG_RE = /(?:_r\d+|_p2|_s2replay)*$/;
const CALLS = ["ADDS", "BOUNDED", "INCONCLUSIVE"];

// ---------------------------------------------------------------- pre-registered rules (P1)
const F5 = ["paste__gaussian_noise__a12", "paste__defocus_blur__a12", "paste__pixelate__a12",
  "paste__jpeg_compression__a12", "occluder_sq__a12"];
const RULES = {
  probeSchema: "t3s_probe/1",
  sets: { F5, PASTE4: F5.slice(0, 4), LOCAL_A12: [...F5, "glare__a12", "dead_sq__a12"],
    GLOBAL: ["gaussian_noise", "defocus_blur", "pixelate", "jpeg_compression", "fog", "brightness", "contrast"].map(c => `global__${c}__s3`),
    SOIL: ["soiling__a12", "soiling__a25"] },
  pooled: ["d1", "knn_l2", "gap_maha", "msp", "maxlogit", "gap", "energy", "entropy"],
  head: ["msp", "maxlogit", "gap", "energy", "entropy"],
  minUnits: 3, support: 3, partial: 2,
  SP1: { map: 0.90, pooled: 0.70, refuteMap: 0.80, refutePooled: 0.80, refuteUnits: 2 },
  SP4: { hit: 0.60, overChance: 0.30 },
  SP5: { map: 0.85, cond: "soiling__a12" },
  SP6: { map: 0.85, head: 0.65, minN: 50, minConds: 3 },
  SP9: { learned: 0.10, conds: ["paste__defocus_blur__a12", "paste__jpeg_compression__a12"] },
  inc: { add: 0.01, eq: 0.02,                           // = atlas/b4_core ADD_DAUC, EQ_BOUND (the frozen rule)
    SP2: ["SP2_map_over_pooled", "F5", ["eval"]], SP3: ["SP3_map_over_pixels", "PASTE4", ["eval"]],
    SP7: ["SP7_std_over_pooled", "F5", ["eval"]], SP8: ["SP8_typing", "LOCAL_A12", "GLOBAL"] },
  ka: { alpha: 0.05, level: 0.99, maxConfFail: 1, accTrained: 0.85, accNull: 0.20 },
  replay: { rel: 1e-6, abs: 1e-9 },
};
const AMENDABLE = {   // claim -> param -> [rule block, key, +1: larger is stricter | -1: smaller is stricter]
  "SP-1": { map: ["SP1", "map", 1], pooled: ["SP1", "pooled", -1] },
  "SP-4": { hit: ["SP4", "hit", 1], overChance: ["SP4", "overChance", 1] },
  "SP-5": { map: ["SP5", "map", 1] },
  "SP-6": { map: ["SP6", "map", 1], head: ["SP6", "head", -1] },
};
const CLAIMS = ["SP-1", "SP-2", "SP-3", "SP-4", "SP-5", "SP-6", "SP-7", "SP-8"];

// ---------------------------------------------------------------- small helpers
const round = (v, d = 4) => (isNum(v) ? Math.round(v * 10 ** d) / 10 ** d : null);
const meanOf = xs => (xs.length && xs.every(isNum) ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const two = a => (isNum(a) ? Math.max(a, 1 - a) : null);
const get = (o, p) => p.split(".").reduce((x, k) => (x === null || x === undefined ? undefined : x[k]), o);
const clone = o => JSON.parse(JSON.stringify(o));
const pc = (r, c) => ((r && r.per_condition) || {})[c] || null;
const au = (r, c, k) => { const p = pc(r, c); const v = p && p.auroc ? p.auroc[k] : null; return isNum(v) ? v : null; };
const valid = (r, c) => { const p = pc(r, c); return !!(p && p.valid === true); };
const prefixMatch = (a, b) => !!a && !!b && (String(a).startsWith(String(b)) || String(b).startsWith(String(a)));
function headAdditiveCall(dauc, ci, dI, rules = RULES) {   // = atlas/b4_core.head_additive_call
  if (!isNum(dauc) || !Array.isArray(ci) || !Array.isArray(dI) || ![ci[0], ci[1], dI[0], dI[1]].every(isNum)) return "INCONCLUSIVE";
  if (dauc >= rules.inc.add && dI[0] > 0) return "ADDS";
  if (ci[1] < rules.inc.eq) return "BOUNDED";
  return "INCONCLUSIVE";
}

// ---------------------------------------------------------------- registry and records
function loadRegistry(p) {
  const j = JSON.parse(fs.readFileSync(path.resolve(REPO, p), "utf8"));
  const units = {};
  for (const m of j.models) units[m.id] = { roles: m.roles || [], trained: m.trained !== false, depth: m.depth_family || null,
    maps: !!(m.layouts && m.layouts.maps) };
  const withRole = r => j.models.filter(m => (m.roles || []).includes(r)).map(m => m.id);
  return { units, sdisc: withRole("Sdisc"), sconf: withRole("Sconf"), anchors: withRole("ANCHOR").filter(i => withRole("Sdisc").includes(i)) };
}
function loadRecords(root) {
  const out = [];
  const dir = path.resolve(REPO, root);
  if (!fs.existsSync(dir)) return out;
  for (const d of fs.readdirSync(dir).sort()) {
    const f = path.join(dir, d, "probe.json");
    if (!fs.existsSync(f)) continue;
    const tag = d.match(TAG_RE)[0], base = d.slice(0, d.length - tag.length);
    let rec = null, err = null;
    try { rec = JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { err = String(e); }
    out.push({ dir: d, base, tag, file: f, rec, err });
  }
  return out;
}
const relaunch = t => { const m = t.match(/_r(\d+)/g); return m ? Math.max(...m.map(x => +x.slice(2))) : 0; };
function selectRecords(all, reg) {
  const disc = {}, conf = {}, replay = {}, unitProblems = {};
  const addP = (id, w) => (unitProblems[id] = unitProblems[id] || []).push(w);
  for (const id of [...reg.sdisc, ...reg.sconf]) {
    const mine = all.filter(x => x.base === id);
    for (const x of mine) if (x.err || !x.rec) addP(id, `${x.dir}: unreadable probe.json`);
    const ok = mine.filter(x => x.rec && x.rec.unit === id);
    const rp = ok.filter(x => x.tag.includes("_s2replay")).sort((a, b) => relaunch(a.tag) - relaunch(b.tag));
    const d = ok.filter(x => !x.tag.includes("_s2replay") && x.rec.phase === "discovery");
    const p2 = d.filter(x => x.tag.includes("_p2"));
    const pick = (p2.length ? p2 : d).sort((a, b) => relaunch(a.tag) - relaunch(b.tag));
    if (pick.length) disc[id] = pick[pick.length - 1];
    // the S2 replay runs the P2 probe: compare it with the S1 run, or with the _p2 re-probe when the probe was fixed at P2
    const s1 = (p2.length ? p2 : d).sort((a, b) => relaunch(a.tag) - relaunch(b.tag));
    if (s1.length && rp.length) replay[id] = { s1: s1[s1.length - 1], s2: rp[rp.length - 1], same_code_rerun: p2.length > 0 };
    const c = ok.filter(x => x.rec.phase === "confirmation");
    if (c.length > 1) addP(id, `probed ${c.length} times in confirmation (touched-once, D7): ${c.map(x => x.dir).join(", ")}`);
    else if (c.length === 1) conf[id] = c[0];
  }
  return { disc, conf, replay, unitProblems };
}

// ---------------------------------------------------------------- the fields this evaluator reads (--check-record)
function schemaProblems(rec) {
  const need = ["schema", "program", "unit", "phase", "sets", "per_condition", "increments", "image_sha256_first16",
    "known_answers.map_gap_identity.pass", "known_answers.pairing.pass", "known_answers.mask_area.pass",
    "known_answers.pixel_hash.pass", "clean.acc_eval", "conformal.n_cal", "conformal.n_test", "conformal.fpr_eval.map_max",
    "conformal.curve_map_max", "collapse", "functional.false_alarms_per_1000_clean", "functional.reference_bytes_float32",
    "code.sha256", "code.core_sha256", "dumps"];
  const miss = need.filter(p => get(rec, p) === undefined);
  if (rec && rec.schema !== undefined && rec.schema !== RULES.probeSchema) miss.push(`schema ${rec.schema} != ${RULES.probeSchema}`);
  if (rec && rec.program !== undefined && rec.program !== "t3s") miss.push(`program ${rec.program} != t3s`);
  for (const c of F5) for (const p of ["valid", "auroc.map_max", "auroc.std_half", "pointing.hit", "pointing.chance", "still_right.n"])
    if (get(pc(rec, c) || {}, p) === undefined) miss.push(`per_condition.${c}.${p}`);
  for (const k of RULES.pooled) if (get(pc(rec, F5[0]) || {}, `auroc.${k}`) === undefined) miss.push(`per_condition.${F5[0]}.auroc.${k}`);
  for (const [key] of Object.values(RULES.inc).filter(Array.isArray))
    for (const p of ["skipped", "conds_pos", "conds_neg"]) if (get(rec, `increments.${key}.${p}`) === undefined) miss.push(`increments.${key}.${p}`);
  return miss;
}

// ---------------------------------------------------------------- per-unit status and predicates
function unitStatus(rec, spec, rules) {
  const why = [];
  const ka = rec.known_answers || {};
  if (!(ka.map_gap_identity && ka.map_gap_identity.pass === true)) why.push("KA-1 map-GAP identity");
  if (!(ka.pairing && ka.pairing.pass === true)) why.push("KA-2 pairing");
  if (!(ka.pixel_hash && ka.pixel_hash.pass === true)) why.push("KA-3 pixel hash");
  if (!(ka.mask_area && ka.mask_area.pass === true)) why.push("KA-5 mask areas");
  const acc = get(rec, "clean.acc_eval");
  if (spec.trained && !(isNum(acc) && acc >= rules.ka.accTrained)) why.push(`KA-6 clean accuracy ${round(acc)} < ${rules.ka.accTrained}`);
  for (const [k, v] of Object.entries(rules.sets)) if (JSON.stringify(get(rec, `sets.${k}`)) !== JSON.stringify(v)) why.push(`condition set ${k} differs`);
  const sp = schemaProblems(rec);
  if (sp.length) why.push(`record fields missing: ${sp.slice(0, 4).join(", ")}`);
  return why;
}
function predSP1(r, rules) {
  const F = rules.sets.F5;
  if (!F.every(c => valid(r, c))) return { ok: false, why: "an F5 condition is missing or invalid" };
  const map = meanOf(F.map(c => au(r, c, "map_max")));
  const per = rules.pooled.map(k => [k, meanOf(F.map(c => two(au(r, c, k))))]);
  if (!isNum(map) || !per.every(([, v]) => isNum(v))) return { ok: false, why: "AUROC missing" };
  const best = per.reduce((a, b) => (b[1] > a[1] ? b : a));
  return { ok: true, map, pooled: best[1], best_pooled: best[0], std_half: meanOf(F.map(c => two(au(r, c, "std_half")))),
    pass: map >= rules.SP1.map && best[1] <= rules.SP1.pooled, refMap: map <= rules.SP1.refuteMap,
    refPooled: best[1] >= rules.SP1.refutePooled,
    margin: Math.min(Math.abs(map - rules.SP1.map), Math.abs(best[1] - rules.SP1.pooled)) };
}
function predInc(r, which, rules) {
  const [key, posSet, neg] = rules.inc[which];
  const x = (r.increments || {})[key];
  if (!x || x.skipped !== false) return { ok: false, why: `${key} skipped` };
  const wantNeg = Array.isArray(neg) ? neg : rules.sets[neg];
  if (JSON.stringify(x.conds_pos) !== JSON.stringify(rules.sets[posSet]) || JSON.stringify(x.conds_neg) !== JSON.stringify(wantNeg))
    return { ok: false, why: `${key}: conditions differ from the registered sets` };
  if (!CALLS.includes(x.call)) return { ok: false, why: `${key}: no call` };
  const frozen = headAdditiveCall(x.dauc, x.dauc_ci, x.dI_ci, rules);
  if (frozen !== x.call) return { ok: false, why: `${key}: call ${x.call} != the frozen rule's ${frozen}` };
  const m = [isNum(x.dauc) ? x.dauc - rules.inc.add : null, (x.dauc_ci || [])[1] - rules.inc.eq, (x.dI_ci || [])[0]]
    .filter(isNum).map(Math.abs);
  return { ok: true, call: x.call, dauc: x.dauc, dauc_ci: x.dauc_ci, dI_ci: x.dI_ci, delong_p: isNum(x.delong_p) ? x.delong_p : null,
    tpr5_head: x.tpr5_head, tpr5_joint: x.tpr5_joint, n_pos: x.n_pos, n_neg: x.n_neg, margin: m.length ? Math.min(...m) : 0 };
}
function predSP4(r, rules) {
  const F = rules.sets.F5;
  if (!F.every(c => valid(r, c))) return { ok: false, why: "an F5 condition is missing or invalid" };
  const hit = meanOf(F.map(c => get(pc(r, c), "pointing.hit"))), ch = meanOf(F.map(c => get(pc(r, c), "pointing.chance")));
  if (!isNum(hit) || !isNum(ch)) return { ok: false, why: "pointing missing" };
  return { ok: true, hit, chance: ch, hit_dil: meanOf(F.map(c => get(pc(r, c), "pointing.hit_dil"))),
    pix_hit: meanOf(F.map(c => get(pc(r, c), "pointing.pix_hit"))),
    pass: hit >= rules.SP4.hit && hit - ch >= rules.SP4.overChance,
    margin: Math.min(Math.abs(hit - rules.SP4.hit), Math.abs(hit - ch - rules.SP4.overChance)) };
}
function predSP5(r, rules) {
  const c = rules.SP5.cond;
  if (!valid(r, c)) return { ok: false, why: `${c} missing or invalid` };
  const a = au(r, c, "map_max");
  if (!isNum(a)) return { ok: false, why: "AUROC missing" };
  const pooled = Math.max(...rules.pooled.map(k => two(au(r, c, k))).filter(isNum));
  return { ok: true, map: a, best_pooled_two_sided: isNum(pooled) ? pooled : null, a25: au(r, "soiling__a25", "map_max"),
    std_half: au(r, c, "std_half"), pass: a >= rules.SP5.map, margin: Math.abs(a - rules.SP5.map) };
}
function predSP6(r, rules) {
  const cs = rules.sets.F5.filter(c => valid(r, c) && get(pc(r, c), "still_right.n") >= rules.SP6.minN && get(pc(r, c), "still_right.auroc"));
  if (cs.length < rules.SP6.minConds) return { ok: false, why: `${cs.length} F5 conditions with >= ${rules.SP6.minN} still-right rows` };
  const sa = (c, k) => { const v = get(pc(r, c), `still_right.auroc.${k}`); return isNum(v) ? v : null; };
  const map = meanOf(cs.map(c => sa(c, "map_max")));
  const hs = rules.head.map(k => meanOf(cs.map(c => two(sa(c, k)))));
  if (!isNum(map) || !hs.every(isNum)) return { ok: false, why: "still-right AUROC missing" };
  const head = Math.max(...hs);
  return { ok: true, conds: cs, n_rows: cs.map(c => get(pc(r, c), "still_right.n")), map, head,
    pass: map >= rules.SP6.map && head <= rules.SP6.head,
    margin: Math.min(Math.abs(map - rules.SP6.map), Math.abs(head - rules.SP6.head)) };
}
const PRED = { "SP-1": predSP1, "SP-2": (r, R) => predInc(r, "SP2", R), "SP-3": (r, R) => predInc(r, "SP3", R),
  "SP-4": predSP4, "SP-5": predSP5, "SP-6": predSP6, "SP-7": (r, R) => predInc(r, "SP7", R), "SP-8": (r, R) => predInc(r, "SP8", R) };

// ---------------------------------------------------------------- labels
function label(claim, per, rules) {
  const ev = Object.values(per).filter(p => p.ok);
  const n = ev.length;
  if (n < rules.minUnits) return `NOT_EVALUABLE (${n} evaluable confirmation units < ${rules.minUnits})`;
  const cnt = f => ev.filter(f).length;
  if (claim === "SP-1") {
    if (cnt(p => p.pass) >= rules.support) return "SUPPORTED";
    if (cnt(p => p.refMap) >= rules.SP1.refuteUnits) return "REFUTED (MAP-BLIND)";
    if (cnt(p => p.refPooled) >= rules.SP1.refuteUnits) return "REFUTED (POOLED-SEES)";
    return cnt(p => p.pass) === rules.partial ? "PARTIAL" : "INCONCLUSIVE";
  }
  if (["SP-2", "SP-3", "SP-7", "SP-8"].includes(claim)) {
    const nm = claim === "SP-3" ? { sup: "GEOMETRY-ADDS-OVER-PIXELS", ref: "PIXELS-SUFFICE", part: "MIXED", none: "MIXED" }
      : { sup: "SUPPORTED", ref: "REFUTED", part: "PARTIAL", none: "INCONCLUSIVE" };
    const adds = cnt(p => p.call === "ADDS"), bounded = cnt(p => p.call === "BOUNDED");
    if (adds >= rules.support) return nm.sup;
    if (bounded >= rules.support) return nm.ref;
    return adds === rules.partial ? nm.part : nm.none;
  }
  const k = cnt(p => p.pass);                         // SP-4, SP-5, SP-6
  return k >= rules.support ? "SUPPORTED" : k === rules.partial ? "PARTIAL" : "REFUTED";
}

// ---------------------------------------------------------------- P2 amendments (D8 b, c)
function applyAmendments(list) {
  const rules = clone(RULES), applied = [], refused = [], withdrawn = {};
  for (const a of list || []) {
    if (!a || a.track !== "T3S") continue;
    if (!CLAIMS.includes(a.claim)) { refused.push({ ...a, why: "unknown claim" }); continue; }
    if (a.action === "withdraw") { withdrawn[a.claim] = String(a.reason || "no reason given"); applied.push(a); continue; }
    const spec = a.action === "threshold" ? (AMENDABLE[a.claim] || {})[a.param] : null;
    if (!spec || !isNum(a.value)) { refused.push({ ...a, why: "only withdrawals and SP-1/4/5/6 thresholds are amendable" }); continue; }
    const [blk, key, dir] = spec, old = rules[blk][key];
    if (dir * (a.value - old) < 0) { refused.push({ ...a, why: `not stricter than the P1 value ${old}` }); continue; }
    rules[blk][key] = a.value;
    applied.push({ ...a, p1_value: old });
  }
  return { rules, applied, refused, withdrawn };
}
function amendmentsAt(p2) {
  if (!p2) return [];
  const b = S.gitShow(p2, "experiments/b4/freeze_P2.json", REPO);
  if (!b) return [];
  try { return (JSON.parse(b.toString("utf8")).amendments || []).filter(a => a && a.track === "T3S"); } catch (e) { return []; }
}

// ---------------------------------------------------------------- provenance (D7, D8)
function realProvenance(opts, sel) {
  const problems = [], unitProblems = {};
  const addU = (id, w) => (unitProblems[id] = unitProblems[id] || []).push(w);
  if (!opts.p1) problems.push("no --p1: a dry run (every claim NOT_EVALUABLE)");
  const rec = S.provenance({ repo: REPO, p1: opts.p1, p2: opts.p2, pRun: opts.pRun, files: FILES });
  if (!opts.p1) return { record: rec, problems, unitProblems };
  if (!rec.p1_ancestor_of_head) problems.push("P1 is not an ancestor of HEAD");
  if (opts.phase === "confirmation" && !opts.p2) problems.push("the confirmation phase needs --p2");
  if (opts.p2 && !rec.p2_ancestor_of_head) problems.push("P2 is not an ancestor of HEAD");
  if (opts.p2 && !S.gitIsAncestor(opts.p1, opts.p2, REPO)) problems.push("P1 is not an ancestor of P2");
  if (!opts.pRun.length) problems.push("no --p-run: the session commits are not checked");
  if (opts.p2) for (const f of EXTRACTION) if (rec.files[f].at_p1 !== rec.files[f].at_p2)
    problems.push(`${f} changed between P1 and P2 (extraction instrument: a new P1' and re-extraction are needed, D7)`);
  const self = rec.files[SELF];
  if (self.worktree !== (opts.p2 ? self.at_p2 : self.at_p1)) problems.push(`${SELF} differs from its frozen version`);
  const at = (f, useP2) => (useP2 ? rec.files[f].at_p2 : rec.files[f].at_p1);
  const probeFixed = !!opts.p2 && rec.files[PROBE].at_p1 !== rec.files[PROBE].at_p2;
  const check = (id, x, useP2) => {
    const code = x.rec.code || {};
    if (!at(PROBE, useP2) || code.sha256 !== at(PROBE, useP2)) addU(id, `${x.dir}: probe code != ${useP2 ? "P2" : "P1"}`);
    if (!at(CORE, useP2) || code.core_sha256 !== at(CORE, useP2)) addU(id, `${x.dir}: atlas/b4_core.py != ${useP2 ? "P2" : "P1"}`);
    if (opts.pRun.length && !opts.pRun.some(c => prefixMatch(code.repo_commit, c))) addU(id, `${x.dir}: repo_commit not in --p-run`);
  };
  for (const [id, x] of Object.entries(sel.disc)) {
    const isP2 = x.tag.includes("_p2");
    if (probeFixed && !isP2) addU(id, "the probe was fixed at P2: this discovery unit needs its _p2 re-probe (D7)");
    check(id, x, isP2);
  }
  for (const [id, x] of Object.entries(sel.conf)) {
    check(id, x, true);
    if (!prefixMatch(get(x.rec, "env.unseal"), opts.p2)) addU(id, `${x.dir}: not unsealed with P2 (env.unseal ${get(x.rec, "env.unseal")})`);
  }
  return { record: rec, problems, unitProblems };
}
function sealCheck(opts, sel) {
  const unitProblems = {}, problems = [];
  if (opts.phase !== "confirmation") return { problems, unitProblems };
  const p = path.resolve(REPO, opts.sealed);
  if (!fs.existsSync(p)) { problems.push(`seal manifest ${opts.sealed} absent`); return { problems, unitProblems }; }
  const ent = JSON.parse(fs.readFileSync(p, "utf8")).entries || [];
  for (const [id, x] of Object.entries(sel.conf)) {
    const e = ent.find(v => v.unit === id && v.layout === "maps");
    const got = get(x.rec, "dumps.0.meta_sha256");
    if (!e || e.status !== "SEALED" || e.meta_sha256 !== got)
      (unitProblems[id] = unitProblems[id] || []).push(`maps dump meta sha ${String(got).slice(0, 12)} != sealed.json`);
  }
  return { problems, unitProblems };
}

// ---------------------------------------------------------------- replay (rule 6, D15)
const IGNORE = new Set(["timing_s", "max_rss_mb", "env", "code", "created", "created_utc", "host", "versions", "out", "tag", "unseal"]);
function claimLeaves(rec) {   // the statistics the claims read, flattened
  const out = {};
  const walk = (o, p) => {
    if (o === null || o === undefined) return;
    if (typeof o === "number") { out[p] = o; return; }
    if (Array.isArray(o)) { o.forEach((v, i) => walk(v, `${p}[${i}]`)); return; }
    if (typeof o === "object") for (const [k, v] of Object.entries(o)) if (!IGNORE.has(k)) walk(v, p ? `${p}.${k}` : k);
  };
  for (const [c, v] of Object.entries(rec.per_condition || {})) {
    walk(v.auroc, `per_condition.${c}.auroc`); walk(v.pointing, `per_condition.${c}.pointing`);
    walk((v.still_right || {}).auroc, `per_condition.${c}.still_right.auroc`);
  }
  for (const [k, v] of Object.entries(rec.increments || {})) for (const f of ["dauc", "dauc_ci", "dI_ci"]) walk(v[f], `increments.${k}.${f}`);
  return out;
}
function replayCheck(opts, sel, reg) {
  const rep = { units: {}, drift: 0, status: "NOT_RUN" };
  for (const id of reg.anchors) {
    const pr = sel.replay[id];
    if (!pr) { rep.units[id] = { status: "MISSING" }; continue; }
    const a = claimLeaves(pr.s1.rec), b = claimLeaves(pr.s2.rec);
    let maxAbs = 0, nBad = 0;
    const first = [];
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const x = a[k], y = b[k];
      if (!isNum(x) || !isNum(y)) { nBad++; if (first.length < 5) first.push(`${k}: one side missing`); maxAbs = Math.max(maxAbs, 1); continue; }
      const d = Math.abs(x - y);
      if (d > Math.max(opts.rules.replay.abs, opts.rules.replay.rel * Math.max(Math.abs(x), Math.abs(y)))) {
        nBad++; maxAbs = Math.max(maxAbs, d); if (first.length < 5) first.push(`${k}: ${x} vs ${y}`);
      }
    }
    rep.units[id] = { status: nBad ? "DRIFT" : "PASS", n_diff: nBad, max_abs: maxAbs, first, s1: pr.s1.dir, s2: pr.s2.dir,
      note: pr.same_code_rerun ? "probe fixed at P2: compared with the _p2 re-probe (same code, both in S2)" : "S1 run vs S2 replay" };
    rep.drift = Math.max(rep.drift, maxAbs);
  }
  const lead = path.resolve(REPO, opts.replay);
  if (fs.existsSync(lead)) {
    try {
      const j = JSON.parse(fs.readFileSync(lead, "utf8"));
      rep.lead = Object.fromEntries(Object.entries(j.units || {}).filter(([k]) => k.startsWith("t3s:")).map(([k, v]) => [k, v.status]));
    } catch (e) { rep.lead = { error: String(e) }; }
  }
  const st = Object.values(rep.units).map(u => u.status);
  rep.status = st.includes("DRIFT") ? "DRIFT" : st.length && st.every(s => s === "PASS") ? "PASS" : "INCOMPLETE";
  return rep;
}

// ---------------------------------------------------------------- INFO blocks
function infoBlocks(reg, disc, conf, rules) {
  const info = {};
  const bj = r => (r ? meanOf(rules.SP9.conds.map(c => au(r, c, "map_max"))) : null);
  const pairs = {};
  for (const id of reg.sdisc) { const u = reg.units[id]; (pairs[u.depth] = pairs[u.depth] || {})[u.trained ? "trained" : "null"] = id; }
  const d9 = Object.fromEntries(Object.entries(pairs).map(([dep, p]) => {
    const t = bj((disc[p.trained] || {}).rec), n = bj((disc[p.null] || {}).rec);
    return [dep, { trained: p.trained, null: p.null, map_max_trained: round(t), map_max_null: round(n), diff: isNum(t) && isNum(n) ? round(t - n) : null }];
  }));
  const ds = Object.values(d9).map(v => v.diff);
  info["SP-9 learned vs random-init (INFO)"] = { pairs: d9, label: ds.length >= 2 && ds.every(v => isNum(v) && v >= rules.SP9.learned)
    ? "LEARNED" : ds.some(v => !isNum(v)) ? "NOT_EVALUABLE (a pair is missing)" : "ARCHITECTURAL-OR-MIXED" };
  const trained = [...Object.entries(disc).filter(([id]) => reg.units[id].trained), ...Object.entries(conf)];
  const xs = [], ys = [], ids = [];
  for (const [id, x] of trained) {
    const p = predSP1(x.rec, rules), l = get(x.rec, "collapse.log10_nc1_ref");
    if (p.ok && isNum(l)) { xs.push(-l); ys.push(p.map - p.pooled); ids.push(id); }
  }
  info["SP-10 collapse link (INFO)"] = { units: ids, rho_neg_log10_nc1_vs_map_advantage: round(S.spearman(xs, ys)), n: ids.length };
  info["SP-11 stage-2 map, hubs (INFO)"] = Object.fromEntries(Object.entries(disc).filter(([id]) => reg.units[id].trained).map(([id, x]) => [id, {
    s2_map_max: round(meanOf(rules.sets.F5.map(c => au(x.rec, c, "s2_map_max")))),
    map_max: round(meanOf(rules.sets.F5.map(c => au(x.rec, c, "map_max")))),
    s2_hit: round(meanOf(rules.sets.F5.map(c => get(pc(x.rec, c), "pointing.s2_hit")))),
    s2_chance: round(meanOf(rules.sets.F5.map(c => get(pc(x.rec, c), "pointing.s2_chance")))) }]));
  const kinds = ["paste__gaussian_noise", "paste__defocus_blur", "paste__pixelate", "paste__jpeg_compression", "occluder_sq",
    "glare", "dead_sq", "soiling"];
  const curve = r => Object.fromEntries(kinds.map(k => [k, Object.fromEntries(["a06", "a12", "a25"].filter(a => pc(r, `${k}__${a}`))
    .map(a => { const c = `${k}__${a}`; return [a, { map_max: round(au(r, c, "map_max")), std_half: round(au(r, c, "std_half")),
      best_pooled_two_sided: round(Math.max(...rules.pooled.map(q => two(au(r, c, q))).filter(isNum))),
      pix_padim: round(au(r, c, "pix_padim")), valid: valid(r, c) }]; }))]));
  const fun = r => ({ false_alarms_per_1000_clean: get(r, "functional.false_alarms_per_1000_clean"),
    tpr_at_cal05_F5: Object.fromEntries(["map_max", "std_half", "d1", "knn_l2", "msp", "pix_padim"].map(k =>
      [k, round(meanOf(rules.sets.F5.map(c => get(pc(r, c), `tpr_cal05.${k}`))))])),
    reference_bytes_float32: get(r, "functional.reference_bytes_float32"), cpu_ms_per_1000: get(r, "functional.timing_s") });
  const units = { ...Object.fromEntries(Object.entries(disc).map(([k, v]) => [k, v])), ...conf };
  info.area_curve = Object.fromEntries(Object.entries(units).map(([id, x]) => [id, curve(x.rec)]));
  info.functional = Object.fromEntries(Object.entries(units).map(([id, x]) => [id, fun(x.rec)]));
  const ps = [], lab = [];
  for (const [id, x] of Object.entries(conf)) for (const w of ["SP2", "SP3", "SP7", "SP8"]) {
    const p = predInc(x.rec, w, rules);
    if (p.ok) { ps.push(p.delong_p); lab.push(`${w}:${id}`); }
  }
  const h = S.holm(ps);
  info["Holm over the increment DeLong p (INFO)"] = Object.fromEntries(lab.map((k, i) => [k, { p: ps[i], holm: h.adjusted[i], reject_05: h.reject[i] }]));
  return info;
}
function predicatesTable(units, rules) {
  return Object.fromEntries(Object.entries(units).map(([id, x]) => [id,
    Object.fromEntries(CLAIMS.map(c => { const p = PRED[c](x.rec, rules); return [c, roundDeep(p)]; }))]));
}
function roundDeep(o) {
  if (isNum(o)) return round(o, 6);
  if (Array.isArray(o)) return o.map(roundDeep);
  if (o && typeof o === "object") return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, roundDeep(v)]));
  return o;
}

// ---------------------------------------------------------------- the evaluation
function evaluate(opts, deps = {}) {
  opts = { root: DEFAULTS.root, registry: DEFAULTS.registry, sealed: DEFAULTS.sealed, replay: DEFAULTS.replay, pRun: [], ...opts };
  if (!["discovery", "confirmation"].includes(opts.phase)) throw new Error(`--phase discovery|confirmation, got ${opts.phase}`);
  const reg = deps.registry || loadRegistry(opts.registry);
  const am = applyAmendments(opts.amendments !== undefined ? opts.amendments : amendmentsAt(opts.p2));
  const rules = am.rules;
  opts.rules = rules;
  const sel = selectRecords(loadRecords(opts.root), reg);
  const prov = (deps.provenance || realProvenance)(opts, sel);
  const seal = (deps.seal || sealCheck)(opts, sel);
  const out = { schema: OUT_SCHEMA, phase: opts.phase, created_utc: new Date().toISOString(), rules_p1: RULES,
    amendments: { applied: am.applied, refused: am.refused, withdrawn: am.withdrawn },
    provenance: { record: prov.record, problems: prov.problems, seal_problems: seal.problems }, units: {}, lane: {}, claims: {} };
  const unitWhy = id => [...(sel.unitProblems[id] || []), ...(prov.unitProblems[id] || []), ...(seal.unitProblems[id] || [])];
  const usable = (set, ids) => {
    const ok = {};
    for (const id of ids) {
      const x = set[id];
      const why = x ? [...unitWhy(id), ...unitStatus(x.rec, reg.units[id], rules)] : ["no record", ...unitWhy(id)];
      out.units[id] = { role: reg.sconf.includes(id) ? "Sconf" : "Sdisc", trained: reg.units[id].trained,
        record: x ? x.dir : null, evaluable: !why.length, why };
      if (!why.length) ok[id] = x;
    }
    return ok;
  };
  const disc = usable(sel.disc, reg.sdisc);
  const conf = opts.phase === "confirmation" ? usable(sel.conf, reg.sconf) : {};
  // lane-level known answers: KA-4 conformal band, KA-3 pixel hashes equal across every unit read
  const units = opts.phase === "confirmation" ? reg.sconf : reg.sdisc;
  const pool = opts.phase === "confirmation" ? sel.conf : sel.disc;
  const band = {}, curves = {};
  let confFail = 0;
  for (const id of units) {
    const x = pool[id];
    if (!x) continue;
    const n = get(x.rec, "conformal.n_cal"), t = get(x.rec, "conformal.n_test"), f = get(x.rec, "conformal.fpr_eval.map_max");
    const b = isNum(n) && isNum(t) ? S.betaBinomBand(n, t, rules.ka.alpha, rules.ka.level) : null;
    const pass = !!b && isNum(f) && f >= b.lo && f <= b.hi;
    band[id] = { fpr: f, band: b ? [b.lo, b.hi] : null, pass };
    if (!pass) confFail++;
    const cv = isNum(n) && isNum(t) ? S.betaBinomCurve(n, t, (get(x.rec, "conformal.curve_map_max") || []).map(v => v.alpha), rules.ka.level) : [];
    curves[id] = (get(x.rec, "conformal.curve_map_max") || []).map((v, i) => ({ alpha: v.alpha, fpr: v.fpr, band: cv[i] ? [cv[i].lo, cv[i].hi] : null,
      inside: !!cv[i] && isNum(v.fpr) && v.fpr >= cv[i].lo && v.fpr <= cv[i].hi }));
  }
  const hashes = {};
  for (const x of [...Object.values(sel.disc), ...Object.values(sel.conf)])
    for (const [s, h] of Object.entries(x.rec.image_sha256_first16 || {})) (hashes[s] = hashes[s] || new Set()).add(h);
  const hashBad = Object.entries(hashes).filter(([, v]) => v.size > 1).map(([s]) => s);
  const laneWhy = [...prov.problems, ...seal.problems];
  if (confFail > rules.ka.maxConfFail) laneWhy.push(`KA-4: the calibrated clean false-alarm rate is outside the 99% band on ${confFail} units`);
  if (hashBad.length) laneWhy.push(`KA-3: pixels differ across units for ${hashBad.slice(0, 5).join(", ")}`);
  out.known_answers = { "KA-4 conformal band (map_max, alpha 0.05)": band, "KA-4 rule-5 curve": curves,
    "KA-3 cross-unit pixel hashes": { splits: Object.keys(hashes).length, differing: hashBad } };
  out.lane = { status: laneWhy.length ? "NOT_EVALUABLE" : "OK", why: laneWhy };
  const rp = replayCheck(opts, sel, reg);
  out.replay = rp;
  out.discovery_info = { note: "the frozen predicates on the discovery units (INFO; the rules were registered at P1)",
    predicates: predicatesTable(Object.fromEntries(Object.entries(disc).filter(([id]) => reg.units[id].trained)), rules) };
  out.info = infoBlocks(reg, disc, conf, rules);
  if (opts.phase === "discovery") {
    out.claims = Object.fromEntries(CLAIMS.map(c => [c, { label: "not evaluated in the discovery phase" }]));
    return out;
  }
  for (const c of CLAIMS) {
    const per = Object.fromEntries(Object.entries(conf).map(([id, x]) => [id, PRED[c](x.rec, rules)]));
    const ev = Object.values(per).filter(p => p.ok);
    const margin = ev.length ? Math.min(...ev.map(p => p.margin)) : null;
    let lab = laneWhy.length ? `NOT_EVALUABLE (${laneWhy[0]})` : label(c, per, rules);
    if (!laneWhy.length && rp.status === "DRIFT") lab = isNum(margin) && margin < rp.drift ? `NOT_EVALUABLE (REPLAY-DRIFT ${round(rp.drift, 6)} > margin ${round(margin, 6)})` : `${lab} (REPLAY-DRIFT)`;
    if (am.withdrawn[c]) lab = `INFO (withdrawn at P2: ${am.withdrawn[c]}; would read ${lab})`;
    if (am.refused.some(a => a.claim === c)) lab = `NOT_EVALUABLE (refused P2 amendment)`;
    out.claims[c] = { label: lab, n_evaluable: ev.length, margin: round(margin, 6), per_unit: roundDeep(per) };
  }
  return out;
}

// ---------------------------------------------------------------- output
function summary(out) {
  const lines = [`[t3s_eval] ${out.phase}: lane ${out.lane.status}${out.lane.why.length ? " (" + out.lane.why.join("; ") + ")" : ""}`];
  for (const [id, u] of Object.entries(out.units)) lines.push(`  ${id.padEnd(16)} ${u.role} ${u.evaluable ? "evaluable" : "NOT evaluable: " + u.why.join("; ")}`);
  for (const [c, v] of Object.entries(out.claims)) lines.push(`  ${c}: ${v.label}`);
  for (const [k, v] of Object.entries(out.info)) if (v && v.label) lines.push(`  ${k}: ${v.label}`);
  lines.push(`  replay: ${out.replay.status}`);
  return lines.join("\n");
}
function writeOnce(p, obj) {
  const f = path.resolve(p);
  if (fs.existsSync(f)) { console.error(`[t3s_eval] ${p} exists: results are append-only`); return false; }
  fs.mkdirSync(path.dirname(f), { recursive: true });
  fs.writeFileSync(f, JSON.stringify(obj, null, 1) + "\n");
  return true;
}

// ---------------------------------------------------------------- self-test: fixtures
const INC_VALUES = { ADDS: { dauc: 0.05, dauc_ci: [0.03, 0.07], dI_ci: [0.01, 0.03], delong_p: 1e-6 },
  BOUNDED: { dauc: 0.002, dauc_ci: [-0.004, 0.008], dI_ci: [-0.001, 0.002], delong_p: 0.4 },
  INCONCLUSIVE: { dauc: 0.012, dauc_ci: [-0.001, 0.03], dI_ci: [-0.002, 0.01], delong_p: 0.1 } };
function fixture(id, o = {}) {
  const phase = o.phase || "confirmation", soil = o.soil !== undefined ? o.soil : phase === "confirmation";
  const local = ["gaussian_noise", "defocus_blur", "pixelate", "jpeg_compression"].flatMap(c => ["a06", "a12", "a25"].map(a => `paste__${c}__${a}`))
    .concat(["occluder_sq__a06", "occluder_sq__a12", "occluder_sq__a25", "glare__a12", "glare__a25", "dead_sq__a06", "dead_sq__a12"])
    .concat(soil ? RULES.sets.SOIL : []);
  const per = {};
  for (const c of [...local, ...RULES.sets.GLOBAL]) {
    const f5 = F5.includes(c), g = c.startsWith("global__");
    const mapA = f5 ? (o.map !== undefined ? o.map : 0.95) : c === "soiling__a12" ? (o.soilMap !== undefined ? o.soilMap : 0.92) : 0.9;
    const pooledA = f5 && o.pooled !== undefined ? o.pooled : 0.6;
    const auroc = { map_max: mapA, map_top: mapA, map_area: 0.8, map_gini: 0.7, std_half: 0.8, d1: pooledA, knn_l2: 0.6, gap_maha: 0.62,
      msp: 0.55, maxlogit: 0.55, gap: 0.55, energy: 0.55, entropy: 0.45, pix_padim: 0.9, pix_sat: 0.5, pix_lap_dev: 0.6,
      s2_map_max: 0.93, s2_map_top: 0.9 };
    if (o.rand) for (const k of ["map_max", "map_top"]) auroc[k] = 0.6;
    const headA = o.stillHead !== undefined ? o.stillHead : 0.55;
    per[c] = { split: g ? c : `fault__${c}`, kind: g ? "global" : c.split("__")[0], n: 1000, auroc, acc: 0.9, argmax_unchanged: 0.95,
      valid: !(o.invalid || []).includes(c), tpr_cal05: { map_max: 0.8, std_half: 0.4, d1: 0.1, knn_l2: 0.1, msp: 0.08, pix_padim: 0.7 },
      still_right: { n: o.stillN !== undefined ? o.stillN : 700, auroc: { map_max: o.stillMap !== undefined ? o.stillMap : 0.9, map_top: 0.9,
        map_area: 0.8, map_gini: 0.7, std_half: 0.8, d1: 0.55, knn_l2: 0.55, gap_maha: 0.55, msp: headA, maxlogit: headA, gap: headA,
        energy: headA, entropy: 1 - headA } } };
    if (!g) per[c].pointing = { hit: o.hit !== undefined ? o.hit : 0.8, chance: 0.19, hit_dil: 0.95, chance_dil: 0.44, pix_hit: 0.9,
      s2_hit: 0.7, s2_chance: 0.1 };
  }
  const inc = (key, pos, neg, call) => ({ conds_pos: RULES.sets[pos], conds_neg: Array.isArray(neg) ? neg : RULES.sets[neg],
    base: ["head_T1"], bundle: ["map_max"], skipped: false, degenerate: null, call, n_pos: 5000, n_neg: 1000, auc_head: 0.6,
    tpr5_head: 0.1, tpr5_joint: 0.6, ...INC_VALUES[call] });
  const hashes = { ref: "r".repeat(16), cal: "c".repeat(16), eval: "e".repeat(16),
    ...Object.fromEntries(Object.values(per).map(v => [v.split, (o.hashSalt || "") + v.split.slice(0, 16 - (o.hashSalt || "").length)])) };
  return {
    schema: RULES.probeSchema, program: "t3s", unit: id, phase, layout: null, tag: o.tag || "", sets: clone(RULES.sets),
    per_condition: per, image_sha256_first16: hashes,
    increments: { SP2_map_over_pooled: inc("SP2", "F5", ["eval"], o.sp2 || "ADDS"), SP3_map_over_pixels: inc("SP3", "PASTE4", ["eval"], o.sp3 || "ADDS"),
      SP7_std_over_pooled: inc("SP7", "F5", ["eval"], o.sp7 || "ADDS"), SP8_typing: inc("SP8", "LOCAL_A12", "GLOBAL", o.sp8 || "ADDS"),
      INFO_soiling_map_over_pooled: soil ? inc("x", "SOIL", ["eval"], "ADDS") : { skipped: true, conds_pos: RULES.sets.SOIL, conds_neg: ["eval"] } },
    known_answers: { map_gap_identity: { pass: o.ka1 !== false }, pairing: { pass: true }, mask_area: { pass: true }, pixel_hash: { pass: true } },
    clean: { acc_cal: 0.93, acc_eval: o.acc !== undefined ? o.acc : 0.93 },
    conformal: { alpha: 0.05, n_cal: 1000, n_test: 1000, fpr_eval: { map_max: o.fpr !== undefined ? o.fpr : 0.05 },
      curve_map_max: [0.02, 0.03, 0.04, 0.05].map(a => ({ alpha: a, fpr: o.fpr !== undefined ? o.fpr : a })) },
    collapse: { log10_nc1_ref: o.nc !== undefined ? o.nc : -1.2 },
    functional: { false_alarms_per_1000_clean: { map_max: 50 }, reference_bytes_float32: { map_padim: 1064960 },
      timing_s: { map_max_ms_per_1000: 5 } },
    code: { sha256: "0".repeat(64), core_sha256: "1".repeat(64), repo_commit: "f".repeat(40) },
    env: { unseal: phase === "confirmation" ? "p2fixture" : null },
    dumps: [{ path: `results/b4${phase === "confirmation" ? "c" : "d"}_${id}_maps/dump`, meta_sha256: `meta_${id}`, sealed: phase === "confirmation", splits_read: [] }],
    timing_s: { total: 60 }, max_rss_mb: 900 };
}
function writeFixtures(root, recs) {
  fs.rmSync(root, { recursive: true, force: true });
  for (const [dir, rec] of Object.entries(recs)) {
    fs.mkdirSync(path.join(root, dir), { recursive: true });
    fs.writeFileSync(path.join(root, dir, "probe.json"), JSON.stringify(rec));
  }
}
function selftest() {
  const results = [];
  const ck = (name, c, detail = "") => { results.push(!!c); console.log(`${c ? "PASS" : "FAIL"} ${name}${c ? "" : "  " + JSON.stringify(detail).slice(0, 400)}`); };
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "t3s_eval_"));
  try {
    const reg = loadRegistry(DEFAULTS.registry);
    ck("registry: 4 Sdisc (2 trained hubs + 2 random-init nulls), 4 Sconf, both anchors are Sdisc, all with maps",
      reg.sdisc.length === 4 && reg.sconf.length === 4 && reg.anchors.length === 2
      && reg.sdisc.filter(i => reg.units[i].trained).length === 2 && [...reg.sdisc, ...reg.sconf].every(i => reg.units[i].maps), reg);
    ck("frozen HEAD-ADDITIVE port", headAdditiveCall(0.05, [0.03, 0.07], [0.01, 0.03]) === "ADDS"
      && headAdditiveCall(0.002, [-0.004, 0.008], [-0.001, 0.002]) === "BOUNDED" && headAdditiveCall(0.012, [-0.001, 0.03], [-0.002, 0.01]) === "INCONCLUSIVE"
      && headAdditiveCall(null, null, null) === "INCONCLUSIVE");
    ck("two-sided AUROC", two(0.2) === 0.8 && two(0.7) === 0.7 && two(null) === null);
    const hubs = reg.sdisc.filter(i => reg.units[i].trained), nulls = reg.sdisc.filter(i => !reg.units[i].trained);
    const base = (mod = {}, cmod = {}) => {
      const recs = {};
      for (const id of hubs) recs[id] = fixture(id, { phase: "discovery", ...(mod[id] || {}) });
      for (const id of nulls) recs[id] = fixture(id, { phase: "discovery", rand: true, acc: 0.1, ...(mod[id] || {}) });
      for (const id of reg.sconf) recs[id] = fixture(id, { ...(cmod.all || {}), ...(cmod[id] || {}) });
      return recs;
    };
    const stub = () => ({ record: { stub: true }, problems: [], unitProblems: {} });
    const sealed = path.join(tmp, "sealed.json");
    fs.writeFileSync(sealed, JSON.stringify({ entries: reg.sconf.map(u => ({ unit: u, layout: "maps", status: "SEALED", meta_sha256: `meta_${u}` })) }));
    let n = 0;
    const run = (recs, extra = {}, deps = {}) => {
      const root = path.join(tmp, `run${n++}`, "b4_t3s");
      writeFixtures(root, recs);
      return evaluate({ phase: "confirmation", root, sealed, replay: path.join(tmp, "none.json"), p2: "p2fixture", amendments: [], ...extra },
        { provenance: stub, registry: reg, ...deps });
    };
    // ---- the all-supported fixture, read from files (a non-dry run of the rules) ----
    const all = run(base());
    const want = { "SP-1": "SUPPORTED", "SP-2": "SUPPORTED", "SP-3": "GEOMETRY-ADDS-OVER-PIXELS", "SP-4": "SUPPORTED", "SP-5": "SUPPORTED",
      "SP-6": "SUPPORTED", "SP-7": "SUPPORTED", "SP-8": "SUPPORTED" };
    ck("all-supported fixture: every claim at its SUPPORTED label", CLAIMS.every(c => all.claims[c].label === want[c]),
      Object.fromEntries(CLAIMS.map(c => [c, all.claims[c].label])));
    ck("lane OK, 8 units evaluable, SP-9 LEARNED, Holm over 16 p-values", all.lane.status === "OK"
      && Object.values(all.units).every(u => u.evaluable) && all.info["SP-9 learned vs random-init (INFO)"].label === "LEARNED"
      && Object.keys(all.info["Holm over the increment DeLong p (INFO)"]).length === 16, all.lane);
    ck("check-record: a fixture record carries every field read", schemaProblems(fixture("resnet20_s3")).length === 0, schemaProblems(fixture("resnet20_s3")));
    // ---- every claim flipped ----
    const two_ = reg.sconf.slice(0, 2), three = reg.sconf.slice(0, 3);
    const on = (ids, o) => Object.fromEntries(ids.map(i => [i, o]));
    const flips = [
      ["SP-1", on(two_, { map: 0.78 }), "REFUTED (MAP-BLIND)"],
      ["SP-1", on(two_, { pooled: 0.85 }), "REFUTED (POOLED-SEES)"],
      ["SP-1", on(two_, { pooled: 0.75 }), "PARTIAL"],
      ["SP-1", on([reg.sconf[0]], { pooled: 0.25 }), "SUPPORTED"],           // two-sided: 0.25 -> 0.75 fails 1 unit only
      ["SP-1", on(two_, { pooled: 0.15 }), "REFUTED (POOLED-SEES)"],         // an AUROC of 0.15 is a usable flipped detector
      ["SP-2", on(three, { sp2: "BOUNDED" }), "REFUTED"],
      ["SP-2", on(two_, { sp2: "INCONCLUSIVE" }), "PARTIAL"],
      ["SP-2", on(three, { sp2: "INCONCLUSIVE" }), "INCONCLUSIVE"],
      ["SP-3", on(three, { sp3: "BOUNDED" }), "PIXELS-SUFFICE"],
      ["SP-3", on(two_, { sp3: "BOUNDED" }), "MIXED"],
      ["SP-4", on(three, { hit: 0.5 }), "REFUTED"],
      ["SP-4", on(two_, { hit: 0.45 }), "PARTIAL"],
      ["SP-5", on(three, { soilMap: 0.7 }), "REFUTED"],
      ["SP-6", on(three, { stillHead: 0.7 }), "REFUTED"],
      ["SP-6", on(three, { stillMap: 0.8 }), "REFUTED"],
      ["SP-7", on(three, { sp7: "BOUNDED" }), "REFUTED"],
      ["SP-8", on(three, { sp8: "BOUNDED" }), "REFUTED"],
    ];
    for (const [c, cm, lab] of flips) {
      const r = run(base({}, cm));
      ck(`flip ${c} -> ${lab}`, r.claims[c].label === lab, { got: r.claims[c].label, per: Object.fromEntries(Object.entries(r.claims[c].per_unit).map(([k, v]) => [k, v.pass !== undefined ? v.pass : v.call])) });
    }
    const r1 = run(base({}, on([reg.sconf[0]], { pooled: 0.25 })));
    ck("two-sided pooled blindness: an AUROC of 0.25 counts as 0.75 (> 0.70) and fails that unit",
      r1.claims["SP-1"].per_unit[reg.sconf[0]].pass === false && r1.claims["SP-1"].per_unit[reg.sconf[0]].pooled === 0.75);
    const r9 = run(base(on(nulls, { rand: false })));
    ck("flip SP-9 -> ARCHITECTURAL-OR-MIXED (random-init map as good as trained)", r9.info["SP-9 learned vs random-init (INFO)"].label === "ARCHITECTURAL-OR-MIXED");
    // ---- NOT_EVALUABLE runs ----
    const miss = base(); for (const u of two_) delete miss[u];
    const rm = run(miss);
    ck("two Sconf records missing -> every claim NOT_EVALUABLE", CLAIMS.every(c => rm.claims[c].label.startsWith("NOT_EVALUABLE")));
    const rk = run(base({}, on(two_, { ka1: false })));
    ck("KA-1 fails on two units -> NOT_EVALUABLE", CLAIMS.every(c => rk.claims[c].label.startsWith("NOT_EVALUABLE")) && !rk.units[two_[0]].evaluable);
    const racc = run(base({}, on(two_, { acc: 0.5 })));
    ck("KA-6 plumbing accuracy fails on two units -> NOT_EVALUABLE", racc.claims["SP-1"].label.startsWith("NOT_EVALUABLE"));
    const rf = run(base({}, on(two_, { fpr: 0.12 })));
    ck("KA-4 conformal FPR outside the band on two units -> lane NOT_EVALUABLE", rf.lane.status === "NOT_EVALUABLE"
      && CLAIMS.every(c => rf.claims[c].label.startsWith("NOT_EVALUABLE")), rf.lane);
    const rf1 = run(base({}, on([reg.sconf[0]], { fpr: 0.12 })));
    ck("KA-4 on one unit only -> lane OK (flagged)", rf1.lane.status === "OK" && rf1.known_answers["KA-4 conformal band (map_max, alpha 0.05)"][reg.sconf[0]].pass === false);
    const rh = run(base({}, on([reg.sconf[0]], { hashSalt: "x" })));
    ck("KA-3 pixels differ across units -> lane NOT_EVALUABLE", rh.lane.status === "NOT_EVALUABLE" && rh.lane.why.some(w => w.startsWith("KA-3")));
    const tw = base(); const dup = { ...tw, [`${reg.sconf[0]}_r2`]: fixture(reg.sconf[0], { tag: "_r2" }) };
    const rt = run(dup);
    ck("a Sconf unit probed twice (touched-once) is not evaluable", !rt.units[reg.sconf[0]].evaluable && rt.claims["SP-1"].n_evaluable === 3, rt.units[reg.sconf[0]]);
    const bad = base(); bad[reg.sconf[0]].sets.F5 = F5.slice(0, 4);
    ck("a record with another condition set is not evaluable", !run(bad).units[reg.sconf[0]].evaluable);
    const rinv = run(base({}, on(three, { invalid: ["soiling__a12"] })));
    ck("soiling invalid on three units -> SP-5 NOT_EVALUABLE, SP-1 unaffected", rinv.claims["SP-5"].label.startsWith("NOT_EVALUABLE")
      && rinv.claims["SP-1"].label === "SUPPORTED");
    const rbadcall = base(); rbadcall[reg.sconf[0]].increments.SP2_map_over_pooled.call = "BOUNDED";   // inconsistent with its CIs
    ck("a call that disagrees with the frozen rule is not counted", run(rbadcall).claims["SP-2"].per_unit[reg.sconf[0]].ok === false);
    const rseal = run(base(), {}, { seal: undefined });
    ck("sealed.json matches -> units evaluable", Object.values(rseal.units).every(u => u.evaluable));
    const bs = base(); bs[reg.sconf[1]].dumps[0].meta_sha256 = "other";
    ck("sealed meta sha mismatch -> that unit not evaluable", !run(bs).units[reg.sconf[1]].evaluable);
    // ---- replay drift ----
    const rr = base(); const a0 = reg.anchors[0];
    rr[`${a0}_s2replay`] = fixture(a0, { phase: "discovery", tag: "_s2replay" });
    ck("identical replay -> PASS", run(rr).replay.units[a0].status === "PASS");
    const rd = base(); rd[`${a0}_s2replay`] = fixture(a0, { phase: "discovery", tag: "_s2replay", map: 0.95 + 0.004 });
    const rdr = run(rd);
    ck("replay drift 0.004: labels tagged REPLAY-DRIFT; a claim with margin < drift NOT_EVALUABLE", rdr.replay.status === "DRIFT"
      && rdr.claims["SP-2"].label === "SUPPORTED (REPLAY-DRIFT)" && rdr.claims["SP-1"].label === "SUPPORTED (REPLAY-DRIFT)", Object.fromEntries(CLAIMS.map(c => [c, rdr.claims[c].label])));
    const rp2 = base(); rp2[`${a0}_p2`] = fixture(a0, { phase: "discovery", tag: "_p2", map: 0.97 });
    rp2[`${a0}_s2replay`] = fixture(a0, { phase: "discovery", tag: "_s2replay", map: 0.97 });
    const rpr = run(rp2);
    ck("probe fixed at P2: the _p2 re-probe is the discovery record and the replay is compared with it",
      rpr.units[a0].record === `${a0}_p2` && rpr.replay.units[a0].status === "PASS" && /_p2/.test(rpr.replay.units[a0].note), rpr.replay.units[a0]);
    const rd2 = base({}, { all: { map: 0.902 } }); rd2[`${a0}_s2replay`] = fixture(a0, { phase: "discovery", tag: "_s2replay", map: 0.955 });
    ck("replay drift 0.005 > SP-1 margin 0.002 -> SP-1 NOT_EVALUABLE", run(rd2).claims["SP-1"].label.startsWith("NOT_EVALUABLE (REPLAY-DRIFT"));
    // ---- P2 amendments ----
    const ra = run(base(), { amendments: [{ track: "T3S", claim: "SP-4", action: "withdraw", reason: "fixture" },
      { track: "T3S", claim: "SP-5", action: "threshold", param: "map", value: 0.95, reason: "stricter" },
      { track: "T1", claim: "SP-6", action: "withdraw" }] });
    ck("amendments: withdraw -> INFO; a stricter threshold applies (SP-5 0.92 < 0.95 -> REFUTED); other tracks ignored",
      ra.claims["SP-4"].label.startsWith("INFO (withdrawn") && ra.claims["SP-5"].label === "REFUTED" && ra.claims["SP-6"].label === "SUPPORTED",
      Object.fromEntries(CLAIMS.map(c => [c, ra.claims[c].label])));
    const rl = run(base(), { amendments: [{ track: "T3S", claim: "SP-1", action: "threshold", param: "pooled", value: 0.8 }] });
    ck("a looser threshold is refused -> the claim is NOT_EVALUABLE", rl.claims["SP-1"].label === "NOT_EVALUABLE (refused P2 amendment)" && rl.amendments.refused.length === 1);
    const rn = run(base(), { amendments: [{ track: "T3S", claim: "SP-2", action: "threshold", param: "add", value: 0.05 }] });
    ck("the core's HEAD-ADDITIVE bounds are not amendable here", rn.amendments.refused.length === 1 && rn.claims["SP-2"].label.startsWith("NOT_EVALUABLE"));
    // ---- discovery phase ----
    const rdisc = run(base(), { phase: "discovery" });
    ck("discovery phase: no labels, predicates on the two trained hubs", Object.values(rdisc.claims).every(v => v.label.startsWith("not evaluated"))
      && Object.keys(rdisc.discovery_info.predicates).length === 2 && rdisc.discovery_info.predicates[hubs[0]]["SP-1"].pass === true);
    // ---- the real provenance path and the CLI (non-stubbed): fixture code never matches P1 -> NOT_EVALUABLE ----
    const cliRoot = path.join(tmp, "cli", "b4_t3s"), cliOut = path.join(tmp, "cli", "eval.json");
    writeFixtures(cliRoot, base());
    const args = [__filename, "--phase", "confirmation", "--root", cliRoot, "--sealed", sealed, "--replay", path.join(tmp, "none.json"),
      "--p1", "HEAD", "--p2", "HEAD", "--p-run", "f".repeat(40), "--json", cliOut];
    const c1 = spawnSync(process.execPath, args, { encoding: "utf8" });
    const j = fs.existsSync(cliOut) ? JSON.parse(fs.readFileSync(cliOut, "utf8")) : null;
    ck("CLI with the real provenance: fixture records do not carry the P1/P2 probe sha -> every claim NOT_EVALUABLE",
      c1.status === 0 && j && CLAIMS.every(c => j.claims[c].label.startsWith("NOT_EVALUABLE"))
      && reg.sconf.every(u => j.units[u].why.some(w => /probe code != P2/.test(w))), (c1.stderr || "").slice(0, 300));
    const c2 = spawnSync(process.execPath, args, { encoding: "utf8" });
    ck("CLI: the output JSON is written once (append-only)", c2.status === 2);
    const cdry = spawnSync(process.execPath, [__filename, "--phase", "confirmation", "--root", cliRoot, "--sealed", sealed], { encoding: "utf8" });
    ck("CLI dry run (no --p1) -> NOT_EVALUABLE", cdry.status === 0 && /NOT_EVALUABLE/.test(cdry.stdout), cdry.stdout.slice(0, 300));
    const one = path.join(tmp, "one.json");
    fs.writeFileSync(one, JSON.stringify(fixture("resnet20_s3")));
    const cc = spawnSync(process.execPath, [__filename, "--check-record", one], { encoding: "utf8" });
    const bf = path.join(tmp, "bad.json"); const brec = fixture("resnet20_s3"); delete brec.per_condition[F5[1]].pointing; fs.writeFileSync(bf, JSON.stringify(brec));
    const cb = spawnSync(process.execPath, [__filename, "--check-record", bf], { encoding: "utf8" });
    ck("--check-record: 0 on a complete record, 1 on a record missing a field", cc.status === 0 && cb.status === 1, [cc.stdout, cb.stdout]);
  } catch (e) {
    ck("self-test ran without an exception", false, String(e && e.stack));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  const ok = results.every(Boolean);
  console.log(`t3s_eval selftest ${ok ? "PASS" : "FAIL"} (${results.filter(Boolean).length}/${results.length})`);
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------- CLI
function main(argv) {
  const arg = k => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
  if (argv.includes("--selftest")) return selftest();
  if (argv.includes("--check-record")) {
    const f = arg("--check-record");
    let rec = null;
    try { rec = JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { console.log(`[t3s_eval] ${f}: ${e}`); return 1; }
    const p = schemaProblems(rec);
    console.log(p.length ? `[t3s_eval] ${f}: missing ${p.join(", ")}` : `[t3s_eval] ${f}: every field the evaluator reads is present`);
    return p.length ? 1 : 0;
  }
  const phase = arg("--phase");
  if (!["discovery", "confirmation"].includes(phase)) { console.error("usage: see the header of scripts/t3s_eval.js"); return 2; }
  const out = evaluate({ phase, p1: arg("--p1") || null, p2: arg("--p2") || null,
    pRun: (arg("--p-run") || "").split(",").filter(Boolean), root: arg("--root") || DEFAULTS.root,
    registry: arg("--registry") || DEFAULTS.registry, sealed: arg("--sealed") || DEFAULTS.sealed, replay: arg("--replay") || DEFAULTS.replay });
  console.log(summary(out));
  if (arg("--json") && !writeOnce(arg("--json"), out)) return 2;
  return 0;
}
module.exports = { RULES, evaluate, schemaProblems, headAdditiveCall, applyAmendments, label, PRED };
if (require.main === module) process.exit(main(process.argv.slice(2)));
