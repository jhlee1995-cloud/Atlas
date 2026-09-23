#!/usr/bin/env bash
# a4b_eval_fixture.sh -- runs the frozen A4b evaluator (scripts/a4b_eval.js) on committed atlases copied under A4b names
# into a scratch results/ tree, in the matched, interp and below scenarios, and checks its labels (A4b review item 14;
# docs/plans/STAGE2B.md "Local evaluation"). Run it before the pre-registration commit and after any evaluator change.
# node + git only (Windows Git Bash or Linux); writes nothing inside the repo; the scratch tree is deleted at exit.
#
#   bash scripts/a4b_eval_fixture.sh            # all scenarios; exit 1 on the first failed expectation
#   A4B_FIXTURE_KEEP=1 bash scripts/a4b_eval_fixture.sh   # keep the scratch tree (path printed)
#
# Stand-ins (the numbers are Stage 1/1b/2 values, the labels are what the rules must give for them):
#   resnet20 band _st3      <- atlas_v1_resnet20_{s0hub,s1,s2}_st2, atlas_v1_resnet20_{s3,s4} (and their compares)
#   s1, s2, twin, "on-side" <- atlas_v1_resnet56_s0hub (id HIGH, sep_ratio HIGH, bridge HIGH, nc1 LOW vs B20+)
#   "off-side" rung         <- atlas_v1_resnet56_e40 (id HIGH, sep_ratio LOW, bridge LOW, nc1 IN)
#   margins                 <- margin_v1_resnet56_s0hub (M2 fails) and margin_v1_resnet20_s1/s2 (M2 holds)
#   train.json, critic.json, i2.json, check.json, ladder.json, maxprob_ties.json, and the pair deformation.json
#   files that no committed compare has, are synthesized; A4b meta.git_commit / created / stage_b_git_commit are set
#   to HEAD and now (G0e); the evaluator runs with --p HEAD --p-run HEAD.
#   scenario matched: e50 0.9230 (off-side), e60 0.9255, e70 0.9280, s12m 0.9262 (on-side), s13m 0.9250 (off-side)
#            -> M11 e60; D-ID CONFIRMED; D-COLL SEED-SENSITIVE; M56-b REJECT with attribution SPLIT
#   scenario interp: e50 0.9315, e60 0.9320, e70 0.9330 -> interpolation e40_st3 .. e50 (t 0.5447); CONFIRMED-1;
#            M56 attribution and the M56-a DEPTH reading NOT_EVALUABLE
#   scenario below: e50 0.9190, e60 0.9195, e70 0.9200, e90 0.9240, s12m 0.9262, s13m 0.9300 -> M11 e90; CONFIRMED;
#            with a critic S5 that disagrees with the recomputation, so every promotion is blocked
#   variants of matched: G0e FAIL (--p-run HEAD~1), T56 core FAIL (twin critic S9 FAIL), G0b FAIL (no i2.json),
#            R0 FAIL (train_resnet56_s2 accuracy 0.930: every item that reads s1 or s2 is NOT_EVALUABLE)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d)"
if [[ "${A4B_FIXTURE_KEEP:-0}" == 1 ]]; then echo "[fixture] scratch tree kept: $W"; else trap 'rm -rf "$W"' EXIT; fi
HEAD7="$(git -C "$REPO" rev-parse --short HEAD)"

build() {   # build <scenario>: $W/<scenario>/results
  node - "$REPO" "$W/$1/results" "$1" "$HEAD7" <<'EOF'
const fs = require("fs"), path = require("path");
const [REPO, OUT, SC, HEAD7] = process.argv.slice(2);
const SRC = path.join(REPO, "results");
const rd = p => JSON.parse(fs.readFileSync(p, "utf8"));
const wr = (p, o) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, typeof o === "string" ? o : JSON.stringify(o)); };
const now = new Date().toISOString().replace("T", " ").slice(0, 19);            // pod clock format, UTC
const SHARP = ["luminance_mean", "spectral_slope", "saturation_mean", "hue_sin", "colorfulness", "orientation_entropy",
               "class", "corruption_family", "corruption_type", "severity"];
// an A4b run: the atlas under its A4b name, made "this session" at HEAD (G0e)
function run(from, to) {
  const a = rd(path.join(SRC, from, "atlas.json"));
  a.exp_id = to; a.meta = { ...a.meta, git_commit: HEAD7, created: now, exp_id: to };
  wr(path.join(OUT, to, "atlas.json"), a);
  wr(path.join(OUT, to, "provenance.json"), { exp_id: to, stage_b_git_commit: HEAD7, dump_meta: a.meta });
}
const orig = n => { wr(path.join(OUT, n, "atlas.json"), rd(path.join(SRC, n, "atlas.json"))); };   // a committed run, as is
const dfm = (b, a, from) => wr(path.join(OUT, b, `compare_vs_${a}`, "deformation.json"), from ? rd(from) : from);
const synthDfm = (b, a, cka) => wr(path.join(OUT, b, `compare_vs_${a}`, "deformation.json"),
  { a, b, per_layer: { penult: { cka_test: cka, relrep_argmax_agree_ood_c100: 0.57, relrep_argmax_chance_ood_c100: 0.124 } } });
const train = (name, seed, epochs, acc) => wr(path.join(OUT, `train_${name}`, "train.json"),
  { recipe: { arch: "cifar10_resnet56", seed, norm: "chenyaofo", epochs }, final_test_acc_10k: acc });
const critic = (runs, o = {}) => ({ runs, verdict: "fixture", checks: {
  synthetic_refusal: [{ name: "sources", status: "PASS", detail: "all runs real" }],
  norm_consistency: [{ name: "input_norm", status: "PASS", detail: "{}" }],
  holdout_hygiene: runs.map(r => ({ name: `holdout:${r}`, status: "PASS", detail: "declared" })),
  scalar_stability: [{ name: "penult/class_centers.sep_ratio", status: o.S3 || "PASS", detail: "" },
                     { name: "penult/margin_typeb.auc_margin_typeb", status: "PASS", detail: "" }],
  id_profile_stability: [{ name: "id_profile[0,1]", status: o.S1 || "PASS", detail: "" }],
  adjacency_stability: [{ name: "penult/adjacency_spearman[0,1]", status: o.S5 || "PASS", detail: "" }],
  decodability_stability: SHARP.map(fc => ({ name: `decod/${fc}[0,1]`, status: "PASS", detail: "profile_spearman=1.00 mean_abs_delta=0.000" })),
  commit_agreement: [{ name: "commit/class", status: o.S8 || "PASS", detail: "" }],
  panel_agreement: [{ name: "penult/cka_test[0,1]", status: o.S9 || "PASS", detail: "" }] } });
const R20 = "atlas_v1_resnet20", R56 = "atlas_v1_resnet56", HUB = `${R56}_s0hub`, E40 = `${R56}_e40`;

// committed inputs the evaluator reads as they are
wr(path.join(OUT, "norm_check_resnet20", "norm_check.json"), rd(path.join(SRC, "norm_check_resnet20", "norm_check.json")));
wr(path.join(OUT, "train_resnet56_e40", "train.json"), rd(path.join(SRC, "train_resnet56_e40", "train.json")));
for (const n of [`${R20}_s0hub_st2`, `${R20}_s1_st2`, `${R20}_s2_st2`, `${R20}_s3`, `${R20}_s4`, HUB, E40]) orig(n);
for (const s of ["s0hub", "s1", "s2", "s3", "s4"]) orig(`margin_v1_resnet20_${s}`);
orig("margin_v1_resnet56_s0hub");

// resnet20 band _st3 and its pair compares (P20)
const ST3 = { s0hub: `${R20}_s0hub_st2`, s1: `${R20}_s1_st2`, s2: `${R20}_s2_st2`, s3: `${R20}_s3`, s4: `${R20}_s4` };
for (const [s, from] of Object.entries(ST3)) { run(from, `${R20}_${s}_st3`); run(`margin_v1_resnet20_${s}`, `margin_v1_resnet20_${s}_st3`);
  synthDfm(`${R20}_${s}_st3`, from, 1.0); }                                                   // G0c same-space
for (const [b, a, from] of [["s1", "s0hub", `${R20}_s1_st2/compare_vs_${R20}_s0hub_st2`], ["s2", "s0hub", `${R20}_s2_st2/compare_vs_${R20}_s0hub_st2`],
  ["s2", "s1", `${R20}_s2_st2/compare_vs_${R20}_s1_st2`], ["s3", "s0hub", `${R20}_s3/compare_vs_${R20}_s0hub`], ["s3", "s1", `${R20}_s3/compare_vs_${R20}_s1`],
  ["s3", "s2", `${R20}_s3/compare_vs_${R20}_s2`], ["s4", "s0hub", `${R20}_s4/compare_vs_${R20}_s0hub`], ["s4", "s1", `${R20}_s4/compare_vs_${R20}_s1`],
  ["s4", "s2", `${R20}_s4/compare_vs_${R20}_s2`], ["s4", "s3", `${R20}_s4/compare_vs_${R20}_s3`]])
  dfm(`${R20}_${b}_st3`, `${R20}_${a}_st3`, path.join(SRC, from, "deformation.json"));

// resnet56 runs: hub_st3, twin, e40_st3, seeds, rungs, replicates (stand-in per scenario)
const ON = HUB, OFF = E40;
const plan = { hub_st3: ON, s0hub_ref1: ON, e40_st3: OFF, s1: ON, s2: ON };
const acc = {
  matched: { e50: [0.9230, OFF], e60: [0.9255, ON], e70: [0.9280, ON], s12m: [0.9262, ON, 60], s13m: [0.9250, OFF, 60] },
  interp:  { e50: [0.9315, ON], e60: [0.9320, ON], e70: [0.9330, ON] },
  below:   { e50: [0.9190, OFF], e60: [0.9195, OFF], e70: [0.9200, OFF], e90: [0.9240, ON], s12m: [0.9262, ON, 90], s13m: [0.9300, ON, 90] },
}[SC];
for (const [k, v] of Object.entries(acc)) {
  plan[k] = v[1];
  if (k.startsWith("e")) train(`resnet56_${k}`, 11, +k.slice(1), v[0]); else train(`resnet56_${k}`, +k.slice(1, 3), v[2], v[0]);
}
train("resnet56_s1", 1, 200, 0.9431); train("resnet56_s2", 2, 200, 0.9440);
for (const [k, from] of Object.entries(plan)) {
  const to = `${R56}_${k === "hub_st3" ? "s0hub_st3" : k}`;
  run(from, to);
  for (const s of ["s0hub", "s1", "s2"])                                                          // X / XM compares
    dfm(to, `${R20}_${s}_st3`, path.join(SRC, HUB, `compare_vs_${R20}_${s}_st2`, "deformation.json"));
}
synthDfm(`${R56}_s0hub_st3`, HUB, 1.0); synthDfm(`${R56}_e40_st3`, E40, 1.0);                    // G0c same-space
synthDfm(`${R56}_s2`, `${R56}_s1`, 0.95);                                                        // W56
synthDfm(`${R56}_s1`, `${R56}_s0hub_st3`, 0.95); synthDfm(`${R56}_s2`, `${R56}_s0hub_st3`, 0.95);   // hub pairs
synthDfm(`${R56}_s0hub_ref1`, `${R56}_s0hub_st3`, 0.9999);                                       // T56
// margins: s1/s2 = the hub margin (M2 fails); matched margins alternate so the attribution is SPLIT when matched
const MG = { s0hub_st3: "margin_v1_resnet56_s0hub", s1: "margin_v1_resnet56_s0hub", s2: "margin_v1_resnet56_s0hub",
  e50: "margin_v1_resnet20_s2", e60: "margin_v1_resnet20_s1", e70: "margin_v1_resnet20_s1", e90: "margin_v1_resnet20_s1",
  s12m: "margin_v1_resnet20_s2", s13m: "margin_v1_resnet56_s0hub" };
for (const [k, from] of Object.entries(MG)) if (k in plan || k === "s0hub_st3") run(from, `margin_v1_resnet56_${k}`);
// critics and the check dir
const s1s2 = [`${R56}_s1`, `${R56}_s2`];
wr(path.join(OUT, "critic_v1_resnet56_s1_s2", "critic.json"), critic(s1s2, SC === "below" ? { S5: "FAIL" } : {}));
wr(path.join(OUT, "critic_v1_resnet56_hub_noise", "critic.json"), critic([`${R56}_s0hub_st3`, `${R56}_s0hub_ref1`]));
wr(path.join(OUT, "critic_margin_v1_resnet56_s1_s2", "critic.json"), critic(s1s2.map(n => n.replace("atlas_", "margin_"))));
const I = path.join(OUT, "instrument_check_a4b");
wr(path.join(I, "check.json"), { status: "PASS", bitwise_identical: true });
const i2runs = Object.fromEntries(["s1", "s2", "s12m", "s13m"].filter(k => k in plan).map(k => [`${R56}_${k}`, { status: "PASS", problems: [] }]));
wr(path.join(I, "i2.json"), { status: "PASS", runs: i2runs });
wr(path.join(I, "ladder.json"), { matched: { status: "matched", e_star: 60 }, interp: { status: "interpolate", e_star: null },
  below: { status: "matched", e_star: 90 } }[SC]);
wr(path.join(I, "maxprob_ties.json"), { [`${R56}_s1`]: { half_tie_confmatched: 0.002 }, [`${R56}_s2`]: { half_tie_confmatched: 0.02 } });
wr(path.join(I, "code_diff.txt"), `HEAD ${HEAD7}\n`);
wr(path.join(I, "versions.json"), { python: "3.12.3", numpy: "2.1.2", scipy: "1.18.1", sklearn: "1.9.1" });
EOF
}

evaluate() {   # evaluate <scenario> <label> [a4b_eval.js args]: $W/<scenario>/<label>.{json,txt}
  local sc="$1" lab="$2"; shift 2
  node "$REPO/scripts/a4b_eval.js" --root "$W/$sc/results" --json "$W/$sc/$lab.json" "$@" > "$W/$sc/$lab.txt"
}

expect() {     # expect <scenario> <label>: the pre-registered labels for this stand-in tree
  node - "$W/$1/$2.json" "$2" <<'EOF'
const fs = require("fs");
const [file, lab] = process.argv.slice(2);
const o = JSON.parse(fs.readFileSync(file, "utf8"));
const bad = [];
const eq = (what, got, want) => { if (JSON.stringify(got) !== JSON.stringify(want)) bad.push(`${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); };
const has = (what, s, sub) => { if (typeof s !== "string" || !s.includes(sub)) bad.push(`${what}: ${JSON.stringify(s)} lacks ${JSON.stringify(sub)}`); };
const out = o.D.reduce((m, d) => ({ ...m, [d.field]: d.outcome }), {});
const promoted = t => typeof t === "string" && t.startsWith("d56 PROMOTE (");
if (lab === "matched") {
  eq("G0e", o.G0e.status, "PASS"); eq("G0c", o.G0c.status, "IDENTICAL"); eq("ladder", [o.ladder.status, o.ladder.M11, o.ladder.pod_e_star], ["matched", 60, 60]);
  eq("Mc", o.ladder.Mc.map(m => m.evaluable), [true, true]); eq("guard", o.guard, []); eq("blocked", o.promotion_blocked_by, []);
  eq("D", out, { id: "CONFIRMED", sep_ratio: "SEED-SENSITIVE", bridge: "SEED-SENSITIVE", nc1: "SEED-SENSITIVE" });
  eq("D_COLL", o.D_COLL, "SEED-SENSITIVE"); eq("row11 D-ID", o.row11["D-ID"], "\u2705");
  eq("d56 promote", ["1", "3", "4", "5", "6a", "7"].map(k => promoted(o.PROMO[k])), [true, true, true, true, true, true]);
  eq("XM n", o.LEVEL.D1.xm_n, 15); eq("M56-b", [o.M56.b_row9_d56.tag, o.M56.b_row9_d56.m2_failure_attribution], ["d56 REJECT", "SPLIT"]);
  eq("M56-c ties", o.M56.c_confidence.tie_call, ["TIES-EXCLUDED", "TIE-ARTIFACT-POSSIBLE"]);
  eq("T56", o.T56.fail, false); eq("KILL56", o.KILL56, false); eq("rule 8 D1", o.rule8.D1_gt_max, true);
} else if (lab === "interp") {
  eq("ladder", [o.ladder.status, o.ladder.M11, o.ladder.interp && o.ladder.interp.lo.E, o.ladder.interp && o.ladder.interp.hi.E], ["interpolate", null, 40, 50]);
  if (!(Math.abs(o.ladder.interp.t - 0.5447) < 1e-3)) bad.push(`t ${o.ladder.interp.t}`);
  eq("Mc", o.ladder.Mc.map(m => m.trained), [false, false]);
  eq("D", out, { id: "CONFIRMED-1", sep_ratio: "CONFIRMED-1", bridge: "CONFIRMED-1", nc1: "CONFIRMED-1" });
  eq("row11", [o.row11["D-ID"], o.row11["D-COLL"]], ["\u{1F7E1}", "\u{1F7E1}"]); eq("XM n", o.LEVEL.D1.xm_n, 5);
  eq("M56 attr", [o.M56.b_row9_d56.m2_failure_attribution, o.M56.b_row9_d56.m2_failure_attribution_reason], [null, "NOT_EVALUABLE (interpolation branch)"]);
  eq("M56-a", o.M56.a_auc.improves_with_scale_at_matched_acc, "NOT_EVALUABLE (interpolation branch)");
} else if (lab === "below") {
  eq("ladder", [o.ladder.status, o.ladder.M11, o.ladder.in_window], ["matched", 90, [90]]);
  eq("D", out, { id: "CONFIRMED", sep_ratio: "CONFIRMED", bridge: "CONFIRMED", nc1: "CONFIRMED" });
  eq("mismatch", o.core_mismatch, ["S5"]); has("row11 blocked", o.row11["D-ID"], "promotion blocked");
  has("row 1 blocked", o.PROMO["1"], "d56 PROMOTE blocked"); has("row 7", o.PROMO["7"], "ONE-PAIR");
} else if (lab === "g0efail") {
  eq("G0e", o.G0e.status, "FAIL"); eq("D-ID", out.id, "CONFIRMED"); has("row11 blocked", o.row11["D-ID"], "G0e FAIL");
  has("row 1", o.PROMO["1"], "d56 PROMOTE blocked (G0e FAIL");
} else if (lab === "twinfail") {
  eq("T56", o.T56.fail, true); eq("D", Object.values(out), ["NOT_EVALUABLE", "NOT_EVALUABLE", "NOT_EVALUABLE", "NOT_EVALUABLE"]);
  has("row 1", o.PROMO["1"], "no d56 verdict (T56"); eq("M56-b promotable", o.M56.b_row9_d56.promotable, false);
} else if (lab === "g0bfail") {
  eq("G0b s1", o.G0b.atlases.atlas_v1_resnet56_s1, ["i2 MISSING"]); eq("D-ID", out.id, "NOT_EVALUABLE");
  has("D-ID why", o.D[0].not_evaluable_because.join("; "), "G0b");
} else if (lab === "r0fail") {
  const NE = "NOT_EVALUABLE (R0 FAIL)", b = o.M56.b_row9_d56;
  eq("R0", o.R0_56, false); eq("guard", o.guard, ["R0 FAIL"]);
  eq("D", Object.values(out), ["NOT_EVALUABLE", "NOT_EVALUABLE", "NOT_EVALUABLE", "NOT_EVALUABLE"]);
  has("row11", o.row11["D-ID"], "no verdict");
  eq("d56 tags", ["1", "3", "4", "5", "6a", "7"].map(k => o.PROMO[k].startsWith("no d56 verdict (R0 FAIL")), [true, true, true, true, true, true]);
  eq("SHAPE", Object.values(o.SHAPE).every(v => v === NE), true);
  eq("LEVEL", Object.values(o.LEVEL).map(v => [v.label, v.matched]), [[NE, NE], [NE, NE], [NE, NE], [NE, NE]]);
  eq("[ACC]", Object.values(o.acc_fields).map(v => v.labels), [[NE, NE], [NE, NE], [NE, NE]]);
  eq("M56-b", [b.tag, b.promotable, b.m2_failure_attribution, b.m2_failure_attribution_reason], [NE, false, null, NE]);
  eq("M56-a", [o.M56.a_auc.all_ge_080, o.M56.a_auc.scale_robust], [NE, NE]);
  eq("M56-c", [o.M56.c_confidence.E1, o.M56.c_confidence.tie_call], [[NE, NE], [NE, NE]]);
  eq("rule 8", o.rule8, NE); has("row 8", o.row8_info, NE);
}
if (bad.length) { console.log(`[fixture] ${lab} FAIL\n  ` + bad.join("\n  ")); process.exit(1); }
console.log(`[fixture] ${lab} PASS: D-ID ${o.D_ID}, D-COLL ${o.D_COLL}, M56-b ${o.M56 ? o.M56.b_row9_d56.tag : "-"}, G0e ${o.G0e.status}`);
EOF
}

node --check "$REPO/scripts/a4b_eval.js"
( cd "$REPO" && node scripts/a4b_eval.js --dry-run > "$W/dry.txt" ) && echo "[fixture] --dry-run OK ($(grep -c . "$W/dry.txt") lines)"
for sc in matched interp below; do
  build "$sc"
  evaluate "$sc" "$sc" --p HEAD --p-run HEAD
  expect "$sc" "$sc"
done
evaluate matched g0efail --p HEAD --p-run HEAD~1 && expect matched g0efail
node -e 'const fs = require("fs"), p = process.argv[1], c = JSON.parse(fs.readFileSync(p, "utf8"));
  c.checks.panel_agreement[0].status = "FAIL"; fs.writeFileSync(p, JSON.stringify(c));' "$W/matched/results/critic_v1_resnet56_hub_noise/critic.json"
evaluate matched twinfail --p HEAD --p-run HEAD && expect matched twinfail
build matched && rm "$W/matched/results/instrument_check_a4b/i2.json"
evaluate matched g0bfail --p HEAD --p-run HEAD && expect matched g0bfail
build matched && node -e 'const fs = require("fs"), p = process.argv[1], t = JSON.parse(fs.readFileSync(p, "utf8"));
  t.final_test_acc_10k = 0.930; fs.writeFileSync(p, JSON.stringify(t));' "$W/matched/results/train_resnet56_s2/train.json"
evaluate matched r0fail --p HEAD --p-run HEAD && expect matched r0fail
echo "[fixture] all scenarios PASS"
