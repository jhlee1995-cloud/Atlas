#!/usr/bin/env node
// scripts/b4_make_registry.js -- builds experiments/b4/models.json, THE batch-4 CIFAR registry (docs/plans/
// B4_INTEGRATION.md D3), and prints the volume / GPU estimate per group. Self-contained: every number below is cited.
//   node scripts/b4_make_registry.js experiments/b4/models.json          (refuses to overwrite)
//   node scripts/b4_make_registry.js --check experiments/b4/models.json  (regenerates in memory; exit 1 if different)
// Roles: ANCHOR (rule-6 anchors, extracted first), D (16 old trained nets), N (2 random-init nulls), Dnew (5 hub nets,
// the smallest of each family), C (12 hub nets, SEALED, the PRIMARY units), F / F20 (fresh seeds 31/32 at depth 56 / 20,
// sealed fit + eval), K = Kls + Kwd (knob lane, sealed), R2 (8 spent seeds with a sealed never-read eval layout),
// Sdisc / Sconf (lane S maps), STdisc / STconf (streams).
// Sources: chenyaofo/pytorch-cifar-models README at 786c1625 (top-1 %); GitHub release API, 2026-09-23 (asset bytes;
// digest null upstream; T2 review #4); T2 registry draft (f16dims / penult / madds / est. GPU s; T2 review E: tap dims
// checked against the source at the pinned commit); upstream logs branch (best-val epoch); results/train_*/train.json
// (old checkpoint paths); committed atlas meta (anchors).
"use strict";
const fs = require("fs");

// arch -> [family, f16dims (sum of non-penult GAP tap dims), penult, first-tap dim, pre-tap dim, MAdds (M),
//          README top-1 %, release asset bytes, file tag, best-val epoch, T2 est. GPU s]
const ARCH = {
  cifar10_resnet20: ["resnet", 352, 64, 16, 64, 40.81, 92.60, 1139055, "4118986f", 183, 41],
  cifar10_resnet32: ["resnet", 352, 64, 16, 64, 69.12, 93.53, 1944567, "ef93fc4d", 186, 41],
  cifar10_resnet44: ["resnet", 352, 64, 16, 64, 97.44, 94.01, 2750015, "2a3cabcb", 187, 42],
  cifar10_resnet56: ["resnet", 352, 64, 16, 64, 125.75, 94.37, 3555463, "187c023a", 198, 42],
  cifar10_vgg11_bn: ["vgg", 2048, 512, 64, 512, 153.29, 92.79, 39068509, "eaeebf42", 200, 43],
  cifar10_vgg13_bn: ["vgg", 2048, 512, 64, 512, 228.79, 94.00, 39814235, "c01e4a43", 197, 44],
  cifar10_vgg16_bn: ["vgg", 2048, 512, 64, 512, 313.73, 94.16, 61080472, "6ee7ea24", 195, 46],
  cifar10_vgg19_bn: ["vgg", 2048, 512, 64, 512, 398.66, 93.91, 82346709, "57191229", 193, 47],
  cifar10_mobilenetv2_x0_5: ["mobilenetv2", 376, 1280, 16, 160, 27.97, 92.88, 2986233, "ca14ced9", 193, 41],
  cifar10_mobilenetv2_x0_75: ["mobilenetv2", 568, 1280, 24, 240, 59.31, 93.72, 5688121, "a53c314e", 192, 41],
  cifar10_mobilenetv2_x1_0: ["mobilenetv2", 744, 1280, 32, 320, 87.98, 93.79, 9193273, "fe6a5b48", 190, 42],
  cifar10_mobilenetv2_x1_4: ["mobilenetv2", 1048, 1792, 48, 448, 170.07, 94.22, 17635961, "3bbbd6e2", 183, 43],
  cifar10_shufflenetv2_x0_5: ["shufflenetv2", 456, 1024, 24, 192, 10.90, 90.13, 1554833, "1308b4e9", 193, 40],
  cifar10_shufflenetv2_x1_0: ["shufflenetv2", 1068, 1024, 24, 464, 45.00, 92.98, 5230673, "98807be3", 187, 41],
  cifar10_shufflenetv2_x1_5: ["shufflenetv2", 1608, 1024, 24, 704, 94.26, 93.55, 10164241, "296694dd", 197, 42],
  cifar10_shufflenetv2_x2_0: ["shufflenetv2", 2220, 2048, 24, 976, 187.81, 93.81, 21707473, "ec31611c", 197, 43],
  cifar10_repvgg_a0: ["repvgg", 576, 1280, 48, 192, 489.08, 94.39, 31582451, "ef08a50e", 193, 49],
  cifar10_repvgg_a1: ["repvgg", 768, 1280, 64, 256, 851.33, 94.89, 51541043, "38d2431b", 196, 56],
  cifar10_repvgg_a2: ["repvgg", 1120, 1408, 64, 384, 1850.10, 94.98, 107555782, "09488915", 193, 74],
};
// the 18 old nets: id, weights (null = hub), run group, full recipe, depth family (results/train_*/train.json)
const M = "/workspace/models";
const OLD = [
  ["resnet20_hub", null, "r20hub", true], ["resnet20_s1", `${M}/resnet20_s1_chenyaofo.pt`, "r20s1", true],
  ["resnet20_s2", `${M}/resnet20_s2_chenyaofo.pt`, "r20s2", true], ["resnet20_s3", `${M}/resnet20_s3_chenyaofo.pt`, "r20s3", true],
  ["resnet20_s4", `${M}/resnet20_s4_chenyaofo.pt`, "r20s4", true], ["resnet20_rand", `${M}/resnet20_rand99_chenyaofo.pt`, "r20rand", false],
  ["resnet56_hub", null, "r56hub", true], ["resnet56_s1", `${M}/resnet56_s1_chenyaofo.pt`, "r56s1", true],
  ["resnet56_s2", `${M}/resnet56_s2_chenyaofo.pt`, "r56s2", true],
  ["resnet56_e10", `${M}/resnet56_s11_e10_chenyaofo.pt`, "r56s11", false], ["resnet56_e20", `${M}/resnet56_s11_e20_chenyaofo.pt`, "r56s11", false],
  ["resnet56_e40", `${M}/resnet56_s11_e40_chenyaofo.pt`, "r56s11", false], ["resnet56_e50", `${M}/resnet56_s11_e50_chenyaofo.pt`, "r56s11", false],
  ["resnet56_e60", `${M}/resnet56_s11_e60_chenyaofo.pt`, "r56s11", false], ["resnet56_e70", `${M}/resnet56_s11_e70_chenyaofo.pt`, "r56s11", false],
  ["resnet56_s12m", `${M}/resnet56_s12m_chenyaofo.pt`, "r56s12", false], ["resnet56_s13m", `${M}/resnet56_s13m_chenyaofo.pt`, "r56s13", false],
  ["resnet56_rand", `${M}/resnet56_rand99_chenyaofo.pt`, "r56rand", false],
];
const ZOO = ["resnet32", "resnet44", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "mobilenetv2_x0_5", "mobilenetv2_x0_75",
  "mobilenetv2_x1_0", "mobilenetv2_x1_4", "shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0",
  "repvgg_a0", "repvgg_a1", "repvgg_a2"];
const DNEW = ["resnet32", "vgg11_bn", "mobilenetv2_x0_5", "shufflenetv2_x0_5", "repvgg_a0"];   // smallest per family (D3)
const R2 = ["resnet20_s1", "resnet20_s2", "resnet20_s3", "resnet20_s4", "resnet56_s1", "resnet56_s2", "resnet56_s12m", "resnet56_s13m"];
const SDISC = ["resnet20_hub", "resnet56_hub", "resnet20_rand", "resnet56_rand"];
const SCONF = ["resnet20_s3", "resnet20_s4", "resnet56_s12m", "resnet56_s13m"];
const STDISC = ["resnet20_hub", "resnet56_hub", "resnet56_e40"];
const STCONF_SPENT = ["resnet20_s1", "resnet20_s3", "resnet20_s4", "resnet56_s1", "resnet56_s12m", "resnet56_s13m"];
const ROWS = {   // = scripts/b4_extract.py ROWS (tests/test_b4_extract.py pins the agreement); D4
  fit: { n_ref: 10000, test: [0, 5000], c10c: [0, 2000], ood: [0, 2000], faults: [3500, 5000], fault_salt: 0 },
  eval: { test: [5000, 10000], c10c: [5000, 10000], ood: [2000, 7000], faults: [8500, 10000], fault_salt: 1 },
  maps: { n_ref: 10000, ref_stride: 2, cal: [7500, 8500], eval: [8500, 9500], faults: [8500, 9500], c10c: [8500, 9500] },
  roles: { H: [0, 2000], A: [2000, 3500], B: [3500, 5000], EVAL: [5000, 10000], MAPS: [7500, 9500], RESERVE_test: [9500, 10000],
           c10c_reserve: [2000, 5000] },
};
const COMMITTED = id => {                         // the committed atlas of every old net (T3 review K1 mapping)
  const m = id.match(/^resnet(20|56)_(.+)$/); if (!m) return null;
  const [, d, t] = m;
  if (t === "hub") return `results/atlas_v1_resnet${d}_s0hub_st3`;
  if (t === "rand") return `results/atlas_v1_resnet${d}_rand`;
  if (d === "20") return `results/atlas_v1_resnet20_${t}_st3`;
  if (t === "e40") return "results/atlas_v1_resnet56_e40_st3";
  return `results/atlas_v1_resnet56_${t}`;
};

// ---- storage per layout (bytes), following scripts/b4_extract.py's storage rule ----------------------------------------
const OVER = 68;                                   // logits f32 (40) + labels, rows int64 (16) + preds (12)
const FACT = 52;                                   // 13 pixel factors float32 (fit / eval only)
function fitBytes(a, extras) {
  const [, f16, P, first, pre] = a;
  const all = 10000 + 5000 + 10 * 3 * 2000 + 2 * 2000 + 9 * 1500, pen32 = 10000 + 5000 + 2 * 2000;
  const x4 = 5 * 3 * 2000 + (extras ? 4 * 2 * 2000 : 0);
  return all * 2 * f16 + pen32 * 4 * P + (all - pen32) * 2 * P + x4 * 2 * (first + pre + P) + (all + x4) * (OVER + FACT);
}
function evalBytes(a, extras) {
  const [, f16, P, first, pre] = a;
  const all = 5000 + 10 * 3 * 5000 + 2 * 5000 + 9 * 1500, pen32 = 5000 + 2 * 5000;
  const x4 = 5 * 2 * 5000 + (extras ? 4 * 2 * 5000 : 0);
  return all * 2 * f16 + pen32 * 4 * P + (all - pen32) * 2 * P + x4 * 2 * (first + pre + P) + (all + x4) * (OVER + FACT);
}
function mapsBytes(stage2, conf) {                 // ResNet: penult 64, headmap 64x8x8, stage2map 32x16x16, pixels, masks
  const local = conf ? 21 : 19, rows = 5000 + 1000 + 1000 + local * 1000 + 7 * 1000, pen32 = 7000;
  const per = 2 * 64 + 2 * 64 * 64 + (stage2 ? 2 * 32 * 256 : 0) + 3072 + OVER;
  return rows * per + pen32 * 2 * 64 + local * 1000 * 1024;
}
const gb = b => +(b / 1e9).toFixed(3);

const U = [];
function add(id, arch, roles, o) {
  const a = ARCH[arch]; if (!a) throw new Error("no arch " + arch);
  const sealedFit = roles.some(r => ["C", "F", "F20", "K"].includes(r));
  const extrasFit = sealedFit;                                        // D4: extras only for C, F, F20, K
  const layouts = { fit: { dump: `results/${sealedFit ? "b4c" : "b4d"}_${id}/dump`, sealed: sealedFit, extras: extrasFit,
                           gb: gb(fitBytes(a, extrasFit)) } };
  if (roles.some(r => ["F", "F20", "R2"].includes(r))) {
    const ex = roles.some(r => ["F", "F20"].includes(r));               // R2: no extras (D4)
    layouts.eval = { dump: `results/b4c_${id}_eval/dump`, sealed: true, extras: ex, gb: gb(evalBytes(a, ex)) };
  }
  if (roles.includes("Sdisc") || roles.includes("Sconf")) {
    const conf = roles.includes("Sconf"), stage2 = roles.includes("Sdisc") && !roles.includes("N");
    layouts.maps = { dump: `results/${conf ? "b4c" : "b4d"}_${id}_maps/dump`, sealed: conf,
                     maps: stage2 ? ["headmap", "stage2map"] : ["headmap"], holdout_faults: conf, pixels: true, rows: [7500, 9500],
                     gb: gb(mapsBytes(stage2, conf)) };
  }
  const rowsTot = 138500 + (layouts.eval ? 268500 : 0) + (layouts.maps ? 35000 : 0);
  U.push({ id, roles, arch, family: a[0], depth_family: o.depth_family, run_group: o.run_group, source: o.source,
           weights: o.weights, norm: "chenyaofo", full_recipe: o.full_recipe, trained: o.trained !== false,
           selected: o.selected, readme_top1: o.source === "hub" ? a[6] : null,
           hub_asset: o.source === "hub" ? { file: `${arch}-${a[8]}.pt`, bytes: a[7], tag: a[8] } : null,
           f16dims: a[1], penult: a[2], x4dims: a[3] + a[4] + a[2], madds: a[5], anchor: o.anchor || null,
           knob: o.knob || null, train: o.train || null, layouts, est_gpu_s: Math.round(a[10] * rowsTot / 100000) });
}
for (const [id, w, rg, full] of OLD) {
  const d = id.startsWith("resnet20") ? 20 : 56, arch = `cifar10_resnet${d}`, r = [];
  r.push(/_rand$/.test(id) ? "N" : "D");
  if (R2.includes(id)) r.push("R2");
  if (SDISC.includes(id)) r.push("Sdisc");
  if (SCONF.includes(id)) r.push("Sconf");
  if (STDISC.includes(id)) r.push("STdisc");
  if (STCONF_SPENT.includes(id)) r.push("STconf");
  if (id === "resnet20_hub" || id === "resnet56_hub") r.push("ANCHOR");
  add(id, arch, r, { source: w ? "file" : "hub", weights: w, run_group: rg, full_recipe: full, trained: !/_rand$/.test(id),
                     depth_family: `resnet${d}`, anchor: COMMITTED(id),
                     selected: w ? "last epoch" : `best-val epoch ${ARCH[arch][9]} (upstream log)` });
}
for (const z of ZOO) {
  const arch = `cifar10_${z}`;
  add(z, arch, [DNEW.includes(z) ? "Dnew" : "C"], { source: "hub", weights: null, run_group: z, full_recipe: true,
      depth_family: z, selected: `best-val epoch ${ARCH[arch][9]} (upstream log)` });
}
const fresh = (d, s) => add(`resnet${d}_s${s}`, `cifar10_resnet${d}`, [d === 56 ? "F" : "F20", "STconf"], {
  source: "file", weights: `${M}/resnet${d}_s${s}_chenyaofo.pt`, run_group: `r${d}s${s}`, full_recipe: true,
  depth_family: `resnet${d}`, selected: "last epoch",
  train: { script: "scripts/train_second_seed.py", args: `--seed ${s} --norm chenyaofo --arch cifar10_resnet${d}` } });
for (const s of [31, 32]) fresh(56, s);
for (const s of [31, 32]) fresh(20, s);
for (const s of [31, 32]) {
  add(`resnet56_s${s}_ls10`, "cifar10_resnet56", ["K", "Kls"], { source: "file", weights: `${M}/resnet56_s${s}_ls10_chenyaofo.pt`,
      run_group: `r56s${s}ls`, full_recipe: false, depth_family: "resnet56", selected: "last epoch", knob: "label_smoothing 0.1",
      train: { script: "scripts/b4_train_knob.py", args: `--seed ${s} --norm chenyaofo --arch cifar10_resnet56 --label-smoothing 0.1` } });
  add(`resnet56_s${s}_wd5e5`, "cifar10_resnet56", ["K", "Kwd"], { source: "file", weights: `${M}/resnet56_s${s}_wd5e5_chenyaofo.pt`,
      run_group: `r56s${s}wd`, full_recipe: false, depth_family: "resnet56", selected: "last epoch", knob: "weight_decay 5e-5",
      train: { script: "scripts/train_second_seed.py", args: `--seed ${s} --norm chenyaofo --arch cifar10_resnet56 --wd 5e-5` } });
}

const REG = {
  schema: "b4_models/1", generator: "scripts/b4_make_registry.js", pinned_at: "P1",
  hub: { repo: "chenyaofo/pytorch-cifar-models", ref: "786c16252c0fc58ee9adac063f8337cc4a7a497a",
         recipe: "every chenyaofo CIFAR-10 model: SGD nesterov lr 0.1 momentum 0.9 wd 5e-4 batch 256, cosine T_max 200, 200 epochs, cross-entropy, mean (0.4914, 0.4822, 0.4465) std (0.2023, 0.1994, 0.201) (logs branch, logs/cifar10/<model>/default.log, read 2026-09-23). The released weights are the best-val checkpoint (epoch 183-200), selected upstream on the full CIFAR-10 test set." },
  rows: ROWS,
  confirmation_only: ["impulse_noise", "glass_blur", "zoom_blur", "frost", "elastic_transform", "speckle_noise", "gaussian_blur",
                      "spatter", "saturate", "exposure_global", "soiling"],
  models: U,
};
const text = JSON.stringify(REG, null, 1) + "\n";
if (/[^\x00-\x7f]/.test(text)) throw new Error("registry must be ASCII");

const args = process.argv.slice(2);
if (args[0] === "--check") {
  const cur = fs.readFileSync(args[1], "utf8").replace(/\r\n/g, "\n");
  if (cur !== text) { console.error(`${args[1]} differs from the generator output`); process.exit(1); }
  console.log(`${args[1]} == generator output (${U.length} units)`);
  process.exit(0);
}
const out = args[0];
if (!out) { console.error("usage: node scripts/b4_make_registry.js <out.json> | --check <json>"); process.exit(2); }
if (fs.existsSync(out)) { console.error(`refusing to overwrite ${out} (use --check)`); process.exit(1); }
fs.writeFileSync(out, text);

// ---- summary ---------------------------------------------------------------------------------------------------------
const roles = ["ANCHOR", "D", "N", "Dnew", "C", "F", "F20", "K", "R2", "Sdisc", "Sconf", "STdisc", "STconf"];
for (const r of roles) console.log(r.padEnd(7), String(U.filter(m => m.roles.includes(r)).length).padStart(2), U.filter(m => m.roles.includes(r)).map(m => m.id).join(" "));
const sum = f => U.reduce((s, m) => s + f(m), 0);
const line = (k, v) => console.log(k.padEnd(52), v.toFixed(2), "GB");
line("fit, discovery-readable (D, N, Dnew; b4d_*)", sum(m => m.layouts.fit.sealed ? 0 : m.layouts.fit.gb));
line("fit, sealed C (12 zoo; b4c_*)", sum(m => m.roles.includes("C") ? m.layouts.fit.gb : 0));
line("fit, sealed F + F20 + K", sum(m => m.roles.some(r => ["F", "F20", "K"].includes(r)) ? m.layouts.fit.gb : 0));
line("eval, sealed F + F20", sum(m => m.layouts.eval && !m.roles.includes("R2") ? m.layouts.eval.gb : 0));
line("eval, sealed R2 (8 spent seeds)", sum(m => m.roles.includes("R2") ? m.layouts.eval.gb : 0));
line("maps, discovery (Sdisc)", sum(m => m.layouts.maps && !m.layouts.maps.sealed ? m.layouts.maps.gb : 0));
line("maps, sealed (Sconf)", sum(m => m.layouts.maps && m.layouts.maps.sealed ? m.layouts.maps.gb : 0));
line("TOTAL new dumps", sum(m => m.layouts.fit.gb + (m.layouts.eval ? m.layouts.eval.gb : 0) + (m.layouts.maps ? m.layouts.maps.gb : 0)));
console.log("GPU extraction estimate (s):", sum(m => m.est_gpu_s), " units:", U.length, "->", out);
