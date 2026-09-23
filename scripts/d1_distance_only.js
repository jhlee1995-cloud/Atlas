// D1 (Stage 1b, docs/plans/STAGE1.md amendment 2; also A4 R56-3b): distance-only penult adjacency = Spearman
// (average ranks for ties) over the 45 upper-triangle Euclidean distances between
// per_layer.penult.class_centers.centers of two atlases. Reproduces the Stage 1 values: s1-s2 0.950,
// hub-s1 0.955, hub-s2 0.963, twin 0.993, null 0.644 (results/atlas_v1_resnet20_s1/SESSION.md:101-102).
// Usage, from the repo root (node only; the atlases are the pulled results/*/atlas.json):
//   node scripts/d1_distance_only.js                      # the Stage 1b pairs listed in DEFAULT_PAIRS
//   node scripts/d1_distance_only.js s3:s4 s1:s2          # short names = results/atlas_v1_resnet20_<name>
//   node scripts/d1_distance_only.js atlas_v1_resnet20_s0hub_st2:atlas_v1_resnet56_s0hub   # results/<name>
//   node scripts/d1_distance_only.js results/x:results/y  # any result dirs (a name containing "/")
const DEFAULT_PAIRS = ["s3:s4", "s0hub:s3", "s0hub:s4", "s1:s2", "s1:s3", "s1:s4", "s2:s3", "s2:s4"];
const path = require("path");
const rank = a => { const o = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), r = new Array(a.length);
  for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
    for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
const pearson = (x, y) => { const n = x.length, mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let sxy = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sx += (x[i] - mx) ** 2; sy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sx * sy); };
const dir = n => n.includes("/") ? n : path.join("results", n.startsWith("atlas_") ? n : `atlas_v1_resnet20_${n}`);
const dists = name => { const C = require(path.resolve(process.cwd(), dir(name), "atlas.json")).per_layer.penult.class_centers.centers;
  const d = []; for (let i = 0; i < C.length; i++) for (let j = i + 1; j < C.length; j++)
    d.push(Math.sqrt(C[i].reduce((s, v, k) => s + (v - C[j][k]) ** 2, 0))); return d; };
const pairs = process.argv.length > 2 ? process.argv.slice(2) : DEFAULT_PAIRS;
for (const p of pairs) { const [a, b] = p.split(":");
  try { console.log(`${a}-${b} distance-only rho ${pearson(rank(dists(a)), rank(dists(b))).toFixed(3)}`); }
  catch (e) { console.log(`${a}-${b} not computable: ${String(e.message).split("\n")[0]}`); } }
