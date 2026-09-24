#!/usr/bin/env node
// scripts/b4_stats.js -- shared statistics and provenance helpers for the three batch-4 evaluators (t1_eval.js,
// collapse_laws.js, t3s_eval.js; docs/plans/B4_INTEGRATION.md D5, D12, D14). Owner: T1. Node standard library only.
//   const S = require("./b4_stats.js");         node scripts/b4_stats.js --selftest
// INTERFACE (stable at P1)
//   betaBinomBand(nCal, nTest, alpha, level = 0.99) -> {l, a, b, mean, lo, hi, level}
//       the exact central `level` band of the realised conformal false-alarm rate k / nTest when p = (1 + #{cal >= s}) /
//       (nCal + 1) is flagged at p <= alpha: theta ~ Beta(l, nCal + 1 - l), l = floor(alpha (nCal + 1)), k ~ Bin(nTest,
//       theta) (the beta-binomial). ONE band for every track (D14; replaces T2's 3-SD band).
//   betaBinomCurve(nCal, nTest, alphas = [0.02, 0.03, 0.04, 0.05], level) -> [{alpha, lo, hi, mean}]   (rule 5 curve)
//   holm(pvals, alpha = 0.05) -> {adjusted, reject}   Holm step-down; null p-values stay null (= atlas/b4_core.holm)
//   rankAvg(x), pearson(x, y), spearman(x, y)          ties averaged; null when < 3 points or no variance
//   famDemean(values, families, minSize = 2) -> values minus their family mean (null outside families of >= minSize)
//   famDemeanedSpearman(pred, y, families, minSize = 2, minN = 6) -> Pearson of family-demeaned ranks (D12 WIN clause)
//   fileSha256(path, {eol: true}) -> hex (CRLF -> LF first, = atlas/b4_core.code_sha256)
//   git(args, repo) -> stdout | null; gitHead(repo); gitIsAncestor(a, b, repo); gitShow(rev, path, repo) -> Buffer | null;
//   gitSha256At(rev, path, repo) -> hex | null; provenance({repo, p1, p2, pRun, files}) -> record for eval JSON
//   isNum(v)
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { execFileSync } = require("child_process");

const REPO = path.resolve(__dirname, "..");
const isNum = v => typeof v === "number" && Number.isFinite(v);

// ---- log-gamma (Lanczos, g = 7, n = 9; |relative error| < 1e-13 for x > 0) ---------------------------------------------
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LG[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaBinomBand(nCal, nTest, alpha, level = 0.99) {
  const l = Math.floor(alpha * (nCal + 1));
  if (l < 1) return { l, a: l, b: nCal + 1 - l, mean: 0, lo: 0, hi: 0, level, degenerate: true };
  const a = l, b = nCal + 1 - l;
  const lbeta = (p, q) => lgamma(p) + lgamma(q) - lgamma(p + q);
  const lp = [];
  for (let k = 0; k <= nTest; k++)
    lp.push(lgamma(nTest + 1) - lgamma(k + 1) - lgamma(nTest - k + 1) + lbeta(k + a, nTest - k + b) - lbeta(a, b));
  const m = Math.max(...lp);
  const pk = lp.map(v => Math.exp(v - m));
  const tot = pk.reduce((s, v) => s + v, 0);
  const qlo = (1 - level) / 2, qhi = 1 - qlo;
  let cdf = 0, lo = null, hi = null;
  for (let k = 0; k <= nTest; k++) {
    cdf += pk[k] / tot;
    if (lo === null && cdf >= qlo) lo = k / nTest;
    if (hi === null && cdf >= qhi) hi = k / nTest;
  }
  return { l, a, b, mean: l / (nCal + 1), lo, hi, level };
}

function betaBinomCurve(nCal, nTest, alphas = [0.02, 0.03, 0.04, 0.05], level = 0.99) {
  return alphas.map(al => { const r = betaBinomBand(nCal, nTest, al, level); return { alpha: al, lo: r.lo, hi: r.hi, mean: r.mean }; });
}

function holm(pvals, alpha = 0.05) {
  const idx = pvals.map((p, i) => [p, i]).filter(([p]) => isNum(p)).sort((x, y) => x[0] - y[0]);
  const adjusted = pvals.map(() => null), m = idx.length;
  let run = 0;
  idx.forEach(([p, i], j) => { run = Math.max(run, Math.min(1, (m - j) * p)); adjusted[i] = run; });
  return { adjusted, reject: adjusted.map(q => (isNum(q) ? q <= alpha : null)) };
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function rankAvg(x) {
  const idx = x.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), r = new Array(x.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function pearson(x, y) {
  if (x.length < 3) return null;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
}
function spearman(x, y) {
  const ok = x.map((v, i) => isNum(v) && isNum(y[i]));
  const xs = x.filter((_, i) => ok[i]), ys = y.filter((_, i) => ok[i]);
  return xs.length >= 3 ? pearson(rankAvg(xs), rankAvg(ys)) : null;
}
function famDemean(values, families, minSize = 2) {
  const by = {};
  values.forEach((v, i) => { if (isNum(v)) (by[families[i]] = by[families[i]] || []).push(i); });
  const out = values.map(() => null);
  for (const ix of Object.values(by)) {
    if (ix.length < minSize) continue;
    const m = mean(ix.map(i => values[i]));
    for (const i of ix) out[i] = values[i] - m;
  }
  return out;
}
function famDemeanedSpearman(pred, y, families, minSize = 2, minN = 6) {
  const ok = pred.map((p, i) => isNum(p) && isNum(y[i]));
  const by = {};
  ok.forEach((o, i) => { if (o) (by[families[i]] = by[families[i]] || []).push(i); });
  const keep = Object.values(by).filter(ix => ix.length >= minSize).flat();
  if (keep.length < minN) return null;
  const rp = rankAvg(keep.map(i => pred[i])), ry = rankAvg(keep.map(i => y[i])), fam = keep.map(i => families[i]);
  return pearson(famDemean(rp, fam, 1), famDemean(ry, fam, 1));
}

// ---- provenance --------------------------------------------------------------------------------------------------------
function fileSha256(p, opts = { eol: true }) {
  let b = fs.readFileSync(p);
  if (opts.eol) b = Buffer.from(b.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(b).digest("hex");
}
function git(args, repo = REPO) {
  try { return execFileSync("git", ["-C", repo, ...args], { stdio: ["ignore", "pipe", "ignore"] }).toString(); } catch (e) { return null; }
}
const gitHead = (repo = REPO) => { const s = git(["rev-parse", "HEAD"], repo); return s ? s.trim() : null; };
function gitIsAncestor(a, b, repo = REPO) {
  try { execFileSync("git", ["-C", repo, "merge-base", "--is-ancestor", a, b], { stdio: "ignore" }); return true; } catch (e) { return false; }
}
function gitShow(rev, p, repo = REPO) {
  try { return execFileSync("git", ["-C", repo, "show", `${rev}:${p}`], { stdio: ["ignore", "pipe", "ignore"], maxBuffer: 1 << 30 }); } catch (e) { return null; }
}
function gitSha256At(rev, p, repo = REPO) {
  const b = gitShow(rev, p, repo);
  return b ? crypto.createHash("sha256").update(Buffer.from(b.toString("latin1").replace(/\r\n/g, "\n"), "latin1")).digest("hex") : null;
}
function provenance({ repo = REPO, p1 = null, p2 = null, pRun = [], files = [] } = {}) {
  const head = gitHead(repo);
  const rec = { head, p1, p2, p_run: pRun, p1_ancestor_of_head: p1 ? gitIsAncestor(p1, "HEAD", repo) : null,
                p2_ancestor_of_head: p2 ? gitIsAncestor(p2, "HEAD", repo) : null, files: {} };
  for (const f of files) {
    const wt = fs.existsSync(path.join(repo, f)) ? fileSha256(path.join(repo, f)) : null;
    rec.files[f] = { worktree: wt, at_p1: p1 ? gitSha256At(p1, f, repo) : null, at_p2: p2 ? gitSha256At(p2, f, repo) : null };
    rec.files[f].unchanged_since_p1 = p1 ? rec.files[f].worktree === rec.files[f].at_p1 : null;
  }
  return rec;
}

// ---- self-test ---------------------------------------------------------------------------------------------------------
function selftest() {
  const ok = [];
  const chk = (name, c, detail = "") => { ok.push(!!c); console.log(c ? "PASS" : "FAIL", name, c ? "" : detail); };
  chk("lgamma(5) = ln 24", Math.abs(lgamma(5) - Math.log(24)) < 1e-12);
  chk("lgamma(0.5) = ln sqrt(pi)", Math.abs(lgamma(0.5) - 0.5 * Math.log(Math.PI)) < 1e-12);
  const R = betaBinomBand(2500, 1500, 0.05), Mm = betaBinomBand(750, 1500, 0.05);
  chk("X4-1R band n_cal 2500, n_test 1500 = [0.032, 0.070] (T1 review A2)", Math.abs(R.lo - 0.032) < 7e-4 && Math.abs(R.hi - 0.070) < 7e-4, JSON.stringify(R));
  chk("X4-1M band n_cal 750, n_test 1500 = [0.027, 0.077] (T1 review A2)", Math.abs(Mm.lo - 0.027) < 7e-4 && Math.abs(Mm.hi - 0.077) < 7e-4, JSON.stringify(Mm));
  chk("band mean = l / (n + 1)", Math.abs(R.mean - Math.floor(0.05 * 2501) / 2501) < 1e-15);
  // Monte Carlo of the conformal FPR itself (exchangeable continuous scores) falls inside the band ~99% of the time
  let s = 12345; const U = () => ((s = (s * 48271) % 2147483647) / 2147483647);
  let inside = 0; const reps = 400, n = 750, nt = 1500;
  for (let r = 0; r < reps; r++) {
    const cal = Array.from({ length: n }, U).sort((a, b) => a - b);
    const thr = cal[n - Math.floor(0.05 * (n + 1))];                  // p <= alpha  <=>  s > the (n+1-l)-th smallest
    let k = 0; for (let j = 0; j < nt; j++) if (U() > thr) k++;
    if (k / nt >= Mm.lo && k / nt <= Mm.hi) inside++;
  }
  chk("simulated conformal FPR inside the 99% band in >= 97% of 400 runs", inside / reps >= 0.97, inside / reps);
  const cv = betaBinomCurve(2500, 1500);
  chk("rule-5 curve is monotone in alpha", cv.every((c, i) => i === 0 || (c.lo >= cv[i - 1].lo && c.hi >= cv[i - 1].hi)));
  const h = holm([0.01, 0.04, 0.03, 0.005, null]);
  chk("Holm adjusted p", JSON.stringify(h.adjusted.map(v => (v === null ? null : +v.toFixed(6)))) === JSON.stringify([0.03, 0.06, 0.06, 0.02, null]), JSON.stringify(h));
  chk("Holm reject at 0.05", JSON.stringify(h.reject) === JSON.stringify([true, false, false, true, null]));
  chk("spearman ties", Math.abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-12);
  const fam = ["a", "a", "a", "b", "b", "b", "c"];
  // within every family pred and y agree in order, while the family means run the other way
  const y = [1, 2, 3, 11, 12, 13, 50], pred = [21, 22, 23, 1, 2, 3, 0];
  chk("family-demeaned Spearman sees the within-family order (+1), pooled sees the offset (< 0)",
      Math.abs(famDemeanedSpearman(pred, y, fam) - 1) < 1e-12 && spearman(pred, y) < 0);
  chk("families below minSize are dropped", famDemean([1, 2, 3], ["a", "a", "b"])[2] === null);
  const tmp = path.join(require("os").tmpdir(), `b4_stats_${process.pid}`);
  fs.writeFileSync(tmp + "_a", "x\r\ny\r\n"); fs.writeFileSync(tmp + "_b", "x\ny\n");
  chk("fileSha256 ignores CRLF vs LF", fileSha256(tmp + "_a") === fileSha256(tmp + "_b"));
  fs.unlinkSync(tmp + "_a"); fs.unlinkSync(tmp + "_b");
  const hd = gitHead();
  chk("git helpers: HEAD is 40 hex and its own ancestor", hd === null || (/^[0-9a-f]{40}$/.test(hd) && gitIsAncestor(hd, hd)));
  const all = ok.every(Boolean);
  console.log(all ? `selftest PASS (${ok.length} checks)` : "selftest FAIL");
  return all ? 0 : 1;
}

module.exports = { isNum, lgamma, betaBinomBand, betaBinomCurve, holm, rankAvg, pearson, spearman, famDemean,
                   famDemeanedSpearman, fileSha256, git, gitHead, gitIsAncestor, gitShow, gitSha256At, provenance };
if (require.main === module) {
  if (process.argv.includes("--selftest")) process.exit(selftest());
  if (process.argv.includes("--band")) {
    const [n, nt, al] = process.argv.slice(process.argv.indexOf("--band") + 1).map(Number);
    console.log(JSON.stringify(betaBinomBand(n, nt, al)));
    process.exit(0);
  }
  console.log("usage: node scripts/b4_stats.js --selftest | --band <n_cal> <n_test> <alpha>");
}
