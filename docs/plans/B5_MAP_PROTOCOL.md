# Batch 5: Map Protocol pilot (DRAFT)

**Status.** DRAFT, 2026-09-24. Not binding. The owner OK'd the outline on 2026-09-24 (tracks M0-M3, autonomy level B,
DINO/CLIP deferred to batch 6), then an external review. This draft applies the review decisions in section 9. The next
step is design, then implementation and a review workflow, as in batch 4. Summary for the owner (Korean):
`docs/reviews/B4_TAKEAWAYS_B5_APPROACH_2026-09-24.md`.

Batch 5 in one paragraph. Batches 1-4 checked axes we had named: 18 factors and the corruption families. The owner's
final goal is a shared space over senses such as hearing and touch. Those senses have no axis names we can write down in
advance. Batch 5 therefore checks the instrument first, on images:
- Can we record a network's internal space without assuming axes?
- Can we find its axes blind?
- Do the axes we find predict anything on conditions we have not seen?
- Which of them matter for performance, beyond what pixels and the head already give?

---

## 1. What batch 4 hands to batch 5

Sources: `results/b4/SESSION.md` and `ATLAS_STATUS.md` rows 12-19.

| batch-4 result | consequence for batch 5 |
|---|---|
| Clean-input error information is a head function (row 12) | The head is a fixed baseline "brain" and a registered axis |
| Shift information is early and largely pixel-level (rows 13, 14, 16; PC-2b, X3-1, X6-5) | The pixel baseline is mandatory in every comparison |
| Harm lead and benign-change suppression were not shown (X3-3, X3-6) | Every sweep records accuracy loss (the harm curve); the consequence axis in section 4 |
| Collapse is ordered within a family but predicts no model-level outcome (row 18; P7) | M3 tests structure (trajectories), not collapse level |
| The CIFAR-10 hub models and their CIFAR-10-C results have been read (C units, VGG11 outlier) | They cannot confirm a prediction. Sealed material is new (section 6) |

---

## 2. Protocol stages

```
[0 question card] → [1 stimulus design] → [2 extraction] → [3 axis-free record] ─(stability gate)─┐
                          ↑                                                                        ↓
                          └── [6c operationalise] ← [6b critique] ← [6a propose] ← [6 residual] ← [5 axis scoring] ← [4 axis library]
                                                          ↓
                                         [7 pre-registered confirmation] → [8 map card]
```

**0. Question card.** One page per unit of work: models, taps, input domain, the question, and what is out of scope.

**1. Stimulus design.**
- Every input carries full metadata: source image id, class, every applied transform and its parameters, and its
  generation order.
- Five kinds of input:
  - base images;
  - minimal pairs (one factor changed);
  - sweeps (one factor in 7 levels);
  - crosses (two factors);
  - out-of-range inputs.
- Rows are split into reference, discovery and sealed confirmation before any extraction.

**2. Extraction.**
- Every tap, GAP plus a subsampled spatial map.
- The same stimuli also go through three fixed baseline brains:
  - pixels (the input itself, plus the pixel factor set);
  - a random-init network of the same architecture;
  - the head (stored logits).
- Integrity uses the existing hashing, sealing and replay (`atlas/`, `scripts/b4_*`).

**3. Axis-free record.**
- Per tap, a relational object that does not depend on coordinates: distance/Gram matrices (Euclidean and cosine),
  kNN graphs at several k, local ID and density.
- **Stability gate.** All four checks must pass before stages 4-6 run on a unit:

  | check | criterion |
  |---|---|
  | bootstrap | the structure is stable under stimulus subsampling |
  | seeds | seeds agree (CKA, relrep, kNN overlap) |
  | information | class and transform decodability can be recovered from the record |
  | baselines | the record is separable from the pixel and random-init records |

**4. Axis library.**
- An axis is a function of the metadata: continuous, categorical, or a minimal-pair difference.
- Registered at the start:
  - the 18 existing factors (`atlas/factors/`);
  - the transform parameters;
  - class;
  - the head.

**5. Axis scoring.**
- One battery for every axis × tap:
  - partial RSA (controlling the other axes and the pixel and random-init records);
  - a cross-validated probe;
  - minimal-pair displacement (magnitude and cross-image direction consistency);
  - sweep curvature and monotonicity;
  - the layer trajectory (where the axis appears and where it washes out).
- Two separate scores come out of this (section 4.2).

**6. Residual structure.** Partition the relational structure over the known axes. Extract candidate directions from the
unexplained remainder (residual PCA, clustering, sparse autoencoder). Keep only candidates that reproduce across seeds.

**6a-6c. Automatic candidate loop** (autonomy level B: autonomous up to screening; registration and sealed confirmation
need owner approval).
- **6a. Propose.** Batch 5 allows three sources: residual directions, a descriptor bank, and model sensitivity
  (top Jacobian directions). Vision-language descriptions (VisDiff-style) wait for batch 6.
- **6b. Critique.** Check for duplication of a library axis, confounding with a known axis, and whether a stimulus can
  be generated.
- **6c. Operationalise.** Turn the candidate into an intervention stimulus (parametric transform), with a null-edit
  control. Freeze the predicted representation movement before the sealed rows are opened.
- **Budget and stopping.** A cap on candidates and compute per round. BH-FDR at screening. The loop stops after two
  rounds with no drop in the residual fraction.

**7. Confirmation.** Pre-registered claims are scored once on sealed rows, new models and new corruptions. The existing
evaluator and two-verifier procedure is reused.

**8. Map card.** One card per model:
- tap × axis tables;
- the explained and residual fractions;
- cross-seed and cross-model alignment;
- confirmed versus candidate axes;
- the controller column (section 4.3).

---

## 3. Tracks

### M0. Housekeeping
Commit the batch-4 step-E outputs, which are untracked: `results/b4/SESSION.md`, `eval_t1.json`, `t2_eval.json`,
`eval_t3s.json`, `b4a_seals.json`, and the ATLAS_STATUS and knowledge README edits. Commit this draft and the review
summary with them.

### M1. Protocol validation (gate; low information value by design)
- **Stimuli.** About 2000 CIFAR-10 test images, taken from rows the batch-4 confirmation did not use or from a fresh
  split fixed at P1. Six pixel factors:
  - brightness
  - contrast
  - Gaussian blur
  - Gaussian noise
  - rotation
  - hue

  Each factor has 7 levels, plus minimal pairs, a small set of crosses, and CIFAR-10-C as out-of-range input.
- **Units.**
  - ResNet20 s1-s4 (seed axis) and ResNet56 s1/s2 (depth);
  - random-init twins;
  - the pixel and head baselines.
- **Role.** M1 is a sanity check and a gate: M2 and M3 are uninterpretable without it. It is not counted as a
  contribution.
- **Pass criteria (to be fixed at P1).**
  - The stability gate passes on every trained unit.
  - All 6 planted factors are recovered as axes.
  - Each factor is separated from the pixel and random-init baselines where it is learned, and reported as
    pixel-visible where it is not.

### M2. Blind rediscovery (highest priority)

**Design**
- Hide one of the 6 factors from the library, run stages 6-6c, and check whether the loop finds that factor again. Repeat
  for each of the 6.
- Controls:
  - **None hidden.** Nothing is removed. Any "new axis" the loop reports here counts as a false discovery. This measures
    the loop's tendency to find something anyway.
  - **Random init.** The same loop on random-init networks must not report a learned axis.

**Success.** All six steps are required. Naming a residual pattern is not enough.
1. The loop finds structure matching the hidden factor in the residual, on discovery rows. Detection must come from the
   residual or sensitivity path, not the descriptor bank.
2. It generates a candidate factor from that structure.
3. It generates new minimal-pair and sweep stimuli for the candidate. The descriptor bank excludes the exact generator
   parameters of the hidden factors.
4. It pre-registers the predicted representation movement: direction, ordering, and the tap where the factor washes out.
5. The predictions hold on sealed rows.
6. The same learned structure does not appear in the random-init control.

**Match criterion.** Subspace alignment with the hidden factor's direction must reach a threshold. Both the threshold
and the naming rule are fixed at P1, before any loop run.

### M3. Corruption × model size (second priority)

**Questions**
1. Does the corruption displacement type (coherent shift for blur, spread for noise; row 5) hold across families and
   sizes?
2. Does the tap where corruption information washes out move with model size?
3. Secondary: the VGG11 robustness outlier (post-hoc in batch 4) as a stress test. Does a representation signature
   associated with it predict robustness in other models?

**Discovery material**
- The CIFAR-10 hub models, 17 in five families, on discovery rows of the new stimuli.
- The M1/M2 rules.

**Sealed confirmation (new material only; section 6)**
- New stimuli and new corruption transforms, never run on any model. The corruption transforms must not be in
  CIFAR-10-C.
- The CIFAR-100 hub models of the same upstream, 19 in five families (section 6).

**Prediction form (fixed at P1 after M1/M2 discovery)**
- A factor type is kept to a given block and lost after it.
- A family keeps a factor longer.
- Units with a given trajectory have a larger detector increment on the matching corruption.
- A VGG11-like signature predicts robustness on the new corruptions.

---

## 4. Axis classification and scores (review decisions 2, 3, 5 and 6)

### 4.1 Two independent axes, not a ladder
- **Source:** pixel-visible / architectural (present in the random-init network) / learned.
- **Consequence:** none / associated with accuracy loss / associated with the need for a controller action.
- Controller candidacy is decided on consequence. Source decides where to read the axis, which is a cost choice. A
  pixel-visible axis with consequence is a valid controller input, read from pixels. Batch 4's stream detection and
  family identification are examples.
- An axis with a confirmed "none" consequence is kept as a **suppression list** entry. It feeds the benign-change HOLD
  that batch 4 could not show (X3-6), and later F2 gating.

### 4.2 Two scores per axis, kept separate

| representation score | utility score |
|---|---|
| stability (bootstrap) | accuracy-loss association (harm curve along the sweep) |
| seed consistency | failure-risk association (per sample) |
| separability | increment over pixel + head (primary contrast, 4.3) |
| sweep monotonicity and curvature | cross-model transfer (M3) |
| layer-trajectory consistency | lead time: **N/A in batch 5** (no temporal streams) |

A high representation score with a low utility score never makes an axis a controller candidate.

### 4.3 Primary contrast and the controller column
- **Primary:** the representation's increment over **pixel + head** on the utility targets.
- **INFO only:** pixel only, representation only, pixel + representation, head only, head + representation, pixel +
  head. The ladder applies to utility scores only. Representation scores use partial RSA against the pixel and
  random-init records.
- **Controller column on each map card:** the recommended monitoring tap, whether pixels suffice, and the consequence
  class.
- **Two grades per batch:** the primary question (map and axes) and the final goal (the controller function table in
  `docs/knowledge/controller-functions-feasibility.md`). The report lists which cells moved.

---

## 5. Batch-level outcome grade (pre-registered at P1)

| grade | needs | meaning |
|---|---|---|
| A | blind rediscovery (M2 criteria 1-3) | exploratory tool |
| A+B | plus baseline separation (M2 criterion 6, M1 separation) | representation discovery method |
| A+B+C | plus out-of-sample prediction (M2 criteria 4-5 and M3 sealed confirmation of trajectories or damage) | predictive representation framework |

---

## 6. Sealed material

- **CIFAR-10 hub models (17).** Their CIFAR-10-C behaviour and the VGG11 outlier were read in batch 4. They are
  discovery units for M3 and are never used to confirm a robustness prediction.
- **CIFAR-100 hub models (19),** checked on 2026-09-24 with the GitHub release API
  (`chenyaofo/pytorch-cifar-models`, 1.2-108 MB each, about 0.5 GB total):
  - resnet20/32/44/56;
  - vgg11/13/16/19_bn;
  - mobilenetv2 x0_5/x0_75/x1_0/x1_4;
  - shufflenetv2 x0_5/x1_0/x1_5/x2_0;
  - repvgg a0/a1/a2.

  They are never measured before P2. Apply the batch-4 weight checks: size against the release API, sha256 prefix
  against the tag, and the README gate. Use the one-sided gate from B4a, because the group-B upstream augmentation bug
  may apply here too; check the logs branch at design time.
- **New corruption transforms.** A set not in CIFAR-10-C (for example vignetting, chromatic aberration, lens distortion,
  rolling shutter, colour cast, JPEG at new qualities), fixed at P1. Whether to also use CIFAR-100-C is decided at design
  time, after its source is verified.
- **Sealed rows** of every new stimulus set, split at P1.

---

## 7. Out of scope for batch 5

- **DINO/CLIP.** Batch 6 tests whether the map generalises to headless encoders.
- **Generative (semantic) edits.** Batch 5 uses parametric pixel transforms only, where ground truth is certain.
- **Multiple senses and real time axes.** The sweeps reserve the place for a time axis.
- **Closed loop and phase C.**
- **Batch-4 controller follow-ups** (`results/b4/SESSION.md` "What batch 5 should do"): transfer of the shift-time joint
  model, routing by response, the stream harm reference, X4 re-registration, batch estimators. Two exceptions are folded
  in: the pixel baselines (stage 2) and the calibration-row check (stage 3 bootstrap). T2's model-level laws are retired;
  P5a and P6b are kept as candidates. The T3S spatial maps are stored in stage 2 but not analysed.

---

## 8. Sessions and budget

The sequence is the same as batch 4:
1. design;
2. implementation, then a review workflow;
3. P1 (pre-registration, including the M2 match criterion, the grades in section 5 and the M3 predictions template);
4. S1 extraction on an RTX 4090, with sealed dumps;
5. discovery on CPU, including the M2 loop;
6. P2 freeze, with the M3 predictions filled in;
7. S2 confirmation;
8. E: evaluators and two verifiers, then a Korean summary.

**Size.**
- Extraction is about 10^5 32×32 images × about 45 units (6 M1 + random-init twins + 17 CIFAR-10 hub + 19 CIFAR-100 hub).
  That is a few GPU-hours.
- Storage: pooled taps plus a subsampled spatial map. The relational records are computed on the pod. The design must
  fit the 75 GB volume.
- Estimated cost: about $4-7. The balance is $175.

**Open design items.**
- Stimulus counts per condition and the row split.
- Relational metrics and the k values.
- The stability-gate thresholds.
- The M2 alignment threshold and naming rule.
- The descriptor-bank contents, excluding the generator parameters.
- The new corruption list.
- Verification of the CIFAR-100-C source.
- CIFAR-100 group-B README provenance.
- Random-init seeds per architecture.
- The candidate and compute caps per loop round.

---

## 9. External review (2026-09-24): decisions applied

| review item | decision | where |
|---|---|---|
| Framing: batch 4 narrowed the hypotheses | adopted | section 1 |
| M2 = discover → predict → sealed confirmation | adopted, strengthened with a no-hidden control and a match criterion fixed at P1 | M2 |
| Level A/B/C ladder | modified into two independent axes (source × consequence); a pixel-visible axis can be a controller input | 4.1 |
| Separate representation and utility scores | adopted; lead time marked N/A | 4.2 |
| Seven-way comparison ladder | reduced: one primary contrast (over pixel + head); the rest INFO | 4.3 |
| M3 as a pre-registered prediction test | adopted, with sealed material replaced: the CIFAR-10 hub CIFAR-10-C results were already read, so confirmation uses new stimuli, new corruptions and the CIFAR-100 hub models | M3, section 6 |
| VGG11 as a transfer test | adopted as a secondary item (one outlier among 17 gives low power) | M3 |
| Priority M2 > M3 > controller > M1 | adopted for importance; M1 stays first in execution as a gate | M1 |
| Minimum success A / A+B / A+B+C | adopted as the pre-registered batch grade | section 5 |
| Novelty or "strong contribution" as the yardstick | set aside: Atlas grades functional value; prior art (MAIA and others) is adopted and refined, not avoided. The substance (out-of-sample prediction, baseline separation) is kept | — |
| "No performance consequence means no controller relevance" | partly rejected: confirmed-benign axes form the suppression list for HOLD and F2 | 4.1 |
| Many candidate sources raise researcher degrees of freedom | addressed by sealing, caps, FDR and a pre-fixed criterion; batch-5 sources limited to three | 6a |
