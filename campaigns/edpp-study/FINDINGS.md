# EDPP Empirical Study — Findings

Canonical findings doc. Detail/lab-notebook: `out/diag/SESSION_LOG.md`. Headline table:
`out/diag/SUMMARY.md`. Backlog: `TODO.md`. Repro: `out/diag/REPRO.md`.
Superseded artifacts (do NOT cite): `out/diag/ARCHIVE_round-robin-artifact.md` (harness
misconfig) and `out/diag/ARCHIVE_lagged-z-ttft-artifact.md` (the lagged-`z_ttft` flaw — see
"Responsive z_ttft" below).

EDPP is a Lyapunov **time-average** optimizer → report MEANS alongside p99 (means are arguably what it
targets). All results below use the llm-d **weighted PD default** (`precise-prefix-cache:2,queue-depth:1`).

## Framing — two distinct questions
- **Q1: should this workload run disaggregated at all?** Provisioning/topology choice; EDPP doesn't
  control it. Comparing EDPP to `never@4` answers Q1, not EDPP's quality.
- **Q2: given a disaggregated deployment, does EDPP disaggregate the RIGHT requests?** The algorithm's
  actual correctness — baselines hold topology FIXED (always / never-in-split / oracle).
- Q1 for uniform decode-bound synth is settled (below). **Q2 is still the priority** — but the
  responsive-`z_ttft` fix (below) materially changed EDPP's behavior, so prior Q2-flavored verdicts
  ("EDPP actively hurts") are retracted.

## Responsive z_ttft — the fix that changed the EDPP story (2026-06-29)

**The earlier "EDPP is the worst decider / partial disaggregation actively hurts / SLO 0.75" verdict
was an artifact of an engineering flaw, not the algorithm.** `z_ttft` (the TTFT SLO virtual queue)
was updated only at full request completion, so on a saturating decode pool the TTFT-miss signal
arrived ~100s+ late: `z_ttft` was 0 for **81%** of decisions and first went positive **81% through
the run**, leaving the SLO term dormant while EDPP kept loading the failing pod. (The competing
"HOL-blind `ttft_d` predictor" diagnosis was disproved: `ttft_d` enters only as
`z_ttft·(TTFT_P−TTFT_D)`, so it is multiplied by zero exactly when `z_ttft=0`; its accuracy does not
move the decision. Full provenance: `ARCHIVE_lagged-z-ttft-artifact.md`.)

**Fix:** credit `z_ttft` continuously from each in-flight request's observed elapsed wait (a certain
lower bound on its TTFT miss), trued up at first token (new `OnFirstToken` hook) or, as a fallback,
at completion. Same total contribution per request as before — credited earlier — so the virtual
queue's fixed point and the drift-plus-penalty guarantee are unchanged; only the feedback delay is
cut. Design: `docs/superpowers/specs/2026-06-29-edpp-responsive-z-ttft-design.md`.

## Workload: synth (decode-bound), rate 2.0, 2000 reqs, equal 4-node HW

KEY RESULTS (responsive `z_ttft`; pre-fix numbers archived):
- **`never@4` wins** (TTFT p99 0.16s, SLO 100%). Disaggregation does NOT help this decode-bound
  workload at equal hardware — outcome tracks DECODE-CAPABLE NODE COUNT. **(Q1: don't disaggregate;
  add decode nodes.)** Unchanged by the fix (never/always are static deciders).
- **EDPP now tracks `always` within a fixed topology** — it is no longer the worst decider:

  | arm | disagg | decoded | TTFT mean | TTFT p99 | TTFT-SLO@2s |
  |---|---|---|---|---|---|
  | never@4 | 0 | 2000 | 0.07s | 0.16s | 100% |
  | edpp@1P3D | 1625 | 2000 | 1.25s | 19.6s | 99% |
  | edpp@2P2D | 1625 | 2000 | 3.55s | 180s | 98% |
  | edpp@3P1D | 1633 | 1218 | 7.60s | 427s | 98% |
  | always@2P2D | 2000 | — | 0.08s | 0.16s | 100% |

  edpp@2P2D went **75% → 98%** TTFT-SLO (p99 518s → 180s) vs the pre-fix binary, by disaggregating
  breaching requests early. `z_ttft` first positive at **16%** of the run (was 81%), positive for
  **84%** of decisions (was 19%), disagg 52% → 81%.
- **3P1D (1 decode node) is the genuine saturation collapse** — only 1218/2000 decode (nothing to
  balance); the disaggregated prefill still gets fast first-token (SLO 98% on TTFT), but decode is
  capacity-bound. Q1 limit, not an EDPP defect.
- `prefix-threshold` ≈ `always` on synth (tiny inputs trip threshold-16 → ~100% disagg).

## Load-dependence (knee) — RE-MEASURED under the fix; the "rate-1.0 cliff" is GONE
edpp@2P2D across rates (pre-fix → responsive `z_ttft`):

| rate | disagg% | TTFT mean | TTFT p99 | SLO@2s |
|---|---|---|---|---|
| 0.5 | 0.0% | 0.07s → 0.07s | 0.2s → 0.2s | 100% (byte-identical; non-saturated ⇒ untouched) |
| 1.0 | 71.4% | **15.48s → 0.18s** | **162.1s → 0.2s** | 100% (the old sharp cliff is eliminated) |
| 1.5 | 80.4% | 50.74s → 0.54s | 382.0s → 5.1s | 99% |
| 2.0 | 81.2% | 56.32s → 3.55s | 518.3s → 180.2s | 98% |
| 3.0 | 83.2% | 48.29s → 3.46s | 577.4s → 141.0s | 98% |

EDPP now stays healthy through rate 1.5 and degrades gracefully only at 2.0–3.0 (genuine 2-decode-node
saturation). The pre-fix "harmless only at 0.5; sharp cliff at 1.0" finding is RETIRED.

## ITL decision path — RE-MEASURED; conclusion REINFORCED
Tightening `τ_itl` 150→50ms (rate 2.0) now makes EDPP disaggregate MORE under the fix (58% → **90%**),
but **ITL mean stays 72ms** (unchanged; > 50ms target). Disaggregation moves only PREFILL; decode stays
on the same 2 nodes, so ITL is floored by decode capacity. **EDPP's sole lever (prefill placement) is
matched to TTFT/HOL but mismatched to ITL/decode-capacity** — the fix sharpens the response but cannot
beat the capacity floor. NOTE `z_itl` still has the SAME completion-lag flaw `z_ttft` just had
(TODO 10b: apply the responsive-update treatment), though it won't move this capacity-bound ITL.

## Still pre-fix (lower priority)
- **RAG (prefill-bound)** under the corrected default + per-class SLOs — never re-measured post-fix
  (TODO 8). Does the decode-node-count story invert for prefill-bound?

## Anomalies & instrument audits (kept)
- The round-robin "collapse/161s/1192" numbers were a harness misconfiguration (default
  `--routing-policy round-robin` + no per-pool scorers → decode pool unbalanced). Fixed by defaulting
  PD pools to llm-d weighted. **Lesson: audit the routing config (the "knob") before blaming the
  algorithm.** (Archived: `ARCHIVE_round-robin-artifact.md`.)
- The "EDPP actively hurts" verdict was a control-law flaw (lagged `z_ttft`), not the algorithm.
  **Lesson: when a feedback controller looks bad, check WHEN/from-what its feedback updates.**
- `--num-instances N` alone is fatal for PD (needs role'd instances) — use explicit P/D splits.
- per-request ITL is recorded unconditionally on replay (`itl_mean_us`); `--record-itl` is observe-only.
- `slo_class` valid set: critical/standard/sheddable/batch/background (NOT "interactive").

## Q2 — heterogeneous favorable workload: EDPP FAILS (externality-blind) [2026-06-29]
First real Q2 test (requests DIFFER, so there's a right subset to disaggregate). Workload
(`specs/hetero/hetero.yaml`): A (85%, small prefill / substantial decode / tight 200ms TTFT) should stay
LOCAL; B (15%, ~16k prefill / tiny decode / batch SLO) should DISAGGREGATE (its prefill interferes with
co-resident A's TTFT — mechanism verified). 2P2D, decode ADEQUATE (bottleneck = interference, not capacity).

| arm | A TTFT p99 | A SLO viol@200ms |
|---|---|---|
| never-in-split (B local) | 259ms | 2.7% |
| always / prefix-16 (all disagg) | 271ms | 2.5% |
| **edpp** | **256ms** | **2.5%** |
| **ORACLE (disagg B only, A local)** | **51ms** | **0.0%** |

**A 5× tail win (259→51ms) + 100% SLO is achievable; EDPP captures none of it.** EDPP disaggregates
A 0% (correct) but B only 2% — it keeps the interfering big-prefill requests LOCAL, so A is no better
than never-in-split. **Structural, not tuning:** B's huge `W_p` enters only via `balance_term_d =
q_d·(W_p/W*_d)`, and `q_d≈0` when decode is adequate; B's own TTFT is far under its loose 5s SLO so
`z_ttft(batch)=0`. Worse, B's loose SLO inflates `W*_d` (∝τ_ttft), making B's balance term *smaller*
than A's — so lowering `V` disaggregates A before B (backwards). **Root cause: EDPP judges each request
by its OWN class SLO pressure + shared backlog; B's prefill harming A's TTFT is an externality on
neighbors that the rule cannot express.** Detail + fix direction in SESSION_LOG / TODO.

## Prefill-bound RAG (inference-perf batch-summarization) — agg wins; EDPP over-disaggregates [2026-06-29]
Real catalog workload (prefill_decode_ratio 30): vector-qa (78%, ~2k in, standard τ 500ms) + doc-read
(22%, ~60k in, batch). GPU-matched 16 GPUs (4×TP4). vector-qa TTFT SLO-violation@500ms:

| arm | r=1.0 | r=3.0 | r=6.0 |
|---|---|---|---|
| **agg-4 (NO disagg)** | **0%** | **9%** | **53%** |
| prefix-thresh 2P2D | 15% | 76% | 98% |
| edpp 2P2D | 11% | 36% | 90% |
| edpp 1P3D | 11% | 35% | 69% |

**Q1: agg-4 wins at every rate** — GPU-matched dedicated-role splits steal flexible capacity (agg's 4
nodes all prefill via chunked interleaving ≈ 2× a 2P-pool's throughput) and short prefills wait behind
doc-read 60k prefills in the prefill pool. **Q2: EDPP over-disaggregates short vector-qa (60-76%)** —
better than prefix-threshold (≈100%) but far worse than agg. MECHANISM: EDPP predicts `ttft_p < ttft_d`
(disagg faster) for short reqs, but `ttft_p` under-predicts prefill-pool congestion because the clogging
doc-read prefills are RUNNING, not WAITING (`q_p≈0`) — the SAME waiting-vs-running blindness as `ttft_d`
on synth, now on the prefill side. Caveats: GPU-matched framing, single seed.

## SYNTHESIS — EDPP never beats no-disaggregation at equal HW (4 workloads)
- **synth (decode-bound):** `never@4` wins; EDPP ranges harmless→harmful (responsive-`z_ttft` fix makes
  it *behave* sensibly but it still can't add decode nodes).
- **hetero (mixed, decode-adequate):** EDPP *under*-disaggregates the big-prefill B (externality-blind);
  oracle shows a 5× A-TTFT win EDPP misses.
- **RAG (prefill-bound):** `agg-4` wins; EDPP *over*-disaggregates short requests (`ttft_p` blind to
  running prefill-pool congestion).
- **vector-qa-only (clean single class):** `agg` wins on COMPLETIONS (893/1000 @ 0% viol vs PD splits
  ≤532); EDPP completes fewest (294) + adds violations — worst arm. Role dedication starves the
  under-provisioned side; "0% TTFT" on PD splits is a truncation mirage (read completions, not
  TTFT-of-completed).

Common roots: (1) role dedication wastes flexible capacity → lower completion throughput at equal HW;
(2) predictors see only WAITING backlog, blind to RUNNING occupancy (`ttft_d` and `ttft_p`); (3) the
own-class SLO + pool-backlog rule misjudges *when* to disaggregate and the cross-request *externality*.
**CAVEAT (untested — the real open door):** all comparisons are GPU-MATCHED (same 16 GPUs repartitioned).
Disaggregation's production value is INDEPENDENT SCALING (add prefill nodes without stealing from decode),
which equal-HW cannot show. Fair claim: *at equal HW, repartitioning into P/D roles (and EDPP's routing
within it) does not pay off on these workloads* — NOT *disaggregation never helps*. The honest next test,
if pursued, is a non-GPU-matched (independent-scaling) comparison.

## Open (priority: Q2)
- **The externality term** (fix direction from the result above): weight a request's prefill
  interference cost by the *co-resident decode pool's* SLO pressure (the victims'), not the deciding
  request's own `z`. This is the missing ingredient for EDPP to capture the favorable mechanism.
- Robustness: re-run the hetero workload with amplified B (frequency/size) — the structural finding is
  robust but the magnitude (2.7% interference signal) is one operating point.
- Still open: `z_itl` responsive-update (TODO 10b), per-instance backlog coherence (TODO 12),
  RAG re-measure (TODO 8).

## Stage A — estimator-validation harness (2026-07-01)

New `--pd-outcome-trace` CSV (run + replay) emits one row per request pairing realized
admission delay (per prefill/decode sub-req, `T_adm = schedule − enqueue`), TTFT, mean ITL,
E2E, and completion, keyed by `request_id`. `analyze/estimator_validation.py` joins it against
`--edpp-decision-trace` on `request_id` and reports predicted-vs-realized bias, split by
disaggregated × slo_class over completed requests. This validates the *shipped* (waiting-only,
occupancy-blind) forward estimators; it is the baseline Stage C's occupancy-aware roll-forward
will be re-measured against. Measurement-only — no decider/routing change.

**Anchor run** — synth @ 2P2D, rate 2.0, 5000 reqs (bake `blis run --num-instances 4
--trace-output`; replay `--num-instances 4 --prefill-instances 2 --decode-instances 2
--pd-decider edpp --edpp-coeffs coeffs-llama70b-h100-tp4.json --edpp-tau-ttft 2s
--edpp-tau-itl 150ms`). EDPP disaggregated 4545/5000 (91%); 455 kept local. Bias
(median ratio realized/predicted, and tail):

- **ttft_p (disagg, batch):** median 2.08× under-pred; predicted p50 63ms vs realized p50 120ms;
  tail tight (p99 pred 93ms / real 190ms). For disaggregated requests TTFT is prefill-dominated,
  so the decode queue does not enter TTFT — it lands in E2E/ITL.
- **ttft_d (local path, HOL-blind):** median 2.74× but **tail catastrophic — p90 realized 324s vs
  predicted 0.55s (~590×), p99 real 366s vs pred 0.68s**. This is the archived "predicted 14.7s vs
  realized 542s" HOL-blindness (`ttft_d` built from waiting backlog, ignoring RUNNING decode
  occupancy) — reproduced, confirming the harness.
- **Decode admission delay (waiting-only estimate qd_raw/mu_d_nom):** **median ~905× under-pred —
  predicted p50 0.31s vs realized p50 285s** under 2-decode-node saturation. The clearest single
  quantitative case for Stage C (occupancy-aware admission-delay roll-forward).
- **Prefill admission:** median 1.33× (predicted ~0 vs realized ~15.6ms) — the estimator predicts
  no prefill wait; real wait is small.

**Conclusion:** the harness works end-to-end and reproduces the qualitative, tail-heavy
under-prediction of the shipped occupancy-blind estimators. The waiting-only decode signal is
optimistic by ~3 orders of magnitude under saturation. This is the Stage-A baseline.

**Reproduction (one command).** `bash campaigns/edpp-study/repro_stage_a.sh` (tracked) rebuilds
`blis` if needed, bakes the trace, replays 2P2D@edpp with both traces, and writes the bias report to
`campaigns/edpp-study/out/stage_a/bias.json` (out/ gitignored). Deterministic (INV-6): the run above
is byte-identical across invocations. Inputs pinned by the script: model
`meta-llama/llama-3.3-70b-instruct`, coeffs `scripts/calibration/coeffs-llama70b-h100-tp4.json`, spec
`campaigns/edpp-study/specs/synth_rate2.0.yaml` (5000 reqs). Runtime ~110s. **Checkpoint —
`bias.json` must show** (else the harness, not the estimator, regressed): total 5000 / completed 5000;
`disagg=True,class=batch` n=4545 with ttft median-ratio ≈2.085, decode-admission median-ratio ≈904.8
(pred p50 ≈0.312s, real p50 ≈285.8s), prefill-admission median-ratio ≈1.326; `disagg=False,
class=unknown` n=455 with ttft median-ratio ≈2.740, real p90 ≈324s vs pred p90 ≈0.553s. The two traces
the analysis joins are produced by `--edpp-decision-trace` (needs `--trace-level decisions`) and
`--pd-outcome-trace` (no trace-level needed) — both emitted by the same replay invocation.

**Known limitations / follow-ups (not blockers):**
1. Local (non-disaggregated) requests carry no `ParentRequest`, so their `slo_class`/`input_tokens`
   are empty/0 in the outcome CSV (populated only from a parent's `OriginalRequest`). The analysis
   buckets them as `class=unknown` (was: silently dropped by NaN groupby — fixed). A proper fix
   populates class/tokens for local rows from the original request stream.
2. The outcome `completed` flag uses the `RequestE2Es[id] > 0` proxy, which marked 5000/5000 here
   vs the sim's `completed_requests` = 4545 — the proxy is looser than the sim's completion
   definition; reconcile before using `completed` as a hard filter.
3. Realized decode `T_adm` (p50 285s) coexists with realized TTFT p50 120ms on the same disagg
   requests — the decode-side TTFT is measured from decode arrival (omits the pre-decode wait),
   the long-standing "decode-side TTFT understated" issue. Stage A surfaces it; it is orthogonal
   to this harness.
4. PD+EDPP *replay* on this full config completed 4545/5000 (2-decode-node saturation, expected) —
   NOT a parity bug. An earlier smoke run's 23/50 was a small-config horizon artifact, not an
   engine defect. `--pd-outcome-trace` admission-time columns match run byte-for-byte.

## Stage B — corrected work model + per-request validation (2026-07-02)

Corrected the EDPP work formulas to equal the work the **active (trained-physics) latency model
actually charges**, and validated per-request against realized trajectory work via a new
`--edpp-work-trace` CSV (run + replay). Same frozen coeffs, no refit.

**Corrected formulas** (`sim/edpp_coeffs.go`):
- `Wp(a_p, a_r) = C_pf·a_p + C_attn·a_p·(a_r + a_p/2)` — the trained-physics basis (**+ a_p/2**), matching
  `trained_physics_model.go`'s per-step attention charge `ti·(si + ti/2)` (si = full prompt) that
  `C_attn` was calibrated to. Replaces the old no-cache `C_pf·a_p + (C_attn/2)·a_p²`.
- `Wd(a_r, o) = C0·o + C1·o·(a_r + (o−1)/2)` — exact discrete decode sum `Σ_{k=0}^{o−1}(C0+C1(a_r+k))`,
  matching the model's per-decode-step charge `C0 + C1·ProgressIndex`. Replaces the old
  `N̂_out·(C0 + C1·NomDecodeCtx)` fixed-nominal form (`NomDecodeCtx` retained only for the
  `selectedDecodeState` fallback).

**Validation (synth@2P2D and rag@2P2D, rate 2.0, 5000 reqs each; `bash campaigns/edpp-study/repro_stage_b.sh`):**

| workload | single-chunk prefill `max_abs_rel_err` | decode `max_abs_rel_err` | chunked prefill (n, max err) |
|---|---|---|---|
| synth | 5.6e-16 (n=4701) | median 0, p90 2.5e-4 (n=4456; small preemption-tail, max ~10%) | n=299, max 0.28 |
| rag   | 5.8e-16 (n=1418) | 1.2e-15 (n=5000, fully exact) | n=3582, max 0.22 |

**Model is exact.** Single-chunk prefill and decode work match realized to float precision on both
workloads — the corrected `W_p`/`W_d` ARE the closed forms of the active latency model's per-request
charge. The chunked-prefill residual (realized < closed, median ~−0.9%) is the **documented,
expected** `C_attn·(a_p² − Σs_r²)/2` term: the closed form assumes single-chunk prefill, so chunked
requests accrue slightly less attention work. This is reported, not an error.

**Correction effect:** per-request `wp_closed / wp_closed_nocache_old` median ≈ 1.04 on both (the linear
`C_pf·a_p` term dominates at these operating points; the ~3× attention-basis change is only visible
where `a_p ≈ a_r`). NOTE: synth here shows median `cache_hit_frac` 0.91 (heavy prefix reuse — synth is
NOT a no-cache workload as earlier assumed), rag 0.18. The corrected cross term (`a_r` dependence) is
what the old no-cache form dropped.

**Expected decision shift (NOT a regression):** the corrected `W_p` attention term differs ~3× from the
old form at no-cache, so EDPP routing decisions shift vs pre-Stage-B. This is the correction working.
Precise pre/post disagg-fraction quantification is a deferred follow-up (requires a pre-Task-1 rebuild);
the per-request work exactness above is the correctness gate, not the decision delta.

### Documented latency-model fidelity gap (deferred, NOT fixed here)

The default trained-physics latency model charges prefill attention on a **full-input-length** basis
(`ti·(a_r + ti/2)`), which over-counts physically-causal (triangular, `≈ N²/2`) attention by up to ~3×
at single-chunk. `C_attn` is calibrated to this basis, and Stage B's `W_p` matches it **deliberately**
so EDPP's congestion estimate is consistent with the simulator it runs in — the correct choice for
routing-decision fidelity. The **roofline** backend (`roofline.go`) uses the physically-causal
prior-context basis (`≈ a_r − a_p/2`), which matches design-doc §3.6. So §3.6's causal form is
physics-pure but inconsistent with the default model; Stage B's `+a_p/2` matches the default model.
Fixing the trained-physics basis to causal + refitting `C_attn` (invalidating the frozen coeffs and all
prior findings) is a separate, deferred task. If a future study runs the roofline backend, `W_p` must
switch to the causal basis; the accumulator already reads whichever model is active.

**Reproduce:** `bash campaigns/edpp-study/repro_stage_b.sh` → `campaigns/edpp-study/out/stage_b/{synth,rag}_bias.json`
(~4 min, deterministic). Checkpoint: single-chunk prefill and decode `max_abs_rel_err` < 1e-6 on both;
chunked residual is the `Σs²` term.

## Stage C — occupancy-aware admission estimator + fidelity ablation (2026-07-02)

Made EDPP's admission-delay estimate pluggable (replacing the occupancy-blind `qD/muDec`·`qP/muPf`
terms in `ttft_d`/`ttft_p`) with four deployable variants — `waiting` (shipped strawman), `little`
(Little's law, aggregate), `fluid` (occupancy-conditioned mean-field `N_ahead/X̂_dep`), `rollforward`
(true per-request deterministic batch look-ahead) — plus `fluid_oracle`/`rollforward_oracle`
(true-`o_r`, logging-only; INV-9 guard forbids them as routing drivers). New `--edpp-admission-trace`
logs, per request×pool, the realized admission delay + all six predictions. `local_t_adm` capture
(Stage A gap) closed. Default `waiting` → decider byte-identical.

**Microbenchmark** (`repro_stage_c.sh`): 1P1D, `--pd-decider edpp`, synth rate 2.0; T1 forces all-local
(`--edpp-c-xfer 100s`) to isolate `ttft_d`; T2 forces all-disagg (`--edpp-c-xfer 0s`). Routing is
forced via the transfer-penalty knob (EDPP's `Decide` must run to assemble the estimator contexts, so
`never`/`always` deciders can't be used). Median ratio realized/predicted (>1 = under-prediction):

**T1 local pool — `ttft_d` isolation, saturated (realized p50 ≈ 560s):**
| estimator | median ratio realized/pred |
|---|---|
| waiting (shipped) | **57.3×** under-predicts |
| little | n/a (predicts 0) |
| fluid | 2.6e6× (anomalous — see below) |
| **rollforward** | **1.29× — near-exact** |

**Headline: the true `rollforward` estimator collapses the shipped estimator's ~57× admission
under-prediction to ~1.3× on the saturated local decode pool** — the direct fix for the mechanism
Stage A measured (~905×). N̂_out-prediction error was ~0 here (`rollforward` ≈ `rollforward_oracle`),
so the residual 1.3× is estimator *form*, not output-length prediction.

**Open issues the ablation surfaced (the clean 4-point monotonic collapse is NOT yet achieved):**
1. **`fluid` is anomalous** — it under-predicts by ~1e6× (predicts ~µs where reality is ~100s). The
   mean-field `N_ahead/X̂_dep` with the current `RemainingStepsEst`/free-slot inputs collapses to
   near-zero on the local path. Needs debugging (likely `RemainingStepsEst` mis-scaled or the free-slot
   early-return misfiring). It should sit between `waiting` and `rollforward`, not below both.
2. **`little` is inert** — predicts 0 (ratio n/a). Despite the Task-3 `DispatchRate→AdmissionRate`
   wiring, the admission rate is effectively unavailable at decision time in this run (DispatchRate 0
   until first completion, and/or not surfacing). Needs a reliable per-instance admission-rate signal.
3. **Prefill-pool estimators are broken** — realized prefill admission is a constant ~15.5ms and no
   estimator tracks it; the snapshot enrichment added only the *decode* running batch (`RunningDecode`),
   so the prefill-pool context has no occupancy. `ttft_p` cannot be validated until the prefill pool is
   enriched symmetrically.
4. **`rollforward` OVER-predicts on the disagg decode path** (T2 decode: 0.41×, i.e. predicts ~2.5×
   too high). In disaggregation the decode sub-request is admitted quickly after transfer, so the
   full-batch-drain assumption over-estimates; the 905×-analog under-prediction lives on the *local*
   (kept-local-under-saturation) path, which T1 captures and `rollforward` fixes.

**Status:** the estimator infrastructure (pluggable interface, four variants + oracle, INV-9 guard,
`--edpp-admission-trace`, `local_t_adm`) is complete and reviewed. The **proposed `rollforward`
estimator is validated on its target scenario** (57×→1.3× on the saturated local pool). The `fluid`
under-prediction, `little` inactivity, and prefill-pool enrichment are **follow-ups required before the
full monotonic ablation and the paper's fidelity figure are publishable**.

**Reproduce:** `bash campaigns/edpp-study/repro_stage_c.sh` → `out/stage_c/{t1,t2}_ablation.json`.
Checkpoint: T1 local `waiting` median ratio ≈ 57×, `rollforward` ≈ 1.3×.
Limitations carry Stage B's (trained-physics attention basis); single saturating operating point.

### Stage C follow-up: utilization sweep (validity of the saturating operating point)

The T1/T2 microbenchmark uses a single saturating operating point (synth rate 2.0 on one decode
engine). Under overload the backlog grows and admission delay is non-stationary (each request waits
longer than the last), so the reported median ratios summarize a moving target — a deliberate stress
test, not a steady-state claim. The per-request *conditional* comparison (predicted vs realized at each
request's own decision instant) is valid regardless, and is precisely why the conditional `rollforward`
tracks the growing delay while the aggregate `little` cannot. REQUIRED FOLLOW-UP before the paper's
fidelity figure: a **utilization sweep** at loads approaching but below single-engine capacity, where
admission delay is large but bounded/stationary, so predicted-vs-realized has a well-defined
steady-state meaning. Pair it with the fluid/little/prefill fixes.

### Stage C CORRECTION (2026-07-05): the 57×→1.3× headline is oracle-contaminated

A root-cause investigation of `fluid`'s under-prediction surfaced a **measurement-validity bug**: the
deployable `rollforward`/`fluid` and their `_oracle` variants are the same impl and read
`RunningReqState.TrueRemaining` whenever it is ≥0. Enabling `--edpp-admission-trace` turns on oracle
mode, which POPULATES `TrueRemaining`. So in the T1/T2 microbenchmark the "deployable" `rollforward`
was actually reading oracle remaining — which is why `rollforward` and `rollforward_oracle` were
identical (both 1.29×) and the decomposition showed ~0 N̂_out error. **The 1.29× "rollforward fixes
57×→1.3×" result is oracle-fed; the true deployable-`N̂_out` number is UNKNOWN pending a fix that
prevents deployable estimators from reading `TrueRemaining`.** Also found: `RemainingStepsEst` collapses
to 1 (RemainingDecodeWork never populated + a mean-based fallback that goes negative under saturation),
and `N̂_out` is biased low under saturation (survivorship — only short requests have completed), which
inherently limits the *deployable* estimators. These are being addressed as a fix-cluster (fluid wave
mean-field + oracle/deployable separation + N̂_out handling + little admission-rate + prefill enrichment
+ CLI flag). Treat all Stage C ablation numbers above as PROVISIONAL until the fix-cluster lands.

### Stage C — DE-CONFOUNDED RESULTS after fix-cluster (2026-07-05, supersedes the provisional/1.29× numbers)

All six fix-cluster items landed (fluid wave form; censored per-request remaining-steps; windowed
admission-rate; oracle/deployable separation; prefill enrichment; `--edpp-tadm-estimator`) plus the
second round (rollforward queue-waves; `little` pool-filtered-snapshot fix). `repro_stage_c.sh` re-run,
median ratio realized/predicted (>1 = under-prediction, <1 = over-prediction; ~1 = accurate):

| pool (workload) | waiting | little | fluid | rollforward | rollforward_oracle |
|---|---|---|---|---|---|
| **local (T1, realized p50 ≈ 560s)** | **57.3×** | 1.30× | **1.16×** | **1.16×** | 1.25× |
| decode (T1) | 12.9× | 1.81× | 0.98× | 0.98× | 0.98× |
| decode (T2) | 4.05× | 2.70× | 0.34× | 0.34× | 0.34× |
| local (T2) | 1430× | 8.4× | 0.35× | 0.35× | 0.35× |

**Findings:**
- **Occupancy-blindness is the dominant error.** `waiting` (waiting-backlog ÷ drain) under-predicts
  admission delay 57×–1430× on saturated pools — the mechanism Stage A first measured.
- **The occupancy-aware estimators fix it.** On the target T1 local pool, `fluid`/`rollforward` reach
  1.16× (near-exact) and `little` 1.30×. The key ingredient is accounting for the queue-ahead
  (`fluid`'s wave form; `rollforward`'s queue-depth-aware departure walk).
- **`fluid` and `rollforward` converge** (identical on T1 local/decode): once `rollforward` rolls
  through `⌈(QueueDepth+1)/BatchSize⌉` waves, its deep-queue behavior IS the fluid wave form — so the
  "true per-request" estimator and the mean-field agree where the queue dominates. The per-request
  walk's extra fidelity matters only for shallow queues.
- **Oracle ≈ deployable** (T1 local 1.16 vs 1.25; decode identical): after the **censored `N̂_out`
  floor** (a request that produced `k` tokens has `o_r ≥ k`), the deployable estimate closes almost all
  of the gap to the true-remaining oracle. The `N̂_out`-prediction residual is small here.
- **T2 (disaggregated) decode/local slightly OVER-predict (0.34–0.35×):** a disaggregated decode
  sub-request is admitted quickly after transfer, so realized decode admission is smaller than the
  full-queue-drain the occupancy estimators assume. Still far better than `waiting`.
- **Prefill pool reads ~0 (nan ratio) — correct physics, not a bug.** The prefill pool is unsaturated
  at this operating point (QueueDepth=0, free slot), so the occupancy estimators correctly return ~0
  (`waiting` is live on 2086 prefill rows). Validating prefill estimators requires a prefill-saturating
  workload (follow-up).

**Retraction:** the earlier "rollforward fixes 57×→1.3×" headline was oracle-contaminated (deployable
estimators were reading the oracle `TrueRemaining`). De-confounded, the honest headline is:
**occupancy-blind `waiting` (57×–1430×) → occupancy-aware `fluid`/`rollforward` (~1.16× on the saturated
local pool, converging), with `little` a decent aggregate (1.3×) and the censored-`N̂_out` deployable
estimate ≈ the oracle.** Remaining follow-ups: the utilization sweep (bounded/stationary operating
points), prefill-saturating validation, and the prefill oracle-semantics nuance (prefill "remaining" is
known input length, not a hidden variable — the oracle/deployable split is decode-specific).

---

## Utilization sweep — decode-pool admission fidelity vs load (2026-07-06)

**Purpose.** Harden the single-point Stage C result (measured at forced heavy overload — a
*non-stationary* stress test) by measuring decode-pool admission-estimator fidelity across a range of
**bounded, stationary** operating points below capacity. Topology = Stage C T1 (1P1D, EDPP,
`--edpp-c-xfer 100s` ⇒ all-local on a single decode engine), synth, measurement-only. Reproduce:
`bash campaigns/edpp-study/repro_utilization_sweep.sh` → `out/utilization_sweep/sweep.json` (+`sweep.png`).

**Capacity.** λ* = **0.75 req/s** (composite saturation detector; lowest coarse-scan rate not STABLE —
0.1/0.25/0.5 STABLE, 0.75 first non-STABLE). The ρ grid `{0.5,0.7,0.85,0.9,0.95,0.98}·λ*` yields the
points below; the achieved ρ̂ is measured per point (`responses_per_sec/λ*`).

**Result — the sweep worked, and it reveals a STEP FUNCTION, not a smooth fidelity curve.**
Admission delay is bounded and **stationary** at every point (admission-delay drift = median(2nd
half)/median(1st half) ≈ 1.0 throughout), and grows only gently with load:

| ρ̂ | verdict | realized adm p50 | ITL mean | `waiting` | `fluid`/`rollforward` | `little` |
|------|-----------|-----------------|----------|-----------|-----------------------|----------|
| 0.50 | STABLE    | 29.0 ms | 27.2 ms | ≈0 (1e14× off) | **predict 0** (nan) | over-pred ~17× |
| 0.69 | STABLE    | 33.1 ms | 36.1 ms | ≈0            | **predict 0** (nan) | over-pred ~17× |
| 0.84 | STABLE    | 38.8 ms | 47.4 ms | ≈0            | **predict 0** (nan) | over-pred ~25× |
| 0.88 | STABLE    | 40.8 ms | 52.8 ms | ≈0            | **predict 0** (nan) | over-pred ~16× |
| 0.93 | STABLE    | 44.2 ms | 59.5 ms | ≈0            | **predict 0** (nan) | over-pred ~18× |
| 0.95 | OVERLOADED| 46.9 ms | 64.4 ms | ≈0            | over-pred ~90×      | over-pred ~2× |

(Ratios are `realized/predicted`; a `predict 0` yields nan/astronomical ratios, so read the **signed
error** at these points — it ≈ the full realized floor, i.e. the estimators contribute ~0.)

**Mechanism (why the occupancy estimators predict 0 below the cliff).** Every occupancy-aware estimator
opens with the same free-slot early-return (`fluid` at `sim/admission_estimator.go:66`, `rollforward` at
`:84`): `if BatchSize < MaxBatchSize && FreeKVBlocks >= ReqKVNeed { return 0 }`. Below saturation a slot and KV are free, so it fires and
returns **exactly 0** — modelling "a slot is free ⇒ admitted instantly." But the next `FormBatch` only
runs after the **current in-progress decode step finishes**, so a request enqueued mid-step still waits
the residual of that step (≈ one `T_iter`) plus dispatch-tick cadence before admission. The estimators
model **slot/KV availability**, not the **time to the next admission opportunity**. `waiting` (QWork/Mu)
also predicts ≈0 because the waiting queue is near-empty below capacity.

**This dropped term is confirmed to be ≈ one decode iteration.** The realized sub-saturation admission
floor tracks ITL (inter-token latency ≈ decode step time `T_iter`) closely and climbs with it as the
batch fills: 29↔27 ms at ρ̂=0.5, 44↔60 ms at ρ̂=0.93. So the floor the estimators omit **is** the
residual-step / `FormBatch`-cadence latency, ≈ `T_iter`.

**Interpretation (the honest headline).** For this decode-bound long-job workload the admission-delay
curve is a **step function**: a small (~30–47 ms), routing-irrelevant floor ≈ one decode step for all
ρ̂ below the saturation cliff, then an explosion above it (Stage C's 560s regime). There is **no wide
"large-but-bounded" band** to draw a smooth fidelity curve through — the transition is sharp, and
refining λ*/the grid cannot manufacture a band the physics does not produce. Consequences:
- **Stage C's 57×→1.16× win is real but regime-specific** — it validates the estimators where slot-wait
  dominates (heavy overload). Below the cliff the estimators correctly report ~0 *slot-wait*, but omit
  the O(`T_iter`) admission-opportunity latency that is the entire admission delay there.
- **This gap is routing-irrelevant at these SLOs:** the floor (~30–47 ms) is ≪ τ_ttft = 2 s, so z_ttft
  does not engage on it and it cannot change an EDPP routing decision. It is a documented modelling gap,
  not a routing defect.
- The occupancy-aware estimators are therefore *sufficient for routing* (accurate exactly in the regime
  that crosses the SLO and drives decisions) while structurally incomplete as a general
  admission-latency model (they miss the sub-saturation step-cadence floor).

**Reproduction checkpoint** (a correct re-run must reproduce): λ* = 0.75; admission-delay drift ≈ 1.0 at
every retained point (stationary); realized adm p50 rising ~29→47 ms across ρ̂ 0.5→0.95 and tracking ITL
mean; occupancy-aware estimators predict 0 (nan ratio, signed error ≈ realized) at all STABLE points;
the sole OVERLOADED point at ρ̂≈0.95. If instead a smooth monotonic bias curve appears, the harness or
the topology changed. **Numerical caveat:** below the cliff the realized delay is a small floor and the
estimators predict 0, so the *ratio* is meaningless — read the **signed error** (ms), not the ratio, in
this regime (the analyzer flags `ratio_meaningful` against a `ratio_floor_us` for exactly this reason).

**FLOOR IMPLEMENTED (2026-07-06, commit 1f2d4bd; design
`docs/superpowers/specs/2026-07-06-edpp-admission-floor-design.md`).** The occupancy-aware estimators
(`fluid`/`rollforward`/`little`, + oracles) now lower-bound their admission estimate by one `T_iter` via
`flooredTAdm(est,ctx)=max(est,TIter)` (the wait for the current decode step to finish before the next
`FormBatch`). `waiting` is left unfloored — it is the occupancy-blind strawman, and leaving it untouched
preserves the byte-identical default driver.

Re-run result (`repro_utilization_sweep.sh`) — the floor closes the sub-saturation gap:

| ρ̂ | verdict | realized p50 | `waiting` | `fluid`/`rollforward`/`little` (floored) |
|------|-----------|-------------|-----------|------------------------------------------|
| 0.50 | STABLE    | 29.0 ms | ≈0 (huge ratio) | **1.26×** |
| 0.69 | STABLE    | 33.1 ms | ≈0 | **1.17×** |
| 0.84 | STABLE    | 38.8 ms | ≈0 | **1.12×** |
| 0.88 | STABLE    | 40.8 ms | ≈0 | **1.06×** |
| 0.93 | STABLE    | 44.2 ms | ≈0 | **1.06×** |
| 0.95 | OVERLOADED| 46.8 ms | ≈0 | **1.05×** |

The occupancy-aware estimators now track the realized floor to ≈1.05–1.26× across the whole STABLE band
(best near ρ→1; slightly high at low load because one `T_iter` marginally under-shoots the realized
floor); `waiting` still predicts ≈0 (unfloored strawman, still the occupancy-blind baseline).

**Stage C overload headline preserved, with one honest nuance.** Re-running `repro_stage_c.sh`: T1 local
`waiting` = **57.3×** (unchanged), occupancy-aware `fluid`/`rollforward` = **~1.24×** (`little` 1.30×,
oracles ≈ deployable). The occupancy-aware figure shifted slightly from the pre-floor **1.16×** — this is
NOT the floor leaking into the saturated path: `max(est, T_iter)` is a no-op for genuinely saturated rows
(est ≈ 560 s ≫ `T_iter`), so no saturated prediction changed. The T1 run aggregates over the whole
trajectory including the sub-saturation ramp-up, and those early free-slot rows now floor to `T_iter`
instead of 0, nudging the median. Per-row the saturated predictions are identical; the headline (blind
`waiting` 57× vs occupancy-aware ~1.2×) stands, and the occupancy-aware estimators are now consistent
(~1.05–1.26×) across the entire load range instead of collapsing to 0 below the cliff.

The floor is estimator-fidelity only — routing-irrelevant (the floored delay ~30–47 ms ≪ τ_ttft = 2 s, so
`z_ttft` never engages on it) — and `waiting` (the default driver) is byte-identical.

## Counterfactual regret — per-decision hindsight diagnosis of reduced-EDPP (2026-07-07)

**Purpose.** An *exact* per-decision one-step-deviation regret for the shipped (reduced) EDPP
decider. Capture the policy's realized per-request `(decode, prefill)` plan, then for each sampled
request replay the *entire* plan with only that one request's action changed to each alternative
`a ∈ 𝒜 \ {plan(r)}`, and measure the change in aggregate goodput (`slo_attainment`). Regret(r) =
`max_a goodput(dev_{r,a}) − goodput(baseline)`, clamped at 0. Positive regret ⇒ some single
alternative decision would have improved total goodput in hindsight, i.e. EDPP left goodput on the
table at that decision. This is a *local* diagnostic (one decision moves at a time), **not** the
global P/D optimum — that is the later MILP yardstick's job.

**Setup.** Topology **1P2D** (`--num-instances 3 --prefill-instances 1 --decode-instances 2`),
decider `edpp` with the frozen Llama-70B/H100-TP4 coeffs, `τ_ttft=2s`, `τ_itl=150ms`,
SLO `ttft batch=2s, itl batch=150ms`. **Admission-delay driver = `rollforward` (occupancy-aware).**
`repro_counterfactual.sh` defaults to `--edpp-tadm-estimator rollforward` — we are evaluating EDPP's
routing QUALITY, so it must route with the occupancy-aware estimator, not blis's default occupancy-blind
`waiting` strawman (`ESTIMATOR=waiting` reproduces the older blind-driver run cited in the comparison
below; the estimator-BIAS scripts `repro_stage_a`/`_b` intentionally keep `waiting`). Trace:
`specs/synth_cf.yaml` — the decode-bound synthetic-data workload, shrunk to **800 requests at
aggregate_rate 2.0** so the cluster saturates (baseline goodput <1.0) while each single-request
deviation still has measurable leverage on the aggregate (the full 5000-req trace dilutes one row to
≈0). `|𝒜| = 2 decode × (1 prefill + local) = 4`, so 3 deviations per sampled request. Runs use **K=10**
(30–36 sim runs); scale K up for tighter statistics.

**Self-consistency gate (REQUIRED, passed).** Replaying the captured plan via `--pd-plan` reproduced
the baseline exactly (replay `slo_attainment` == baseline, INV-6/INV-13). The fixed-plan decider is a
faithful record/replay of EDPP's decisions, so the regret below is meaningful.

**Result (K=10, `rollforward` driver).** baseline_goodput **0.99**, mean_regret **0.006**, total_regret
**0.06**, frac_positive **0.60** (6 of 10 sampled). **All 6 positive-regret decisions have the same
hindsight-best: pin the decode node to `instance_1`** (3 as `instance1-local`, 3 as `instance1-instance0`
— i.e. get decode onto `instance_1`, whether kept local or disaggregated); each recovers ~0.01 goodput.
The other 4 sampled decisions have zero regret (baseline is locally best).

**Occupancy-aware vs occupancy-blind driver (the check that prompted this):**

| driver | baseline goodput | total regret | frac positive | where regret concentrates |
|---|---|---|---|---|
| `waiting` (occupancy-blind, blis default) | 0.9775 | 0.1387 | 0.70 | decode-node placement (kept-local) |
| **`rollforward` (occupancy-aware, now default)** | **0.99** | **0.06** | **0.60** | decode-node placement (`instance_1`) |

Routing with the occupancy-aware estimator both **raises baseline goodput (0.9775→0.99)** and **roughly
halves the leftover regret (0.1387→0.06)** — a concrete reason to drive routing with `rollforward`, not
`waiting`.

**Interpretation (the finding survives the occupancy-aware driver).** Even with `rollforward` driving,
reduced-EDPP leaves ~0.06 goodput on the table, and the residual is **still decode-node placement, not a
P/D-split error**: every positive-regret decision is improved by putting decode on `instance_1`
(regardless of local-vs-disagg). This is exactly the pool-average-structure critique — EDPP commits no
per-instance decode target and delegates `d` to the default scorer, and that delegated placement is
where the hindsight-better single decision lives. The occupancy-aware estimator sharpens the P/D
*decision* (fewer, cleaner disagg calls, higher baseline) but cannot fix decode-node *selection*, which
it does not control. **This is the direct empirical motivation for the full-joint rule [C]** (choose `d`
by the drift objective over all of ℳ, not a scorer): the open hypothesis it must confirm is that the
joint argmin recovers this residual `instance_1` regret. Caveat unchanged: this is a LOCAL one-step
deviation, not the global optimum (the MILP yardstick's job).

*Caveat:* because EDPP's local decisions carry an empty decode instance in the outcome trace, a
"deviation" on a local request also *pins* a decode target the baseline had left free — so this
regret blends "should have disaggregated" with "should have pinned a better decode". The gate passing
(empty-override replay == baseline) confirms the baseline is captured faithfully; the deviations are
genuine alternatives to it.

**Hand-case validation.** On a tiny idle 2-request 1P2D trace the harness reproduces the known answers
(recorded in `repro_counterfactual.sh`): all-local baseline under a loose SLO ⇒ regret 0 (local is
already optimal, no invented gains); all-disaggregate baseline under a tight `ttft=40ms` SLO (so the
KV-transfer hop misses while local meets) ⇒ positive regret with a *local* hindsight-best.

**Reproduce:** `bash campaigns/edpp-study/repro_counterfactual.sh` (`K=…`, `TARGET_POLICY=…`,
`SPEC=…` overridable). Diagnostic only (local one-step deviation), not the global optimum; it is the
shared fixed-plan (`--pd-plan`) infrastructure for the later full-joint decider and the MILP yardstick.

## Joint mechanism (sub-project 1) — reduced vs `--edpp-joint` sweep (2026-07-08)

**What.** The reduced EDPP decider delegates the decode-node choice `d` to the composable scorer and
only decides local-vs-disaggregate for a *fixed* `d`. The **joint** decider (`--edpp-joint`) instead
enumerates every `(decode, prefill)` candidate over the pools and picks the drift-plus-penalty argmin.
The counterfactual-regret finding above closed on a concrete hypothesis: reduced-EDPP's leftover regret
is decode-node *placement* (every positive-regret decision was improved by moving decode to
`instance_1`), which is exactly what a joint argmin over `d` should be able to recover. This is the test
of that hypothesis. **Scope: HOMOGENEOUS hardware** — the joint objective's only active levers here are
cache warmth and per-instance occupancy; per-instance hardware heterogeneity (`θ_i`) is deferred. It is
a **LOCAL diagnostic** (the regret is one-step-deviation, not the global P/D optimum — the MILP
yardstick's job).

**Correctness gates (all PASS, run first — see `repro_joint.sh` stage 1).**
- **(a) byte-identical off-path.** Two reduced-EDPP runs (`--edpp-joint` OFF) produce byte-identical
  metrics **and** decision trace — the joint plumbing does not perturb the shipped reduced default (INV-6).
- **(b) §5.5 reduction.** `TestJoint_ReducesToScorerSliceMatchesReduced` passes: the joint objective
  restricted to the scorer's single decode reproduces the reduced local-vs-disagg decision.
- **(c) joint-plan self-consistency.** Capturing the joint decider's realized `(d,p)` plan and replaying
  it via `--pd-plan` reproduces the joint baseline `slo_attainment` exactly (INV-6/INV-13). **Caveat that
  drove a harness fix:** the `--pd-outcome-trace` records `decode_instance` only for *disaggregated*
  requests (empty for local); on replay an empty decode falls back to the scorer's decode. That is
  faithful for reduced (which never overrides the scorer's decode) but NOT for joint, whose whole point
  is to override the *local* decode. So the joint plan is captured from the **`--edpp-joint-trace`**
  (which logs `joint_d`/`joint_p` for every request), not the outcome trace. With that source, replay ==
  baseline (gate passes). The joint routing itself is faithful; only the outcome-trace capture path
  could not represent a local-decode override.

**Sweep (K=4 sampled deviations/policy/cell — SMALL, to bound runtime at `K·(|𝒜|−1)` sims per policy;
`|𝒜|=4` at 1P2D, `6` at 2P2D. Numbers shift slightly with K; the *direction* is stable across K=4/6).**
Cells = {1P2D, 2P2D} × {`synth_cf` cache-uniform (shared 2000-tok system prompt), `synth_asym`
cache-asymmetric (unique large prompts, no shared prefix)}. Both policies use
`--edpp-tadm-estimator rollforward`. SLO/τ: `ttft 2s`, `itl 150ms`.

| cell | workload | goodput reduced | goodput joint | regret reduced | regret joint | any-divergence | p-divergence | dir: joint lower-J / tie |
|---|---|---|---|---|---|---|---|---|
| 1P2D | synth_cf (uniform) | **0.990** | 0.979 | 0.030 | **0.0225** | 0.320 | 0.000 | 10% / 90% |
| 2P2D | synth_cf (uniform) | **0.990** | 0.979 | 0.030 | **0.0225** | 0.320 | 0.000 | 10% / 90% |
| 1P2D | synth_asym (asym)  | **1.000** | 0.999 | **0.000** | 0.0012 | 0.328 | 0.000 | 20% / 80% |
| 2P2D | synth_asym (asym)  | **1.000** | 0.999 | **0.000** | 0.0012 | 0.536 | **0.260** | 15% / 85% |

> **CORRECTION (2026-07-09, K=50) — the "~25% regret cut" below was a small-sample (K=4) artifact and is
> RETRACTED.** Re-running the 1P2D `synth_cf` cell with K=50 sampled requests (same seed; see
> `campaigns/edpp-study/repro_policy_comparison.sh` / `out/policy_cmp_k50/`) reverses the direction:
> **reduced total_regret 0.2188 (frac_positive 0.44) vs joint 0.3213 (frac_positive 0.46)** — joint's regret
> is ~47% *higher*, and its goodput is lower (0.979 vs 0.990). So on this homogeneous decode-bound cell
> **joint-EDPP does NOT beat reduced-EDPP — it is slightly worse on both goodput and regret.** The K=4
> numbers in the table/prose below are kept for provenance but do NOT support the "joint recovers
> decode-placement regret" hypothesis at tight statistics. This strengthens the overall verdict (joint's
> value needs heterogeneous `θ_i`, not visible on homogeneous hardware) — it does not weaken it. Note
> `total_regret` is a sum over K, so the K=4 (0.030) and K=50 (0.2188) magnitudes are not comparable; the
> reduced-vs-joint *direction* at fixed K is what flipped.

**Honest reading — joint does NOT uniformly win.**
- **Cache-uniform (`synth_cf`), both topologies:** joint **cuts leftover regret ~25%** (0.030→0.0225),
  confirming the hypothesis directionally — the joint argmin recovers part of the decode-placement regret
  the reduced rule leaves by delegating `d` to the scorer. But it **trades a hair of goodput** (0.990→0.979):
  moving decode to the drift-optimal node is better in one-step hindsight yet the greedy per-request
  argmin over-corrects slightly on the realized run. Divergence is all on `d` (`p_div=0`, only one
  prefill node matters here); on the 32% of decisions where joint overrides the scorer's decode, it picks
  a **strictly lower-J** candidate 10% of the time and a **J-tie (deterministic lower-index/lower-occupancy
  break)** the other 90% — i.e. most overrides are occupancy tie-breaks, not large objective gaps.
- **Cache-asymmetric (`synth_asym`), both topologies:** the loose batch SLO is met by *everyone* — reduced
  goodput is **1.000 with ZERO regret** (already optimal), so there is nothing for joint to recover.
  Joint instead introduces a **tiny positive regret (0.0012) and a hair of goodput loss** (1.000→0.99875):
  a clean case where **joint ties-to-loses**. This is the intended stress of the asymmetric spec, and the
  finding is that the joint lever has no headroom when the reduced rule is already optimal.
- **The one place the prefill lever actually fires: 2P2D synth_asym** — `p_div=0.260` (vs 0 everywhere
  else) and disagg-share on divergent rows jumps to 0.57. With unique large prompts and two prefill nodes,
  the two decode/prefill nodes genuinely diverge in cache warmth, so the joint objective reroutes *prefill*
  (not just decode) across candidates — exactly the `a_p`-differs regime the asymmetric spec was built to
  create. It still doesn't convert to goodput here (SLO already met), but it demonstrates the joint
  objective exercising the P-placement degree of freedom the reduced rule cannot.
- **1P2D == 2P2D on `synth_cf`** (byte-identical rows): the workload is decode-bound with two decode nodes
  in both topologies, so the extra prefill node in 2P2D is not on the critical path and prefill placement
  never diverges — an observation, not a bug.

**Verdict.** On homogeneous hardware the joint mechanism is a **modest, workload-dependent** change: it
recovers ~25% of reduced-EDPP's decode-placement regret on the cache-uniform decode-bound workload
(directionally confirming the regret hypothesis) while shaving a hair of realized goodput, and it
ties-to-slightly-loses on the cache-asymmetric workload where the reduced rule is already optimal. Its
distinctive P-rerouting only activates under cache asymmetry with ≥2 prefill nodes (2P2D synth_asym), and
even there does not pay off at this loose SLO. The larger expected win — per-instance hardware
heterogeneity (`θ_i`) — is deferred; these cells cannot show it. Divergence is dominated by
occupancy/cache tie-breaks (~80–90% of overrides are J-ties), not large objective gaps; the argmin
invariant holds (no divergent row picks a strictly higher J, verified by `joint_divergence.py`).

**Reproduce:** `bash campaigns/edpp-study/repro_joint.sh` (`K=…` overridable; K=4 default keeps the
4-cell sweep bounded). Artifacts per cell in `out/joint/<topo>_<workload>/` (`regret_*/regret.json`,
`divergence.json`). Divergence analyzer + self-test: `analyze/joint_divergence.py` (`selftest`
subcommand) and `analyze/test_joint_divergence.py`.

## Estimator-accuracy figures (paper) — work model + ttft_d (2026-07-07/08)

Paper figures validating the two things EDPP must *estimate* before it can measure them: per-request
**work** (`W_p`/`W_d`) and the **local time-to-first-token** (`ttft_d = T_adm + prefill_time`). All on
the **trained-physics** latency model, llama-70b-h100-tp4 frozen coeffs.

### Fig 1 — work model vs load (`out/work_sweep/fig1_work_model.png`)
Realized per-request trajectory work (summed per-step from the DES step engine — the SAME coefficients
the closed form uses; see `simulator.go accumulateStepWork`) vs the closed-form `W_p`/`W_d`, over offered
load 0.5–3.0 for synth + rag. **This validates the closed form against the model's own per-step physics,
NOT against hardware** (that's `observe`/`calibrate`). Result: single-chunk prefill and decode are
**float-exact and load-invariant** (|rel err| ~3e-16 / ~0); the only residual is **chunked prefill**
(the documented `C_attn·(a_p²−Σsᵣ²)/2` term), bounded and rising mildly with load (synth p99 8%→19%
across the sweep; rag ~21%). Reproduce: `bash campaigns/edpp-study/repro_work_model_sweep.sh`.

### Figs A/B/C — ttft_d on a single collocated instance (`out/ttft_d_local/fig_{admission,prefill,ttft}.png`)
Setup: ONE collocated engine (1P1D, `--edpp-c-xfer 100s` ⇒ every request local, no routing confound),
sweeping offered load **ρ = arrival rate / λ\*** (λ\* = plateau throughput μ from one overloaded probe
run — warmup-insensitive, unlike a throughput-ratio threshold) from ρ=0.5 (underload) to ρ=1.5 (overload).
Three synthetic single-client archetypes spanning the prefill/decode spectrum: **synth** (decode-heavy),
**mixed** (balanced, `specs/mixed_rate1.0.yaml`), **prefill** (prefill-heavy, `specs/prefill_rate1.0.yaml`).
Real **rag is NOT used here** — its 15k–80k-tok prompts make a single-instance overload run take ~30 min,
intractable to sweep. Estimators compared: **fluid** and **rollforward** only (`waiting` dropped);
`--edpp-tadm-estimator` selects the one driving `ttft_d`. Metric: median over local requests of realized
vs estimated, decomposed as `ttft_d = admission + prefill`.

**Key result.** Up to capacity (ρ≤1) both fluid and rollforward track realized admission to **~1.0–1.2×**
— reproducing the Stage C utilization-sweep result *in the collocated setting*. In **overload (ρ>1)** the
estimators under-predict admission by 1–2 orders of magnitude (mixed ρ=1.5: realized 20 s vs est ~78 ms):
they are **snapshot roll-forwards** — they project draining the queue observed at decision time, so they
are blind to the non-stationary queue growth that continues *after* the request arrives. Stage C never
saw this because its sweep stopped at ρ≈0.98 (sub-capacity by design); the two results are complementary,
not contradictory. fluid ≈ rollforward on these workloads (they differ only in `N̂_out` prediction, which
decode-bound/short-output archetypes don't stress). The TTFT figure is the clean bottom-line comparison;
the admission/prefill split is only meaningful once `local_t_adm` is correctly recorded (see gap below).

**Instrumentation gap fixed (cluster.go `BuildPDOutcomeRecords`).** The `--pd-outcome-trace` path
formerly hardcoded `local_t_adm=0` for local requests, so the outcome trace silently reported zero local
admission even at 20 s TTFT. The fix captures the local enqueue instant (`localEnqueueTimes`, now
populated under `recordPDOutcomes` too) so `local_t_adm = local_schedule − local_enqueue` is emitted like
the prefill/decode legs. The realized admission was always correct in the `--edpp-admission-trace`
(`realized_t_adm`); the plotter reads it from there.

**Reproduce:**
```
bash campaigns/edpp-study/repro_work_model_sweep.sh      # Fig 1 (work model vs load)
bash campaigns/edpp-study/repro_ttft_d_local.sh          # Figs A/B/C (ttft_d vs load × 3 archetypes)
# figures land in out/work_sweep/ and out/ttft_d_local/ (both gitignored)
```
Analysis scripts: `analyze/work_model_sweep.py`, `analyze/ttft_d_local.py`.

### ttft_p — Phase 1: single 1P1D disaggregated pipeline (`out/ttft_p_local/fig_{ttft_p,prefill_adm}.png`)
Mirror of the ttft_d figure for the DISAGGREGATED path: 1P1D, `--pd-decider edpp --edpp-c-xfer 0s` ⇒ EDPP
disaggregates ~98.5% of requests onto the single prefill + single decode engine (no pool/routing
confound; `--pd-decider edpp` is required — only EDPP's `Decide` computes/logs `ttft_p`). Validates
`ttft_p = prefill-pool admission + prefill compute + KV transfer + decode-side first-token` against
realized outcomes, fluid & rollforward, same ρ=offered/λ\* plateau-probe load axis, same synth/mixed/
prefill archetypes. λ\*: synth 0.72, mixed 2.99, prefill 7.47 req/s — note prefill's disagg λ\* (7.47)
**exceeds its collocated λ\* (5.66)** because disaggregation frees the decode engine from prefill
contention (the expected prefill-heavy disaggregation payoff). Decomposition option (b): two figures —
total `ttft_p`, plus the prefill-pool admission component.

**Key results.**
- **`fig_ttft_p` (total, the deliverable):** realized TTFT rises with load; `ttft_p` is essentially FLAT
  (fluid ≡ rollforward) — mild OVER-prediction at low load, accurate near ρ=1, UNDER-prediction above.
  Crucially **no overload explosion** (TTFT bounded ~100–180ms even at ρ=1.5, vs ttft_d's 20 s): on the
  disaggregated path prefill runs on the uncontended engine, so decode backlog hits ITL/E2E, not
  first-token. Same snapshot-based load-insensitivity as ttft_d, far gentler failure — the disaggregation
  TTFT payoff, visible.
- **`fig_prefill_adm` (component):** prefill-pool admission is **accurately estimated (~1.06×: realized
  15.6ms vs est 16.6ms)** but nearly FLAT and small — on 1P1D the single prefill engine is not the
  bottleneck, so this term barely moves with load. The one real signal is prefill-workload at ρ=1.5,
  where realized prefill admission rises to 24.8ms (prefill engine finally saturates) and the estimate
  misses the onset. NOTE the auto-zoomed y-axis (~15–25ms) visually exaggerates the ~1ms gap.
- **fluid ≡ rollforward in `ttft_p`**: they agree on prefill-pool admission (which dominates ttft_p's
  admission term); the decode-pool 2× over-prediction (Stage C T2) enters ttft_p only as a tiny
  first-token component, so it doesn't surface.

**Limitation / Phase 2.** The prefill-admission decomposition is idle on 1P1D because the prefill engine
has slack. To make prefill-pool admission the load-sensitive term it needs a **prefill-bottlenecked
topology (e.g. 1P3D)**, plus 2P2D / 3P1D — deferred to Phase 2 (multi-instance pools need load-balancing
`--prefill/decode-routing-scorers` to avoid shared-prefix pinning, as the ttft_d k>1 exploration found).

**Reproduce:** `bash campaigns/edpp-study/repro_ttft_p_local.sh` → `out/ttft_p_local/fig_{ttft_p,prefill_adm}.png`.
Analysis script: `analyze/ttft_p_local.py`.

## Policy comparison — all five §5 deciders (2026-07-09)

*(The "§5 deciders" = the baseline table in the joint-routing formulation,
`docs/design/2026-06-30-pd-joint-routing-problem-formulation.md`, section 5.)* Goodput
(`slo_attainment`) + one-step counterfactual regret (`total_regret`, K=4) for `never` / `always` /
`prefix-threshold` / reduced-EDPP / joint-EDPP, same cells as the joint sweep. Reproduce:
`bash campaigns/edpp-study/repro_policy_comparison.sh` (artifacts in `out/policy_cmp/<cell>/`).

| cell | never | always | prefix-threshold | reduced-EDPP | joint-EDPP |
|---|---|---|---|---|---|
| 1P2D synth_cf (cache-uniform, decode-bound) | **0.399** / NA | **1.000** / 0.000 | **1.000** / 0.000 | 0.990 / 0.030 | 0.979 / 0.0225 |
| 2P2D synth_cf | 0.399 / NA | 1.000 / 0.000 | 1.000 / 0.000 | 0.990 / 0.030 | 0.979 / 0.0225 |
| 1P2D synth_asym (loose SLO) | 1.000 / NA | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.999 / 0.0012 |
| 2P2D synth_asym | 1.000 / NA | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.999 / 0.0012 |

(`never` regret = NA: its all-local run produces no `--pd-outcome-trace` rows — "need
PD/disaggregation enabled" — so no plan to sweep. Goodput is unaffected. This is the deferred
`--pd-outcome-trace`-omits-local-`decode_instance` gap.)

**Honest reading (this reframes the joint-mechanism result).**
- **On synth_cf (decode-bound), full disaggregation is optimal and trivial baselines win.** `always`
  and `prefix-threshold` reach **goodput 1.000 with ZERO one-step regret** — they beat BOTH EDPP
  variants (reduced 0.990, joint 0.979). `never` collapses to **0.399** (all-local on 2 mixed nodes ⇒
  decode saturation). So: (i) this workload SHOULD disaggregate (Q1: never ≪ always), and (ii) the
  right amount is *all of it* — EDPP's selectivity (it keeps ~some requests local) is a small
  **liability** here, and joint's decode-node reshuffling costs a hair more than reduced.
- **So the "regret" reduced/joint leave is regret *against the always baseline being available*** — EDPP
  leaves goodput on the table precisely by NOT fully disaggregating like `always` does, and neither EDPP
  variant reaches the trivial baselines' 1.000 on this workload. **(K=50 update — supersedes the K=4
  numbers above.)** At K=50 sampled requests, reduced total_regret **0.2188** vs joint **0.3213** — **joint
  leaves MORE on the table than reduced, not less.** The K=4 "joint ~25% lower" was a small-sample artifact
  (see the CORRECTION banner in the "Joint mechanism" section). So on this homogeneous decode-bound cell the
  ordering is `always ≈ prefix-threshold (optimal) > reduced-EDPP > joint-EDPP`.
- **synth_asym (loose SLO): everyone ties at ~1.000** — the batch SLO is met by all policies, so there
  is no separation; joint again shaves a hair (0.999).
- **The takeaway for the paper.** On uniform/decode-bound synth at equal hardware, the standard
  baselines (`always`/`prefix-threshold`) are already optimal, so neither reduced- nor joint-EDPP can
  demonstrate value — echoing the equal-HW Q2 verdict in the EDPP worklog memory. The joint mechanism's
  distinctive lever (per-instance cost via `θ_i`) is invisible here because all decode instances are
  identical; `always`/`never` cannot express a per-instance cost tradeoff, so a workload where they
  CANNOT be optimal (heterogeneous hardware — sub-project 2) is required to show the joint rule's edge.

## T-A — cheap heterogeneity de-risk (hand analysis) (2026-07-12)

Goal (TODO.md "ROADMAP", T-A): before investing in the heterogeneity simulator work, confirm/refute that
the joint rule (`--edpp-joint`) CAN beat `always`/`never` under heterogeneity.

**Hand analysis of the joint objective `J`** (formulation
`docs/design/2026-06-30-pd-joint-routing-problem-formulation.md`, section 5.3), comparing `J(M1,local)` vs
`J(M2,local)` for two decode instances at equal occupancy — the choice is driven by which node is *cheaper*
for the request:
- **Cache-warm node** (smaller uncached `a_p`) → smaller `W_p`/`T̂` → joint prefers it. **But the shipped
  scorer is `precise-prefix-cache`, already cache-aware**, so `always`/`never`+scorer route to the warm node
  too — **joint has NO edge here.** This explains why the cache-asymmetric `synth_asym` cells were null
  (see "Policy comparison").
- **Faster-hardware node** (smaller `θ_i`: `C0`/`C1`/`α`) → smaller `W_d`/`T_iter` → joint prefers it. **The
  scorer is NOT hardware-cost-aware** (prefix/queue/kv only) — so the baselines CANNOT preferentially pick
  the faster node. **This is joint's real, unique edge.**

**Conclusion: the thesis is not dead — the mechanism has headroom, but specifically for HARDWARE-`θ`
heterogeneity, not cache.** So there is no cheap empirical shortcut via cache asymmetry.

**Feasibility of the empirical opportunity test.** (a) EDPP builds ONE global latency model
(`sim/cluster/cluster.go:437`) — per-instance `θ_i` in the *decider* is a code change (T-B). (b) The
*simulator* CAN serve a heterogeneous decode pool via a node-pool bundle: placement is role-blind,
deterministic first-fit in pool-declaration order (`sim/cluster/infra_placement.go:184`,
`PlaceInstance(..., gpuType="")`), so two GPU pools sized to land prefill→poolA, decode→poolA, decode→poolB
put the two decode nodes on different hardware (`simCfg.HWConfig` per instance drives latency) — **no code
change**. So the fixed-plan brute-force "opportunity" test (does the optimum beat `always`/`never` under
hardware heterogeneity — no `θ_i` needed) is feasible, but requires authoring a node-pool bundle
(2 GPU types + capacity-forced placement) — real setup, not free. The joint-*capture* test additionally
needs per-instance `θ_i` (T-B). Since both share the node-pool serving setup, the efficient path is likely
minimal T-B (node-pool serving + `θ_i` indexing) rather than a separate opportunity-only harness.

**T-A spike outcome (node-pool serving feasibility) — a small code change is needed.** Verified: the
per-GPU hardware map `hw_config_by_gpu` (documented in `docs/reference/configuration.md`, with e.g.
H100=1979 TFLOPS/3.35 TB/s vs A100=1248/2.0) is a `DeploymentConfig` field (`sim/cluster/deployment.go:166`)
but is **NEVER populated by any non-test code** (no assignment in `cmd/`), and `PolicyBundle`
(`sim/bundle.go`) parses strictly (`KnownFields(true)`) without an `hw_config_by_gpu` field — so a
`--policy-config` YAML carrying it would ERROR. Net: **no code-free path to heterogeneous decode serving.**
Node-pools DO load via `--policy-config` and placement is deterministic/role-blind
(`sim/cluster/infra_placement.go:184`), so the ONLY missing piece is wiring `hw_config_by_gpu` from the
bundle into `DeploymentConfig.HWConfigByGPU` (small: mirror the `bundle.NodePools` wiring at
`cmd/root.go:1721`). This minimal change unblocks BOTH the opportunity test (fixed-plan brute-force, no
`θ_i`) AND is a prerequisite for T-B. Path: (1) wire `hw_config_by_gpu`; (2) confirm two decode nodes run
at different speeds; (3) brute-force opportunity test (optimum vs always/never under hardware heterogeneity);
(4) if headroom → per-instance `θ_i` in EDPP (rest of T-B).

---

## Hardware-θ opportunity test (T-A steps 1–3, 2026-07-12) — HEADROOM CONFIRMED; joint captures it

**Setup.** `hw_config_by_gpu` bundle wiring now merged (branch `feat/edpp-estimator-validation`,
commits `60dcdaa..b07a79f`), so a `--policy-config` bundle can serve a heterogeneous decode pool. Bundle
`campaigns/edpp-study/specs/hetero_hw/bundle_1p2d.yaml`: 1P2D where placement (first-fit, TP=4 ⇒ 4 GPUs/instance)
lands `instance_0`=prefill(H100), `instance_1`=decode(H100 **fast**), `instance_2`=decode(A100 **slow**,
deliberately crippled to 400 TFLOPS/0.7 TB/s). Workload `specs/hetero_hw/synth_hw.yaml`: homogeneous
decode-bound (isl 256, osl 512, prefix 0, batch SLO), N=60 @ rate 1.0 — so the ONLY interesting lever is
*which decode instance* a request lands on. SLO `--slo-itl batch=50ms` (fast idle ITL ~17ms MEETS, slow
~74ms VIOLATES); `--slo-ttft batch=10s` loose.

**Serving heterogeneity confirmed (step 2).** Pinning all 60 decode → `instance_1` gives ITL 16.97ms /
e2e 8.7s; all → `instance_2` gives ITL 73.84ms / e2e 37.7s (4.3× ITL, 4.4× latency) — driven purely by
`hw_config_by_gpu`. The fast node alone serves all 60 within SLO (no saturation at this load).

**Opportunity test (step 3): the optimum beats every hardware-blind policy.** Fixed-plan fast-fraction
sweep is exactly linear (goodput = #routed-to-fast / 60: 0→0.25→0.50→0.75→1.00) — at this SLO each
fast-routed req meets, each slow-routed req violates, no contention. So the fixed-plan **optimum = all-fast
= goodput 1.00**. Hardware-BLIND decode routing (always decider, various scorers), single seed 42:

| decode routing (hardware-blind)                     | goodput | fast/slow split |
|-----------------------------------------------------|---------|-----------------|
| default PD profile `precise-prefix-cache:2,queue-depth:1` | 0.00    | 0 / 60          |
| `queue-depth:1`                                     | 0.40    | 24 / 36         |
| `active-requests:1`                                 | 0.75    | 45 / 15         |
| `load-balance` / `kv-utilization` / `running-requests` | 0.77    | 46 / 14         |
| **fixed-plan optimum (all-fast)**                   | **1.00**| 60 / 0          |

Two honest caveats on the baseline: (1) the DEFAULT profile is *pathological* here — `precise-prefix-cache`
(weight 2, prefix_length 0) pins all decode to one instance (the slow one) → 0.00; the naive "0→1" headline
overstates the gap. (2) Load-aware blind scorers *implicitly* exploit speed (the fast node drains faster ⇒
looks less loaded ⇒ attracts more) and reach 0.77 "for free" without any hardware knowledge. **But no
hardware-blind policy closes the last ~0.23 to the optimum** — that residual is the genuine hardware-θ headroom.

**KEY RESULT — joint-EDPP captures the headroom; reduced-EDPP does not (4 seeds).** Placing the actual
deciders on this scale (`--edpp-coeffs coeffs-llama70b-h100-tp4.json`, τ_ttft 10s, τ_itl 50ms, rollforward):

| seed | reduced-EDPP (default profile) | **joint-EDPP** | best blind (always+load-balance) | optimum |
|------|-------------------------------|----------------|----------------------------------|---------|
| 42   | 0.00                          | 0.97           | 0.77                             | 1.00    |
| 7    | 0.00                          | 0.97           | 0.72                             | 1.00    |
| 123  | 0.00                          | 1.00           | 0.73                             | 1.00    |
| 2024 | 1.00 (scorer-luck)            | 0.98           | 0.77                             | 1.00    |

Robust ordering: **joint (0.97–1.00) ≫ best hardware-blind (0.72–0.77) > reduced (0.0, scorer-luck-dependent)**.
joint routes 58/60 → fast. **This is the FIRST workload where joint STRICTLY and SUBSTANTIALLY beats reduced**
— the exact opposite of the homogeneous-hardware result (Joint mechanism / K=50 correction: joint ≤ reduced there).

**MECHANISM (important, and it revises a prior belief).** joint uses a SINGLE homogeneous θ, so it does NOT
model the speed gap directly. It prefers the fast node because the slow node visibly ACCUMULATES more
congestion: joint's per-instance `Q_i` term (`qByInstance`) + per-candidate occupancy-aware `T̂` (rollforward)
are both higher for the backed-up slow candidate, so argmin J avoids it. reduced-EDPP delegates decode
selection to the scorer, so it inherits the scorer's blindness (0.0 pathological / 0.77 load-aware) exactly.
So the prior claim "joint's value REQUIRES heterogeneous θ_i / is invisible on homogeneous HW" is too strong:
**joint captures most hardware-θ headroom REACTIVELY via per-instance congestion, before any per-instance θ_i.**
Per-instance θ_i (T-B) becomes the PROACTIVE refinement — predict slowness before congestion accrues, and
close the residual 0.97→1.00 — not the thing that first unlocks the win.

**Decision gate: PASSED.** Headroom exists (optimum 1.00 vs best blind 0.77) and joint already realizes it
(≈0.97–1.00), giving the paper a concrete positive joint-vs-reduced result that was absent on homogeneous HW.

**Caveats / next (rigor before paper-grade).** One archetype (homogeneous decode-bound, small prefill),
one speed gap (~4.3× ITL), one binary-separating SLO, N=60, 4 seeds. The optimum here is DEGENERATE
("all-fast", because the load fits entirely on the fast node) — the natural harder test is load that
SATURATES the fast node, forcing a non-trivial speed-weighted split, to show joint computes the RIGHT split
(not just "avoid the slow node") and to expose where reactive-congestion joint falls short of proactive θ_i.
Artifacts: `campaigns/edpp-study/specs/hetero_hw/`, out `/tmp/hwopp` (regenerate via the commands in this section).

### Saturating-regime follow-up (2026-07-12) — reactive joint = blind load-balance; only θ_i closes the gap

The opportunity test above used an UNDER-capacity regime (rate 1.0, the fast node alone serves all load),
so its optimum was DEGENERATE ("all-fast") and joint's reactive congestion signal won easily. This follow-up
builds the harder NON-DEGENERATE case: cap per-instance concurrency (`--max-num-running-reqs 8`), short
decode (osl 64, both nodes meet an 8s e2e SLO when un-queued), and push arrival RATE until the fast node
SATURATES so the optimum is a genuine interior speed-weighted split. Same 1P2D fast-H100/slow-A100 bundle.

**Fixed-plan optimum is a non-trivial interior split.** At rate 10 (fast saturated: all-fast goodput 0.48,
all-slow 0.03), the fast-fraction sweep peaks at **~86% fast / 14% slow → goodput 0.96** — dramatically above
both extremes and above a naive 50/50 (~0.5). So using BOTH nodes, weighted toward the fast one, is optimal.

**Every reactive/congestion-aware policy converges to ~77% fast and UNDERSHOOTS the optimum (3 seeds):**

| seed | joint | blind load-balance | reduced+load-balance | fixed-plan optimum (86% fast) |
|------|-------|--------------------|----------------------|-------------------------------|
| 42   | 0.823 | 0.840              | 0.830                | 0.960                         |
| 7    | 0.818 | 0.835              | 0.810                | 0.902                         |
| 123  | 0.890 | 0.953              | 0.877                | 0.975                         |

(reduced default profile still 0.02 — the `precise-prefix-cache` all-to-slow pin.) **Joint gives NO advantage
over a plain hardware-blind load-balancer here (0.82 vs 0.84, sometimes marginally worse), and all three
cluster at ~77% fast vs the optimal 86%** — leaving ~0.08–0.14 goodput unclaimed every seed.

**WHY (the T-B motivation, sharpened).** Reactive congestion/load signals EQUALIZE queue depth / occupancy
across instances. Queue-equalization is NOT the goodput-optimal split: because the fast node drains 4.3×
faster, the optimum OVER-loads it relative to equal-queue (push it harder — its requests still meet SLO).
Joint's per-instance `Q_i`/`T̂` terms equalize just like the blind load-balancer, so joint converges to the
same ~77% fast. Closing the last ~0.1 to the 86% optimum requires PROACTIVE per-instance speed knowledge —
i.e. per-instance `θ_i` in the decider (T-B) — which reactive signals provably cannot supply.

**Synthesis of the two regimes (the paper's hardware-θ story).**
- UNDER-capacity (fast has spare room): reactive joint WINS big (0.97 vs blind 0.77) — its congestion term
  suffices to "avoid the slow node."
- SATURATING (must use both, optimum is an interior speed-weighted split): reactive joint = blind
  load-balance (~0.82), both undershoot the optimum (~0.96); per-instance `θ_i` is REQUIRED to hit the split.
So hardware heterogeneity creates real headroom in BOTH regimes; joint captures the easy one reactively, and
the hard one is the concrete, quantified case for T-B. Repro: `repro_hetero_hw.sh` (SAT=1 mode).

## Per-instance θ_i (T-B) result — the acceptance experiment (2026-07-14)

**What.** The T-B change (per-instance `θ_i` in the joint decider, keyed by GPU type via a `coeffs_by_gpu`
bundle; design `docs/superpowers/specs/2026-07-14-edpp-per-instance-theta-design.md`) is now merged. This is
the acceptance run against the **saturating-regime bar** set by the "Saturating-regime follow-up" above.
The joint arm additionally carries `coeffs_by_gpu: {H100: coeffs-llama70b-h100-tp4.json, A100:
coeffs-llama70b-a100crippled-tp4.json}` — the fast H100 file and the slow-A100 file fit from the SAME 400
TFLOPS / 0.7 TB/s HWConfig the slow pool executes on (decode c0 8.87 vs 5.35, c1 0.228 vs 0.048 — ~4.8×
costlier decode), so the decider's θ_i matches execution. Same 1P2D fast-H100 / slow-A100 bundle, rate 10,
`--max-num-running-reqs 8`, seeds 42/7/123. Deterministic (INV-6): byte-identical across re-runs.
Reproduce: `SAT=1 THETA=1 SEEDS="42 7 123" bash campaigns/edpp-study/repro_hetero_hw.sh`.

**Acceptance bar (design §7).** PASS = θ_i-joint shifts the realized fast-share from ~77% toward ~86% AND
goodput from ~0.82 toward ~0.96, **beating reduced-EDPP and blind load-balance across the 3 seeds.**

**Saturating regime — result (goodput; realized fast/slow decode split from per-instance completions):**

| seed | **θ_i-joint (T-B)** | joint (homog θ) | blind load-balance | reduced+load-balance | fixed-plan optimum (86% fast) |
|------|---------------------|-----------------|--------------------|----------------------|-------------------------------|
| 42   | **0.877  318/82 (80% fast)** | 0.823  309/91 (77%) | 0.840  311/89 (78%) | 0.835  310/90 (78%) | 0.960  344/56 (86%) |
| 7    | **0.685  380/20 (95% fast)** | 0.820  308/92 (77%) | 0.835  307/93 (77%) | 0.823  308/92 (77%) | 0.902  344/56 (86%) |
| 123  | **0.750  389/11 (97% fast)** | 0.943  314/86 (78%) | 0.953  316/84 (79%) | 0.938  316/84 (79%) | 0.975  344/56 (86%) |

**VERDICT: the acceptance bar is NOT met — θ_i-joint OVER-corrects (an honest, publishable finding, design §9).**
θ_i moves the split in the RIGHT direction on every seed (fast-share always ≥ homogeneous joint's ~77%), so it
DOES unstick the queue-equalizing split the reactive signals were pinned at — the design's core mechanism claim
("θ_i makes the split speed-sensitive") holds. But per-instance `θ_i`, plugged into the work terms alone,
**over-weights the fast node and shoots PAST the 86% optimum on 2 of 3 seeds** (95%, 97% fast), where the fast
node's fixed concurrency cap (8) then bottlenecks and goodput COLLAPSES **below every reactive baseline** (0.685
and 0.750, vs blind ~0.84/0.95, reduced ~0.82/0.94, homogeneous joint ~0.82/0.94). Only seed 42 shows the
intended partial improvement (0.877, 80% fast — beating blind 0.840, reduced 0.835, homogeneous joint 0.823).
So θ_i-joint neither reaches ~86%/~0.96 nor reliably beats reduced/blind: it wins one seed and regresses two.

**Mechanism (why it overshoots).** The homogeneous joint sat at ~77% fast because its per-instance congestion
term (`Q_i`) + occupancy-aware `T̂` EQUALIZE queues — a stable, capacity-respecting split that merely undershoots
the speed-weighted optimum. Adding per-instance `θ_i` makes the slow-A100 decode candidate's work term ~4.8×
larger, and that proactive work-cost gap OVERWHELMS the reactive `Q_i` congestion signal that had regulated the
split: argmin J now prefers the fast node so strongly that congestion no longer pulls load back once the fast
node's queue builds. The result is an UN-regulated over-allocation to fast, past the goodput-optimal interior
point, into the fast node's cap-8 saturation. The design's §2 non-goal — deferring per-instance `Z^I_i` and any
capacity/congestion coupling of the θ term — is exactly what makes this overshoot: **naive work-model θ_i
supplies the speed signal but removes the capacity governor, and the two are needed together to land on ~86%.**

**Under-capacity regime — NO regression (design §7 requirement, PASS).** With the same `coeffs_by_gpu` bundle at
rate 1.0 (the regime joint already wins reactively), θ_i-joint is **byte-identical to homogeneous joint** every
seed — it does not perturb the regime where reactive congestion alone suffices:

| seed | reduced (dflt) | joint (homog) | **θ_i-joint** | best blind (load-balance) | optimum (all-fast) |
|------|----------------|---------------|---------------|---------------------------|--------------------|
| 42   | 0.000          | 0.967         | **0.967**     | 0.767                     | 1.000              |
| 7    | 0.000          | 0.967         | **0.967**     | 0.717                     | 1.000              |
| 123  | 0.000          | 1.000         | **1.000**     | 0.733                     | 1.000              |

(θ_i ≡ homogeneous joint here because the fast node has spare room, so the argmin's speed-vs-congestion tradeoff
never binds — no candidate is ever pushed into saturation. ~0.967 ≈ the design's ~0.97 no-regression floor.)

**The honest headline (a bound on per-instance work-model knowledge).** Per-instance `θ_i` in the joint work
model is **necessary but not sufficient** to hit the saturating-regime optimum. It unsticks the reactive
queue-equalization (which provably could not exceed ~77%) and makes the split genuinely speed-aware, but on its
own it over-corrects — it lacks the capacity/congestion coupling that would stop over-loading the fast node at
its interior optimum. Reaching ~86%/~0.96 therefore requires `θ_i` **combined with** a per-instance capacity
governor (the deferred `Z^I_i` / occupancy-coupled work term, design §2 non-goal), not the work-model θ alone.
This is the bound design §9 anticipated: it quantifies what per-instance work-model knowledge achieves (a
speed-aware but un-regulated split) versus the residual that needs congestion/ITL coupling. For the paper, the
positive T-A story stands (reactive joint captures the under-capacity headroom, ~0.97 vs blind ~0.77); the
saturating regime is now a characterized limitation with a concrete next step, not an unexplained gap.

**Durable wiring guard.** The `coeffs_by_gpu` → decider path is guarded end-to-end by
`cmd/edppcoeffs_bundle_wiring_test.go::TestCoeffsByGPU_RunCmdLiteralWiring_DecodeSplitObservable` (child-process
`blis run` on a heterogeneous 1P2D joint scenario, asserting the realized decode split shifts toward the fast
node WITH `coeffs_by_gpu` vs WITHOUT; RED when the `EDPPCoeffsByGPU:` literal in `cmd/root.go` is severed).

---

## SLO-class heterogeneity (2026-07-17) — per-class machinery is COUNTERPRODUCTIVE; externality-blindness confirmed

**Setup.** The third heterogeneity axis, never tested before. Prefill-bound 16000/16, 1P2D, cap 16,
decode scorer pinned `queue-depth:1`, rate 10. Rate chosen from a probe as the point where goodput is
well below 1 (≈40% of requests miss, so routing decides WHO survives) but EDPP has not yet collapsed
(rate 8 = nothing to allocate; rate 12+ = EDPP already broken). **Both classes have IDENTICAL sizes**
(16000/16, 50/50) so the ONLY difference is the SLO — a win cannot be attributed to workload het.
critical: ttft 1794ms / itl 100ms / e2e 1600ms. batch: 60s / 500ms / 60s (loose; it does not care).
EDPP gets per-class targets via `--edpp-tau-ttft-classes` / `--edpp-tau-itl-classes`.

**Hypothesis (stated before running):** `least-ttft` is class-blind by construction and CANNOT
prioritise; EDPP's per-class `z` queues can. Critical is 50% of load and ~60% of requests are savable,
so a class-aware policy could save nearly all critical (~1.0) while a class-blind one saves ~60% of
each class (~0.6). Predicted edpp ≈ 0.9–1.0, least-ttft ≈ 0.6.

**Result — critical-class goodput (batch = 1.000 everywhere by construction):**

| seed | never | always | least-ttft | **edpp single-τ (class-blind)** | edpp per-class τ |
|------|-------|--------|-----------|-------------------------------|------------------|
| 42   | 0.163 | 0.039  | 0.581     | **0.884**                     | 0.752            |
| 7    | —     | —      | 0.636     | **0.836**                     | 0.891            |
| 123  | —     | —      | 0.456     | **0.856**                     | 0.800            |
| mean | —     | —      | 0.558     | **0.859**                     | 0.814            |

**HYPOTHESIS REFUTED.** Per-class targets did NOT help — they HURT (mean 0.859 → 0.814, worse on 2/3
seeds). The one axis predicted to be EDPP's structural home does not deliver.

**MECHANISM — confirmed from EDPP's own decision trace (`--edpp-decision-trace`), seed 42:**

| | batch disagg | batch peak z_ttft | critical disagg | critical gp | batch e2e_p99 |
|---|---|---|---|---|---|
| edpp single-τ   | 55.0% | 239.3 | 27.1% | **0.884** | 3424ms |
| edpp per-class τ| 28.8% | **0** | 43.4% | 0.752 | **1879ms** |

The machinery did EXACTLY what it was designed to do: batch's `z_ttft` is exactly 0 (its TTFT term is
fully off), prefill-server access shifted away from batch (55%→29%) and toward critical (27%→43%).
**And critical still got worse.** Because batch did not disappear — with its TTFT term gone, nothing
pushed it to disaggregate (the transfer penalty went unopposed), so batch ran its 16000-token prefill
LOCALLY, on the decode servers critical needs. The deprioritised class ended up FASTER (e2e 3424→1879ms)
while the priority class paid.

**THE FINDING: routing cannot sacrifice a class.** Lowering a class's SLO target does not make it yield
capacity — it makes it *selfish*: it stops optimising its own latency and takes the cheapest resource,
which is the contended one. Batch's decision is computed from batch's own SLO; the harm it does to
critical is invisible to the rule. **This is the SAME externality-blindness that killed the Type-A/B
workload experiment** (FINDINGS "Q2": "EDPP judges each request by its OWN class SLO + shared backlog;
B-harms-A is an EXTERNALITY the rule can't express"), now reproduced in a cleaner setting AND shown to
be made WORSE by the per-class machinery. One structural flaw, two of the three heterogeneity axes.
Sacrificing a class requires admission/scheduling (shed or defer), which EDPP does not control.

**COROLLARY — a correction to the spectrum framing.** EDPP single-τ (0.859) beats `least-ttft` (0.558)
here by a wide margin, and at single-class rate 10 it also leads (0.646 vs 0.537). The earlier
"the drift/z/V machinery is not what wins" read came from rate 8 (both at ceiling: 0.917 vs 0.854) and
rate 16 (EDPP collapsed: 0.071 vs 0.375). There is a **moderate-contention band where the machinery
genuinely wins**, bounded above by the overload collapse. Corrected characterisation:
- light contention → least-ttft ≈ edpp (both near ceiling)
- **moderate contention → edpp > least-ttft (machinery earns its keep)**
- extreme overload → edpp collapses, least-ttft robust

**Caveats.** One archetype, one rate, 3 seeds, reduced path only (joint routing still never exercised).
Repro: probe + the four arms are single `blis run` commands; see STUDY_REPORT §8 for the pattern
(`--edpp-tau-ttft-classes` toggles the machinery; `--edpp-decision-trace` gives per-class disagg + z).

---

## Term ablation (2026-07-17) — the drift term is BOTH the win and the collapse; z ~inert; V·c_xfer negligible

**Question.** EDPP beats `least-ttft` at moderate contention (rate 10) but collapses under overload.
WHICH TERM is responsible? Ablate the reduced rule term-by-term using existing flags only.

**Method (no code change).** The reduced rule is `disagg iff lhs > rhs`, with
`lhs = balanceTermD - balanceTermP` (congestion drift, Q_i) and
`rhs = transferTerm + ttftTerm + itlTerm`.
- `--edpp-v 0` zeroes `transferTerm`.
- `--edpp-tau-ttft 999s` makes TTFT unviolatable ⇒ `z_ttft ≡ 0` ⇒ `ttftTerm = 0`.
  (`z_itl` is ALREADY 0 here: measured ITL ≈34ms vs τ_itl=100ms. τ_itl is held at 100ms so the
  normalizer μ_D = 1 − α_D/τ_itl is UNCHANGED — the ablation removes z without rescaling drift.
  With rhs=0 the surviving comparison `lhs > 0` is invariant to the τ_ttft scaling of W*.)
- Both off ⇒ `lhs > 0` = PURE congestion drift ("disagg iff the decode queue is more backed up
  than the prefill pool"). `least-ttft` = neither drift nor z nor V.

**Integrity check (decision trace, seed 42, rate 10) — the ablation is real:**

| arm | peak z_ttft | max abs ttft_term | max abs transfer | disagg% |
|-----|------------|-------------------|------------------|---------|
| drift only | **0** | **0** | **0** | 37% |
| full edpp  | 435.6 | 25.74 | **0.00078** | 41% |

Note `max|transfer_term| = 0.0008` vs `ttft_term = 25.7` — **the transfer penalty, i.e. the
formulation's stated objective, is FOUR ORDERS OF MAGNITUDE below the constraint terms.** It barely
moves the rule.

**Result — prefill-bound 16000/16, goodput.** Rate 10, 3 seeds:

| seed | least-ttft | drift only | drift + z | full edpp |
|------|-----------|-----------|-----------|-----------|
| 42   | 0.537     | **0.713** | 0.667     | 0.646     |
| 7    | 0.713     | **0.821** | 0.787     | 0.787     |
| 123  | 0.558     | 0.762     | **0.838** | 0.817     |
| mean | 0.603     | **0.765** | 0.764     | 0.750     |

Across load (seed 42):

| rate | least-ttft | drift only | full edpp |
|------|-----------|-----------|-----------|
| 8    | 0.854     | **0.929** | 0.917     |
| 10   | 0.537     | **0.713** | 0.646     |
| 12   | **0.500** | 0.100     | 0.275     |
| 16   | **0.375** | 0.071     | 0.071     |

**FINDINGS.**
1. **The congestion-drift term delivers the ENTIRE win** over `least-ttft` at moderate contention
   (+0.16 mean at rate 10). The earlier "drift is a crude externality proxy" hypothesis is SUPPORTED.
2. **The drift term is ALSO the cause of the overload collapse.** drift-only is the WORST arm at
   rate 12 (0.100 vs least-ttft 0.500) — removing drift entirely (least-ttft) is what survives overload.
   One term, both the win and the failure.
3. **The `z` virtual queues are ~inert at moderate load** (0.765 → 0.764) and only ever act as
   DAMAGE CONTROL for drift under overload (drift-only 0.100 → full 0.275, still << least-ttft 0.500).
   The time-average-constraint machinery has no demonstrated standalone value in this cell.
4. **`V·c_xfer` is negligible (4 orders down) and slightly harmful** (0.764 → 0.750).

**INTERPRETATION — this independently confirms the work-vs-value diagnosis.** `q_i·ΔW_i` is
WORK-weighted. At moderate load work ≈ value-at-risk (a loaded queue really does hold savable
requests), so "don't dump there" is correct and the term wins. Under overload every queue is huge, so
the term balances work between two hopelessly-backed-up queues — whereas a VALUE-weighted version
would price those queues at ~0 (they hold doomed requests; dumping there is free) and steer the
savable requests to wherever they can still make the deadline. **The drift term is doing the right
job in the wrong currency.**

**Consequence for the design.** This is NOT a reason to leave Neely — a virtual queue can measure
anything. Keep the drift structure; change what the queue measures (value-at-risk, not work); drop
the transfer penalty (noise); `z` appears subsumed by a correct drift term.

**Caveats.** Rate-10 ablation is 3 seeds; the load-range row is seed 42 only. One archetype
(prefill-bound), reduced path only (joint still never exercised). `z_itl` was inert throughout
(ITL never approached its target), so this ablates z_ttft specifically.

### Widened term ablation (2026-07-17) — full rule wins 0/12 cells; z's sign FLIPS by archetype

Widened the ablation to all 4 archetypes x rates {8,12,16} x seeds {42,7,123} (repro:
`MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh`).
Means over 3 seeds; four cells sit at the 1.000 ceiling and carry no signal.

| archetype | rate | least-ttft | drift-only | drift+z | full | winner |
|-----------|------|-----------|-----------|---------|------|--------|
| decode        | 8  | 0.133 | **0.267** | 0.135 | 0.133 | drift-only |
| decode        | 12 | 0.133 | **0.237** | 0.135 | 0.133 | drift-only |
| decode        | 16 | 0.133 | **0.233** | 0.137 | 0.133 | drift-only |
| mixed         | 16 | 0.750 | **0.803** | 0.747 | 0.708 | drift-only |
| prefill_lean  | 16 | 0.754 | 0.696 | **0.793** | 0.771 | drift+z |
| prefill_bound | 8  | 0.904 | **0.971** | 0.921 | 0.925 | drift-only |
| prefill_bound | 12 | **0.492** | 0.264 | 0.408 | 0.414 | least-ttft |
| prefill_bound | 16 | **0.397** | 0.061 | 0.062 | 0.064 | least-ttft |

**HEADLINE: the shipped full rule is the best arm in 0 of 12 cells.** It is dominated everywhere by
one of its own ablations.

**CONFIRMED — drift is the load-bearing term.** Best in 5 of the 8 informative cells.

**REFINED — the collapse is NOT universal.** drift-only collapses only on prefill_bound at rate>=12
(the most extreme archetype at extreme load), where `least-ttft` wins. Note high seed variance there
(drift-only @ r12: 0.100 / 0.529 / 0.163) — the collapse is bimodal, i.e. a tipping point, not a
smooth degradation.

**REFUTED (an earlier claim in the previous section) — `z` is NOT inert; its SIGN FLIPS:**

| archetype | drift+z − drift-only | |
|-----------|---------------------|---|
| decode        | **−0.110** | z badly HURTS |
| mixed         | −0.018 | ~inert |
| prefill_lean  | +0.030 | z helps |
| prefill_bound | +0.032 | z helps |

The earlier "z is inert" was drawn from ONE cell (prefill_bound rate 10) where it happened to be
neutral. Across the spectrum z is the difference between 0.24 and 0.135 in decode-bound.

**MECHANISM — this explains the long-standing "EDPP under-disaggregates on decode-bound" anomaly.**
`z_ttft` prices the TTFT *cost* of disaggregating (the KV transfer) but is BLIND to the decode
capacity that disaggregating BUYS. In decode-bound the prompt is tiny, so disagg's TTFT cost is
essentially just the transfer — small but positive — so `ttftTerm > 0` raises `rhs` and vetoes
disaggregation. But disagg is exactly right there (`always` 0.271 >> `never` 0.133). **So z drives
EDPP to behave like `never` on decode-bound** — the anomaly first seen in the spectrum sweep
(edpp = never = 0.133 vs always = 0.271) and unexplained until now. Drift wants to disaggregate; z
overrules it; z is wrong.

**THE UNIFYING STATEMENT.** `z_ttft` prices the deciding request's OWN TTFT. `V*c_xfer` prices the
deciding request's OWN transfer. Only the drift term prices the effect on EVERYONE ELSE — and it
does so in WORK, not VALUE. *Every term prices the deciding request's own experience; the one term
that prices the externality uses the wrong currency.* That single sentence accounts for all four
failures on record: the decode-bound veto, the prefill-bound collapse, the Type-A/B workload failure,
and the SLO-class backfire.

**Caveats.** Reduced path only (joint still never exercised). One topology (1P2D), cap 16, single
`--edpp-tau-itl 100ms` throughout (z_itl was inert in every run — measured ITL never approached the
target, so this ablates z_ttft specifically). Four cells are at the ceiling and carry no information.

### JOINT ablation (2026-07-17) — gap 01 CLOSED: joint routing does NOT rescue EDPP

**The gap this closes.** Every experiment before this used the REDUCED rule (a scorer picks the
decode instance; EDPP only chooses local-vs-disagg). EDPP's actual thesis — the joint (d,p) argmin,
motivated by formulation §1's coupling argument — had never been exercised in any spectrum, class,
or ablation cell. `JOINT=1` added to the tracked harness; the same term ablations apply to the joint
path (`--edpp-v 0` zeroes its transfer term, `--edpp-tau-ttft 999s` zeroes its z). In joint mode EDPP
enumerates all (d,p) and picks the argmin ITSELF, overriding the decode scorer entirely.
Repro: `JOINT=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh`

**Result — reduced vs JOINT, mean of 3 seeds (^ joint better by >0.02, v joint worse):**

| archetype | rate | drift-only red→joint | drift+z red→joint | full red→joint |
|-----------|------|---------------------|-------------------|----------------|
| decode        | 8  | 0.267 → 0.286 | 0.135 → 0.139 | 0.133 → 0.136 |
| decode        | 12 | 0.237 → **0.296 ^** | 0.135 → 0.136 | 0.133 → 0.133 |
| decode        | 16 | 0.233 → 0.221 | 0.137 → 0.139 | 0.133 → 0.133 |
| mixed         | 12 | 0.999 → **0.843 v** | 1.000 → 1.000 | 1.000 → 0.978 v |
| mixed         | 16 | 0.803 → **0.681 v** | 0.747 → 0.724 v | 0.708 → **0.626 v** |
| prefill_lean  | 16 | 0.696 → 0.679 | 0.793 → **0.728 v** | 0.771 → **0.711 v** |
| prefill_bound | 8  | 0.971 → **0.992 ^** | 0.921 → **0.954 ^** | 0.925 → **0.953 ^** |
| prefill_bound | 12 | 0.264 → 0.238 v | 0.408 → 0.403 | 0.414 → 0.396 |
| prefill_bound | 16 | 0.061 → 0.062 | 0.062 → **0.161 ^** | 0.064 → **0.161 ^** |

**JOINT IS A WASH.** Helps on prefill_bound 8/16 and decode 12; HURTS on mixed 12/16, prefill_lean 16,
prefill_bound 12. It fixes NONE of the three diagnosed failures:
1. **The decode-bound veto survives** — joint full = 0.133–0.136, still exactly `never` (vs `always`
   0.271). Giving EDPP the decode choice does not stop z from vetoing disaggregation.
2. **The overload collapse survives** — joint prefill_bound@16 = 0.161 vs `least-ttft` 0.397.
3. **The full rule still wins in NO cell**, joint or reduced.

**THE FINDING — the lossy decomposition is NOT the binding problem.** Formulation §1's coupling
argument (the value of disaggregating depends on which decode node; the best decode node depends on
whether you disaggregate) motivates the ENTIRE joint construction and is the project's central
hypothesis. It is real but SECOND-ORDER: enumerating (d,p) jointly with the same terms does not help,
because **choosing jointly in the wrong currency is still choosing in the wrong currency.** The term
currency (work, and own-experience pricing) dominates the decomposition loss.

**Sharpest evidence.** On `mixed`, joint's only gift is that EDPP picks the decode instance instead
of the queue-depth scorer — and handed that control, the work-currency drift term picks WORSE than
plain load balancing (0.803 → 0.681). §5.5's "provably no worse" holds for the *objective as
formulated*; it does not protect goodput when the objective itself is mis-specified.

**Consequence.** Do NOT build the joint `least-ttft` baseline — joint does not move the needle, so a
joint baseline would measure nothing. The redesign target is unchanged and now better isolated: fix
what the terms MEASURE (value-at-risk, externality), not which action set they are minimised over.

**Caveats.** 1P2D only (|A| = 2 decode x 2 prefill-choices = 4 candidates — a wider pool might give
joint more to work with). Cap 16. 3 seeds. z_itl inert throughout.

---

## E10 / F12 — Oracle output-length control: `o_r` estimation is NOT the cause (2026-07-17)

**Question.** Is the overload collapse / decode-bound veto an artifact of output-length (`N̂_out`)
estimation error, rather than the work-vs-value currency (F11)?

**Mechanism.** New flag `--edpp-oracle-output-len` (DIAGNOSTIC / UPPER-BOUND / INV-9-violating):
substitutes the routed request's TRUE `len(req.OutputTokens)` for the per-class `N̂_out` when charging
its OWN decode work (`reqNHatOut` → joint `W_d` at `edpp.go:794/875`, and the `qdWork` backlog in
`OnRoute`). Co-resident remaining stays estimated/censored. Loud CLI warning; off by default (flat path
byte-identical).

```bash
ORACLE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
JOINT=1 ORACLE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```

**Result — near-no-op.** Reduced `prefill_bound` byte-identical est vs oracle; joint `full` r16
0.161 → 0.167; decode veto `full` = 0.133 unmoved; only `drift-only` decode-bound moves +0.02–0.03
(warmup transient). **Reason:** every archetype has a CONSTANT output length, so `N̂_out` converges to
the true `o_r` — almost no error to remove.

**F12.** Output-length estimation is not the cause of the collapse or the veto. Closes the "maybe it's
bad `o_r` estimates" escape hatch. Caveat: these constant-output workloads can't stress the estimate;
a VARIABLE-output workload is needed to test that dimension (still untested).

---

## E11 / F13 — Size-aware `c_xfer`: `least-ttft`'s overload robustness was a transfer-cost artifact (2026-07-17)

**Question (raised by V.).** The decider assumed a flat `--edpp-c-xfer 5ms`, but the transfer cost
depends on the KV size moved. Is 5ms wrong, and does it matter?

**The mismatch.** The DES *executes* a size-based KV transfer
(`sim/cluster/pd_events.go`: `base + blocks·blockSize·kvBytesPerToken / bandwidth`, added to the
disaggregated request's TTFT), while EDPP *decided* with a flat 5ms. Measured real transfers
(llama-70b TP4, 25 GB/s, from the DES `--log debug` `duration=` line):

| archetype | prefill tok | real transfer | flat c_xfer | error |
|-----------|-------------|---------------|-------------|-------|
| decode 256/512      | 256   | 1.1 ms  | 5 ms | 4.5× too big |
| mixed 2048/128      | 2048  | 7.0 ms  | 5 ms | ~right |
| prefill_lean 8192/64| 8192  | 27.1 ms | 5 ms | 5.4× too small |
| prefill_bound 16000/16 | 16000 | 52.7 ms | 5 ms | 10.5× too small |

**Mechanism.** New flag `--edpp-c-xfer-size-aware` (deployable, input-only): EDPP computes
`c_xfer = XferBaseUs + ⌈a_r/blockSize⌉·blockSize·KVBytesPerTokenPerGPU / bandwidth` per request
(`cXferUsFor`, `edpp.go`), mirroring the executor. KVBytesPerToken + bandwidth + base plumbed from the
cluster config. Applied to BOTH `ttftP` (→ `least-ttft`, `z_ttft`) AND the penalty term. Off by default
(flat path byte-identical). NOT used by `drift-only` (its decision reduces to `sign(q_d/W*−q_p/W*)`).

```bash
CXSIZE=1 MODE=ablate RATES="8 12 16" SEEDS="42 7 123" bash campaigns/edpp-study/repro_spectrum.sh
```

**Result (reduced, mean of 3 seeds), flat → size-aware:**

| archetype | rate | least-ttft | drift-only | drift+z | full |
|-----------|------|-----------|-----------|---------|------|
| prefill_bound | 12 | 0.492 → 0.242 | 0.264 → **0.264** | 0.408 → 0.238 | 0.414 → 0.233 |
| prefill_bound | 16 | **0.397 → 0.101** | 0.061 → **0.061** | 0.062 → 0.065 | 0.064 → 0.065 |
| mixed         | 16 | 0.750 → 0.368 | 0.803 → **0.803** | 0.747 → 0.410 | 0.708 → 0.417 |
| decode (veto) | 8  | 0.133 → 0.152 | 0.250 → **0.250** | 0.138 → 0.169 | 0.133 → 0.146 |

`drift-only` is **byte-identical** in every cell (clean invariant check — it never reads `c_xfer`).

**F13.** `least-ttft`'s "5–6× robustness under overload" (E4) was an artifact of under-charged transfer:
correcting to the true ~53ms collapses it (prefill_bound r16 0.397 → 0.101), closing the gap to
`drift-only` (0.061) to ~0.04. It was over-disaggregating because disagg looked cheap, landing near the
interior optimum by luck. **Core diagnosis intact:** `drift-only` invariant, `full` still wins 0/12, and
the decode veto survives the correct SMALLER cost (so F7 is NOT a `c_xfer` artifact). Reinforces F11:
true transfer pricing makes `least-ttft` more accurate about its OWN TTFT and thus more globally wrong —
it declines the disaggregation that helps the system.

**Note on the penalty term (V.'s follow-up).** The anomaly applies to the penalty too
(`transferPenalty` now takes the per-request `c_xfer`), but per F8 even the 10× scale-up leaves it
~0.008 vs `z_ttft`'s 25.7 — corrected for honesty, still a non-driver.

---

## E12 — Value-at-risk drift ORACLE vs `least-ttft` (the one live idea, tested at its ceiling)

Design: `docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md`. The §7 "one live idea" made
concrete: keep Neely's drift structure but re-price the drift term in **value-at-risk** — the marginal
goodput destroyed among the co-residents on the candidate decode instance — instead of in **work** (µs).
Built as a DIAGNOSTIC ORACLE (`--edpp-rule var`, kernels `--edpp-var-metric flip|util|hazard`): it reads
each co-resident's TRUE remaining decode steps (un-censored `TrueRemaining`, a gated INV-9 violation) to
project its completion under local vs disagg placement, then flips the reduced rule's LHS from the
work-currency balance term to `VaR_local − VaR_disagg`. Completion model = **full B+1 re-timing**
(co-resident per-iter time recomputed at batch `B+1`, `kv + a_r` after the routed request joins). g()'s
composite (TTFT ∧ mean-ITL ∧ E2E ≤ deadline) uses the SAME thresholds as the goodput metric (new
`--edpp-tau-e2e`). Composed with `--edpp-oracle-output-len` for a fully clean ceiling.

**The bar (§1):** oracle VaR must **clearly beat** the one-line `least-ttft` rule on the archetypes where
`least-ttft` ties/wins. If a perfect oracle can't, the value-currency idea is dead.

```bash
COEFFS=scripts/calibration/coeffs-llama70b-h100-tp4.json bash campaigns/edpp-study/repro_var_oracle.sh
```

**Result — contested cells only (mean of 3 seeds; cells where every arm scores 1.000 omitted):**

| archetype | rate | never | always | least-ttft | edpp | var:flip | var:util | var:hazard |
|-----------|------|-------|--------|-----------|------|----------|----------|-----------|
| decode (256/512)        | 4  | 0.414 | **1.000** | 0.414 | 0.414 | 0.981 | 0.981 | **1.000** |
| decode                  | 8  | 0.133 | 0.295 | 0.152 | 0.146 | 0.276 | 0.276 | **0.293** |
| decode                  | 16 | 0.133 | 0.267 | 0.133 | 0.133 | 0.214 | 0.214 | **0.267** |
| mixed (2048/128)        | 16 | 0.303 | 0.928 | 0.368 | 0.417 | 0.711 | 0.843 | **0.929** |
| prefill_lean (8192/64)  | 16 | 0.249 | 0.065 | 0.771 | **0.786** | 0.726 | 0.747 | 0.189 |
| prefill_bound (16000/16)| 8  | 0.276 | 0.033 | 0.947 | **0.967** | 0.878 | 0.943 | 0.958 |
| prefill_bound           | 16 | 0.049 | 0.026 | **0.101** | 0.065 | 0.082 | 0.090 | 0.079 |

**F14 — the oracle CLEARS the bar on the decode-bound and balanced archetypes, and only there.**
On `decode` and `mixed` — exactly where `least-ttft` is blind to the co-resident externality (it prices
only the deciding request's own TTFT) — VaR beats it decisively: `var:util` 0.843 vs 0.368 and `var:hazard`
0.929 vs 0.368 at mixed r16; `var:*` roughly doubles `least-ttft` across the saturated `decode` cells. This
is the design's central prediction confirmed: supplying the externality `least-ttft` lacks is worth a large
goodput gain when the decode pool is the contended resource. **On the prefill-bound archetypes the oracle
does NOT beat `least-ttft`** — it ties at moderate load (`var:util` 0.943 vs 0.947 at prefill_bound r8) and
neither clearly wins under deep overload (r16, all ≈ 0.08–0.10). There `least-ttft`'s own-TTFT signal is
already near-optimal and the placement externality is prefill-side, which VaR models only coarsely.

**F15 — kernel ranking: `util` is the robust pick; `hazard` over-disaggregates; the predicted B-trap did
not fire.** `var:hazard` is strongest on decode/balanced (it tracks `always`, which is right there) but
**collapses on `prefill_lean`** (0.189, tracking `always`'s 0.065) — it disaggregates too aggressively when
prefill is the bottleneck. `var:util` (kernel B) never collapses and captures most of the decode/balanced
win, making it the best-behaved kernel — notably, the §7-predicted "saturating utility ⇒ doomed-request
neglect" trap did **not** materialise at these operating points (constant-output, single-class). That is
"not observed here," not "refuted": the trap is a heterogeneous/multi-class phenomenon this grid does not
stress. `var:flip` is the noisy hyperparameter-free ceiling — directionally identical to `util`, lower.

**Verdict.** PARTIAL clearance. The value-currency idea has real, large merit **on the decode-bound and
balanced regimes** — enough to justify building a deployable approximation *for those regimes* (co-resident
remaining from the censored `N̂_out` estimate rather than the oracle; deferred per §7). It does **not**
rescue the prefill-bound regime, where `least-ttft`/`edpp` already win and `hazard` actively hurts. This is
the nuanced, per-archetype verdict §9 anticipated — not a single operating point. The honest boundary from
§7 stands: VaR gives externality-aware *placement*, not *triage*; a saturating utility yields neglect, not
capacity-yielding, because EDPP holds no admission/scheduling lever.

**F16 — joint path, HOMOGENEOUS hardware (`JOINT=1`; `least-ttft` is reduced-only ⇒ n/a).** The headline
verdict is unchanged: `var:*` clears the bar vs reduced `least-ttft` on decode/balanced (joint `var:util`
mixed r16 mean 0.796 vs reduced `least-ttft` 0.368) and not on prefill-bound (joint `var:util` prefill_lean
r16 0.743 ≈ reduced `least-ttft` 0.771; prefill_bound r8 0.911 vs 0.947). But two things stand out. (1)
**Joint-VaR beats joint-`edpp` (dpp argmin) almost everywhere** — mixed r16 0.796 vs 0.325, prefill_lean r16
`var:util` 0.743 vs `edpp` 0.622 — so VaR's win is not a reduced-path artifact; it improves the joint rule
too. (2) **Joint-`edpp` is no better than (often slightly worse than) reduced-`edpp`** on this homogeneous
grid (mixed r16 0.325 vs 0.417; prefill_lean r16 0.622 vs 0.786), consistent with F10: the joint
(decode,prefill) argmin earns nothing when every candidate shares the same θ_i. The ceiling on the prefill
archetypes is still set by the reduced path. `var:hazard` stays the over-aggressive kernel (collapses on
prefill_lean, tracking `always`), though it is oddly the least-bad arm under prefill_bound deep overload
(r16, all arms ≈ 0.05–0.31, noise-dominated). **This isolates the open question:** joint's structural value
requires heterogeneous θ_i (fast/slow decode nodes) — the axis F10/E5 identify and this homogeneous sweep
cannot exercise. A heterogeneous-θ_i VaR run (fast H100 + crippled-A100 pools via `--edpp-coeffs-by-gpu`,
mirroring `repro_hetero_hw.sh THETA=1`) is the natural next test, because a fast node destroys *less*
goodput per unit work — precisely the externality VaR prices and the work-currency θ_i drift over-corrected
on in E5.

**F17 — heterogeneous θ_i, saturating regime: VaR does NOT beat work-currency θ_i-joint dpp — it
over-routes to the fast node.** Ran the test above (`repro_var_oracle_hetero.sh`): a saturating 1P2D
deployment with the two decode instances on different hardware (fast H100 + crippled A100), per-instance
θ_i via `coeffs_by_gpu`, joint (decode,prefill) argmin. The goodput optimum is a NON-degenerate interior
decode split (~86% fast). Mean of 4 seeds:

| arm | goodput | realized fast-share |
|-----|---------|--------------------|
| optimum (fixed plan, 86% fast) | 0.949 | 86% |
| **θ_i-joint dpp** (work currency) | **0.928** | 80% |
| blind load-balance | 0.878 | 78% |
| var:util | 0.859 | ~91% |
| var:hazard | 0.859 | ~91% |
| var:flip | 0.470 | **100%** |

**VaR over-weights the fast node** (~91% fast, past the 86% optimum) and lands BELOW work-currency
θ_i-joint dpp (0.928) — and `var:util`/`hazard` even below blind load-balance. `var:flip` pins **100%**
onto the fast node every seed and craters (0.470): under a loose E2E deadline (8s) the fast node's
co-residents never cross their deadline, so the flip count on fast is identically 0 → VaR sees adding to
fast as always free → it never stops. This is the mirror image of dpp's failure, not a fix for it: the
goodput-currency externality is correctly signed but MIS-CALIBRATED on heterogeneous hardware — a fast
node destroys so little goodput per unit work that VaR concentrates load there past the throughput
optimum. Work-currency θ_i-drift, which DOES feel the fast node's rising per-iter time as it fills, stays
closer to the optimum here. **Net: the value-currency win is real on homogeneous decode/balanced (F14),
but it does NOT extend to heterogeneous provisioning** — the deployable-approximation recommendation stays
scoped to the decode/balanced regime. A calibration fix (tighter deadline sensitivity, or a hazard band
scaled to the node's own per-iter time) is a possible future lever, but is not part of this finding.

**F18 — drift-plus-VaR unifies both regimes: keep the congestion drift AND add the VaR externality.**
The F17 diagnosis said pure VaR over-routes because it prices the delay to the current batch and is blind
to the standing queue that over-concentration builds — exactly the snapshot-blindness the admission
estimator shows in overload. The fix is to keep the Lyapunov work-congestion drift term (which DOES feel a
node's backlog grow) and ADD the VaR externality, rather than replacing congestion with VaR:
`cost(i) = w · congestion_i + VaR_i + self terms` (new `--edpp-var-congestion [--edpp-var-congestion-weight w]`).

Mechanism (why one weight works in both regimes): on HOMOGENEOUS hardware the congestion term is symmetric
across identical decode nodes, so it cancels out of the argmin and VaR does the discriminating; on
HETEROGENEOUS hardware the fast node's backlog grows asymmetrically as it fills, so congestion bites and
reins in the over-routing. The weight only needs to be large enough to dominate WHERE congestion is
asymmetric; it is inert where it is symmetric. Verified with `w=10000`, `util` kernel, joint path:

| regime / cell | dpVaR:util (w=1e4) | joint-dpp | pure-VaR:util | best baseline |
|---|---|---|---|---|
| homog decode r8 (3 seeds) | 0.281 | 0.139 | 0.293 | never/always 0.19/0.29 |
| homog mixed r16 (3 seeds) | 0.860 | 0.325 | 0.796 | always 0.928 |
| hetero saturating (4 seeds) | **0.9995** | 0.928 | 0.859 | static-opt 0.949 |

On heterogeneous, dpVaR:util reaches ~1.00 on all four seeds — beating θ_i-joint dpp, pure VaR, blind
load-balance, and even the static-fraction optimum (it adapts per request, landing at ~82% fast vs the
static 86%). On homogeneous it preserves the VaR win (≈ pure VaR, ~2–3× over dpp), because congestion
cancels. **dpVaR:util is the only rule near-optimal across BOTH regimes** — always wins homog-mixed but
collapses on prefill/heterogeneous; dpp wins heterogeneous but loses homog decode/balanced; pure VaR is the
reverse. Only the `util` kernel unifies (hazard+congestion still over-routes, ~0.86 on hetero).

**Weight sensitivity + auto-normalization (caveat 2, now addressed).** The raw congestion weight is NOT a
knife-edge: sweeping `w` shows a broad plateau — heterogeneous dpVaR:util mean goodput 0.894 (w=100) →
0.965 (w=1e3) → 0.9995 (w=1e4) → 0.990 (w=1e5), and homogeneous mixed-r16 holds 0.79–0.80 across
w∈[100,1e4]. Any `w`∈[1e3,1e4] wins both regimes (a full order of magnitude wide), exactly as a Neely `V`
knob is presented. The large absolute value is only because congestion (work-µs, normalized) and VaR
(goodput units) live on different scales. Adding per-decision min-max normalization of the two terms across
the joint candidates (`--edpp-var-normalize`) makes the weight SCALE-FREE: the win moves to `w≈1` (hetero
seed42: w=0.5→1.000, w=1→1.000, w=2→0.970, w=5→0.978) and holds across seeds at `w=1` (hetero ~0.999 over
4 seeds; homog decode-r8 ~0.30 vs dpp ~0.14, mixed-r16 ~0.86 vs dpp ~0.33 over 3 seeds). A zero-spread guard
in the normalizer makes symmetric congestion (identical hardware) cancel automatically — the mechanism is
now built in, not hand-set. So drift-plus-VaR:util + auto-normalization at `w=1` is the clean form.

**Remaining honest scope:** (1) still an ORACLE (un-censored co-resident remaining + true o_r) — a ceiling;
a deployable variant (censored N̂_out) is untested. (2) one heterogeneous topology (H100 + crippled-A100,
1P2D, one saturating rate) and the homogeneous archetypes at specific rates — broader topology / rate /
heterogeneity-ratio robustness is future work. (3) the util kernel unifies; hazard does not. Within those
bounds this is a positive, unifying algorithmic result with a principled (scale-free) weight, not a diagnosis.

**F19 — the honest, deployable result: drift-plus-VaR is minimax-regret-adaptive, and F18's "unification"
was a constant-output ceiling.** Two follow-ups closed the F18 caveats and, in doing so, corrected the
claim.

*(a) Deployable ≈ oracle.* A DEPLOYABLE variant (`--edpp-var-deployable`) estimates each co-resident's
remaining steps from the censored per-class N̂_out (`max(N̂_out − StepsDone, 1)`, INV-9-safe) instead of the
oracle true remaining. It matches the oracle within noise everywhere tested (hetero constant: 0.925 vs
0.944; homog: ~equal) — so the result does NOT depend on reading hidden output lengths. Even the ORACLE
degrades under output variance, so the limit is the *mechanism*, not the estimate.

*(b) Variable output shrinks the win — it is minimax-regret, not domination.* On constant output dpVaR
beats dpp outright on heterogeneous (F18). On realistic VARIABLE output (lognormal σ=0.4, CV≈0.42) the
advantage narrows: dpVaR **ties** the regime winner rather than beating it. Regime × baseline dominance
grid (deployable, variable output, mean(min) over 3 seeds; `repro_var_dominance.sh`):

| archetype | never | always | least-ttft | dpp | dpVaR(deploy) |
|-----------|-------|--------|-----------|-----|---------------|
| decode r16        | 0.133 | **0.736** | 0.607 | 0.527 | 0.724 |
| mixed r16         | 0.356 | **1.000** | 0.708 | 0.621 | 0.946 |
| prefill_lean r16  | 0.204 | 0.061 | 0.806 | 0.819 | **0.853** |
| prefill_bound r8  | 0.399 | 0.046 | 0.954 | **0.968** | 0.964 |
| heterogeneous     | –     | 0.332 | n/a   | **0.942** | 0.900 |

dpVaR is NOT ≥ every baseline everywhere (trails `always` 5pts on mixed, `dpp` 4pts on heterogeneous). The
real claim is **minimax regret**: deployed blind, dpVaR is within ~5% of the best rule in every regime and
never craters, whereas every simple rule collapses somewhere. Worst-case regret across the grid: **dpVaR
0.054** vs dpp 0.38 vs least-ttft ≥0.29 vs always 0.92 — a 5–17× reduction. **This is the operational
value:** the rule you deploy when you cannot predict workload or hardware, cutting worst-case goodput loss
from 30–92% to ~5%. Scope: σ=0.4 (the heterogeneous gap to dpp widens modestly at higher CV — a
variance-axis figure is the natural follow-up); 3 seeds; util kernel; one heterogeneous topology. This
supersedes F18's "unifies both regimes" framing (true only on constant output) with the deployable,
variable-output minimax-regret result.

**F20 — historical comparison against a Kairos-inspired adaptation.** F19's grid compared only against
simple rules. We added a physics-normalized adaptation inspired by **Kairos** ("Towards Load-Aware
Prefill Deflection for Disaggregated LLM Serving", arXiv:2607.02043) as `--edpp-rule kairos`, plus
llm-d's shipped `prefix-threshold(16)` decider.

*Fidelity correction (2026-07-31).* These frozen results are not a reproduction of the published
algorithm. The historical arm omitted the paper's α=1.3 margin and request TTFT-SLO gate, used the
arriving request's TBT class rather than the strictest resident class, solved chunk size continuously,
added an occupancy-aware decode-admission estimate, approximated queued prompt lengths, and added KV
transfer to the regular-path estimate. These differences have mixed directional bias. The historical
numbers therefore compare against a **Kairos-inspired, physics-normalized adaptation**, not published
Kairos. The corrected implementation now exposes `--edpp-rule kairos-paper`; only fresh arms from that
mode may be used for a published-Kairos comparison. The old arm remains available as
`--edpp-rule kairos-adapted` (with `kairos` retained as its compatibility alias).

**Grid (deployable arms, variable output σ=0.4, mean over 3 seeds):**

| archetype | never | always | prefix16 | kairos* | least-ttft | dpp | dpVaR |
|-----------|-------|--------|----------|---------|-----------|-----|-------|
| decode r16       | 0.133 | **0.736** | 0.736 | **0.736** | 0.607 | 0.527 | 0.724 |
| mixed r16        | 0.356 | **1.000** | 1.000 | 0.883 | 0.708 | 0.621 | 0.946 |
| prefill_lean r16 | 0.204 | 0.061 | 0.061 | **0.856** | 0.806 | 0.819 | 0.853 |
| prefill_bound r8 | 0.399 | 0.046 | 0.046 | 0.890 | 0.954 | **0.968** | 0.964 |
| heterogeneous    | –     | 0.332 | 0.332 | 0.332 | 0.328 | **0.942** | 0.900 |

**Worst-case regret within this historical arm set:** dpVaR **0.054** | dpp 0.38 |
kairos-adapted* 0.61 | least-ttft 0.61 |
never 0.65 | always = prefix16 0.92.

The earlier interpretation that these rows establish an advantage over published Kairos is withdrawn.
They establish an advantage over the evaluated adapted arm only. Homogeneous and heterogeneous
comparisons against published Kairos require rerunning `kairos-paper` on the frozen workloads and seeds.

**Secondary finding: llm-d's shipped PD decider degenerates to `always`.** `prefix-threshold(16)` is
byte-identical to `always` in all five cells — every prompt exceeds the 16-token uncached threshold — so
it inherits `always`'s collapses (0.046 prefill_bound, 0.332 heterogeneous). A property of a shipped
production config, independent of our rule.

Scope unchanged from F19: σ=0.4, 3 seeds, one heterogeneous topology, simulation with coefficients fit to
the simulator's own latency model. Harness: `repro_var_dominance.sh` (its historical rows sweep the
adapted arm's β and must not be reused as `kairos-paper` results).

## Structural / topology ablations (2026-07-23) — dpVaR's edge survives the heterogeneity ratio AND the fleet shape

F19/F20 fixed the heterogeneity at ONE accelerator ratio (the crippled-A100 cell, decode-θ ratio ~4.8×)
and ONE topology (1P2D). This ablation varies both — the two axes the paper's threats section named as
open. Design: `docs/superpowers/specs/2026-07-23-edpp-structural-ablations-design.md`. Both experiments
reuse the F20 deployable arm set (σ=0.4, 3 seeds, best-β Kairos, rollforward + size-aware c_xfer). Paper
subsection: `infocom/joint-pd-routing.tex` `sec:eval:structural` (frozen tab:grid/tab:regret).

**F21 — the heterogeneity ratio at which the TTFT-currency rules break, and where dpVaR's margin peaks.**
Uniform-Nx slow node (scale BOTH tflops_peak and bw_peak_tbs by 1/N via `repro_theta_by_gpu.sh`); the
fitted decode per-token coefficient `c1` comes out at ratio EXACTLY N (memory-bound), so the ratio the
rule sees equals the knob. Coeff ladder: `scripts/calibration/coeffs-llama70b-ratio{1p0,1p2,1p5,2p0,3p0,5p0}-tp4.json`.
N=5 (c1=0.238) ≈ the published crippled-A100 cell (c1=0.228, N≈4.79), and at N=5 seed 42 reproduces the
tab:grid WORST-seed column exactly (dpVaR 0.777, dpp ~0.865) — the sweep extends the headline, it does
not restate it. 1P2D, rate 10, cap 8, mean over 3 seeds:

| N (=θ_slow/θ_fast) | always | kairos* | least-ttft | dpp | dpVaR | best static split |
|---|---|---|---|---|---|---|
| 1.0 | 0.783 | 0.783 | 0.768 | **1.000** | **1.000** | 1.000 |
| 1.2 | 0.554 | 0.554 | 0.554 | **1.000** | **1.000** | 1.000 |
| 1.5 | 0.462 | 0.462 | 0.462 | **1.000** | **1.000** | 1.000 |
| 2.0 | 0.421 | 0.421 | 0.421 | **1.000** | **1.000** | 0.998 |
| 3.0 | 0.379 | 0.379 | 0.379 | **0.994** | 0.941 | 0.990 |
| 5.0 | 0.328 | 0.328 | 0.322 | **0.940** | 0.889 | 0.888 |

**Worst-case regret vs the per-N best static split, across N∈[1,5]:** always/kairos/least-ttft **0.611** |
dpp **0.000** | dpVaR **0.049** (matches the 0.054 headline). The TTFT-currency rules (least-ttft, Kairos)
fall below half their homogeneous goodput by N=1.5 and never recover. dpp and dpVaR track the best static
split across the whole range; dpVaR's margin over least-ttft grows monotonically and reaches **0.57 at
N=5**. Note the oracle is the best *static* decode split (per-N grid search) — the joint rules route
adaptively and can EXCEED it, so their regret dips slightly negative at high N. Harness:
`repro_hetero_ratio_sweep.sh`; fig `out/hetero_ratio/hetero_ratio_sweep.png` → `infocom/figures/hetero_ratio_sweep.png`.

**F22 — the cancel/bind normalization holds as the fleet SHAPE changes; Kairos is provisioning-fragile.**
GPU-matched 16 accelerators provisioned as 1P3D / 2P2D / 3P1D (num-instances 4), HOMOGENEOUS hardware (no
bundle → the first-fit placement fragility does not apply). Four workload archetypes × 7 arms × 3 seeds,
one CSV, two framings. **Framing A — worst-case regret per topology (across the four archetypes):**

| topology | never | always | kairos* | least-ttft | dpp | dpVaR |
|---|---|---|---|---|---|---|
| 1P3D | 0.79 | 0.95 | 0.007 | 0.23 | 0.34 | **0.025** |
| 2P2D | 0.79 | 0.95 | 0.13 | 0.29 | 0.32 | **0.004** |
| 3P1D | 0.95 | 0.94 | **0.59** | 0.64 | 0.10 | **0.003** |

dpVaR's worst case stays ≤0.025 on every shape, matching its 1P2D headline — the §norm congestion term
still cancels where instances match and binds where they differ, with the shared weight fixed. NEW: Kairos
is near-optimal on the decode-heavy 1P3D shape (0.007) and collapses to 0.59 on the prefill-heavy 3P1D
shape, so its exposure follows the PROVISIONING as well as the workload (F20 showed only the workload
axis). dpp is worst on decode-heavy provisioning (0.34 on 1P3D) and best on 3P1D (0.10), the mirror image.

**Framing B — P:D provisioning adaptivity (the TaiChi answer).** Fixing the rule and reading its goodput
across the three provisionings, dpVaR sits at the per-provisioning BEST in all 16 archetype×provisioning
cells, so it adapts the effective split online with the one shared weight — vs TaiChi's offline,
minutes-scale reconfiguration. Fig `out/topo_matrix/pd_provisioning.png` → `infocom/figures/pd_provisioning.png`.
Harness: `repro_topology_matrix.sh`.

**Paper-vs-harness naming caveat (flagged, not silently resolved).** The fleet is built with
`--decode-instances` (decode-only pods that prefill LOCALLY on the collocated path, per
`TestDisaggregation_NonDisaggRoutedToDecodePoolOnly`), while the paper calls them "mixed (M)". The
published tab:grid uses the same setup, so the subsection uses paper M-notation (1P3M/2P2M/3P1M). True
mixed pods = `--prefill-decode-instances` (SharedInstances). The reduced rule charges the prefill side
with the global `d.coeffs`, so per-instance prefill-θ heterogeneity needs `--edpp-joint`.

Scope: σ=0.4, 3 seeds, one model/TP, homogeneous multi-topology (heterogeneity ratio is a separate axis,
F21), coefficients fit to the simulator's own latency model.

## F23 — collocated-prefill externality folded into the default rule; headline corrects to 0.042, topology re-run at 10 seeds (2026-07-25)

Decision: be true to the physics. The drift-plus-VaR rule now prices the first-token (TTFT) risk of
collocated mid-prefill occupants on the candidate decode instance by default. A request an earlier
collocated placement left mid-prefill lives in `decSnap.RunningPrefill`, and the decode-side VaR terms
skipped it, so a new collocate placement could delay that occupant's first token without the rule charging
for it. The term that prices it (`--edpp-var-colloc-prefill`, `varCollocPrefillLocal/Disagg`, INV-9-safe)
was default-off; it is now default-on. The paper's published Algorithm 1 already computed this re-timing
(`ω_j = min(n_c, remaining_j)`), so the flip aligns the default rule with the printed algorithm. The flag
survives as an ablation switch (`--edpp-var-colloc-prefill=false`). This SUPERSEDES the F19/F20 headline
(0.054 → 0.042) and the F22 topology row (3-seed 0.025/0.004/0.003 → 10-seed 0.033/0.015/0.006).

**Only the `var`+`joint` arm can move** (the flag reads only under `--edpp-rule var`). Non-var arms
reproduce bit-exactly, so every delta below is attributable to the term (verified: the colloc-OFF ablation
reproduces the old dpVaR baselines exactly).

**1P2D grid (3-seed, faithful):** dpVaR row moves balanced 0.946→**0.958**, prefill-lean 0.853→**0.858**,
prefill-bound 0.964→**0.957**; decode (0.724) and heterogeneous (0.900) unchanged. Worst-case regret
**0.054 → 0.042** (balanced). Kairos comparison: homogeneous-only 0.117 vs **0.042** (~2.8×); including
heterogeneity 0.61 vs 0.042 (~14.5×, envelope caveat unchanged). On prefill-lean dpVaR now edges Kairos
0.858 vs 0.856, a two-thousandth margin inside the seed spread — reported as a near-tie, not a lead.

**hetero_ratio_sweep:** 0.000 delta at every N (worst-case regret 0.049 unchanged). That config has no
mid-prefill collocation pressure, so the term is inert — a good null check.

**Topology matrix RE-RUN AT 10 SEEDS** (seeds 42 7 123 1 2 3 99 256 512 1024) to average out the seed-42
3P1D pathology the reviewer flagged (paper line 615). Framing-A worst-case regret per topology:

| topology | faithful (ON) | ablation (colloc-OFF) |
|---|---|---|
| 1P3D | 0.033 | 0.033 |
| 2P2D | 0.015 | 0.015 |
| 3P1D | **0.006** | **0.072** |

The term's effect is concentrated exactly where collocation is heaviest. On 3P1D, three prefill feeders
pour into one decode instance, so that instance carries many mid-prefill occupants, and pricing their
first-token risk cuts worst-case regret 12× (0.072 → 0.006). On 1P3D there is no such pressure (three
decode instances, little collocation) and the two are byte-identical. Physically sensible, and it makes the
topology robustness claim rest on the term rather than in spite of it.

Paper updated: tab:grid dpVaR column, tab:regret (0.042), tab:topo (10-seed 0.033/0.015/0.006),
pd_provisioning.png (ten-seed means), and all prose (factor of nine vs dpp, ~2.8× / ~14.5× vs Kairos,
limitations note states three seeds for the main grid and ten for the topology matrix). PDF builds clean,
18 pages. Harness: default binary for ON; `sed 's/--edpp-var-deployable/& --edpp-var-colloc-prefill=false/'`
on the repro scripts for the OFF ablation. ON data in `out/topo_matrix_10s/`, OFF in `out/topo_matrix_10s_off/`.
