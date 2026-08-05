# INFOCOM 2027 artifact and reproduction guide

This guide describes the `infocom-implementation` branch of
`vishakha-ramani/inference-sim`. The branch contains the simulator changes,
frozen inputs, registered experiment protocols, campaign runners, and compact
results used by the paper. It merges upstream `inference-sim/main` through
commit `3340de78` while retaining the paper's causal prefill-attention model,
joint routing policy, scheduler-rollout predictions, resident projections, and
decision instrumentation. Nothing in this artifact requires or modifies an
upstream branch.

The large raw campaign tree is intentionally not versioned. It is regenerated
under `campaigns/edpp-study/out/`; compact machine-readable results and reports
are tracked under `campaigns/edpp-study/results/infocom-2027/`.

## Requirements and setup

- Go 1.24 with toolchain 1.24.4 (declared in `go.mod`)
- Python 3.10 or newer
- PyYAML
- A CPU-only machine; a multicore machine is useful for `--jobs 12`

From a fresh clone:

```bash
git clone git@github.com:vishakha-ramani/inference-sim.git
cd inference-sim
git checkout infocom-implementation

python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install PyYAML

go build -o blis main.go
go test ./... -count=1
```

The campaign runners invoke the repository-root `./blis` binary. They are
resumable: existing complete run artifacts are reused unless `--force` is
given. Reduce `--jobs` if the host has fewer cores or limited memory.

## What this branch adds

The implementation is concentrated in these files:

| File | Role |
|---|---|
| `sim/edpp.go` | Joint `(decode, prefill)` action enumeration, routing score, per-class feedback, per-instance work state, and policy callbacks |
| `sim/edpp_var.go` | Arriving-request and resident SLO-value projections, including local and remote externalities |
| `sim/edpp_scheduler_rollout.go` | Frozen-snapshot scheduler rollout used to predict admission, first-token, and completion events |
| `sim/edpp_coeffs.go` | Calibrated phase-work and iteration-time model |
| `sim/edpp_kairos.go` | Paper-faithful Kairos comparator and its FIFO prefill state |
| `sim/cluster/cluster.go` | Cluster integration, live routing snapshots, lifecycle callbacks, and policy state updates |
| `sim/trace/edpp_joint_csv.go` | Per-candidate score, component, work, admission, and chosen-action traces |
| `cmd/root.go` | CLI configuration and trace wiring for the policy and ablations |

The corresponding `*_test.go` files cover the score identity, resident
projections, causal attention, scheduler rollout, callbacks, heterogeneous
coefficients, and trace fields. Upstream's lazy request source, token IDs, LoRA
state, and corrected disaggregated request timing are integrated rather than
forked.

The paper-specific experimental material is under
`campaigns/edpp-study/`:

- `PUBLIC-*-PROTOCOL.md` freezes selection, held-out seeds, validity gates, and
  analysis before each registered campaign.
- `run_public_*.py` builds workload specs, launches BLIS, validates terminal
  accounting and candidate traces, and writes the registered summaries.
- `workloads/public-closeout/` contains the three frozen public-workload
  marginals: interactive chat, reasoning, and deep research.
- `inputs/hetero-realistic-1p2d.yaml` defines the H100/A100 fleet without
  depending on a generated file under `out/`.
- `results/infocom-2027/` contains the compact final artifacts and checksums.

The earlier `feat/edpp-occupancy-predictor` work is not merged. It is an
obsolete experimental predecessor superseded by the scheduler rollout in this
branch.

## Frozen model and workload inputs

The final campaigns use Llama 3.3 70B at TP=4 and the following calibrated
iteration-time coefficients:

- `scripts/calibration/coeffs-llama70b-h100-tp4.json`
- `scripts/calibration/coeffs-llama70b-a100real-tp4.json`

The absolute CSV paths recorded inside those JSON files document calibration
provenance; the CSVs are not needed to reproduce the simulator experiments.
The A100 file and heterogeneous bundle are tracked specifically so a fresh
clone has every runtime input.

The public workload files are:

- `campaigns/edpp-study/workloads/public-closeout/interactive-chat-single-turn.yaml`
- `campaigns/edpp-study/workloads/public-closeout/reasoning-single-turn.yaml`
- `campaigns/edpp-study/workloads/public-closeout/deep-research-single-turn.yaml`

## Full paper reproduction

Run from the repository root. The order matters: the first campaign freezes
capacity and static-plan selections consumed by the ablation, counterfactual,
and stress runners.

```bash
python3 campaigns/edpp-study/run_public_load_static_benchmark.py --stage all --jobs 12
python3 campaigns/edpp-study/run_public_externality_decomposition_ablation.py --stage all --jobs 12
python3 campaigns/edpp-study/run_public_joint_counterfactual.py --stage all --jobs 12
python3 campaigns/edpp-study/run_public_mixed_burst_benchmark.py --stage all --jobs 12
python3 campaigns/edpp-study/run_public_final_topology_sweep.py --stage all --jobs 12
```

The default output directories are, in the same order:

```text
campaigns/edpp-study/out/public_load_static_benchmark_ttft_rollout_v2/
campaigns/edpp-study/out/public_externality_decomposition_ablation_ttft_rollout_v2/
campaigns/edpp-study/out/public_joint_counterfactual_ttft_rollout_v2/
campaigns/edpp-study/out/public_mixed_burst_benchmark_ttft_rollout_v2/
campaigns/edpp-study/out/public_final_topology_sweep_ttft_rollout_v2/
```

`--stage all` also performs each runner's registered selection or smoke stages
where applicable. Do not add `--force` when resuming an interrupted campaign;
use it only to recompute already complete runs.

## Fast smoke checks

After the main campaign has produced its capacity selection, each runner has a
small smoke stage:

```bash
python3 campaigns/edpp-study/run_public_load_static_benchmark.py --stage smoke --jobs 6
python3 campaigns/edpp-study/run_public_externality_decomposition_ablation.py --stage smoke --jobs 4
python3 campaigns/edpp-study/run_public_joint_counterfactual.py --stage smoke --jobs 4
python3 campaigns/edpp-study/run_public_mixed_burst_benchmark.py --stage smoke --jobs 2
python3 campaigns/edpp-study/run_public_final_topology_sweep.py --stage smoke --jobs 12
```

For a smoke-only fresh clone, seed the dependent runners with the tracked
capacity selection first:

```bash
mkdir -p campaigns/edpp-study/out/public_load_static_benchmark_ttft_rollout_v2
cp campaigns/edpp-study/results/infocom-2027/main/capacity_selection.json \
  campaigns/edpp-study/out/public_load_static_benchmark_ttft_rollout_v2/capacity_selection.json
```

These checks exercise workload generation, heterogeneous coefficients,
terminal accounting, action enumeration, and exact chosen-action argmins. They
are sanity checks, not substitutes for the held-out campaigns.

## Expected checkpoints

The 1,060 registered evaluation runs reported by the paper are the four
held-out confirmation cohorts below. Selection/calibration runs and the
counterfactual diagnostic are additional.

| Campaign | Confirmation runs | Expected checkpoint |
|---|---:|---|
| Main load/fleet benchmark | 432 | Causal-externality worst static-plan gap `0.0100`, equal-cell mean goodput `0.9209`; least-TTFT `0.0542`, Kairos `0.1050`, llm-d `0.3517` worst gaps |
| Externality/decomposition ablation | 288 | Full minus own-only `+0.0154`; full minus decode-first `+0.0485`; full minus resident-only `-0.0016` with a confidence interval crossing zero |
| Mixed/shift/burst stress | 160 | Causal-externality worst gap `0.0031`; least-TTFT `0.1110`, Kairos `0.0924`, llm-d `0.2421` |
| Topology sweep | 180 | Causal-externality worst gap `0.0100`; least-TTFT `0.0800`, llm-d `0.0859` |

The counterfactual campaign samples 144 online decisions across 18 cells and
runs 432 one-request forced deviations, plus 18 online and 18 exact replay-gate
runs. It should agree with a best-goodput forced placement for 91.0% of sampled
decisions. Of the 13 misses, 12 choose the wrong decoder and one chooses local
when remote is better.

Every confirmation report should show zero hard-invalid runs, drops, timeouts,
and output-length caps. Candidate-trace checks should report zero score-identity,
candidate-count, and chosen-argmin violations.

## Comparing regenerated results

The compact reference artifacts and their provenance are described in
`campaigns/edpp-study/results/infocom-2027/README.md`. Verify the checked-in
copies first:

```bash
shasum -a 256 -c campaigns/edpp-study/results/infocom-2027/CHECKSUMS.sha256
```

Then compare a regenerated campaign with its reference. For example:

```bash
diff -u \
  campaigns/edpp-study/results/infocom-2027/main/confirmation_result.json \
  campaigns/edpp-study/out/public_load_static_benchmark_ttft_rollout_v2/confirmation_result.json

diff -u \
  campaigns/edpp-study/results/infocom-2027/main/PUBLIC-LOAD-STATIC-BENCHMARK.md \
  campaigns/edpp-study/out/public_load_static_benchmark_ttft_rollout_v2/PUBLIC-LOAD-STATIC-BENCHMARK.md
```

Use the analogous `ablation`, `counterfactual`, `stress`, and `topology`
subdirectories for the other runners. Raw metrics, generated specs, stdout,
plans, and candidate traces remain under the ignored `out/` tree so the
validity and analysis can be audited without inflating the Git repository.

## Scope of reproduction

The commands above reproduce the CPU-only simulator evaluation and the paper's
reported policy comparisons. Repeating the real-vLLM validation or refitting
the coefficient files is a separate hardware experiment requiring the stated
H100/A100 GPU configurations and the calibration workflow under
`scripts/calibration/`.
