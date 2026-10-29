# Config attribution study

Can BLIS explain *why* a deployment config wins, not just *which* one? This study breaks each
metric into the pieces that sum to it and compares those pieces across configs.

Local study branch `study/config-attribution`, based on upstream `main` (adfca981). Nothing
here is committed or pushed.

## Layout

| Path | What it is |
|---|---|
| `charts/where-time-goes-32rps.png`, `charts/where-time-goes-64rps.png` | Slack-ready charts: llama-3.1-70b on H100, chatbot preset, TP 2/4/8. Share of total per piece; label is the total. |
| `charts/leaderboard-why-qwen3-14b.png` | Four "why does this config rank above that one" panels for the leaderboard's qwen3-14b group (6 req/s). Made by `scripts/png_leaderboard.py`. |
| `charts/where-the-time-goes.html` | Interactive version (metric, load, share/absolute, table view). Open it in a browser. |
| `data/breakdown.json` | Per-config breakdown of TTFT, ITL, E2E and GPU-time per token (the charts' data). |
| `data/counterfactual.json` | TP4 vs TP8 cost gap split into direct cost vs batching effect. |
| `data/sweep.json` | First per-term ledger sweep, TP4 vs TP8. |
| `data/lb_explain.json` | The leaderboard's qwen3-14b group re-run with the breakdown. |
| `data/lb_results.json` | Copy of `inference-sim/leaderboard` `prototypes/results.json`. |
| `charts/compare-<name>/` | Layer 2 comparisons from `scripts/compare.py`: `ttft.png`, `itl.png`, `e2e.png`, `cost.png`, `summary.md`. |
| `scripts/` | The prototype scripts that produced the above. `compare.py` + `attrib.py` are the current ones. |
| `runs/` | Raw BLIS outputs from `compare.py` (not tracked). |

## The simulator change

`sim/latency/study_term_ledger.go`, plus small hooks in `trained_physics_model.go`,
`latency.go` and `cmd/root.go`. Everything is off unless an environment variable is set, and
stdout is byte-identical with it on.

- `BLIS_STUDY_TERM_LEDGER=<file>` records every step's named trained-physics terms, totals per
  term, and per request, the terms of the steps it spent in prefill, decode, and in attempts
  discarded by preemption.
- `BLIS_STUDY_SHADOW_TP=<n>` also prices every executed batch under TP=n (counterfactual).

## Layer 2: compare configs (`scripts/compare.py`)

Vary one BLIS knob; the first value is the baseline. Each metric gets one PNG: the absolute
breakdown per value, then per compared value the impact of each piece (ms) and each piece's own
% change. Every run is checked that its pieces sum to what BLIS reports.

```bash
python3 study/config-attribution/scripts/compare.py \
  --model meta-llama/llama-3.1-70b-instruct --hardware H100 --tp 4 \
  --workload chatbot --rate 32 --vary tp=4,8 --name llama70b-tp4-vs-tp8
```

`--vary` accepts any `blis run` flag (`hardware=H100,A100-SXM`, `max-num-seqs=8,32,256`,
`rate=8,32,64`, ...). Extra BLIS flags go after `--`. The catalog defaults to
`../blis-catalog` next to the repo (override with `BLIS_CATALOG`).

## Layer 3: direct vs knock-on (`compare.py --cause`)

For `--vary tp=...` or `--vary hardware=...`, add `--cause`. Each config's run also prices every
batch it executed under the other config (`BLIS_STUDY_SHADOW_TP` / `BLIS_STUDY_SHADOW_HARDWARE`),
in both directions. Each change then splits into **direct** (same batches, new costs) and
**knock-on** (different batches and queueing), averaged over both orders; the two always sum to
the layer-2 change. A third panel per comparison row shows the split per component. Other knobs
only change scheduling, so their whole effect is knock-on and the split is skipped.

Sanity check: at matched concurrency (`--concurrency 64 --vary tp=4,8 --cause`), batch sizes are
equal and knock-on drops to ~0–1% for ITL, E2E and cost.

## Running the older scripts

They are prototypes. They expect to be run from the worktree root (BLIS reads `defaults.yaml`
from the working directory), with `blis` built there and the catalog cloned at
`/tmp/blis-catalog` (tag 0.1.1). They write raw run outputs to `/tmp/attrib/` as scratch space.

```bash
go build -o blis main.go
python3 study/config-attribution/scripts/breakdown.py   # chatbot TP×load breakdown
python3 study/config-attribution/scripts/png.py         # charts (reads /tmp/attrib/bd_slim.json)
```

## Caveats

- This is the breakdown as BLIS's trained-physics model computes it, not a GPU measurement.
- Decode compute is priced at zero (issue #1123).
- trained-physics is being replaced by `blis-latency-kernel`, which reports a per-resource
  breakdown (`StepTimeInto`) but composes step time as a per-layer max, so its resource
  totals do not sum to the step time.
