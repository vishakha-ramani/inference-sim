#!/usr/bin/env bash
# Protocol v3, Phase A: derive SLO targets from never@3M at 0.7 x its capacity.
#
# WHY THIS REPLACES THE v2.1 DERIVATION. v2.1 read p99s from an `always` probe at
# rate 0.2 and multiplied by 5x/4x/3x. Two problems. The probe ran the
# DISAGGREGATED path, so every target was anchored to one policy's routing
# choice -- and on the prefill cells that is the path which degrades first. And
# an idle probe cannot see load-dependent latency at all, so the ITL target came
# out wider than any achievable ITL and never bound.
#
# v3 instead drives `never` on a plain aggregated 3-instance fleet -- no PD
# split, no PD decider, just queue-depth load balancing -- and reads the
# MEAN latencies at 0.7 x that fleet's measured saturation throughput. Policy-
# neutral, because no PD machinery is involved, and at a loaded operating point,
# so ITL has actually grown by the time we read it.
#
# TARGETS ARE THE MEANS, NOT THE p99s. A target set at the mean is aggressive by
# construction: never@3M itself misses roughly half of its own requests at the
# rate the target was read from. That is deliberate. Under v2.1 the ITL target
# was so loose that not one request missed it in any cell, so the evaluation
# reported a two-dimensional good indicator while claiming three. Targets that
# bind are what let policies separate. The p99 columns are printed alongside for
# reference.
#
# CAPACITY IS MEASURED, NOT INFERRED FROM A RATIO. The v2.1 knee criterion
# (achieved/offered >= 0.95) is contaminated: responses_per_sec is
# CompletedRequests / SimEndedTime, so it divides a fixed request count by a
# duration that includes the post-arrival drain, and it is non-monotone in rate
# near the knee (see RESULTS_capacity.md section 4). Here capacity is the
# throughput PLATEAU read in deep overload, where the measurement is
# unambiguous, and it is cross-checked against the analytical bound from
# gamma_cap.py.
#
# CAVEAT TO STATE IN THE PAPER: the targets are read off one fleet shape at one
# operating point, so never@3M is the reference the targets were calibrated on.
# It is policy-neutral in the sense that no PD machinery is involved, not in the
# sense that every fleet shape is treated symmetrically.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="meta-llama/llama-3.3-70b-instruct"
D=campaigns/edpp-study/out/slo_v3; OUT=$D/out; mkdir -p "$OUT"
CAP=256          # simulator default; v2.1 used 16, which pinned ITL (see RESULTS_capacity.md)
ARRIVAL_S=400    # seconds of arrivals per run

jget(){ python3 -c "import json;m=json.load(open('$1'));print(m.get('$2',0))" 2>/dev/null||echo 0; }

spec(){ # in out rate n seed file
  python3 - "$1" "$2" "$3" "$4" "$5" > "$6" <<'PY'
import sys, math
inp,out,rate,n,seed = sys.argv[1:6]
mu = math.log(float(out)) - 0.08
print(f"""version: "2"
seed: {seed}
category: language
aggregate_rate: {rate}
num_requests: {n}
clients:
  - {{id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {{process: poisson}}, input_distribution: {{type: constant, params: {{value: {inp}}}}}, output_distribution: {{type: lognormal, params: {{mu: {mu:.4f}, sigma: 0.4, min: 4, max: {int(float(out))*8+16}}}}}, prefix_group: g, prefix_length: 0}}""")
PY
}

cat > "$D/bundle.yaml" <<'YAML'
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
coeffs_by_gpu:
  H100: scripts/calibration/coeffs-llama70b-h100-tp4.json
  A100: scripts/calibration/coeffs-llama70b-a100crippled-tp4.json
YAML

NOSLO=(--slo-ttft "standard=9999s" --slo-itl "standard=9999s" --slo-e2e "standard=9999s")
# Aggregated 3-instance fleet: no prefill/decode split, no PD decider.
AGG=(--num-instances 3 --routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP")

run(){ # name in out rate tag extra...
  local NAME=$1 IN=$2 O=$3 R=$4 TAG=$5; shift 5
  local N; N=$(python3 -c "print(int($R*$ARRIVAL_S))")
  spec "$IN" "$O" "$R" "$N" 42 "$D/w_${NAME}_${TAG}.yaml"
  ./blis run --model "$MODEL" --workload-spec "$D/w_${NAME}_${TAG}.yaml" "${AGG[@]}" "$@" \
    "${NOSLO[@]}" --pd-decider never --seed 42 \
    --metrics-path "$OUT/${NAME}_${TAG}.json" >/dev/null 2>&1 || true
}

# name, in, out, overload probe rate (~2x the analytical aggregated bound:
# 1.5 x cap_coll from gamma_cap.py --batch-cap 256, since 3M has three
# collocated instances where the 1P2D coll figure counts two).
#   decode 58.7 | mixed 86.4 | prefill_lean 22.5 | prefill_bound 12.6 | hetero see bundle
echo "== Phase A1: capacity plateau (deep overload, never@3M) =="
printf "%-14s %8s %10s %12s %10s\n" cell probe_rate achieved analytic_est ratio
while read -r NAME IN O PROBE EST EXTRA; do
  EX=(); [ "$EXTRA" = "bundle" ] && EX=(--policy-config "$D/bundle.yaml")
  run "$NAME" "$IN" "$O" "$PROBE" overload ${EX[@]+"${EX[@]}"}
  ACH=$(jget "$OUT/${NAME}_overload.json" responses_per_sec)
  printf "%-14s %8s %10.3f %12s %10s\n" "$NAME" "$PROBE" "$ACH" "$EST" \
    "$(python3 -c "print(f'{$ACH/$EST:.3f}')")"
  echo "$NAME $ACH" >> "$D/capacity.txt"
done <<'ROWS'
decode        256   512 120 58.7 -
mixed         2048  128 175 86.4 -
prefill_lean  8192  64   45 22.5 -
prefill_bound 16000 16   25 12.6 -
hetero        256   64  400 272  bundle
ROWS

echo
echo "== Phase A2: operating point at 0.7 x measured capacity =="
printf "%-14s %8s %9s %9s %9s %9s %9s %9s %9s\n" \
  cell rate ttft_mean ttft_p99 itl_mean itl_p99 e2e_mean e2e_p99 achieved
while read -r NAME IN O _ _ EXTRA; do
  EX=(); [ "$EXTRA" = "bundle" ] && EX=(--policy-config "$D/bundle.yaml")
  CAPM=$(grep "^$NAME " "$D/capacity.txt" | tail -1 | awk '{print $2}')
  R=$(python3 -c "print(round(0.7*$CAPM, 2))")
  run "$NAME" "$IN" "$O" "$R" op07 ${EX[@]+"${EX[@]}"}
  M="$OUT/${NAME}_op07.json"
  printf "%-14s %8s %9.1f %9.1f %9.2f %9.2f %9.1f %9.1f %9.2f\n" "$NAME" "$R" \
    "$(jget "$M" ttft_mean_ms)" "$(jget "$M" ttft_p99_ms)" \
    "$(jget "$M" itl_mean_ms)"  "$(jget "$M" itl_p99_ms)" \
    "$(jget "$M" e2e_mean_ms)"  "$(jget "$M" e2e_p99_ms)" \
    "$(jget "$M" responses_per_sec)"
done <<'ROWS'
decode        256   512 120 58.7 -
mixed         2048  128 175 86.4 -
prefill_lean  8192  64   45 22.5 -
prefill_bound 16000 16   25 12.6 -
hetero        256   64  400 272  bundle
ROWS

echo
echo "SLO targets = the MEAN columns above (p99 shown for reference)."
echo "Capacity in $D/capacity.txt"
echo DONE
