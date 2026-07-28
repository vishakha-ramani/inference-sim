#!/usr/bin/env bash
# Protocol v3, Phase A2: read the SLO targets off never@3M at 0.7 x capacity.
#
# Phase A1 (repro_slo_derivation_v3.sh) measures each cell's capacity as the
# throughput plateau in deep overload and writes it to out/slo_v3/capacity.txt.
# This script consumes that file, so the expensive overload runs are not
# repeated. Run A1 first if capacity.txt is missing.
#
# THE TARGETS ARE THE MEANS. A target set at the mean is aggressive on purpose:
# never@3M itself misses roughly half its own requests at the rate the target
# was read from. Under v2.1 the ITL target was so loose that not one request
# missed it in any cell, so the evaluation reported a two-dimensional good
# indicator while claiming three. Targets that bind are what let policies
# separate. The p99 columns are printed for reference only.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="meta-llama/llama-3.3-70b-instruct"
D=campaigns/edpp-study/out/slo_v3; OUT=$D/out; mkdir -p "$OUT"
CAP=256
ARRIVAL_S=400
FRAC=${FRAC:-0.7}

jget(){ python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m.get(sys.argv[2],0))" "$1" "$2" 2>/dev/null || echo 0; }

spec(){ # in out rate n file
  python3 - "$1" "$2" "$3" "$4" > "$5" <<'PY'
import sys, math
inp, out, rate, n = sys.argv[1:5]
mu = math.log(float(out)) - 0.08
print(f"""version: "2"
seed: 42
category: language
aggregate_rate: {rate}
num_requests: {n}
clients:
  - {{id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {{process: poisson}}, input_distribution: {{type: constant, params: {{value: {inp}}}}}, output_distribution: {{type: lognormal, params: {{mu: {mu:.4f}, sigma: 0.4, min: 4, max: {int(float(out))*8+16}}}}}, prefix_group: g, prefix_length: 0}}""")
PY
}

printf '%-14s %7s %10s %9s %9s %8s %10s %10s %9s\n' \
  cell rate ttft_mean ttft_p99 itl_mean itl_p99 e2e_mean e2e_p99 achieved

while read -r NAME IN OUTLEN ISHET; do
  CAPM=$(awk -v n="$NAME" '$1==n {v=$2} END {print v}' "$D/capacity.txt")
  RATE=$(python3 -c "print(round($FRAC*$CAPM, 2))")
  NREQ=$(python3 -c "print(int($RATE*$ARRIVAL_S))")
  W="$D/tgt_${NAME}.yaml"
  spec "$IN" "$OUTLEN" "$RATE" "$NREQ" "$W"
  M="$OUT/${NAME}_op.json"
  if [ "$ISHET" = "hetero" ]; then
    ./blis run --model "$MODEL" --workload-spec "$W" \
      --num-instances 3 --routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP" \
      --policy-config "$D/bundle.yaml" \
      --slo-ttft "standard=9999s" --slo-itl "standard=9999s" --slo-e2e "standard=9999s" \
      --pd-decider never --seed 42 --metrics-path "$M" >/dev/null 2>&1 || true
  else
    ./blis run --model "$MODEL" --workload-spec "$W" \
      --num-instances 3 --routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP" \
      --slo-ttft "standard=9999s" --slo-itl "standard=9999s" --slo-e2e "standard=9999s" \
      --pd-decider never --seed 42 --metrics-path "$M" >/dev/null 2>&1 || true
  fi
  TM=$(jget "$M" ttft_mean_ms); TP=$(jget "$M" ttft_p99_ms)
  IM=$(jget "$M" itl_mean_ms);  IP=$(jget "$M" itl_p99_ms)
  EM=$(jget "$M" e2e_mean_ms);  EP=$(jget "$M" e2e_p99_ms)
  AC=$(jget "$M" responses_per_sec)
  printf '%-14s %7s %10.1f %9.1f %9.2f %8.2f %10.1f %10.1f %9.2f\n' \
    "$NAME" "$RATE" "$TM" "$TP" "$IM" "$IP" "$EM" "$EP" "$AC"
done <<'ROWS'
decode        256   512 homog
mixed         2048  128 homog
prefill_lean  8192  64  homog
prefill_bound 16000 16  homog
hetero        256   64  hetero
ROWS

echo
echo "SLO targets = ttft_mean / itl_mean / e2e_mean at ${FRAC}x measured capacity."
echo DONE
