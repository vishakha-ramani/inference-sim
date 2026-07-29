#!/usr/bin/env bash
# The fair baseline: never@3M (three MIXED instances, no PD split) against
# dpvar@1P2D, at matched offered rates, with the v3-derived SLO targets.
#
# WHY. The ladder compared dpvar@1P2D against never@1P2D, where `never` cannot
# use the prefill-only instance at all and so runs on 2 of 3 instances. That is
# not a deployment anyone would choose: if you are not disaggregating you would
# not provision a prefill-only box. The realistic alternative to PD routing is an
# aggregated fleet of the same size, so never@3M is the baseline that matters.
#
# Capacity already measured: never@3M reaches 20.49 (prefill_lean) and 11.86
# (prefill_bound) vs dpvar@1P2D's 17.69 and 9.38. Aggregation wins on THROUGHPUT.
# The open question is GOODPUT: whether protecting decode from collocated prefill
# interference buys enough SLO attainment to offset the lower capacity.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/fair_baseline; OUT="$D/out"; mkdir -p "$OUT"
CAP=256; ARRIVAL_S="${ARRIVAL_S:-300}"; SEEDS="${SEEDS:-42}"
[[ -x ./blis ]] || go build -o blis main.go

spec(){ python3 - "$1" "$2" "$3" "$4" "$5" > "$6" <<'PY'
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
jg(){ python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); p=d['per_class']['standard']
print('%.3f %.3f %.1f'%(d['responses_per_sec'],p['slo_attainment'],p['ttft_p99_ms']))
" "$1" 2>/dev/null||echo "0 0 0"; }

printf '%-14s %-14s %6s %9s %8s %12s\n' cell arm rate achieved slo ttft_p99_ms
while read -r NAME IN O T I E RATES; do
  for R in $RATES; do
    for s in $SEEDS; do
      W="$D/w_${NAME}_${R}_${s}.yaml"; N=$(python3 -c "print(int($R*$ARRIVAL_S))")
      spec "$IN" "$O" "$R" "$N" "$s" "$W"
      SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
      EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
      # never@3M: three mixed instances, no PD split declared at all
      M="$OUT/${NAME}_${R}_never3m_${s}.json"
      ./blis run --model "$MODEL" --workload-spec "$W" \
        --num-instances 3 --routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP" \
        "${SLO[@]}" --pd-decider never --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
      read -r A SL TP <<<"$(jg "$M")"
      printf '%-14s %-14s %6s %9s %8s %12s\n' "$NAME" "never@3M" "$R" "$A" "$SL" "$TP"
      # dpvar@1P2D, for side-by-side at the same rate
      M="$OUT/${NAME}_${R}_dpvar1p2d_${s}.json"
      ./blis run --model "$MODEL" --workload-spec "$W" \
        --num-instances 3 --prefill-instances 1 --decode-instances 2 \
        --decode-routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP" "${SLO[@]}" \
        --pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint \
        --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 \
        --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms" \
        --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
      read -r A SL TP <<<"$(jg "$M")"
      printf '%-14s %-14s %6s %9s %8s %12s\n' "$NAME" "dpvar@1P2D" "$R" "$A" "$SL" "$TP"
    done
  done
done <<'ROWS'
prefill_lean   8192  64 212.0 27.37 1946.8 12 15 18
prefill_bound 16000  16 343.3 26.23  739.8 7 9 11
ROWS
echo; echo "If never@3M wins on BOTH throughput and goodput, PD disaggregation does not"
echo "pay on these homogeneous cells and the motivation must come from hetero."
echo DONE
