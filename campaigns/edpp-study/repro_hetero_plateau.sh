#!/usr/bin/env bash
# Locate dpvar@1P2D's capacity on the heterogeneous fleet, and use it as a
# mechanism test.
#
# The hetero comparison drove both arms at 70/90/110, anchored on never@3M's
# measured capacity of 109.10. dpvar kept up at all three, so its own plateau is
# unmeasured and we cannot state what utilization it was running at.
#
# The plateau also discriminates HOW dpvar uses the slow instance. With the ITL
# target enforced (alpha_D = 69.31 ms > tau_itl = 42.74 ms, feasible decode batch
# zero), the analytic 1P2D ceilings are:
#     slow instance used as a DECODE instance   -> 143.57 req/s
#     slow instance used as a PREFILL instance  -> 252.01 req/s
# A plateau near 250 is evidence the rule quarantines the slow box into prefill.
# A plateau near 145 says it does not, and the goodput win comes from somewhere
# else. never@3M's own measured plateau is 109.10 against an analytic 324.76.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/hetero_plateau; OUT="$D/out"; mkdir -p "$OUT"
CAP=256; ARRIVAL_S="${ARRIVAL_S:-300}"; SEEDS="${SEEDS:-42}"
T=80.5; I=42.74; E=2783.3
cp campaigns/edpp-study/out/hetero_fair/bundle.yaml "$D/bundle.yaml"
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
bd=p.get('slo_attainment_by_dim',{}) or {}
print('%.3f %.3f %.3f %.3f'%(d['responses_per_sec'],p['slo_attainment'],bd.get('itl',0),bd.get('ttft',0)))
" "$1" 2>/dev/null||echo "0 0 0 0"; }

printf '%-12s %7s %10s %8s %8s %8s\n' arm offered achieved slo d_itl d_ttft
for R in 150 200 300; do
  for s in $SEEDS; do
    W="$D/w_${R}_${s}.yaml"; N=$(python3 -c "print(int($R*$ARRIVAL_S))")
    spec 256 64 "$R" "$N" "$s" "$W"
    SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
    EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
    M="$OUT/${R}_dpvar_${s}.json"
    ./blis run --model "$MODEL" --workload-spec "$W" --num-instances 3 \
      --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" \
      --max-num-running-reqs "$CAP" --policy-config "$D/bundle.yaml" "${SLO[@]}" \
      --pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint \
      --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 \
      --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms" \
      --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
    printf '%-12s %7s %10s %8s %8s %8s\n' "dpvar@1P2D" "$R" $(jg "$M")
  done
done
echo; echo "Analytic 1P2D ceilings with ITL enforced: slow-as-decode 143.57, slow-as-prefill 252.01."
echo "never@3M measured plateau 109.10 (analytic 324.76)."
echo DONE
