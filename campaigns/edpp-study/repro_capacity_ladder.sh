#!/usr/bin/env bash
# Follow-up to repro_capacity_plateau.sh: test the adaptive-headroom claim IN the
# domain where the paper claims it.
#
# The plateau run offered 2.5x capacity. There dpvar landed BETWEEN always and
# never on both prefill cells (lean 10.88 vs never 13.66; bound 5.27 vs never
# 7.91), far short of the pd_best ceiling. But that is deep overload, which the
# paper's own Assumption 3 excludes: the collocated admission estimator
# under-predicts by one to two orders of magnitude once a standing queue forms,
# so the rule is being scored on an input it is documented not to trust.
#
# The claim that matters is narrower and testable: at rates BETWEEN never's
# ceiling and pd_best, only an interior assignment can keep up. If dpvar sustains
# those rates and never cannot, the headroom claim holds where it is made.
#
#   prefill_lean : never measured 13.66, pd_best 20.01 -> probe 12 / 15 / 18
#   prefill_bound: never measured  7.91, pd_best 11.78 -> probe  7 /  9 / 11
#
# Read achieved throughput AND slo_attainment. At rate 15 on prefill_lean, never
# is above its own ceiling and must fall behind; dpvar should not.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/capacity_ladder; OUT="$D/out"; mkdir -p "$OUT"
CAP=256
ARRIVAL_S="${ARRIVAL_S:-300}"
SEEDS="${SEEDS:-42}"
[[ -x ./blis ]] || go build -o blis main.go

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

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2
      --decode-routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP")

jg(){ python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); p=d['per_class']['standard']
print('%.3f %.3f %.1f'%(d['responses_per_sec'],p['slo_attainment'],p['ttft_p99_ms']))
" "$1" 2>/dev/null||echo "0 0 0"; }

printf '%-14s %-7s %6s %9s %9s %8s %12s\n' cell arm rate offered achieved slo ttft_p99_ms

while read -r NAME IN O T I E RATES; do
  for R in $RATES; do
    for s in $SEEDS; do
      W="$D/w_${NAME}_${R}_${s}.yaml"
      N=$(python3 -c "print(int($R*$ARRIVAL_S))")
      spec "$IN" "$O" "$R" "$N" "$s" "$W"
      SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
      EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
      VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint
           --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1
           --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")
      for arm in always never dpvar; do
        M="$OUT/${NAME}_${R}_${arm}_${s}.json"
        case $arm in
          always) ARGS=(--pd-decider always) ;;
          never)  ARGS=(--pd-decider never)  ;;
          dpvar)  ARGS=("${VVF[@]}")         ;;
        esac
        ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
          "${ARGS[@]}" --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
        read -r ACH SLOA TP99 <<<"$(jg "$M")"
        printf '%-14s %-7s %6s %9s %9s %8s %12s\n' "$NAME" "$arm" "$R" "$R" "$ACH" "$SLOA" "$TP99"
      done
    done
  done
done <<'ROWS'
prefill_lean   8192  64 212.0 27.37 1946.8 12 15 18
prefill_bound 16000  16 343.3 26.23  739.8 7 9 11
ROWS

echo
echo "READ: at a rate above never's measured ceiling (lean >13.66, bound >7.91),"
echo "never must fall behind. If dpvar keeps up there, the headroom claim holds"
echo "in-domain. If dpvar also falls behind, the adaptive claim has no support."
echo DONE
