#!/usr/bin/env bash
# Validate the fluid capacity model against measured policy plateaus.
#
# WHY. The paper's motivating table (never/always/pd_best per regime) is entirely
# analytic: it comes from gamma_cap.py's work model, never from measurement. The
# whole restructured paper rests on it, so it has to be checked before the
# rewrite, not after.
#
# WHAT. Drive three policies on the 1P2D topology deep into overload and read the
# achieved throughput plateau. The fluid model predicts a DIFFERENT ceiling for
# each, and the ordering is what the paper claims:
#
#   cell           always   never   pd_best (adaptive ceiling)
#   prefill_lean     8.32   15.02   20.01
#   prefill_bound    4.15    8.41   11.78
#
# `always` strands the decode pool behind one prefill instance. `never` cannot
# use the prefill-only instance at all. Only an assignment that puts SOME prefill
# on the mixed instances (interior theta, 0.703 on prefill_lean) reaches pd_best.
# So a measured dpvar plateau above never's 15.02 is direct evidence that the
# rule found an interior assignment -- no per-instance instrumentation needed.
#
# Concurrency 256 (simulator default), matching protocol v3. SLO targets are the
# v3 derived values; they do not gate throughput for never/always but the edpp
# rule consumes them.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/capacity_plateau; OUT="$D/out"; mkdir -p "$OUT"
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

jget(){ python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m.get(sys.argv[2],0))" "$1" "$2" 2>/dev/null||echo 0; }

printf '%-14s %-7s %-6s %8s %10s %10s %8s\n' cell arm seed probe achieved predicted ratio

# name in out probe_rate ttft itl e2e  pred_always pred_never pred_pd
while read -r NAME IN O PROBE T I E PA PN PP; do
  for s in $SEEDS; do
    W="$D/w_${NAME}_${s}.yaml"
    N=$(python3 -c "print(int($PROBE*$ARRIVAL_S))")
    spec "$IN" "$O" "$PROBE" "$N" "$s" "$W"
    SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
    EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
    VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint
         --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1
         --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")

    for arm in always never dpvar; do
      M="$OUT/${NAME}_${arm}_${s}.json"
      case $arm in
        always) ARGS=(--pd-decider always); PRED=$PA ;;
        never)  ARGS=(--pd-decider never);  PRED=$PN ;;
        dpvar)  ARGS=("${VVF[@]}");         PRED=$PP ;;
      esac
      ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
        "${ARGS[@]}" --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
      ACH=$(jget "$M" responses_per_sec)
      printf '%-14s %-7s %-6s %8s %10.3f %10s %8s\n' "$NAME" "$arm" "$s" "$PROBE" "$ACH" "$PRED" \
        "$(python3 -c "print(f'{$ACH/$PRED:.3f}')" 2>/dev/null||echo NA)"
    done
  done
done <<'ROWS'
prefill_lean   8192  64 50 212.0 27.37 1946.8  8.32 15.02 20.01
prefill_bound 16000  16 30 343.3 26.23  739.8  4.15  8.41 11.78
ROWS

echo
echo "PASS CRITERION: measured <= predicted (the fluid value is an upper bound),"
echo "and the ORDERING always < never < dpvar holds on both cells."
echo "A dpvar plateau above never's prediction proves an interior assignment."
echo DONE
