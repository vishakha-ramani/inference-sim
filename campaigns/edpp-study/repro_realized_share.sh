#!/usr/bin/env bash
# What share of requests does each arm actually disaggregate?
#
# WHY
# ---
# The goodput ladder found that a plan holding the throughput-optimal share f*
# fixed beats the adaptive rule at every rate, on both prefill cells. That result
# has two possible causes and they call for different fixes.
#
#   The aggregate mixture is wrong. The rule settles on a share away from f*, so
#   it is solving for the wrong split. Measurable here.
#
#   The mixture is right and the assignment is not. The rule lands near f* but
#   splits the wrong requests, or splits them at the wrong moments. That would
#   show up as a matching share with a worse goodput, which is the more
#   interesting finding.
#
# `printPDMetrics` (cmd/root.go) writes "Disaggregated Requests: N" to stdout on
# every run, so the realized share is N over the injected count. The ladder sent
# stdout to /dev/null, hence this pass.
#
# Two arms serve as controls. `always` must report a share of 1.000 and the plan
# must report f* exactly. If either misses, the reading is not what we think.
#
# Rates are the sub-ceiling points of the ladder, where `always` is under its own
# capacity and the goodput comparison carries information.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
LADDER=campaigns/edpp-study/out/goodput_ladder
D=campaigns/edpp-study/out/realized_share; OUT="$D/out"; mkdir -p "$OUT"
SEEDS="${SEEDS:-42}"
[[ -x ./blis ]] || go build -o blis main.go

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2
      --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 256)

# cell in out tau_ttft tau_itl tau_e2e fstar rates
rows(){ case "$1" in
  prefill_lean)  echo "8192  64 212.0 27.37 1946.8 0.39 4 6 8" ;;
  prefill_bound) echo "16000 16 343.3 26.23  739.8 0.34 2 3 4" ;;
esac; }

printf '%-14s %-9s %6s %11s %11s %9s %9s\n' cell arm rate injected disagg share f_star

for cell in prefill_lean prefill_bound; do
  read -r IN O T I E FSTAR RATES <<<"$(rows "$cell")"
  SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
  EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware
      --edpp-tau-itl "${I}ms")
  VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint
       --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1
       --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")

  for R in $RATES; do
    for s in $SEEDS; do
      # Reuse the ladder's own spec and plan so the trace is byte-identical.
      W="$LADDER/w_${cell}_${R}_${s}.yaml"
      N=$(python3 -c "print(int($R*300))")
      P="$LADDER/plan_${cell}_${s}_${N}.csv"
      [[ -f "$W" ]] || { echo "missing $W (run the ladder first)"; exit 1; }
      for arm in always dpvar "plan@f*"; do
        case $arm in
          always)   ARGS=(--pd-decider always); tag=always ;;
          dpvar)    ARGS=("${VVF[@]}");         tag=dpvar  ;;
          "plan@f*")ARGS=(--pd-plan "$P");      tag=planfs ;;
        esac
        SO="$OUT/${cell}_${R}_${tag}_${s}.out"
        ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
          "${ARGS[@]}" --seed "$s" --metrics-path "$OUT/${cell}_${R}_${tag}_${s}.json" \
          >"$SO" 2>/dev/null || true
        read -r INJ DIS SH <<<"$(python3 -c "
import re,sys,json
txt=open('$SO').read()
m=re.search(r'Disaggregated Requests:\s*([0-9]+)', txt)
dis=int(m.group(1)) if m else -1
d=json.load(open('$OUT/${cell}_${R}_${tag}_${s}.json'))
inj=d['injected_requests']
print(inj, dis, ('%.4f'%(dis/inj)) if inj and dis>=0 else 'n/a')")"
        printf '%-14s %-9s %6s %11s %11s %9s %9s\n' "$cell" "$arm" "$R" "$INJ" "$DIS" "$SH" "$FSTAR"
      done
    done
  done
done

echo
echo "READ: always must show 1.0000 and plan@f* must show f_star. If dpvar's share"
echo "sits away from f_star, the rule is solving for the wrong split. If it sits on"
echo "f_star while losing goodput, the split is right and the per-request choice is"
echo "what costs it."
echo DONE
