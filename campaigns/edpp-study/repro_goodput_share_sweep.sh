#!/usr/bin/env bash
# Goodput against the share of arrivals that disaggregate, at fixed sub-ceiling
# rates. Finds the goodput-optimal share and asks whether it moves.
#
# THE QUESTION
# ------------
# A plan holding the throughput-optimal share phi* fixed beats the adaptive rule at
# every rate on both prefill cells. phi* comes from a work-conservation model that
# knows nothing about the SLO targets, so there is no reason it should also
# maximize goodput. Two readings follow, and they say opposite things about the
# paper.
#
#   The goodput-optimal share moves with load or with workload. Then no fixed
#   share is deployable, an operator cannot pick one in advance, and the rule's
#   premise holds. The rule then has to FIND the share, and the measured
#   under-disaggregation says how it currently fails to.
#
#   One share works everywhere. Then an operator can set it once and the paper
#   needs a different argument for an adaptive rule.
#
# This sweep decides which. It also gives the goodput-optimal share as a target
# the rule can be measured against, which is a sharper test than comparing
# against phi*.
#
# WHY SUB-CEILING RATES
# ---------------------
# Goodput is only informative where the arms can serve the offered load. Above
# capacity every arm misses everything and the curve flattens at zero. Each cell
# runs at two rates below `always`'s own ceiling, chosen from the ladder, so the
# load dependence is visible:
#
#   prefill_lean   4 and 8    (always caps at 8.28, measured ceiling 19.75)
#   prefill_bound  2 and 4    (always caps at 4.15, measured ceiling 11.64)
#
# WHAT THE ENDPOINTS MEAN
# -----------------------
# Share 0 collocates everything on the two mixed instances and leaves the
# dedicated prefill instance idle, so it is `never` on this topology. It is not a
# baseline the paper compares against, and it is a legitimate point of the family.
# Share 1 splits every request, which is `always`.
#
# The realized share is read back off stdout on every run as a control. It must
# equal the requested share.
#
# Usage:
#   campaigns/edpp-study/repro_goodput_share_sweep.sh
#   SEEDS="42 7" campaigns/edpp-study/repro_goodput_share_sweep.sh
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
D=campaigns/edpp-study/out/${RUN:-goodput_share}; OUT="$D/out"; mkdir -p "$OUT"
SEEDS="${SEEDS:-42}"
SHARES="${SHARES:-0,0.10,0.20,0.30,0.34,0.39,0.50,0.65,0.80,1.0}"
CELLS="${CELLS:-prefill_lean prefill_bound}"
[[ -x ./blis ]] || go build -o blis main.go

# cell in out tau_ttft tau_itl tau_e2e phistar rates
#
# Rates sit below `always`'s own capacity on the 1P2D fleet, so every share is a
# live competitor. Those capacities are 8.28 on prefill_lean, 4.15 on
# prefill_bound, 34.00 on mixed, and 40.38 on decode.
#
# The two archetypes added below are the ones where phi* is far from the prefill
# cells' 0.34 to 0.39. decode wants every request split and mixed wants about
# half, so if the goodput-optimal share tracks phi* at all, it has to move here.
rows(){ case "$1" in
  prefill_lean)  echo "8192  64 212.0 27.37 1946.8 0.39  300 4 8"   ;;
  prefill_bound) echo "16000 16 343.3 26.23  739.8 0.34  300 2 4"   ;;
  decode)        echo "256  512  49.5 21.93 11293.1 1.00  300 12 24" ;;
  mixed)         echo "2048 128  94.8 28.51  3724.4 0.49  300 12 24" ;;
esac; }

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

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2
      --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 256)

printf '%-14s %6s %7s %9s %8s %8s %8s %8s %9s %8s %s\n' \
  cell rate share achieved goodput g_ttft g_itl g_e2e ttft_p99 realized gate

for cell in $CELLS; do
  read -r IN O T I E PHISTAR AS RATES <<<"$(rows "$cell")"
  SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
  for R in $RATES; do
    for s in $SEEDS; do
      N=$(python3 -c "print(int($R*$AS))")
      W="$D/w_${cell}_${R}_${s}.yaml"; spec "$IN" "$O" "$R" "$N" "$s" "$W"
      for PHI in ${SHARES//,/ }; do
        P="$D/plan_${cell}_${R}_${s}_${PHI}.csv"
        python3 campaigns/edpp-study/make_pd_plan.py --n "$N" --phi "$PHI" > "$P"
        M="$OUT/${cell}_${R}_${PHI}_${s}.json"; SO="${M%.json}.out"
        ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
          --pd-plan "$P" --seed "$s" --metrics-path "$M" >"$SO" 2>/dev/null || true
        read -r ACH G GT GI GE TP99 REAL FLAG <<<"$(python3 -c "
import json,re
d=json.load(open('$M')); p=d['per_class']['standard']; dim=p.get('slo_attainment_by_dim',{})
txt=open('$SO').read(); m=re.search(r'Disaggregated Requests:\s*([0-9]+)', txt)
inj=d['injected_requests']; dis=int(m.group(1)) if m else -1
real=('%.4f'%(dis/inj)) if inj and dis>=0 else 'n/a'
bad=[]
if d['dropped_unservable']: bad.append('DROPPED=%d'%d['dropped_unservable'])
if d['still_queued']: bad.append('QUEUED=%d'%d['still_queued'])
if d['still_running']: bad.append('RUNNING=%d'%d['still_running'])
e2e_s=d.get('e2e_mean_ms',0)/1000.0   # top-level field; per_class has only e2e_p99_ms
expect=d['injected_requests']/($AS + e2e_s) if ($AS + e2e_s) else 0
if expect and d['responses_per_sec'] < 0.95*min($R, expect): bad.append('saturated')
print('%.3f %.4f %.4f %.4f %.4f %.1f %s %s'%(
  d['responses_per_sec'], p['slo_attainment'], dim.get('ttft',0), dim.get('itl',0),
  dim.get('e2e',0), p.get('ttft_p99_ms',0), real, ','.join(bad) if bad else 'ok'))")"
        printf '%-14s %6s %7s %9s %8s %8s %8s %8s %9s %8s %s\n' \
          "$cell" "$R" "$PHI" "$ACH" "$G" "$GT" "$GI" "$GE" "$TP99" "$REAL" "$FLAG"
      done
    done
  done
done

echo
echo "READ: the goodput-optimal share is the peak of each rate's curve. Compare it"
echo "against phi_star (lean 0.39, bound 0.34), which maximizes THROUGHPUT. If the two"
echo "differ, or if the peak moves between the two rates or between the two cells,"
echo "then no fixed share is deployable. The realized column must equal the share."
echo DONE
