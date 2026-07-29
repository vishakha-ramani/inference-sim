#!/usr/bin/env bash
# Goodput against offered load on a 1P2D fleet, at rates where every arm is a
# live competitor.
#
# WHY THIS EXISTS
# ---------------
# Every `always` goodput on record reads 0.000, and every one of those readings
# was taken at a rate above `always`'s own capacity. On prefill_lean that capacity
# is 8.28 req/s and the ladder ran at 12, 15, and 18. On prefill_bound it is 4.15
# and the ladder ran at 7, 9, and 11. A saturated policy misses every deadline by
# construction, so those readings say nothing about routing. They restate the
# capacity result already established.
#
# This ladder puts at least two rates per cell BELOW `always`'s ceiling, where it
# serves everything offered and its goodput reports its routing rather than its
# saturation. Only there does a goodput comparison carry information.
#
# THE THREE ARMS
# --------------
#   always      --pd-decider always. Every request splits. The static corner an
#               operator gets by turning disaggregation on and leaving it.
#   dpvar       the full rule. Reads the deployable co-resident estimate.
#   plan@f*     --pd-plan holding the throughput-optimal share f* fixed. This is
#               the static arrangement that maximizes CAPACITY, measured at 0.39
#               on prefill_lean and 0.34 on prefill_bound. It is the reference an
#               adaptive rule has to beat to earn its complexity.
#
# `never` appears in neither form. never@3M is a different fleet and grades no
# policy here. never@1P2D leaves the dedicated prefill instance idle, measured at
# a mean batch occupancy of zero, so it competes with two instances against three.
#
# WHAT f* IS AND IS NOT
# ---------------------
# f* maximizes throughput. Nothing says it maximizes goodput. Raising the share
# that disaggregates shortens the prefill queue and enlarges the decode batch on
# the mixed instances, which trades first-token time against inter-token time, and
# a work-conservation model cannot see that trade. So plan@f* is a reference point
# and not a goodput ceiling. If dpvar beats it, the gap is what reacting to state
# buys. If plan@f* wins, a fixed share suffices on that cell and we should say so.
#
# RATES
# -----
# Fractions of the MEASURED ceiling (19.75 on prefill_lean, 11.64 on
# prefill_bound, from repro_mix_ceiling.sh), chosen so the ladder brackets
# `always`'s ceiling from both sides:
#
#   prefill_lean   4   6   8  12  16     always caps at  8.28  (0.42 of ceiling)
#   prefill_bound  2   3   4   6   9     always caps at  4.15  (0.36 of ceiling)
#
# The first two rates on each cell leave `always` under its own capacity. The
# third sits within a few percent of it. The last two are past it, where `always`
# is expected to collapse and the other two arms should not.
#
# TARGETS
# -------
# The v3 derived targets from specs/grid_v3/cells.txt, read off a three-mixed
# reference fleet at seven tenths of its capacity. That fleet runs none of the
# policies here, so no arm supplies the bar that grades it. Targets sit at the
# reference fleet's MEAN latency, so it misses about half its own requests at the
# rate they were read from. A target a generous margin satisfies would rank
# nothing.
#
# GATES
# -----
# A run is reported with a flag when it shed requests, left work unfinished, or
# failed to keep up with the offered rate. The last is not a defect for `always`
# above its ceiling, which is the point of those columns, so the flag is
# descriptive and the row still counts.
#
# Usage:
#   campaigns/edpp-study/repro_goodput_ladder.sh
#   CELLS=prefill_lean SEEDS="42 7 123" campaigns/edpp-study/repro_goodput_ladder.sh
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/${RUN:-goodput_ladder}
OUT="$D/out"; mkdir -p "$OUT"

CAP=256
ARRIVAL_S="${ARRIVAL_S:-300}"
SEEDS="${SEEDS:-42 7}"
CELLS="${CELLS:-prefill_lean prefill_bound}"

[[ -x ./blis ]] || go build -o blis main.go

# name  in  out  tau_ttft_ms  tau_itl_ms  tau_e2e_ms  fstar  rates...
rows_for(){ case "$1" in
  prefill_lean)  echo "8192  64 212.0 27.37 1946.8 0.39 4 6 8 12 16" ;;
  prefill_bound) echo "16000 16 343.3 26.23  739.8 0.34 2 3 4 6 9"  ;;
esac; }

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

read_row(){ python3 -c "
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print('0 0 0 0 0 0 0 0 0'); raise SystemExit
p=d['per_class']['standard']; dim=p.get('slo_attainment_by_dim',{})
print('%.3f %.4f %.4f %.4f %.4f %.1f %d %d %d'%(
  d['responses_per_sec'], p['slo_attainment'],
  dim.get('ttft',0), dim.get('itl',0), dim.get('e2e',0),
  p.get('ttft_p99_ms',0), d['dropped_unservable'], d['still_queued'], d['still_running']))" "$1"; }

printf '%-14s %-9s %6s %9s %8s %8s %8s %8s %10s %s\n' \
  cell arm rate achieved goodput g_ttft g_itl g_e2e ttft_p99 gate

for cell in $CELLS; do
  read -r IN O T I E FSTAR RATES <<<"$(rows_for "$cell")"
  SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
  EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware
      --edpp-tau-itl "${I}ms")
  VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint
       --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1
       --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")

  for R in $RATES; do
    for s in $SEEDS; do
      N=$(python3 -c "print(int($R*$ARRIVAL_S))")
      W="$D/w_${cell}_${R}_${s}.yaml"
      spec "$IN" "$O" "$R" "$N" "$s" "$W"
      P="$D/plan_${cell}_${s}_${N}.csv"
      [[ -f "$P" ]] || python3 campaigns/edpp-study/make_pd_plan.py --n "$N" --f "$FSTAR" > "$P"

      for arm in always dpvar "plan@f*"; do
        case $arm in
          always)   ARGS=(--pd-decider always); tag=always ;;
          dpvar)    ARGS=("${VVF[@]}");         tag=dpvar  ;;
          "plan@f*")ARGS=(--pd-plan "$P");      tag=planfs ;;
        esac
        M="$OUT/${cell}_${R}_${tag}_${s}.json"
        ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
          "${ARGS[@]}" --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
        read -r ACH G GT GI GE TP99 DR Q RU <<<"$(read_row "$M")"
        FLAG=$(python3 -c "
ach,r,dr,q,ru = $ACH,$R,$DR,$Q,$RU
b=[]
if dr: b.append('DROPPED=%d'%dr)
if q:  b.append('QUEUED=%d'%q)
if ru: b.append('RUNNING=%d'%ru)
if ach < 0.97*r: b.append('saturated')
print(','.join(b) if b else 'ok')")
        printf '%-14s %-9s %6s %9s %8s %8s %8s %8s %10s %s\n' \
          "$cell" "$arm" "$R" "$ACH" "$G" "$GT" "$GI" "$GE" "$TP99" "$FLAG"
      done
    done
  done
done

echo
echo "READ:"
echo "  The rates below always's ceiling (lean 4 and 6, bound 2 and 3) are the ones"
echo "  that carry information. There always serves everything offered, so its"
echo "  goodput reports its routing. If dpvar and plan@f* lead there, the win is"
echo "  routing quality at equal throughput. If they only lead at the higher rates,"
echo "  the win is capacity and the capacity section already made it."
echo "  plan@f* is the throughput-optimal STATIC share. It is a reference, not a"
echo "  ceiling on goodput, because f* is chosen without reference to the targets."
echo DONE
