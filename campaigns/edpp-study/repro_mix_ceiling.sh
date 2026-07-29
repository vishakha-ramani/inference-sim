#!/usr/bin/env bash
# Measure the EXPERIMENTAL capacity ceiling of a 1P2D fleet by sweeping the
# mixing fraction f under a forced plan, and locate f*.
#
# WHY THIS EXISTS
# ---------------
# Every "fraction of ceiling" number in the study divides by an ANALYTIC ceiling
# that has never been measured. `always` and `never` are measurable because their
# assignment is fixed, and both reproduce their predictions (always to 0.999).
# The ceiling itself is not a policy, so it cannot be measured by running one.
#
# --pd-plan makes it measurable. It forces a per-request (decode, prefill) plan,
# so a plan holding a fixed fraction f of requests disaggregated realises one
# point of the fluid capacity curve. Sweeping f traces the curve and its peak is
# the ceiling.
#
# WHAT f MEANS
# ------------
# f = fraction of requests whose prefill runs on instance 0 (the prefill-only
# instance) and whose decode runs on a mixed instance. The remaining 1 - f are
# served whole on a mixed instance (prefill_instance = local, so disaggregation
# does not fire). Decode is split evenly across the mixed pool.
#
#   f = 0  is exactly `never` on 1P2D   (nothing uses the prefill instance)
#   f = 1  is exactly `always` on 1P2D  (every request splits)
#
# Those two endpoints are run BOTH as plans and as --pd-decider never / always,
# and they must agree. That is the harness self-check: if a plan at f = 1 does not
# reproduce `always`, the plan mechanism is wrong and no interior point is
# trustworthy.
#
# WHY THE OFFERED RATE VARIES WITH f
# ----------------------------------
# Capacity is only observable when the server is the bottleneck, so every point
# must be overloaded. But the offered rate cannot simply be large: what triggers
# `dropped_unservable` is the BACKLOG, and a single large rate sheds requests at
# the f values whose capacity is lowest. A run with drops is not a capacity
# measurement. So each point is offered a fixed modest multiple (default 1.15x) of
# its own PREDICTED capacity, which keeps the backlog comparable across f.
#
# Trusting the prediction that way would be circular, so every run is gated:
#   - dropped_unservable, still_queued, still_running must all be zero
#   - achieved must be below 0.97 x offered, i.e. genuinely saturated
# A point failing either gate is printed with a flag and must not be read as a
# capacity.
#
# WHY responses_per_sec IS STILL SLIGHTLY LOW
# -------------------------------------------
# It is completed requests over total simulation duration, and the duration
# includes the transient while the batch fills. The duration is linear in N,
# duration = T0 + N/C with T0 about 3 s, so the understatement is T0/(T0 + N/C).
# At the default N that is under 2%, and it biases DOWNWARD, which is the correct
# side of an upper bound. N is chosen large enough for that and small enough that
# the backlog never reaches the shedding threshold.
#
# WHAT TO READ
# ------------
# The measured column should rise, peak near the predicted f*, and fall. A peak
# above `never`'s measured ceiling is only reachable by putting prefill onto the
# decode instances, so the peak's location is direct evidence of the interior
# mixing the fluid model predicts.
#
# Usage:
#   campaigns/edpp-study/repro_mix_ceiling.sh                     # both cells
#   CELLS=prefill_lean campaigns/edpp-study/repro_mix_ceiling.sh
#   N=6000 SEEDS="42 7" campaigns/edpp-study/repro_mix_ceiling.sh
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
# RUN names the output subdirectory so a re-run of a subset does not overwrite
# valid measurements from an earlier pass.
D=campaigns/edpp-study/out/${RUN:-mix_ceiling}
OUT="$D/out"
mkdir -p "$OUT"

CAP=256                                   # --max-num-running-reqs (simulator default)
N="${N:-4600}"                            # requests per run; see the note above
OVERLOAD="${OVERLOAD:-1.15}"              # offered rate as a multiple of C_pred(f)
SEEDS="${SEEDS:-42}"
F_VALUES="${F_VALUES:-0,0.10,0.20,0.28,0.31,0.34,0.39,0.45,0.55,0.70,1.0}"
CELLS="${CELLS:-prefill_lean prefill_bound}"

[[ -x ./blis ]] || go build -o blis main.go

# Cell definitions: input tokens and mean output tokens. Same four homogeneous
# cells as specs/grid_v3/cells.txt; only the two prefill cells are swept, because
# the decode cell's predicted f* sits at the boundary f = 1 (nothing to locate)
# and the balanced cell's is sensitive to the batch cap.
cell_in()  { case "$1" in prefill_lean) echo 8192;; prefill_bound) echo 16000;;
                          decode) echo 256;; mixed) echo 2048;; esac; }
cell_out() { case "$1" in prefill_lean) echo 64;;   prefill_bound) echo 16;;
                          decode) echo 512;; mixed) echo 128;; esac; }

# Workload spec: constant input length, lognormal outputs (sigma 0.4), Poisson
# arrivals, no prefix reuse. Byte-identical generator to repro_capacity_ladder.sh
# so the ceilings measured here are comparable with the always/never plateaus
# already on record.
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

# No SLO targets are passed. This measures throughput capacity, and a target
# would only add a column that reads 0 everywhere in overload. Goodput at
# sub-capacity rates is a separate experiment.

# Read achieved throughput plus the four conservation counters the gates need.
read_metrics(){ python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    print('0 0 0 0 0 0'); raise SystemExit
print('%.4f %d %d %d %d %.1f'%(d['responses_per_sec'],d['injected_requests'],
      d['dropped_unservable'],d['still_queued'],d['still_running'],
      d['vllm_estimated_duration_s']))" "$1"; }

# Achieved batch occupancy per role, from the per-instance JSON on stdout. The
# metrics-path file is cluster-level only, and the cluster figure blends the
# prefill-only instance (which runs a batch near one) with the mixed pool, so it
# cannot be compared against the capacity model's per-instance B.
read_xbar(){ python3 -c "
import sys,json,re
txt=open(sys.argv[1]).read() if len(sys.argv)>1 else ''
per={}
for m in re.finditer(r'\{[^{}]*\"instance_id\"[^{}]*\}', txt, re.S):
    try: d=json.loads(m.group(0))
    except Exception: continue
    per[d.get('instance_id')]=d.get('mean_running_batch',0.0)
mixed=[v for k,v in per.items() if k not in ('instance_0','cluster')]
print('%.2f %.2f'%(sum(mixed)/len(mixed) if mixed else 0.0, per.get('instance_0',0.0)))" "$1" 2>/dev/null || echo "0 0"; }

# One run. arm is either a plan fraction ("f=0.39") or a reference decider.
run_one(){ # cell in out offered n seed tag extra_args...
  local cell=$1 in=$2 out=$3 offered=$4 n=$5 seed=$6 tag=$7; shift 7
  local w="$D/w_${cell}_${tag}_${seed}.yaml"
  local m="$OUT/${cell}_${tag}_${seed}.json"
  local so="$OUT/${cell}_${tag}_${seed}.stdout"
  spec "$in" "$out" "$offered" "$n" "$seed" "$w"
  ./blis run --model "$MODEL" --workload-spec "$w" "${TOPO[@]}" \
    "$@" --seed "$seed" --metrics-path "$m" >"$so" 2>/dev/null || true
  read_metrics "$m"
}

# Gate a run and return a verdict string.
verdict(){ # achieved offered inj drop queued running
  python3 -c "
import sys
ach,off,inj,drop,q,run = float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
bad=[]
if drop: bad.append('DROPPED=%d'%drop)
if q:    bad.append('QUEUED=%d'%q)
if run:  bad.append('RUNNING=%d'%run)
if off and ach > 0.97*off: bad.append('NOT-SATURATED')
print(','.join(bad) if bad else 'ok')" "$1" "$2" "$3" "$4" "$5" "$6"; }

printf '%-14s %-10s %8s %9s %10s %8s %9s %8s %7s %s\n' \
  cell arm offered C_pred measured ratio duration Xb_mix Xb_pf gate

for cell in $CELLS; do
  IN=$(cell_in "$cell"); O=$(cell_out "$cell")

  # Predicted capacity and offered rate for each f, from mix_cap.py so the model
  # lives in exactly one place.
  GRID=$(python3 campaigns/edpp-study/mix_cap.py --emit-grid "$cell" \
           --f-values "$F_VALUES" --overload "$OVERLOAD")

  for s in $SEEDS; do
    # --- self-check: the two endpoints as real deciders -------------------
    # never must match the f=0 plan, always must match the f=1 plan. The rates
    # are queried for f=0 and f=1 explicitly rather than read out of GRID, so the
    # self-check still runs when F_VALUES covers only part of the range (e.g. a
    # re-run of the peak region).
    ENDS=$(python3 campaigns/edpp-study/mix_cap.py --emit-grid "$cell" \
             --f-values "0,1.0" --overload "$OVERLOAD")
    NEVER_C=$(awk 'NR==1{printf "%.2f", $2}' <<<"$ENDS")
    NEVER_R=$(awk 'NR==1{printf "%.1f", $3}' <<<"$ENDS")
    ALWAYS_C=$(awk 'NR==2{printf "%.2f", $2}' <<<"$ENDS")
    ALWAYS_R=$(awk 'NR==2{printf "%.1f", $3}' <<<"$ENDS")

    read -r A INJ DR Q RU DUR <<<"$(run_one "$cell" "$IN" "$O" "$NEVER_R" "$N" "$s" ref-never --pd-decider never)"
    read -r XM XP <<<"$(read_xbar "$OUT/${cell}_ref-never_${s}.stdout")"
    printf '%-14s %-10s %8s %9s %10s %8.3f %9s %8s %7s %s\n' "$cell" "ref:never" "$NEVER_R" "$NEVER_C" "$A" \
      "$(python3 -c "print($A/$NEVER_C)")" "$DUR" "$XM" "$XP" "$(verdict "$A" "$NEVER_R" "$INJ" "$DR" "$Q" "$RU")"

    read -r A INJ DR Q RU DUR <<<"$(run_one "$cell" "$IN" "$O" "$ALWAYS_R" "$N" "$s" ref-always --pd-decider always)"
    read -r XM XP <<<"$(read_xbar "$OUT/${cell}_ref-always_${s}.stdout")"
    printf '%-14s %-10s %8s %9s %10s %8.3f %9s %8s %7s %s\n' "$cell" "ref:always" "$ALWAYS_R" "$ALWAYS_C" "$A" \
      "$(python3 -c "print($A/$ALWAYS_C)")" "$DUR" "$XM" "$XP" "$(verdict "$A" "$ALWAYS_R" "$INJ" "$DR" "$Q" "$RU")"

    # --- the f sweep -------------------------------------------------------
    while read -r F CPRED OFF; do
      R=$(python3 -c "print(f'{$OFF:.1f}')")
      P="$D/plan_${cell}_f${F}_${s}.csv"
      python3 campaigns/edpp-study/make_pd_plan.py --n "$N" --f "$F" > "$P"
      read -r A INJ DR Q RU DUR <<<"$(run_one "$cell" "$IN" "$O" "$R" "$N" "$s" "f$F" --pd-plan "$P")"
      read -r XM XP <<<"$(read_xbar "$OUT/${cell}_f${F}_${s}.stdout")"
      printf '%-14s %-10s %8s %9.2f %10s %8.3f %9s %8s %7s %s\n' "$cell" "f=$F" "$R" "$CPRED" "$A" \
        "$(python3 -c "print($A/$CPRED)")" "$DUR" "$XM" "$XP" "$(verdict "$A" "$R" "$INJ" "$DR" "$Q" "$RU")"
    done <<<"$GRID"
  done
done

echo
echo "READ:"
echo "  ref:never must match the f=0 row and ref:always the f=1 row. If they do"
echo "  not, the plan mechanism is wrong and no interior row means anything."
echo "  The measured column should peak near the predicted f* (lean 0.387,"
echo "  bound 0.337). A peak above never's measured ceiling (lean 13.66, bound"
echo "  7.89) is reachable only by prefilling on the decode instances."
echo "  ratio = measured / C_pred; it must stay at or below 1, since C_pred is a"
echo "  fluid upper bound, and it understates by under 2% from the fill transient."
echo "  Any row whose gate is not 'ok' is NOT a capacity measurement."
echo DONE
