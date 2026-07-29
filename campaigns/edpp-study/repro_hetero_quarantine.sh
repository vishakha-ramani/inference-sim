#!/usr/bin/env bash
# Does adaptive routing earn its keep on heterogeneous hardware, against a static
# plan that can also quarantine the slow accelerator?
#
# WHY THIS IS THE DECIDING EXPERIMENT
# ----------------------------------
# On homogeneous hardware a single fixed share beats the adaptive rule by 3.4x on
# worst-case regret across four archetypes. The rule's remaining case is hardware
# heterogeneity, which is also the reason the paper gives for taking a
# disaggregated deployment as given.
#
# The trap to avoid is comparing the rule only against shares. A --pd-plan names
# the DECODE INSTANCE per request, not just a share, so a static plan can steer
# decode away from a slow accelerator. That is the same quarantine mechanism the
# adaptive rule is supposed to provide. If a static plan captures it, adaptivity
# buys nothing here either, and we should say so.
#
# THE FLEET
# ---------
# 1P2D on the realistic mixed bundle. Measured, not assumed, by reading
# per-instance ITL from a probe run:
#
#   instance_0   fast H100, prefill-only
#   instance_1   fast H100, mixed        ITL 17.48 ms
#   instance_2   SLOW A100, mixed        ITL 27.20 ms
#
# The A100 coefficients come from real hardware specs (624 TFLOPS, 2.039 TB/s),
# giving alpha_D = 25.56 ms against the H100's 16.61 ms, a factor of 1.54. An
# earlier bundle used a fabricated 0.7 TB/s and is not used here.
#
# WHY THE SLOW INSTANCE IS SPECIAL
# --------------------------------
# The derived ITL target is 25.04 ms and the slow instance's per-iteration floor
# is 25.56 ms. It therefore cannot meet the inter-token target on a single request
# at any batch size. Every request it decodes misses. This is a property of the
# hardware against the SLO, not a congestion effect, so no amount of headroom
# removes it.
#
# The margin is 2 percent. It survives with honest coefficients and it is a knife
# edge. A different fleet mix moves the target, because the target is the
# reference fleet's mean ITL, and could erase the effect entirely.
#
# WHAT MAKES THIS LOAD-DEPENDENT
# ------------------------------
# Steering decode off the slow instance costs capacity. Fluid capacities on this
# fleet:
#
#   all split, decode shared across both mixed   273.7 req/s
#   collocate on the two mixed                   236.6
#   decode on the fast instance only             185.5
#
# So quarantine gives up 32 percent of capacity to gain ITL feasibility. Below
# about 185 req/s that should win. Above it, quarantine saturates and the
# arrangements that use the slow instance for decode take over, buying capacity at
# the cost of ITL misses. If the best arrangement moves across that boundary, no
# fixed plan is safe and adaptivity has something a constant cannot do.
#
# THE ARMS
# --------
# Every plan holds phi = 1, so prefill always runs on instance_0 and the DECODE
# assignment is the only knob. psi is the share of decodes sent to the slow
# instance.
#
#   plan psi=0.00   quarantine. The slow instance never decodes.
#   plan psi=0.10   mostly quarantined, a tenth of decode load on the slow box.
#   plan psi=0.25
#   plan psi=0.50   even split, which is what `always` does.
#   always          the decider. Must match psi=0.50, which is the self-check.
#   dpvar           the full rule.
#
# RATES
# -----
# 120, 180, 240 req/s, bracketing the 185.5 quarantine capacity from both sides.
# All three sit below `always`'s 273.7, so `always` is never saturated by
# construction and the comparison stays honest.
#
# Usage:
#   campaigns/edpp-study/repro_hetero_quarantine.sh
#   RATES="120 180" SEEDS=42 campaigns/edpp-study/repro_hetero_quarantine.sh
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
BUNDLE="${BUNDLE:-campaigns/edpp-study/out/hetero_real_cmp/bundle.yaml}"
D=campaigns/edpp-study/out/${RUN:-hetero_quarantine}; OUT="$D/out"; mkdir -p "$OUT"

IN=256; O=64                       # the hetero cell of specs/grid_v3/cells.txt
T=53.7; I=25.04; E=1635.8          # targets re-derived on the realistic A100 fleet
ARRIVAL_S="${ARRIVAL_S:-120}"
RATES="${RATES:-120 180 240}"
SEEDS="${SEEDS:-42 7}"
PSIS="${PSIS:-0 0.10 0.25 0.50}"
[[ -x ./blis ]] || go build -o blis main.go
[[ -f "$BUNDLE" ]] || { echo "missing bundle $BUNDLE"; exit 1; }

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
      --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 256
      --policy-config "$BUNDLE")
SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware
    --edpp-tau-itl "${I}ms")
VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint
     --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1
     --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")

# Per-instance decode counts say where the arm actually sent its decodes, which is
# what the whole experiment is about. instance_2 is the slow one.
row(){ python3 -c "
import json,sys,re
m=json.load(open(sys.argv[1])); p=m['per_class']['standard']
dim=p.get('slo_attainment_by_dim',{})
txt=open(sys.argv[2]).read()
per={}
for mm in re.finditer(r'\{[^{}]*\"instance_id\"[^{}]*\}', txt, re.S):
    try: d=json.loads(mm.group(0))
    except Exception: continue
    per[d.get('instance_id')]=d
dec1=per.get('instance_1',{}).get('completed_requests',0)
dec2=per.get('instance_2',{}).get('completed_requests',0)
tot=dec1+dec2
slowshare=(dec2/tot) if tot else 0.0
itl2=per.get('instance_2',{}).get('itl_mean_ms',0.0)
bad=[]
if m['dropped_unservable']: bad.append('DROPPED=%d'%m['dropped_unservable'])
if m['still_queued']: bad.append('QUEUED=%d'%m['still_queued'])
if m['still_running']: bad.append('RUNNING=%d'%m['still_running'])
e2e_s=m.get('e2e_mean_ms',0)/1000.0
expect=m['injected_requests']/(float(sys.argv[3])+e2e_s) if (float(sys.argv[3])+e2e_s) else 0
if expect and m['responses_per_sec'] < 0.95*min(float(sys.argv[4]), expect): bad.append('saturated')
print('%.2f %.4f %.4f %.4f %.4f %.1f %.4f %.2f %s'%(
  m['responses_per_sec'], p['slo_attainment'], dim.get('ttft',0), dim.get('itl',0),
  dim.get('e2e',0), p.get('ttft_p99_ms',0), slowshare, itl2,
  ','.join(bad) if bad else 'ok'))" "$1" "$2" "$3" "$4"; }

printf '%-13s %6s %9s %8s %8s %8s %8s %10s %9s %9s %s\n' \
  arm rate achieved goodput g_ttft g_itl g_e2e ttft_p99 slow_dec itl_slow gate

for R in $RATES; do
  for s in $SEEDS; do
    N=$(python3 -c "print(int($R*$ARRIVAL_S))")
    W="$D/w_${R}_${s}.yaml"; spec "$IN" "$O" "$R" "$N" "$s" "$W"
    for PSI in $PSIS; do
      P="$D/plan_${R}_${s}_psi${PSI}.csv"
      python3 campaigns/edpp-study/make_pd_plan.py --n "$N" --phi 1.0 --psi "$PSI" > "$P"
      M="$OUT/${R}_psi${PSI}_${s}.json"; SO="${M%.json}.out"
      ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
        --pd-plan "$P" --seed "$s" --metrics-path "$M" >"$SO" 2>/dev/null || true
      printf '%-13s %6s %9s %8s %8s %8s %8s %10s %9s %9s %s\n' \
        "psi=$PSI" "$R" $(row "$M" "$SO" "$ARRIVAL_S" "$R")
    done
    for arm in always dpvar; do
      case $arm in
        always) ARGS=(--pd-decider always) ;;
        dpvar)  ARGS=("${VVF[@]}")         ;;
      esac
      M="$OUT/${R}_${arm}_${s}.json"; SO="${M%.json}.out"
      ./blis run --model "$MODEL" --workload-spec "$W" "${TOPO[@]}" "${SLO[@]}" \
        "${ARGS[@]}" --seed "$s" --metrics-path "$M" >"$SO" 2>/dev/null || true
      printf '%-13s %6s %9s %8s %8s %8s %8s %10s %9s %9s %s\n' \
        "$arm" "$R" $(row "$M" "$SO" "$ARRIVAL_S" "$R")
    done
  done
done

echo
echo "READ:"
echo "  slow_dec is the share of decodes that landed on the SLOW instance, and"
echo "  itl_slow is its mean ITL against a 25.04 ms target. always must match"
echo "  psi=0.50 on both, which is the self-check."
echo "  Below 185 req/s quarantine (psi=0) should lead, because the slow instance"
echo "  cannot meet ITL at any batch size. At 240 it is over its 185.5 capacity and"
echo "  must fall behind. If the best psi MOVES across that boundary, no fixed plan"
echo "  is safe. If one psi wins everywhere, adaptivity buys nothing here either."
echo "  dpvar earns its keep only by beating the best psi at EVERY rate."
echo DONE
