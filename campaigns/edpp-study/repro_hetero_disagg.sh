#!/usr/bin/env bash
# The one place PD routing may still have a case: heterogeneous hardware.
#
# WHY. On homogeneous cells the aggregated fleet beats dpvar@1P2D on throughput
# AND on all three SLO dimensions including ITL, so the interference argument for
# disaggregation does not hold there. Hetero is different for a structural
# reason: the crippled A100 has alpha_D = 69.31 ms against a 42.74 ms ITL target,
# so its largest ITL-feasible decode batch is ZERO. It cannot meet ITL on any
# request it decodes, at any batch size.
#
# An aggregated fleet has no way to express that. queue-depth balancing feeds the
# slow box regardless, which is why never@3M measured only 109.10 req/s against
# an analytic 324.76 (34%). Disaggregation provides the missing mechanism: send
# the slow instance prefill work, where no ITL constraint applies.
#
# The analytic capacity still favours aggregation (287 vs 252 with an ORACLE
# router). The hypothesis is that a REAL aggregated router cannot reach 287, and
# that PD routing beats what it actually achieves.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/out/hetero_disagg; OUT="$D/out"; mkdir -p "$OUT"
CAP=256; ARRIVAL_S="${ARRIVAL_S:-300}"; SEEDS="${SEEDS:-42}"
T=53.7; I=25.04; E=1635.8      # re-derived on realistic A100 hardware
[[ -x ./blis ]] || go build -o blis main.go

cat > "$D/bundle.yaml" <<'YAML'
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 624.0,  bw_peak_tbs: 2.039, mfu_prefill: 0.5, mfu_decode: 0.5}
coeffs_by_gpu:
  H100: scripts/calibration/coeffs-llama70b-h100-tp4.json
  A100: scripts/calibration/coeffs-llama70b-a100real-tp4.json
YAML

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
print('%.3f %.3f %.1f %.3f %.3f %.3f'%(d['responses_per_sec'],p['slo_attainment'],p['ttft_p99_ms'],
  bd.get('ttft',0),bd.get('itl',0),bd.get('e2e',0)))
" "$1" 2>/dev/null||echo "0 0 0 0 0 0"; }

printf '%-14s %6s %9s %8s %11s %7s %7s %7s\n' arm rate achieved slo ttft_p99 d_ttft d_itl d_e2e
for R in 170 220 270; do
  for s in $SEEDS; do
    W="$D/w_${R}_${s}.yaml"; N=$(python3 -c "print(int($R*$ARRIVAL_S))")
    spec 256 64 "$R" "$N" "$s" "$W"
    SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
    EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
    for ARM in never always; do
      M="$OUT/${R}_${ARM}_${s}.json"
      ./blis run --model "$MODEL" --workload-spec "$W" --num-instances 3 \
        --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" \
        --max-num-running-reqs "$CAP" --policy-config "$D/bundle.yaml" "${SLO[@]}" \
        --pd-decider $ARM --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
      printf '%-14s %6s %9s %8s %11s %7s %7s %7s\n' "$ARM@1P2D" "$R" $(jg "$M")
    done
    M="$OUT/${R}_dpvar1p2d_${s}.json"
    ./blis run --model "$MODEL" --workload-spec "$W" --num-instances 3 \
      --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" \
      --max-num-running-reqs "$CAP" --policy-config "$D/bundle.yaml" "${SLO[@]}" \
      --pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint \
      --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 \
      --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms" \
      --seed "$s" --metrics-path "$M" >/dev/null 2>&1 || true
    printf '%-14s %6s %9s %8s %11s %7s %7s %7s\n' "dpvar@1P2D" "$R" $(jg "$M")
  done
done
echo; echo "never@3M plateau previously measured at 109.10 (analytic 324.76, ratio 0.34)."
echo "If dpvar@1P2D beats never@3M here on goodput, PD routing's case is heterogeneity."
echo DONE
