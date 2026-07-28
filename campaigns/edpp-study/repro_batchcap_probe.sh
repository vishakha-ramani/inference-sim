#!/usr/bin/env bash
# Runs down the decode-cell capacity gap: is the binder the batch cap
# (--max-num-running-reqs) rather than work conservation?
#
# Analytic prediction (frozen H100 coeffs, 256 in / 512 out, |D| = 2):
#   T_iter(B) = alpha_D + B * (C0 + C1 * Lbar),  Lbar = 256 + out/2 = 512
#             = 16613.5 + B * 29.72   us
#   per-instance req/s = B / (out * T_iter(B))
#   fleet decode capacity = |D| * that
#     B=16  -> 1.829 * 2 = 3.657 req/s     (v2.1 harnesses)
#     B=64  -> 1.700 * 2 = ... see table   (alpha amortizes)
#     B=256 -> 3.216 * 2 = 6.433 req/s     (simulator default)
# If the cap is the binder, achieved throughput must plateau at these values and
# ITL must stay pinned near alpha_D at B=16 but grow with B at higher caps.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="meta-llama/llama-3.3-70b-instruct"
D=campaigns/edpp-study/out/batchcap; OUT=$D/out; mkdir -p "$OUT"

jget(){ python3 -c "import json;m=json.load(open('$1'));print(m.get('$2',0))" 2>/dev/null||echo 0; }

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

NOSLO=(--slo-ttft "standard=9999s" --slo-itl "standard=9999s" --slo-e2e "standard=9999s")

echo "cap rate arm achieved offered_ratio itl_mean_ms itl_p99_ms ttft_p99_ms e2e_p99_ms"
for CAP in 16 64 256; do
  # Sweep well past the predicted plateau for each cap so saturation is visible.
  case $CAP in
    16)  RATES="2.0 3.0 3.5 4.0 6.0" ;;
    64)  RATES="3.0 5.0 7.0 9.0 12.0" ;;
    256) RATES="4.0 6.0 8.0 10.0 14.0" ;;
  esac
  for R in $RATES; do
    for ARM in always never; do
      W="$D/w_${CAP}_${R}_${ARM}.yaml"
      # 300 s of steady state or 600 requests, whichever is larger, so the
      # achieved/offered ratio is not dominated by the ramp.
      N=$(python3 -c "print(max(600, int($R*300)))")
      spec 256 512 "$R" "$N" 42 "$W"
      M="$OUT/c${CAP}_r${R}_${ARM}.json"
      ./blis run --model "$MODEL" --workload-spec "$W" \
        --num-instances 3 --prefill-instances 1 --decode-instances 2 \
        --decode-routing-scorers "queue-depth:1" --max-num-running-reqs "$CAP" \
        "${NOSLO[@]}" --pd-decider "$ARM" --seed 42 --metrics-path "$M" >/dev/null 2>&1 || true
      ach=$(jget "$M" responses_per_sec)
      printf "%s %s %s %s %s %s %s %s %s\n" "$CAP" "$R" "$ARM" "$ach" \
        "$(python3 -c "print(f'{$ach/$R:.3f}')" 2>/dev/null || echo NA)" \
        "$(jget "$M" itl_mean_ms)" "$(jget "$M" itl_p99_ms)" \
        "$(jget "$M" ttft_p99_ms)" "$(jget "$M" e2e_p99_ms)"
    done
  done
done
echo DONE
