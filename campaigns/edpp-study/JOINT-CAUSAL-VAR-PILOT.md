# Joint corrected-causal-VaR pilot

## Policy under test

The joint policy evaluates every feasible action for an arriving request.
With (P) prefill instances and (D) decode instances, the action set has
(D(P+1)) members. Each decode instance contributes one local action and one
remote action for every prefill instance.

Every action uses the corrected deployable causal-VaR calculation. The
calculation uses candidate-specific cache state and exact marginal prefill work
over the chunks that overlap each co-resident. The policy selects the minimum
VaR action. The existing decode and prefill scorers order ties within a fixed
(10^{-9}) tolerance. A remaining local-versus-remote tie resolves to local.

The matched decomposed control uses the same estimator, coefficients, snapshot,
and tie rule. The existing decode scorer first fixes the decode destination.
Corrected causal VaR then selects among the local action and every remote
prefill action for that decode instance.

## Pilot protocol

The pilot uses four paired seeds. It covers 1P3D, 2P2D, and 3P1D fleets,
synthetic, RAG, and shared-prefix traffic, and two offered loads. The offered
rates come from the earlier 1P2D campaign. They are not normalized to each
topology's capacity. The 3P1D cells are therefore a structural one-decode
control, not a cross-topology performance comparison.

The primary endpoint is paired goodput uplift from joint selection. The trace
also computes router-induced predicted VaR regret. For each request, this is
the minimum VaR on the scorer-selected decode instance minus the minimum VaR
over the full action set.

## Four-seed result

| topology | workload | load | joint uplift | 95% paired interval | W/T/L | requests with positive router regret |
|---|---|---|---:|---:|---:|---:|
| 1P3D | synthetic | medium | 0.000 | [0.000, 0.000] | 0/4/0 | 0.0% |
| 1P3D | synthetic | near-high | 0.000 | [0.000, 0.000] | 0/4/0 | 0.0% |
| 1P3D | RAG | medium | +0.0195 | [+0.0022, +0.0368] | 4/0/0 | 60.7% |
| 1P3D | RAG | near-high | +0.0395 | [+0.0055, +0.0736] | 4/0/0 | 60.5% |
| 1P3D | shared | medium | 0.000 | [0.000, 0.000] | 0/4/0 | 55.6% |
| 1P3D | shared | near-high | +0.0063 | [+0.0023, +0.0102] | 4/0/0 | 53.8% |
| 2P2D | synthetic | medium | 0.000 | [0.000, 0.000] | 0/4/0 | 0.0% |
| 2P2D | synthetic | near-high | 0.000 | [0.000, 0.000] | 0/4/0 | 0.0% |
| 2P2D | RAG | medium | +0.0765 | [-0.1229, +0.2759] | 3/0/1 | 48.0% |
| 2P2D | RAG | near-high | +0.0877 | [-0.0504, +0.2259] | 3/0/1 | 47.0% |
| 2P2D | shared | medium | -0.0219 | [-0.2257, +0.1820] | 3/0/1 | 42.3% |
| 2P2D | shared | near-high | +0.1004 | [+0.0594, +0.1414] | 4/0/0 | 44.7% |
| 3P1D | all tested cells | both | 0.000 | [0.000, 0.000] | 0/24/0 | 0.0% |

Synthetic traffic has zero predicted VaR in these cells. Both policies follow
the same scorer tie path and reproduce each other exactly. The 3P1D fleet has
one decode destination. Joint decode selection is therefore identical to the
decomposed control by construction.

The pilot supports the mechanism in 1P3D and in the 2P2D near-high
shared-prefix cell. The two 2P2D RAG means are positive but unresolved with
four seeds. The 2P2D shared-medium mean is negative and unresolved. The next
experiment must calibrate medium and near-high load separately for each
topology before making a topology-general performance claim.

All 144 runs conserved requests and drained their queues. Unservable requests
remain zero-good outcomes in the denominator. The complete machine-readable
result is `out/joint_causal_var_v2/joint_routing_pilot_result.json`.
