# EmpiricalDPP → llm-d feasibility study

**Date:** 2026-06-11
**Question:** Can the EmpiricalDPP (EDPP) P/D disaggregation decider — validated in BLIS
(`sim/disaggregation_edpp.go`) — be ported to a real llm-d stack, and where does it go?
**Method:** read-only source investigation of `llm-d-router` (module
`github.com/llm-d/llm-d-router`, shallow clone, 2026-06-11) against EDPP's signal needs.

## Verdict: GO — feasible, with work

EDPP is portable to llm-d, but **not as a drop-in `deciderPlugin`**. It must be a
**multi-hook plugin modeled on the existing `predictedlatency` plugin**: one instance that
implements the decision hook *and* the request-lifecycle hooks, holds its learned state, and
**maintains its own per-endpoint metrics from the request lifecycle** (because the metrics
EDPP needs are not all in llm-d's datalayer). The prefill-time signal is missing — same as
BLIS — so the TTFT proxy carries over unchanged. No blocking obstacle was found; the work is
real but bounded.

## Where it goes

- **Actuation (the disaggregate decision):** the PD-decider slot
  `llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/`, sibling to
  `prefix_based_pd_decider.go` / `always_disagg_pd_decider.go`; interface
  `deciderPlugin.disaggregate(ctx, request, endpoint) bool` (`decider_plugin.go:12-15`);
  registered in `cmd/epp/runner/runner.go`. The disagg profile handler calls it with the
  pre-selected decode endpoint (`disagg_profile_handler.go:319`).
- **Learning + aggregates (the rest of EDPP):** the `requestcontrol` framework hooks
  (`pkg/epp/framework/interface/requestcontrol/plugins.go:28-99`). One plugin object can
  register as `DataProducer` (pre-schedule), `PreRequest` (post-schedule), and
  `ResponseBodyProcessor` (completion) simultaneously — `request_control_config.go:85-108`
  type-asserts each interface and adds the same instance to every matching hook list, so
  shared state is naturally shared. **Template: the `predictedlatency` plugin**
  (`pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/`), which already
  pairs a scheduling-time scorer with a completion hook and keeps per-endpoint state
  (`prefillTokensInFlight sync.Map`) + a request-ID-keyed TTL store.

## Signal-by-signal map

| EDPP signal | When | llm-d source | Verdict |
|---|---|---|---|
| `u` = uncached input tokens | decide | `PrefixCacheMatchInfo` on endpoint (`prefix_based_pd_decider.go:119-136`) | **AVAILABLE** |
| per-pod queue depth | decide | `Endpoint.GetMetrics().WaitingQueueSize` (`datalayer/metrics.go:26-42`) | **AVAILABLE** |
| per-pod running / KV util | decide | `GetMetrics().RunningRequestsSize`, `.KVCacheUsagePercent` | **AVAILABLE** |
| prefill- vs decode-pool identity | decide | `llm-d.ai/role` labels + `filter/bylabel/roles.go` | **AVAILABLE** |
| per-pod AvgInTokens / AvgOutTokens / ITL | decide | NOT in datalayer `Metrics` (Prometheus system-level only) | **MISSING → plugin must accumulate** |
| all decode + all prefill endpoints at decide time | decide | decider gets ONE endpoint (`decider_plugin.go:14`); full set only pre-schedule in `DataProducer.Produce(ctx,request,endpoints)` and director (`director.go:284-305`) | **AVAILABLE-WITH-WORK** |
| TTFT (observed) | completion | `FirstTokenTimestamp − RequestReceivedTimestamp` (`handlers/server.go:115-117`) | **AVAILABLE** |
| mean ITL / TPOT (observed) | completion | computed at `EndOfStream` (`predictedlatency/requestcontrol_hooks.go:173`) | **AVAILABLE** |
| input/output token counts, serving endpoint | completion | `Usage.PromptTokens/CompletionTokens`, `TargetPod` (`server.go:109-119`) | **AVAILABLE** |
| **per-request prefill-time** | completion | no explicit signal in the response path | **MISSING → use TTFT proxy (as in BLIS)** |
| action (REMOTE/LOCAL) + stashed `u` carried decide→completion | both | `request.PutAttribute/GetAttribute` (`scheduling/attributes.go:21-71`) or request-ID store | **AVAILABLE** |
| mutable learned state (rate_P/D, p, ttft, Z) across requests | all | singleton plugin instance + `sync.Map`/TTL store (`predictedlatency/plugin.go:69-117`) | **AVAILABLE** |

## Recommended architecture (one plugin, three hooks)

Mirror `predictedlatency`. A single `EmpiricalDPPPlugin` instance:

1. **`DataProducer.Produce(ctx, request, endpoints)`** — pre-schedule, has the candidate set.
   Compute the drift aggregates `Q_D`, `Q_P` from the plugin's per-endpoint state (queue depth
   from live metrics × the plugin's *own* learned per-endpoint avg-tokens/ITL/rate), tag pods by
   role, and stash the result + current learned scalars on the request via `PutAttribute`.
2. **`deciderPlugin.disaggregate(ctx, request, endpoint)`** — read the stashed aggregates + `u`
   (from `PrefixCacheMatchInfo`) + learned `rate_D/rate_P/Δp/ΔTTFT/Z`, evaluate the rule
   `u·(Q_D·rate_D − Q_P·rate_P) + V·Δp > Z·ΔTTFT`, apply ε-exploration, stash the chosen action.
3. **`ResponseBodyProcessor.ResponseBody(...)` at `EndOfStream`** — observe TTFT (and TTFT-proxy
   prefill-time), mean ITL, tokens, serving endpoint; update the per-action EWMAs
   (rate_P/rate_D, p_local/remote, ttft_local/remote), update `Z`, and refresh the per-endpoint
   running averages that feed step 1.

This keeps the BLIS decider logic intact; the porting work is **the plumbing that BLIS got for
free from `RoutingSnapshot`** — in llm-d the plugin must build that per-endpoint view itself.

## Work items / risks

- **Per-endpoint metric accumulation (main effort).** AvgIn/AvgOut/ITL are not in the datalayer,
  so the plugin maintains them as per-endpoint EWMAs from completions (same mechanism EDPP
  already uses for its global learned state — extend to per-endpoint, keyed by `NamespacedName`).
- **Decision-time aggregate access — RESOLVED.** `DataProducer.Produce` receives the **full
  cross-pool candidate set** (all prefill+decode pods), before per-profile filtering:
  `director.go:284` locates all `endpointCandidates`, `:292` snapshots them, `:294` runs the
  DataProducer plugins on that full set. So `Q_D`/`Q_P` can be computed there from live per-pod
  queue depth × the plugin's per-endpoint learned EWMAs, and stashed on the request.
- **Prefill-time = TTFT proxy.** No direct prefill-duration signal at completion — identical to
  BLIS. Documented limitation; revisit if vLLM exposes a per-request prefill metric later.
- **No upstream interface change required** for a first version (use DataProducer + attributes).
  A cleaner long-term option is extending `deciderPlugin` to receive pool-wide endpoints, but that
  is an upstream PR, not needed to start.

## Bottom line

EDPP can run on a real llm-d stack as a `predictedlatency`-style multi-hook plugin. The algorithm
(rule, ε-exploration, two-population EWMAs, Z) ports unchanged; the work is reconstructing the
per-endpoint snapshot view and wiring the three lifecycle hooks. The one fidelity caveat —
prefill-time via TTFT proxy — already matches the BLIS implementation. Next step (separate, when a
cluster is available): an implementation plan for `EmpiricalDPPPlugin` + EPP config, validated
first against the `predictedlatency` plugin's patterns.

## References
- BLIS decider: `sim/disaggregation_edpp.go`; campaign result: `[[edpp-campaign-operating-point]]`.
- llm-d-router clone (read-only) at `~/git-repos/llm-git-repos/llm-d-router`, commit per shallow
  clone 2026-06-11. File:line citations above are from that tree and may shift on update.
