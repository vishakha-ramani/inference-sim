package cluster

import "github.com/inference-sim/inference-sim/sim"

// DefaultCacheSignalDelay is the default propagation delay for prefix cache
// signals in microseconds (50ms). Only affects precise-prefix-cache and
// no-hit-lru scorers; has no observable effect on other routing policies.
// Models aggregate signal staleness from production llm-d metrics polling.
// Set to 0 for oracle mode (live cache state).
const DefaultCacheSignalDelay int64 = 50_000

// DeploymentConfig describes a cluster where all instances share identical
// hardware and model configuration. NumInstances must be >= 1.
type DeploymentConfig struct {
	sim.SimConfig // Embeds all instance-level config (horizon, seed, KV, batch, latency, policy)

	NumInstances int

	// Online routing pipeline configuration (PR4+)
	AdmissionPolicy       string  // "always-admit" (default) or "token-bucket"
	AdmissionLatency      int64   // microseconds, default 0
	RoutingLatency        int64   // microseconds, default 0
	TokenBucketCapacity   float64 // max tokens, default 10000
	TokenBucketRefillRate float64 // tokens/second, default 1000

	// Routing policy configuration (PR6, evolved in PR17)
	RoutingPolicy        string             // "round-robin" (default), "least-loaded", "weighted", "always-busiest"
	RoutingScorerConfigs []sim.ScorerConfig // for weighted routing scorer pipeline (nil = use defaults)

	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // number of counterfactual candidates, default 0
	// RecordRoutingDecisions enables per-candidate routing-decision capture for the
	// --routing-decision-trace CSV (every prefill/decode/standard target selection).
	RecordRoutingDecisions bool

	// Snapshot staleness configuration (H3 experiment, unified in #463)
	// When > 0, all Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization)
	// use Periodic refresh with this interval (microseconds). 0 = Immediate (default).
	SnapshotRefreshInterval int64

	// Cache signal propagation delay for precise prefix cache scoring (issue #919).
	// Only affects routing when precise-prefix-cache or no-hit-lru scorers are active;
	// has no observable effect on other routing policies (round-robin, least-loaded, etc.).
	// When > 0, those scorers query a periodically-refreshed stale snapshot of each
	// instance's KV cache block hash map instead of live state.
	// Models the asynchronous KV event propagation delay in production llm-d.
	// Default: DefaultCacheSignalDelay (50ms). Feeds into ObservabilityConfig.CacheBlocks.
	// 0 = oracle mode (scorers read live cache state with zero delay).
	// Units: microseconds of simulated time.
	CacheSignalDelay int64

	// Phase 1A: Node pool infrastructure (optional â empty = backward-compatible mode).
	// When non-empty, activates PlacementManager for GPU inventory tracking.
	NodePools []NodePoolConfig

	// Phase 1A: Instance lifecycle configuration (loading delay, warm-up, drain policy).
	// Zero value is safe: no loading delay, no warm-up, WAIT drain policy.
	InstanceLifecycle InstanceLifecycleConfig

	// PD disaggregation configuration (PR1)
	// When both PrefillInstances and DecodeInstances are 0, disaggregation is disabled
	// and the pipeline is unchanged (BC-PD-1).
	PrefillInstances int // Number of instances dedicated to prefill (0 = disabled)
	DecodeInstances  int // Number of instances dedicated to decode (0 = disabled)
	// SharedInstances (issue #1276, GAP-5) â number of instances serving both
	// prefill and decode (shared-role pods, llm-d "prefill-decode" / legacy "both"
	// role labels). When > 0, ValidatePoolTopology accepts
	// prefill+decode+shared <= total, and BuildPoolMembershipFromIndices
	// assigns PoolRolePrefillDecode to the last `shared` indices after
	// the prefill-only and decode-only ranges.
	SharedInstances          int
	PDDecider                string         // Disaggregation decider: "" or "never" (default), "always", "prefix-threshold", "edpp"
	PDPrefixThreshold        int            // Default non-cached token threshold for prefix-threshold decider (PR6)
	PDPrefixThresholdByClass map[string]int // Optional SLO-class overrides; absent classes use PDPrefixThreshold
	PDPlanPath               string         // Path to a fixed-plan CSV; when set, forces a FixedPlanDecider (overrides PDDecider). Counterfactual-regret harness / offline yardstick.

	// EDPP (Lyapunov drift-plus-penalty) decider knobs â used only when PDDecider == "edpp".
	// All durations are microseconds. See sim/edpp.go and the design doc for semantics.
	EDPPTauTTFTUs                       int64            // default Ï_ttft: time-average TTFT SLO target (Âµs)
	EDPPTauITLUs                        int64            // default Ï_itl: time-average ITL SLO target (Âµs)
	EDPPTauRefUs                        int64            // fixed reference Ï for the transfer-penalty normalization (Âµs)
	EDPPTauTTFTByClassUs                map[string]int64 // per-SLO-class Ï_ttft overrides (Âµs); nil = defaults for all
	EDPPTauITLByClassUs                 map[string]int64 // per-SLO-class Ï_itl overrides (Âµs); nil = defaults for all
	EDPPTauE2EUs                        int64            // default Ï_e2e for the VaR E2E composite (Âµs); 0 â E2E conjunct disabled. Used by EDPPRule "var" and "var-prefill".
	EDPPTauE2EByClassUs                 map[string]int64 // per-SLO-class Ï_e2e overrides (Âµs); nil = defaults for all
	EDPPV                               float64          // V: penalty/stability tradeoff knob (larger â fewer offloads)
	EDPPCXferUs                         int64            // c_xfer: KV-transfer cost paid when routing P (Âµs)
	EDPPNomPrefillTokens                int              // S_nom: nominal prefill chunk for the fixed prefill normalizer
	EDPPNomDecodeCtx                    int              // L_nom: nominal decode context for the fixed decode normalizer
	EDPPCoeffs                          sim.EDPPCoeffs   // frozen E3 latency-law coefficients; required when PDDecider == "edpp"
	EDPPTAdmEstimator                   string           // admission-delay estimator that DRIVES routing ("" â waiting); deployable-only, oracle names rejected by NewEDPPDecider
	EDPPJoint                           bool             // when true, EDPP enumerates all (decode, prefill) candidates and picks the drift-plus-penalty argmin (--edpp-joint); false â reduced fixed-d rule
	EDPPJointCausalVar                  bool             // opt-in corrected causal-VaR-only joint policy; scorer breaks VaR ties only
	EDPPDecomposedCausalVar             bool             // corrected causal VaR after the existing scorer fixes decode placement
	EDPPJointSLOExternality             bool             // constrained joint policy: projected net good plus per-instance capacity shadow prices
	EDPPDecomposedSLOExternality        bool             // same constrained score after the existing scorer fixes decode placement
	EDPPSLOExternalityNoExternality     bool             // ablation: remove the causal SLO externality term only
	EDPPSLOExternalityNoOwnGood         bool             // ablation: remove the arriving-request projected-good term only
	EDPPSLOExternalityNoCapacity        bool             // ablation: remove the capacity-shadow-price term only
	EDPPSLOExternalityOccupancyCapacity bool             // replace marginal-work/nominal-mu capacity queues with physical occupancy-time queues
	EDPPRule                            string           // EDPP reduced-path decision rule: "" / "dpp" (default) | "least-ttft" | "var" (DIAGNOSTIC ORACLE)
	EDPPVarMetric                       string           // EDPP VaR scoring kernel for EDPPRule "var"/"var-prefill": "flip" (default) | "util" | "hazard"
	EDPPVarPrefillWeight                float64          // simplified var-prefill prefill-queue stability weight λ_p (0 disables the term; CLI default 1.0)
	EDPPVarKeepCongestion               bool             // EDPP drift-plus-VaR when EDPPRule=="var": keep the congestion drift and ADD the VaR externality (instead of replacing it)
	EDPPVarCongestionWeight             float64          // EDPP drift-plus-VaR congestion weight: cost = weightÂ·congestion + VaR (0 â 1.0)
	EDPPVarNormalize                    bool             // EDPP drift-plus-VaR auto-normalization: per-decision min-max normalize congestion vs VaR so the weight is scale-free
	EDPPVarNormalizeFloorScale          float64          // EDPP normalization spread floor scale: ε₀ = scale·(dwork/W*); spreads below ε₀ are compressed out instead of amplified (0 ⇒ 1.0)
	EDPPVarDeployable                   bool             // DEPLOYABLE VaR: estimate co-resident remaining from censored N̂_out instead of the oracle true remaining (INV-9-safe)
	EDPPVarCollocPrefill                bool             // DEPLOYABLE VaR extra: also price the first-token VaR of collocated prefill occupants on the decode instance (INV-9-safe; default ON — the rule prices this externality)
	EDPPVarGoodputObjective             bool             // Add arriving-request predicted goodput: VaR−good_r for "var", or the local-vs-disagg reward difference for "var-prefill". Upper bound with EDPPOracleOutputLen; off preserves prior behavior.
	EDPPTTFTOverlapAware                bool             // Reduced-path TTFT ablation: overlap remote prefill+transfer with decode-queue drainage instead of serially adding both delays.
	EDPPVarExactPrefillOverlap          bool             // VaR ablation: price only exact marginal prefill work over the chunks that overlap each co-resident.
	EDPPPathSpecificPrefillWork         bool             // Reduced-path ablation: use each path's own prefix-cache state for prefill work, TTFT, and backlog booking.
	EDPPKairosAlpha                     float64          // Kairos paper TTFT margin α (EDPPRule=="kairos-paper"); 0 ⇒ 1.3
	EDPPKairosBeta                      float64          // Kairos TBT safety margin β (all Kairos modes); 0 ⇒ 1.0
	EDPPJointTrace                      bool             // when true (joint mode only), record the per-decision scorer-vs-joint divergence trace (--edpp-joint-trace); pure instrumentation, no routing effect
	EDPPJointCandidateTrace             bool             // record every joint candidate's causal-VaR breakdown; pure instrumentation
	EDPPOracleOutputLen                 bool             // DIAGNOSTIC / UPPER-BOUND ONLY (--edpp-oracle-output-len): charge the routed request's OWN decode work with its TRUE output length instead of NÌ_out. Violates INV-9; never deployable.
	EDPPCXferSizeAware                  bool             // --edpp-c-xfer-size-aware: EDPP computes c_xfer per request from KV size (mirrors the DES KV-transfer executor), instead of the flat EDPPCXferUs. Deployable (input-only).

	// E/P/D disaggregation configuration (GAP-4, issue #1264).
	// When EncodeInstances == 0 (default), the encode stage is disabled and the
	// pipeline is byte-identical to the pre-PR simulator (BC-EPD-1).
	EncodeInstances int    // Number of instances dedicated to encoding multimodal input (0 = disabled)
	EncodeDecider   string // Encode decider: "", "never" (default), "always", "multimodal"

	// PD KV transfer configuration (PR2)
	PDTransferBandwidthGBps float64 // Inter-instance KV transfer bandwidth in GB/s (default 25.0)
	PDTransferBaseLatencyMs float64 // Inter-instance KV transfer base latency in ms (default 0.05)
	PDTransferContention    bool    // Enable fair-share bandwidth contention model (--pd-transfer-contention, INV-P2-2)

	// Per-pool routing scorer configuration (PR2)
	// When nil, both pools use the main RoutingScorerConfigs.
	PrefillScorerConfigs []sim.ScorerConfig // Scorer configs for prefill pool routing
	DecodeScorerConfigs  []sim.ScorerConfig // Scorer configs for decode pool routing

	// Per-pool hardware overrides
	// When empty (all nil/zero), all instances use the global SimConfig (BC-P2-1).
	PrefillOverrides PoolOverrides // Hardware overrides for prefill pool instances
	DecodeOverrides  PoolOverrides // Hardware overrides for decode pool instances

	// Phase 1C: Model autoscaler pipeline (issue #692).
	// Zero value is safe: ModelAutoscalerIntervalUs=0 disables the autoscaler entirely (INV-6).
	ModelAutoscalerIntervalUs      float64   `yaml:"model_autoscaler_interval_us,omitempty"`       // tick interval in Î¼s; 0 = autoscaler disabled
	HPAScrapeDelay                 DelaySpec `yaml:"hpa_scrape_delay,omitempty"`                   // HPA scrape lag: time from WVA metric emission to HPA acting; zero = same-tick actuation; Mean/Stddev in seconds
	ScaleUpStabilizationWindowUs   float64   `yaml:"scale_up_stabilization_window_us,omitempty"`   // HPA scale-up stabilization window in Î¼s; 0 = act on first signal (HPA default)
	ScaleDownStabilizationWindowUs float64   `yaml:"scale_down_stabilization_window_us,omitempty"` // HPA scale-down stabilization window in Î¼s; 0 = no stabilization (pass immediately). Set to 300,000,000 (= 5 minutes) to match the Kubernetes HPA default.
	// AutoscalerAnalyzerConfig holds V2SaturationAnalyzer thresholds.
	// Zero values are safe: NewClusterSimulator applies WVA reference defaults
	// (KvCacheThreshold=0.8, ScaleUpThreshold=0.8, ScaleDownBoundary=0.4, AvgInputTokens=512).
	AutoscalerAnalyzerConfig V2SaturationAnalyzerConfig `yaml:"autoscaler_analyzer,omitempty"`

	// Phase 1B-1a: tier-ordered admission shedding config (issue #809).
	// TierShedMinPriority=0 rejects sheddable tiers (priority < 0) under overload.
	// Set to 3 (Standard) for Standard-and-above protection, or -3 to admit all.
	TierShedThreshold   int `yaml:"tier_shed_threshold,omitempty"`
	TierShedMinPriority int `yaml:"tier_shed_min_priority,omitempty"`

	// GAIE-legacy admission thresholds (issue #1014). Only used when AdmissionPolicy = "gaie-legacy".
	GAIEQDThreshold float64 // queue depth threshold per instance (default 5)
	GAIEKVThreshold float64 // KV cache utilization threshold (default 0.8)

	// Phase 1B-2a: per-tenant fair-share budgets (issue #811).
	// Key: TenantID string. Value: fraction of total cluster capacity (0.0â1.0).
	// Zero value is safe: nil = no enforcement (all tenants unlimited).
	TenantBudgets map[string]float64 `yaml:"tenant_budgets,omitempty"`

	// Flow control configuration (issue #882, GIE parity).
	// When FlowControlEnabled is false (default), the gateway queue is bypassed
	// and requests flow directly from admission to routing (BC-1 pass-through).
	FlowControlEnabled              bool             `yaml:"flow_control_enabled,omitempty"`
	FlowControlDetector             string           `yaml:"flow_control_detector,omitempty"`                // "never" (default), "utilization", "concurrency"
	FlowControlDispatchOrder        string           `yaml:"flow_control_dispatch_order,omitempty"`          // "fifo" (default), "priority", "slo-deadline"
	FlowControlSLOTargets           map[string]int64 `yaml:"flow_control_slo_targets,omitempty"`             // SLO class â TTFT target Âµs for slo-deadline ordering
	FlowControlMaxQueueDepth        int              `yaml:"flow_control_max_queue_depth,omitempty"`         // 0 = unlimited
	FlowControlQueueDepthThreshold  float64          `yaml:"flow_control_queue_depth_threshold,omitempty"`   // for utilization detector
	FlowControlKVCacheUtilThreshold float64          `yaml:"flow_control_kv_cache_util_threshold,omitempty"` // for utilization detector
	FlowControlMaxConcurrency       int              `yaml:"flow_control_max_concurrency,omitempty"`         // for concurrency detector
	FlowControlPerBandCapacity      int              `yaml:"flow_control_per_band_capacity,omitempty"`       // 0 = unlimited; max requests per priority band
	FlowControlUsageLimitThreshold  float64          `yaml:"flow_control_usage_limit_threshold,omitempty"`   // per-band HoL blocking ceiling (1.0=no HoL, <1.0 gates lower bands earlier)
	FlowControlFairnessPolicy       string           `yaml:"flow_control_fairness_policy,omitempty"`         // "global-strict" (default), "round-robin"
	FlowControlRequestTTL           int64            `yaml:"flow_control_request_ttl,omitempty"`             // microseconds; 0 = disabled (default). GIE parity: DefaultRequestTTL.
	FlowControlQueueShedding        bool             `yaml:"flow_control_queue_shedding,omitempty"`          // BLIS-extra: cross-band shedding on full queue (not in llm-d). Default false.
	FlowControlDispatchTickInterval int64            `yaml:"flow_control_dispatch_tick_interval,omitempty"`  // Âµs between periodic dispatch ticks (default 1000 = 1ms, llm-d parity). 0 = use default.
	FlowControlInFlightEviction     bool             `yaml:"flow_control_in_flight_eviction,omitempty"`      // BLIS-extra: evict sheddable in-flight requests when saturated (not in llm-d). Default false.

	// Issue #893: per-GPU-type hardware calibration for roofline and trained-physics backends.
	// Key: GPU type string (e.g., "A100", "H100"). Value: HardwareCalib for that GPU.
	// When non-nil and a pool's gpu_type is found in the map, the matched HardwareCalib
	// overrides simCfg.HWConfig at instance construction time (both sync and deferred paths),
	// ensuring pool-placed instances use the correct roofline hardware coefficients
	// (TFlopsPeak, BwPeakTBs) rather than the CLI --gpu calibration.
	// Zero value (nil) is safe: no override, backward-compatible with all existing callers.
	HWConfigByGPU map[string]sim.HardwareCalib `yaml:"hw_config_by_gpu,omitempty"`

	// per-GPU-type Î¸_i for the EDPP decider; nil = homogeneous
	EDPPCoeffsByGPU map[string]sim.EDPPCoeffs `yaml:"edpp_coeffs_by_gpu,omitempty"`
}

// ToSimConfig returns the embedded SimConfig for per-instance construction.
// WorkloadConfig is an empty struct: cluster mode generates workload centrally
// and injects requests via InjectRequestOnline.
func (d DeploymentConfig) ToSimConfig() sim.SimConfig {
	return d.SimConfig
}

// EffectivePrefillTP returns the tensor parallelism degree used by the prefill pool.
// Used for KV transfer sizing in both NewClusterSimulator (upfront validation) and
// KVTransferStartedEvent.Execute (runtime). Note: resolveConfigForRole independently
// applies PrefillOverrides via ResolvePoolConfig.
func (d DeploymentConfig) EffectivePrefillTP() int {
	if d.PrefillOverrides.TP != nil {
		return *d.PrefillOverrides.TP
	}
	return d.TP
}

// resolveConfigForRole returns the SimConfig appropriate for an instance in the given pool role.
// For PoolRolePrefill: applies PrefillOverrides to the global SimConfig.
// For PoolRoleDecode: applies DecodeOverrides to the global SimConfig.
// For PoolRolePrefillDecode (shared-role): applies DecodeOverrides â "decode wins"
// precedence per issue #1276 D-2 (matches the spirit of llm-d's allowsNoLabel=true
// decode default; a shared pod is decode-capable in every deployment, and picking
// decode avoids oversizing TP for pods that also do prefill).
// For any other role (including 0/unset): returns the global SimConfig unchanged.
// The global SimConfig is never mutated.
func (d DeploymentConfig) resolveConfigForRole(role PoolRole) sim.SimConfig {
	switch role {
	case PoolRolePrefillDecode:
		return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)
	case PoolRolePrefill:
		return ResolvePoolConfig(d.SimConfig, d.PrefillOverrides)
	case PoolRoleDecode:
		return ResolvePoolConfig(d.SimConfig, d.DecodeOverrides)
	case PoolRoleEncode:
		// Encode pool uses the global SimConfig in this PR; per-pool
		// overrides are a follow-up (GAP-4 design doc D6).
		return d.SimConfig
	default:
		return d.SimConfig
	}
}
