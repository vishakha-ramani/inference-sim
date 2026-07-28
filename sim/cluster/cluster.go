package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"reflect"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/sirupsen/logrus"
)

// ClusterSimulator orchestrates N InstanceSimulator replicas behind a shared clock.
// Events from all instances are processed in global timestamp order;
// ties are broken by lowest instance index for determinism.
type ClusterSimulator struct {
	config            DeploymentConfig
	instances         []*InstanceSimulator
	rng               *sim.PartitionedRNG
	clock             int64
	hasRun            bool
	aggregatedMetrics *sim.Metrics

	// Online routing pipeline fields
	clusterEvents     ClusterEventQueue
	seqCounter        int64
	admissionLatency  int64
	routingLatency    int64
	admissionPolicy   sim.AdmissionPolicy
	priorityMap       *sim.SLOPriorityMap
	snapshotProvider  *CachedSnapshotProvider
	routingPolicy     sim.RoutingPolicy
	rejectedRequests  int            // EC-2: count of requests rejected by admission policy
	routingRejections int            // I13: count of requests rejected at routing (no routable instances)
	shedByTier        map[string]int // per-SLOClass shedding: admission rejections + gateway queue shed + in-flight evictions
	// injectedByClass: per-SLOClass arrival counter. Incremented in ClusterArrivalEvent.Execute
	// before any drop/route/admission decision. Goodput denominator (issue #1409, BC-5).
	injectedByClass       map[string]int64
	trace                 *trace.SimulationTrace    // nil when trace-level is "none" (BC-1: zero overhead)
	preGeneratedRequests  []*sim.Request            // Pre-generated requests (all workload paths unified)
	inFlightRequests      map[string]int            // instance ID → dispatched-but-not-completed count (#463)
	evictionTracker       *EvictionTracker          // tracks routed sheddable requests for in-flight eviction (nil unless --in-flight-eviction set)
	gatewayEvicted        int                       // count of requests evicted in-flight from instances (INV-1: gw_evicted)
	gatewayExpired        int                       // count of requests expired from gateway queue via TTL (INV-1: gw_expired)
	requestTTL            int64                     // gateway queue request TTL in microseconds; 0 = disabled
	dispatchTickInterval  int64                     // µs between periodic dispatch ticks (default 1000 = 1ms, llm-d parity)
	dispatchTickPending   bool                      // true when a GatewayDispatchTickEvent is already scheduled
	poolMembership        map[string]PoolRole       // instance ID → pool role (nil when disaggregation disabled)
	disaggregationDecider sim.DisaggregationDecider // PD disaggregation decider (nil when disabled)
	sloFeedback           sim.SLOFeedbackDecider    // non-nil when the decider consumes realized-SLO feedback (EDPP)

	// PD disaggregation state (PR2)
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions map[string]string         // prefill sub-req ID → parent ID
	pendingDecodeCompletions  map[string]string         // decode sub-req ID → parent ID
	localAdmitTimes           map[string]int64          // request ID → first local-admission tick (non-disagg); populated only when recordPDOutcomes
	recordPDOutcomes          bool                      // gate: capture per-request admission times for --pd-outcome-trace

	// Stage B work trace: per-request realized-vs-closed work model CSV (--edpp-work-trace).
	recordWorkTrace bool
	workCoeffs      sim.EDPPCoeffs
	workByInstance  map[string]map[string]sim.ReqWork // instanceID → reqID → work

	// Stage C admission trace: per-request realized-vs-predicted admission delay CSV
	// (--edpp-admission-trace). recordAdmissionTrace gates capture (zero-cost when off).
	// admissionCtx keys parent/local request ID → the per-pool AdmissionContext(s) the
	// EDPP decider assembled at decision time. localEnqueueTimes captures the local
	// (non-disaggregated) enqueue instant so local_t_adm = local_schedule − local_enqueue.
	recordAdmissionTrace bool
	admissionCoeffs      sim.EDPPCoeffs
	admissionCtx         map[string]*capturedAdmission
	localEnqueueTimes    map[string]int64
	// transfersInitiated counts every request that reaches
	// KVTransferStartedEvent (issue #1343: both successful reservations
	// that enter the transfer pipeline AND drop-at-start cases where the
	// decode pod was unroutable or reservation failed — dropAtStart also
	// bumps this counter). This preserves the pre-#1343 "attempts"
	// semantics so INV-PD-3 (initiated == completed) still holds: every
	// drop-at-start also schedules a zero-duration
	// KVTransferCompletedEvent that increments transfersCompleted.
	transfersInitiated int
	// transfersCompleted counts every KVTransferCompletedEvent that fires —
	// successful promotions, late drops (decode pod became non-routable
	// mid-transfer), and degenerate completions scheduled by dropAtStart.
	transfersCompleted      int
	pdPrefillCompletedCount int               // prefill sub-requests that completed (for INV-1 correction)
	pdDecodeCompletedCount  int               // decode sub-requests that completed (for INV-1 in-flight tracking)
	pdDecodeTimedOutCount   int               // decode sub-requests that timed out (for INV-1 in-flight tracking)
	droppedAtDecodeKV       int               // requests dropped due to insufficient KV at decode
	prefillRoutingPolicy    sim.RoutingPolicy // nil = use main routingPolicy
	decodeRoutingPolicy     sim.RoutingPolicy // nil = use main routingPolicy

	// E/P/D disaggregation state (GAP-4, issue #1264).
	// encodeDecider is nil when --encode-instances == 0, which disables the encode stage.
	encodeDecider           sim.EncodeDecider
	encodeRoutingRejections int // INV-1 term: requests rejected at encode routing (empty encode pool)

	// Transfer contention state (--pd-transfer-contention flag, INV-P2-2)
	activeTransfers                int
	peakConcurrentTransfers        int
	transferDepthSum               int64
	transferStartCount             int64
	contentionBookkeepingCorrupted bool

	// Phase 1A: node/GPU placement manager. Nil when NodePools is empty (backward-compat).
	placement *PlacementManager

	// Phase 1B-2a: per-tenant fair-share tracker. Nil when TenantBudgets is nil (backward-compat).
	tenantTracker *TenantTracker

	// Phase 1C: model autoscaler pipeline. Nil when ModelAutoscalerIntervalUs == 0 (backward-compat, INV-6).
	autoscaler      *autoscalerPipeline
	pendingArrivals int // count of ClusterArrivalEvents not yet executed; used by scheduleNextTick to stop ticking when all work is done

	// sessionCallback is the raw onRequestDone parameter for session follow-up
	// generation in PD mode. Called from detectDecodeCompletions with the original
	// request (which carries SessionID). Separate from the per-instance closure to
	// avoid double-notifying tenantTracker (issue #884). Nil for non-session workloads.
	sessionCallback func(*sim.Request, int64) []*sim.Request

	// cacheQueryFn maps instance IDs to KV cache query functions for precise
	// prefix cache scoring. Built after instance construction; deferred instances
	// are added in NodeReadyEvent.Execute. Nil when no instances exist yet.
	cacheQueryFn map[string]func([]sim.TokenID) int

	// Cache block staleness is managed by CachedSnapshotProvider via
	// ObservabilityConfig.CacheBlocks (unified in #1060).

	// Flow control state (issue #882, GIE parity).
	// When flowControlEnabled is false, these fields are nil/zero (BC-1 pass-through).
	flowControlEnabled   bool
	saturationDetector   sim.SaturationDetector
	gatewayQueue         *GatewayQueue
	flowControlAdmission *FlowControlAdmission // typed ref for outcome inspection + completion dispatch; nil when flow control disabled

	progressHook               sim.ProgressHook
	simClockProgressIntervalUs int64
	nextSnapshotClockUs        int64
}

// effectiveAnalyzerConfig applies WVA reference defaults to zero-valued fields.
// Zero values mean "not configured by caller" — fill with defaults so callers
// only need to set ModelAutoscalerIntervalUs to enable the autoscaler.
func effectiveAnalyzerConfig(cfg V2SaturationAnalyzerConfig) V2SaturationAnalyzerConfig {
	if cfg.KvCacheThreshold == 0 {
		cfg.KvCacheThreshold = 0.8
	}
	if cfg.ScaleUpThreshold == 0 {
		cfg.ScaleUpThreshold = 0.8
	}
	if cfg.ScaleDownBoundary == 0 {
		cfg.ScaleDownBoundary = 0.4
	}
	if cfg.AvgInputTokens == 0 {
		cfg.AvgInputTokens = 512
	}
	return cfg
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// All workload generation now happens externally — requests are passed in directly.
// onRequestDone is an optional callback invoked when a request reaches a terminal state
// (completed, length-capped, timed out, or dropped). The callback returns follow-up
// requests which are routed through the cluster pipeline (not injected locally).
// Pass nil for non-session workloads.
// Panics if config.NumInstances < 1.
func NewClusterSimulator(config DeploymentConfig, requests []*sim.Request, onRequestDone func(*sim.Request, int64) []*sim.Request) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
	}

	// Validate pool topology and overrides early (before instance construction).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 || config.EncodeInstances > 0 {
		if err := ValidatePoolTopology(config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.EncodeInstances, config.NumInstances); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.PrefillOverrides.Validate("prefill pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.DecodeOverrides.Validate("decode pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
	}

	// Validate KV bytes per token derivation early so KVTransferStartedEvent never
	// encounters a configuration error at runtime (the panic there is now unreachable).
	// Covers pure-shared clusters too (issue #1276): a shared-role pod can perform
	// prefill and therefore source a KV transfer.
	if config.PrefillInstances > 0 || config.SharedInstances > 0 {
		if config.EffectivePrefillTP() <= 0 {
			panic("ClusterSimulator: PD disaggregation requires prefill TP > 0 (set --tp or --prefill-tp)")
		}
		if _, err := latency.KVBytesPerToken(config.ModelConfig, config.EffectivePrefillTP()); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: PD disaggregation requires valid ModelConfig for KV transfer sizing: %v", err))
		}
	}

	// PDTransferContention is valid for any PD-enabled deployment, including pure-shared
	// (shared pod → shared pod KV transfer is possible when prefill and decode land on
	// different shared pods). Only reject when PD is entirely disabled. (#1276)
	if config.PDTransferContention && config.PrefillInstances == 0 && config.DecodeInstances == 0 && config.SharedInstances == 0 {
		panic("ClusterSimulator: PDTransferContention requires PD disaggregation (--prefill-instances, --decode-instances, or --prefill-decode-instances must be set)")
	}

	// Build pre-construction pool membership so instance construction can resolve per-pool config.
	// When disaggregation is disabled (all pool counts are 0), prePoolMembership is nil
	// and all instances use the global config (backward-compatible).
	var prePoolMembership map[string]PoolRole
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 || config.EncodeInstances > 0 {
		prePoolMembership = BuildPoolMembershipFromIndices(config.NumInstances, config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.EncodeInstances)
	}

	// instances and instanceMap are populated by the unified construction+placement loop below.
	// Declared here so they are available throughout NewClusterSimulator.
	instanceMap := make(map[InstanceID]*InstanceSimulator, config.NumInstances)

	// Initialize trace collector if tracing is enabled (BC-1: nil when none)
	var simTrace *trace.SimulationTrace
	tracingEnabled := config.TraceLevel != "" && trace.TraceLevel(config.TraceLevel) != trace.TraceLevelNone
	if tracingEnabled || config.RecordRoutingDecisions {
		simTrace = trace.NewSimulationTrace(trace.TraceConfig{
			Level:                  trace.TraceLevel(config.TraceLevel),
			CounterfactualK:        config.CounterfactualK,
			RecordRoutingDecisions: config.RecordRoutingDecisions,
		})
	}

	// Extract PartitionedRNG before struct literal so routing policy can use SubsystemRouter.
	// The routing policy exclusively owns the SubsystemRouter partition — do not reuse
	// cs.rng.ForSubsystem(SubsystemRouter) elsewhere to avoid interleaving RNG draws.
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed))

	// Construct SLO priority map from config overrides (nil-safe: defaults used when empty).
	priorityMap := sim.NewSLOPriorityMap(config.SLOPriorityOverrides)

	// Bypass generic factory for policies needing custom params (research.md D-2).
	var admissionPolicy sim.AdmissionPolicy
	switch config.AdmissionPolicy {
	case "tier-shed":
		if config.TierShedMinPriority == 0 {
			logrus.Warn("[cluster] tier-shed: TierShedMinPriority=0 rejects sheddable tiers (priority < 0) under overload; set tier_shed_min_priority: 3 for Standard-and-above protection")
		}
		admissionPolicy = sim.NewTierShedAdmission(config.TierShedThreshold, config.TierShedMinPriority, priorityMap)
	case "gaie-legacy":
		qdThreshold := config.GAIEQDThreshold
		if qdThreshold == 0 {
			qdThreshold = 5.0 // GAIE DefaultQueueDepthThreshold (config.go:31)
		}
		kvThreshold := config.GAIEKVThreshold
		if kvThreshold == 0 {
			kvThreshold = 0.8 // GAIE DefaultKVCacheUtilThreshold (config.go:33)
		}
		admissionPolicy = sim.NewGAIELegacyAdmission(qdThreshold, kvThreshold, priorityMap)
	default:
		admissionPolicy = sim.NewAdmissionPolicy(config.AdmissionPolicy, config.TokenBucketCapacity, config.TokenBucketRefillRate)
	}

	cs := &ClusterSimulator{
		config:               config,
		instances:            make([]*InstanceSimulator, 0, config.NumInstances),
		rng:                  rng,
		preGeneratedRequests: requests,
		clusterEvents:        make(ClusterEventQueue, 0),
		admissionLatency:     config.AdmissionLatency,
		routingLatency:       config.RoutingLatency,
		admissionPolicy:      admissionPolicy,
		priorityMap:          priorityMap,
		snapshotProvider:     nil, // set after unified construction loop below
		routingPolicy:        nil, // set after instance construction (needs cacheQueryFn from instances)
		trace:                simTrace,
		inFlightRequests:     make(map[string]int, config.NumInstances),
		shedByTier:           make(map[string]int),
		injectedByClass:      make(map[string]int64),
	}

	// PD disaggregation: set pool membership (topology already validated above).
	// Decider construction is deferred until after cs.cacheQueryFn is built
	// (PrefixThresholdDecider consumes the map).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 {
		cs.poolMembership = prePoolMembership
		cs.parentRequests = make(map[string]*ParentRequest)
		cs.pendingPrefillCompletions = make(map[string]string)
		cs.pendingDecodeCompletions = make(map[string]string)
		cs.localAdmitTimes = make(map[string]int64)
		cs.localEnqueueTimes = make(map[string]int64)

		// Per-pool routing policies and the disaggregation decider are created after
		// the construction loop (both need cacheQueryFn from instances).

		logrus.Infof("[cluster] PD disaggregation enabled: %d prefill, %d decode, %d shared (prefill-decode) instances, decider=%q",
			config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.PDDecider)

		// Issue #1276 D-2: shared-role pods resolve config via decode-wins precedence.
		// Warn when PrefillOverrides would have been applicable but is silently discarded.
		if config.SharedInstances > 0 && !reflect.DeepEqual(config.PrefillOverrides, config.DecodeOverrides) {
			logrus.Infof("[cluster] shared-role pods use DecodeOverrides (PrefillOverrides ignored per PR1276 decode-wins rule)")
		}
	}

	// Phase 1A: initialize PlacementManager when node pools are configured.
	// Must happen BEFORE the unified construction loop so cs.placement is set.
	if len(config.NodePools) > 0 {
		provisionRng := rng.ForSubsystem(subsystemNodeProvisioning)
		loadingRng := rng.ForSubsystem(subsystemInstanceLoading)
		cs.placement = NewPlacementManager(config.NodePools, provisionRng, loadingRng, 0)
	}

	// Unified construction+placement loop: construct each InstanceSimulator AFTER placement
	// so the pool's GPU type (authoritative) is used instead of the CLI flag (SC-004).
	// TP=0 in ModelHardwareConfig means "not configured" — treat as 1 GPU per instance.
	tpDegree := config.TP
	if tpDegree < 1 {
		tpDegree = 1 // default to TP=1 when not explicitly set (R3: defensive correction with comment)
	}
	for idx := 0; idx < config.NumInstances; idx++ {
		id := InstanceID(fmt.Sprintf("instance_%d", idx))
		role := PoolRole(0)
		if prePoolMembership != nil {
			role = prePoolMembership[string(id)]
		}
		simCfg := config.resolveConfigForRole(role)

		if cs.placement != nil {
			// NodePools path: placement determines GPU type (authoritative).
			// Pass "" as gpuType so PlacementManager selects any available pool
			// (the pool's gpu_type is the authoritative source, not the CLI --gpu flag).
			nodeID, gpuIDs, matchedGPUType, err := cs.placement.PlaceInstance(id, config.Model, "", tpDegree)
			if err != nil {
				// No capacity — defer construction until NodeReadyEvent.
				// Pass "" as gpuType (any pool) to match AddPending's placement semantics.
				cs.placement.AddPending(id, config.Model, "", tpDegree, simCfg)
				continue
			}
			// Placement succeeded: use pool's GPU type (SC-004: pool-authoritative, not CLI flag).
			// Set GPU label and, when HWConfigByGPU is provided, override HWConfig so that
			// roofline and trained-physics backends use the pool's hardware coefficients (issue #893).
			simCfg.GPU = matchedGPUType
			if hc, ok := config.HWConfigByGPU[matchedGPUType]; ok {
				if hc.TFlopsPeak <= 0 || hc.BwPeakTBs <= 0 {
					panic(fmt.Sprintf("HWConfigByGPU[%q]: TFlopsPeak and BwPeakTBs must be positive, got TFlopsPeak=%v BwPeakTBs=%v",
						matchedGPUType, hc.TFlopsPeak, hc.BwPeakTBs))
				}
				simCfg.HWConfig = hc
			}
			// Phase 1C: look up CostPerHour for this matched GPU type (issue #692).
			var poolCostPerHour float64
			for i := range config.NodePools {
				if config.NodePools[i].GPUType == matchedGPUType {
					poolCostPerHour = config.NodePools[i].CostPerHour
					break
				}
			}
			inst := NewInstanceSimulator(id, simCfg)
			inst.Model = config.Model
			inst.nodeID = nodeID
			inst.allocatedGPUIDs = gpuIDs
			inst.TPDegree = tpDegree
			inst.CostPerHour = poolCostPerHour
			inst.warmUpRemaining = config.InstanceLifecycle.WarmUpRequestCount
			if config.InstanceLifecycle.WarmStartInitialInstances {
				// Pre-deployed cluster: startup instances skip loading delay and start Active.
				// Autoscaler-added instances (via direct_actuator.go) still use LoadingDelay.
				if inst.warmUpRemaining > 0 {
					inst.TransitionTo(sim.InstanceStateWarmingUp)
				} else {
					inst.TransitionTo(sim.InstanceStateActive)
				}
			} else {
				inst.TransitionTo(sim.InstanceStateLoading)
				cs.scheduleInstanceLoadedEvent(inst)
			}
			cs.instances = append(cs.instances, inst)
			instanceMap[id] = inst
			cs.inFlightRequests[string(id)] = 0
		} else {
			// No NodePools: the CLI --gpu flag (config.ModelHardwareConfig.GPU, accessed via
			// DeploymentConfig's embedded SimConfig) is the authoritative source (backward-compat).
			// simCfg.GPU is already set — resolveConfigForRole returns config.SimConfig as-is
			// for the default role, preserving ModelHardwareConfig.GPU from the CLI flag.
			inst := NewInstanceSimulator(id, simCfg)
			inst.Model = config.Model
			inst.warmUpRemaining = config.InstanceLifecycle.WarmUpRequestCount
			if inst.warmUpRemaining > 0 {
				inst.TransitionTo(sim.InstanceStateWarmingUp)
			} else {
				inst.TransitionTo(sim.InstanceStateActive)
			}
			cs.instances = append(cs.instances, inst)
			instanceMap[id] = inst
			cs.inFlightRequests[string(id)] = 0
		}
	}

	// Initialize snapshot provider with exactly the placed instances.
	// Deferred instances are registered via CachedSnapshotProvider.AddInstance
	// when NodeReadyEvent.Execute constructs them (Phase 4, T017).
	cs.snapshotProvider = NewCachedSnapshotProvider(instanceMap, newObservabilityConfig(config.SnapshotRefreshInterval, config.CacheSignalDelay))

	// Build cacheQueryFn from the unified snapshot provider (#1060).
	// When CacheSignalDelay > 0, CachedSnapshotProvider manages stale snapshots.
	// When CacheSignalDelay == 0, oracle mode: closures query live instance state.
	cs.cacheQueryFn = cs.snapshotProvider.BuildCacheQueryFn()

	// Create routing policies now that cacheQueryFn is available.
	cs.routingPolicy = sim.NewRoutingPolicyWithCache(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem(sim.SubsystemRouter), cs.cacheQueryFn)
	if len(config.PrefillScorerConfigs) > 0 {
		cs.prefillRoutingPolicy = sim.NewRoutingPolicyWithCache("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"), cs.cacheQueryFn)
	}
	if len(config.DecodeScorerConfigs) > 0 {
		cs.decodeRoutingPolicy = sim.NewRoutingPolicyWithCache("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"), cs.cacheQueryFn)
	}

	// PD disaggregation: construct the decider now that cacheQueryFn is available.
	// PrefixThresholdDecider consumes the per-pod cache-query map; other deciders
	// ignore state but share the same construction point.
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 {
		switch {
		case config.PDPlanPath != "":
			// Fixed-plan decider (counterfactual-regret harness / offline yardstick):
			// forces a supplied per-request (decode, prefill) plan. Takes precedence
			// over --pd-decider. Total plan (missing request is fatal, R1); INV-9.
			plan, err := sim.LoadFixedPlanCSV(config.PDPlanPath)
			if err != nil {
				logrus.Fatalf("[cluster] --pd-plan: %v", err)
			}
			cs.disaggregationDecider = sim.NewFixedPlanDecider(plan)
		case config.PDDecider == "prefix-threshold":
			cs.disaggregationDecider = sim.NewPrefixThresholdDecider(config.PDPrefixThreshold, int(config.BlockSizeTokens), cs.cacheQueryFn)
		case config.PDDecider == "edpp":
			// EDPP recovers α/δ by finite-difference on a latency model (no live scrape in
			// BLIS), so build the model here and inject it. Prefill-pool backlogs come from a
			// closure over the snapshot provider; the decode pool arrives via RouterState.
			lm, err := latency.NewLatencyModel(config.LatencyCoeffs, config.ModelHardwareConfig)
			if err != nil {
				logrus.Fatalf("[cluster] EDPP decider: latency model construction failed: %v", err)
			}
			prefillSnapshots := func() []sim.RoutingSnapshot {
				return cs.buildPoolFilteredSnapshots(PoolRolePrefill)
			}
			// Size-aware c_xfer (opt-in): derive per-GPU KV bytes/token from the model config +
			// prefill TP so EDPP's transfer cost matches the DES executor (pd_events.go). Validated
			// non-error at construction when PD is enabled (see the KVBytesPerToken guard above).
			var edppKVBytesPerTok float64
			if config.EDPPCXferSizeAware {
				if v, err := latency.KVBytesPerToken(config.ModelConfig, config.EffectivePrefillTP()); err == nil {
					edppKVBytesPerTok = v
				} else {
					logrus.Fatalf("[cluster] EDPP --edpp-c-xfer-size-aware: cannot derive KV bytes/token: %v", err)
				}
			}
			cs.disaggregationDecider = sim.NewEDPPDecider(sim.EDPPConfig{
				TauTTFTUs:              config.EDPPTauTTFTUs,
				TauITLUs:               config.EDPPTauITLUs,
				TauRefUs:               config.EDPPTauRefUs,
				TauTTFTByClassUs:       config.EDPPTauTTFTByClassUs,
				TauITLByClassUs:        config.EDPPTauITLByClassUs,
				TauE2EUs:               config.EDPPTauE2EUs,
				TauE2EByClassUs:        config.EDPPTauE2EByClassUs,
				V:                      config.EDPPV,
				CXferUs:                config.EDPPCXferUs,
				NomPrefillTokens:       config.EDPPNomPrefillTokens,
				NomDecodeCtx:           config.EDPPNomDecodeCtx,
				BlockSize:              int(config.BlockSizeTokens),
				ChunkTokens:            int(config.BatchConfig.MaxScheduledTokens),
				TraceEnabled:           trace.TraceLevel(config.TraceLevel) == trace.TraceLevelDecisions,
				Coeffs:                 config.EDPPCoeffs,
				CoeffsByGPU:            config.EDPPCoeffsByGPU,
				TAdmEstimator:          config.EDPPTAdmEstimator,
				Joint:                  config.EDPPJoint,
				Rule:                   config.EDPPRule,
				VarMetric:              config.EDPPVarMetric,
				VarKeepCongestion:      config.EDPPVarKeepCongestion,
				VarCongestionWeight:    config.EDPPVarCongestionWeight,
				VarNormalize:           config.EDPPVarNormalize,
				VarNormalizeFloorScale: config.EDPPVarNormalizeFloorScale,
				VarDeployable:          config.EDPPVarDeployable,
				VarCollocPrefill:       config.EDPPVarCollocPrefill,
				VarGoodputObjective:    config.EDPPVarGoodputObjective,
				KairosBeta:             config.EDPPKairosBeta,
				JointTraceEnabled:      config.EDPPJoint && config.EDPPJointTrace,
				OracleOutputLen:        config.EDPPOracleOutputLen,
				CXferSizeAware:         config.EDPPCXferSizeAware,
				KVBytesPerTokenPerGPU:  edppKVBytesPerTok,
				XferBandwidthGBps:      config.PDTransferBandwidthGBps,
				XferBaseUs:             config.PDTransferBaseLatencyMs * 1000.0,
			}, lm, cs.cacheQueryFn, prefillSnapshots)
			// Inject the shadow prefill scorer used ONLY to populate the joint divergence
			// trace's scorer_p (logging-only). It runs a DEDICATED-RNG copy of the prefill
			// routing policy so shadow evaluation never perturbs production routing decisions
			// (INV-6). Wired only when the joint divergence trace is active.
			if config.EDPPJoint && config.EDPPJointTrace {
				if ed, ok := cs.disaggregationDecider.(*sim.EDPPDecider); ok {
					shadowScorers := config.PrefillScorerConfigs
					if len(shadowScorers) == 0 {
						shadowScorers = config.RoutingScorerConfigs
					}
					shadowPolicy := sim.NewRoutingPolicyWithCache("weighted", shadowScorers, config.BlockSizeTokens, rng.ForSubsystem("edpp-joint-shadow-prefill"), cs.cacheQueryFn)
					ed.SetPrefillScorer(func(req *sim.Request, snaps []sim.RoutingSnapshot) string {
						if len(snaps) == 0 {
							return ""
						}
						return shadowPolicy.Route(req, &sim.RouterState{Snapshots: snaps, Clock: cs.clock}).TargetInstance
					})
				}
			}
		default:
			cs.disaggregationDecider = sim.NewDisaggregationDecider(config.PDDecider)
		}
		// Capture the SLO-feedback hook once: deciders that track realized SLOs (EDPP)
		// get OnComplete callbacks from the per-request completion site (R4: single point).
		if fb, ok := cs.disaggregationDecider.(sim.SLOFeedbackDecider); ok {
			cs.sloFeedback = fb
		}
	}

	// E/P/D disaggregation (GAP-4, issue #1264): construct the encode decider
	// only when the encode pool is active. When EncodeInstances == 0, cs.encodeDecider
	// stays nil and the encode stage in executeDisaggregatedRouting is a no-op,
	// preserving byte-for-byte behavior for pre-PR runs (BC-EPD-1).
	if config.EncodeInstances > 0 {
		cs.encodeDecider = sim.NewEncodeDecider(config.EncodeDecider)
		logrus.Infof("[cluster] E/P/D enabled: %d encode instances, decider=%q", config.EncodeInstances, config.EncodeDecider)
	}

	// Phase 1C: initialize autoscaler pipeline when ModelAutoscalerIntervalUs > 0 (issue #692).
	// Zero interval disables the autoscaler entirely (INV-6 backward-compat).
	// The default WVA pipeline (DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine →
	// DirectActuator) is wired here. Tests that need custom components replace cs.autoscaler
	// after construction via same-package access.
	// R3: validate autoscaler float64 fields — NaN/Inf/negative values are configuration errors.
	if math.IsNaN(config.ModelAutoscalerIntervalUs) || math.IsInf(config.ModelAutoscalerIntervalUs, 0) {
		panic("ModelAutoscalerIntervalUs must not be NaN or Inf")
	}
	if config.ModelAutoscalerIntervalUs < 0 {
		panic("ModelAutoscalerIntervalUs must be ≥0 (0 = disabled)")
	}
	if math.IsNaN(config.ScaleUpStabilizationWindowUs) || math.IsInf(config.ScaleUpStabilizationWindowUs, 0) || config.ScaleUpStabilizationWindowUs < 0 {
		panic("ScaleUpStabilizationWindowUs must be a finite non-negative number")
	}
	if math.IsNaN(config.ScaleDownStabilizationWindowUs) || math.IsInf(config.ScaleDownStabilizationWindowUs, 0) || config.ScaleDownStabilizationWindowUs < 0 {
		panic("ScaleDownStabilizationWindowUs must be a finite non-negative number")
	}
	if math.IsNaN(config.HPAScrapeDelay.Mean) || math.IsInf(config.HPAScrapeDelay.Mean, 0) || config.HPAScrapeDelay.Mean < 0 {
		panic("HPAScrapeDelay.Mean must be a finite non-negative number")
	}
	if math.IsNaN(config.HPAScrapeDelay.Stddev) || math.IsInf(config.HPAScrapeDelay.Stddev, 0) || config.HPAScrapeDelay.Stddev < 0 {
		panic("HPAScrapeDelay.Stddev must be a finite non-negative number")
	}
	if config.ModelAutoscalerIntervalUs > 0 {
		// Wire the default WVA pipeline: DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator.
		// effectiveAnalyzerConfig fills zero fields with WVA reference defaults so callers only need interval_us.
		// Tests that need custom components (stubs, nopActuator) replace cs.autoscaler after construction (same-package access).
		analyzerCfg := effectiveAnalyzerConfig(config.AutoscalerAnalyzerConfig)
		cs.autoscaler = newAutoscalerPipeline(
			&DefaultCollector{},
			NewV2SaturationAnalyzer(analyzerCfg),
			&UnlimitedEngine{},
			NewDirectActuator(cs),
			rng.ForSubsystem(subsystemAutoscaler),
		)
	}

	// Flow control: per-band gateway queue with FlowControlAdmission policy (issue #882, #1191).
	// When disabled (default), the pipeline is unchanged — requests flow directly
	// from admission to routing (BC-1 pass-through equivalence).
	// Must be initialized BEFORE TenantBudgetAdmission so budget wrapping decorates FlowControlAdmission.
	if config.FlowControlEnabled {
		dispatchOrder := config.FlowControlDispatchOrder
		if dispatchOrder == "" {
			dispatchOrder = "fifo"
		}
		cs.flowControlEnabled = true
		gq := NewGatewayQueue(dispatchOrder, config.FlowControlMaxQueueDepth, cs.priorityMap)
		if config.FlowControlSLOTargets != nil {
			gq.SetSLOTargets(config.FlowControlSLOTargets)
		}
		if config.FlowControlPerBandCapacity > 0 {
			gq.SetPerBandCapacity(config.FlowControlPerBandCapacity)
		}
		if config.FlowControlUsageLimitThreshold > 0 && config.FlowControlUsageLimitThreshold < 1.0 {
			gq.SetUsageLimitThreshold(config.FlowControlUsageLimitThreshold)
		}
		switch config.FlowControlFairnessPolicy {
		case "round-robin":
			gq.SetFairnessPolicy(NewRoundRobinPolicy())
		case "", "global-strict":
			// default — already set
		default:
			panic(fmt.Sprintf("ClusterSimulator: unknown fairness policy %q (must be global-strict or round-robin)", config.FlowControlFairnessPolicy))
		}
		cs.gatewayQueue = gq
		cs.requestTTL = config.FlowControlRequestTTL
		if cs.requestTTL < 0 {
			panic(fmt.Sprintf("ClusterSimulator: FlowControlRequestTTL must be >= 0, got %d", cs.requestTTL))
		}
		if config.FlowControlQueueShedding {
			gq.SetSheddingEnabled(true)
		}
		cs.dispatchTickInterval = config.FlowControlDispatchTickInterval
		if cs.dispatchTickInterval < 0 {
			panic(fmt.Sprintf("ClusterSimulator: FlowControlDispatchTickInterval must be >= 0, got %d", cs.dispatchTickInterval))
		}
		if cs.dispatchTickInterval == 0 {
			cs.dispatchTickInterval = 1000 // default 1ms (llm-d parity)
		}
		cs.saturationDetector = sim.NewSaturationDetector(
			config.FlowControlDetector,
			config.FlowControlQueueDepthThreshold,
			config.FlowControlKVCacheUtilThreshold,
			config.FlowControlMaxConcurrency,
		)
		fcAdmission := NewFlowControlAdmission(gq)
		cs.flowControlAdmission = fcAdmission
		cs.admissionPolicy = fcAdmission
		if config.FlowControlInFlightEviction {
			cs.evictionTracker = NewEvictionTracker()
		}
		fairness := config.FlowControlFairnessPolicy
		if fairness == "" {
			fairness = "global-strict"
		}
		logrus.Infof("[cluster] flow control enabled: detector=%q, dispatch=%q, fairness=%q, maxDepth=%d, perBandCapacity=%d, requestTTL=%d, queueShedding=%v, inFlightEviction=%v",
			config.FlowControlDetector, dispatchOrder, fairness, config.FlowControlMaxQueueDepth, config.FlowControlPerBandCapacity, config.FlowControlRequestTTL, config.FlowControlQueueShedding, config.FlowControlInFlightEviction)
	}

	// Phase 1B-2a: initialize TenantTracker when TenantBudgets is configured (issue #811).
	// totalCapacity = NumInstances × MaxRunningReqs (batch size proxy for cluster-wide capacity).
	// Wraps whichever admission policy is active (including FlowControlAdmission when flow control enabled).
	if config.TenantBudgets != nil {
		totalCapacity := config.NumInstances * int(config.MaxRunningReqs)
		if len(config.TenantBudgets) > 0 && totalCapacity == 0 {
			logrus.Warnf("[cluster] tenant_budgets configured but totalCapacity=0 (NumInstances=%d, MaxRunningReqs=%d); all budgeted tenants will be immediately over-budget — set max_running_reqs > 0",
				config.NumInstances, config.MaxRunningReqs)
		}
		cs.tenantTracker = NewTenantTracker(config.TenantBudgets, totalCapacity)
		cs.admissionPolicy = sim.NewTenantBudgetAdmission(cs.admissionPolicy, cs.tenantTracker, cs.priorityMap)
	}

	// Startup warning: horizon too small for pipeline (BC-1)
	pipelineLatency := cs.admissionLatency + cs.routingLatency
	if cs.config.Horizon > 0 && cs.config.Horizon < pipelineLatency {
		logrus.Warnf("[cluster] horizon (%d) < pipeline latency (%d); no requests can complete — increase --horizon or reduce admission/routing latency",
			cs.config.Horizon, pipelineLatency)
	}

	// Store raw callback for PD session follow-up (issue #884).
	cs.sessionCallback = onRequestDone

	// Wire OnRequestDone callback on each instance (BC-9: follow-ups route through cluster pipeline).
	// The callback pushes follow-up requests as ClusterArrivalEvents, ensuring they go through
	// admission → routing → instance injection. The callback returns nil so the per-instance
	// simulator does not inject locally.
	// Phase 1B-2a: also notify tenantTracker on completion when budgets are configured.
	if onRequestDone != nil || cs.tenantTracker != nil || cs.evictionTracker != nil || cs.sloFeedback != nil {
		for _, inst := range cs.instances {
			inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
				// Phase 1B-2a: release tenant in-flight slot on every terminal state.
				if cs.tenantTracker != nil {
					cs.tenantTracker.OnComplete(req.TenantID)
				}
				// Remove from eviction tracker on normal completion (BC-3).
				if cs.evictionTracker != nil {
					cs.evictionTracker.Untrack(req.ID)
				}
				// EDPP virtual-queue feedback (no-op unless an SLO-feedback decider is set).
				cs.feedSLOFeedback(req)
				if onRequestDone == nil {
					return nil
				}
				nextReqs := onRequestDone(req, tick)
				for _, next := range nextReqs {
					cs.pushArrival(next, next.ArrivalTime)
				}
				return nil // don't inject locally — route through cluster pipeline
			}
			if cs.sloFeedback != nil {
				inst.sim.OnAdmit = func(req *sim.Request, tick int64) {
					cs.feedAdmission(req)
					cs.recordAdmissionTime(req, tick)
				}
				inst.sim.OnFirstToken = func(req *sim.Request, tick int64) {
					cs.feedFirstToken(req, tick)
				}
			}
		}
	}

	// VaR drift rule (--edpp-rule var, design 2026-07-21): the value-at-risk externality needs
	// each decode co-resident's state (StepsDone, arrival, first-token, class), which populates
	// only when per-instance admission detail is on. Enable it. The ORACLE flavor additionally
	// populates the un-censored true remaining steps (a gated INV-9 violation, loud CLI warning);
	// the DEPLOYABLE flavor (--edpp-var-deployable) leaves TrueRemaining censored and estimates
	// remaining from the per-class N̂_out instead (INV-9-safe).
	if config.EDPPRule == "var" {
		oracle := !config.EDPPVarDeployable
		for _, inst := range cs.instances {
			inst.sim.SetAdmissionDetail(oracle)
		}
	}

	return cs
}

// registerInstanceCacheQueryFn adds a cacheQueryFn entry for a single instance,
// choosing between stale (snapshot) and oracle (live) modes based on
// ObservabilityConfig.CacheBlocks (#1060).
// Called from NodeReadyEvent.Execute (deferred instances).
// Precondition: cs.cacheQueryFn must be non-nil (initialised before calling).
func (cs *ClusterSimulator) registerInstanceCacheQueryFn(id InstanceID, inst *InstanceSimulator) {
	if cs.snapshotProvider.IsStaleCacheMode() {
		// Stale mode: register with CachedSnapshotProvider; the closure delegates
		// to CacheQuery at call time, picking up refreshed snapshots automatically.
		cs.snapshotProvider.AddCacheInstance(id, inst)
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []sim.TokenID) int {
			return cs.snapshotProvider.CacheQuery(idStr, tokens)
		}
	} else {
		// Oracle mode: closure captures inst directly for live-state queries.
		idStr := string(id)
		cs.cacheQueryFn[idStr] = func(tokens []sim.TokenID) int {
			return inst.GetCachedBlockCount(tokens)
		}
	}
}

// pushArrival enqueues a ClusterArrivalEvent and increments pendingArrivals
// as a paired operation. All ClusterArrivalEvent pushes MUST go through this
// method — it is the single enforcement point for the pendingArrivals
// co-invariant used by scheduleNextTick (autoscaler.go). Direct heap.Push of
// ClusterArrivalEvent outside this method is prohibited.
// The paired decrement occurs in ClusterArrivalEvent.Execute (cluster_event.go).
//
// NOTE: When ClusterSubsystem hooks are introduced (see discussion #1033 PR D),
// OnRequestDone hooks that generate follow-ups should return them to the
// coordinator rather than calling pushArrival directly, preserving this method
// as the single enforcement point. See also issue #1041.
func (cs *ClusterSimulator) pushArrival(req *sim.Request, timeUs int64) {
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &ClusterArrivalEvent{time: timeUs, request: req},
		seqID: cs.nextSeqID(),
	})
	cs.pendingArrivals++
}

// Run executes the cluster simulation using online routing pipeline:
// generates requests centrally, schedules ClusterArrivalEvents, runs a shared-clock
// event loop processing cluster events before instance events, then finalizes.
// Panics if called more than once.
func (c *ClusterSimulator) Run() error {
	if c.hasRun {
		panic("ClusterSimulator.Run() called more than once")
	}
	c.hasRun = true

	// 1. Use pre-generated requests (all workload paths now pre-generate)
	requests := c.preGeneratedRequests
	if len(requests) == 0 {
		logrus.Warn("[cluster] no requests provided — simulation will produce zero results")
	}

	// 2. Schedule ClusterArrivalEvents (NC-1: no pre-dispatch before event loop)
	heap.Init(&c.clusterEvents)

	// Phase 1C: schedule the first ScalingTickEvent when the autoscaler is enabled (T015).
	// The autoscaler is enabled when ModelAutoscalerIntervalUs > 0 AND cs.autoscaler is non-nil.
	// Zero-interval guard: no tick is ever scheduled when interval is 0 (INV-6).
	if c.autoscaler != nil && c.config.ModelAutoscalerIntervalUs > 0 {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &ScalingTickEvent{At: c.clock},
			seqID: c.nextSeqID(),
		})
	}

	for _, req := range requests {
		c.pushArrival(req, req.ArrivalTime)
	}

	// 3. Shared-clock event loop (BC-4: cluster events before instance events)
	for {
		// Find earliest cluster event time
		clusterTime := int64(math.MaxInt64)
		if len(c.clusterEvents) > 0 {
			clusterTime = c.clusterEvents[0].event.Timestamp()
		}

		// Find earliest instance event time
		instanceTime := int64(math.MaxInt64)
		instanceIdx := -1
		for idx, inst := range c.instances {
			if inst.HasPendingEvents() {
				t := inst.PeekNextEventTime()
				if t < instanceTime {
					instanceTime = t
					instanceIdx = idx
				}
			}
		}

		// Both queues empty: done
		if clusterTime == math.MaxInt64 && instanceIdx == -1 {
			break
		}

		// BC-4: Cluster events at time T processed before instance events at time T
		// Using <= ensures cluster events drain first when timestamps are equal
		if clusterTime <= instanceTime {
			entry := heap.Pop(&c.clusterEvents).(clusterEventEntry)
			c.clock = entry.event.Timestamp()
			if c.clock > c.config.Horizon {
				break
			}
			entry.event.Execute(c)
		} else {
			prevClusterClock := c.clock
			c.clock = instanceTime
			if c.clock > c.config.Horizon {
				break
			}
			inst := c.instances[instanceIdx]
			instID := string(inst.ID())

			// Snapshot counters BEFORE processing the event
			completedBefore := inst.Metrics().CompletedRequests
			droppedBefore := inst.Metrics().DroppedUnservable
			timedOutBefore := inst.Metrics().TimedOutRequests

			ev := inst.ProcessNextEvent()

			// If ProcessNextEvent() skipped a cancelled TimeoutEvent (lazy
			// cancellation — inst.Clock was not advanced), restore c.clock.
			// A no-op orphaned timeout must not advance the cluster clock.
			if te, ok := ev.(*sim.TimeoutEvent); ok && te.Request.State == sim.StateCompleted {
				c.clock = prevClusterClock
			}

			// Completion-based decrement (#463, BC-3, BC-7): InFlightRequests tracks the full
			// dispatch-to-completion window. Decrement by the number of newly completed,
			// dropped-unservable, or timed-out requests.
			completedAfter := inst.Metrics().CompletedRequests
			droppedAfter := inst.Metrics().DroppedUnservable
			timedOutAfter := inst.Metrics().TimedOutRequests
			delta := (completedAfter - completedBefore) + (droppedAfter - droppedBefore) + (timedOutAfter - timedOutBefore)
			if delta > 0 {
				c.inFlightRequests[instID] -= delta
				if c.inFlightRequests[instID] < 0 {
					// Warn-and-clamp: inFlightRequests is a best-effort routing signal
					// (INV-7); it recovers from delta mis-accounting and does not corrupt
					// deterministic metrics. Contrast with activeTransfers (contention
					// subsystem) which uses a hard error because contention metrics are
					// meaningless once the counter is wrong.
					logrus.Warnf("inFlightRequests[%s] went negative (%d) after delta=%d (completed=%d, dropped=%d, timedOut=%d) — bookkeeping bug",
						instID, c.inFlightRequests[instID], delta, completedAfter-completedBefore, droppedAfter-droppedBefore, timedOutAfter-timedOutBefore)
					c.inFlightRequests[instID] = 0
				}
				// T042: consume warm-up slots for newly completed requests (Phase 1A).
				// Each completion on a WarmingUp instance counts against the warm-up budget.
				completionDelta := int(completedAfter - completedBefore)
				for i := 0; i < completionDelta; i++ {
					if inst.IsWarmingUp() {
						inst.ConsumeWarmUpRequest()
					}
				}

			}

			// T042: drain completion accounting (Phase 1A).
			// When a Draining instance has no more queued or running requests,
			// transition it to Terminated and release its GPU allocations.
			if inst.State == sim.InstanceStateDraining && inst.QueueDepth() == 0 && inst.BatchSize() == 0 {
				inst.TransitionTo(sim.InstanceStateTerminated)
				c.releaseInstanceGPUs(inst)
				c.snapshotProvider.RemoveCacheInstance(inst.ID())
				delete(c.cacheQueryFn, string(inst.ID()))
				// I1: a non-zero inFlightRequests at termination time indicates a bookkeeping bug —
				// a missing completion event or an early-termination race.
				if c.inFlightRequests[instID] != 0 {
					logrus.Warnf("[cluster] instance %s terminated with inFlightRequests=%d — bookkeeping bug",
						instID, c.inFlightRequests[instID])
				}
			}

			// PD disaggregation: detect prefill/decode sub-request completions.
			// Set-membership (.Has) so a shared-role pod (PoolRolePrefillDecode)
			// fires both detectors — issue #1276 BC-5.
			if c.poolsConfigured() {
				role := c.poolMembership[instID]
				if role.Has(PoolRolePrefill) {
					c.detectPrefillCompletions(inst)
					c.detectPrefillTimeouts(inst)
				}
				if role.Has(PoolRoleDecode) {
					c.detectDecodeCompletions(inst)
				}
			}
		}

		c.maybeDeliverProgressSnapshot(false)
	}

	c.maybeDeliverProgressSnapshot(true)

	// 4. Finalize all instances (populates StillQueued/StillRunning)
	for _, inst := range c.instances {
		inst.Finalize()
	}

	// 5. Post-simulation invariant: inFlightRequests should match StillQueued + StillRunning
	// MUST be after Finalize() — StillQueued/StillRunning are zero until Finalize populates them.
	// NOTE: A mismatch can occur legitimately if requests were routed near the horizon but their
	// ArrivalEvent/QueuedEvent hadn't fired yet (request is in the instance event queue, not in
	// WaitQ or RunningBatch). This is an edge case, not a bookkeeping bug.
	for _, inst := range c.instances {
		instID := string(inst.ID())
		inflight := c.inFlightRequests[instID]
		m := inst.Metrics()
		expectedInFlight := m.StillQueued + m.StillRunning
		if inflight != expectedInFlight {
			logrus.Warnf("post-simulation: inFlightRequests[%s] = %d, expected %d (StillQueued=%d + StillRunning=%d) — may indicate bookkeeping bug or requests in event pipeline at horizon",
				instID, inflight, expectedInFlight, m.StillQueued, m.StillRunning)
		}
	}

	c.aggregatedMetrics = c.aggregateMetrics()

	// R1/INV-1: PD disaggregation conservation correction.
	// Each disaggregated request generates two sub-requests (prefill + decode) that
	// complete on separate instances. aggregateMetrics() naively sums CompletedRequests
	// across all instances, double-counting: prefill completion + decode completion = 2
	// for each original request. Subtract prefill completions to restore correct count.
	if c.pdPrefillCompletedCount > 0 {
		c.aggregatedMetrics.CompletedRequests -= c.pdPrefillCompletedCount
	}
	// Requests dropped at decode KV allocation: the prefill sub-request already
	// completed (counted above and subtracted), but the original request is lost.
	// Count as DroppedUnservable for INV-1 conservation.
	if c.droppedAtDecodeKV > 0 {
		c.aggregatedMetrics.DroppedUnservable += c.droppedAtDecodeKV
	}
	// In-flight PD transfers: requests whose prefill completed but decode hasn't
	// finished or been dropped yet (e.g., simulation ended at bounded horizon while
	// KV transfer was in progress). These requests were subtracted from CompletedRequests
	// but don't appear in any instance's StillQueued/StillRunning/DroppedUnservable.
	// Count them as StillRunning for conservation.
	//
	// Distinguish four sub-states of "prefill completed but decode not done":
	// - pendingDecodeCompletions: decode sub-requests already injected into instances
	//   (appear in instance StillQueued/StillRunning via Finalize — do NOT add again)
	// - pdInTransfer: requests still in KV transfer or cluster event queue
	//   (not on any instance — must be added to StillRunning)
	// - timed-out prefills: entries may remain in pendingPrefillCompletions but
	//   pdPrefillCompletedCount was NOT incremented; the timeout is already counted
	//   in instance TimedOutRequests → aggregated via aggregateMetrics(). No correction needed.
	// - timed-out decodes: counted in pdDecodeTimedOutCount; already in instance
	//   TimedOutRequests via aggregateMetrics(). Subtracted here to keep pdInTransfer = 0.
	pdInTransfer := c.pdPrefillCompletedCount - c.pdDecodeCompletedCount - c.pdDecodeTimedOutCount - c.droppedAtDecodeKV - len(c.pendingDecodeCompletions)
	if pdInTransfer > 0 {
		c.aggregatedMetrics.StillRunning += pdInTransfer
	} else if pdInTransfer < 0 {
		logrus.Warnf("[cluster] pdInTransfer = %d (negative): prefillCompleted=%d, decodeCompleted=%d, decodeTimedOut=%d, droppedAtDecodeKV=%d, pendingDecode=%d — bookkeeping bug in PD disaggregation accounting",
			pdInTransfer, c.pdPrefillCompletedCount, c.pdDecodeCompletedCount, c.pdDecodeTimedOutCount, c.droppedAtDecodeKV, len(c.pendingDecodeCompletions))
	}

	// INV-PD-6: Project sub-request metrics to parent-request granularity.
	// aggregateMetrics() merges per-instance maps keyed by sub-request IDs
	// (req_N_prefill, req_N_decode). Replace with parent-keyed entries so
	// user-facing distributions reflect the full request lifecycle.
	c.projectPDMetrics()

	// Post-simulation contention bookkeeping checks (INV-P2-2)
	if c.contentionBookkeepingCorrupted {
		return fmt.Errorf("contention bookkeeping corrupted: activeTransfers went negative during simulation — contention metrics are invalid")
	}
	if c.config.PDTransferContention && c.activeTransfers != 0 {
		logrus.Warnf("[cluster] post-simulation: activeTransfers = %d (expected 0), initiated=%d completed=%d — contention metrics (PeakConcurrentTransfers, MeanTransferQueueDepth) may be inflated if horizon cut off in-flight transfers",
			c.activeTransfers, c.transfersInitiated, c.transfersCompleted)
	}

	// Flow control: log gateway queue state at simulation end
	if c.flowControlEnabled && c.gatewayQueue.Len() > 0 {
		logrus.Warnf("[cluster] %d requests remain in gateway queue at simulation end", c.gatewayQueue.Len())
	}

	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by admission policy %q — no requests completed",
				c.rejectedRequests, c.config.AdmissionPolicy)
		} else if c.aggregatedMetrics.TimedOutRequests > 0 {
			logrus.Warnf("[cluster] no requests completed — %d of %d requests timed out (client timeout exceeded, likely KV pressure)",
				c.aggregatedMetrics.TimedOutRequests,
				c.aggregatedMetrics.TimedOutRequests+c.aggregatedMetrics.DroppedUnservable)
		} else {
			logrus.Warnf("[cluster] no requests completed — horizon may be too short or workload too small")
		}
	}

	return nil
}

// nextSeqID returns the next monotonically increasing sequence ID for event ordering.
func (c *ClusterSimulator) nextSeqID() int64 {
	id := c.seqCounter
	c.seqCounter++
	return id
}

// SetProgressHook registers an optional hook that receives periodic state
// snapshots during cluster simulation execution. Must be called before Run().
// When hook is nil (default), there is zero behavioral or performance impact.
// simClockIntervalUs controls the minimum simulation-clock interval (microseconds)
// between periodic snapshots. If simClockIntervalUs <= 0, only the final snapshot
// is delivered.
func (c *ClusterSimulator) SetProgressHook(hook sim.ProgressHook, simClockIntervalUs int64) {
	c.progressHook = hook
	if simClockIntervalUs > 0 {
		c.simClockProgressIntervalUs = simClockIntervalUs
		c.nextSnapshotClockUs = simClockIntervalUs
	}
}

func (c *ClusterSimulator) maybeDeliverProgressSnapshot(isFinal bool) {
	if c.progressHook == nil {
		return
	}
	if !isFinal && (c.simClockProgressIntervalUs <= 0 || c.clock < c.nextSnapshotClockUs) {
		return
	}

	activeCount := 0
	instanceSnaps := make([]sim.InstanceSnapshot, 0, len(c.instances))
	for _, inst := range c.instances {
		if inst.State == sim.InstanceStateTerminated {
			continue
		}
		if inst.State == sim.InstanceStateActive || inst.State == sim.InstanceStateWarmingUp {
			activeCount++
		}
		instanceSnaps = append(instanceSnaps, sim.InstanceSnapshot{
			ID:                string(inst.ID()),
			QueueDepth:        inst.QueueDepth(),
			BatchSize:         inst.BatchSize(),
			KVUtilization:     inst.KVUtilization(),
			KVFreeBlocks:      inst.FreeKVBlocks(),
			KVTotalBlocks:     inst.TotalKVBlocks(),
			CacheHitRate:      inst.CacheHitRate(),
			PreemptionCount:   inst.PreemptionCount(),
			CompletedRequests: inst.Metrics().CompletedRequests,
			InFlightRequests:  c.inFlightRequests[string(inst.ID())],
			TimedOutRequests:  inst.Metrics().TimedOutRequests,
			State:             inst.State,
			Model:             inst.Model,
		})
	}

	var gatewayQueueDepth, gatewayQueueShed int
	if c.gatewayQueue != nil {
		gatewayQueueDepth = c.gatewayQueue.Len()
		gatewayQueueShed = c.gatewayQueue.ShedCount()
	}

	clock := c.clock
	if isFinal {
		clock = min(c.clock, c.config.Horizon)
	}

	snap := sim.ProgressSnapshot{
		Clock:             clock,
		TotalCompleted:    c.completedRequestsTotal(),
		TotalTimedOut:     c.timedOutRequestsTotal(),
		TotalDropped:      c.droppedRequestsTotal(),
		TotalInputTokens:  c.inputTokensTotal(),
		TotalOutputTokens: c.outputTokensTotal(),
		TotalPreemptions:  c.preemptionsTotal(),
		InstanceSnapshots: instanceSnaps,
		RejectedRequests:  c.rejectedRequests,
		RoutingRejections: c.routingRejections,
		GatewayQueueDepth: gatewayQueueDepth,
		GatewayQueueShed:  gatewayQueueShed,
		GatewayEvicted:    c.gatewayEvicted,
		GatewayExpired:    c.gatewayExpired,
		ActivePDTransfers: c.activeTransfers,
		ActiveInstances:   activeCount,
		TotalInstances:    len(c.instances),
		IsFinal:           isFinal,
	}
	if len(c.shedByTier) > 0 {
		snap.ShedByTier = make(map[string]int, len(c.shedByTier))
		for k, v := range c.shedByTier {
			snap.ShedByTier[k] = v
		}
	}
	c.progressHook.OnProgress(snap)
	if !isFinal {
		c.nextSnapshotClockUs += c.simClockProgressIntervalUs
	}
}

func (c *ClusterSimulator) completedRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().CompletedRequests
	}
	return total
}

func (c *ClusterSimulator) timedOutRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TimedOutRequests
	}
	return total
}

func (c *ClusterSimulator) droppedRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().DroppedUnservable
	}
	return total
}

func (c *ClusterSimulator) inputTokensTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TotalInputTokens
	}
	return total
}

func (c *ClusterSimulator) outputTokensTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TotalOutputTokens
	}
	return total
}

func (c *ClusterSimulator) preemptionsTotal() int64 {
	var total int64
	for _, inst := range c.instances {
		total += inst.Metrics().PreemptionCount
	}
	return total
}

// addLiveInstance constructs, registers, and activates an InstanceSimulator for a
// placement that succeeded while the cluster is already running.
// Called from NodeReadyEvent.Execute (deferred placement) and DirectActuator.scaleUp
// (autoscaler direct placement). NOT used by the NewClusterSimulator startup path,
// which bulk-initialises the snapshot provider with a full instance map.
//
// simCfg must already have GPU and HWConfig set (pool-authoritative, SC-004).
// Returns true on success. On false, GPU allocations have been released — callers
// must not touch the instance and should skip/continue.
//
// Maintenance note: if you add a new field to instance initialisation here, also
// check NewClusterSimulator's construction loop (which does NOT call this method).
func (cs *ClusterSimulator) addLiveInstance(
	id InstanceID,
	model string,
	simCfg sim.SimConfig,
	nodeID string,
	gpuIDs []string,
	tpDegree int,
	costPerHour float64,
) bool {
	inst := NewInstanceSimulator(id, simCfg)
	inst.Model = model
	inst.nodeID = nodeID
	inst.allocatedGPUIDs = gpuIDs
	inst.TPDegree = tpDegree
	inst.CostPerHour = costPerHour
	inst.warmUpRemaining = cs.config.InstanceLifecycle.WarmUpRequestCount
	inst.TransitionTo(sim.InstanceStateLoading)

	if cs.snapshotProvider == nil {
		// snapshotProvider is nil — can only happen in unit tests that bypass
		// NewClusterSimulator. Release GPUs so they are not held by a phantom
		// instance (R1: no silent data loss).
		logrus.Warnf("[cluster] addLiveInstance: snapshotProvider is nil for instance %s — releasing GPUs and skipping", id)
		cs.releaseInstanceGPUs(inst)
		return false
	}
	cs.snapshotProvider.AddInstance(id, inst)

	cs.scheduleInstanceLoadedEvent(inst)
	cs.instances = append(cs.instances, inst)
	cs.inFlightRequests[string(id)] = 0

	// Register with cacheQueryFn for precise prefix scoring.
	// registerInstanceCacheQueryFn handles both oracle and stale modes (R23).
	if cs.cacheQueryFn != nil {
		cs.registerInstanceCacheQueryFn(id, inst)
	}

	// Wire OnRequestDone callback — mirrors startup path in NewClusterSimulator (R4).
	onRequestDone := cs.sessionCallback
	if onRequestDone != nil || cs.tenantTracker != nil || cs.evictionTracker != nil || cs.sloFeedback != nil {
		inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
			if cs.tenantTracker != nil {
				cs.tenantTracker.OnComplete(req.TenantID)
			}
			if cs.evictionTracker != nil {
				cs.evictionTracker.Untrack(req.ID)
			}
			// EDPP virtual-queue feedback (no-op unless an SLO-feedback decider is set).
			cs.feedSLOFeedback(req)
			if onRequestDone == nil {
				return nil
			}
			nextReqs := onRequestDone(req, tick)
			for _, next := range nextReqs {
				cs.pushArrival(next, next.ArrivalTime)
			}
			return nil // don't inject locally — route through cluster pipeline
		}
		if cs.sloFeedback != nil {
			inst.sim.OnAdmit = func(req *sim.Request, tick int64) {
				cs.feedAdmission(req)
				cs.recordAdmissionTime(req, tick)
			}
			inst.sim.OnFirstToken = func(req *sim.Request, tick int64) {
				cs.feedFirstToken(req, tick)
			}
		}
	}

	return true
}

// poolsConfigured returns true if PD disaggregation pool topology is active.
func (c *ClusterSimulator) poolsConfigured() bool {
	return c.poolMembership != nil
}

// PoolMembership returns a copy of the pool role membership map (R8: no exported mutable maps).
// Returns nil when disaggregation is disabled.
func (c *ClusterSimulator) PoolMembership() map[string]PoolRole {
	if c.poolMembership == nil {
		return nil
	}
	result := make(map[string]PoolRole, len(c.poolMembership))
	for k, v := range c.poolMembership {
		result[k] = v
	}
	return result
}

// ParentRequests returns a sorted slice of defensive copies of parent request tracking records.
// Each ParentRequest struct is copied by value so callers cannot mutate lifecycle timestamps (R8).
// Note: OriginalRequest and DecodeSubReq are shared *sim.Request pointers — callers must not mutate via them.
// Panics if called before Run() completes. Returns an empty (non-nil) slice when disaggregation is disabled,
// allowing callers to range over the result without a nil check.
func (c *ClusterSimulator) ParentRequests() []*ParentRequest {
	if !c.hasRun {
		panic("ClusterSimulator.ParentRequests() called before Run()")
	}
	result := make([]*ParentRequest, 0, len(c.parentRequests))
	for _, pr := range c.parentRequests {
		cp := *pr
		result = append(result, &cp)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

// buildPoolFilteredSnapshots constructs routing snapshots filtered to a specific pool role.
// Filters by IsRoutable() for parity with buildRouterState (R23), then by pool role.
// Model filter is intentionally omitted: all instances in a DeploymentConfig share config.Model,
// so pool-role filtering is sufficient. If multi-model PD clusters are added, add model filtering here.
// Preserves instance order from c.instances for determinism (R2).
//
// INV-7: refreshes stale cache snapshots before sampling so the disaggregated routing
// path observes the same cache-block view as buildRouterState under Periodic mode.
func (c *ClusterSimulator) buildPoolFilteredSnapshots(role PoolRole) []sim.RoutingSnapshot {
	// Refresh stale cache snapshots if interval has elapsed (#919, #1060).
	// No-op when CacheBlocks.Mode != Periodic (oracle mode).
	c.snapshotProvider.RefreshCacheIfNeeded(c.clock)

	allSnapshots := make([]sim.RoutingSnapshot, 0, len(c.instances))
	for _, inst := range c.instances {
		if !inst.IsRoutable() {
			continue
		}
		snap := c.snapshotProvider.Snapshot(inst.ID(), c.clock)
		snap.GPUType = inst.GPU()
		snap.InFlightRequests = c.inFlightRequests[string(inst.ID())]
		// Admission-rate signals feed the EDPP `little` estimator (λ_adm ≈ completion
		// rate, §3.8). The EDPP disaggregation path sources its snapshots here (decode-pool
		// state via DisaggregationDecisionEvent, prefill-pool state via the decider's
		// prefillSnapshots closure) rather than via buildRouterState, so these fields must
		// be populated here too for parity — otherwise AdmissionContext.AdmissionRate is
		// always 0 and `little` predicts 0 on every decision. Gated on admission detail so
		// the default path pays nothing (zero-cost; LatencyStats() only when enabled).
		if inst.AdmissionDetailEnabled() {
			snap.DispatchRate = inst.LatencyStats().DispatchRate
			snap.AdmissionRate = inst.WindowedAdmissionRate(c.clock)
		}
		allSnapshots = append(allSnapshots, snap)
	}
	return FilterSnapshotsByPool(allSnapshots, c.poolMembership, role)
}

// detectPrefillCompletions checks for newly completed prefill sub-requests on the given instance
// and schedules KV transfer events for each.
// R2/INV-6: Collects completed IDs into a sorted slice before processing to ensure
// deterministic nextSeqID() assignment regardless of Go's random map iteration order.
// feedSLOFeedback delivers a completed request's realized TTFT and mean ITL to an
// SLO-feedback decider (EDPP). It is a no-op unless such a decider is configured.
// Only requests that produced a first token and at least one ITL sample are fed;
// timed-out or zero-output requests carry no usable latency signal and are skipped.
//
// For PD-disaggregated requests this fires for the decode sub-request, so TTFT is
// measured from decode-side arrival (it omits the prefill+transfer prefix). The design
// accepts crude realized signals — the virtual queues self-correct over time (§5.1) —
// and the ITL signal that drives the disaggregation-payoff term is exact regardless of
// where it is measured.
func (c *ClusterSimulator) feedSLOFeedback(req *sim.Request) {
	if c.sloFeedback == nil {
		return
	}
	// Conservation key: the ID OnRoute used (Defect 1). For a PD-disaggregated request
	// this completion fires for the decode sub-request (ID == parent.ID+"_decode"), so
	// resolve back to the parent identity. Non-disaggregated requests complete under
	// their own ID (key == req.ID).
	key := c.edppConservationKey(req)

	// Conservation must hold for EVERY terminal state of a routed request, but the
	// virtual-queue feedback (z) needs a usable realized signal. A request that
	// produced a first token AND at least one ITL sample carries that signal and gets
	// the full OnComplete (bump z + update N̂_out; the waiting backlog was already
	// drained at admission via OnAdmit). A request that reached completion without a
	// usable latency signal (e.g. a single-token output has TTFT but no inter-token
	// latency, or a timed-out/zero-output request) must still conserve — Forget is
	// a no-op for the backlog (already drained at admission) but skips polluting z
	// (Defect 2: the guarded early-return used to leak this work).
	ttftUs := req.FirstTokenTime - req.ArrivalTime
	if !req.TTFTSet || len(req.ITL) == 0 || ttftUs < 0 {
		c.sloFeedback.Forget(key)
		return
	}
	var sum int64
	for _, v := range req.ITL {
		sum += v
	}
	c.sloFeedback.OnComplete(req, key, ttftUs, sum/int64(len(req.ITL)))
}

// edppConservationKey returns the stable conservation key OnRoute used for req: the
// original/parent request ID. For a decode sub-request (PD-disaggregated path) the
// per-instance Request.ID is parent.ID+"_decode", so we map it back to the parent via
// the persistent parentRequests index. Falls back to req.ID when no parent is found
// (non-disaggregated request, or a sub-request whose parent record was already pruned).
func (c *ClusterSimulator) edppConservationKey(req *sim.Request) string {
	if !req.IsDecodeSubRequest {
		return req.ID
	}
	if parentID, ok := c.pendingDecodeCompletions[req.ID]; ok {
		return parentID
	}
	// pendingDecodeCompletions may already be drained; fall back to scanning the
	// persistent parent index by DecodeSubReqID.
	for parentID, parent := range c.parentRequests {
		if parent != nil && parent.DecodeSubReqID == req.ID {
			return parentID
		}
	}
	return req.ID
}

// admissionConservationKey resolves a just-admitted request to the OnRoute key
// (the parent/original request ID) and which backlog share its admission drains.
// prefillSide=true ⇒ this is the prefill sub-request of a P-routed request
// (drains Q_p); prefillSide=false ⇒ a decode sub-request or a normal D-routed
// request (drains Q_d). known=false ⇒ the request was not routed under EDPP
// bookkeeping (skip).
func (cs *ClusterSimulator) admissionConservationKey(req *sim.Request) (key string, prefillSide bool, known bool) {
	if req.IsDecodeSubRequest {
		if parent, ok := cs.pendingDecodeCompletions[req.ID]; ok {
			return parent, false, true
		}
		return "", false, false
	}
	if parent, ok := cs.pendingPrefillCompletions[req.ID]; ok {
		return parent, true, true // prefill sub-request of a disaggregated request
	}
	return req.ID, false, true // normal (D-routed) request: key is its own ID
}

// feedAdmission notifies an SLO-feedback decider that a routed request entered a
// running batch, so it can drain the waiting backlog (EDPP waiting-only, §6.2).
func (cs *ClusterSimulator) feedAdmission(req *sim.Request) {
	if cs.sloFeedback == nil {
		return
	}
	key, prefillSide, known := cs.admissionConservationKey(req)
	if !known {
		return
	}
	cs.sloFeedback.OnAdmit(key, prefillSide)
}

// SetRecordPDOutcomes enables per-request admission-time capture for --pd-outcome-trace.
// The setter only flips the gate bool; it does not allocate localAdmitTimes (which is
// init'd solely in the PD-enabled cluster construction path). recordAdmissionTime returns
// early when the gate is off, and no read path dereferences localAdmitTimes when PD is off,
// so enabling the gate outside a PD deployment is safe (BuildPDOutcomeRecords simply yields
// zero records, which the CLI warns about).
func (cs *ClusterSimulator) SetRecordPDOutcomes(v bool) { cs.recordPDOutcomes = v }

// BuildPDOutcomeRecords assembles one realized-outcome record per request for the
// --pd-outcome-trace estimator-validation harness (Stage A). It walks disaggregated
// parents and locally-admitted requests, pairs each with its realized TTFT/ITL/E2E
// from m (keyed by original request ID), and returns records sorted by RequestID
// (INV-6). A t_adm is emitted only when both its enqueue and schedule instants are
// set; otherwise it stays zero. "disaggregated" means a distinct decode instance was
// used. Completion follows the metrics convention: RequestE2Es[id] > 0.
//
// Local t_adm: the non-disaggregated enqueue instant is captured in localEnqueueTimes
// (at routing time, when recordPDOutcomes or recordAdmissionTrace is set), so
// LocalTAdm = local_schedule − local_enqueue is emitted like the prefill/decode legs.
// It stays zero only if the enqueue instant was never recorded (e.g. request never routed).
func (cs *ClusterSimulator) BuildPDOutcomeRecords(m *sim.Metrics) []trace.PDOutcomeRecord {
	recs := make([]trace.PDOutcomeRecord, 0, len(cs.parentRequests)+len(cs.localAdmitTimes))
	tadm := func(enq, sched int64) int64 {
		if enq > 0 && sched >= enq {
			return sched - enq
		}
		return 0
	}
	realized := func(id string) (ttft, itl, e2e float64, done bool) {
		e2e = m.RequestE2Es[id]
		return m.RequestTTFTs[id], m.RequestITLs[id], e2e, e2e > 0
	}

	for id, p := range cs.parentRequests {
		ttft, itl, e2e, done := realized(id)
		class, in := "", 0
		if p.OriginalRequest != nil {
			class = p.OriginalRequest.SLOClass
			in = len(p.OriginalRequest.InputTokens)
		}
		recs = append(recs, trace.PDOutcomeRecord{
			RequestID: id, SLOClass: class, InputTokens: in,
			Disaggregated:   p.DecodeInstanceID != "" && p.DecodeInstanceID != p.PrefillInstanceID,
			PrefillInstance: string(p.PrefillInstanceID), DecodeInstance: string(p.DecodeInstanceID),
			PrefillEnqueue: p.PrefillEnqueueTime, PrefillSchedule: p.PrefillScheduleTime, PrefillTAdm: tadm(p.PrefillEnqueueTime, p.PrefillScheduleTime),
			DecodeEnqueue: p.DecodeEnqueueTime, DecodeSchedule: p.DecodeScheduleTime, DecodeTAdm: tadm(p.DecodeEnqueueTime, p.DecodeScheduleTime),
			RealizedTTFT: ttft, RealizedMeanITL: itl, RealizedE2E: e2e, Completed: done,
		})
	}

	for id, admit := range cs.localAdmitTimes {
		if _, isParent := cs.parentRequests[id]; isParent {
			continue // already emitted as a disagg/local-via-parent record
		}
		ttft, itl, e2e, done := realized(id)
		enq := cs.localEnqueueTimes[id] // routing/enqueue instant (0 if unrecorded)
		// Local records carry class/size/instance from the request metrics, so the
		// trace supports per-instance placement analysis for collocated requests too
		// (a collocated request prefills and decodes on its one serving instance).
		class, in, inst := "", 0, ""
		if rm, ok := m.Requests[id]; ok {
			class, in, inst = rm.SLOClass, rm.NumPrefillTokens, rm.HandledBy
		}
		recs = append(recs, trace.PDOutcomeRecord{
			RequestID: id, SLOClass: class, InputTokens: in, Disaggregated: false,
			PrefillInstance: inst, DecodeInstance: inst,
			LocalEnqueue: enq, LocalSchedule: admit, LocalTAdm: tadm(enq, admit),
			RealizedTTFT: ttft, RealizedMeanITL: itl, RealizedE2E: e2e, Completed: done,
		})
	}

	sort.Slice(recs, func(i, j int) bool { return recs[i].RequestID < recs[j].RequestID })
	return recs
}

// recordAdmissionTime captures the first admission instant of a request for the
// --pd-outcome-trace estimator-validation harness (Stage A). No-op unless
// recordPDOutcomes is set. A prefill sub-request sets its parent's
// PrefillScheduleTime; a decode sub-request sets DecodeScheduleTime; a normal
// (non-disaggregated) request is recorded in localAdmitTimes. OnAdmit can fire
// twice under preemption re-admit, so the first (earliest) time is kept.
func (cs *ClusterSimulator) recordAdmissionTime(req *sim.Request, tick int64) {
	if !cs.recordPDOutcomes {
		return
	}
	if req.IsDecodeSubRequest {
		if pid, ok := cs.pendingDecodeCompletions[req.ID]; ok {
			if p := cs.parentRequests[pid]; p != nil && p.DecodeScheduleTime == 0 {
				p.DecodeScheduleTime = tick
			}
		}
		return
	}
	if pid, ok := cs.pendingPrefillCompletions[req.ID]; ok {
		if p := cs.parentRequests[pid]; p != nil && p.PrefillScheduleTime == 0 {
			p.PrefillScheduleTime = tick
		}
		return
	}
	if _, seen := cs.localAdmitTimes[req.ID]; !seen {
		cs.localAdmitTimes[req.ID] = tick
	}
}

// feedFirstToken trues up an SLO-feedback decider's TTFT virtual queue when a request
// produces its first token (tick = absolute first-token time), keyed on whatever OnRoute
// registered — the parent ID for a PD request, the request ID otherwise (firstTokenKey).
//
// Resolving the PREFILL sub-request to its parent is load-bearing, not cosmetic. A PD
// request's first token is produced on the prefill side, so if that event does not reach
// the decider under the parent key the parent's awaiting-record is never trued up and never
// deleted: every later credit pass keeps adding (now − arrival − τ) for a request that has
// already finished, and the TTFT deficit queue integrates phantom lateness without bound.
// OnFirstToken is idempotent (a second call finds no record), so mapping both sub-request
// kinds to the parent is safe whichever one fires first.
func (cs *ClusterSimulator) feedFirstToken(req *sim.Request, tick int64) {
	if cs.sloFeedback == nil {
		return
	}
	cs.sloFeedback.OnFirstToken(cs.firstTokenKey(req), tick)
}

// firstTokenKey resolves req to the key OnRoute registered for the TTFT virtual queue.
// A decode sub-request resolves through edppConservationKey; a prefill sub-request is
// resolved here via the pending-prefill index, falling back to a scan of the parent index.
func (cs *ClusterSimulator) firstTokenKey(req *sim.Request) string {
	if k := cs.edppConservationKey(req); k != req.ID {
		return k // decode sub-request, already resolved to its parent
	}
	if pid, ok := cs.pendingPrefillCompletions[req.ID]; ok {
		return pid
	}
	for pid, parent := range cs.parentRequests {
		if parent != nil && parent.PrefillSubReqID == req.ID {
			return pid
		}
	}
	return req.ID
}

func (c *ClusterSimulator) detectPrefillCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	// Phase 1: collect completed sub-request IDs (sorted for determinism)
	var completedIDs []string
	for subReqID, parentID := range c.pendingPrefillCompletions {
		parent := c.parentRequests[parentID]
		if parent == nil || string(parent.PrefillInstanceID) != instID {
			continue
		}
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			completedIDs = append(completedIDs, subReqID)
		}
	}
	sort.Strings(completedIDs)

	// Phase 2: process in deterministic order
	for _, subReqID := range completedIDs {
		parentID := c.pendingPrefillCompletions[subReqID]
		parent := c.parentRequests[parentID]
		parent.PrefillCompleteTime = c.clock
		delete(c.pendingPrefillCompletions, subReqID)
		c.pdPrefillCompletedCount++

		// Schedule KV transfer
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &KVTransferStartedEvent{
				time:      c.clock,
				parentReq: parent,
			},
			seqID: c.nextSeqID(),
		})
	}
}

// releaseDecodeInFlight releases the decode-pod in-flight reservation taken at
// disaggregated routing (llm-d parity, see executeDisaggregatedRouting). Used on
// the paths where the decode sub-request is never injected onto the decode
// instance — so the per-instance OnRequestDone delta never decrements it: a
// transfer-start drop, a transfer-complete late drop, or a prefill timeout.
// Warn-and-clamp on negative mirrors the OnRequestDone decrement: inFlightRequests
// is a best-effort routing signal (INV-7), not a hard-conservation counter.
func (c *ClusterSimulator) releaseDecodeInFlight(decodeInstID string) {
	// Defensive: narrow event-unit tests construct ClusterSimulator as a struct
	// literal without inFlightRequests (NewClusterSimulator always makes it). With
	// no map there was no reservation to release.
	if c.inFlightRequests == nil {
		return
	}
	c.inFlightRequests[decodeInstID]--
	if c.inFlightRequests[decodeInstID] < 0 {
		logrus.Warnf("[cluster] inFlightRequests[%s] went negative releasing a decode reservation — bookkeeping bug; clamping to 0",
			decodeInstID)
		c.inFlightRequests[decodeInstID] = 0
	}
}

// detectPrefillTimeouts releases the decode-pod in-flight reservation for
// disaggregated requests whose prefill sub-request timed out on this prefill
// instance. Such a request never reaches KVTransferStartedEvent (no decode
// sub-request is ever created or injected), so without this its decode
// reservation — taken at routing in executeDisaggregatedRouting — would leak for
// the rest of the run, permanently biasing the decode load signal away from that
// pod. The prefill timeout itself is already counted in the prefill instance's
// TimedOutRequests (decrementing the prefill-side inFlightRequests via the
// OnRequestDone delta); this only handles the decode-side reservation. The parent
// is removed from pendingPrefillCompletions so it is processed at most once.
// R2/INV-6: collect IDs into a sorted slice before processing for determinism.
func (c *ClusterSimulator) detectPrefillTimeouts(inst *InstanceSimulator) {
	instID := string(inst.ID())
	var timedOutSubReqIDs []string
	for subReqID, parentID := range c.pendingPrefillCompletions {
		parent := c.parentRequests[parentID]
		if parent == nil || string(parent.PrefillInstanceID) != instID {
			continue
		}
		if parent.PrefillSubReq != nil && parent.PrefillSubReq.State == sim.StateTimedOut {
			timedOutSubReqIDs = append(timedOutSubReqIDs, subReqID)
		}
	}
	sort.Strings(timedOutSubReqIDs)

	for _, subReqID := range timedOutSubReqIDs {
		parentID := c.pendingPrefillCompletions[subReqID]
		parent := c.parentRequests[parentID]
		c.releaseDecodeInFlight(string(parent.DecodeInstanceID))
		delete(c.pendingPrefillCompletions, subReqID)
		// Conservation cleanup: the parent never completes, so release any backlog
		// its OnRoute added (mirrors the decode-timeout path in detectDecodeCompletions).
		if c.sloFeedback != nil {
			c.sloFeedback.Forget(parentID)
		}
	}
}

// detectDecodeCompletions checks for newly completed or timed-out decode sub-requests
// on the given instance and sets the parent request's CompletionTime.
// R2/INV-6: Collects IDs into sorted slices before processing for determinism.
func (c *ClusterSimulator) detectDecodeCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	// Phase 1: collect completed and timed-out sub-request IDs (sorted for determinism)
	var completedIDs []string
	var timedOutIDs []string
	for subReqID, parentID := range c.pendingDecodeCompletions {
		parent := c.parentRequests[parentID]
		if parent == nil || string(parent.DecodeInstanceID) != instID {
			continue
		}
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			completedIDs = append(completedIDs, subReqID)
		} else if parent.DecodeSubReq != nil && parent.DecodeSubReq.State == sim.StateTimedOut {
			timedOutIDs = append(timedOutIDs, subReqID)
		}
	}
	sort.Strings(completedIDs)
	sort.Strings(timedOutIDs)

	// Phase 2: process completions in deterministic order
	for _, subReqID := range completedIDs {
		parent := c.parentRequests[c.pendingDecodeCompletions[subReqID]]
		// Include PostDecodeFixedOverhead so parent.CompletionTime represents the
		// client-visible completion time, matching non-PD E2E semantics (issue #846).
		// For roofline (overhead=0), value is byte-identical to before.
		// No zero-output guard needed: decode sub-requests always carry the full
		// output token list from the original request (set in KVTransferCompletedEvent.Execute).
		parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeCompletedCount++

		// Issue #884: trigger session follow-up for the original (parent) request.
		// The per-instance OnRequestDone fires for the decode sub-request (no
		// SessionID), so SessionManager never sees PD completions. We call
		// sessionCallback directly with the original request to generate follow-ups.
		if c.sessionCallback != nil {
			// Value copy to avoid mutating the shared *sim.Request pointer
			// (contract at ParentRequests: callers must not mutate via OriginalRequest).
			origCopy := *parent.OriginalRequest
			origCopy.State = sim.StateCompleted
			// Use the decode sub-request's actual ProgressIndex for accurate context
			// accumulation (session.go:163). For length-capped decode sub-requests
			// (BC-5 force-completion), MaxOutputLen overstates the actual output;
			// DecodeSubReq.ProgressIndex reflects the true final position.
			// (blis replay passes onRequestDone=nil, so this code never runs in replay mode.)
			origCopy.ProgressIndex = parent.DecodeSubReq.ProgressIndex
			nextReqs := c.sessionCallback(&origCopy, parent.CompletionTime)
			for _, next := range nextReqs {
				c.pushArrival(next, next.ArrivalTime)
			}
		}
	}

	// Phase 3: process timed-out decode sub-requests (INV-11 session completeness).
	// Non-PD equivalent: TimeoutEvent.Execute calls OnRequestDone with StateTimedOut →
	// SessionManager cancels the session. The PD path needs the same treatment.
	for _, subReqID := range timedOutIDs {
		parentID := c.pendingDecodeCompletions[subReqID]
		parent := c.parentRequests[parentID]
		parent.CompletionTime = c.clock
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeTimedOutCount++

		// Conservation cleanup (Defect 2): a timed-out decode sub-request never reaches
		// a normal completion, so release the backlog the parent's OnRoute added. No z
		// bump / N̂_out update — there is no realized SLO signal. Keyed by parent ID.
		if c.sloFeedback != nil {
			c.sloFeedback.Forget(parentID)
		}

		if c.sessionCallback != nil {
			origCopy := *parent.OriginalRequest
			origCopy.State = sim.StateTimedOut
			origCopy.ProgressIndex = parent.DecodeSubReq.ProgressIndex
			// SessionManager.OnComplete cancels the session for StateTimedOut (session.go:112).
			// No follow-ups expected, but handle defensively.
			nextReqs := c.sessionCallback(&origCopy, parent.CompletionTime)
			for _, next := range nextReqs {
				c.pushArrival(next, next.ArrivalTime)
			}
		}
	}
}

// Clock returns the cluster's current simulation clock.
func (c *ClusterSimulator) Clock() int64 {
	return c.clock
}

// Instances returns the slice of InstanceSimulators.
func (c *ClusterSimulator) Instances() []*InstanceSimulator {
	return c.instances
}

// AggregatedMetrics returns the merged metrics across all instances.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) AggregatedMetrics() *sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.AggregatedMetrics() called before Run()")
	}
	return c.aggregatedMetrics
}

// RejectedRequests returns the count of requests rejected by the admission policy (EC-2).
// Returns 0 if AlwaysAdmit is used or if no requests were rejected by TokenBucket.
func (c *ClusterSimulator) RejectedRequests() int {
	return c.rejectedRequests
}

// RoutingRejections returns the count of requests rejected at routing because no
// routable instances were available (I13). Distinct from admission rejections.
func (c *ClusterSimulator) RoutingRejections() int {
	return c.routingRejections
}

// EncodeRoutingRejections returns the count of requests rejected at the encode
// routing stage because the encode pool has zero routable instances (GAP-4,
// issue #1264). Always zero when --encode-instances 0.
func (c *ClusterSimulator) EncodeRoutingRejections() int {
	return c.encodeRoutingRejections
}

// ShedByTier returns a copy of per-SLOClass rejection counts recorded during admission.
// Populated unconditionally for every admission rejection, regardless of policy.
// Returns a defensive copy so callers cannot mutate the internal counter (R8).
// Panics if called before Run() completes.
func (c *ClusterSimulator) ShedByTier() map[string]int {
	if !c.hasRun {
		panic("ClusterSimulator.ShedByTier() called before Run()")
	}
	result := make(map[string]int, len(c.shedByTier))
	for k, v := range c.shedByTier {
		result[k] = v
	}
	return result
}

// InjectedByClass returns a defensive copy of the per-SLOClass arrival counter.
// Incremented in ClusterArrivalEvent.Execute before any drop/route/admission
// decision; used as the goodput denominator (issue #1409, BC-5/BC-6).
// Panics if called before Run() completes (R8 mirror of ShedByTier).
func (c *ClusterSimulator) InjectedByClass() map[string]int64 {
	if !c.hasRun {
		panic("ClusterSimulator.InjectedByClass() called before Run()")
	}
	result := make(map[string]int64, len(c.injectedByClass))
	for k, v := range c.injectedByClass {
		result[k] = v
	}
	return result
}

// gpuInventory computes the current GPU inventory for Engine.Optimize().
// Phase 1C (T012): returns free GPU slots per VariantSpec.
//
// Free slots for a variant = total GPUs of that GPU type on Ready nodes
//   - GPUs held by Loading instances of that GPU type
//   - GPUs held by Active/WarmingUp instances of that GPU type
//   - GPUs held by Draining instances of that GPU type (hold GPUs until drain completes)
//
// Pending (Scheduling) instances are NOT subtracted.
// Terminated instances are NOT subtracted.
//
// Returns an empty inventory when cs.placement is nil (no NodePools configured, backward-compat).
func (c *ClusterSimulator) gpuInventory() GPUInventory {
	if c.placement == nil {
		return GPUInventory{byVariant: make(map[VariantSpec]int)}
	}

	// Step 1: count total GPUs on Ready nodes per GPUType.
	totalByGPUType := make(map[string]int)
	for _, node := range c.placement.nodesByID {
		if node.State == NodeStateReady {
			totalByGPUType[node.GPUType] += node.TotalGPUs
		}
	}

	// Step 2: subtract GPUs used by Loading, Active (incl. WarmingUp), and Draining instances.
	// Also populate seenVariants so every GPU type with Ready capacity appears in the result,
	// even when there are no active instances of that GPU type (enables scale-up from zero).
	clusterTPDegree := c.config.TP
	if clusterTPDegree < 1 {
		clusterTPDegree = 1
	}
	usedByGPUType := make(map[string]int)
	seenVariants := make(map[VariantSpec]struct{})
	// Seed from Ready node GPU types so zero-instance pools appear in inventory.
	for gpuType, total := range totalByGPUType {
		if total > 0 {
			seenVariants[VariantSpec{GPUType: gpuType, TPDegree: clusterTPDegree}] = struct{}{}
		}
	}
	for _, inst := range c.instances {
		switch inst.State {
		case sim.InstanceStateLoading, sim.InstanceStateWarmingUp, sim.InstanceStateActive, sim.InstanceStateDraining:
			if inst.GPU() != "" {
				usedByGPUType[inst.GPU()] += inst.TPDegree
				if inst.TPDegree > 0 {
					seenVariants[VariantSpec{GPUType: inst.GPU(), TPDegree: inst.TPDegree}] = struct{}{}
				}
			}
		}
	}

	// Step 3: build byVariant — same raw free count for each variant of the same GPUType.
	// Callers must use Variants() to iterate (R2: map iteration is non-deterministic).
	byVariant := make(map[VariantSpec]int, len(seenVariants))
	for v := range seenVariants {
		free := totalByGPUType[v.GPUType] - usedByGPUType[v.GPUType]
		if free < 0 {
			logrus.Warnf("[autoscaler] gpuInventory: variant %+v has negative free slots (%d) — bookkeeping inconsistency; clamping to 0", v, free)
			free = 0
		}
		byVariant[v] = free
	}
	return GPUInventory{byVariant: byVariant}
}

// GatewayQueueDepth returns the number of requests still in the gateway queue
// at simulation end. Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueDepth() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.Len()
}

// GatewayQueueShed returns the number of requests shed (evicted victims) from the gateway queue
// due to capacity limits. Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueShed() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.ShedCount()
}

// GatewayQueueRejected returns the number of requests rejected from the gateway queue
// (queue full, incoming could not displace any entry). Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueRejected() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.RejectedCount()
}

// GatewayEvicted returns the number of requests evicted in-flight from instances
// due to gateway-level eviction (INV-1: gw_evicted bucket).
func (c *ClusterSimulator) GatewayEvicted() int {
	return c.gatewayEvicted
}

// GatewayExpired returns the number of requests expired from the gateway queue
// via TTL (INV-1: gw_expired bucket).
func (c *ClusterSimulator) GatewayExpired() int {
	return c.gatewayExpired
}

// tryDispatchFromGatewayQueue attempts to dispatch one request from the gateway queue.
// Called on enqueue and by the periodic GatewayDispatchTickEvent (llm-d parity).
// Builds fresh RouterState at dispatch time for late binding (BC-3).
// Returns true if a request was dispatched, false if saturated or queue empty.
func (c *ClusterSimulator) tryDispatchFromGatewayQueue() bool {
	if c.gatewayQueue == nil || c.gatewayQueue.Len() == 0 {
		return false
	}
	// Schedule periodic dispatch tick if not already pending (BC-3).
	// Demand-driven: only active while queue is non-empty.
	if c.dispatchTickInterval > 0 && !c.dispatchTickPending {
		c.dispatchTickPending = true
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &GatewayDispatchTickEvent{
				At:       c.clock + c.dispatchTickInterval,
				Interval: c.dispatchTickInterval,
			},
			seqID: c.nextSeqID(),
		})
	}
	// Build fresh state for late binding (BC-3)
	state := buildRouterState(c, nil)
	sat := c.saturationDetector.Saturation(state)
	if math.IsNaN(sat) || math.IsInf(sat, 0) {
		panic(fmt.Sprintf("tryDispatchFromGatewayQueue: saturation=%f is not finite — detector bug", sat))
	}

	// Eviction trigger (BC-1): if saturated and a non-sheddable request is waiting,
	// evict one sheddable in-flight request to free capacity.
	if sat >= 1.0 && c.evictionTracker != nil && c.evictionTracker.Len() > 0 {
		if c.gatewayQueue.HasNonSheddableWaiting() {
			c.tryEvictOne()
			return false
		}
	}

	// Per-band HoL blocking: DequeueGated checks saturation against per-band ceilings
	// and halts dispatch if any band's ceiling is exceeded (GIE parity).
	req := c.gatewayQueue.DequeueGated(sat)
	if req == nil {
		logrus.Debugf("[cluster] tryDispatch: held (saturation=%.2f, snapshots=%d, queueLen=%d)",
			sat, len(state.Snapshots), c.gatewayQueue.Len())
		return false
	}
	req.GatewayDispatchTime = c.clock

	// Schedule routing (BC-9). RoutingDecisionEvent.Execute branches internally on
	// cs.poolsConfigured() for the disaggregated vs standard path — no fork here.
	heap.Push(&c.clusterEvents, clusterEventEntry{
		event: &RoutingDecisionEvent{
			time:    c.clock + c.routingLatency,
			request: req,
		},
		seqID: c.nextSeqID(),
	})
	return true
}

// tryEvictOne pops the most-evictable request and schedules its termination.
func (c *ClusterSimulator) tryEvictOne() {
	victim, instanceID := c.evictionTracker.Pop()
	if victim == nil {
		return
	}
	heap.Push(&c.clusterEvents, clusterEventEntry{
		event: &GatewayEvictionEvent{
			time:           c.clock,
			request:        victim,
			targetInstance: instanceID,
		},
		seqID: c.nextSeqID(),
	})
}

// Trace returns the decision trace collected during simulation.
// Returns nil if trace-level was "none" (default).
func (c *ClusterSimulator) Trace() *trace.SimulationTrace {
	return c.trace
}

// PerInstanceMetrics returns the metrics for each individual instance.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) PerInstanceMetrics() []*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetrics() called before Run()")
	}
	metrics := make([]*sim.Metrics, len(c.instances))
	for i, inst := range c.instances {
		metrics[i] = inst.Metrics()
	}
	return metrics
}

// PerInstanceMetricsByID returns a map of instance ID → *sim.Metrics.
// Panics if called before Run() completes (R1).
// The returned map is a new map (R8), but the *sim.Metrics values are live pointers to
// instance-owned structs — callers must not mutate fields through them.
func (c *ClusterSimulator) PerInstanceMetricsByID() map[string]*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetricsByID() called before Run()")
	}
	result := make(map[string]*sim.Metrics, len(c.instances))
	for _, inst := range c.instances {
		result[string(inst.ID())] = inst.Metrics()
	}
	return result
}

// PeakConcurrentTransfers returns the maximum number of KV transfers in flight simultaneously.
// Returns 0 when --pd-transfer-contention is disabled (backward-compat).
func (c *ClusterSimulator) PeakConcurrentTransfers() int {
	return c.peakConcurrentTransfers
}

// MeanTransferQueueDepth returns the mean number of active concurrent transfers sampled at each
// transfer initiation event (arrival-weighted mean, not a time-average). Specifically:
//
//	sum(activeTransfers at each start event) / count(start events)
//
// The activeTransfers count is taken post-increment, so it includes the initiating transfer
// itself. For example, with fully sequential transfers the mean is exactly 1.0.
//
// This is not equivalent to a time-averaged queue depth (Little's Law denominator); it measures
// how many transfers were in flight at the moment each new transfer began, including the new one.
// Returns 0 when --pd-transfer-contention is disabled or no transfers occurred.
func (c *ClusterSimulator) MeanTransferQueueDepth() float64 {
	if c.transferStartCount == 0 {
		return 0
	}
	return float64(c.transferDepthSum) / float64(c.transferStartCount)
}

// mergeFloat64Map merges src into dst, logging a warning on duplicate keys.
func mergeFloat64Map(dst, src map[string]float64, mapName string) {
	for k, v := range src {
		if _, exists := dst[k]; exists {
			logrus.Warnf("aggregateMetrics: duplicate request ID %q in %s", k, mapName)
		}
		dst[k] = v
	}
}

// mergeInt64Map merges src into dst, logging a warning on duplicate keys.
func mergeInt64Map(dst, src map[string]int64, mapName string) {
	for k, v := range src {
		if _, exists := dst[k]; exists {
			logrus.Warnf("aggregateMetrics: duplicate request ID %q in %s", k, mapName)
		}
		dst[k] = v
	}
}

func (c *ClusterSimulator) aggregateMetrics() *sim.Metrics {
	merged := sim.NewMetrics()
	for _, inst := range c.instances {
		m := inst.Metrics()
		merged.CompletedRequests += m.CompletedRequests
		merged.TotalInputTokens += m.TotalInputTokens
		merged.TotalOutputTokens += m.TotalOutputTokens
		merged.TTFTSum += m.TTFTSum
		merged.ITLSum += m.ITLSum
		if m.SimEndedTime > merged.SimEndedTime {
			merged.SimEndedTime = m.SimEndedTime
		}
		merged.KVBlocksUsed += m.KVBlocksUsed
		if m.PeakKVBlocksUsed > merged.PeakKVBlocksUsed {
			merged.PeakKVBlocksUsed = m.PeakKVBlocksUsed
		}
		merged.NumWaitQRequests = append(merged.NumWaitQRequests, m.NumWaitQRequests...)
		merged.NumRunningBatchRequests = append(merged.NumRunningBatchRequests, m.NumRunningBatchRequests...)

		// Merge per-request maps. IDs are globally unique (centrally generated as "request_N").
		// Duplicate IDs indicate a workload generation bug.
		mergeFloat64Map(merged.RequestTTFTs, m.RequestTTFTs, "RequestTTFTs")
		mergeFloat64Map(merged.RequestE2Es, m.RequestE2Es, "RequestE2Es")
		mergeFloat64Map(merged.RequestITLs, m.RequestITLs, "RequestITLs")
		mergeInt64Map(merged.RequestSchedulingDelays, m.RequestSchedulingDelays, "RequestSchedulingDelays")
		mergeFloat64Map(merged.RequestCompletionTimes, m.RequestCompletionTimes, "RequestCompletionTimes")

		for k, v := range m.Requests {
			if _, exists := merged.Requests[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in Requests", k)
			}
			merged.Requests[k] = v
		}
		merged.AllITLs = append(merged.AllITLs, m.AllITLs...)
		merged.RequestStepCounters = append(merged.RequestStepCounters, m.RequestStepCounters...)
		merged.PreemptionCount += m.PreemptionCount
		merged.KVAllocationFailures += m.KVAllocationFailures
		merged.DroppedUnservable += m.DroppedUnservable
		merged.LengthCappedRequests += m.LengthCappedRequests
		merged.TimedOutRequests += m.TimedOutRequests
		merged.CacheHitRate += m.CacheHitRate
		merged.KVThrashingRate += m.KVThrashingRate
		merged.StillQueued += m.StillQueued
		merged.StillRunning += m.StillRunning
	}
	if n := len(c.instances); n > 0 {
		merged.CacheHitRate /= float64(n)
		merged.KVThrashingRate /= float64(n)
	}

	// T042: apply warm-up TTFT factor to requests served during warm-up (Phase 1A, R23).
	// C4 (known simplification): The penalty is applied post-hoc to recorded TTFTs rather than
	// during token generation. This means scheduling decisions during warm-up don't see inflated
	// TTFTs. Acceptable for Phase 1A; a pre-hoc model would require latency model integration.
	// Applied uniformly across all TTFT recording paths.
	// warmUpRequestIDs is cleared unconditionally to prevent unbounded memory growth,
	// even when factor <= 1.0 (e.g., default config where effectiveWarmUpFactor returns 1.0).
	factor := c.config.InstanceLifecycle.effectiveWarmUpFactor()
	for _, inst := range c.instances {
		if factor > 1.0 {
			for _, reqID := range inst.WarmUpRequestIDs() {
				if ttft, ok := merged.RequestTTFTs[reqID]; ok {
					// Guard against propagating corrupt TTFT values (R3, R11)
					if !math.IsNaN(ttft) && !math.IsInf(ttft, 0) {
						newTTFT := ttft * factor
						// I34: Guard against Inf from large factor * large TTFT
						if math.IsInf(newTTFT, 0) {
							continue
						}
						// I1: Keep TTFTSum consistent with per-request TTFT adjustments.
						// Convert the TTFT delta (microseconds) to int64 ticks for TTFTSum.
						merged.TTFTSum += int64(newTTFT - ttft)
						merged.RequestTTFTs[reqID] = newTTFT
					}
				}
			}
		}
		inst.clearWarmUpRequestIDs()
	}

	return merged
}

// projectPDMetrics replaces sub-request entries in per-request metric maps
// with parent-level entries. For each ParentRequest:
//   - Completed parents (CompletionTime > 0, DecodeInstanceID != ""):
//     sub-request entries are replaced with parent-keyed entries using
//     true user-facing values (e.g., E2E = CompletionTime - ArrivalTime).
//   - Incomplete/dropped parents: sub-request entries are removed
//     (these requests did not complete successfully).
//
// This is a no-op when disaggregation is not active (parentRequests is empty).
func (c *ClusterSimulator) projectPDMetrics() {
	if len(c.parentRequests) == 0 {
		return
	}
	m := c.aggregatedMetrics

	for _, parent := range c.parentRequests {
		pfx := parent.PrefillSubReqID // "req_N_prefill"
		dec := parent.DecodeSubReqID  // "req_N_decode"
		pid := parent.ID              // "req_N"
		completed := parent.CompletionTime > 0 && parent.DecodeInstanceID != ""

		// E2E = parent.CompletionTime - parent.ArrivalTime
		// (arrival → prefill → transfer → decode → completion).
		delete(m.RequestE2Es, pfx)
		delete(m.RequestE2Es, dec)
		if completed {
			e2e := parent.CompletionTime - parent.ArrivalTime
			if e2e < 0 {
				// INV-3/INV-5 violation: completion before arrival. Should never occur
				// after the clusterTime fix in EnqueueDecodeSubRequest.
				logrus.Errorf("[cluster] projectPDMetrics: negative E2E for %s (completionTime=%d arrivalTime=%d); skipping",
					pid, parent.CompletionTime, parent.ArrivalTime)
			} else {
				m.RequestE2Es[pid] = float64(e2e)
			}
		}

		// TTFT: user-visible time-to-first-token for PD disaggregation.
		// In llm-d, the first token reaches the user from the decode pod, not
		// prefill: prefill completes → KV transfers → decode pod recomputes last
		// prompt token and samples first output token. User-visible TTFT =
		// prefillTTFT + transferDuration + firstDecodeStep. See issue #930.
		//
		// Read prefill TTFT before deleting sub-request keys (R1: no silent data loss).
		// Gate on completed: dropped-request TTFTs must not enter the distribution.
		prefillTTFT, hasPrefillTTFT := m.RequestTTFTs[pfx]
		delete(m.RequestTTFTs, pfx)
		delete(m.RequestTTFTs, dec)
		if completed {
			if hasPrefillTTFT && parent.TransferStartTime > 0 && parent.TransferCompleteTime >= parent.TransferStartTime && parent.DecodeSubReq != nil && len(parent.DecodeSubReq.ITL) > 0 {
				transferDuration := float64(parent.TransferCompleteTime - parent.TransferStartTime)
				firstDecodeStep := float64(parent.DecodeSubReq.ITL[0])
				newTTFT := prefillTTFT + transferDuration + firstDecodeStep
				m.RequestTTFTs[pid] = newTTFT
				// BC-3: Keep TTFTSum consistent with the TTFT adjustment.
				m.TTFTSum += int64(newTTFT - prefillTTFT)
			} else if hasPrefillTTFT {
				// Defensive fallback: use prefill-only TTFT if decode data unavailable.
				m.RequestTTFTs[pid] = prefillTTFT
				logrus.Warnf("[cluster] projectPDMetrics: parent %s missing decode ITL or TransferCompleteTime; using prefill TTFT", pid)
			} else {
				logrus.Warnf("[cluster] projectPDMetrics: completed parent %s has no prefill TTFT (key %s)", pid, pfx)
			}
		}

		// Scheduling delay = prefill sub-request's delay
		// (the real user-facing delay, not the decode pipeline cumulative latency).
		prefillDelay, hasPrefillDelay := m.RequestSchedulingDelays[pfx]
		delete(m.RequestSchedulingDelays, pfx)
		delete(m.RequestSchedulingDelays, dec)
		if completed && hasPrefillDelay {
			m.RequestSchedulingDelays[pid] = prefillDelay
		}

		// Requests metadata keyed by parent ID, HandledBy set to decode instance.
		delete(m.Requests, pfx)
		delete(m.Requests, dec)
		if completed {
			if parent.OriginalRequest == nil {
				panic(fmt.Sprintf("projectPDMetrics: parent %s has nil OriginalRequest", pid))
			}
			rm := sim.NewRequestMetrics(parent.OriginalRequest, float64(parent.ArrivalTime)/1e6)
			rm.HandledBy = string(parent.DecodeInstanceID)
			m.Requests[pid] = rm
		}

		// ITL from decode sub-request (prefill ITL is 0 noise).
		decodeITL, hasDecodeITL := m.RequestITLs[dec]
		delete(m.RequestITLs, pfx)
		delete(m.RequestITLs, dec)
		if completed && hasDecodeITL {
			m.RequestITLs[pid] = decodeITL
		}

		// Completion time from parent lifecycle tracking.
		delete(m.RequestCompletionTimes, pfx)
		delete(m.RequestCompletionTimes, dec)
		if completed {
			m.RequestCompletionTimes[pid] = float64(parent.CompletionTime)
		}
	}
}

// routingTraceOn reports whether per-candidate routing-decision tracing
// (--routing-decision-trace) is active.
func (cs *ClusterSimulator) routingTraceOn() bool {
	return cs.trace != nil && cs.trace.Config.RecordRoutingDecisions
}

// recordRoutingDecisionTrace captures one target selection's full candidate set
// for the --routing-decision-trace CSV. chosenID is taken from decision.TargetInstance
// (so a post-Route decode-pod override is reflected). No-op unless routingTraceOn().
func (cs *ClusterSimulator) recordRoutingDecisionTrace(stage, reqID string, decision sim.RoutingDecision, snapshots []sim.RoutingSnapshot) {
	cands, regret := buildRoutingTraceCandidates(decision.TargetInstance, decision, snapshots)
	cs.trace.RecordRoutingDecision(trace.RoutingDecisionTraceRecord{
		Clock:          cs.clock,
		Stage:          stage,
		RequestID:      reqID,
		ChosenInstance: decision.TargetInstance,
		Regret:         regret,
		Candidates:     cands,
	})
}

// executeStandardRouting performs non-disaggregated routing: select a target over
// all routable instances, record the decision, increment in-flight/tenant counters,
// record warm-up, and inject the request into the target instance. Used when pool
// topology is not configured (plain DES routing). Called by RoutingDecisionEvent.Execute.
//
// Parameter `time` is the scheduled event time (already advanced by routingLatency
// from admission). Injection happens at `time` — no additional offset.
func (cs *ClusterSimulator) executeStandardRouting(req *sim.Request, time int64) {
	state := buildRouterState(cs, req)

	// Guard: if no routable instances are available (e.g., all model-M instances are Loading
	// or Draining), routing policies panic on empty snapshot sets. Treat as rejection instead.
	// Uses Warn so users understand why requests are dropping (visible at default log level).
	// I13: Use routingRejections counter to distinguish from admission rejections.
	if len(state.Snapshots) == 0 {
		logrus.Warnf("[cluster] req %s: no routable instances for model %q — request rejected at routing (all instances may be Loading or Draining)", req.ID, req.Model)
		cs.routingRejections++
		return
	}

	state.CaptureScorerBreakdown = cs.routingTraceOn()
	decision := cs.routingPolicy.Route(req, state)
	logrus.Debugf("[cluster] req %s → instance %s (reason=%s)", req.ID, decision.TargetInstance, decision.Reason)

	if cs.routingTraceOn() {
		cs.recordRoutingDecisionTrace("standard", req.ID, decision, state.Snapshots)
	}

	// #181: Stamp request with assigned instance for per-request metrics
	req.AssignedInstance = decision.TargetInstance

	// Record routing decision if tracing is enabled (BC-3, BC-4, BC-5, BC-6)
	if cs.trace != nil {
		record := trace.RoutingRecord{
			RequestID:      req.ID,
			Clock:          cs.clock,
			ChosenInstance: decision.TargetInstance,
			Reason:         decision.Reason,
			Scores:         copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			record.Candidates, record.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				state.Snapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordRouting(record)
	}

	// Find target instance, increment in-flight count, and inject request
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			// Increment in-flight AFTER target validation — gives next routing decision
			// visibility into this routing decision (#170)
			cs.inFlightRequests[decision.TargetInstance]++
			// Phase 1B-2a: track tenant in-flight count for fair-share enforcement.
			if cs.tenantTracker != nil {
				cs.tenantTracker.OnStart(req.TenantID)
			}

			// T042: record warm-up requests for TTFT factor application (Phase 1A).
			warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
			if warmUpCount > 0 && len(inst.WarmUpRequestIDs()) < warmUpCount {
				inst.RecordWarmUpRequest(req.ID)
			}

			inst.InjectRequestOnline(req, time)
			// Track routed sheddable requests for in-flight eviction (BC-3).
			if cs.evictionTracker != nil {
				cs.evictionTracker.Track(req, decision.TargetInstance, cs.priorityMap)
			}
			return
		}
	}

	// Should never reach here (policy contract ensures valid target)
	panic(fmt.Sprintf("executeStandardRouting: invalid TargetInstance %q", decision.TargetInstance))
}

// executeDisaggregatedRouting performs PD disaggregation routing: select a decode pod
// first (llm-d parity), then decide whether to disaggregate. If disaggregate=false,
// inject directly to the selected decode pod. If disaggregate=true, store the decode
// pod in a ParentRequest and schedule a PrefillRoutingEvent.
//
// Parameter `time` is the scheduled event time (already advanced by routingLatency
// from admission). Both the non-disaggregated injection and the PrefillRoutingEvent
// fire at `time` — no additional offset.
func (cs *ClusterSimulator) executeDisaggregatedRouting(req *sim.Request, time int64) {
	// Step 1: route to decode pool first (llm-d parity: decode pod always selected first).
	filteredSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(filteredSnapshots) == 0 {
		logrus.Warnf("[cluster] req %s: no routable instances in decode pool — request rejected at routing", req.ID)
		cs.routingRejections++
		return
	}
	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock, CaptureScorerBreakdown: cs.routingTraceOn()}
	policy := cs.decodeRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decodeDecision := policy.Route(req, state)
	logrus.Debugf("[cluster] req %s: decode pod pre-selected → %s", req.ID, decodeDecision.TargetInstance)

	// Step 2: disaggregation decision with decode pod known. Pass the full decode-pool
	// RouterState so the decider can query per-pod state (cache presence, load) and
	// optionally reconsider the decode pod via DisaggregationDecision.DecodePodOverride.
	// state.SelectedInstance tells the decider which snapshot was pre-selected by
	// the decode routing policy — PrefixThresholdDecider queries the selected
	// pod's cacheQueryFn closure for per-pod prefix cache state (matches llm-d's
	// PrefixBasedPDDecider reading endpoint.Get(PrefixCacheMatchInfoKey)).
	state.SelectedInstance = decodeDecision.TargetInstance
	disaggDecision := cs.disaggregationDecider.Decide(req, state)
	logrus.Debugf("[cluster] req %s: disaggregate=%v", req.ID, disaggDecision.Disaggregate)

	// Stage C: snapshot the assembled per-pool AdmissionContext(s) so BuildAdmissionRecords
	// can recompute all six estimator predictions at end of run (--edpp-admission-trace).
	// hasPrefill is true iff this request disaggregates (a prefill row is emitted then).
	if cs.recordAdmissionTrace && disaggDecision.AdmissionCtxDecode != nil {
		cap := &capturedAdmission{decodeCtx: *disaggDecision.AdmissionCtxDecode, hasPrefill: disaggDecision.Disaggregate}
		if disaggDecision.AdmissionCtxPrefill != nil {
			cap.prefillCtx = *disaggDecision.AdmissionCtxPrefill
		}
		cs.admissionCtx[req.ID] = cap
	}

	// If the decider overrode the decode pod (joint D+P policies), retarget.
	// Empty string = keep the pod pre-selected by the decode routing policy.
	// The override must be a member of the decode-pool snapshot set; the downstream
	// instance lookup panics otherwise (see decodeInst == nil guard below).
	if disaggDecision.DecodePodOverride != "" {
		decodeDecision.TargetInstance = disaggDecision.DecodePodOverride
	}

	// Record the decode target selection for --routing-decision-trace (after any
	// decider override, so the chosen instance reflects the final decode pod).
	if cs.routingTraceOn() {
		cs.recordRoutingDecisionTrace("decode", req.ID, decodeDecision, state.Snapshots)
	}

	// Record disaggregation decision if tracing is enabled (BC-PD-17).
	if cs.trace != nil {
		cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
			RequestID:    req.ID,
			Clock:        cs.clock,
			Disaggregate: disaggDecision.Disaggregate,
		})
		// EDPP rule-term trace: present only when the EDPP decider has tracing enabled.
		if et := disaggDecision.EDPPTrace; et != nil {
			cs.trace.RecordEDPPDecision(trace.EDPPDecisionRecord{
				RequestID: req.ID, Clock: cs.clock,
				Class: et.Class, SkipReason: et.SkipReason,
				Ap: et.Ap, Wp: et.Wp, DeltaPfChunk: et.DeltaPfChunk,
				QdRaw: et.QdRaw, QpRaw: et.QpRaw, Qd: et.Qd, Qp: et.Qp,
				MuDNom: et.MuDNom, MuPNom: et.MuPNom, WStarD: et.WStarD, WStarP: et.WStarP,
				TauTTFT: et.TauTTFT, TauITL: et.TauITL,
				TTFTP: et.TTFTP, TTFTD: et.TTFTD, ITLP: et.ITLP, ITLD: et.ITLD,
				ZTTFT: et.ZTTFT, ZITL: et.ZITL,
				BalanceTermD: et.BalanceTermD, BalanceTermP: et.BalanceTermP,
				TransferTerm: et.TransferTerm, TTFTTerm: et.TTFTTerm, ITLTerm: et.ITLTerm,
				LHS: et.LHS, RHS: et.RHS, Disaggregate: et.Disaggregate,
			})
		}
		// Joint scorer-vs-joint divergence trace: present only under --edpp-joint-trace.
		if jt := disaggDecision.EDPPJointTrace; jt != nil {
			cs.trace.RecordEDPPJointDecision(trace.EDPPJointDecisionRecord{
				RequestID: req.ID, Clock: cs.clock,
				Class:   jt.Class,
				ScorerD: jt.ScorerD, JointD: jt.JointD,
				ScorerP: jt.ScorerP, JointP: jt.JointP,
				AgreeD: jt.AgreeD, AgreeP: jt.AgreeP,
				JScorer: jt.JScorer, JJoint: jt.JJoint,
				Disaggregate: jt.Disaggregate,
			})
		}
	}

	// Fire OnRoute exactly once at the routing-commit point. This is the single live
	// OnRoute call: both the D-path and P-path below are terminal dispatch points, so
	// one call here guarantees exactly-once semantics. The conservation key is the
	// original request ID (req.ID); for a disaggregated request the completing decode
	// sub-request has a different ID (req.ID+"_decode"), so the completion side must
	// correlate back to req.ID — see feedSLOFeedback / edppConservationKey.
	if cs.sloFeedback != nil {
		ap := len(req.InputTokens) // uncached-prompt upper bound; INV-9 safe
		// The decode instance is pre-selected here (decode-first routing). For the reduced
		// path the prefill instance is chosen later by PrefillRoutingEvent, so PrefillPodHint
		// is empty and per-instance prefill attribution is deferred (pool scalars unaffected).
		// For the joint path (--edpp-joint), the decider has already committed the argmin's
		// prefill node p* in PrefillPodHint, so pass it here to populate the per-instance q_p.
		cs.sloFeedback.OnRoute(req, req.ID, disaggDecision.Disaggregate, ap, decodeDecision.TargetInstance, disaggDecision.PrefillPodHint)
	}

	// Find the target decode instance object (used in both paths below).
	var decodeInst *InstanceSimulator
	for _, inst := range cs.instances {
		if string(inst.ID()) == decodeDecision.TargetInstance {
			decodeInst = inst
			break
		}
	}
	if decodeInst == nil {
		// The routing policy contract requires Route to return a TargetInstance from the
		// provided snapshot set; a panic here indicates a policy implementation bug.
		panic(fmt.Sprintf("executeDisaggregatedRouting: invalid decode TargetInstance %q returned by routing policy", decodeDecision.TargetInstance))
	}

	// Encode stage (GAP-4, issue #1264). Synchronous under option A
	// (zero-duration): we make a routing decision on the encode pool, record
	// a trace entry, and carry the chosen encode instance ID forward to the
	// parent (for the disagg path) or discard it after trace recording (for
	// the non-disagg path). No encode sub-request is injected into an instance.
	// No-op when cs.encodeDecider == nil (the default when --encode-instances 0),
	// preserving byte-for-byte pre-PR behavior (BC-EPD-1).
	var encodeInstanceID string
	if cs.encodeDecider != nil && cs.encodeDecider.ShouldEncode(req, decodeDecision.TargetInstance) {
		encodeSnapshots := cs.buildPoolFilteredSnapshots(PoolRoleEncode)
		if len(encodeSnapshots) == 0 {
			logrus.Warnf("[cluster] req %s: no routable instances in encode pool — request rejected at encode routing", req.ID)
			cs.encodeRoutingRejections++
			return
		}
		encodeState := &sim.RouterState{Snapshots: encodeSnapshots, Clock: cs.clock, CaptureScorerBreakdown: cs.routingTraceOn()}
		// Encode routing uses the main routingPolicy in this PR; per-pool scorer
		// config is a follow-up (design doc D6).
		encodeDecision := cs.routingPolicy.Route(req, encodeState)
		encodeInstanceID = encodeDecision.TargetInstance
		logrus.Debugf("[cluster] req %s: encode pod selected → %s", req.ID, encodeInstanceID)

		if cs.routingTraceOn() {
			cs.recordRoutingDecisionTrace("encode", req.ID, encodeDecision, encodeState.Snapshots)
		}

		if cs.trace != nil {
			record := trace.EncodeRoutingRecord{
				ParentRequestID: req.ID,
				Clock:           cs.clock,
				ChosenInstance:  encodeInstanceID,
				Scores:          copyScores(encodeDecision.Scores),
			}
			if cs.trace.Config.CounterfactualK > 0 {
				record.Candidates, record.Regret = computeCounterfactual(
					encodeInstanceID, encodeDecision.Scores,
					encodeSnapshots, cs.trace.Config.CounterfactualK,
				)
			}
			cs.trace.RecordEncodeRouting(record)
		}
	}

	if !disaggDecision.Disaggregate {
		// Step 3a: local path — inject directly to the selected decode pod.
		// Non-disaggregated requests route exclusively to the decode pool, not to all
		// instances via buildRouterState().
		req.AssignedInstance = decodeDecision.TargetInstance

		// Record standard routing trace for BC-TRACE-COMPAT: consumers expect
		// len(tr.Routings) == numRequests.
		if cs.trace != nil {
			record := trace.RoutingRecord{
				RequestID:      req.ID,
				Clock:          cs.clock,
				ChosenInstance: decodeDecision.TargetInstance,
				Reason:         decodeDecision.Reason,
				Scores:         copyScores(decodeDecision.Scores),
			}
			if cs.trace.Config.CounterfactualK > 0 {
				record.Candidates, record.Regret = computeCounterfactual(
					decodeDecision.TargetInstance, decodeDecision.Scores,
					filteredSnapshots, cs.trace.Config.CounterfactualK,
				)
			}
			cs.trace.RecordRouting(record)
		}

		cs.inFlightRequests[decodeDecision.TargetInstance]++
		if cs.tenantTracker != nil {
			cs.tenantTracker.OnStart(req.TenantID)
		}
		warmUpCount := cs.config.InstanceLifecycle.WarmUpRequestCount
		if warmUpCount > 0 && len(decodeInst.WarmUpRequestIDs()) < warmUpCount {
			decodeInst.RecordWarmUpRequest(req.ID)
		}
		// The local (non-disaggregated) enqueue instant, so local_t_adm =
		// local_schedule − local_enqueue. Needed by BOTH the --edpp-admission-trace
		// (Stage C) and the --pd-outcome-trace (Stage A) harnesses; OnAdmit later records
		// the schedule instant into localAdmitTimes.
		if cs.recordAdmissionTrace || cs.recordPDOutcomes {
			if _, seen := cs.localEnqueueTimes[req.ID]; !seen {
				cs.localEnqueueTimes[req.ID] = time
			}
		}
		decodeInst.InjectRequestOnline(req, time)
		if cs.evictionTracker != nil {
			cs.evictionTracker.Track(req, decodeDecision.TargetInstance, cs.priorityMap)
		}
		return
	}

	// Step 3b: disaggregated path — decode pod pre-selected, route prefill next.
	parent := NewParentRequest(req, cs.config.BlockSizeTokens)
	parent.DecodeInstanceID = InstanceID(decodeDecision.TargetInstance)
	parent.PrefillPodHint = disaggDecision.PrefillPodHint
	if encodeInstanceID != "" {
		parent.EncodeInstanceID = InstanceID(encodeInstanceID)
	}
	cs.parentRequests[parent.ID] = parent

	// Reserve the decode pod's in-flight load signal NOW, at selection — llm-d
	// parity (EPP InFlightLoadProducer.PreRequest increments the decode endpoint's
	// requestTracker synchronously right after Schedule(), holding it through the
	// whole prefill+transfer+decode window). The local (non-disaggregated) path
	// increments inFlightRequests at selection too (Step 3a above); the
	// disaggregated path used to defer the decode increment to
	// KVTransferCompletedEvent (after prefill+transfer), leaving the decode load
	// signal blind for the entire transfer window so a burst of disaggregated
	// requests all saw every decode pod as empty. This reservation is released
	// exactly once: normally by the decode sub-request's completion via the
	// per-instance OnRequestDone delta (the decode sub-request is injected on this
	// instance at KVTransferCompletedEvent), or — for paths where the decode
	// sub-request is never injected — explicitly at the transfer-start drop, the
	// transfer-complete late drop, or a prefill timeout (see pd_events.go and
	// detectPrefillTimeouts).
	cs.inFlightRequests[string(parent.DecodeInstanceID)]++

	// Create prefill sub-request: same input, no output (completes after prefill).
	// InputTokens is a slice-header alias of req.InputTokens (#1445) — the
	// sub-request views the same underlying token buffer, no flatten. If
	// Request.InputTokens ever becomes lazy/chained, this site must update.
	prefillSubReq := &sim.Request{
		ID:           parent.PrefillSubReqID,
		InputTokens:  req.InputTokens,
		MaxOutputLen: req.MaxOutputLen,
		Deadline:     req.Deadline,
		PrefixGroup:  req.PrefixGroup,
		State:        sim.StateQueued,
		ArrivalTime:  req.ArrivalTime,
		TenantID:     req.TenantID,
		SLOClass:     req.SLOClass,
		Model:        req.Model,
	}
	// Retain the prefill sub-request so a prefill timeout can release the decode
	// reservation above (the decode sub-request is never created in that case).
	parent.PrefillSubReq = prefillSubReq

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &PrefillRoutingEvent{
			time:      time,
			request:   prefillSubReq,
			parentReq: parent,
		},
		seqID: cs.nextSeqID(),
	})
}

// capturedAdmission holds the per-pool AdmissionContext(s) the EDPP decider assembled
// for one request at routing-decision time, so BuildAdmissionRecords can recompute every
// estimator's prediction at end of run against the same inputs the live decision saw.
// hasPrefill is true only for a disaggregated request (a prefill row is emitted then).
type capturedAdmission struct {
	decodeCtx  sim.AdmissionContext
	prefillCtx sim.AdmissionContext
	hasPrefill bool
}

// EnableAdmissionTrace turns on per-request admission-context capture for the
// --edpp-admission-trace companion trace (Stage C). It (a) flips SetAdmissionDetail(true)
// on every instance so RunningDecode/TrueRemaining/DispatchRate populate (oracle mode —
// logging-only, INV-9); (b) reuses the Stage A per-request time correlation
// (SetRecordPDOutcomes) to compute realized t_adm; and (c) tells the EDPP decider to
// attach its assembled AdmissionContext(s) to each decision. Zero-cost when never called.
func (cs *ClusterSimulator) EnableAdmissionTrace(coeffs sim.EDPPCoeffs) {
	cs.recordAdmissionTrace = true
	cs.admissionCoeffs = coeffs
	cs.admissionCtx = make(map[string]*capturedAdmission)
	if cs.localEnqueueTimes == nil {
		cs.localEnqueueTimes = make(map[string]int64)
	}
	// Realized t_adm reuses Stage A's parent/local schedule/enqueue capture.
	cs.SetRecordPDOutcomes(true)
	// Oracle admission detail: populate RunningDecode/TrueRemaining/DispatchRate. INV-9:
	// oracle predictions are logged only; the Task 6 guard prevents oracle-as-router-driver.
	for _, inst := range cs.instances {
		inst.sim.SetAdmissionDetail(true)
	}
	if d, ok := cs.disaggregationDecider.(*sim.EDPPDecider); ok {
		d.SetCaptureAdmissionContext(true)
	}
}

// BuildAdmissionRecords assembles one realized-vs-predicted admission-delay record per
// pool-term for the --edpp-admission-trace companion trace. It walks disaggregated
// parents (prefill + decode rows) and locally-admitted requests (a local row), pairs
// each with the AdmissionContext captured at decision time, computes realized t_adm from
// the Stage A schedule/enqueue instants, runs all six estimators against the captured
// context, and returns rows sorted by request_id (INV-6). Records with no captured
// context are skipped (e.g. empty-prompt / fully-cached early returns in the decider).
func (cs *ClusterSimulator) BuildAdmissionRecords() []trace.AdmissionRecord {
	recs := make([]trace.AdmissionRecord, 0, len(cs.admissionCtx))
	tadm := func(enq, sched int64) float64 {
		if enq > 0 && sched >= enq {
			return float64(sched - enq)
		}
		return 0
	}
	// stripOracle returns a deployable copy of the captured context with every running
	// request's oracle TrueRemaining censored to -1, forcing the deployable estimators to
	// fall back to the N̂_out-based RemainingStepsEst (INV-9: deployable path never sees
	// oracle remaining). The Running slice is deep-copied so the original/captured context
	// — used unchanged for the _oracle variants — is not mutated.
	stripOracle := func(c sim.AdmissionContext) sim.AdmissionContext {
		rc := make([]sim.RunningReqState, len(c.Running))
		copy(rc, c.Running)
		for i := range rc {
			rc[i].TrueRemaining = -1
		}
		c.Running = rc
		return c
	}
	mk := func(id, pool string, realized float64, ctx sim.AdmissionContext) trace.AdmissionRecord {
		// INV-9 asymmetry: only decode Running carries oracle (o_r-derived) remaining, so
		// only decode/local rows are stripped for the deployable prediction. Prefill
		// remaining is known input (inLen − ProgressIndex) — deployable == oracle, no strip.
		deployable := ctx
		if pool != "prefill" {
			deployable = stripOracle(ctx)
		}
		p := func(name string, c sim.AdmissionContext) float64 {
			est, err := sim.NewAdmissionEstimator(name)
			if err != nil {
				return 0
			}
			return est.EstimateTAdm(c)
		}
		return trace.AdmissionRecord{
			RequestID: id, Pool: pool, RealizedTAdm: realized,
			TAdmPredWaiting:           p("waiting", deployable),
			TAdmPredLittle:            p("little", deployable),
			TAdmPredFluid:             p("fluid", deployable),
			TAdmPredRollforward:       p("rollforward", deployable),
			TAdmPredFluidOracle:       p("fluid_oracle", ctx),
			TAdmPredRollforwardOracle: p("rollforward_oracle", ctx),
		}
	}

	for id, cap := range cs.admissionCtx {
		if p, isParent := cs.parentRequests[id]; isParent && cap.hasPrefill {
			recs = append(recs, mk(id, "prefill", tadm(p.PrefillEnqueueTime, p.PrefillScheduleTime), cap.prefillCtx))
			recs = append(recs, mk(id, "decode", tadm(p.DecodeEnqueueTime, p.DecodeScheduleTime), cap.decodeCtx))
			continue
		}
		// Local (non-disaggregated) request: realized t_adm = local schedule − local enqueue.
		realized := tadm(cs.localEnqueueTimes[id], cs.localAdmitTimes[id])
		recs = append(recs, mk(id, "local", realized, cap.decodeCtx))
	}

	sort.Slice(recs, func(i, j int) bool {
		if recs[i].RequestID != recs[j].RequestID {
			return recs[i].RequestID < recs[j].RequestID
		}
		return recs[i].Pool < recs[j].Pool
	})
	return recs
}

// EnableWorkTrace turns on per-request work accumulation on every instance simulator.
func (cs *ClusterSimulator) EnableWorkTrace(coeffs sim.EDPPCoeffs) {
	cs.recordWorkTrace = true
	cs.workCoeffs = coeffs
	for _, inst := range cs.instances {
		inst.sim.SetWorkTrace(coeffs)
	}
}

// gatherWorkByInstance snapshots each instance's per-request work accumulators.
func (cs *ClusterSimulator) gatherWorkByInstance() map[string]map[string]sim.ReqWork {
	out := make(map[string]map[string]sim.ReqWork)
	for _, inst := range cs.instances {
		out[string(inst.id)] = inst.sim.WorkAccumulators()
	}
	return out
}

// BuildWorkTraceRecords gathers live instance accumulators and correlates
// prefill/decode sub-requests back to parents. Sorted by request_id (INV-6).
func (cs *ClusterSimulator) BuildWorkTraceRecords() []trace.WorkTraceRecord {
	return cs.buildWorkTraceRecordsFrom(cs.gatherWorkByInstance())
}

func (cs *ClusterSimulator) buildWorkTraceRecordsFrom(byInst map[string]map[string]sim.ReqWork) []trace.WorkTraceRecord {
	claimed := make(map[string]map[string]bool) // instanceID → reqID → claimed by a parent
	mark := func(inst, id string) {
		if claimed[inst] == nil {
			claimed[inst] = map[string]bool{}
		}
		claimed[inst][id] = true
	}
	get := func(inst, id string) (sim.ReqWork, bool) {
		m := byInst[inst]
		if m == nil {
			return sim.ReqWork{}, false
		}
		w, ok := m[id]
		return w, ok
	}
	mk := func(id, slo string, ar, ap, o int64, chunks int, pfWork, decWork float64) trace.WorkTraceRecord {
		chf := 0.0
		if ar > 0 {
			chf = 1.0 - float64(ap)/float64(ar)
		}
		return trace.WorkTraceRecord{
			RequestID: id, SLOClass: slo, Ar: ar, ApRealized: ap, ORealized: o,
			PrefillChunks: chunks, CacheHitFrac: chf,
			RealizedPrefillWork: pfWork, RealizedDecodeWork: decWork,
			WpClosed:           cs.workCoeffs.Wp(int(ap), int(ar)),
			WdClosed:           cs.workCoeffs.Wd(int(ar), float64(o)),
			WpClosedNoCacheOld: cs.workCoeffs.CPf*float64(ap) + (cs.workCoeffs.CAttn/2.0)*float64(ap)*float64(ap),
		}
	}

	recs := make([]trace.WorkTraceRecord, 0)
	for pid, p := range cs.parentRequests {
		pf, _ := get(string(p.PrefillInstanceID), p.PrefillSubReqID)
		dec, _ := get(string(p.DecodeInstanceID), p.DecodeSubReqID)
		mark(string(p.PrefillInstanceID), p.PrefillSubReqID)
		mark(string(p.DecodeInstanceID), p.DecodeSubReqID)
		slo, ar := "", int64(0)
		if p.OriginalRequest != nil {
			slo = p.OriginalRequest.SLOClass
			ar = util.Len64(p.OriginalRequest.InputTokens)
		}
		recs = append(recs, mk(pid, slo, ar, pf.ApRealized, dec.ORealized, pf.PrefillChunks, pf.RealizedPrefillWork, dec.RealizedDecodeWork))
	}
	for inst, m := range byInst {
		for id, w := range m {
			if claimed[inst][id] {
				continue
			}
			recs = append(recs, mk(id, w.SLOClass, w.Ar, w.ApRealized, w.ORealized, w.PrefillChunks, w.RealizedPrefillWork, w.RealizedDecodeWork))
		}
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].RequestID < recs[j].RequestID })
	return recs
}

// EDPPDeficitStats reports the SLO-deficit virtual-queue occupancy accumulated by the EDPP
// decider over this run, when EDPP is the active PD decider. The second return is false for
// every other decider. Pure instrumentation: it answers whether the rule's time-average SLO
// constraints ever bound, which the goodput metric alone cannot show.
func (cs *ClusterSimulator) EDPPDeficitStats() (sim.EDPPDeficitStats, bool) {
	if d, ok := cs.disaggregationDecider.(*sim.EDPPDecider); ok && d != nil {
		return d.DeficitQueueStats(), true
	}
	return sim.EDPPDeficitStats{}, false
}
