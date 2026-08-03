// Package trace provides decision-trace recording for cluster-level policy analysis.
// This package has no dependencies on sim/ or sim/cluster/ — it stores pure data types.
package trace

// AdmissionDecisionRecord captures a single admission policy decision (admit/reject).
type AdmissionDecisionRecord struct {
	RequestID string
	Clock     int64
	Admitted  bool
	Reason    string
}

// AdmissionRecord is one request's realized admission delay vs the six admission-delay
// estimators' predictions, for the --edpp-admission-trace companion trace (Stage C).
// Pool is "prefill", "decode", or "local" per the term the row describes. RealizedTAdm
// and all predictions are microseconds. The oracle predictions (…Oracle) are computed
// against a true-remaining-populated context and are logging-only (INV-9). Emitted only
// under --edpp-admission-trace; rows are sorted by request_id (INV-6).
type AdmissionRecord struct {
	RequestID string
	Pool      string

	RealizedTAdm              float64
	TAdmPredWaiting           float64
	TAdmPredLittle            float64
	TAdmPredFluid             float64
	TAdmPredRollforward       float64
	TAdmPredFluidOracle       float64
	TAdmPredRollforwardOracle float64
}

// CandidateScore captures a counterfactual candidate instance with its score and state.
type CandidateScore struct {
	InstanceID       string
	Score            float64
	QueueDepth       int
	BatchSize        int
	InFlightRequests int
	KVUtilization    float64
	FreeKVBlocks     int64
}

// RoutingRecord captures a single routing policy decision with optional counterfactual analysis.
type RoutingRecord struct {
	RequestID      string
	Clock          int64
	ChosenInstance string
	Reason         string
	Scores         map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates     []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret         float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DisaggregationRecord captures a PD disaggregation decision.
// When Disaggregate=true, the request follows the disaggregated path. Paired
// PrefillRoutingRecord and KVTransferRecord are recorded for the same RequestID
// except in two drop scenarios:
//
//  1. No routable prefill pool instances (routingRejections++): no downstream records.
//  2. Decode-side drop (droppedAtDecodeKV++): PrefillRoutingRecord exists but
//     KVTransferRecord is absent. Two sub-cases, both counted the same:
//     - At KVTransferStartedEvent: decode pod became non-routable, or
//     ReserveTransferredKV fails (insufficient decode KV). Drop fires at
//     transfer start; no KVTransferCompletedEvent is scheduled. Issue #1343.
//     - At KVTransferCompletedEvent: decode pod became non-routable between
//     transfer start and transfer complete. Reserved KV is released.
//
// To detect case 1: check for the absence of a PrefillRoutingRecord with a
// matching ParentRequestID in the trace. To detect case 2: check the
// absence of a KVTransferRecord for a given ParentRequestID.
// Note: DecodeRoutingRecord is never emitted; the decode pod is pre-selected at
// executeDisaggregatedRouting time (no second routing decision).
type DisaggregationRecord struct {
	RequestID string
	// Clock is the simulation time at which the routing (and disaggregation)
	// decision was made: admission_time + routingLatency. Since #1261 unified
	// the PD and standard routing paths under RoutingDecisionEvent, this value
	// reflects the routing-fire time rather than the admission time.
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing to decode pool
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// EncodeRoutingRecord captures an encode pool routing decision (GAP-4, issue #1264).
// Emitted when an EncodeDecider approves encoding for a request during
// executeDisaggregatedRouting. Under option A (zero-duration encode), the record
// reflects a routing decision only — no encode sub-request is injected.
// ParentRequestID equals the request ID (one encode record per parent).
type EncodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Nil when scoring is disabled.
	Scores     map[string]float64
	Candidates []CandidateScore // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64          // max(alternative scores) - score(chosen); >= 0
}

// EDPPDecisionRecord captures the intermediate terms of one EDPP (E14) rule evaluation
// for a request, for diagnostic analysis of why the decider chose P or D. It is a flat
// mirror of sim.EDPPDecisionTrace (this package has no dependency on sim/), plus the
// request ID and the decision clock. Recorded only when the EDPP decider has tracing
// enabled and trace-level=decisions. The two sides compose exactly:
//
//	LHS = BalanceTermD − BalanceTermP
//	RHS = TransferTerm + TTFTTerm + ITLTerm + PrefillStabilityTerm
//	Disaggregate = LHS > RHS
//
// For the reduced VaR rules, the VaR totals expose the co-resident
// populations. SelfGoodLocal/SelfGoodDisagg expose the optional arriving-
// request reward behind:
// LHS = VarLocalTotal − VarDisaggTotal + SelfGoodDisagg − SelfGoodLocal.
//
// On early-return paths SkipReason names the path ("empty-prompt"/"fully-cached") and the
// term fields are zero.
type EDPPDecisionRecord struct {
	RequestID              string
	Clock                  int64
	Class                  string
	SkipReason             string
	Ap                     int
	Wp                     float64
	ApPrefill              int
	WpPrefill              float64
	DeltaPfChunk           float64
	QdRaw                  float64
	QpRaw                  float64
	Qd                     float64
	Qp                     float64
	MuDNom                 float64
	MuPNom                 float64
	WStarD                 float64
	WStarP                 float64
	TauTTFT                float64
	TauITL                 float64
	TTFTP                  float64
	TTFTD                  float64
	TAdmP                  float64
	TAdmD                  float64
	RemoteLead             float64
	LocalService           float64
	DisaggFirst            float64
	ITLP                   float64
	ITLD                   float64
	ZTTFT                  float64
	ZITL                   float64
	BalanceTermD           float64
	BalanceTermP           float64
	TransferTerm           float64
	TTFTTerm               float64
	ITLTerm                float64
	PrefillStabilityTerm   float64
	VarLocalDecode         float64
	VarLocalCollocPrefill  float64
	VarLocalTotal          float64
	VarDisaggDecode        float64
	VarDisaggCollocPrefill float64
	VarDisaggPrefillPool   float64
	VarDisaggTotal         float64
	SelfGoodLocal          float64
	SelfGoodDisagg         float64
	KairosMode             string
	KairosAlpha            float64
	KairosAlphaThreshold   float64
	KairosTTFTGateRequired bool
	KairosTTFTGatePassed   bool
	KairosResidentTauITL   float64
	KairosTBTBudget        float64
	KairosFirstChunk       float64
	KairosMinChunk         float64
	KairosChunkSteps       int
	LHS                    float64
	RHS                    float64
	Disaggregate           bool
}

// EDPPJointDecisionRecord captures the scorer-vs-joint divergence for one joint
// (--edpp-joint) routing decision: the decode-routing scorer's pick and the shadow
// prefill-scorer pick vs the joint argmin's (decode, prefill) nodes, plus the objective J
// at each. Flat mirror of sim.EDPPJointDecisionTrace (this package has no dependency on
// sim/), plus request ID and decision clock. Recorded only under --edpp-joint-trace.
// JJoint <= JScorer by construction (the argmin ranges over a superset of the scorer slice).
type EDPPJointDecisionRecord struct {
	RequestID    string
	Clock        int64
	Class        string
	ScorerD      string
	JointD       string
	ScorerP      string
	JointP       string
	AgreeD       bool
	AgreeP       bool
	JScorer      float64
	JJoint       float64
	Disaggregate bool
}

// EDPPJointCandidateRecord is one candidate action from a joint EDPP decision.
// There are D(P+1) rows per request: one local action per decode node and one
// disaggregated action per decode/prefill pair.
type EDPPJointCandidateRecord struct {
	RequestID, Class                            string
	Clock                                       int64
	DecodePod, PrefillPod                       string
	Local, Chosen, RouterDecode                 bool
	VarDecode, VarCollocPrefill                 float64
	VarPrefillPool, VarTotal, BestVar           float64
	ChosenVarRegret                             float64
	SLOExternality, OwnGood                     float64
	NetGoodCost                                 float64
	CapacityQueueDecode, CapacityQueuePrefill   float64
	CapacityDemandDecode, CapacityDemandPrefill float64
	CapacityDecode, CapacityPrefill             float64
	CapacityTotal, Score, BestScore             float64
	ChosenScoreRegret                           float64
}

// PDOutcomeRecord is one request's realized outcome for EDPP estimator validation
// (Stage A). Joined against EDPPDecisionRecord on RequestID. Times are microseconds,
// absolute; zero means the phase was not reached. Emitted only under --pd-outcome-trace.
type PDOutcomeRecord struct {
	RequestID       string
	SLOClass        string
	InputTokens     int
	Disaggregated   bool
	PrefillInstance string
	DecodeInstance  string

	PrefillEnqueue  int64
	PrefillSchedule int64
	PrefillTAdm     int64
	DecodeEnqueue   int64
	DecodeSchedule  int64
	DecodeTAdm      int64
	LocalEnqueue    int64
	LocalSchedule   int64
	LocalTAdm       int64

	RealizedTTFT    float64
	RealizedMeanITL float64
	RealizedE2E     float64
	Completed       bool
}

// WorkTraceRecord is one request's realized trajectory work vs the closed-form
// work model (Stage B validation). Times/work in µs. Emitted only under
// --edpp-work-trace. See docs/superpowers/specs/2026-07-01-edpp-work-model-design.md.
type WorkTraceRecord struct {
	RequestID     string
	SLOClass      string
	Ar            int64   // full prompt length len(InputTokens)
	ApRealized    int64   // Σ new prefill tokens actually processed (excludes cached prefix)
	ORealized     int64   // realized output length (decode steps)
	PrefillChunks int     // number of prefill steps (1 = single-chunk)
	CacheHitFrac  float64 // 1 - ApRealized/Ar

	RealizedPrefillWork float64 // Σ per-step prefill δ (active latency model basis)
	RealizedDecodeWork  float64 // Σ per-step decode δ

	WpClosed           float64 // Wp(ApRealized, Ar) — corrected closed form
	WdClosed           float64 // Wd(Ar, ORealized) — corrected closed form
	WpClosedNoCacheOld float64 // old shipped form C_pf·ApRealized + (C_attn/2)·ApRealized² (for delta reporting)
}

// RoutingTraceCandidate is one candidate instance considered during a routing
// target selection, captured for the --routing-decision-trace CSV.
type RoutingTraceCandidate struct {
	InstanceID     string
	IsChosen       bool
	CompositeScore float64            // weighted composite (RoutingDecision.Scores); 0 for non-scoring policies
	ScorerScores   map[string]float64 // scorer name → raw clamped [0,1] score; nil for non-scoring policies
	QueueDepth     int
	BatchSize      int
	// InFlightRequests is the dispatched-but-not-completed count as the router saw
	// it, INCLUDING decode targets reserved at selection but not yet transferred
	// (the in-flight reservation, commit 6a97a2f) — so reserved-pending decodes are
	// reflected here.
	InFlightRequests int
	KVUtilization    float64
	FreeKVBlocks     int64
}

// RoutingDecisionTraceRecord captures one routing target selection (prefill,
// decode, standard, or encode) with the full candidate set, for the
// --routing-decision-trace CSV. One record per selection; the CSV writer emits
// one row per candidate.
type RoutingDecisionTraceRecord struct {
	Clock          int64
	Stage          string // "standard" | "prefill" | "decode" | "encode"
	RequestID      string
	ChosenInstance string
	Regret         float64 // best composite − chosen composite (≥0); 0 for non-scoring policies
	Candidates     []RoutingTraceCandidate
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
// TransferDuration is always >= 0; negative values are clamped to 0 with a warning in
// KVTransferCompletedEvent.Execute() (sim/cluster/pd_events.go) if INV-PD-4 is ever violated.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock)
	TransferDuration  int64 // microseconds; >= 0 (clamped at recording site)
	NumKVBlocks       int64
	PrefillInstanceID string
	DecodeInstanceID  string
}
