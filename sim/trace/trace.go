package trace

// TraceLevel controls the verbosity of decision tracing.
type TraceLevel string

const (
	// TraceLevelNone disables tracing (zero overhead).
	TraceLevelNone TraceLevel = "none"
	// TraceLevelDecisions captures all admission and routing policy decisions.
	TraceLevelDecisions TraceLevel = "decisions"
)

// validTraceLevels maps accepted trace level strings.
var validTraceLevels = map[TraceLevel]bool{
	TraceLevelNone:      true,
	TraceLevelDecisions: true,
	"":                  true, // empty defaults to none
}

// IsValidTraceLevel returns true if the given level string is a recognized trace level.
func IsValidTraceLevel(level string) bool {
	return validTraceLevels[TraceLevel(level)]
}

// TraceConfig controls trace collection behavior.
type TraceConfig struct {
	Level           TraceLevel
	CounterfactualK int // number of counterfactual candidates per routing decision
	// RecordRoutingDecisions enables full per-candidate routing-decision capture
	// (every prefill/decode/standard target selection, all candidates, per-scorer
	// scores) for the --routing-decision-trace CSV. Independent of CounterfactualK.
	RecordRoutingDecisions bool
}

// SimulationTrace collects decision records during a cluster simulation.
type SimulationTrace struct {
	Config          TraceConfig
	Admissions      []AdmissionDecisionRecord
	Routings        []RoutingRecord
	Disaggregations []DisaggregationRecord
	PrefillRoutings []PrefillRoutingRecord
	DecodeRoutings  []DecodeRoutingRecord
	EncodeRoutings  []EncodeRoutingRecord // GAP-4 (issue #1264)
	KVTransfers     []KVTransferRecord
	EDPPDecisions   []EDPPDecisionRecord // EDPP rule-term traces (when EDPP tracing enabled)
	// EDPPJointDecisions holds scorer-vs-joint divergence records (when --edpp-joint-trace set).
	EDPPJointDecisions  []EDPPJointDecisionRecord
	EDPPJointCandidates []EDPPJointCandidateRecord
	// RoutingDecisions holds per-candidate routing-decision traces (every
	// prefill/decode/standard target selection) when Config.RecordRoutingDecisions.
	RoutingDecisions []RoutingDecisionTraceRecord
}

// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:              config,
		Admissions:          make([]AdmissionDecisionRecord, 0),
		Routings:            make([]RoutingRecord, 0),
		Disaggregations:     make([]DisaggregationRecord, 0),
		PrefillRoutings:     make([]PrefillRoutingRecord, 0),
		DecodeRoutings:      make([]DecodeRoutingRecord, 0),
		EncodeRoutings:      make([]EncodeRoutingRecord, 0),
		KVTransfers:         make([]KVTransferRecord, 0),
		EDPPDecisions:       make([]EDPPDecisionRecord, 0),
		EDPPJointDecisions:  make([]EDPPJointDecisionRecord, 0),
		EDPPJointCandidates: make([]EDPPJointCandidateRecord, 0),
		RoutingDecisions:    make([]RoutingDecisionTraceRecord, 0),
	}
}

// RecordAdmission appends an admission decision record.
func (st *SimulationTrace) RecordAdmission(record AdmissionDecisionRecord) {
	st.Admissions = append(st.Admissions, record)
}

// RecordRouting appends a routing decision record.
func (st *SimulationTrace) RecordRouting(record RoutingRecord) {
	st.Routings = append(st.Routings, record)
}

// RecordDisaggregation appends a disaggregation decision record.
func (st *SimulationTrace) RecordDisaggregation(record DisaggregationRecord) {
	st.Disaggregations = append(st.Disaggregations, record)
}

// RecordPrefillRouting appends a prefill pool routing decision record.
func (st *SimulationTrace) RecordPrefillRouting(record PrefillRoutingRecord) {
	st.PrefillRoutings = append(st.PrefillRoutings, record)
}

// RecordDecodeRouting appends a decode pool routing decision record.
func (st *SimulationTrace) RecordDecodeRouting(record DecodeRoutingRecord) {
	st.DecodeRoutings = append(st.DecodeRoutings, record)
}

// RecordKVTransfer appends a KV transfer event record.
func (st *SimulationTrace) RecordKVTransfer(record KVTransferRecord) {
	st.KVTransfers = append(st.KVTransfers, record)
}

// RecordEncodeRouting appends an encode pool routing decision record (GAP-4, #1264).
func (st *SimulationTrace) RecordEncodeRouting(record EncodeRoutingRecord) {
	st.EncodeRoutings = append(st.EncodeRoutings, record)
}

// RecordEDPPDecision appends an EDPP rule-term trace record.
func (st *SimulationTrace) RecordEDPPDecision(record EDPPDecisionRecord) {
	st.EDPPDecisions = append(st.EDPPDecisions, record)
}

// RecordEDPPJointDecision appends a scorer-vs-joint divergence trace record.
func (st *SimulationTrace) RecordEDPPJointDecision(record EDPPJointDecisionRecord) {
	st.EDPPJointDecisions = append(st.EDPPJointDecisions, record)
}

func (st *SimulationTrace) RecordEDPPJointCandidate(record EDPPJointCandidateRecord) {
	st.EDPPJointCandidates = append(st.EDPPJointCandidates, record)
}

// RecordRoutingDecision appends a per-candidate routing-decision trace record.
func (st *SimulationTrace) RecordRoutingDecision(record RoutingDecisionTraceRecord) {
	st.RoutingDecisions = append(st.RoutingDecisions, record)
}
