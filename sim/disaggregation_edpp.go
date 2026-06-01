package sim

import "math"

// ObservationUpdater extends TTFTUpdater with the additional feedback signals
// that EmpiricalDPPDecider needs. Implemented by EmpiricalDPPDecider.
//
// Call sites in sim/cluster/cluster.go:
//
//   1. Same site as TTFTUpdater (~line 1259) — after KV transfer completes:
//        if updater, ok := c.disaggregationDecider.(sim.ObservationUpdater); ok {
//            updater.UpdateTransferObservation(
//                float64(parent.TransferCompleteTime - parent.TransferStartTime),
//                float64(parent.TransferStartTime - parent.ArrivalTime), // ≈ prefill time
//            )
//        }
//
//   2. After every completed request (decode completion, ~line 1251):
//        if updater, ok := c.disaggregationDecider.(sim.ObservationUpdater); ok {
//            // itlMeanUs ≈ total decode time / output token count
//            updater.UpdateRequestCompletion(itlMeanUs)
//        }
type ObservationUpdater interface {
	TTFTUpdater
	// UpdateTransferObservation is called once per completed disaggregated request.
	// transferUs: observed KV transfer duration (TransferCompleteTime - TransferStartTime).
	// prefillApproxUs: arrival-to-transfer-start duration, a proxy for prefill service time.
	UpdateTransferObservation(transferUs, prefillApproxUs float64)
	// UpdateRequestCompletion is called once per completed request of any kind.
	// itlMeanUs: mean inter-token latency for this request's decode phase.
	UpdateRequestCompletion(itlMeanUs float64)
}

// EmpiricalDPPDecider is a self-calibrating drift-plus-penalty disaggregation
// policy. It preserves the DPP structure but eliminates all analytically-derived
// constants, replacing them with empirically estimated quantities updated from
// observed request completions.
//
// Decision rule:
//
//	V · η · Q_D  >  Q_P  +  Z · κ̂
//
// Quantities:
//
//	Q_P, Q_D — aggregate prefill/decode queue depths at decision time (from RouterState).
//	V        — adaptive penalty weight; increases when observed ITL > itlTarget,
//	           decreases when ITL is comfortable. Bounded to [vMin, vMax].
//	η        — fixed queue weight (default 1.0; operator-specified).
//	Z        — virtual TTFT queue: Z(t+1) = max(0, Z(t) + TTFT_obs − d).
//	           Z grows when disaggregated requests miss the TTFT SLO, suppressing
//	           future disaggregation. Unlike DriftPlusPenaltyDecider, there is no
//	           additive constant V·c_D/2 to drown Z, so Z can actually swing decisions.
//	κ̂       — empirical KV-transfer cost ratio: EWMA(ΔT_obs / W_P_approx).
//	           Starts at kappaInit; adapts from observed transfer and prefill durations.
//	           Represents "how many prefill-service-time units does one KV transfer cost."
//
// V update (per epoch of epochSize completed requests):
//
//	V(t+1) = clip( V(t) + alpha · (ITL_epoch_mean − itlTarget) / itlTarget, vMin, vMax )
//
// This drives V toward the value where observed ITL matches the operator's target.
// When ITL is too high, V rises (more disaggregation to offload prefill).
// When ITL is already good, V falls (less disaggregation, TTFT improves).
type EmpiricalDPPDecider struct {
	// --- Operator-specified policy knobs ---
	eta         float64 // queue weight ratio η
	ttftSloUs   float64 // TTFT SLO target d (μs)
	itlTargetUs float64 // ITL target (μs); V adapts to track this
	vMin        float64 // lower bound on V (prevents collapse to NeverDisaggregate)
	vMax        float64 // upper bound on V (prevents runaway to AlwaysDisaggregate)
	alpha       float64 // V step size per epoch (as a fraction of relative ITL error)
	epochSize   int     // number of request completions per V update
	kappaAlpha  float64 // EWMA smoothing factor for κ̂ updates (0 < kappaAlpha ≤ 1)

	// --- Adaptive state (updated from observations) ---
	v             float64 // current penalty weight V
	kappaHat      float64 // current empirical cost ratio κ̂
	virtualQueueZ float64 // virtual TTFT queue Z (μs accumulated)

	// --- Epoch accumulator for V update ---
	epochCount  int
	epochITLSum float64
}

// EmpiricalDPPConfig holds constructor parameters for EmpiricalDPPDecider.
// Separate from CLI flag parsing to keep disaggregation.go test-friendly.
type EmpiricalDPPConfig struct {
	Eta         float64 // default 1.0
	TTFTSloMs   float64 // TTFT SLO in ms; converted to μs internally
	ITLTargetMs float64 // ITL target in ms; converted to μs internally
	VInit       float64 // initial V; default 1.0
	VMin        float64 // default 0.05
	VMax        float64 // default 50.0
	Alpha       float64 // default 0.1
	EpochSize   int     // default 50
	KappaInit   float64 // initial κ̂; default 0.114 (empirical from 2P+2D campaign)
	KappaAlpha  float64 // EWMA factor; default 0.05
}

// NewEmpiricalDPPDecider creates an EmpiricalDPPDecider from config.
// Panics if Eta ≤ 0, ITLTargetMs ≤ 0, EpochSize ≤ 0.
func NewEmpiricalDPPDecider(cfg EmpiricalDPPConfig) *EmpiricalDPPDecider {
	if cfg.Eta <= 0 {
		panic("EmpiricalDPPDecider: Eta must be > 0")
	}
	if cfg.ITLTargetMs <= 0 {
		panic("EmpiricalDPPDecider: ITLTargetMs must be > 0")
	}
	if cfg.EpochSize <= 0 {
		panic("EmpiricalDPPDecider: EpochSize must be > 0")
	}
	vInit := cfg.VInit
	if vInit <= 0 {
		vInit = 1.0
	}
	kappaInit := cfg.KappaInit
	if kappaInit <= 0 {
		kappaInit = 0.114 // empirical ΔT/W_P from trained-physics qwen3-14b 2P+2D runs
	}
	kappaAlpha := cfg.KappaAlpha
	if kappaAlpha <= 0 {
		kappaAlpha = 0.05
	}
	return &EmpiricalDPPDecider{
		eta:         cfg.Eta,
		ttftSloUs:   cfg.TTFTSloMs * 1000,
		itlTargetUs: cfg.ITLTargetMs * 1000,
		vMin:        cfg.VMin,
		vMax:        cfg.VMax,
		alpha:       cfg.Alpha,
		epochSize:   cfg.EpochSize,
		kappaHat:    kappaInit,
		kappaAlpha:  kappaAlpha,
		v:           vInit,
	}
}

// Decide disaggregates iff V·η·Q_D > Q_P + Z·κ̂.
// Handles nil state (Q_P = Q_D = 0 → never disaggregate at zero load).
func (e *EmpiricalDPPDecider) Decide(_ *Request, state *RouterState) DisaggregationDecision {
	var qD, qP float64
	if state != nil {
		for _, s := range state.Snapshots {
			qD += float64(s.QueueDepth)
		}
		for _, s := range state.PrefillSnapshots {
			qP += float64(s.QueueDepth)
		}
	}
	lhs := e.v * e.eta * qD
	rhs := qP + e.virtualQueueZ*e.kappaHat
	return DisaggregationDecision{Disaggregate: lhs > rhs}
}

// UpdateTTFT updates Z after a disaggregated request completes.
// Z accumulates TTFT excess over the SLO target.
// Implements TTFTUpdater (and thus ObservationUpdater).
func (e *EmpiricalDPPDecider) UpdateTTFT(ttftUs float64) {
	e.virtualQueueZ = math.Max(0, e.virtualQueueZ+ttftUs-e.ttftSloUs)
}

// UpdateTransferObservation refines κ̂ from observed KV-transfer and prefill durations.
// Called once per completed disaggregated request.
// transferUs:     TransferCompleteTime − TransferStartTime (actual KV transfer duration)
// prefillApproxUs: TransferStartTime − ArrivalTime (includes queue wait; upper bound on W_P)
func (e *EmpiricalDPPDecider) UpdateTransferObservation(transferUs, prefillApproxUs float64) {
	if prefillApproxUs <= 0 {
		return
	}
	observed := transferUs / prefillApproxUs
	// EWMA update: κ̂ ← (1−α)·κ̂ + α·observed
	e.kappaHat = (1-e.kappaAlpha)*e.kappaHat + e.kappaAlpha*observed
}

// UpdateRequestCompletion records an ITL observation and, at each epoch boundary,
// updates V toward the ITL target.
// itlMeanUs: mean inter-token latency for this request (μs).
func (e *EmpiricalDPPDecider) UpdateRequestCompletion(itlMeanUs float64) {
	if itlMeanUs <= 0 {
		return
	}
	e.epochCount++
	e.epochITLSum += itlMeanUs

	if e.epochCount >= e.epochSize {
		epochMean := e.epochITLSum / float64(e.epochCount)
		// Relative ITL error: positive → observed too high → increase V (disaggregate more)
		relError := (epochMean - e.itlTargetUs) / e.itlTargetUs
		e.v = math.Max(e.vMin, math.Min(e.vMax, e.v+e.alpha*relError))
		e.epochCount = 0
		e.epochITLSum = 0
	}
}

// CurrentV exposes the current adaptive V for metrics/logging.
func (e *EmpiricalDPPDecider) CurrentV() float64 { return e.v }

// CurrentKappa exposes the current empirical κ̂ for metrics/logging.
func (e *EmpiricalDPPDecider) CurrentKappa() float64 { return e.kappaHat }

// CurrentZ exposes the current virtual TTFT queue Z (μs) for metrics/logging.
func (e *EmpiricalDPPDecider) CurrentZ() float64 { return e.virtualQueueZ }

// Compile-time checks.
var (
	_ DisaggregationDecider = (*EmpiricalDPPDecider)(nil)
	_ ObservationUpdater    = (*EmpiricalDPPDecider)(nil)
)
