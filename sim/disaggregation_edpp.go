package sim

import "math"

// TTFTUpdater is an optional interface for disaggregation deciders that maintain
// internal state based on observed request TTFT (time-to-first-token).
// Implemented by EmpiricalDPPDecider to update the virtual TTFT queue Z after
// each completed disaggregated request.
type TTFTUpdater interface {
	UpdateTTFT(ttftUs float64)
}

// ObservationUpdater extends TTFTUpdater with additional feedback signals that
// EmpiricalDPPDecider needs to adapt its empirical parameters.
//
// Call sites in sim/cluster/cluster.go:
//
//  1. After KV transfer completes (same site as TTFTUpdater):
//     if updater, ok := c.disaggregationDecider.(sim.ObservationUpdater); ok {
//         updater.UpdateTransferObservation(
//             float64(parent.TransferCompleteTime - parent.TransferStartTime),
//             float64(parent.TransferStartTime - parent.ArrivalTime),
//         )
//     }
//
//  2. After every completed request (decode completion):
//     if updater, ok := c.disaggregationDecider.(sim.ObservationUpdater); ok {
//         updater.UpdateRequestCompletion(itlMeanUs)
//     }
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

// EmpiricalDPPDecider is a self-calibrating disaggregation policy that uses the
// Drift-Plus-Penalty structure without analytically-derived constants. All system
// parameters are estimated from observed request completions.
//
// Decision rule:
//
//	V · η · Q_D  >  Q_P  +  Z · κ̂
//
// Quantities:
//
//	Q_P, Q_D — aggregate prefill/decode queue depths at decision time (RouterState).
//	V        — adaptive penalty weight. Updated each epoch:
//	           V += α · (ITL_epoch_mean − ITLtarget) / ITLtarget
//	           When observed ITL exceeds the target, V rises (more disaggregation).
//	           When ITL is comfortable, V falls (less disaggregation, lower TTFT).
//	η        — fixed queue weight ratio (default 1.0; operator-specified).
//	Z        — virtual TTFT queue: Z(t+1) = max(0, Z(t) + TTFT_obs − d).
//	           Grows when disaggregated requests miss the TTFT SLO, suppressing
//	           future disaggregation. No additive constant drowns Z here.
//	κ̂       — empirical KV-transfer cost ratio: EWMA(ΔT_obs / W_P_approx).
//	           Starts at kappaInit; adapts from observed transfer durations.
//
// Operator-specified goals: ITLtarget (ms) and TTFT SLO d (ms). No system
// parameters (c_D, W_P) need to be provided.
type EmpiricalDPPDecider struct {
	// Policy knobs (operator-specified)
	eta         float64 // queue weight ratio η
	ttftSloUs   float64 // TTFT SLO target d (μs)
	itlTargetUs float64 // ITL target (μs); V adapts toward this
	vMin        float64 // lower bound on V
	vMax        float64 // upper bound on V
	alpha       float64 // V step size per epoch
	epochSize   int     // completed requests per V update
	kappaAlpha  float64 // EWMA smoothing factor for κ̂ (0 < kappaAlpha ≤ 1)

	// Adaptive state (updated from observations)
	v             float64 // current penalty weight V
	kappaHat      float64 // current empirical cost ratio κ̂
	virtualQueueZ float64 // virtual TTFT queue Z (μs)

	// Epoch accumulator for V update
	epochCount  int
	epochITLSum float64
}

// EmpiricalDPPConfig holds constructor parameters. Separate from CLI flag
// parsing to keep disaggregation_edpp.go test-friendly.
type EmpiricalDPPConfig struct {
	Eta         float64 // default 1.0
	TTFTSloMs   float64 // TTFT SLO in ms; converted to μs internally
	ITLTargetMs float64 // ITL target in ms; converted to μs internally
	VInit       float64 // initial V; default 1.0
	VMin        float64 // default 0.05
	VMax        float64 // default 50.0
	Alpha       float64 // default 0.1
	EpochSize   int     // default 50
	KappaInit   float64 // initial κ̂; default 0.114 (empirical ΔT/W_P from 2P+2D runs)
	KappaAlpha  float64 // EWMA factor; default 0.05
}

// NewEmpiricalDPPDecider creates an EmpiricalDPPDecider.
// Panics if Eta ≤ 0, ITLTargetMs ≤ 0, or EpochSize ≤ 0.
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
		kappaInit = 0.114
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
// Implements TTFTUpdater (and thus ObservationUpdater).
func (e *EmpiricalDPPDecider) UpdateTTFT(ttftUs float64) {
	e.virtualQueueZ = math.Max(0, e.virtualQueueZ+ttftUs-e.ttftSloUs)
}

// UpdateTransferObservation refines κ̂ from observed KV-transfer and prefill durations.
func (e *EmpiricalDPPDecider) UpdateTransferObservation(transferUs, prefillApproxUs float64) {
	if prefillApproxUs <= 0 {
		return
	}
	observed := transferUs / prefillApproxUs
	e.kappaHat = (1-e.kappaAlpha)*e.kappaHat + e.kappaAlpha*observed
}

// UpdateRequestCompletion records an ITL observation and updates V at each epoch boundary.
func (e *EmpiricalDPPDecider) UpdateRequestCompletion(itlMeanUs float64) {
	if itlMeanUs <= 0 {
		return
	}
	e.epochCount++
	e.epochITLSum += itlMeanUs
	if e.epochCount >= e.epochSize {
		epochMean := e.epochITLSum / float64(e.epochCount)
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
