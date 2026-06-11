package sim

import (
	"fmt"
	"math"
	"math/rand"
)

// CompletionObserver is implemented by disaggregation deciders that learn from
// observed request completions. The cluster simulator calls ObserveCompletion
// once per completed request — for BOTH local (non-disaggregated) and remote
// (disaggregated) requests — so the decider can maintain two-population
// empirical estimates and a virtual TTFT queue.
//
// Call sites in sim/cluster/cluster.go:
//
//  1. Remote (disaggregated) completions — detectDecodeCompletions:
//     obs.ObserveCompletion(true,
//     float64(parent.TransferCompleteTime-parent.ArrivalTime),   // ttftUs
//     c.instanceMeanITL(decodeSubReqID),                          // itlMeanUs
//     float64(parent.PrefillCompleteTime-parent.PrefillEnqueueTime), // prefillTimeUs
//     parent.OriginalRequest.DisaggUncachedTokens)
//
//  2. Local (non-disaggregated) completions — the OnRequestDone closure:
//     obs.ObserveCompletion(false,
//     req.FirstTokenTime,        // ttftUs   (TTFT proxy, arrival-relative)
//     meanITL,                   // itlMeanUs
//     req.FirstTokenTime,        // prefillTimeUs (BLIS TTFT proxy — see note below)
//     req.DisaggUncachedTokens)
type CompletionObserver interface {
	// ObserveCompletion records a completed request.
	//   disaggregated:  true if the request was routed to the dedicated prefill pool (REMOTE).
	//   ttftUs:         observed time-to-first-token, microseconds.
	//   itlMeanUs:      mean inter-token latency for the decode phase, microseconds (≤0 = skip ITL update).
	//   prefillTimeUs:  prefill-phase duration, microseconds (≤0 = skip rate update).
	//   uncachedTokens: uncached input-token count used by the disaggregation decision.
	ObserveCompletion(disaggregated bool, ttftUs, itlMeanUs, prefillTimeUs float64, uncachedTokens int)
}

// ewma is a cold-started exponentially-weighted moving average. The first
// observation sets the value directly (no drag from a 0 init); subsequent
// observations blend with smoothing factor beta. Until the first observation,
// value() returns 0 — which is exactly what the EDPP decision rule needs to be
// "drift-free" while the populations are still empty.
type ewma struct {
	v    float64
	seen bool
}

func (e *ewma) update(x, beta float64) {
	if !e.seen {
		e.v = x
		e.seen = true
		return
	}
	e.v = (1-beta)*e.v + beta*x
}

func (e ewma) value() float64 { return e.v }

// EmpiricalDPPDecider is a self-calibrating P/D disaggregation policy derived
// from Lyapunov drift-plus-penalty. Every system parameter is learned from
// observed request completions; there are no analytically-derived physics
// constants.
//
// Decision rule (work-units, microseconds throughout):
//
//	disaggregate ⟺  u·(Q_D·rate_D − Q_P·rate_P) + V·Δp  >  Z·ΔTTFT
//
// where, for the request being decided:
//
//	u       — uncached input tokens (same signal PrefixThresholdDecider uses).
//	Q_D     — Σ over decode snapshots of QueueDepth·AvgOutTokens·ITL  (decode backlog, µs).
//	Q_P     — Σ over prefill snapshots of QueueDepth·AvgInTokens·rate_P (prefill backlog, µs).
//	rate_P  — learned prefill rate on the dedicated prefill pool (µs / uncached token), REMOTE pop.
//	rate_D  — learned prefill rate on the mixed decode pool        (µs / uncached token), LOCAL pop.
//	Δp      — p_local − p_remote: two-population ITL difference (the mixing-penalty signal).
//	ΔTTFT   — ttft_remote − ttft_local: two-population TTFT difference.
//	V       — fixed operator knob trading ITL (Δp) against the drift term. NOT adapted.
//	Z       — virtual TTFT queue: Z ← max(0, Z + TTFT_obs − d), updated on EVERY request.
//	d       — TTFT SLO target (µs).
//
// The drift term is rate-weighted because prefill work is resource-dependent:
// mixed prefill on a decode server (rate_D) is slower than dedicated prefill on
// a prefill server (rate_P), so the Q_D/Q_P backlogs do not collapse to a
// symmetric w_P·(Q_D−Q_P). See memory [[edpp-clean-derivation]] for the full
// Lyapunov derivation — implemented here verbatim.
//
// All EWMA state inits to 0, so before the populations warm up every term is 0
// and the rule yields false (local). ε-exploration (epsilon) is what bootstraps
// the REMOTE population: with probability ε the chosen action is flipped, using
// a seeded RNG for reproducibility.
//
// BLIS proxy note: BLIS exposes no isolated prefill-time metric for local
// (non-disaggregated) requests, so prefillTimeUs for the LOCAL population is the
// observed TTFT (FirstTokenTime), which bundles queue wait with prefill service.
// This is a deliberate degraded-fidelity proxy — on llm-d the real vLLM
// prefill-time metric replaces it without any change to this decider. When the
// physics does not model prefill→decode interference, rate_D ≈ rate_P and
// Δp ≈ 0, so the decider self-degrades to drift + SLO (Z). TODO(llm-d): feed the
// vLLM prefill-time metric instead of the TTFT proxy.
type EmpiricalDPPDecider struct {
	// Policy knobs (operator-specified; fixed for the run).
	v         float64 // penalty weight V (fixed)
	ttftSloUs float64 // TTFT SLO target d (µs)
	epsilon   float64 // ε-exploration probability
	beta      float64 // EWMA smoothing factor

	// Uncached-token estimation (mirrors PrefixThresholdDecider).
	blockSize  int
	cacheQuery map[string]func([]int) int

	rng *rand.Rand

	// Learned two-population state.
	rateP, rateD          ewma    // prefill rate µs/uncached-token: REMOTE, LOCAL
	pLocal, pRemote       ewma    // ITL µs/token: LOCAL, REMOTE
	ttftLocal, ttftRemote ewma    // TTFT µs: LOCAL, REMOTE
	z                     float64 // virtual TTFT queue Z (µs)
}

// EmpiricalDPPConfig holds constructor parameters. Separate from CLI flag
// parsing to keep disaggregation_edpp.go test-friendly.
type EmpiricalDPPConfig struct {
	V         float64 // fixed penalty weight; default 1.0
	TTFTSloMs float64 // TTFT SLO d in ms; converted to µs internally; default 100.0
	Epsilon   float64 // ε-exploration probability; default 0.05
	Beta      float64 // EWMA smoothing factor; default 0.05
	Seed      int64   // RNG seed for ε-exploration (reproducibility)
}

// NewEmpiricalDPPDecider creates an EmpiricalDPPDecider.
//
// blockSize and cacheQuery mirror PrefixThresholdDecider: cacheQuery[selected]
// returns the cached-block count for the pre-selected decode pod, and
// uncached tokens = len(InputTokens) − cachedBlocks·blockSize. When cacheQuery
// is nil or the selected pod is unknown, the decider falls back to
// u = len(InputTokens) (treats the request as fully uncached).
//
// Defaults are applied for non-positive V and Beta; TTFTSloMs defaults to 100ms
// when ≤ 0. Epsilon is NOT defaulted — a zero-value config disables exploration
// (the recommended 0.05 default is provided by the --edpp-epsilon CLI flag).
// Panics if Epsilon ∉ [0,1] or Beta > 1.
func NewEmpiricalDPPDecider(cfg EmpiricalDPPConfig, blockSize int, cacheQuery map[string]func([]int) int) *EmpiricalDPPDecider {
	v := cfg.V
	if v <= 0 {
		v = 1.0
	}
	sloMs := cfg.TTFTSloMs
	if sloMs <= 0 {
		sloMs = 100.0
	}
	// ε=0 (exploration off) is a valid, respected value, so it is NOT coerced.
	// The recommended 0.05 default lives in the CLI flag (--edpp-epsilon); a
	// zero-value config therefore disables exploration.
	eps := cfg.Epsilon
	beta := cfg.Beta
	if beta <= 0 {
		beta = 0.05
	}
	if eps < 0 || eps > 1 {
		panic(fmt.Sprintf("EmpiricalDPPDecider: Epsilon must be in [0,1], got %g", eps))
	}
	if beta > 1 {
		panic(fmt.Sprintf("EmpiricalDPPDecider: Beta must be ≤ 1, got %g", beta))
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	return &EmpiricalDPPDecider{
		v:          v,
		ttftSloUs:  sloMs * 1000,
		epsilon:    eps,
		beta:       beta,
		blockSize:  blockSize,
		cacheQuery: cacheQuery,
		rng:        rand.New(rand.NewSource(cfg.Seed)),
	}
}

// uncachedTokens returns the input tokens of req not cached on the pre-selected
// decode pod. Mirrors PrefixThresholdDecider.Decide; falls back to
// len(req.InputTokens) when no per-pod cache information is available.
func (e *EmpiricalDPPDecider) uncachedTokens(req *Request, state *RouterState) int {
	n := len(req.InputTokens)
	if n == 0 || e.cacheQuery == nil || state == nil || state.SelectedInstance == "" {
		return n
	}
	fn, ok := e.cacheQuery[state.SelectedInstance]
	if !ok || fn == nil {
		return n
	}
	u := n - fn(req.InputTokens)*e.blockSize
	if u < 0 {
		return 0
	}
	return u
}

// Decide applies the drift-plus-penalty rule, then ε-exploration. The
// decision-time uncached-token count is stashed on req.DisaggUncachedTokens so
// the matching ObserveCompletion can attribute the prefill rate correctly.
func (e *EmpiricalDPPDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	u := e.uncachedTokens(req, state)
	req.DisaggUncachedTokens = u

	var qD, qP float64
	if state != nil {
		for _, s := range state.Snapshots {
			qD += float64(s.QueueDepth) * s.AvgOutTokens * s.ITL
		}
		for _, s := range state.PrefillSnapshots {
			qP += float64(s.QueueDepth) * s.AvgInTokens * e.rateP.value()
		}
	}

	drift := float64(u) * (qD*e.rateD.value() - qP*e.rateP.value())
	deltaP := e.pLocal.value() - e.pRemote.value()
	deltaTTFT := e.ttftRemote.value() - e.ttftLocal.value()

	disaggregate := drift+e.v*deltaP > e.z*deltaTTFT

	// ε-exploration: flip the action with probability ε to keep both
	// populations fresh (and to bootstrap REMOTE from the all-zero cold start).
	// Guarded so ε=0 consumes no RNG (fully deterministic, drift-only).
	if e.epsilon > 0 && e.rng.Float64() < e.epsilon {
		disaggregate = !disaggregate
	}

	return DisaggregationDecision{Disaggregate: disaggregate}
}

// ObserveCompletion updates the two-population EWMAs (by action) and the virtual
// TTFT queue Z (on every request). Implements CompletionObserver.
func (e *EmpiricalDPPDecider) ObserveCompletion(disaggregated bool, ttftUs, itlMeanUs, prefillTimeUs float64, uncachedTokens int) {
	u := math.Max(1, float64(uncachedTokens))
	if disaggregated {
		if prefillTimeUs > 0 {
			e.rateP.update(prefillTimeUs/u, e.beta)
		}
		if itlMeanUs > 0 {
			e.pRemote.update(itlMeanUs, e.beta)
		}
		if ttftUs > 0 {
			e.ttftRemote.update(ttftUs, e.beta)
		}
	} else {
		if prefillTimeUs > 0 {
			e.rateD.update(prefillTimeUs/u, e.beta)
		}
		if itlMeanUs > 0 {
			e.pLocal.update(itlMeanUs, e.beta)
		}
		if ttftUs > 0 {
			e.ttftLocal.update(ttftUs, e.beta)
		}
	}
	// Z is a constraint queue over ALL requests, not just disaggregated ones.
	e.z = math.Max(0, e.z+ttftUs-e.ttftSloUs)
}

// CurrentV exposes the fixed penalty weight V for metrics/logging.
func (e *EmpiricalDPPDecider) CurrentV() float64 { return e.v }

// CurrentZ exposes the current virtual TTFT queue Z (µs) for metrics/logging.
func (e *EmpiricalDPPDecider) CurrentZ() float64 { return e.z }

// CurrentRates exposes the learned prefill rates (rate_P, rate_D) for logging.
func (e *EmpiricalDPPDecider) CurrentRates() (rateP, rateD float64) {
	return e.rateP.value(), e.rateD.value()
}

// Compile-time checks.
var (
	_ DisaggregationDecider = (*EmpiricalDPPDecider)(nil)
	_ CompletionObserver    = (*EmpiricalDPPDecider)(nil)
)
