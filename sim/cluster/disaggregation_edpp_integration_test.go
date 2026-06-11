package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestEDPP_EndToEnd_TogglesAndStaysBounded is the integration smoke for the
// EmpiricalDPPDecider: it drives a real 2P+2D cluster with --pd-decider edpp
// and ε-exploration enabled, exercising BOTH completion observation call sites
// (REMOTE in detectDecodeCompletions, LOCAL in the OnRequestDone closure).
//
// It asserts the run completes, the virtual TTFT queue Z stays finite and
// non-negative (the formula self-limits — the PT(16) flooding failure mode does
// not manifest), and the decider actually toggles (some requests disaggregate,
// some do not) rather than collapsing to never/always.
func TestEDPP_EndToEnd_TogglesAndStaysBounded(t *testing.T) {
	const numRequests = 40
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.PDDecider = "edpp"
	config.TraceLevel = "decisions"
	// ε=0.5 forces a mix of LOCAL/REMOTE actions from the all-zero cold start,
	// so both observation populations get fed and the toggle assertion is
	// deterministic under the fixed seed.
	config.EDPPEpsilon = 0.5
	config.EDPPBeta = 0.1

	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)

	// The decider must be the empirical-DPP observer.
	decider, ok := cs.disaggregationDecider.(*sim.EmpiricalDPPDecider)
	if !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EmpiricalDPPDecider", cs.disaggregationDecider)
	}

	mustRun(t, cs)

	// Z bounded: finite, non-negative (no runaway from flooding the prefill pool).
	z := decider.CurrentZ()
	if math.IsNaN(z) || math.IsInf(z, 0) || z < 0 {
		t.Errorf("virtual queue Z = %g, want finite and ≥ 0", z)
	}

	// The decider toggled: at least one disaggregated, at least one local.
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	var disagg, total int
	for _, rec := range tr.Disaggregations {
		total++
		if rec.Disaggregate {
			disagg++
		}
	}
	if total == 0 {
		t.Fatal("no disaggregation decisions recorded")
	}
	if disagg == 0 || disagg == total {
		t.Errorf("decider did not toggle: %d/%d disaggregated (want strictly between)", disagg, total)
	}

	// The learned rates populated (both populations were observed).
	if rp, rd := decider.CurrentRates(); rp == 0 && rd == 0 {
		t.Error("neither rate_P nor rate_D learned a value; observation plumbing not feeding the decider")
	}
}
