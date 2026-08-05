package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// edppTraceConfig builds a 2P+2D disagg config driving the EDPP decider, with the
// EDPP knobs and block size set (the constructor panics on zero values).
func edppTraceConfig(traceLevel string) DeploymentConfig {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.PDDecider = "edpp"
	config.TraceLevel = traceLevel
	config.BlockSizeTokens = 16
	config.EDPPTauTTFTUs = 500_000
	config.EDPPTauRefUs = 500_000
	config.EDPPTauITLUs = 100_000
	config.EDPPV = 1.0
	config.EDPPCXferUs = 5_000
	config.EDPPNomPrefillTokens = 512
	config.EDPPNomDecodeCtx = 2048
	config.EDPPCoeffs = sim.EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0}
	return config
}

// TestPDTrace_EDPP_RecordsDecisionTerms: with trace-level decisions and the EDPP decider,
// one EDPPDecisionRecord is emitted per disaggregation decision, and its intermediate
// terms compose into LHS/RHS and the recorded decision.
func TestPDTrace_EDPP_RecordsDecisionTerms(t *testing.T) {
	const numRequests = 5
	cs := NewClusterSimulator(edppTraceConfig("decisions"), NewSliceRequestSource(newTestRequests(numRequests)), nil)
	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.EDPPDecisions) == 0 {
		t.Fatal("expected EDPP decision records, got 0")
	}
	// One EDPP trace per disaggregation decision (recorded at the same call site).
	if len(tr.EDPPDecisions) != len(tr.Disaggregations) {
		t.Errorf("EDPPDecisions=%d != Disaggregations=%d", len(tr.EDPPDecisions), len(tr.Disaggregations))
	}
	for i, r := range tr.EDPPDecisions {
		if r.RequestID == "" {
			t.Errorf("EDPPDecisions[%d]: empty RequestID", i)
		}
		if r.SkipReason != "" {
			continue // early-return path: term fields intentionally zero
		}
		if math.Abs(r.LHS-(r.BalanceTermD-r.BalanceTermP)) > 1e-9 {
			t.Errorf("EDPPDecisions[%d]: LHS %v != BalanceTermD-BalanceTermP", i, r.LHS)
		}
		if math.Abs(r.RHS-(r.TransferTerm+r.TTFTTerm+r.ITLTerm)) > 1e-9 {
			t.Errorf("EDPPDecisions[%d]: RHS %v != sum of components", i, r.RHS)
		}
		if r.Disaggregate != (r.LHS > r.RHS) {
			t.Errorf("EDPPDecisions[%d]: Disaggregate %v inconsistent with LHS %v > RHS %v", i, r.Disaggregate, r.LHS, r.RHS)
		}
	}
}

// TestPDTrace_EDPP_NoRecordsWhenTraceNone: trace-level none ⇒ no trace, no EDPP records,
// and no panic (zero-overhead path).
func TestPDTrace_EDPP_NoRecordsWhenTraceNone(t *testing.T) {
	cs := NewClusterSimulator(edppTraceConfig("none"), NewSliceRequestSource(newTestRequests(5)), nil)
	mustRun(t, cs)
	if tr := cs.Trace(); tr != nil && len(tr.EDPPDecisions) != 0 {
		t.Errorf("expected no EDPP records when trace-level none, got %d", len(tr.EDPPDecisions))
	}
}
