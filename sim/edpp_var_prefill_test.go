package sim

import (
	"math"
	"testing"
)

func newVarPrefillTestDecider(t *testing.T, prefillWeight float64) *EDPPDecider {
	t.Helper()
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "var-prefill"
	cfg.VarMetric = "hazard"
	cfg.VarDeployable = true
	cfg.VarPrefillWeight = prefillWeight
	cfg.TraceEnabled = true
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	return NewEDPPDecider(cfg, newTestAffineModel(), coldCacheQuery("D0", "P0"), prefill)
}

// The simplified rule is intentionally reduced: the existing decode scorer owns
// decode load balancing, while this rule decides only local-vs-disaggregated
// prefill. Flags that reintroduce joint routing, decode congestion, or
// normalized congestion are rejected. The arriving request's own-good reward
// is an explicit ablation of this reduced policy.
func TestVarPrefill_RejectsIncompatibleObjectives(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*EDPPConfig)
	}{
		{"joint", func(c *EDPPConfig) { c.Joint = true }},
		{"decode-congestion", func(c *EDPPConfig) { c.VarKeepCongestion = true }},
		{"normalization", func(c *EDPPConfig) { c.VarNormalize = true }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := defaultTestEDPPConfig()
			cfg.Rule = "var-prefill"
			cfg.VarMetric = "util"
			tc.mutate(&cfg)
			assertPanics(t, func() {
				NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
			})
		})
	}
}

func TestVarPrefill_OwnGoodAddsExactRequestValueDifference(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "var-prefill"
	cfg.VarMetric = "util"
	cfg.VarDeployable = true
	cfg.VarGoodputObjective = true
	cfg.VarPrefillWeight = 1
	cfg.TraceEnabled = true
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	d := NewEDPPDecider(
		cfg, newTestAffineModel(), coldCacheQuery("D0", "P0"), prefill,
	)
	req := &Request{ID: "r", InputTokens: make([]TokenID, 400)}
	state := &RouterState{
		SelectedInstance: "D0",
		Snapshots: []RoutingSnapshot{{
			ID: "D0", BatchSize: 1, KvTokensInUse: 2048,
		}},
	}
	tr := d.Decide(req, state).EDPPTrace
	if tr == nil {
		t.Fatal("expected decision trace")
	}
	want := tr.VarLocalTotal - tr.VarDisaggTotal +
		tr.SelfGoodDisagg - tr.SelfGoodLocal
	if math.Abs(tr.LHS-want) > 1e-12 {
		t.Fatalf("LHS = %v, want exact VaR + own-good difference %v", tr.LHS, want)
	}
	if tr.SelfGoodLocal <= 0 || tr.SelfGoodDisagg <= 0 {
		t.Fatalf(
			"own-good trace not populated: local=%v disagg=%v",
			tr.SelfGoodLocal, tr.SelfGoodDisagg,
		)
	}
}

// With no co-residents, VaR is zero. The only nonzero simplified-policy
// charge must therefore be the normalized prefill waiting-queue stability
// term. Decode backlog, virtual queues, transfer penalty, and TTFT/ITL self
// terms must not leak into the decision.
func TestVarPrefill_OnlyChargesPrefillQueueStabilityWithoutVaR(t *testing.T) {
	const weight = 2.0
	d := newVarPrefillTestDecider(t, weight)
	req := &Request{ID: "r", InputTokens: make([]TokenID, 400)}
	n := d.normFor(req.SLOClass)

	// qp = Qp/W*p = 1. Decode backlog and virtual queues are deliberately huge;
	// the simplified rule must ignore them.
	d.qpWork = n.wStarP
	d.qdWork = 1e15
	d.ensureZ(req.SLOClass).zTTFT = 1e15
	d.ensureZ(req.SLOClass).zITL = 1e15

	state := &RouterState{
		SelectedInstance: "D0",
		Snapshots:        []RoutingSnapshot{{ID: "D0", BatchSize: 1, KvTokensInUse: 2048}},
	}
	tr := d.Decide(req, state).EDPPTrace
	if tr == nil {
		t.Fatal("expected decision trace")
	}

	wp := d.coeffs.Wp(len(req.InputTokens), len(req.InputTokens))
	want := weight * (wp / n.wStarP)
	if math.Abs(tr.PrefillStabilityTerm-want) > 1e-12 {
		t.Fatalf("prefill stability = %v, want %v", tr.PrefillStabilityTerm, want)
	}
	if tr.LHS != 0 || math.Abs(tr.RHS-want) > 1e-12 {
		t.Fatalf("simplified comparison LHS/RHS = %v/%v, want 0/%v", tr.LHS, tr.RHS, want)
	}
	if tr.TransferTerm != 0 || tr.TTFTTerm != 0 || tr.ITLTerm != 0 {
		t.Fatalf("removed self terms leaked into objective: transfer=%v ttft=%v itl=%v",
			tr.TransferTerm, tr.TTFTTerm, tr.ITLTerm)
	}
	if tr.Disaggregate {
		t.Fatal("zero VaR benefit must not overcome positive prefill-queue stability cost")
	}
}

func TestVarPrefill_ZeroWeightIsVaROnlyAblation(t *testing.T) {
	d := newVarPrefillTestDecider(t, 0)
	req := &Request{ID: "r", InputTokens: make([]TokenID, 400)}
	n := d.normFor(req.SLOClass)
	d.qpWork = 10 * n.wStarP
	state := &RouterState{
		SelectedInstance: "D0",
		Snapshots:        []RoutingSnapshot{{ID: "D0"}},
	}
	tr := d.Decide(req, state).EDPPTrace
	if tr == nil {
		t.Fatal("expected decision trace")
	}
	if tr.PrefillStabilityTerm != 0 || tr.RHS != 0 {
		t.Fatalf(
			"zero-weight VaR-only ablation charged stability: term=%v rhs=%v",
			tr.PrefillStabilityTerm, tr.RHS,
		)
	}
}

// Local and remote prefill do not necessarily have the same prefix cache.
// In the evaluated one-prefill-node topology the reduced rule can observe both
// locations exactly, so the remote TTFT/stability operands must use P0's a_p
// rather than silently reusing D0's.
func TestVarPrefill_PathSpecificPrefillWorkUsesRemoteCache(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "var-prefill"
	cfg.VarMetric = "util"
	cfg.VarDeployable = true
	cfg.VarPrefillWeight = 1
	cfg.PathSpecificPrefillWork = true
	cfg.TraceEnabled = true
	cache := map[string]func([]TokenID) int{
		"D0": func([]TokenID) int { return 10 }, // 160 cached => a_p^D=840
		"P0": func([]TokenID) int { return 50 }, // 800 cached => a_p^P=200
	}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	d := NewEDPPDecider(cfg, newTestAffineModel(), cache, prefill)
	n := d.normFor("")
	d.qpWork = n.wStarP // q_p=1 makes the remote-work stability operand visible

	req := &Request{ID: "r", InputTokens: make([]TokenID, 1000)}
	state := &RouterState{
		SelectedInstance: "D0",
		Snapshots:        []RoutingSnapshot{{ID: "D0"}},
	}
	tr := d.Decide(req, state).EDPPTrace
	if tr == nil {
		t.Fatal("expected decision trace")
	}
	if tr.Ap != 840 || tr.ApPrefill != 200 {
		t.Fatalf("path a_p local/remote = %d/%d, want 840/200", tr.Ap, tr.ApPrefill)
	}
	wantLocal := d.coeffs.Wp(840, 1000)
	wantRemote := d.coeffs.Wp(200, 1000)
	if tr.Wp != wantLocal || tr.WpPrefill != wantRemote {
		t.Fatalf(
			"path Wp local/remote = %v/%v, want %v/%v",
			tr.Wp, tr.WpPrefill, wantLocal, wantRemote,
		)
	}
	wantStability := wantRemote / n.wStarP
	if math.Abs(tr.PrefillStabilityTerm-wantStability) > 1e-12 {
		t.Fatalf(
			"stability used wrong path work: got %v, want %v",
			tr.PrefillStabilityTerm, wantStability,
		)
	}
}

// Backlog conservation must use the same path-specific observable work as the
// decision. The cluster historically passed a cold full-prompt upper bound to
// OnRoute; with the ablation enabled the decider corrects it using the sole P
// node's cache before booking Q_p.
func TestVarPrefill_PathSpecificOnRouteBooksRemoteWork(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "var-prefill"
	cfg.VarMetric = "util"
	cfg.VarDeployable = true
	cfg.PathSpecificPrefillWork = true
	cache := map[string]func([]TokenID) int{
		"D0": func([]TokenID) int { return 10 },
		"P0": func([]TokenID) int { return 50 },
	}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	d := NewEDPPDecider(cfg, newTestAffineModel(), cache, prefill)
	req := &Request{ID: "r", InputTokens: make([]TokenID, 1000)}

	d.OnRoute(req, req.ID, true, 1000, "D0", "")
	want := d.coeffs.Wp(200, 1000)
	if d.qpWork != want {
		t.Fatalf("Qp booked work = %v, want remote-cache Wp %v", d.qpWork, want)
	}
	if pending := d.pending[req.ID]; pending.wp != want {
		t.Fatalf("pending remote work = %v, want %v", pending.wp, want)
	}
}

// VaR remains the benefit side of the simplified comparison. With an active
// decode co-resident, local prefill delays it immediately while disaggregated
// prefill arrives after its one estimated remaining step; with an empty prefill
// queue, that positive VaR difference must select disaggregation.
func TestVarPrefill_VaRCanSelectDisaggregation(t *testing.T) {
	d := newVarPrefillTestDecider(t, 1)
	req := &Request{ID: "r", InputTokens: make([]TokenID, 400)}
	state := &RouterState{
		Clock:            10_000,
		SelectedInstance: "D0",
		Snapshots: []RoutingSnapshot{{
			ID: "D0", BatchSize: 1, KvTokensInUse: 2048,
			RunningDecode: []RunningReqState{{
				StepsDone: 0, SLOClass: "", ArrivalUs: 0,
				FirstTokenUs: 1_000, TTFTSet: true,
			}},
		}},
	}
	tr := d.Decide(req, state).EDPPTrace
	if tr == nil {
		t.Fatal("expected decision trace")
	}
	if tr.LHS <= 0 {
		t.Fatalf("VaR(local)-VaR(disagg) = %v, want positive", tr.LHS)
	}
	if got := tr.VarLocalDecode + tr.VarLocalCollocPrefill; math.Abs(got-tr.VarLocalTotal) > 1e-12 {
		t.Fatalf("local VaR components sum to %v, total=%v", got, tr.VarLocalTotal)
	}
	if got := tr.VarDisaggDecode + tr.VarDisaggCollocPrefill + tr.VarDisaggPrefillPool; math.Abs(got-tr.VarDisaggTotal) > 1e-12 {
		t.Fatalf("disagg VaR components sum to %v, total=%v", got, tr.VarDisaggTotal)
	}
	if got := tr.VarLocalTotal - tr.VarDisaggTotal; math.Abs(got-tr.LHS) > 1e-12 {
		t.Fatalf("VaR total difference = %v, LHS=%v", got, tr.LHS)
	}
	if tr.PrefillStabilityTerm != 0 || tr.RHS != 0 {
		t.Fatalf("empty prefill queue cost = %v/%v, want 0/0", tr.PrefillStabilityTerm, tr.RHS)
	}
	if !tr.Disaggregate {
		t.Fatal("positive VaR benefit with no prefill-queue cost should disaggregate")
	}
}
