package sim

import "testing"

// newEDPP builds a decider with ε disabled (deterministic, drift-only) and the
// given V/β. cacheQuery defaults to nil (u = len(InputTokens)).
func newEDPP(v, beta float64) *EmpiricalDPPDecider {
	return NewEmpiricalDPPDecider(EmpiricalDPPConfig{V: v, Beta: beta, Epsilon: 0, Seed: 1}, 1, nil)
}

// decodeState builds a RouterState with a single decode snapshot contributing
// Q_D = qd·avgOut·itl and an optional single prefill snapshot.
func decodeState(qd int, avgOut, itl float64) *RouterState {
	return &RouterState{Snapshots: []RoutingSnapshot{{ID: "d0", QueueDepth: qd, AvgOutTokens: avgOut, ITL: itl}}}
}

func reqN(n int) *Request {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = i + 1
	}
	return &Request{ID: "r", InputTokens: tokens}
}

// --- Constructor -----------------------------------------------------------

func TestEDPP_ConstructorDefaults(t *testing.T) {
	d := NewEmpiricalDPPDecider(EmpiricalDPPConfig{}, 0, nil)
	if d.v != 1.0 {
		t.Errorf("default V = %g, want 1.0", d.v)
	}
	if d.ttftSloUs != 100_000 {
		t.Errorf("default d = %g µs, want 100000", d.ttftSloUs)
	}
	// Epsilon is not defaulted (zero-value ⟹ exploration off); β defaults to 0.05.
	if d.epsilon != 0 || d.beta != 0.05 {
		t.Errorf("default ε=%g β=%g, want 0/0.05", d.epsilon, d.beta)
	}
	if d.blockSize != 1 {
		t.Errorf("default blockSize = %d, want 1 (non-positive coerced)", d.blockSize)
	}
}

func TestEDPP_ConstructorPanics(t *testing.T) {
	cases := []EmpiricalDPPConfig{{Epsilon: 1.5}, {Beta: 2.0}}
	for _, cfg := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("expected panic for cfg %+v", cfg)
				}
			}()
			NewEmpiricalDPPDecider(cfg, 1, nil)
		}()
	}
}

// --- Decision rule ---------------------------------------------------------

// Cold start: every EWMA is 0, so drift = Δp = ΔTTFT = 0 and the strict
// inequality 0 > 0 is false ⟹ local.
func TestEDPP_ColdStartIsLocal(t *testing.T) {
	d := newEDPP(1, 0.05)
	if d.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
		t.Error("cold-start decision should be local (false)")
	}
}

// Drift dominance: a large Q_D·rate_D with no prefill backlog makes the drift
// term strongly positive ⟹ disaggregate.
func TestEDPP_DriftDisaggregates(t *testing.T) {
	d := newEDPP(1, 0.05)
	d.rateD = ewma{v: 2.0, seen: true} // µs/uncached-token on the mixed decode pool
	// Q_D = 10·100·50 = 50000; u = 10; drift = 10·(50000·2 − 0) = 1e6 > 0.
	if !d.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
		t.Error("positive drift should disaggregate (true)")
	}
}

// Mixed-prefill backlog term: a decode snapshot with NO decode-generation work
// (AvgOutTokens=0, ITL=0) but pending local prefill (AvgInTokens>0) and a warmed
// rate_D must still contribute to Q_D and disaggregate. Without the AvgInTokens·rate_D
// term, Q_D would be 0 and the decision would be local — so this isolates the fix.
func TestEDPP_MixedPrefillBacklog(t *testing.T) {
	mk := func(avgIn float64) *EmpiricalDPPDecider {
		d := newEDPP(1, 0.05)
		d.rateD = ewma{v: 2.0, seen: true}
		return d
	}
	// AvgInTokens=100 ⟹ Q_D = 10·(0·0 + 100·2) = 2000; drift = 10·(2000·2) = 40000 > 0.
	state := &RouterState{Snapshots: []RoutingSnapshot{
		{ID: "d0", QueueDepth: 10, AvgOutTokens: 0, ITL: 0, AvgInTokens: 100},
	}}
	if !mk(100).Decide(reqN(10), state).Disaggregate {
		t.Error("pending mixed-prefill backlog (AvgInTokens·rate_D) should disaggregate")
	}
	// Same snapshot with AvgInTokens=0 ⟹ Q_D=0 ⟹ drift=0 ⟹ local. Confirms the
	// new term — not the decode-gen term — is what flips the decision.
	zeroIn := &RouterState{Snapshots: []RoutingSnapshot{
		{ID: "d0", QueueDepth: 10, AvgOutTokens: 0, ITL: 0, AvgInTokens: 0},
	}}
	if mk(0).Decide(reqN(10), zeroIn).Disaggregate {
		t.Error("zero decode backlog (no decode-gen, no pending prefill) should stay local")
	}
}

// Z·ΔTTFT suppression: with the same positive drift, a large virtual queue Z
// and ttft_remote > ttft_local push the RHS above the LHS ⟹ suppressed to local;
// shrinking Z flips it back to disaggregate.
func TestEDPP_ZSuppression(t *testing.T) {
	build := func(z float64) *EmpiricalDPPDecider {
		d := newEDPP(1, 0.05)
		d.rateD = ewma{v: 2.0, seen: true}          // drift = 1e6 (as above)
		d.ttftRemote = ewma{v: 200_000, seen: true} // ΔTTFT = 100000
		d.ttftLocal = ewma{v: 100_000, seen: true}
		d.z = z
		return d
	}
	// z=100 ⟹ RHS = 100·100000 = 1e7 > 1e6 ⟹ suppressed.
	if build(100).Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
		t.Error("large Z·ΔTTFT should suppress disaggregation")
	}
	// z=1 ⟹ RHS = 1e5 < 1e6 ⟹ disaggregate.
	if !build(1).Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
		t.Error("small Z·ΔTTFT should not suppress disaggregation")
	}
}

// V·Δp term: with no drift, a positive ITL gap (p_local > p_remote) and z=0
// makes LHS = V·Δp > 0 ⟹ disaggregate; a negative gap ⟹ local. Also checks V scales it.
func TestEDPP_PenaltyTerm(t *testing.T) {
	mk := func(pLocal, pRemote, v float64) *EmpiricalDPPDecider {
		d := newEDPP(v, 0.05)
		d.pLocal = ewma{v: pLocal, seen: true}
		d.pRemote = ewma{v: pRemote, seen: true}
		return d
	}
	empty := &RouterState{} // no snapshots ⟹ no drift
	if !mk(50_000, 10_000, 2).Decide(reqN(10), empty).Disaggregate {
		t.Error("Δp > 0 with z=0 should disaggregate")
	}
	if mk(10_000, 50_000, 2).Decide(reqN(10), empty).Disaggregate {
		t.Error("Δp < 0 should stay local")
	}
}

// --- ε-exploration ---------------------------------------------------------

func TestEDPP_EpsilonExtremes(t *testing.T) {
	// ε=0 ⟹ never flip ⟹ cold start always local.
	d0 := NewEmpiricalDPPDecider(EmpiricalDPPConfig{Epsilon: 0, Seed: 7}, 1, nil)
	for i := 0; i < 50; i++ {
		if d0.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
			t.Fatal("ε=0 must never flip the cold-start local decision")
		}
	}
	// ε=1 ⟹ always flip ⟹ cold start always disaggregate.
	d1 := NewEmpiricalDPPDecider(EmpiricalDPPConfig{Epsilon: 1, Seed: 7}, 1, nil)
	for i := 0; i < 50; i++ {
		if !d1.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate {
			t.Fatal("ε=1 must always flip the cold-start local decision")
		}
	}
}

func TestEDPP_EpsilonSeededDeterminism(t *testing.T) {
	mk := func() *EmpiricalDPPDecider {
		return NewEmpiricalDPPDecider(EmpiricalDPPConfig{Epsilon: 0.5, Seed: 123}, 1, nil)
	}
	a, b := mk(), mk()
	for i := 0; i < 100; i++ {
		da := a.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate
		db := b.Decide(reqN(10), decodeState(10, 100, 50)).Disaggregate
		if da != db {
			t.Fatalf("same-seed deciders diverged at step %d", i)
		}
	}
}

// --- Uncached-token estimation --------------------------------------------

func TestEDPP_UncachedTokens(t *testing.T) {
	// No cacheQuery ⟹ u = len(InputTokens).
	d := newEDPP(1, 0.05)
	req := reqN(40)
	d.Decide(req, &RouterState{})
	if req.DisaggUncachedTokens != 40 {
		t.Errorf("fallback u = %d, want 40", req.DisaggUncachedTokens)
	}

	// With cacheQuery: u = len − cachedBlocks·blockSize = 40 − 2·16 = 8.
	cq := map[string]func([]int) int{"d0": func([]int) int { return 2 }}
	dc := NewEmpiricalDPPDecider(EmpiricalDPPConfig{Epsilon: 0, Seed: 1}, 16, cq)
	req2 := reqN(40)
	dc.Decide(req2, &RouterState{Snapshots: []RoutingSnapshot{{ID: "d0"}}, SelectedInstance: "d0"})
	if req2.DisaggUncachedTokens != 8 {
		t.Errorf("cacheQuery u = %d, want 8", req2.DisaggUncachedTokens)
	}

	// Over-cached (clamped to 0): 16 tokens, 2 blocks·16 = 32 cached ⟹ u = 0.
	req3 := reqN(16)
	dc.Decide(req3, &RouterState{Snapshots: []RoutingSnapshot{{ID: "d0"}}, SelectedInstance: "d0"})
	if req3.DisaggUncachedTokens != 0 {
		t.Errorf("over-cached u = %d, want 0 (clamped)", req3.DisaggUncachedTokens)
	}
}

// --- Observation -----------------------------------------------------------

// REMOTE observation updates only the remote population; LOCAL only the local
// population; Z updates on both.
func TestEDPP_ObserveRoutesByAction(t *testing.T) {
	d := newEDPP(1, 0.05)

	// REMOTE: prefill=10000/u=100 ⟹ rateP=100; pRemote=30000; ttftRemote=200000.
	d.ObserveCompletion(true, 200_000, 30_000, 10_000, 100)
	rp, rd := d.CurrentRates()
	if rp != 100 || rd != 0 {
		t.Errorf("after REMOTE obs rateP=%g rateD=%g, want 100/0", rp, rd)
	}
	if d.pRemote.value() != 30_000 || d.pLocal.value() != 0 {
		t.Errorf("REMOTE obs touched the wrong ITL population: pRemote=%g pLocal=%g", d.pRemote.value(), d.pLocal.value())
	}
	// Z = max(0, 0 + 200000 − 100000) = 100000.
	if d.CurrentZ() != 100_000 {
		t.Errorf("Z after REMOTE = %g, want 100000", d.CurrentZ())
	}

	// LOCAL: prefill=5000/u=50 ⟹ rateD=100; pLocal=20000; ttftLocal=50000.
	d.ObserveCompletion(false, 50_000, 20_000, 5_000, 50)
	rp, rd = d.CurrentRates()
	if rp != 100 || rd != 100 {
		t.Errorf("after LOCAL obs rateP=%g rateD=%g, want 100/100", rp, rd)
	}
	if d.pLocal.value() != 20_000 {
		t.Errorf("LOCAL obs pLocal=%g, want 20000", d.pLocal.value())
	}
	// Z = max(0, 100000 + 50000 − 100000) = 50000 (updated on LOCAL too).
	if d.CurrentZ() != 50_000 {
		t.Errorf("Z after LOCAL = %g, want 50000", d.CurrentZ())
	}
}

// EWMA cold-start: first observation sets the value directly; the second blends
// with β. With β=0.5: rateP = 0.5·100 + 0.5·200 = 150.
func TestEDPP_EWMAColdStart(t *testing.T) {
	d := newEDPP(1, 0.5)
	d.ObserveCompletion(true, 1, 1, 10_000, 100) // rateP ← 100 (direct)
	if rp, _ := d.CurrentRates(); rp != 100 {
		t.Fatalf("first obs rateP=%g, want 100 (direct set)", rp)
	}
	d.ObserveCompletion(true, 1, 1, 20_000, 100) // obs 200, blend → 150
	if rp, _ := d.CurrentRates(); rp != 150 {
		t.Errorf("blended rateP=%g, want 150", rp)
	}
}

// Non-positive prefill/ITL signals are skipped (no rate/ITL update) but Z still
// advances on the TTFT observation.
func TestEDPP_ObserveSkipsNonPositiveSignals(t *testing.T) {
	d := newEDPP(1, 0.05)
	d.ObserveCompletion(false, 120_000, 0, 0, 50) // prefill=0, itl=0 ⟹ skipped
	if _, rd := d.CurrentRates(); rd != 0 {
		t.Errorf("rateD=%g, want 0 (prefill=0 skipped)", rd)
	}
	if d.pLocal.value() != 0 {
		t.Errorf("pLocal=%g, want 0 (itl=0 skipped)", d.pLocal.value())
	}
	if d.CurrentZ() != 20_000 { // 120000 − 100000
		t.Errorf("Z=%g, want 20000 (advanced on TTFT)", d.CurrentZ())
	}
}

// --- Invariants ------------------------------------------------------------

// INV-9: the decision must not depend on OutputTokens.
func TestEDPP_INV9_OracleBoundary(t *testing.T) {
	d := newEDPP(1, 0.05)
	d.rateD = ewma{v: 2.0, seen: true}
	state := decodeState(10, 100, 50)

	r1 := &Request{ID: "a", InputTokens: make([]int, 10), OutputTokens: nil}
	r2 := &Request{ID: "b", InputTokens: make([]int, 10), OutputTokens: make([]int, 9999)}
	if d.Decide(r1, state) != d.Decide(r2, state) {
		t.Error("decision depends on OutputTokens — INV-9 violation")
	}
}

func TestEDPP_InterfaceCompliance(t *testing.T) {
	var _ DisaggregationDecider = (*EmpiricalDPPDecider)(nil)
	var _ CompletionObserver = (*EmpiricalDPPDecider)(nil)
}
