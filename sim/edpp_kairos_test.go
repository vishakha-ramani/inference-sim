package sim

import (
	"math"
	"testing"
)

func newKairosTestDecider(t *testing.T, rule string, configure func(*EDPPConfig), prefill ...RoutingSnapshot) *EDPPDecider {
	t.Helper()
	cfg := defaultTestEDPPConfig()
	cfg.Rule = rule
	cfg.ChunkTokens = 256
	cfg.CXferUs = 0
	cfg.TraceEnabled = true
	if configure != nil {
		configure(&cfg)
	}
	return NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot {
		return prefill
	})
}

func TestKairosPaper_DefaultAlphaAndMargin(t *testing.T) {
	prefill := RoutingSnapshot{ID: "p0"}
	state := &RouterState{Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 1, RunningDecode: []RunningReqState{{SLOClass: "standard"}},
	}}}
	req := makeReq("r", 100, "standard")

	withDefault := newKairosTestDecider(t, "kairos-paper", nil, prefill)
	got := withDefault.Decide(req, state)
	if got.Disaggregate || got.DecodePodOverride != "d0" {
		t.Fatalf("alpha=1.3 decision = %+v, want deflection to d0", got)
	}
	if math.Abs(withDefault.kairosAlpha-1.3) > 1e-12 {
		t.Fatalf("default alpha = %v, want 1.3", withDefault.kairosAlpha)
	}
	if got.EDPPTrace == nil || got.EDPPTrace.KairosAlpha != 1.3 || got.EDPPTrace.KairosMode != "paper" {
		t.Fatalf("paper trace = %+v, want mode=paper alpha=1.3", got.EDPPTrace)
	}

	withAlphaOne := newKairosTestDecider(t, "kairos-paper", func(cfg *EDPPConfig) {
		cfg.KairosAlpha = 1
	}, prefill)
	got = withAlphaOne.Decide(req, state)
	if !got.Disaggregate {
		t.Fatalf("alpha=1 decision = %+v, want regular prefill because decode TTFT is slower", got)
	}
}

func TestKairosPaper_EnforcesRequestTTFTGate(t *testing.T) {
	d := newKairosTestDecider(t, "kairos-paper", func(cfg *EDPPConfig) {
		cfg.TauTTFTUs = 2050 // decode estimate is 2100; alpha margin still passes
	}, RoutingSnapshot{ID: "p0"})
	state := &RouterState{Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 1, RunningDecode: []RunningReqState{{SLOClass: "standard"}},
	}}}
	got := d.Decide(makeReq("r", 100, "standard"), state)
	if !got.Disaggregate {
		t.Fatalf("decision = %+v, want regular prefill when decode TTFT misses request SLO", got)
	}
	if got.EDPPTrace == nil || !got.EDPPTrace.KairosTTFTGateRequired || got.EDPPTrace.KairosTTFTGatePassed {
		t.Fatalf("TTFT gate trace = %+v, want required=true passed=false", got.EDPPTrace)
	}
}

func TestKairosPaper_UsesStrictestResidentTBTTarget(t *testing.T) {
	d := newKairosTestDecider(t, "kairos-paper", func(cfg *EDPPConfig) {
		cfg.TauITLByClassUs = map[string]int64{"critical": 1500}
	}, RoutingSnapshot{ID: "p0"})
	req := makeReq("r", 100, "standard")

	strict := &RouterState{Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 1, RunningDecode: []RunningReqState{{SLOClass: "critical"}},
	}}}
	got := d.Decide(req, strict)
	if !got.Disaggregate {
		t.Fatalf("strict-resident decision = %+v, want regular prefill because no chunk fits 1500us", got)
	}

	loose := &RouterState{Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 1, RunningDecode: []RunningReqState{{SLOClass: "standard"}},
	}}}
	got = d.Decide(req, loose)
	if got.Disaggregate || got.DecodePodOverride != "d0" {
		t.Fatalf("standard-resident decision = %+v, want deflection to d0", got)
	}
	if got.EDPPTrace == nil || got.EDPPTrace.KairosResidentTauITL != float64(d.cfg.TauITLUs) {
		t.Fatalf("resident target trace = %+v, want %d", got.EDPPTrace, d.cfg.TauITLUs)
	}
}

func TestKairosPaper_DiscreteChunkSearch(t *testing.T) {
	c := defaultTestEDPPConfig().Coeffs
	ttft, schedule, ok := kairosDiscreteDeflectTTFT(c, 0, 0, 600, 4000, kairosDiscreteCandidates(2048), 32)
	if !ok {
		t.Fatal("discrete schedule unexpectedly infeasible")
	}
	want := []float64{256, 256, 88}
	if len(schedule) != len(want) {
		t.Fatalf("schedule = %v, want %v", schedule, want)
	}
	for i := range want {
		if schedule[i] != want[i] {
			t.Fatalf("schedule = %v, want %v", schedule, want)
		}
	}
	continuous := kairosMaxSafeChunk(c, c.tIterDecode(0, 0, 0), 0, 4000)
	if continuous != 300 {
		t.Fatalf("continuous safe chunk = %v, want 300", continuous)
	}
	if schedule[0] == continuous {
		t.Fatalf("paper search used continuous chunk %v instead of discrete 256", continuous)
	}
	if ttft <= 0 {
		t.Fatalf("ttft = %v, want positive", ttft)
	}
}

func TestKairosModeSeparationAndExactPrefillTokens(t *testing.T) {
	prefill := RoutingSnapshot{ID: "p0", QueueDepth: 99, PrefillTokensAhead: 100}
	paper := newKairosTestDecider(t, "kairos-paper", nil, prefill)
	adapted := newKairosTestDecider(t, "kairos-adapted", nil, prefill)
	req := makeReq("r", 100, "standard")

	paperTTFT, _ := paper.kairosPaperPrefillTTFT(req, 100)
	if paperTTFT != 4000 {
		t.Fatalf("paper prefill TTFT = %v, want 4000 from exact 100 queued tokens + own execution", paperTTFT)
	}
	adaptedTTFT, _ := adapted.kairosAdaptedPrefillTTFT(req, 100)
	if adaptedTTFT <= paperTTFT {
		t.Fatalf("adapted TTFT = %v, want larger than exact paper TTFT %v due queue-depth approximation", adaptedTTFT, paperTTFT)
	}

	state := &RouterState{Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 1, RunningDecode: []RunningReqState{{SLOClass: "standard"}},
	}}}
	if got := paper.Decide(req, state); got.Disaggregate {
		t.Fatalf("paper decision = %+v, want deflection under alpha margin", got)
	}
	if got := adapted.Decide(req, state); got.Disaggregate {
		// The large adapted queue estimate also makes deflection attractive; assert
		// identity through the trace instead of forcing equal decisions.
		if got.EDPPTrace == nil || got.EDPPTrace.KairosMode != "adapted" {
			t.Fatalf("adapted trace = %+v, want explicit adapted identity", got.EDPPTrace)
		}
	}
}

func TestKairosPaper_RejectsAlphaBelowOne(t *testing.T) {
	assertPanics(t, func() {
		_ = newKairosTestDecider(t, "kairos-paper", func(cfg *EDPPConfig) {
			cfg.KairosAlpha = 0.99
		}, RoutingSnapshot{ID: "p0"})
	})
}

func TestKairosPaper_SelectsLowestTTFTPrefillAndHintsIt(t *testing.T) {
	d := newKairosTestDecider(t, "kairos-paper", nil,
		RoutingSnapshot{ID: "p0", PrefillTokensAhead: 10_000},
		RoutingSnapshot{ID: "p1", PrefillTokensAhead: 0},
	)
	req := makeReq("r", 100, "standard")

	ttft, prefillID := d.kairosPaperPrefillTTFT(req, 100)
	if prefillID != "p1" {
		t.Fatalf("paper prefill ID = %q, want least-loaded p1", prefillID)
	}
	if ttft != 2000 {
		t.Fatalf("paper prefill TTFT = %v, want 2000 on p1", ttft)
	}

	got := d.Decide(req, &RouterState{})
	if !got.Disaggregate || got.PrefillPodHint != "p1" {
		t.Fatalf("paper decision = %+v, want disaggregation hinted to p1", got)
	}
}

func TestKairosPrefillTieBreaksByInstanceID(t *testing.T) {
	d := newKairosTestDecider(t, "kairos-paper", nil,
		RoutingSnapshot{ID: "p1"},
		RoutingSnapshot{ID: "p0"},
	)
	_, prefillID := d.kairosPaperPrefillTTFT(makeReq("r", 100, "standard"), 100)
	if prefillID != "p0" {
		t.Fatalf("equal-TTFT prefill ID = %q, want deterministic p0", prefillID)
	}
}
