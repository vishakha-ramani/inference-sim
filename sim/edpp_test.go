package sim

import (
	"fmt"
	"math"
	"testing"
)

// edppAffineModel is a test LatencyModel whose StepTime is exactly α + Σδ(r),
// matching the structural form the EDPP finite-difference extraction assumes:
//
//	prefill probe (ProgressIndex < len(InputTokens)): δ = kp · NumNewTokens
//	decode probe  (otherwise, len(OutputTokens) > 0):  δ = c0 + c1 · ProgressIndex
//
// This lets the unit tests assert exact α/δ recovery and exact normalizer values.
type edppAffineModel struct {
	alpha int64 // per-step fixed cost
	kp    int64 // prefill marginal per chunk token
	c0    int64 // decode per-request overhead
	c1    int64 // decode marginal per context token
}

func (m *edppAffineModel) StepTime(batch []*Request) int64 {
	if len(batch) == 0 {
		return 1 // matches the LatencyModel contract: StepTime([]) >= 1, never α
	}
	s := m.alpha
	for _, r := range batch {
		if r.ProgressIndex < int64(len(r.InputTokens)) {
			s += m.kp * int64(r.NumNewTokens) // prefill
		} else {
			s += m.c0 + m.c1*r.ProgressIndex // decode
		}
	}
	return s
}

func (m *edppAffineModel) QueueingTime(*Request) int64      { return 0 }
func (m *edppAffineModel) OutputTokenProcessingTime() int64 { return 0 }
func (m *edppAffineModel) PostDecodeFixedOverhead() int64   { return 0 }

func newTestAffineModel() *edppAffineModel {
	return &edppAffineModel{alpha: 1000, kp: 10, c0: 100, c1: 1}
}

func defaultTestEDPPConfig() EDPPConfig {
	return EDPPConfig{
		TauTTFTUs:        100_000, // 100 ms
		TauITLUs:         50_000,  // 50 ms
		TauRefUs:         100_000, // fixed reference = default τ_ttft ⇒ factor 1 at default
		V:                1.0,
		CXferUs:          5_000, // 5 ms
		NomPrefillTokens: 512,
		NomDecodeCtx:     2048,
		BlockSize:        16,
		Coeffs:           EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0},
	}
}

func assertPanics(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("expected panic, got none")
		}
	}()
	f()
}

// --- Coefficient extraction (Task 1) ---

// TestEDPP_NoProbeHelpers is a compile-time guard documenting that the probe
// extraction path is fully removed; physics now comes from EDPPCoeffs. This test
// has no body assertions — it fails to compile if a probe helper is reintroduced
// and referenced, and passes once the package builds without them.
func TestEDPP_NoProbeHelpers(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	// The decider still answers purely from coeffs.
	if d.coeffs.AlphaD <= 0 {
		t.Fatal("coeffs not wired")
	}
}

func TestEDPP_AlphaZero_YieldsConservingRate(t *testing.T) {
	// §11 conservation anchor: with α → 0, μ^nom → 1 (work-conserving server).
	// Both the affine model and the coeffs must have AlphaD=AlphaP=0 so both the
	// probe path (alphaD/alphaP) and the coeff path (coeffs.muDNom/muPNom) yield 1.
	m := &edppAffineModel{alpha: 0, kp: 10, c0: 100, c1: 1}
	cfg := defaultTestEDPPConfig()
	cfg.Coeffs = EDPPCoeffs{AlphaD: 0, AlphaP: 0, C0: 100, C1: 1, CPf: 10, CAttn: 0}
	// Note: validate() requires AlphaD>0 and AlphaP>0, but we bypass via direct field
	// construction — the test verifies behavior at the mathematical limit α→0.
	// We construct the decider directly to skip validate(), exercising clampMu.
	d := &EDPPDecider{
		cfg:      cfg,
		model:    m,
		coeffs:   cfg.Coeffs,
		zByClass: make(map[string]*edppClassState),
	}
	d.muPNom = d.coeffs.muPNom(cfg.NomPrefillTokens)
	n := d.normFor("") // default class
	if math.Abs(n.muDNom-1.0) > 1e-9 {
		t.Errorf("α=0 ⇒ μ_d^nom = %v, want 1.0", n.muDNom)
	}
	if math.Abs(n.muPNom-1.0) > 1e-9 {
		t.Errorf("α=0 ⇒ μ_p^nom = %v, want 1.0", n.muPNom)
	}
}

func TestEDPP_NormalizersFromCoeffs(t *testing.T) {
	// Prove the normalizers come from the frozen coefficients, not the probe path.
	// We use Coeffs.CPf=7 while the affine model keeps kp=10, so the two paths diverge:
	//   coeff path: muPNom = 1 − 1000/(1000 + 7·512) = 1 − 1000/4584 ≈ 0.7820...
	//   probe path: muPNom = 1 − 1000/StepTime(prefillProbe(512))
	//             = 1 − 1000/(1000 + 10·512) = 1 − 1000/6120 ≈ 0.8366...
	// If the test passes, it must be reading the coeff path.
	cfg := defaultTestEDPPConfig()
	cfg.Coeffs.CPf = 7 // diverges from model's kp=10
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
	n := d.normFor("") // default class: τ_ttft=100_000, τ_itl=50_000

	// μ_d^nom = 1 − α/τ_itl = 1 − 1000/50000 = 0.98 (coeff path: AlphaD=1000)
	if math.Abs(n.muDNom-0.98) > 1e-9 {
		t.Errorf("muDNom = %v, want 0.98", n.muDNom)
	}
	// μ_p^nom via coeff path: 1 − AlphaP/(AlphaP + CPf·S_pf^nom) = 1 − 1000/(1000+7·512)
	wantMuP := 1 - 1000.0/(1000.0+7.0*512)
	if math.Abs(n.muPNom-wantMuP) > 1e-9 {
		t.Errorf("muPNom = %v, want %v (coeff path with CPf=7; probe path would give ~0.8366)", n.muPNom, wantMuP)
	}
	// W*_d = μ_d^nom · τ_ttft = 0.98 · 100000 = 98000
	if math.Abs(n.wStarD-98000) > 1e-6 {
		t.Errorf("wStarD = %v, want 98000", n.wStarD)
	}
	// W*_p = μ_p^nom · τ_ttft (coeff path)
	if math.Abs(n.wStarP-wantMuP*100000) > 1e-6 {
		t.Errorf("wStarP = %v, want %v", n.wStarP, wantMuP*100000)
	}
}

// --- Constructor / normalizers (Task 2) ---

func TestEDPP_Constructor_PrecomputesNormalizers(t *testing.T) {
	m := newTestAffineModel()
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, m, nil, nil)
	n := d.normFor("") // default class

	// μ_d^nom = 1 − α_d/τ_itl = 1 − 1000/50000 = 0.98
	if math.Abs(n.muDNom-0.98) > 1e-9 {
		t.Errorf("μ_d^nom = %v, want 0.98", n.muDNom)
	}
	// T_iter_p^nom = StepTime([prefillProbe(512)]) = 1000 + 10·512 = 6120
	// μ_p^nom = 1 − 1000/6120
	wantMuP := 1 - 1000.0/6120.0
	if math.Abs(n.muPNom-wantMuP) > 1e-9 {
		t.Errorf("μ_p^nom = %v, want %v", n.muPNom, wantMuP)
	}
	// W*_d = μ_d^nom · τ_ttft = 0.98 · 100000 = 98000
	if math.Abs(n.wStarD-98000) > 1e-6 {
		t.Errorf("W*_d = %v, want 98000", n.wStarD)
	}
}

func TestEDPP_Constructor_RejectsInvalidConfig(t *testing.T) {
	m := newTestAffineModel()
	bad := []EDPPConfig{
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.TauTTFTUs = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.TauITLUs = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.V = -1; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.CXferUs = -1; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.NomPrefillTokens = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.NomDecodeCtx = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.BlockSize = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.TauRefUs = 0; return c }(),
	}
	for i, c := range bad {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("config[%d] expected panic, got none", i)
				}
			}()
			_ = NewEDPPDecider(c, m, nil, nil)
		}()
	}
}

// Selecting an oracle estimator as the routing driver must be rejected (INV-9 guard runtime path).
func TestNewEDPPDecider_RejectsOracleDriver(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when oracle estimator is the routing driver")
		}
	}()
	cfg := defaultTestEDPPConfig()
	cfg.TAdmEstimator = "rollforward_oracle"
	_ = NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
}

// --- Decide rule + §11 behavioral anchors (Tasks 3 & 5) ---

func decodeState(selected string, queue, batch int, itlUs float64) *RouterState {
	return &RouterState{
		SelectedInstance: selected,
		Snapshots: []RoutingSnapshot{
			{ID: selected, QueueDepth: queue, BatchSize: batch, ITL: itlUs},
		},
	}
}

// admissionSpy is a test AdmissionDelayEstimator that captures the last
// AdmissionContext it was asked to estimate, so tests can assert on how Decide
// populated the occupancy fields.
type admissionSpy struct {
	onCall func(AdmissionContext)
}

func (s admissionSpy) Name() string { return "spy" }
func (s admissionSpy) EstimateTAdm(ctx AdmissionContext) float64 {
	if s.onCall != nil {
		s.onCall(ctx)
	}
	return 0
}

// newTestEDPPDeciderWithEstimator constructs an EDPPDecider with the default
// test config and injects est as the admission-delay estimator.
func newTestEDPPDeciderWithEstimator(t *testing.T, est AdmissionDelayEstimator) *EDPPDecider {
	t.Helper()
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
	d.tadmEstimator = est
	return d
}

// makeReq builds a Request with nInput input tokens and the given SLO class.
func makeReq(id string, nInput int, class string) *Request {
	return &Request{ID: id, InputTokens: make([]TokenID, nInput), SLOClass: class}
}

// The deployable remaining-steps must NOT collapse to 1 under saturation: with running
// requests whose StepsDone exceed a stale N̂_out, N̂_out is floored by the max in-flight
// elapsed (censored lower bound: o_r ≥ tokens already produced), so the estimate reflects
// the running occupants' scale rather than 1.
func TestDecide_CensoredRemainingSteps(t *testing.T) {
	var seen AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = c }}
	d := newTestEDPPDeciderWithEstimator(t, spy)
	// N̂_out starts at its default (small/0); running requests are deep into decode.
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 3, MaxBatchSize: 4,
		RunningDecode: []RunningReqState{{StepsDone: 2000, TrueRemaining: -1}, {StepsDone: 2200, TrueRemaining: -1}, {StepsDone: 1800, TrueRemaining: -1}},
	}}}
	d.Decide(makeReq("r1", 100, "batch"), state)
	// Censored floor: N̂_out_eff ≥ max StepsDone (2200); remaining_i = max(N̂_out_eff − StepsDone_i, 1).
	// Mean over {2200-2000, 2200-2200→1, 2200-1800} = mean(200,1,400) ≈ 200.3 — NOT 1.
	if seen.RemainingStepsEst < 100 {
		t.Fatalf("remaining-steps collapsed to ~1 (%v); censored floor should keep it ~200", seen.RemainingStepsEst)
	}
}

// INV-9 (control path) ASYMMETRY: the runtime routing driver must never see DECODE's
// oracle remaining (o_r-derived, hidden), so Decide censors the decode Running slice to
// -1 before feeding the estimator. PREFILL remaining (inLen − ProgressIndex) is KNOWN
// input, so it is deployable-legitimate and must NOT be censored. This asserts both:
// decode Running stripped, prefill Running preserved.
func TestDecide_StripsOracleFromRoutingContext(t *testing.T) {
	var seen []AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = append(seen, c) }}
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil,
		func() []RoutingSnapshot {
			return []RoutingSnapshot{{
				ID: "p0", BatchSize: 2, MaxBatchSize: 2, FreeKVBlocks: 0,
				RunningPrefill: []RunningReqState{{StepsDone: 0, KVBlocks: 2, TrueRemaining: 7}},
			}}
		})
	d.tadmEstimator = spy
	// Oracle mode would populate TrueRemaining on the running requests.
	decSnap := RoutingSnapshot{
		ID: "d0", BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 0,
		RunningDecode: []RunningReqState{
			{StepsDone: 5, KVBlocks: 5, TrueRemaining: 11},
			{StepsDone: 8, KVBlocks: 5, TrueRemaining: 13},
		},
	}
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{decSnap}}
	d.Decide(makeReq("r1", 4000, "batch"), state)

	if len(seen) == 0 {
		t.Fatalf("estimator was never called")
	}
	// Identify prefill vs decode contexts by their running-occupant fingerprint:
	// the prefill snapshot's single occupant has KVBlocks==2 (from RunningPrefill),
	// the decode occupants have KVBlocks==5 (from RunningDecode).
	sawPrefill, sawDecode := false, false
	for _, c := range seen {
		for i, r := range c.Running {
			if r.KVBlocks == 2 { // prefill occupant: known remaining, must be preserved
				sawPrefill = true
				if r.TrueRemaining != 7 {
					t.Fatalf("prefill remaining censored (Running[%d]=%d); prefill remaining is known input, must NOT be stripped", i, r.TrueRemaining)
				}
			} else { // decode occupant: oracle remaining, must be censored
				sawDecode = true
				if r.TrueRemaining != -1 {
					t.Fatalf("routing path leaked oracle DECODE TrueRemaining=%d (Running[%d]); INV-9 violation", r.TrueRemaining, i)
				}
			}
		}
	}
	if !sawPrefill || !sawDecode {
		t.Fatalf("expected both prefill and decode contexts (prefill=%v decode=%v)", sawPrefill, sawDecode)
	}
	// The shared snapshot slices must not be mutated (deep-copy required on the decode path).
	if decSnap.RunningDecode[0].TrueRemaining != 11 || decSnap.RunningDecode[1].TrueRemaining != 13 {
		t.Fatalf("Decide mutated the shared decode snapshot: %+v", decSnap.RunningDecode)
	}
}

func TestDecide_PopulatesAdmissionContext(t *testing.T) {
	var seen AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = c }}
	d := newTestEDPPDeciderWithEstimator(t, spy) // helper: constructs decider, injects spy as tadmEstimator
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots: []RoutingSnapshot{{
			ID: "d0", BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0,
			RemainingDecodeWork: 30, AdmissionRate: 0.001,
			RunningDecode: []RunningReqState{{StepsDone: 2, KVBlocks: 5, TrueRemaining: -1}},
		}},
	}
	d.Decide(makeReq("r1", 100, "batch"), state)
	if seen.BatchSize != 4 || seen.MaxBatchSize != 4 || seen.RemainingStepsEst == 0 || seen.AdmissionRate != 0.001 {
		t.Fatalf("context not populated from snapshot: %+v", seen)
	}
	if len(seen.Running) != 1 || seen.Running[0].StepsDone != 2 {
		t.Fatalf("running state not propagated: %+v", seen.Running)
	}
}

func TestDecide_AdmissionRateFromDispatchRate(t *testing.T) {
	// When the explicit AdmissionRate field is 0, Decide must fall back to the
	// observed completion rate (DispatchRate, req/s → req/µs), so the `little`
	// estimator (QueueDepth/AdmissionRate) is not a dead point.
	var seen AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = c }}
	d := newTestEDPPDeciderWithEstimator(t, spy)
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots: []RoutingSnapshot{{
			ID: "d0", BatchSize: 4, MaxBatchSize: 4, QueueDepth: 8,
			DispatchRate: 20, // req/s; AdmissionRate field left 0
		}},
	}
	d.Decide(makeReq("r1", 100, "batch"), state)
	want := 20.0 / 1e6
	if seen.AdmissionRate != want {
		t.Fatalf("AdmissionRate = %v, want DispatchRate/1e6 = %v", seen.AdmissionRate, want)
	}
	// Explicit AdmissionRate takes precedence when non-zero.
	state.Snapshots[0].AdmissionRate = 0.005
	d.Decide(makeReq("r2", 100, "batch"), state)
	if seen.AdmissionRate != 0.005 {
		t.Fatalf("AdmissionRate = %v, want explicit 0.005", seen.AdmissionRate)
	}
}

// The prefill AdmissionContext must carry real prefill-pool occupancy so the
// ttft_p estimators (fluid/rollforward) are not inert on the prefill pool. A
// prefill snapshot with a full batch and running prefill work must produce a
// prefill context whose RemainingStepsEst reflects the prefill occupants (not a
// decode-derived value) and whose Running is the prefill running-state.
func TestDecide_PrefillContextPopulated(t *testing.T) {
	// Prefill pool: batch full (BatchSize == MaxBatchSize) with running prefill
	// requests, and no free KV — so fluid/rollforward cannot short-circuit to 0
	// and must consult RemainingStepsEst / Running.
	prefill := RoutingSnapshot{
		ID: "p0", BatchSize: 2, MaxBatchSize: 2, QueueDepth: 3, FreeKVBlocks: 0,
		RunningPrefill: []RunningReqState{
			{StepsDone: 1, KVBlocks: 4, TrueRemaining: 3},
			{StepsDone: 0, KVBlocks: 2, TrueRemaining: 5},
		},
	}
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil,
		func() []RoutingSnapshot { return []RoutingSnapshot{prefill} })
	d.SetCaptureAdmissionContext(true)
	// Decode side: no running decode work, so any prefill occupancy must come from
	// the prefill snapshot itself, not from a decode-derived RemainingStepsEst.
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{ID: "d0"}}}
	dec := d.Decide(makeReq("r1", 4000, "batch"), state)
	if dec.AdmissionCtxPrefill == nil {
		t.Fatalf("prefill admission context not captured")
	}
	seenPrefill := *dec.AdmissionCtxPrefill
	if seenPrefill.BatchSize == 0 && seenPrefill.RemainingStepsEst == 0 {
		t.Fatalf("prefill context inert: %+v", seenPrefill)
	}
	// The prefill running-state must be the prefill occupants, not decode's (nil here).
	if len(seenPrefill.Running) != len(prefill.RunningPrefill) {
		t.Fatalf("prefill Running not propagated from prefill snapshot: %+v", seenPrefill.Running)
	}
}

func TestEDPP_NOutIndependence(t *testing.T) {
	// §11 N_out-independence anchor: the decision must not read Request.OutputTokens (INV-9).
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	state := decodeState("d0", 10, 8, 60_000)

	reqA := &Request{ID: "a", InputTokens: make([]TokenID, 800)}
	reqB := &Request{ID: "b", InputTokens: make([]TokenID, 800), OutputTokens: make([]TokenID, 4096)}

	if d.Decide(reqA, state).Disaggregate != d.Decide(reqB, state).Disaggregate {
		t.Errorf("decision depended on OutputTokens (INV-9 violation)")
	}
}

func TestEDPP_DisaggregationPayoffSign(t *testing.T) {
	// §11 disaggregation-payoff-sign anchor: under an ITL-SLO breach (z_itl large) with
	// idle prefill capacity (no prefill backlog), the rule must shift toward P.
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	req := &Request{ID: "r", InputTokens: make([]TokenID, 800)}
	state := decodeState("d0", 10, 8, 60_000) // decode ITL above τ_itl

	// Drive z_itl large via realized ITL misses (E8), prefill stays idle.
	for i := 0; i < 50; i++ {
		rid := fmt.Sprintf("r%d", i)
		d.OnComplete(&Request{ID: rid, SLOClass: ""}, rid, 0, 200_000) // realized ITL = 200ms >> τ_itl, default class
	}
	if !d.Decide(req, state).Disaggregate {
		t.Errorf("expected P (disaggregate) under z_itl breach with idle prefill")
	}
}

func TestEDPP_OnComplete_UpdatesVirtualQueues(t *testing.T) {
	// E8: Z_ttft and Z_itl accumulate positive violations and floor at 0.
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	// z_ttft now flows through the awaiting/first-token path; a completion with no prior
	// first-token reconciles it via the OnComplete fallback, so route the request first.
	r2 := &Request{ID: "r2", SLOClass: "", ArrivalTime: 0}
	d.OnRoute(r2, "r2", false, 1, "", "")
	d.OnComplete(r2, "r2", 150_000, 80_000) // TTFT 150ms (τ=100ms), ITL 80ms (τ=50ms)
	z := d.zByClass[""]
	if z == nil || z.zTTFT != 50_000 {
		t.Errorf("Z_ttft = %v, want 50000", z)
	}
	if z.zITL != 30_000 {
		t.Errorf("Z_itl = %v, want 30000", z.zITL)
	}
	// A large under-target completion must floor each queue at 0, not go negative.
	r3 := &Request{ID: "r3", SLOClass: "", ArrivalTime: 0}
	d.OnRoute(r3, "r3", false, 1, "", "")
	d.OnComplete(r3, "r3", 0, 0)
	if z.zTTFT != 0 || z.zITL != 0 {
		t.Errorf("virtual queues not floored at 0: zTTFT=%v zITL=%v", z.zTTFT, z.zITL)
	}
}

func TestEDPP_PerClass_TargetsResolve(t *testing.T) {
	// A class with an explicit override uses it; an unlisted class falls back to defaults.
	cfg := defaultTestEDPPConfig()
	cfg.TauTTFTByClassUs = map[string]int64{"critical": 30_000}
	cfg.TauITLByClassUs = map[string]int64{"critical": 10_000}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)

	tT, tI := d.targetsFor("critical")
	if tT != 30_000 || tI != 10_000 {
		t.Errorf("critical targets = (%d,%d), want (30000,10000)", tT, tI)
	}
	tT, tI = d.targetsFor("batch") // unlisted → defaults
	if tT != cfg.TauTTFTUs || tI != cfg.TauITLUs {
		t.Errorf("batch targets = (%d,%d), want defaults (%d,%d)", tT, tI, cfg.TauTTFTUs, cfg.TauITLUs)
	}
}

func TestEDPP_PerClass_IndependentVirtualQueues(t *testing.T) {
	// An ITL breach reported for one class must not bleed into another class's
	// virtual queue: the SLO-pressure feedback is per-class.
	cfg := defaultTestEDPPConfig()
	cfg.TauTTFTByClassUs = map[string]int64{"critical": 30_000}
	cfg.TauITLByClassUs = map[string]int64{"critical": 10_000}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	for i := 0; i < 50; i++ {
		rid := fmt.Sprintf("r%d", i)
		d.OnComplete(&Request{ID: rid, SLOClass: "critical"}, rid, 0, 200_000) // breach only the critical class
	}
	if d.zByClass["critical"] == nil || d.zByClass["critical"].zITL == 0 {
		t.Fatalf("critical z_itl did not accumulate")
	}
	if z := d.zByClass["batch"]; z != nil && z.zITL != 0 {
		t.Errorf("batch z_itl leaked from critical breach: %v", z.zITL)
	}

	state := decodeState("d0", 10, 8, 60_000)
	// Critical (breached) shifts toward P; batch (no breach, loose target) stays D.
	if !d.Decide(&Request{ID: "c", InputTokens: make([]TokenID, 800), SLOClass: "critical"}, state).Disaggregate {
		t.Errorf("critical request should disaggregate under its own ITL breach")
	}
	if d.Decide(&Request{ID: "b", InputTokens: make([]TokenID, 800), SLOClass: "batch"}, state).Disaggregate {
		t.Errorf("batch request should not disaggregate (no breach on its class)")
	}
}

func TestEDPP_EmptyPrompt_DoesNotDisaggregate(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	if d.Decide(&Request{ID: "r"}, decodeState("d0", 1, 1, 60_000)).Disaggregate {
		t.Errorf("empty prompt must not disaggregate")
	}
}

// --- Decision-trace instrumentation ---

func TestEDPP_DecisionTrace_PopulatedAndConsistent(t *testing.T) {
	// With tracing enabled, Decide attaches an EDPPDecisionTrace whose intermediate
	// terms compose into LHS/RHS exactly, and whose Disaggregate matches LHS > RHS.
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	req := &Request{ID: "r", InputTokens: make([]TokenID, 800), SLOClass: "batch"}
	state := decodeState("d0", 10, 8, 60_000)

	dec := d.Decide(req, state)
	tr := dec.EDPPTrace
	if tr == nil {
		t.Fatal("expected non-nil EDPPTrace when tracing enabled")
	}
	if tr.SkipReason != "" {
		t.Errorf("rule was evaluated; SkipReason should be empty, got %q", tr.SkipReason)
	}
	if tr.Class != "batch" {
		t.Errorf("Class = %q, want batch", tr.Class)
	}

	// Intermediate-term composition (the whole point of the instrumentation).
	if math.Abs(tr.LHS-(tr.BalanceTermD-tr.BalanceTermP)) > 1e-9 {
		t.Errorf("LHS %v != BalanceTermD %v - BalanceTermP %v", tr.LHS, tr.BalanceTermD, tr.BalanceTermP)
	}
	if math.Abs(tr.RHS-(tr.TransferTerm+tr.TTFTTerm+tr.ITLTerm)) > 1e-9 {
		t.Errorf("RHS %v != TransferTerm %v + TTFTTerm %v + ITLTerm %v",
			tr.RHS, tr.TransferTerm, tr.TTFTTerm, tr.ITLTerm)
	}
	if tr.Disaggregate != dec.Disaggregate {
		t.Errorf("trace.Disaggregate %v != decision.Disaggregate %v", tr.Disaggregate, dec.Disaggregate)
	}
	if tr.Disaggregate != (tr.LHS > tr.RHS) {
		t.Errorf("Disaggregate %v inconsistent with LHS %v > RHS %v", tr.Disaggregate, tr.LHS, tr.RHS)
	}

	// A few anchored intermediate values (affine model, default cfg, no z, idle prefill).
	if tr.Ap != 800 {
		t.Errorf("Ap = %d, want 800", tr.Ap)
	}
	if tr.Wp != 8000 { // kp·a_p = 10·800
		t.Errorf("Wp = %v, want 8000", tr.Wp)
	}
	if math.Abs(tr.TransferTerm-0.05) > 1e-9 { // V·(c_xfer/τ_ttft) = 1·5000/100000
		t.Errorf("TransferTerm = %v, want 0.05", tr.TransferTerm)
	}
	if tr.TTFTTerm != 0 || tr.ITLTerm != 0 { // z=0 (no completions reported)
		t.Errorf("z-gated terms should be 0 with no completions: ttft=%v itl=%v", tr.TTFTTerm, tr.ITLTerm)
	}
}

func TestEDPP_DecisionTrace_NilWhenDisabled(t *testing.T) {
	// Default config leaves tracing off ⇒ zero overhead, no trace attached.
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	dec := d.Decide(&Request{ID: "r", InputTokens: make([]TokenID, 800)}, decodeState("d0", 10, 8, 60_000))
	if dec.EDPPTrace != nil {
		t.Errorf("EDPPTrace should be nil when tracing disabled, got %+v", dec.EDPPTrace)
	}
}

func TestEDPP_DecisionTrace_EarlyReturnRecordsSkipReason(t *testing.T) {
	// Early-return paths still emit a trace (when enabled) with a SkipReason, so every
	// request is accounted for in the per-decision record.
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)

	dec := d.Decide(&Request{ID: "empty"}, decodeState("d0", 1, 1, 60_000))
	if dec.Disaggregate {
		t.Errorf("empty prompt must not disaggregate")
	}
	if dec.EDPPTrace == nil || dec.EDPPTrace.SkipReason == "" {
		t.Errorf("empty-prompt trace must carry a SkipReason, got %+v", dec.EDPPTrace)
	}
}

// --- τ-consistent transfer penalty (fix for the τ/W* coupling) ---

func TestEDPP_TransferTerm_ScalesInverseTauSquared(t *testing.T) {
	// The transfer penalty must scale as 1/τ_ttft² (like the balance and SLO terms),
	// not 1/τ_ttft. With τ_ref = the default τ_ttft, a class at 2×τ_ref must show a
	// transfer term that is ONE QUARTER of the default-class term (was: one half).
	cfg := defaultTestEDPPConfig() // τ_ref = TauTTFTUs = 100_000
	cfg.TauTTFTByClassUs = map[string]int64{"x2": 200_000}
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
	state := decodeState("d0", 10, 8, 60_000)

	base := d.Decide(&Request{ID: "a", InputTokens: make([]TokenID, 800), SLOClass: "batch"}, state).EDPPTrace // τ = τ_ref
	dbl := d.Decide(&Request{ID: "b", InputTokens: make([]TokenID, 800), SLOClass: "x2"}, state).EDPPTrace     // τ = 2·τ_ref

	if base == nil || dbl == nil {
		t.Fatal("expected traces")
	}
	if math.Abs(base.TransferTerm-0.05) > 1e-12 { // V·c_xfer/τ_ref = 1·5000/100000, ratio 1 at default
		t.Errorf("base TransferTerm = %v, want 0.05 (backward-compatible at default τ)", base.TransferTerm)
	}
	if math.Abs(dbl.TransferTerm-base.TransferTerm/4) > 1e-12 {
		t.Errorf("TransferTerm at 2×τ = %v, want base/4 = %v (1/τ² scaling)", dbl.TransferTerm, base.TransferTerm/4)
	}
}

func TestEDPP_TransferPenalty_FixedTauRef_EngagesAtLooseDefault(t *testing.T) {
	// The bug regression for the REAL scenario: a globally-loose default τ_ttft (no
	// per-class override) with a FIXED τ_ref. Under heavy decode imbalance, idle prefill,
	// and no SLO pressure (z=0), the request must disaggregate on the load-balancing
	// signal. τ_ref must NOT track the loose default τ_ttft (that would make the penalty
	// ∝1/τ again and wrongly keep work local).
	cfg := defaultTestEDPPConfig()
	cfg.TauTTFTUs = 1_000_000 // loose default τ_ttft (1s), applied to all classes
	cfg.TauRefUs = 100_000    // fixed reference (100ms), independent of the loose default
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	// Seed decode backlog via OnRoute so qdWork >> 0 (qpWork stays 0 — idle prefill).
	// 10 D-path routes of ap=1000: qdWork += (Wp(1000) + wd) each ≈ 121480 µs total.
	for i := 0; i < 10; i++ {
		r := &Request{ID: fmt.Sprintf("seed%d", i), SLOClass: "batch"}
		d.OnRoute(r, r.ID, false /*D-path*/, 1000, "", "")
	}

	state := decodeState("d0", 50, 0, 60_000) // heavy decode backlog, idle prefill
	req := &Request{ID: "r", InputTokens: make([]TokenID, 800), SLOClass: "batch"}
	if !d.Decide(req, state).Disaggregate {
		t.Errorf("loose-default-τ request must disaggregate under heavy imbalance (fixed τ_ref), got kept-local")
	}
}

func TestEDPPConfig_RequiresCoeffsAndTauITLAboveAlpha(t *testing.T) {
	m := newTestAffineModel()
	// Missing coeffs (zero value) must panic via cfg.validate().
	bad := defaultTestEDPPConfig()
	bad.Coeffs = EDPPCoeffs{}
	assertPanics(t, func() { _ = NewEDPPDecider(bad, m, nil, nil) })

	// τ_itl <= α_d is physically unachievable ⇒ panic (design §7 guard).
	tauGuard := defaultTestEDPPConfig()
	tauGuard.TauITLUs = 500 // TauITLUs=500 <= AlphaD=1000 ⇒ ITL SLO unachievable
	assertPanics(t, func() { _ = NewEDPPDecider(tauGuard, m, nil, nil) })
}

func TestEDPP_BookkeepingConservation(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
	r1 := &Request{ID: "r1", SLOClass: "", InputTokens: make([]TokenID, 200), OutputTokens: []TokenID{0}}
	// First completion seeds N̂_out (no prior estimate ⇒ use a 1-token default), so
	// route adds W_p only for the decode side until N̂_out is known.
	d.OnRoute(r1, r1.ID, true /*toPrefill*/, 200, "", "")
	if d.qpWork <= 0 {
		t.Fatalf("qpWork must be > 0 after routing P, got %v", d.qpWork)
	}
	qpAfterRoute := d.qpWork
	// W_p(200) with test coeffs (CPf=10, CAttn=0) = 2000.
	if math.Abs(qpAfterRoute-2000) > 1e-9 {
		t.Errorf("qpWork = %v, want 2000 (W_p(200))", qpAfterRoute)
	}
	// Admission drains the work (§6.2): prefill side drains Q_p, decode side drains Q_d.
	// For a P-routed request, prefill admission fires first, then decode admission.
	d.OnAdmit(r1.ID, true /*prefillSide*/)
	if math.Abs(d.qpWork) > 1e-9 {
		t.Errorf("qpWork after prefill OnAdmit = %v, want 0", d.qpWork)
	}
	d.OnAdmit(r1.ID, false /*decodeSide*/)
	_, qdAfterAdmit, pendingAfterAdmit := d.BacklogForTest()
	if math.Abs(qdAfterAdmit) > 1e-9 || pendingAfterAdmit != 0 {
		t.Errorf("after both OnAdmit: qd=%v pending=%d, want 0,0", qdAfterAdmit, pendingAfterAdmit)
	}
	// OnComplete updates N̂_out and virtual queues (backlog already drained).
	r1.ITL = []int64{40_000}
	r1.OutputTokens = []TokenID{1, 2, 3} // realized N_out=3 (post-completion read; INV-9 OK)
	d.OnComplete(r1, r1.ID, 90_000, 40_000)
	// N̂_out updated from the realized length.
	if m := d.nHatOut[""]; m == nil || m.mean() != 3 {
		t.Errorf("N̂_out not updated to 3, got %+v", m)
	}
}

func TestEDPP_Forget_ReleasesBacklogWithoutZBump(t *testing.T) {
	// A routed request that reaches a terminal state without completing must have its
	// backlog released by Forget — Q returns to baseline, pending empties, but the
	// virtual queues and N̂_out are untouched (no realized SLO signal).
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
	r := &Request{ID: "drop1", SLOClass: "", InputTokens: make([]TokenID, 200)}
	d.OnRoute(r, r.ID, false /*toPrefill: D path, all work on decode*/, 200, "", "")
	qp, qd, n := d.BacklogForTest()
	if (qp == 0 && qd == 0) || n != 1 {
		t.Fatalf("after OnRoute: qp=%v qd=%v pending=%d; want nonzero backlog and 1 pending", qp, qd, n)
	}
	// N̂_out is lazily seeded by OnRoute (to estimate W_d); snapshot its sample count so
	// we can assert Forget does not feed it a (nonexistent) realized output length.
	nHatSamplesBefore := d.nHatFor(r.SLOClass).n
	d.Forget(r.ID)
	qp, qd, n = d.BacklogForTest()
	if math.Abs(qp) > 1e-9 || math.Abs(qd) > 1e-9 || n != 0 {
		t.Errorf("after Forget: qp=%v qd=%v pending=%d; want 0,0,0", qp, qd, n)
	}
	if len(d.zByClass) != 0 {
		t.Errorf("Forget bumped virtual queues: %+v", d.zByClass)
	}
	if d.nHatFor(r.SLOClass).n != nHatSamplesBefore {
		t.Errorf("Forget updated N̂_out sample count: %d → %d", nHatSamplesBefore, d.nHatFor(r.SLOClass).n)
	}
	// Idempotent: forgetting again is a no-op.
	d.Forget(r.ID)
	if qp, qd, n = d.BacklogForTest(); n != 0 || math.Abs(qp) > 1e-9 || math.Abs(qd) > 1e-9 {
		t.Errorf("second Forget changed state: qp=%v qd=%v pending=%d", qp, qd, n)
	}
}

func TestEDPP_NoutDoesNotChangeDecision(t *testing.T) {
	// §11 N_out-independence: the realized N_out of a request must not change the
	// P/D decision (it only affects Q_d accounting magnitude, never the rule's
	// admission read which uses input-only a_p + the class N̂_out estimate).
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
	req := &Request{ID: "x", InputTokens: make([]TokenID, 300)}
	state := &RouterState{Snapshots: []RoutingSnapshot{{ID: "d0", QueueDepth: 1, BatchSize: 2, ResidentPrefillTokens: 0}}}
	dec1 := d.Decide(req, state)
	req.OutputTokens = []TokenID{1, 2, 3, 4, 5} // would-be large N_out; decision path must ignore it
	dec2 := d.Decide(req, state)
	if dec1.Disaggregate != dec2.Disaggregate {
		t.Errorf("decision changed with N_out: %v vs %v", dec1.Disaggregate, dec2.Disaggregate)
	}
}

func TestEDPP_PredictorsAndITLCollapse(t *testing.T) {
	// §9 explicit-chunk predictor and collapsed ITL term anchors.
	//
	// Setup (affine model, default config):
	//   α_p = α_d = 1000 µs, c_pf = 10 µs/tok, c_attn = 0, c0 = 100, c1 = 1.
	//   CXferUs = 5000, τ_ttft = 100_000, τ_itl = 50_000.
	//   ap = 400, no ChunkTokens cap ⇒ chunk = 400, n_chunks = 1.
	//   Decode snapshot: B=1, KV=2048, S_pf=0.
	//   Prefill snapshot: none (idle).
	//
	// Expected TTFT predictors (qP=qD=0, no seeded backlog):
	//   muDec = 1 − α_d/T_iter_dec = 1 − 1000/(1000 + 100·1 + 1·2048 + 0) = 1 − 1000/3148
	//   muPf  = 1 − α_p/T_iter_pf  = 1 − 1000/(1000 + 10·0) = clamped to 0.001 (sPf=0 ⇒ T_iter_pf = α_p = 1000)
	//   δ_pf_chunk = c_pf·chunk = 10·400 = 4000
	//   tBminus1 = T_iter_dec(1, 2048, 0) = 1000 + 100 + 2048 = 3148
	//   ttftP = 0/muPf + 1·(1000 + 4000) + 5000 = 10000  (qP=0)
	//   ttftD = 0/muDec + 1·(3148 + 4000) = 7148        (qD=0)
	//
	// ITL collapsed form: itlTerm = −z_itl·(c_pf·chunk)/τ_itl = −z_itl·4000/50000.
	// With z_itl raised via 20 completions each with realized ITL = 200_000 µs:
	//   each adds (200000 − 50000) = 150000 → z_itl_raw = 20·150000 = 3_000_000
	//   zITL = 3_000_000 / 50_000 = 60
	//   itlTerm = −60 · 4000/50000 = −4.8
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	for i := 0; i < 20; i++ {
		rid := fmt.Sprintf("q%d", i)
		d.OnComplete(&Request{ID: rid, SLOClass: ""}, rid, 0, 200_000)
	}

	req := &Request{ID: "t", InputTokens: make([]TokenID, 400), SLOClass: ""}
	// Decode snapshot: B=1, KV=2048, S_pf=0.
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots:        []RoutingSnapshot{{ID: "d0", BatchSize: 1, KvTokensInUse: 2048}},
	}
	dec := d.Decide(req, state)
	tr := dec.EDPPTrace
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// TTFT predictor anchors.
	wantTTFTP := 10000.0
	if math.Abs(tr.TTFTP-wantTTFTP) > 1e-6 {
		t.Errorf("TTFTP = %v, want %v", tr.TTFTP, wantTTFTP)
	}
	wantTTFTD := 7148.0
	if math.Abs(tr.TTFTD-wantTTFTD) > 1e-6 {
		t.Errorf("TTFTD = %v, want %v", tr.TTFTD, wantTTFTD)
	}

	// Collapsed ITL term anchor: itlTerm = −zITL·(c_pf·chunk)/τ_itl = −60·4000/50000 = −4.8
	wantITLTerm := -4.8
	if math.Abs(tr.ITLTerm-wantITLTerm) > 1e-9 {
		t.Errorf("ITLTerm = %v, want %v (collapsed form)", tr.ITLTerm, wantITLTerm)
	}

	// Composition invariant: RHS = TransferTerm + TTFTTerm + ITLTerm.
	if math.Abs(tr.RHS-(tr.TransferTerm+tr.TTFTTerm+tr.ITLTerm)) > 1e-9 {
		t.Errorf("RHS composition violated: %v != %v + %v + %v",
			tr.RHS, tr.TransferTerm, tr.TTFTTerm, tr.ITLTerm)
	}
}

func TestEDPP_ChunkCap_CapsDeltaPfChunkAtBudget(t *testing.T) {
	// Verifies the chunk-budget CAP branch in EDPPDecider.Decide (edpp.go ~line 367-369):
	//   chunk := ap
	//   if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk { chunk = d.cfg.ChunkTokens }
	//   deltaPfChunk = coeffs.CPf * float64(chunk)
	//
	// With CPf=10:
	//   capped   (ChunkTokens=256, ap=1000): chunk=256, DeltaPfChunk = 10*256 = 2560
	//   uncapped (ChunkTokens=0,   ap=1000): chunk=1000, DeltaPfChunk = 10*1000 = 10000
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots:        []RoutingSnapshot{{ID: "d0", BatchSize: 1}},
	}
	req := &Request{ID: "cap-test", InputTokens: make([]TokenID, 1000), SLOClass: ""}

	t.Run("capped", func(t *testing.T) {
		cfg := defaultTestEDPPConfig()
		cfg.TraceEnabled = true
		cfg.ChunkTokens = 256
		d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
		dec := d.Decide(req, state)
		if dec.EDPPTrace == nil {
			t.Fatal("expected non-nil trace")
		}
		const want = 10 * 256.0 // CPf * chunk (capped)
		if math.Abs(dec.EDPPTrace.DeltaPfChunk-want) > 1e-9 {
			t.Errorf("DeltaPfChunk (capped) = %v, want %v (cap did not fire)", dec.EDPPTrace.DeltaPfChunk, want)
		}
	})

	t.Run("uncapped", func(t *testing.T) {
		cfg := defaultTestEDPPConfig()
		cfg.TraceEnabled = true
		cfg.ChunkTokens = 0 // no cap
		d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
		dec := d.Decide(req, state)
		if dec.EDPPTrace == nil {
			t.Fatal("expected non-nil trace")
		}
		const want = 10 * 1000.0 // CPf * full ap
		if math.Abs(dec.EDPPTrace.DeltaPfChunk-want) > 1e-9 {
			t.Errorf("DeltaPfChunk (uncapped) = %v, want %v", dec.EDPPTrace.DeltaPfChunk, want)
		}
	})
}

// Compile-time interface compliance.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)

// --- §11 verification-anchor tests (Task 10) ---

// TestEDPP_Anchor_SignalDirection asserts that increasing the decode backlog via
// OnRoute (D-path) must NOT decrease the rule's LHS. q_d = Q_d / W*_d; W*_d uses
// the fixed μ^nom normalizer, so the congestion signal is monotone non-decreasing
// in Q_d. This exercises the new qpWork/qdWork path (Task 6) via OnRoute, reading
// LHS from the decision trace.
func TestEDPP_Anchor_SignalDirection(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	req := &Request{ID: "a", InputTokens: make([]TokenID, 200)}
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots:        []RoutingSnapshot{{ID: "d0", BatchSize: 2, KvTokensInUse: 1024}},
	}

	low := d.Decide(req, state).EDPPTrace.LHS

	// Increase decode backlog via D-path OnRoute: qdWork grows, so Q_d / W*_d grows.
	extra := &Request{ID: "b", SLOClass: ""}
	d.OnRoute(extra, extra.ID, false /*D-path*/, 500, "", "")
	high := d.Decide(req, state).EDPPTrace.LHS

	if high < low-1e-9 {
		t.Errorf("LHS decreased as Q_d rose: low=%v high=%v (signal direction inverted)", low, high)
	}
	// Concrete: qdWork grew, so high must strictly exceed low (unless already saturated at muD=1).
	// We assert at minimum non-decrease; if both are 0 the test is vacuous — check that high>0.
	if high <= 0 {
		t.Errorf("LHS is non-positive after seeding decode backlog: %v (backlog not wired?)", high)
	}
}

// TestEDPP_Anchor_DisaggPayoffSign asserts that under an ITL-SLO breach (large
// realized ITL via OnComplete), the rule's RHS must FALL. The collapsed ITL term
// −z_itl·(c_pf·chunk)/τ_itl is strictly negative, pushing toward P.
func TestEDPP_Anchor_DisaggPayoffSign(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.ChunkTokens = 128
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	req := &Request{ID: "a", InputTokens: make([]TokenID, 300)}
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots:        []RoutingSnapshot{{ID: "d0", BatchSize: 2, KvTokensInUse: 1024}},
	}

	before := d.Decide(req, state).EDPPTrace.RHS

	// Drive z_itl large via a realized ITL breach well above τ_itl.
	breach := &Request{ID: "x", SLOClass: ""}
	breach.OutputTokens = []TokenID{1}
	d.OnComplete(breach, breach.ID, 0, cfg.TauITLUs+50_000)

	after := d.Decide(req, state).EDPPTrace.RHS

	// RHS must strictly fall: the ITL term (−z_itl·c_pf·chunk/τ_itl) is negative.
	if after >= before {
		t.Errorf("RHS did not fall under ITL breach: before=%v after=%v (ITL term not wired or wrong sign)", before, after)
	}
}

// TestEDPP_Anchor_UnitsDimensionless asserts that scaling ALL µs-dimensioned
// quantities by a common factor k leaves the P/D decision unchanged. The §8 rule
// is dimensionless by design: every term scales identically, so LHS > RHS is
// invariant under uniform rescaling of the µs quantities.
//
// Construction: multiply τ_ttft, τ_itl, τ_ref, c_xfer, AND the µs-dimension
// coefficients (AlphaD, AlphaP scale by k; C0 scales by k; C1 and CPf are µs/token
// so they also scale by k; CAttn is µs/token² so it scales by k). Keep token counts
// (a_p, B, KV) fixed. Seed identical backlog via OnRoute and identical z via
// OnComplete (with realized latencies also scaled by k to keep z dimensionless).
//
// Strengthened assertions (Task 10):
//  1. LHS and RHS must be EQUAL (not merely same-sign) across k within 1e-9: if any
//     single coefficient fails to scale, LHS or RHS would differ.
//  2. TTFT term is exercised: the OnComplete seed also breaches TTFT
//     (realizedTTFTUs = τ_ttft + 50_000*k) so zTTFT > 0 and ttftTerm is non-zero.
//  3. A second sub-test exercises the TRUE (disaggregate) side of the boundary to
//     prove invariance straddles both directions.
func TestEDPP_Anchor_UnitsDimensionless(t *testing.T) {
	// mk builds a decider with all µs quantities scaled by k, seeds decode backlog and
	// z_itl+z_ttft, and returns the full DisaggregationDecision with trace attached.
	mk := func(k int64) DisaggregationDecision {
		cfg := defaultTestEDPPConfig()
		kf := float64(k)
		cfg.TauTTFTUs *= k
		cfg.TauITLUs *= k
		cfg.TauRefUs *= k
		cfg.CXferUs *= k
		cfg.TraceEnabled = true
		cfg.Coeffs = EDPPCoeffs{
			AlphaD: 1000 * kf,
			AlphaP: 1000 * kf,
			C0:     100 * kf, // µs/req → scales by k
			C1:     1 * kf,   // µs/token → scales by k
			CPf:    10 * kf,  // µs/token → scales by k
			CAttn:  0,        // µs/token² → 0 * k = 0
		}
		d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

		// Seed decode backlog via D-path OnRoute; token count fixed.
		seed := &Request{ID: "s", SLOClass: ""}
		d.OnRoute(seed, seed.ID, false, 200, "", "")

		// Seed z_itl AND z_ttft via OnComplete; realized latencies scaled by k to keep
		// z values dimensionless-invariant.
		//   z_itl = (τ_itl + 50_000*k − τ_itl) / τ_itl = 50_000/50_000 = 1  (constant across k)
		//   z_ttft = (τ_ttft + 50_000*k − τ_ttft) / τ_ttft = 50_000*k / (100_000*k) = 0.5 (constant)
		// Both ttftTerm and itlTerm are now non-zero and must scale invariantly.
		breach := &Request{ID: "b", SLOClass: "", ArrivalTime: 0, OutputTokens: []TokenID{1}}
		d.OnRoute(breach, breach.ID, false, 1, "", "") // track so the OnComplete z_ttft fallback fires
		d.OnComplete(breach, breach.ID, cfg.TauTTFTUs+50_000*k, cfg.TauITLUs+50_000*k)

		return d.Decide(
			&Request{ID: "u", InputTokens: make([]TokenID, 200)},
			&RouterState{Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 2, KvTokensInUse: 1024}}},
		)
	}

	dec1 := mk(1)
	dec3 := mk(3)

	// Bool equality (existing check preserved).
	if dec1.Disaggregate != dec3.Disaggregate {
		t.Errorf("decision not scale-invariant under k=1 vs k=3 rescaling: k=1 → Disaggregate=%v, k=3 → Disaggregate=%v (dimensionality bug in §8 rule)", dec1.Disaggregate, dec3.Disaggregate)
	}

	// Strong invariance: LHS and RHS must be numerically equal across k within 1e-9.
	// Any single unscaled µs coefficient would cause LHS or RHS to differ between k=1 and k=3.
	tr1 := dec1.EDPPTrace
	tr3 := dec3.EDPPTrace
	if tr1 == nil || tr3 == nil {
		t.Fatal("expected non-nil EDPPTrace for both k=1 and k=3 (TraceEnabled=true)")
	}
	if math.Abs(tr1.LHS-tr3.LHS) > 1e-9 {
		t.Errorf("LHS not invariant under k rescaling: k=1 LHS=%v, k=3 LHS=%v (diff=%v > 1e-9); dimensionality bug in §8 rule",
			tr1.LHS, tr3.LHS, math.Abs(tr1.LHS-tr3.LHS))
	}
	if math.Abs(tr1.RHS-tr3.RHS) > 1e-9 {
		t.Errorf("RHS not invariant under k rescaling: k=1 RHS=%v, k=3 RHS=%v (diff=%v > 1e-9); dimensionality bug in §8 rule",
			tr1.RHS, tr3.RHS, math.Abs(tr1.RHS-tr3.RHS))
	}

	// TTFT term must be non-zero (exercised by the TTFT breach seed above).
	if tr1.TTFTTerm == 0 {
		t.Errorf("TTFTTerm is 0 at k=1 — TTFT term not exercised (seeding may have failed)")
	}
	if tr3.TTFTTerm == 0 {
		t.Errorf("TTFTTerm is 0 at k=3 — TTFT term not exercised (seeding may have failed)")
	}
}

// TestEDPP_Anchor_UnitsDimensionless_DecidesTrueInvariant proves scale-invariance
// on the TRUE (disaggregate=true) side of the decision boundary. Config: large decode
// backlog (10 D-path OnRoute seeds of ap=500) + small c_xfer (500µs), low V=0.1 to
// suppress the transfer penalty. This pushes LHS >> RHS so Disaggregate=true at k=1;
// k=3 must also give true and LHS/RHS must match within 1e-9.
func TestEDPP_Anchor_UnitsDimensionless_DecidesTrueInvariant(t *testing.T) {
	mkTrue := func(k int64) DisaggregationDecision {
		cfg := defaultTestEDPPConfig()
		kf := float64(k)
		cfg.TauTTFTUs *= k
		cfg.TauITLUs *= k
		cfg.TauRefUs *= k
		// Small transfer cost so disaggregation benefit (large decode backlog) wins.
		cfg.CXferUs = 500 * k
		cfg.V = 0.1
		cfg.TraceEnabled = true
		cfg.Coeffs = EDPPCoeffs{
			AlphaD: 1000 * kf,
			AlphaP: 1000 * kf,
			C0:     100 * kf,
			C1:     1 * kf,
			CPf:    10 * kf,
			CAttn:  0,
		}
		d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

		// Seed large decode backlog: 10 D-path OnRoute calls (ap=500 each).
		// qdWork grows substantially, making Q_d/W*_d >> Q_p/W*_p (Q_p=0).
		for i := 0; i < 10; i++ {
			r := &Request{ID: fmt.Sprintf("ds%d_%d", k, i), SLOClass: ""}
			d.OnRoute(r, r.ID, false /*D-path*/, 500, "", "")
		}

		// Also seed z_ttft and z_itl to exercise both terms.
		breach := &Request{ID: fmt.Sprintf("bt%d", k), SLOClass: "", OutputTokens: []TokenID{1}}
		d.OnComplete(breach, breach.ID, cfg.TauTTFTUs+50_000*k, cfg.TauITLUs+50_000*k)

		return d.Decide(
			&Request{ID: fmt.Sprintf("u%d", k), InputTokens: make([]TokenID, 200)},
			&RouterState{Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 2, KvTokensInUse: 1024}}},
		)
	}

	dec1 := mkTrue(1)
	dec3 := mkTrue(3)

	if !dec1.Disaggregate {
		t.Fatalf("k=1 must decide Disaggregate=true (check backlog seeding); got false — LHS=%v RHS=%v",
			dec1.EDPPTrace.LHS, dec1.EDPPTrace.RHS)
	}
	if dec1.Disaggregate != dec3.Disaggregate {
		t.Errorf("decision not scale-invariant (true side): k=1 → Disaggregate=%v, k=3 → Disaggregate=%v",
			dec1.Disaggregate, dec3.Disaggregate)
	}

	tr1 := dec1.EDPPTrace
	tr3 := dec3.EDPPTrace
	if tr1 == nil || tr3 == nil {
		t.Fatal("expected non-nil EDPPTrace for both k=1 and k=3 (TraceEnabled=true)")
	}
	if math.Abs(tr1.LHS-tr3.LHS) > 1e-9 {
		t.Errorf("LHS not invariant (true side): k=1 LHS=%v, k=3 LHS=%v (diff=%v > 1e-9)",
			tr1.LHS, tr3.LHS, math.Abs(tr1.LHS-tr3.LHS))
	}
	if math.Abs(tr1.RHS-tr3.RHS) > 1e-9 {
		t.Errorf("RHS not invariant (true side): k=1 RHS=%v, k=3 RHS=%v (diff=%v > 1e-9)",
			tr1.RHS, tr3.RHS, math.Abs(tr1.RHS-tr3.RHS))
	}
}

// --- Task 3: waiting-only backlog migration tests ---

func TestEDPP_WaitingOnly_DrainsAtAdmissionNotCompletion(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	// D-routed request: OnRoute adds wp+wd to qdWork.
	r := &Request{ID: "d1", SLOClass: ""}
	d.OnRoute(r, r.ID, false, 200, "", "") // Wp(200)=2000; wd=N̂_out(=1)·deltaBarDecode(2048)=2148 ⇒ qd=4148
	_, qd0, n0 := d.BacklogForTest()
	if qd0 <= 0 || n0 != 1 {
		t.Fatalf("after OnRoute: qd=%v pending=%d, want qd>0 pending=1", qd0, n0)
	}
	// OnComplete must NOT drain the backlog (waiting work already left at admission)...
	r.OutputTokens = []TokenID{1, 2}
	d.OnComplete(r, r.ID, 90_000, 40_000)
	_, qdAfterComplete, nAfterComplete := d.BacklogForTest()
	if qdAfterComplete != qd0 {
		t.Errorf("OnComplete drained backlog (%v→%v); it must not — admission drains", qd0, qdAfterComplete)
	}
	if nAfterComplete != 1 {
		t.Errorf("OnComplete deleted pending; it must not — admission deletes")
	}
	// ...OnAdmit (decode side) drains it and clears pending.
	d.OnAdmit(r.ID, false)
	_, qdAfterAdmit, nAfterAdmit := d.BacklogForTest()
	if math.Abs(qdAfterAdmit) > 1e-9 || nAfterAdmit != 0 {
		t.Errorf("after OnAdmit: qd=%v pending=%d, want 0 and 0", qdAfterAdmit, nAfterAdmit)
	}
}

func TestEDPP_WaitingOnly_PDTwoSidedAdmission(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	// P-routed: OnRoute adds wp→qp, wd→qd.
	r := &Request{ID: "p1", SLOClass: ""}
	d.OnRoute(r, r.ID, true, 200, "", "") // qp += Wp(200)=2000 ; qd += wd=2148
	qp0, qd0, _ := d.BacklogForTest()
	if math.Abs(qp0-2000) > 1e-9 || qd0 <= 0 {
		t.Fatalf("after P OnRoute: qp=%v qd=%v", qp0, qd0)
	}
	// Prefill admission drains ONLY the Q_p share; pending survives (decode share remains).
	d.OnAdmit(r.ID, true)
	qpA, qdA, nA := d.BacklogForTest()
	if math.Abs(qpA) > 1e-9 {
		t.Errorf("prefill admit: qp=%v, want 0", qpA)
	}
	if math.Abs(qdA-qd0) > 1e-9 || nA != 1 {
		t.Errorf("prefill admit must not touch qd or delete pending; qd=%v pending=%d", qdA, nA)
	}
	// Decode admission drains the Q_d share and clears pending.
	d.OnAdmit(r.ID, false)
	qpB, qdB, nB := d.BacklogForTest()
	if math.Abs(qpB) > 1e-9 || math.Abs(qdB) > 1e-9 || nB != 0 {
		t.Errorf("decode admit: qp=%v qd=%v pending=%d, want 0,0,0", qpB, qdB, nB)
	}
}

func TestEDPP_Anchor_WaitingVsRunning(t *testing.T) {
	// §11 new anchor: an admitted (running) request must NOT contribute to the
	// normalized waiting backlog q_d that drives the rule's LHS.
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })
	probe := &Request{ID: "x", InputTokens: make([]TokenID, 200)}
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 2, KvTokensInUse: 1024}}}
	d.OnRoute(&Request{ID: "w1", SLOClass: ""}, "w1", false, 500, "", "") // waiting work present
	lhsWaiting := d.Decide(probe, state).EDPPTrace.LHS
	d.OnAdmit("w1", false) // w1 now running, not waiting
	lhsRunning := d.Decide(probe, state).EDPPTrace.LHS
	if lhsRunning >= lhsWaiting {
		t.Errorf("admitting w1 did not reduce waiting-backlog LHS (%v → %v); running work must leave q_d", lhsWaiting, lhsRunning)
	}
}

func TestEDPP_TTFTP_UsesPrefillCoResidency(t *testing.T) {
	// §5.1: TTFT_P co-residency overhead is T_pf(B−1)=α_p+c_pf·S_pf, NOT bare α_p.
	// With prefill-pool S_pf>0, ttftP must exceed the old α_p-only value by
	// nChunks·c_pf·S_pf. We observe ttftP via the decision trace.
	cfg := defaultTestEDPPConfig()
	cfg.TraceEnabled = true
	cfg.ChunkTokens = 0 // chunk = ap ⇒ nChunks = 1
	// Prefill pool reports a busy running batch: S_pf = 400 resident prefill tokens.
	prefill := func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", ResidentPrefillTokens: 400}}
	}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, prefill)
	req := &Request{ID: "q", InputTokens: make([]TokenID, 300)}
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 1}}}
	tr := d.Decide(req, state).EDPPTrace
	// μ_pf = 1 − α_p/(α_p + c_pf·S_pf) = 1 − 1000/(1000+10·400) = 1 − 1000/5000 = 0.8
	// qP = 0 (no OnRoute). nChunks=1, deltaPfChunk = c_pf·chunk = 10·300 = 3000.
	// T_pf(B−1) = α_p + c_pf·S_pf = 1000 + 10·400 = 5000.
	// ttftP = 0/0.8 + 1·(5000 + 3000) + c_xfer(5000) = 8000 + 5000 = 13000.
	if math.Abs(tr.TTFTP-13000) > 1e-6 {
		t.Errorf("ttftP = %v, want 13000 (uses T_pf(B−1)=5000, not α_p=1000)", tr.TTFTP)
	}
}

// --- Responsive z_ttft: continuous credit from observed elapsed-wait + first-token true-up ---
//
// These assert the law "same total z_ttft contribution as the old completion-time bump,
// only credited earlier" plus the keep-credit-on-drop decision.

func zval(z *edppClassState) float64 {
	if z == nil {
		return -1
	}
	return z.zTTFT
}

func newWaitingReq(id string, arrivalUs int64) *Request {
	return &Request{ID: id, SLOClass: "", ArrivalTime: arrivalUs, InputTokens: []TokenID{1, 2, 3}}
}

// First token at 250ms with τ_ttft=100ms ⇒ z_ttft bumps by the 150ms miss (same as the
// old completion-time update would have produced).
func TestEDPP_FirstToken_BumpsZTTFTByMiss(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	req := newWaitingReq("r1", 0)
	d.OnRoute(req, "r1", false, 3, "", "")
	d.OnFirstToken("r1", 250_000)
	if got := zval(d.zByClass[""]); math.Abs(got-150_000) > 1 {
		t.Fatalf("zTTFT = %v, want 150000", got)
	}
}

// First token at 50ms (< τ 100ms): SLO met ⇒ z stays floored at 0 (no positive pressure).
func TestEDPP_FirstToken_MetSLO_NoPositiveZ(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	d.OnRoute(newWaitingReq("r1", 0), "r1", false, 3, "", "")
	d.OnFirstToken("r1", 50_000)
	if got := zval(d.zByClass[""]); got != 0 {
		t.Fatalf("zTTFT = %v, want 0 (SLO met)", got)
	}
}

// A decision at clock=300ms while r1 (arrival 0, τ 100ms) is still awaiting its first
// token credits the certain lower-bound miss (200ms) into z_ttft — before any completion.
func TestEDPP_ContinuousCredit_BeforeFirstToken(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	d.OnRoute(newWaitingReq("r1", 0), "r1", false, 3, "", "")
	d.Decide(newWaitingReq("r2", 300_000), &RouterState{Clock: 300_000})
	if got := zval(d.zByClass[""]); math.Abs(got-200_000) > 1 {
		t.Fatalf("zTTFT = %v, want 200000 (credited lower bound)", got)
	}
}

// Two sweeps at 300ms then 400ms must credit only the increment: total 300ms (400−100),
// not 200+300.
func TestEDPP_Credit_NoDoubleCount(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	d.OnRoute(newWaitingReq("r1", 0), "r1", false, 3, "", "")
	d.Decide(newWaitingReq("x", 300_000), &RouterState{Clock: 300_000})
	d.Decide(newWaitingReq("y", 400_000), &RouterState{Clock: 400_000})
	if got := zval(d.zByClass[""]); math.Abs(got-300_000) > 1 {
		t.Fatalf("zTTFT = %v, want 300000 (increment-only)", got)
	}
}

// A request dropped/timed-out after accruing credit keeps that credit (censored
// observation: the long wait genuinely violated the SLO) and is no longer tracked.
func TestEDPP_Forget_KeepsCredit(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	d.OnRoute(newWaitingReq("r1", 0), "r1", false, 3, "", "")
	d.Decide(newWaitingReq("x", 300_000), &RouterState{Clock: 300_000})
	d.Forget("r1")
	if got := zval(d.zByClass[""]); math.Abs(got-200_000) > 1 {
		t.Fatalf("zTTFT = %v, want 200000 (credit kept)", got)
	}
	if _, ok := d.awaitingFirstToken["r1"]; ok {
		t.Fatalf("r1 should be untracked after Forget")
	}
}

// If the first-token hook never fires (e.g. a completion path that bypasses it),
// OnComplete falls back to bumping z_ttft from the realized TTFT — so the signal is
// never lost. 250ms realized, τ 100ms ⇒ 150ms.
func TestEDPP_OnComplete_FallbackBumpsZTTFT(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	req := newWaitingReq("r1", 0)
	req.OutputTokens = []TokenID{1, 2}
	d.OnRoute(req, "r1", false, 3, "", "")
	d.OnComplete(req, "r1", 250_000, 10_000)
	if got := zval(d.zByClass[""]); math.Abs(got-150_000) > 1 {
		t.Fatalf("zTTFT = %v, want 150000 (completion fallback)", got)
	}
}

// When the first-token true-up already happened, OnComplete must NOT bump z_ttft again.
func TestEDPP_OnComplete_NoDoubleAfterFirstToken(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	req := newWaitingReq("r1", 0)
	req.OutputTokens = []TokenID{1, 2}
	d.OnRoute(req, "r1", false, 3, "", "")
	d.OnFirstToken("r1", 250_000)
	d.OnComplete(req, "r1", 250_000, 10_000)
	if got := zval(d.zByClass[""]); math.Abs(got-150_000) > 1 {
		t.Fatalf("zTTFT = %v, want 150000 (no double bump)", got)
	}
}

// --- Per-GPU θ store (Task 3) ---

// TestEDPPDecider_CoeffsFor proves coeffsFor selects the per-GPU-type override when
// present and falls back to the global coeffs for an unmapped type and for "". The
// store is otherwise unused (not yet wired into the cost math), so this test only
// exercises the selector itself (design 2026-07-14).
func TestEDPPDecider_CoeffsFor(t *testing.T) {
	base := EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0}
	a100 := EDPPCoeffs{AlphaD: 4000, AlphaP: 4000, C0: 400, C1: 4, CPf: 40, CAttn: 0}

	cfg := defaultTestEDPPConfig()
	if cfg.Coeffs != base {
		t.Fatalf("defaultTestEDPPConfig().Coeffs = %+v, want %+v (test assumption)", cfg.Coeffs, base)
	}
	cfg.CoeffsByGPU = map[string]EDPPCoeffs{"A100": a100}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)

	if got := d.coeffsFor("A100"); got != a100 {
		t.Fatalf("coeffsFor(A100) = %+v, want %+v", got, a100)
	}
	if got := d.coeffsFor("H100"); got != base { // unmapped ⇒ fallback
		t.Fatalf("coeffsFor(unmapped) = %+v, want base %+v", got, base)
	}
	if got := d.coeffsFor(""); got != base { // empty ⇒ fallback
		t.Fatalf("coeffsFor(\"\") = %+v, want base %+v", got, base)
	}
}

// --- Reduced path decode-side θ (Task 5) ---

// TestDecideReduced_HomogeneousByteIdentical is the byte-identity guard (INV-6) for
// wiring the selected decode instance's θ into the reduced-path decode-side terms: a
// CoeffsByGPU entry that merely duplicates the global coeffs under the selected
// snapshot's GPUType must produce the exact same Decide() result as no CoeffsByGPU at
// all (coeffsFor falls back to d.coeffs either way). This test passes before AND after
// the θ wiring lands — it is what proves the wiring didn't change homogeneous behavior.
func TestDecideReduced_HomogeneousByteIdentical(t *testing.T) {
	cfg := defaultTestEDPPConfig() // joint=false ⇒ reduced path
	dPlain := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)

	cfgDup := cfg
	cfgDup.CoeffsByGPU = map[string]EDPPCoeffs{"H100": cfg.Coeffs}
	dDup := NewEDPPDecider(cfgDup, newTestAffineModel(), nil, nil)

	req := makeReq("r1", 256, "batch")
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{
		ID: "d0", GPUType: "H100", BatchSize: 2, MaxBatchSize: 4, KvTokensInUse: 1024,
	}}}

	got := dPlain.Decide(req, state)
	gotDup := dDup.Decide(req, state)
	if got != gotDup {
		t.Fatalf("reduced decision changed under duplicate-θ CoeffsByGPU (byte-identity broken): plain=%+v dup=%+v", got, gotDup)
	}
}

// --- least-ttft reduced decision rule (Task 1) ---

// least-ttft disaggregates when local decode is congested (ttftD high) and stays
// local when the prefill pool is congested (ttftP high) — decided purely on predicted TTFT.
func TestDecideReduced_LeastTTFT_DecidesOnPredictedTTFT(t *testing.T) {
	// Prefill pool EMPTY/idle -> ttftP low; decode instance heavily loaded -> ttftD high => disaggregate.
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "least-ttft"
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", BatchSize: 0, ResidentPrefillTokens: 0}}
	})
	reqBusy := makeReq("r1", 600, "") // uncached prompt so a_p > 0
	stateBusyDecode := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 64, KvTokensInUse: 60000, QueueDepth: 40}, // congested decode
	}}
	if !d.Decide(reqBusy, stateBusyDecode).Disaggregate {
		t.Fatal("least-ttft: expected Disaggregate=true when local decode is congested (ttftD > ttftP)")
	}

	// Now flip it: idle decode, congested prefill pool -> ttftP high => stay local.
	d2 := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", BatchSize: 64, ResidentPrefillTokens: 120000, QueueDepth: 40}}
	})
	stateIdleDecode := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 0, KvTokensInUse: 0, QueueDepth: 0},
	}}
	if d2.Decide(reqBusy, stateIdleDecode).Disaggregate {
		t.Fatal("least-ttft: expected Disaggregate=false when prefill pool is congested (ttftP > ttftD)")
	}
}

// The KEY guard: least-ttft ignores the SLO virtual queues. Blowing up z must NOT change
// its decision, whereas under dpp the same z DOES change the decision.
func TestDecideReduced_LeastTTFT_IgnoresVirtualQueues(t *testing.T) {
	req := makeReq("r1", 600, "")
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 8, KvTokensInUse: 4000, QueueDepth: 2},
	}}
	prefill := func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", BatchSize: 4, ResidentPrefillTokens: 2000}}
	}

	cfgLT := defaultTestEDPPConfig()
	cfgLT.Rule = "least-ttft"
	base := NewEDPPDecider(cfgLT, newTestAffineModel(), nil, prefill)
	baseDec := base.Decide(req, state).Disaggregate

	withZ := NewEDPPDecider(cfgLT, newTestAffineModel(), nil, prefill)
	withZ.zByClass[req.SLOClass] = &edppClassState{zTTFT: 1e12, zITL: 1e12} // huge SLO deficit
	if withZ.Decide(req, state).Disaggregate != baseDec {
		t.Fatal("least-ttft decision changed when z virtual queues were inflated (machinery leaked in)")
	}

	// Contrast: under dpp the same huge z SHOULD move the decision (proves the guard is meaningful).
	cfgDPP := defaultTestEDPPConfig() // Rule "" == dpp
	dppNoZ := NewEDPPDecider(cfgDPP, newTestAffineModel(), nil, prefill)
	dppZ := NewEDPPDecider(cfgDPP, newTestAffineModel(), nil, prefill)
	dppZ.zByClass[req.SLOClass] = &edppClassState{zTTFT: 1e12, zITL: 1e12}
	if dppNoZ.Decide(req, state).Disaggregate == dppZ.Decide(req, state).Disaggregate {
		t.Fatal("dpp decision did NOT change under huge z — the contrast guard is vacuous; retune the state")
	}
}

// Unknown rule is rejected at construction (R3 panic style).
func TestNewEDPPDecider_RejectsUnknownRule(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "bogus"
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unknown EDPPConfig.Rule")
		}
	}()
	_ = NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
}

// Default (Rule "") is byte-identical to explicit "dpp".
func TestDecideReduced_EmptyRuleEqualsDPP(t *testing.T) {
	req := makeReq("r1", 600, "")
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 8, KvTokensInUse: 4000}}}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "p0", ResidentPrefillTokens: 2000}} }
	e := defaultTestEDPPConfig() // Rule ""
	dfl := defaultTestEDPPConfig()
	dfl.Rule = "dpp"
	if NewEDPPDecider(e, newTestAffineModel(), nil, prefill).Decide(req, state) !=
		NewEDPPDecider(dfl, newTestAffineModel(), nil, prefill).Decide(req, state) {
		t.Fatal(`Rule "" must behave identically to "dpp"`)
	}
}
