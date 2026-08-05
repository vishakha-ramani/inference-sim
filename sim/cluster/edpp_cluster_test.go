package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type firstTokenFeedbackSpy struct {
	keys          []string
	ticks         []int64
	completedKeys []string
	completedTTFT []int64
	forgottenKeys []string
}

func (*firstTokenFeedbackSpy) OnRoute(*sim.Request, string, bool, int, string, string) {}
func (*firstTokenFeedbackSpy) OnAdmit(string, bool)                                    {}
func (s *firstTokenFeedbackSpy) OnFirstToken(key string, tick int64) {
	s.keys = append(s.keys, key)
	s.ticks = append(s.ticks, tick)
}
func (s *firstTokenFeedbackSpy) OnComplete(_ *sim.Request, key string, ttft, _ int64) {
	s.completedKeys = append(s.completedKeys, key)
	s.completedTTFT = append(s.completedTTFT, ttft)
}
func (s *firstTokenFeedbackSpy) Forget(key string) {
	s.forgottenKeys = append(s.forgottenKeys, key)
}

// Disaggregated feedback must true up at the client-visible first decode
// token, not at the prefill subrequest's internal completion boundary.
// Normal collocated requests continue to true up from Simulator.OnFirstToken.
func TestEDPPFeedback_UsesFirstDecodeTokenForDisaggregatedRequest(t *testing.T) {
	spy := &firstTokenFeedbackSpy{}
	parent := &ParentRequest{
		ID: "parent", PrefillSubReqID: "parent_prefill", DecodeSubReqID: "parent_decode",
	}
	cs := &ClusterSimulator{
		sloFeedback:               spy,
		parentRequests:            map[string]*ParentRequest{"parent": parent},
		pendingPrefillCompletions: map[string]string{"parent_prefill": "parent"},
		pendingDecodeCompletions:  map[string]string{"parent_decode": "parent"},
	}

	cs.feedFirstToken(&sim.Request{ID: "parent_prefill"}, 100)
	if len(spy.keys) != 0 {
		t.Fatalf("prefill completion emitted client TTFT feedback: keys=%v ticks=%v", spy.keys, spy.ticks)
	}

	cs.recordFirstDecodeToken(&sim.Request{ID: "parent_decode", IsDecodeSubRequest: true}, 250)
	if len(spy.keys) != 1 || spy.keys[0] != "parent" || spy.ticks[0] != 250 {
		t.Fatalf("first decode token feedback = keys=%v ticks=%v, want parent@250", spy.keys, spy.ticks)
	}
	if parent.FirstDecodeTokenTime != 250 {
		t.Fatalf("FirstDecodeTokenTime = %d, want 250", parent.FirstDecodeTokenTime)
	}

	cs.feedFirstToken(&sim.Request{ID: "local"}, 300)
	if len(spy.keys) != 2 || spy.keys[1] != "local" || spy.ticks[1] != 300 {
		t.Fatalf("local first-token feedback = keys=%v ticks=%v, want second event local@300", spy.keys, spy.ticks)
	}
}

// FirstTokenTime is an elapsed duration, whereas ArrivalTime is absolute. A
// second subtraction used to turn perfectly valid late-arriving requests
// negative, route them through Forget, and silently suppress their ITL/N-hat
// feedback.
func TestEDPPFeedback_UsesElapsedFirstTokenTime(t *testing.T) {
	spy := &firstTokenFeedbackSpy{}
	cs := &ClusterSimulator{sloFeedback: spy}
	req := &sim.Request{
		ID: "late-arrival", ArrivalTime: 10_000, FirstTokenTime: 2_000,
		TTFTSet: true, ITL: []int64{500},
	}

	cs.feedSLOFeedback(req)

	if len(spy.completedKeys) != 1 || spy.completedKeys[0] != req.ID {
		t.Fatalf("completed keys = %v, want [%s]", spy.completedKeys, req.ID)
	}
	if len(spy.completedTTFT) != 1 || spy.completedTTFT[0] != 2_000 {
		t.Fatalf("completed TTFT = %v, want [2000]", spy.completedTTFT)
	}
	if len(spy.forgottenKeys) != 0 {
		t.Fatalf("valid completion was forgotten: %v", spy.forgottenKeys)
	}
}

// newTestEDPPDeploymentConfig mirrors newTestDisaggDeploymentConfig but selects the
// EDPP decider and supplies its knobs. Tight τ_itl makes the run breach the ITL SLO
// so the z-feedback path is exercised.
func newTestEDPPDeploymentConfig(numInstances, prefill, decode int) DeploymentConfig {
	modelCfg := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
	}
	hwCfg := sim.HardwareCalib{TFlopsPeak: 1.0, BwPeakTBs: 0.001}
	betas := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0}
	alphas := []float64{100, 1, 100}
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betas, alphas),
			ModelHardwareConfig: sim.NewModelHardwareConfig(modelCfg, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 0),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "edpp",
		EDPPTauTTFTUs:           100_000,
		EDPPTauRefUs:            100_000,
		EDPPTauITLUs:            5_000,
		EDPPV:                   0.1,
		EDPPCXferUs:             1_000,
		EDPPNomPrefillTokens:    512,
		EDPPNomDecodeCtx:        2048,
		EDPPCoeffs:              sim.EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0},
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
	}
}

func TestEDPP_Cluster_WiringAndFeedback(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(5)), nil)

	// The "edpp" name must resolve to an EDPPDecider...
	if _, ok := cs.disaggregationDecider.(*sim.EDPPDecider); !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}
	// ...and it must be registered as the SLO-feedback sink so completions update Z.
	if cs.sloFeedback == nil {
		t.Fatalf("sloFeedback is nil; EDPP completion feedback not wired")
	}

	// End-to-end run must complete without error and produce decode output (full path).
	mustRun(t, cs)
	if cs.AggregatedMetrics().TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0; EDPP run produced no decode output")
	}
}

// Both VaR rules require per-running-request snapshot state. The simplified
// var-prefill mode must enable the same deployable admission-detail substrate
// as the full var rule; otherwise every RunningDecode/RunningPrefill slice is
// nil and all VaR kernels silently evaluate to zero.
func TestEDPP_VarPrefill_EnablesDeployableVaRInputs(t *testing.T) {
	config := newTestEDPPDeploymentConfig(3, 1, 2)
	config.EDPPRule = "var-prefill"
	config.EDPPVarMetric = "util"
	config.EDPPVarDeployable = true
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)

	for _, inst := range cs.instances {
		if !inst.sim.AdmissionDetailEnabled() {
			t.Fatalf("instance %s has admission detail disabled; var-prefill would observe zero co-residents", inst.ID())
		}
	}
}

// Every dedicated resident-externality policy needs the same admission-detail
// substrate as the older var rules. These modes are deployable by construction:
// admission detail supplies phase/age/progress state while true remaining output
// work stays censored.
func TestEDPP_ResidentExternalityPolicies_EnableAdmissionDetail(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*DeploymentConfig)
	}{
		{"joint-causal", func(config *DeploymentConfig) { config.EDPPJointCausalVar = true }},
		{"decomposed-causal", func(config *DeploymentConfig) { config.EDPPDecomposedCausalVar = true }},
		{"joint-slo-externality", func(config *DeploymentConfig) { config.EDPPJointSLOExternality = true }},
		{"decomposed-slo-externality", func(config *DeploymentConfig) { config.EDPPDecomposedSLOExternality = true }},
		{"kairos-paper", func(config *DeploymentConfig) { config.EDPPRule = "kairos-paper" }},
		{"kairos-adapted", func(config *DeploymentConfig) { config.EDPPRule = "kairos-adapted" }},
		{"kairos-compatibility-alias", func(config *DeploymentConfig) { config.EDPPRule = "kairos" }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := newTestEDPPDeploymentConfig(3, 1, 2)
			test.configure(&config)
			cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)
			for _, inst := range cs.instances {
				if !inst.sim.AdmissionDetailEnabled() {
					t.Fatalf("instance %s has admission detail disabled; resident externality would be identically zero", inst.ID())
				}
			}
		})
	}
}

func TestEDPP_NewTTFTPoliciesExposeSchedulerRolloutStateOnEveryCandidate(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*DeploymentConfig)
	}{
		{"causal-externality", func(config *DeploymentConfig) {
			config.EDPPJointSLOExternality = true
			config.EDPPSLOExternalityNoCapacity = true
		}},
		{"joint-least-ttft", func(config *DeploymentConfig) {
			config.EDPPRule = "least-ttft"
			config.EDPPJoint = true
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := newTestEDPPDeploymentConfig(3, 1, 2)
			test.configure(&config)
			cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)
			for _, inst := range cs.instances {
				id := inst.ID()
				snap := cs.snapshotProvider.Snapshot(id, cs.clock)
				if !snap.SchedulerStateObserved {
					t.Fatalf("candidate %s has no scheduler rollout state", id)
				}
				if snap.MaxScheduledTokens <= 0 || snap.MaxBatchSize <= 0 || snap.BlockSizeTokens <= 0 {
					t.Fatalf(
						"candidate %s has incomplete rollout limits: tokens=%d batch=%v block=%d",
						id, snap.MaxScheduledTokens, snap.MaxBatchSize, snap.BlockSizeTokens,
					)
				}
			}
		})
	}
}

// TestEDPP_Cluster_ConservationAfterDisaggregatedCompletion drives requests through
// the real EDPP routing path to completion — including at least one request that is
// actually PD-disaggregated — and asserts the decider's conservation bookkeeping
// returns to baseline: qpWork ≈ qdWork ≈ 0 and the pending map is empty.
//
// This is the test whose absence let Defect 1 ship: before the fix, OnComplete fired
// for the decode sub-request (parent.ID+"_decode") and never matched OnRoute's
// parent-ID pending key, so qp/qd/pending leaked for every disaggregated request and
// this assertion would fail with a positive residual backlog.
func TestEDPP_Cluster_ConservationAfterDisaggregatedCompletion(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	// Force genuine disaggregation through the real routing path. The E14 rule routes P
	// (disaggregate) when offloading prefill relieves ITL-SLO pressure. We set that up
	// deterministically:
	//   - V=0, CXfer=0: no transfer penalty, so the rule is governed purely by the
	//     SLO-pressure term.
	//   - A latency model whose prefill co-scheduling inflates a decode iteration's ITL
	//     (δ_pf-chunk > 0): offloading prefill to the P pool removes that inflation, so
	//     ITL_P < ITL_D and the ITL term favors P.
	//   - Pre-seeded ITL-SLO breaches (z_itl ≫ 0): the standing virtual-queue pressure
	//     turns that ITL advantage into a decisive P decision from the first request.
	// With this setup every request actually disaggregates (asserted via ParentRequests),
	// exercising the decode-sub-request completion path that Defect 1 broke.
	config.EDPPV = 0
	config.EDPPCXferUs = 0
	config.LatencyCoeffs = sim.NewLatencyCoeffs(
		[]float64{1000, 0, 0, 0, 0, 0, 0}, // β₁>0 ⇒ nonzero prefill-chunk ITL inflation
		[]float64{100, 1, 100},
	)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(10)), nil)

	dec, ok := cs.disaggregationDecider.(*sim.EDPPDecider)
	if !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}
	// Seed standing ITL-SLO pressure (z_itl) so the rule offloads prefill to protect
	// decode ITL. Each call reports a realized mean ITL of 500ms ≫ τ_itl=5ms.
	for i := 0; i < 200; i++ {
		dec.OnComplete(&sim.Request{ID: "seed", SLOClass: ""}, "seed", 0, 500_000)
	}

	mustRun(t, cs)

	// At least one request must have actually been disaggregated (P-routed): every
	// disaggregated request becomes a ParentRequest. Without a disaggregated request
	// this test would not exercise the decode-sub-request completion path that Defect 1
	// broke, so it is part of the contract.
	if len(cs.ParentRequests()) == 0 {
		t.Fatalf("no requests were disaggregated; conservation test does not exercise the PD completion path")
	}

	// Observable conservation: all routed work has been released. A leak (Defect 1)
	// leaves a strictly positive residual here.
	qp, qd, pendingLen := dec.BacklogForTest()
	const eps = 1e-6
	if qp > eps || qd > eps {
		t.Errorf("conservation violated: residual backlog qp=%v qd=%v (want ~0); pending=%d", qp, qd, pendingLen)
	}
	if pendingLen != 0 {
		t.Errorf("conservation violated: %d entries still pending after all completions (want 0)", pendingLen)
	}
}

// TestEDPP_Cluster_WaitingBacklogDrainsAtAdmission verifies that after a full run
// the waiting backlog is zero (drained at admission, not at completion). It reuses
// the same forced-disaggregation setup as ConservationAfterDisaggregatedCompletion to
// exercise the P-routed (prefill+decode) admission path, then asserts qp≈0, qd≈0,
// pending==0 via BacklogForTest(). Conservation now holds via ADMISSION drain, not
// via OnComplete.
func TestEDPP_Cluster_WaitingBacklogDrainsAtAdmission(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	config.EDPPV = 0
	config.EDPPCXferUs = 0
	config.LatencyCoeffs = sim.NewLatencyCoeffs(
		[]float64{1000, 0, 0, 0, 0, 0, 0},
		[]float64{100, 1, 100},
	)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(10)), nil)

	dec, ok := cs.disaggregationDecider.(*sim.EDPPDecider)
	if !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}
	// Seed standing ITL-SLO pressure so requests actually disaggregate (P-routed).
	for i := 0; i < 200; i++ {
		dec.OnComplete(&sim.Request{ID: "seed", SLOClass: ""}, "seed", 0, 500_000)
	}

	mustRun(t, cs)

	if len(cs.ParentRequests()) == 0 {
		t.Fatalf("no requests were disaggregated; test does not exercise the P-routed admission path")
	}

	// Conservation via admission drain: all waiting work must be drained by now.
	qp, qd, pendingLen := dec.BacklogForTest()
	const eps = 1e-6
	if qp > eps || qd > eps {
		t.Errorf("waiting backlog not drained at admission: qp=%v qd=%v pending=%d (want ~0)", qp, qd, pendingLen)
	}
	if pendingLen != 0 {
		t.Errorf("pending entries remain after all admissions: %d (want 0)", pendingLen)
	}
}

// TestEDPP_Cluster_ConservationViaAdmission_NormalCompletion closes the coverage gap
// that the two forced-disaggregation conservation tests above DON'T cover. In the
// forced-disaggregation harness every decode sub-request reaches feedSLOFeedback with
// no usable latency signal (!TTFTSet || len(ITL)==0) and routes to Forget(key), which
// zeroes the backlog regardless of whether OnAdmit is wired. So those tests PASS even
// if the OnAdmit drain is removed entirely.
//
// This test instead drives requests that EDPP decides NOT to disaggregate
// (Disaggregate=false, D-routed) using the DEFAULT config (V=0.1, no pre-seeded z ⇒
// LHS<RHS ⇒ no disaggregation). Such requests run to NORMAL completion on the
// decode/shared pool with TTFT set and ≥1 ITL sample, so feedSLOFeedback calls
// OnComplete (NOT Forget). For these requests:
//   - OnRoute adds their work to qdWork,
//   - OnAdmit (entry into the running batch) is the ONLY thing that drains qdWork,
//   - OnComplete does NOT drain (it only bumps z + N̂_out).
//
// Therefore post-run qdWork≈0 holds ONLY if OnAdmit is wired. With the OnAdmit wiring
// removed, Forget never fires for a normally-completed request, so qdWork LEAKS and
// pending stays non-empty — this test fails, which is the whole point (revert-verified).
func TestEDPP_Cluster_ConservationViaAdmission_NormalCompletion(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	// Default V/c_xfer and NO pre-seeded z ⇒ the E14 rule keeps Disaggregate=false, so
	// every request is D-routed and completes normally (TTFT + multiple ITL samples
	// from newTestRequests' multi-token outputs ⇒ OnComplete, not Forget).
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(10)), nil)

	dec, ok := cs.disaggregationDecider.(*sim.EDPPDecider)
	if !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}

	mustRun(t, cs)

	// Requests must complete via the OnComplete path, NOT Forget. The observable proxy:
	// requests completed normally (CompletedRequests>0) and NOTHING disaggregated
	// (ParentRequests()==0). A disaggregated request would route its decode
	// sub-request through Forget; a zero-completion run would exercise nothing.
	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Fatalf("no requests completed; test does not exercise the OnComplete path")
	}
	if len(cs.ParentRequests()) != 0 {
		t.Fatalf("requests were disaggregated (%d parents); their decode sub-requests "+
			"route through Forget, not OnComplete — test must use the non-disaggregated path",
			len(cs.ParentRequests()))
	}

	// Conservation depends entirely on OnAdmit for these normally-completing requests:
	// OnComplete does not drain, and Forget never fires. If OnAdmit is wired, qd≈0 and
	// pending is empty; if the OnAdmit wiring is broken, qd leaks and pending>0.
	qp, qd, pendingLen := dec.BacklogForTest()
	const eps = 1e-6
	if qp > eps || qd > eps {
		t.Errorf("OnAdmit drain not applied for normally-completing requests: "+
			"residual backlog qp=%v qd=%v (want ~0); pending=%d", qp, qd, pendingLen)
	}
	if pendingLen != 0 {
		t.Errorf("OnAdmit drain not applied: %d entries still pending after all "+
			"admissions+completions (want 0)", pendingLen)
	}
}

// TestEDPP_Cluster_ConservationOnNonCompletionTerminal asserts the Defect-2 cleanup:
// a routed request that reaches a terminal state WITHOUT a usable completion signal
// still releases its conservation backlog. Single-token outputs produce a TTFT but no
// inter-token latency (len(req.ITL)==0), so feedSLOFeedback's signal guard would skip
// OnComplete entirely; the Forget path must release their backlog so Q returns to 0.
func TestEDPP_Cluster_ConservationOnNonCompletionTerminal(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	// Single-token outputs (oMean=1): completions carry TTFT but no ITL sample, so they
	// hit the no-usable-signal branch of feedSLOFeedback and must be Forgotten, not
	// silently dropped (the pre-fix early-return leaked their backlog).
	reqs := testGenerateRequests(11, math.MaxInt64, 50.0/1e6, 12,
		0, 300, 40, 100, 500, 1, 0, 1, 1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	dec := cs.disaggregationDecider.(*sim.EDPPDecider)

	mustRun(t, cs)

	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Fatalf("no requests completed; test setup did not route any work")
	}
	qp, qd, pendingLen := dec.BacklogForTest()
	const eps = 1e-6
	if qp > eps || qd > eps || pendingLen != 0 {
		t.Errorf("non-completion-terminal leak: qp=%v qd=%v pending=%d (want 0,0,0)", qp, qd, pendingLen)
	}
}

func TestEDPP_OccupancyCapacity_WiresSchedulerReferenceWidth(t *testing.T) {
	config := newTestEDPPDeploymentConfig(3, 1, 2)
	config.EDPPJointSLOExternality = true
	config.EDPPSLOExternalityOccupancyCapacity = true
	config.BatchConfig = sim.NewBatchConfig(8, 2048, 0)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(2)), nil)
	mustRun(t, cs)

	d, ok := cs.disaggregationDecider.(*sim.EDPPDecider)
	if !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}
	queues := d.SLOCapacityForTest()
	if len(queues) != 3 {
		t.Fatalf("occupancy queues = %d, want one per dedicated instance", len(queues))
	}
	for id, state := range queues {
		if state.Mu != 1 || state.Scale != 1_000_000 {
			t.Fatalf("occupancy queue %s has mu=%g scale=%g, want 1 and 1e6", id, state.Mu, state.Scale)
		}
	}
}
