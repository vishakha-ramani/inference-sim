package sim

import (
	"fmt"
	"testing"
)

// Per-request decision-cost benchmarks for the routing rule (§V overhead claim).
//
// The rule runs on the CPU gateway, once per request, and never touches the GPU.
// These benchmarks time the full decision function a router would call —
// least-TTFT (reduced), drift-plus-penalty (joint), and drift-plus-VaR (joint,
// deployable) — as the decode batch width B (the VaR co-resident loop) and the
// fleet size grow. The analytical cost is O(|M|·|P|) latency-law evaluations plus
// O(|M|·|P|·B) cheap arithmetic; these numbers put an absolute time on it and
// show what the externality's co-resident loop actually costs over a work-only rule.
//
// Caveat: the benchmark uses the affine test latency model (newTestAffineModel).
// The candidate count and co-resident-loop iteration count are exact and model
// independent. The absolute nanoseconds are a lower bound, because the production
// trained-physics law costs more per evaluation than the affine stub while being
// called the same number of times.

// benchCoResidents builds b decode co-residents for one instance's RunningDecode
// slice, each mid-decode with a realized first token (so g()'s TTFT conjunct is
// fixed and the completion projection + kernel run for every one).
func benchCoResidents(b int) []RunningReqState {
	crs := make([]RunningReqState, b)
	for i := range crs {
		crs[i] = RunningReqState{
			StepsDone:     10,
			KVBlocks:      64,
			TrueRemaining: 200,
			SLOClass:      "batch",
			ArrivalUs:     0,
			FirstTokenUs:  1000,
			TTFTSet:       true,
		}
	}
	return crs
}

// benchFleet returns decode instance IDs, prefill instance IDs, and a RouterState
// whose nDecode decode snapshots each carry batchWidth co-residents.
func benchFleet(nDecode, nPrefill, batchWidth int) (decodeIDs, prefillIDs []string, state *RouterState, prefill func() []RoutingSnapshot) {
	decodeIDs = make([]string, nDecode)
	snaps := make([]RoutingSnapshot, nDecode)
	for i := 0; i < nDecode; i++ {
		id := fmt.Sprintf("d%d", i)
		decodeIDs[i] = id
		snaps[i] = RoutingSnapshot{
			ID:            id,
			BatchSize:     batchWidth,
			RunningDecode: benchCoResidents(batchWidth),
		}
	}
	prefillIDs = make([]string, nPrefill)
	psnaps := make([]RoutingSnapshot, nPrefill)
	for i := 0; i < nPrefill; i++ {
		id := fmt.Sprintf("p%d", i)
		prefillIDs[i] = id
		psnaps[i] = RoutingSnapshot{ID: id}
	}
	state = &RouterState{SelectedInstance: decodeIDs[0], Snapshots: snaps, Clock: 5_000_000}
	prefill = func() []RoutingSnapshot { return psnaps }
	return
}

// benchTrainedPhysicsModel builds the production trained-physics latency model
// (H100, TP4 — the paper's Llama-70B provenance) so the iteration-time law the
// decision path evaluates is the real one, not the affine stub. sim/latency's
// factory is registered through the blank import in latency_import_test.go.
func benchTrainedPhysicsModel(tb testing.TB) LatencyModel {
	tb.Helper()
	// Llama-70B-like dense architecture (the paper's provenance), with the
	// intermediate dim the trained-physics MLP term requires.
	mc := ModelConfig{
		NumLayers:       80,
		HiddenDim:       8192,
		NumHeads:        64,
		NumKVHeads:      8,
		IntermediateDim: 28672,
		BytesPerParam:   2, // FP16
	}
	hw := NewModelHardwareConfig(mc, rooflineHWCalib(), "llama-70b", "H100", 4, 1, false, "trained-physics", 0)
	// A valid trained-physics coefficient vector (β₁–β₁₀, α₁–α₃), same shape as the
	// frozen Llama-70B fit; the per-call arithmetic cost is what matters here, not the values.
	coeffs := LatencyCoeffs{
		AlphaCoeffs: []float64{15563.199579, 777.3455, 45.907545},
		BetaCoeffs:  []float64{0.152128, 0.0, 1.36252915, 0.752037, 32.09546717, 4.41684444, 126.024825, 481.8613888, 0.0, 1.94710771},
	}
	m, err := MustNewLatencyModel(coeffs, hw)
	if err != nil {
		tb.Fatalf("MustNewLatencyModel(trained-physics): %v", err)
	}
	return m
}

// benchDecider constructs a decider for the given rule over the given fleet, with a
// cold cache for every instance (a_p = full prompt, so the disaggregation branch is live).
func benchDecider(rule string, model LatencyModel, decodeIDs, prefillIDs []string, prefill func() []RoutingSnapshot) *EDPPDecider {
	cfg := defaultTestEDPPConfig()
	cfg.TauE2EUs = 30_000_000 // 30 s E2E deadline for the VaR composite
	allIDs := append(append([]string{}, decodeIDs...), prefillIDs...)
	switch rule {
	case "least-ttft":
		cfg.Rule = "least-ttft"
	case "dpp":
		cfg.Rule = "" // drift-plus-penalty (default)
		cfg.Joint = true
	case "var":
		cfg.Rule = "var"
		cfg.VarMetric = "util"
		cfg.VarKeepCongestion = true
		cfg.VarNormalize = true
		cfg.VarCongestionWeight = 1.0
		cfg.VarDeployable = true // the shipped rule reads the censored estimate, not the oracle
		cfg.Joint = true
	}
	return NewEDPPDecider(cfg, model, coldCacheQuery(allIDs...), prefill)
}

// benchModels are the two latency backends the Decide benchmarks run under. The
// candidate count and co-resident-loop iterations are identical across both; only
// the per-iteration-time evaluation cost differs, so the ratio is the constant the
// affine stub understates.
func benchModels(tb testing.TB) map[string]LatencyModel {
	return map[string]LatencyModel{
		"affine":  newTestAffineModel(),
		"trained": benchTrainedPhysicsModel(tb),
	}
}

// BenchmarkDecide_ByBatchWidth times one decision on the paper's 1P2D fleet as the
// decode batch width B grows. Only drift-plus-VaR loops over co-residents, so its
// curve should rise with B while least-TTFT and drift-plus-penalty stay flat.
func BenchmarkDecide_ByBatchWidth(b *testing.B) {
	models := benchModels(b)
	for _, mn := range []string{"affine", "trained"} {
		for _, rule := range []string{"least-ttft", "dpp", "var"} {
			for _, bw := range []int{1, 8, 16, 32, 64} {
				b.Run(fmt.Sprintf("%s/%s/B=%d", mn, rule, bw), func(b *testing.B) {
					decodeIDs, prefillIDs, state, prefill := benchFleet(2, 1, bw)
					d := benchDecider(rule, models[mn], decodeIDs, prefillIDs, prefill)
					req := &Request{ID: "r", InputTokens: make([]TokenID, 2000), SLOClass: "batch"}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = d.Decide(req, state)
					}
				})
			}
		}
	}
}

// BenchmarkDecide_ByFleet times one decision at a fixed batch width as the fleet
// grows, exercising the O(|M|·|P|) candidate scaling.
func BenchmarkDecide_ByFleet(b *testing.B) {
	model := benchTrainedPhysicsModel(b)
	fleets := []struct{ nD, nP int }{{2, 1}, {4, 2}, {8, 4}}
	for _, rule := range []string{"least-ttft", "dpp", "var"} {
		for _, f := range fleets {
			b.Run(fmt.Sprintf("%s/%dP%dD", rule, f.nP, f.nD), func(b *testing.B) {
				decodeIDs, prefillIDs, state, prefill := benchFleet(f.nD, f.nP, 16)
				d := benchDecider(rule, model, decodeIDs, prefillIDs, prefill)
				req := &Request{ID: "r", InputTokens: make([]TokenID, 2000), SLOClass: "batch"}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = d.Decide(req, state)
				}
			})
		}
	}
}

// BenchmarkVarDecodeLocal isolates the VaR co-resident inner loop (completion
// projection + kernel) as a function of B, confirming it is cheap per co-resident.
func BenchmarkVarDecodeLocal(b *testing.B) {
	rt := varReTiming{tIter0: 40_000, tIterOverlap: 60_000, tIterAfter: 80_000}
	for _, bw := range []int{1, 8, 16, 32, 64} {
		b.Run(fmt.Sprintf("B=%d", bw), func(b *testing.B) {
			crs := make([]varDecodeCoResident, bw)
			for i := range crs {
				crs[i] = varDecodeCoResident{
					rem: 200, arrivalUs: 0, firstTokenUs: 1000, ttftSet: true,
					slo: varSLO{tauTTFTUs: 100_000, tauITLUs: 50_000, tauE2EUs: 30_000_000},
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = varDecodeLocal(0, crs, rt, 4, varKernelUtil)
			}
		})
	}
}
