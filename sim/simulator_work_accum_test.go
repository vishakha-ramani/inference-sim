package sim

import "testing"

// The accumulator must sum per-step δ (active latency-model basis) to the analytic
// closed form for a single request driven through prefill then decode.
func TestWorkAccumulator_SumsToClosedForm(t *testing.T) {
	c := EDPPCoeffs{C0: 5.0, C1: 0.05, CPf: 6.0, CAttn: 0.001}
	sim := &Simulator{workAcc: map[string]*reqWorkAccum{}, recordWorkTrace: true, workCoeffs: c}

	ar := 100
	// Single-chunk prefill: one step processes all ar tokens (ProgressIndex 0 → ar).
	sim.accumulateStepWork("r1", "batch", &Request{
		ID: "r1", SLOClass: "batch",
		InputTokens: make([]TokenID, ar), NumNewTokens: ar, ProgressIndex: 0,
	})
	// 3 decode steps at ProgressIndex ar, ar+1, ar+2.
	for k := 0; k < 3; k++ {
		sim.accumulateStepWork("r1", "batch", &Request{
			ID: "r1", SLOClass: "batch",
			InputTokens: make([]TokenID, ar), OutputTokens: make([]TokenID, 3),
			NumNewTokens: 1, ProgressIndex: int64(ar + k),
		})
	}
	got := sim.WorkAccumulators()["r1"]

	wantPrefill := 6.0*float64(ar) + 0.001*float64(ar)*(float64(ar)+float64(ar)/2.0) // single chunk
	if diff := got.RealizedPrefillWork - wantPrefill; diff < -1e-6 || diff > 1e-6 {
		t.Fatalf("prefill work = %v, want %v", got.RealizedPrefillWork, wantPrefill)
	}
	var wantDecode float64
	for k := 0; k < 3; k++ {
		wantDecode += 5.0 + 0.05*float64(ar+k)
	}
	if diff := got.RealizedDecodeWork - wantDecode; diff < -1e-6 || diff > 1e-6 {
		t.Fatalf("decode work = %v, want %v", got.RealizedDecodeWork, wantDecode)
	}
	if got.Ar != int64(ar) || got.ApRealized != int64(ar) || got.ORealized != 3 || got.PrefillChunks != 1 {
		t.Fatalf("bad accum meta: %+v", got)
	}
}

func TestWorkAccumulator_DisabledNoAlloc(t *testing.T) {
	sim := &Simulator{recordWorkTrace: false}
	sim.accumulateStepWork("r1", "batch", &Request{ID: "r1", InputTokens: make([]TokenID, 10), NumNewTokens: 10})
	if sim.workAcc != nil {
		t.Fatalf("workAcc must stay nil when disabled, got %v", sim.workAcc)
	}
}
