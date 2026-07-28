package sim

import "testing"

// TestRunningPrefillState_TrueRemainingNotOracleGated is the INV-9 prefill-asymmetry
// regression test. Prefill's remaining = inLen − ProgressIndex is KNOWN at routing time
// (input length is known), so it is NOT oracle-gated: with admission detail ON but ORACLE
// OFF, a running prefill request's TrueRemaining must be populated (not −1). Decode's
// remaining depends on the hidden o_r and stays oracle-gated (verified separately).
func TestRunningPrefillState_TrueRemainingNotOracleGated(t *testing.T) {
	newSim := func() *Simulator {
		cfg := SimConfig{
			KVCacheConfig: NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
			BatchConfig:   NewBatchConfig(256, 2048, 0),
			Seed:          42,
		}
		s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), &spyLatencyModel{})
		if err != nil {
			t.Fatalf("NewSimulator: %v", err)
		}
		// A request still in prefill: ProgressIndex (10) < inLen (32) ⇒ 22 remaining.
		pf := &Request{ID: "pf", InputTokens: make([]TokenID, 32), OutputTokens: make([]TokenID, 5),
			ProgressIndex: 10, NumNewTokens: 1, State: StateRunning}
		s.RunningBatch = &Batch{Requests: []*Request{pf}}
		return s
	}

	// Admission detail ON, ORACLE OFF: prefill TrueRemaining must be the known remaining.
	s := newSim()
	s.SetAdmissionDetail(false)
	pf := s.RunningPrefillState()
	if len(pf) != 1 {
		t.Fatalf("RunningPrefillState: got %d states, want 1", len(pf))
	}
	if want := int64(22); pf[0].TrueRemaining != want {
		t.Fatalf("prefill TrueRemaining oracle-gated: got %d, want %d (inLen−ProgressIndex, known/deployable)", pf[0].TrueRemaining, want)
	}

	// Decode stays oracle-gated: with oracle OFF, decode TrueRemaining must be −1.
	dec := &Request{ID: "dec", InputTokens: make([]TokenID, 8), OutputTokens: make([]TokenID, 20),
		ProgressIndex: 12, NumNewTokens: 1, State: StateRunning}
	s.RunningBatch = &Batch{Requests: []*Request{dec}}
	decStates := s.RunningDecodeState()
	if len(decStates) != 1 {
		t.Fatalf("RunningDecodeState: got %d states, want 1", len(decStates))
	}
	if decStates[0].TrueRemaining != -1 {
		t.Fatalf("decode TrueRemaining must stay oracle-gated (−1 with oracle off), got %d", decStates[0].TrueRemaining)
	}
}
