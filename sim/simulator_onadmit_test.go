package sim

import "testing"

// TestSimulator_OnAdmit_FiresOnFirstAdmission verifies that the OnAdmit callback
// is fired exactly once when a request first enters the running batch.
func TestSimulator_OnAdmit_FiresOnFirstAdmission(t *testing.T) {
	sim := mustNewSimulator(t, newTestSimConfig())
	var admitted []string
	var admittedTick int64 = -1
	sim.OnAdmit = func(req *Request, tick int64) {
		admitted = append(admitted, req.ID)
		admittedTick = tick
	}

	req := &Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  make([]TokenID, 3),
		OutputTokens: make([]TokenID, 4),
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run()

	if len(admitted) == 0 {
		t.Fatalf("OnAdmit not fired for r1; got %v", admitted)
	}
	if admitted[0] != "r1" {
		t.Fatalf("OnAdmit fired with wrong ID; got %v, want r1", admitted)
	}

	count := 0
	for _, id := range admitted {
		if id == "r1" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("OnAdmit fired %d times for r1, want 1", count)
	}

	if admittedTick < 0 {
		t.Errorf("OnAdmit tick = %d, want >= 0", admittedTick)
	}
	// Request has ArrivalTime 0; admitted tick must be >= arrival time.
	if admittedTick < req.ArrivalTime {
		t.Errorf("OnAdmit tick = %d, want >= ArrivalTime %d", admittedTick, req.ArrivalTime)
	}
}

// TestSimulator_OnAdmit_NilIsNoop verifies that leaving OnAdmit nil does not panic.
func TestSimulator_OnAdmit_NilIsNoop(t *testing.T) {
	sim := mustNewSimulator(t, newTestSimConfig())
	// sim.OnAdmit is nil by default — do not set it.

	req := &Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  make([]TokenID, 3),
		OutputTokens: make([]TokenID, 4),
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run() // must not panic

	// Verify the request completed (conservation check).
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("expected 1 completed request, got %d", sim.Metrics.CompletedRequests)
	}
}
