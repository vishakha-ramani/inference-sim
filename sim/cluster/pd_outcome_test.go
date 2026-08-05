package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// recordAdmissionTime must set the correct schedule field on the correct parent
// for disaggregated sub-requests, and record local requests in the local-admit map,
// keeping the FIRST admission time (idempotent under preemption re-admit).
func TestRecordAdmissionTime_RoutesToCorrectSlot(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:            map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:           map[string]int64{},
		recordPDOutcomes:          true,
	}
	parent := &ParentRequest{ID: "r1", PrefillSubReqID: "r1_prefill", DecodeSubReqID: "r1_decode"}
	cs.parentRequests["r1"] = parent
	cs.pendingPrefillCompletions["r1_prefill"] = "r1"
	cs.pendingDecodeCompletions["r1_decode"] = "r1"

	// Prefill sub-request admitted at t=100.
	cs.recordAdmissionTime(&sim.Request{ID: "r1_prefill"}, 100)
	if parent.PrefillScheduleTime != 100 {
		t.Fatalf("PrefillScheduleTime = %d, want 100", parent.PrefillScheduleTime)
	}
	// Decode sub-request admitted at t=250.
	cs.recordAdmissionTime(&sim.Request{ID: "r1_decode", IsDecodeSubRequest: true}, 250)
	if parent.DecodeScheduleTime != 250 {
		t.Fatalf("DecodeScheduleTime = %d, want 250", parent.DecodeScheduleTime)
	}
	// Local (non-disagg) request admitted at t=40, re-admitted at t=90 — keep 40.
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 40)
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 90)
	if got := cs.localAdmitTimes["r2"]; got != 40 {
		t.Fatalf("localAdmitTimes[r2] = %d, want 40 (first admission)", got)
	}
}

// When the flag is off, no local-admit bookkeeping happens (zero-cost gate).
func TestRecordAdmissionTime_DisabledIsNoop(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:            map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:           map[string]int64{},
		recordPDOutcomes:          false,
	}
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 40)
	if len(cs.localAdmitTimes) != 0 {
		t.Fatalf("localAdmitTimes populated while disabled: %v", cs.localAdmitTimes)
	}
}

func TestBuildPDOutcomeRecords_DisaggAndLocal(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:   map[string]*ParentRequest{},
		localAdmitTimes:  map[string]int64{},
		recordPDOutcomes: true,
	}
	// Disaggregated request r1 (disagg derived in builder from distinct decode instance).
	cs.parentRequests["r1"] = &ParentRequest{
		ID: "r1", OriginalRequest: &sim.Request{ID: "r1", InputTokens: make([]sim.TokenID, 512), SLOClass: "standard"},
		PrefillInstanceID: "instance_0", DecodeInstanceID: "instance_2",
		PrefillEnqueueTime: 100, PrefillScheduleTime: 140,
		DecodeEnqueueTime: 900, DecodeScheduleTime: 1200,
		CompletionTime: 43000,
	}
	// Local request r2 (no parent).
	cs.localAdmitTimes["r2"] = 40

	m := sim.NewMetrics()
	m.RequestTTFTs["r1"] = 1500
	m.RequestITLs["r1"] = 30
	m.RequestE2Es["r1"] = 42000
	m.RequestTTFTs["r2"] = 200
	m.RequestITLs["r2"] = 25
	m.RequestE2Es["r2"] = 8000

	recs := cs.BuildPDOutcomeRecords(m)
	if len(recs) != 2 {
		t.Fatalf("want 2 records, got %d", len(recs))
	}
	// Sorted by request_id: r1 then r2.
	if recs[0].RequestID != "r1" || recs[1].RequestID != "r2" {
		t.Fatalf("records not sorted by request_id: %s, %s", recs[0].RequestID, recs[1].RequestID)
	}
	r1 := recs[0]
	if !r1.Disaggregated || r1.PrefillTAdm != 40 || r1.DecodeTAdm != 300 {
		t.Fatalf("r1 disagg fields wrong: disagg=%v prefillTAdm=%d decodeTAdm=%d", r1.Disaggregated, r1.PrefillTAdm, r1.DecodeTAdm)
	}
	if r1.InputTokens != 512 || r1.RealizedE2E != 42000 || !r1.Completed {
		t.Fatalf("r1 metrics wrong: in=%d e2e=%v done=%v", r1.InputTokens, r1.RealizedE2E, r1.Completed)
	}
	r2 := recs[1]
	if r2.Disaggregated || r2.LocalTAdm != 0 || r2.LocalSchedule != 40 {
		t.Fatalf("r2 local fields wrong: disagg=%v localTAdm=%d localSchedule=%d", r2.Disaggregated, r2.LocalTAdm, r2.LocalSchedule)
	}
}

func TestBuildPDOutcomeRecords_CausalityAndNonNegativeTAdm(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:   map[string]*ParentRequest{},
		localAdmitTimes:  map[string]int64{},
		recordPDOutcomes: true,
	}
	cs.parentRequests["r1"] = &ParentRequest{
		ID: "r1", OriginalRequest: &sim.Request{ID: "r1"},
		PrefillInstanceID: "p0", DecodeInstanceID: "d0",
		PrefillEnqueueTime: 100, PrefillScheduleTime: 140,
		DecodeEnqueueTime: 900, DecodeScheduleTime: 1200,
	}
	m := sim.NewMetrics()
	m.RequestE2Es["r1"] = 42000
	for _, r := range cs.BuildPDOutcomeRecords(m) {
		if r.PrefillTAdm < 0 || r.DecodeTAdm < 0 || r.LocalTAdm < 0 {
			t.Fatalf("negative t_adm in %+v", r)
		}
		if r.PrefillSchedule != 0 && r.PrefillSchedule < r.PrefillEnqueue {
			t.Fatalf("schedule before enqueue (prefill) in %+v", r)
		}
		if r.DecodeSchedule != 0 && r.DecodeSchedule < r.DecodeEnqueue {
			t.Fatalf("schedule before enqueue (decode) in %+v", r)
		}
	}
}
