package cluster

import (
	"os"
	"testing"
)

// TestFixedPlan_ForcedRouting_EndToEnd builds a 2P + 2D topology and forces a
// fixed plan via --pd-plan (LoadFixedPlanCSV + NewFixedPlanDecider at the cluster
// construction site). It proves BOTH overrides take effect:
//   - r1 → (decode=instance_2, prefill=local): handled non-disaggregated, no parent.
//   - r2 → (decode=instance_3, prefill=instance_1): disaggregated with the prefill
//     hint forcing the SECOND prefill instance (not the round-robin default), and
//     the decode override forcing instance_3.
func TestFixedPlan_ForcedRouting_EndToEnd(t *testing.T) {
	// 2 prefill (instance_0, instance_1) + 2 decode (instance_2, instance_3).
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	dir := t.TempDir()
	planPath := dir + "/plan.csv"
	csv := "request_id,decode_instance,prefill_instance\n" +
		"r1,instance_2,local\n" +
		"r2,instance_3,instance_1\n"
	if err := os.WriteFile(planPath, []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}
	config.PDPlanPath = planPath

	reqs := newTestRequests(2)
	reqs[0].ID = "r1"
	reqs[1].ID = "r2"

	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	// r1 was forced local ⇒ no ParentRequest (disaggregation did not fire).
	if _, ok := cs.parentRequests["r1"]; ok {
		t.Errorf("r1 forced local but a ParentRequest was created (disaggregation fired)")
	}
	// r1 must have decoded on instance_2 (DecodePodOverride on the local path).
	if !instanceCompleted(cs, "instance_2", "r1") {
		t.Errorf("r1 not completed on instance_2 (decode override on local path failed)")
	}

	// r2 was forced disaggregated with prefill hint instance_1 + decode instance_3.
	parent, ok := cs.parentRequests["r2"]
	if !ok {
		t.Fatalf("r2 forced disaggregated but no ParentRequest created")
	}
	if string(parent.PrefillInstanceID) != "instance_1" {
		t.Errorf("r2 PrefillInstanceID = %q, want instance_1 (PrefillPodHint not honored)", parent.PrefillInstanceID)
	}
	if string(parent.DecodeInstanceID) != "instance_3" {
		t.Errorf("r2 DecodeInstanceID = %q, want instance_3 (DecodePodOverride not honored)", parent.DecodeInstanceID)
	}
}

// PD outcome tracing is enabled by the CLI after cluster construction. Fixed-plan
// replays do not install an EDPP SLO-feedback decider, so this test guards the
// independent admission hook needed to capture both local and disaggregated
// schedule instants.
func TestFixedPlan_PDOutcomeTraceCapturesAdmissionTimes(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	dir := t.TempDir()
	planPath := dir + "/plan.csv"
	csv := "request_id,decode_instance,prefill_instance\n" +
		"r1,instance_2,local\n" +
		"r2,instance_3,instance_1\n"
	if err := os.WriteFile(planPath, []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}
	config.PDPlanPath = planPath

	reqs := newTestRequests(2)
	reqs[0].ID = "r1"
	reqs[1].ID = "r2"

	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	cs.SetRecordPDOutcomes(true)
	mustRun(t, cs)

	records := cs.BuildPDOutcomeRecords(cs.AggregatedMetrics())
	if len(records) != 2 {
		t.Fatalf("outcome records = %d, want 2", len(records))
	}
	for _, record := range records {
		switch record.RequestID {
		case "r1":
			if record.Disaggregated {
				t.Fatalf("r1 disaggregated = true, want false")
			}
			// r1 arrives at simulation tick zero, which is also the trace's
			// "unset" enqueue sentinel; the nonzero schedule is the regression
			// signal that the post-construction hook fired.
			if record.LocalSchedule == 0 {
				t.Fatalf("r1 missing local admission timing: %+v", record)
			}
		case "r2":
			if !record.Disaggregated {
				t.Fatalf("r2 disaggregated = false, want true")
			}
			// Decode may admit immediately (t_adm=0); both nonzero schedule
			// instants prove that each sub-request reached the hook.
			if record.PrefillSchedule == 0 || record.DecodeSchedule == 0 ||
				record.PrefillTAdm <= 0 {
				t.Fatalf("r2 missing disaggregated admission timing: %+v", record)
			}
		default:
			t.Fatalf("unexpected request %q", record.RequestID)
		}
	}
}

func instanceCompleted(cs *ClusterSimulator, instanceID, reqID string) bool {
	for _, inst := range cs.instances {
		if string(inst.ID()) != instanceID {
			continue
		}
		_, done := inst.Metrics().RequestCompletionTimes[reqID]
		return done
	}
	return false
}
