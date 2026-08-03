package sim

import (
	"os"
	"testing"
)

func TestFixedPlanDecider_Local(t *testing.T) {
	d := NewFixedPlanDecider(map[string]FixedPlanAction{
		"r1": {DecodeInstance: "M0", PrefillInstance: "local"},
	})
	got := d.Decide(&Request{ID: "r1"}, nil)
	if got.Disaggregate || got.DecodePodOverride != "M0" || got.PrefillPodHint != "" {
		t.Fatalf("local action: got %+v, want {Disaggregate:false, DecodePodOverride:M0, PrefillPodHint:''}", got)
	}
}

func TestFixedPlanDecider_Disagg(t *testing.T) {
	d := NewFixedPlanDecider(map[string]FixedPlanAction{
		"r2": {DecodeInstance: "M1", PrefillInstance: "P0"},
	})
	got := d.Decide(&Request{ID: "r2"}, nil)
	if !got.Disaggregate || got.DecodePodOverride != "M1" || got.PrefillPodHint != "P0" {
		t.Fatalf("disagg action: got %+v, want {Disaggregate:true, DecodePodOverride:M1, PrefillPodHint:P0}", got)
	}
}

func TestFixedPlanDecider_RoutingPreservingActions(t *testing.T) {
	d := NewFixedPlanDecider(map[string]FixedPlanAction{
		"local":  {PrefillInstance: "local"},
		"remote": {PrefillInstance: "auto"},
	})

	local := d.Decide(&Request{ID: "local"}, nil)
	if local.Disaggregate || local.DecodePodOverride != "" || local.PrefillPodHint != "" {
		t.Fatalf("routing-preserving local action = %+v, want no overrides", local)
	}

	remote := d.Decide(&Request{ID: "remote"}, nil)
	if !remote.Disaggregate || remote.DecodePodOverride != "" || remote.PrefillPodHint != "" {
		t.Fatalf("routing-preserving remote action = %+v, want disaggregate with no overrides", remote)
	}
}

func TestFixedPlanDecider_MissingRequestPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on request absent from plan (plans must be total, R1)")
		}
	}()
	d := NewFixedPlanDecider(map[string]FixedPlanAction{"r1": {DecodeInstance: "M0", PrefillInstance: "local"}})
	d.Decide(&Request{ID: "absent"}, nil)
}

func TestLoadFixedPlanCSV(t *testing.T) {
	dir := t.TempDir()
	p := dir + "/plan.csv"
	if err := os.WriteFile(p, []byte("request_id,decode_instance,prefill_instance\nr1,M0,local\nr2,M1,P0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := LoadFixedPlanCSV(p)
	if err != nil {
		t.Fatal(err)
	}
	if plan["r1"] != (FixedPlanAction{"M0", "local"}) || plan["r2"] != (FixedPlanAction{"M1", "P0"}) {
		t.Fatalf("parsed plan wrong: %+v", plan)
	}
}
