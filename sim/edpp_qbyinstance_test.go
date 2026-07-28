package sim

import (
	"math"
	"testing"
)

func floatEq(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// newTestEDPPDecider builds a decider with known coeffs, mirroring the
// defaultTestEDPPConfig()/newTestAffineModel() setup used elsewhere in edpp_test.go.
func newTestEDPPDecider(t *testing.T) *EDPPDecider {
	t.Helper()
	return NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
}

func TestQByInstance_SumsMatchPoolLevel(t *testing.T) {
	d := newTestEDPPDecider(t)
	// disagg route: prefill work → prefill inst "P0", decode work → decode inst "M1"
	d.OnRoute(&Request{ID: "r1", InputTokens: make([]TokenID, 1000), SLOClass: "batch"}, "r1", true, 1000, "M1", "P0")
	// local route: prefill+decode both → "M0"
	d.OnRoute(&Request{ID: "r2", InputTokens: make([]TokenID, 500), SLOClass: "batch"}, "r2", false, 500, "M0", "")

	q := d.QByInstance()
	// P0 holds r1's prefill work; M1 holds r1's decode work; M0 holds r2's (wp+wd).
	if q["P0"].Wp <= 0 || q["P0"].Wd != 0 {
		t.Fatalf("P0 = %+v, want Wp>0 Wd==0", q["P0"])
	}
	if q["M1"].Wd <= 0 || q["M1"].Wp != 0 {
		t.Fatalf("M1 = %+v, want Wd>0 Wp==0", q["M1"])
	}
	if q["M0"].Wd <= 0 {
		t.Fatalf("M0 = %+v, want Wd>0 (local wp+wd)", q["M0"])
	}
	// INVARIANT: per-instance sums equal the pool-level scalars (byte-identical reduced bookkeeping).
	var sumWp, sumWd float64
	for _, v := range q {
		sumWp += v.Wp
		sumWd += v.Wd
	}
	if !floatEq(sumWp, d.qpWork) || !floatEq(sumWd, d.qdWork) {
		t.Fatalf("per-instance sums (wp=%v wd=%v) != pool-level (qp=%v qd=%v)", sumWp, sumWd, d.qpWork, d.qdWork)
	}
}
