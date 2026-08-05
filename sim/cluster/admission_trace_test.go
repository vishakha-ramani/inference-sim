package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// wantPreds returns the six expected predictions with the same oracle/deployable
// separation BuildAdmissionRecords applies: the four deployable variants
// (waiting/little/fluid/rollforward) see a context with every running request's
// TrueRemaining censored to -1, while the two _oracle variants see the raw context.
func wantPreds(t *testing.T, ctx sim.AdmissionContext) [6]float64 {
	t.Helper()
	deployable := ctx
	rc := make([]sim.RunningReqState, len(ctx.Running))
	copy(rc, ctx.Running)
	for i := range rc {
		rc[i].TrueRemaining = -1
	}
	deployable.Running = rc

	names := [6]string{"waiting", "little", "fluid", "rollforward", "fluid_oracle", "rollforward_oracle"}
	var out [6]float64
	for i, n := range names {
		est, err := sim.NewAdmissionEstimator(n)
		if err != nil {
			t.Fatalf("NewAdmissionEstimator(%q): %v", n, err)
		}
		c := deployable
		if n == "fluid_oracle" || n == "rollforward_oracle" {
			c = ctx
		}
		out[i] = est.EstimateTAdm(c)
	}
	return out
}

func checkPreds(t *testing.T, got [6]float64, want [6]float64) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prediction %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// A local (non-disaggregated) request yields exactly one row with pool="local",
// realized t_adm = local_schedule − local_enqueue, and all six predictions equal
// to each estimator's EstimateTAdm(ctx).
func TestBuildAdmissionRecords_Local(t *testing.T) {
	ctx := sim.AdmissionContext{
		QWork: 4000, Mu: 2.0, BatchSize: 8, MaxBatchSize: 8,
		FreeKVBlocks: 0, ReqKVNeed: 4, TIter: 100, QueueDepth: 3,
		AdmissionRate: 0.001, RemainingStepsEst: 10,
		Running: []sim.RunningReqState{{StepsDone: 2, KVBlocks: 4, TrueRemaining: 5}},
	}
	cs := &ClusterSimulator{
		recordAdmissionTrace: true,
		parentRequests:       map[string]*ParentRequest{},
		localAdmitTimes:      map[string]int64{"r1": 1500},
		localEnqueueTimes:    map[string]int64{"r1": 300},
		admissionCtx: map[string]*capturedAdmission{
			"r1": {decodeCtx: ctx},
		},
	}
	recs := cs.BuildAdmissionRecords()
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d: %+v", len(recs), recs)
	}
	r := recs[0]
	if r.RequestID != "r1" || r.Pool != "local" {
		t.Fatalf("unexpected id/pool: %+v", r)
	}
	if r.RealizedTAdm != 1200 { // 1500 − 300
		t.Fatalf("realized t_adm = schedule−enqueue: got %v want 1200", r.RealizedTAdm)
	}
	want := wantPreds(t, ctx)
	got := [6]float64{r.TAdmPredWaiting, r.TAdmPredLittle, r.TAdmPredFluid, r.TAdmPredRollforward, r.TAdmPredFluidOracle, r.TAdmPredRollforwardOracle}
	checkPreds(t, got, want)
}

// The deployable rollforward must NOT read TrueRemaining even when it is populated
// (oracle mode); only rollforward_oracle may. With RemainingStepsEst ≠ TrueRemaining,
// the two predictions differ, and the oracle one uses TrueRemaining=2 → ~2000µs.
func TestBuildAdmissionRecords_OracleDeployableSeparation(t *testing.T) {
	// One running req: TrueRemaining=2 (oracle) but RemainingStepsEst=50 (deployable estimate).
	// Batch full and no free KV so rollforward must look ahead (not short-circuit to 0).
	ctx := sim.AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
		RemainingStepsEst: 50,
		Running:           []sim.RunningReqState{{StepsDone: 100, KVBlocks: 10, TrueRemaining: 2}},
	}
	cs := &ClusterSimulator{
		recordAdmissionTrace: true,
		parentRequests:       map[string]*ParentRequest{},
		localAdmitTimes:      map[string]int64{"r1": 2000},
		localEnqueueTimes:    map[string]int64{"r1": 0},
		admissionCtx: map[string]*capturedAdmission{
			"r1": {decodeCtx: ctx},
		},
	}
	recs := cs.BuildAdmissionRecords()
	var r trace.AdmissionRecord
	for _, x := range recs {
		if x.RequestID == "r1" && x.Pool == "local" {
			r = x
		}
	}
	// deployable rollforward uses RemainingStepsEst=50 → ~50000µs; oracle uses TrueRemaining=2 → ~2000µs.
	if r.TAdmPredRollforward == r.TAdmPredRollforwardOracle {
		t.Fatalf("deployable and oracle rollforward must differ (oracle leakage); got both %v", r.TAdmPredRollforward)
	}
	if r.TAdmPredRollforwardOracle < 1999 || r.TAdmPredRollforwardOracle > 2001 {
		t.Fatalf("oracle should use TrueRemaining=2 → ~2000, got %v", r.TAdmPredRollforwardOracle)
	}
	// The captured context must not be mutated (still readable as oracle).
	if ctx.Running[0].TrueRemaining != 2 {
		t.Fatalf("captured context Running[0].TrueRemaining mutated: got %v want 2", ctx.Running[0].TrueRemaining)
	}
}

// A disaggregated parent yields two rows (prefill + decode), sorted by request_id,
// each with predictions computed against its own captured context.
func TestBuildAdmissionRecords_Disaggregated(t *testing.T) {
	prefillCtx := sim.AdmissionContext{QWork: 9000, Mu: 3.0, BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 10, ReqKVNeed: 2, TIter: 50, QueueDepth: 1, AdmissionRate: 0.002, RemainingStepsEst: 5}
	decodeCtx := sim.AdmissionContext{QWork: 6000, Mu: 2.0, BatchSize: 8, MaxBatchSize: 8, FreeKVBlocks: 0, ReqKVNeed: 4, TIter: 120, QueueDepth: 4, AdmissionRate: 0.001, RemainingStepsEst: 12, Running: []sim.RunningReqState{{StepsDone: 1, KVBlocks: 4, TrueRemaining: 3}}}

	cs := &ClusterSimulator{
		recordAdmissionTrace: true,
		localAdmitTimes:      map[string]int64{},
		localEnqueueTimes:    map[string]int64{},
		parentRequests: map[string]*ParentRequest{
			"p1": {
				ID:                  "p1",
				PrefillInstanceID:   "i0",
				DecodeInstanceID:    "i1",
				PrefillEnqueueTime:  100,
				PrefillScheduleTime: 250,
				DecodeEnqueueTime:   400,
				DecodeScheduleTime:  1000,
			},
		},
		admissionCtx: map[string]*capturedAdmission{
			"p1": {prefillCtx: prefillCtx, decodeCtx: decodeCtx, hasPrefill: true},
		},
	}
	recs := cs.BuildAdmissionRecords()
	if len(recs) != 2 {
		t.Fatalf("want 2 records (prefill+decode), got %d: %+v", len(recs), recs)
	}
	// Sorted by request_id: both share "p1"; deterministic tie-break by pool.
	byPool := map[string]int{}
	for i, r := range recs {
		byPool[r.Pool] = i
	}
	pi, ok := byPool["prefill"]
	if !ok {
		t.Fatalf("missing prefill row: %+v", recs)
	}
	di, ok := byPool["decode"]
	if !ok {
		t.Fatalf("missing decode row: %+v", recs)
	}
	if recs[pi].RealizedTAdm != 150 { // 250−100
		t.Fatalf("prefill realized t_adm: got %v want 150", recs[pi].RealizedTAdm)
	}
	if recs[di].RealizedTAdm != 600 { // 1000−400
		t.Fatalf("decode realized t_adm: got %v want 600", recs[di].RealizedTAdm)
	}
	wp := wantPreds(t, prefillCtx)
	gp := [6]float64{recs[pi].TAdmPredWaiting, recs[pi].TAdmPredLittle, recs[pi].TAdmPredFluid, recs[pi].TAdmPredRollforward, recs[pi].TAdmPredFluidOracle, recs[pi].TAdmPredRollforwardOracle}
	checkPreds(t, gp, wp)
	wd := wantPreds(t, decodeCtx)
	gd := [6]float64{recs[di].TAdmPredWaiting, recs[di].TAdmPredLittle, recs[di].TAdmPredFluid, recs[di].TAdmPredRollforward, recs[di].TAdmPredFluidOracle, recs[di].TAdmPredRollforwardOracle}
	checkPreds(t, gd, wd)
}

// TestPoolFilteredSnapshots_CarryAdmissionRate is the regression for the `little`
// estimator producing 0 on every EDPP disaggregation decision. The EDPP decode-first
// routing path (DisaggregationDecisionEvent / executeDisaggregatedRouting) and the
// prefill-pool snapshot closure both source their snapshots from
// buildPoolFilteredSnapshots, NOT buildRouterState. buildRouterState populates
// snap.AdmissionRate (from WindowedAdmissionRate) and snap.DispatchRate when admission
// detail is enabled; buildPoolFilteredSnapshots did not, so the AdmissionContext the
// decider assembled always had AdmissionRate=0 → littleEstimator returns 0
// unconditionally. This test records admissions on an instance with admission detail
// enabled and asserts the pool-filtered snapshot carries the same non-zero
// AdmissionRate that buildRouterState reports (parity).
func TestPoolFilteredSnapshots_CarryAdmissionRate(t *testing.T) {
	config := newTestEDPPDeploymentConfig(2, 1, 1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)
	// Enable admission detail on every instance (as --edpp-admission-trace does), so the
	// windowed admission-rate counter records and snapshots may carry it.
	for _, inst := range cs.instances {
		inst.SetAdmissionDetail(false)
	}
	// Record admissions on the prefill-pool instance within the rolling window.
	var prefillInst *InstanceSimulator
	for _, inst := range cs.instances {
		if cs.poolMembership[string(inst.ID())].Has(PoolRolePrefill) {
			prefillInst = inst
			break
		}
	}
	if prefillInst == nil {
		t.Fatal("no prefill-pool instance found")
	}
	cs.clock = 500_000
	prefillInst.recordAdmission(100_000)
	prefillInst.recordAdmission(200_000)
	prefillInst.recordAdmission(300_000)

	wantRate := prefillInst.WindowedAdmissionRate(cs.clock)
	if wantRate <= 0 {
		t.Fatalf("precondition: WindowedAdmissionRate should be >0 after admissions, got %v", wantRate)
	}

	// buildRouterState (the non-PD path) already carries it — establish the target.
	rs := buildRouterState(cs, nil)
	var rsRate float64
	for _, s := range rs.Snapshots {
		if s.ID == string(prefillInst.ID()) {
			rsRate = s.AdmissionRate
		}
	}
	if rsRate <= 0 {
		t.Fatalf("buildRouterState should carry AdmissionRate>0, got %v", rsRate)
	}

	// The PD path must reach parity: the pool-filtered snapshot for the same instance
	// must carry the same non-zero AdmissionRate.
	pool := cs.buildPoolFilteredSnapshots(PoolRolePrefill)
	var pfRate float64
	for _, s := range pool {
		if s.ID == string(prefillInst.ID()) {
			pfRate = s.AdmissionRate
		}
	}
	if pfRate != rsRate {
		t.Fatalf("buildPoolFilteredSnapshots AdmissionRate = %v, want parity with buildRouterState %v (little estimator reads 0 otherwise)", pfRate, rsRate)
	}
}

// TestBuildPoolFilteredSnapshots_CarriesGPUType is a regression for the EDPP
// disaggregation path missing per-instance GPU type on its routing snapshots.
// buildRouterState (the non-PD path) has always set snap.GPUType = inst.GPU()
// (cluster_event.go), but buildPoolFilteredSnapshots — the snapshot source for
// the EDPP decode-first routing path — did not, so any decider logic keyed on
// GPUType (e.g. per-GPU coefficient lookup, coeffs_by_gpu) would silently see ""
// on every disaggregated decision. newTestEDPPDeploymentConfig configures GPU
// "H100" via ModelHardwareConfig, so every instance's InstanceSimulator.GPU()
// returns "H100"; the pool-filtered decode snapshots must carry it too.
func TestBuildPoolFilteredSnapshots_CarriesGPUType(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(1)), nil)

	const wantGPU = "H100"
	for _, inst := range cs.instances {
		if inst.GPU() != wantGPU {
			t.Fatalf("precondition: instance %s GPU() = %q, want %q", inst.ID(), inst.GPU(), wantGPU)
		}
	}

	snaps := cs.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(snaps) == 0 {
		t.Fatal("expected >=1 decode snapshot")
	}
	for _, s := range snaps {
		if s.GPUType != wantGPU {
			t.Fatalf("snapshot %s GPUType = %q, want %q", s.ID, s.GPUType, wantGPU)
		}
	}
}
