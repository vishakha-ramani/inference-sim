package sim

import (
	"math"
	"testing"
)

func newSLOExternalityTestDecider(
	t *testing.T,
	cfgMutate func(*EDPPConfig),
	cache map[string]func([]TokenID) int,
	prefill func() []RoutingSnapshot,
) *EDPPDecider {
	t.Helper()
	cfg := defaultTestEDPPConfig()
	cfg.JointSLOExternality = true
	cfg.TauE2EUs = 500_000
	if cfgMutate != nil {
		cfgMutate(&cfg)
	}
	return NewEDPPDecider(cfg, newTestAffineModel(), cache, prefill)
}

func TestSLOExternalityConfigRejectsAmbiguousModes(t *testing.T) {
	for name, mutate := range map[string]func(*EDPPConfig){
		"joint and decomposed": func(cfg *EDPPConfig) {
			cfg.JointSLOExternality = true
			cfg.DecomposedSLOExternality = true
		},
		"historical and constrained": func(cfg *EDPPConfig) {
			cfg.JointSLOExternality = true
			cfg.JointCausalVar = true
		},
		"ablation without policy": func(cfg *EDPPConfig) {
			cfg.SLOExternalityNoCapacity = true
		},
		"nonpositive V": func(cfg *EDPPConfig) {
			cfg.JointSLOExternality = true
			cfg.V = 0
		},
		"occupancy capacity without policy": func(cfg *EDPPConfig) {
			cfg.SLOExternalityOccupancyCapacity = true
			cfg.SLOCapacityReferenceBatch = 4
		},
		"occupancy capacity without reference width": func(cfg *EDPPConfig) {
			cfg.JointSLOExternality = true
			cfg.SLOExternalityOccupancyCapacity = true
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := defaultTestEDPPConfig()
			mutate(&cfg)
			assertPanics(t, func() { NewEDPPDecider(cfg, newTestAffineModel(), nil, nil) })
		})
	}
}

func TestSLOOccupancyCapacityBooksPaperDemandsAndSharesPrefillBaseline(t *testing.T) {
	d := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.SLOExternalityOccupancyCapacity = true
		cfg.SLOCapacityReferenceBatch = 4
		cfg.ChunkTokens = 32
	}, coldCacheQuery("D0", "D1", "P0"), nil)
	d.refreshSLOCapacity(1_000,
		[]RoutingSnapshot{{ID: "D0"}, {ID: "D1"}},
		[]RoutingSnapshot{{ID: "P0"}},
	)

	local := reqBatch("local-occupancy", 64)
	nOut := d.reqNHatOut(local)
	wp := d.coeffs.Wp(64, 64)
	decode := nOut*d.coeffs.AlphaD/4 + d.coeffs.Wd(64, nOut)
	prefill := 2*d.coeffs.AlphaP + wp
	d.OnRoute(local, local.ID, false, 64, "D0", "")
	assertClose(t, "local collocated occupancy", d.SLOCapacityForTest()["D0"].Q, decode+wp)

	remote := reqBatch("remote-occupancy", 64)
	d.OnRoute(remote, remote.ID, true, 64, "D1", "P0")
	queues := d.SLOCapacityForTest()
	assertClose(t, "remote decode occupancy", queues["D1"].Q, decode)
	assertClose(t, "remote prefill occupancy", queues["P0"].Q, prefill)
	assertClose(t, "collocation baseline saving", decode+prefill-queues["D0"].Q, 2*d.coeffs.AlphaP)
	assertClose(t, "occupancy drain rate", queues["D0"].Mu, 1)
	assertClose(t, "occupancy physical scale", queues["D0"].Scale, 1_000_000)

	d.refreshSLOCapacity(1_100, []RoutingSnapshot{{ID: "D0"}, {ID: "D1"}}, []RoutingSnapshot{{ID: "P0"}})
	assertClose(t, "wall-time drain", d.SLOCapacityForTest()["D0"].Q, math.Max(decode+wp-100, 0))
}

func TestSLOOccupancyCapacityCandidateUsesSameDemandAsCommit(t *testing.T) {
	d := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.SLOExternalityOccupancyCapacity = true
		cfg.SLOCapacityReferenceBatch = 8
		cfg.ChunkTokens = 32
	}, coldCacheQuery("D0"), nil)
	ds := RoutingSnapshot{ID: "D0"}
	d.refreshSLOCapacity(10_000, []RoutingSnapshot{ds}, nil)
	background := reqBatch("background-occupancy", 64)
	d.bookSLOCapacityWork(background, false, "D0", "")
	q := d.SLOCapacityForTest()["D0"].Q

	req := reqBatch("candidate-occupancy", 64)
	ec := &jointEvalCtx{
		req: req, n: d.normFor(req.SLOClass), reqKVNeed: d.reqKVNeed(req),
		nHatOut: d.reqNHatOut(req), nowUs: 10_000,
	}
	score := d.jointSLOExternalityCandidateScore(ec, ds, nil)
	demand := d.sloLocalOccupancy(d.coeffs, 64, 64, ec.nHatOut)
	want := (q / 1_000_000) * (demand / 1_000_000)
	assertClose(t, "traced occupancy queue", score.capacityQueueDecode, q)
	assertClose(t, "traced occupancy demand", score.capacityDemandDecode, demand)
	assertClose(t, "occupancy candidate cross term", score.capacityDecode, want)

	before := q
	d.bookSLOCapacityWork(req, false, "D0", "")
	assertClose(t, "candidate/commit occupancy agreement", d.SLOCapacityForTest()["D0"].Q-before, demand)
}

func assertClose(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s = %g, want %g", label, got, want)
	}
}

func TestSLOCompositeRoutingValueUsesTTFTAndE2EOnly(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100, tauITLUs: 50, tauE2EUs: 500}
	base := sloCompositeValue(slo, 50, 250)
	for name, got := range map[string]float64{
		"TTFT": sloCompositeValue(slo, 150, 250),
		"E2E":  sloCompositeValue(slo, 50, 750),
	} {
		if got >= base {
			t.Errorf("worsening %s produced value %g, want less than baseline %g", name, got, base)
		}
	}
	withoutITL := slo
	withoutITL.tauITLUs = 0
	if got := sloCompositeValue(withoutITL, 50, 250); got != base {
		t.Fatalf("ITL target changed routing value: got %g, want %g", got, base)
	}
}

func TestDecodeCompositeRetainsRealizedTTFTFactor(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100, tauITLUs: 50, tauE2EUs: 500}
	fastFirstToken := varDecodeCoResident{
		rem: 5, arrivalUs: 0, firstTokenUs: 50, ttftSet: true, slo: slo,
	}
	slowFirstToken := fastFirstToken
	slowFirstToken.firstTokenUs = 200

	fastValue := gDecodeComposite(fastFirstToken, 250)
	slowValue := gDecodeComposite(slowFirstToken, 250)
	if fastValue <= slowValue {
		t.Fatalf("decode-resident values fast=%g slow=%g; realized TTFT factor was not retained", fastValue, slowValue)
	}
}

func TestPrefillResidentCompositeIsTTFTOnly(t *testing.T) {
	resident := varPrefillCoResident{
		arrivalUs: 0, remDecodeSteps: 100,
		slo: varSLO{tauTTFTUs: 100, tauITLUs: 10, tauE2EUs: 200},
	}
	base := gCollocComposite(resident, 50, 150)
	if got := gCollocComposite(resident, 50, 10_000); got != base {
		t.Fatalf("prefill resident value changed with unobserved decode horizon: got %g, want %g", got, base)
	}
	if got := gCollocComposite(resident, 150, 10_000); got >= base {
		t.Fatalf("prefill resident value did not decrease with TTFT: got %g, baseline %g", got, base)
	}
}

func TestSLOCapacityQueueDrainsByElapsedServiceNotAdmission(t *testing.T) {
	d := newSLOExternalityTestDecider(t, nil, coldCacheQuery("D0"), nil)
	decode := []RoutingSnapshot{{ID: "D0"}}
	d.refreshSLOCapacity(1_000, decode, nil)
	req := reqBatch("r", 32)
	d.OnRoute(req, "r", false, 32, "D0", "")

	before := d.SLOCapacityForTest()["D0"]
	wantBooked := d.coeffs.Wp(32, 32) + d.coeffs.Wd(32, d.reqNHatOut(req))
	assertClose(t, "booked capacity work", before.Q, wantBooked)

	d.OnAdmit("r", false)
	afterAdmission := d.SLOCapacityForTest()["D0"]
	assertClose(t, "capacity work after admission", afterAdmission.Q, before.Q)

	d.refreshSLOCapacity(1_100, decode, nil)
	afterTime := d.SLOCapacityForTest()["D0"]
	assertClose(t, "capacity work after elapsed service", afterTime.Q, math.Max(before.Q-100*before.Mu, 0))
}

func TestSLOCapacityFullyCachedRequestStillBooksDecodeWork(t *testing.T) {
	cache := map[string]func([]TokenID) int{
		"D0": func(tokens []TokenID) int { return len(tokens) / 16 },
	}
	d := newSLOExternalityTestDecider(t, nil, cache, nil)
	d.refreshSLOCapacity(0, []RoutingSnapshot{{ID: "D0"}}, nil)
	req := reqBatch("cached", 32)
	d.OnRoute(req, "cached", false, 0, "D0", "")

	got := d.SLOCapacityForTest()["D0"].Q
	want := d.coeffs.Wd(32, d.reqNHatOut(req))
	assertClose(t, "fully cached capacity work", got, want)
	if got <= 0 {
		t.Fatal("fully cached request booked no decode work")
	}
	assertClose(t, "fully cached admission work", d.QByInstance()["D0"].Wd, want)
	d.OnAdmit("cached", false)
	assertClose(t, "admitted fully cached waiting work", d.QByInstance()["D0"].Wd, 0)
	assertClose(t, "admitted fully cached capacity work", d.SLOCapacityForTest()["D0"].Q, want)
}

func TestSLOCapacityBooksLocalAndRemotePlacements(t *testing.T) {
	cache := coldCacheQuery("D0", "P0")
	d := newSLOExternalityTestDecider(t, nil, cache, nil)
	decode := []RoutingSnapshot{{ID: "D0"}}
	prefill := []RoutingSnapshot{{ID: "P0"}}
	d.refreshSLOCapacity(0, decode, prefill)

	local := reqBatch("local", 40)
	d.OnRoute(local, "local", false, 40, "D0", "")
	wantLocal := d.coeffs.Wp(40, 40) + d.coeffs.Wd(40, d.reqNHatOut(local))
	assertClose(t, "local decoder work", d.SLOCapacityForTest()["D0"].Q, wantLocal)

	remote := reqBatch("remote", 60)
	d.OnRoute(remote, "remote", true, 60, "D0", "P0")
	wantDecode := wantLocal + d.coeffs.Wd(60, d.reqNHatOut(remote))
	wantPrefill := d.coeffs.Wp(60, 60)
	queues := d.SLOCapacityForTest()
	assertClose(t, "remote decode work", queues["D0"].Q, wantDecode)
	assertClose(t, "remote prefill work", queues["P0"].Q, wantPrefill)
}

func TestSLOWorkBookingUsesCommittedLocationCoefficients(t *testing.T) {
	base := defaultTestEDPPConfig().Coeffs
	decodeCoeffs := base
	decodeCoeffs.C0 = 250
	prefillCoeffs := base
	prefillCoeffs.CPf = 25
	d := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.CoeffsByGPU = map[string]EDPPCoeffs{"decode-gpu": decodeCoeffs, "prefill-gpu": prefillCoeffs}
	}, coldCacheQuery("D0", "P0"), nil)
	d.refreshSLOCapacity(0,
		[]RoutingSnapshot{{ID: "D0", GPUType: "decode-gpu"}},
		[]RoutingSnapshot{{ID: "P0", GPUType: "prefill-gpu"}},
	)
	req := reqBatch("heterogeneous", 100)
	d.OnRoute(req, req.ID, true, 100, "D0", "P0")

	queues := d.SLOCapacityForTest()
	assertClose(t, "heterogeneous decode booking", queues["D0"].Q, decodeCoeffs.Wd(100, d.reqNHatOut(req)))
	assertClose(t, "heterogeneous prefill booking", queues["P0"].Q, prefillCoeffs.Wp(100, 100))
	waiting := d.QByInstance()
	assertClose(t, "heterogeneous decode admission work", waiting["D0"].Wd, decodeCoeffs.Wd(100, d.reqNHatOut(req)))
	assertClose(t, "heterogeneous prefill admission work", waiting["P0"].Wp, prefillCoeffs.Wp(100, 100))
}

func TestSLOExternalityCandidateComponentsSumToScore(t *testing.T) {
	prefillSnaps := []RoutingSnapshot{{ID: "P0"}}
	d := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.V = 3
		cfg.JointCandidateTraceEnabled = true
	}, coldCacheQuery("D0", "D1", "P0"), func() []RoutingSnapshot { return prefillSnaps })
	decodeSnaps := []RoutingSnapshot{{ID: "D0"}, {ID: "D1"}}
	d.refreshSLOCapacity(10_000, decodeSnaps, prefillSnaps)
	d.bookSLOCapacityWork(reqBatch("background", 20_000), false, "D0", "")

	state := &RouterState{Clock: 10_000, SelectedInstance: "D0", Snapshots: decodeSnaps}
	dec := d.Decide(reqBatch("candidate", 200), state)
	if dec.EDPPJointCandidates == nil {
		t.Fatal("missing candidate trace")
	}
	if got := len(dec.EDPPJointCandidates.Candidates); got != 4 {
		t.Fatalf("candidate rows = %d, want D(P+1)=4", got)
	}
	chosen := 0
	for _, row := range dec.EDPPJointCandidates.Candidates {
		assertClose(t, "net-good cost", row.NetGoodCost, row.SLOExternality-row.OwnGood)
		assertClose(t, "capacity total", row.CapacityTotal, row.CapacityDecode+row.CapacityPrefill)
		assertClose(t, "candidate score", row.Score, d.cfg.V*row.NetGoodCost+row.CapacityTotal)
		if row.Chosen {
			chosen++
			assertClose(t, "chosen score", row.Score, row.BestScore)
			assertClose(t, "chosen score regret", row.ChosenScoreRegret, 0)
		}
	}
	if chosen != 1 {
		t.Fatalf("chosen candidate rows = %d, want 1", chosen)
	}
}

func TestSLOExternalityRemoteTTFTOverlapsDecodeQueueDrainage(t *testing.T) {
	prefillSnaps := []RoutingSnapshot{{ID: "P0"}}
	d := newSLOExternalityTestDecider(t, nil, coldCacheQuery("D0", "P0"), func() []RoutingSnapshot { return prefillSnaps })
	d.qByInstance["D0"] = &edppInstWork{wd: 1_000}
	req := reqBatch("remote", 200)
	ds := RoutingSnapshot{ID: "D0", BatchSize: 1}
	ps := prefillSnaps[0]
	ec := &jointEvalCtx{
		req: req, n: d.normFor(req.SLOClass), reqKVNeed: d.reqKVNeed(req),
		nHatOut: d.reqNHatOut(req), nowUs: 10_000,
	}

	score := d.jointSLOExternalityCandidateScore(ec, ds, &ps)
	thetaD := d.coeffsFor(ds.GPUType)
	thetaP := d.coeffsFor(ps.GPUType)
	tAdmD := d.tadmEstimator.EstimateTAdm(d.jointDecodeAdmissionCtx(ec, ds))
	tAdmP := d.tadmEstimator.EstimateTAdm(d.jointPrefillAdmissionCtx(ec, ps))
	apP := d.apForInstance(req, ps.ID)
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	remoteLead := tAdmP + nChunksP*thetaP.tIterPrefill(ps.ResidentPrefillTokens) +
		thetaP.Wp(apP, len(req.InputTokens)) + d.cXferUsFor(req)
	firstDecode := thetaD.tIterDecode(ds.BatchSize+1, ds.KvTokensInUse+int64(len(req.InputTokens)), ds.ResidentPrefillTokens)
	overlapTTFT := d.projectedDisaggTTFT(math.Max(remoteLead, tAdmD), firstDecode)
	serialTTFT := d.projectedDisaggTTFT(remoteLead+tAdmD, firstDecode)
	want := d.jointSelfGoodWithKernel(ec, thetaD, ds, overlapTTFT, varKernelComposite)
	serialValue := d.jointSelfGoodWithKernel(ec, thetaD, ds, serialTTFT, varKernelComposite)

	assertClose(t, "overlap-aware own good", score.ownGood, want)
	if math.Abs(score.ownGood-serialValue) < 1e-9 {
		t.Fatalf("fixture does not distinguish overlap and serial TTFT: value=%g", score.ownGood)
	}
}

func TestDecomposedSLOExternalityKeepsScorerDecode(t *testing.T) {
	d := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.JointSLOExternality = false
		cfg.DecomposedSLOExternality = true
	}, coldCacheQuery("D0", "D1"), nil)
	state := &RouterState{
		Clock: 10_000, SelectedInstance: "D1",
		Snapshots: []RoutingSnapshot{{ID: "D0"}, {ID: "D1", BatchSize: 10, KvTokensInUse: 10_000}},
	}
	dec := d.Decide(reqBatch("decomposed", 200), state)
	if dec.DecodePodOverride != "D1" {
		t.Fatalf("decomposed policy changed scorer decode to %q, want D1", dec.DecodePodOverride)
	}
}

func TestSLOExternalityAblationsRemoveOnlyTheirNamedComponent(t *testing.T) {
	d := newSLOExternalityTestDecider(t, nil, coldCacheQuery("D0"), nil)
	ds := RoutingSnapshot{ID: "D0", BatchSize: 1, RunningDecode: []RunningReqState{{
		StepsDone: 0, ArrivalUs: 0, FirstTokenUs: 1_000, TTFTSet: true,
	}}}
	d.refreshSLOCapacity(10_000, []RoutingSnapshot{ds}, nil)
	d.bookSLOCapacityWork(reqBatch("background", 20_000), false, "D0", "")
	req := reqBatch("scored", 200)
	ec := &jointEvalCtx{
		req: req, n: d.normFor(req.SLOClass), reqKVNeed: d.reqKVNeed(req),
		nHatOut: d.reqNHatOut(req), nowUs: 10_000,
	}
	base := d.jointSLOExternalityCandidateScore(ec, ds, nil)
	if base.externality <= 0 || base.capacityTotal <= 0 {
		t.Fatalf("test fixture is not discriminating: externality=%g capacity=%g", base.externality, base.capacityTotal)
	}

	withoutExternality := *d
	withoutExternality.cfg.SLOExternalityNoExternality = true
	noExt := withoutExternality.jointSLOExternalityCandidateScore(ec, ds, nil)
	assertClose(t, "no-externality own good", noExt.ownGood, base.ownGood)
	assertClose(t, "no-externality capacity", noExt.capacityTotal, base.capacityTotal)
	assertClose(t, "removed externality", noExt.externality, 0)

	withoutCapacity := *d
	withoutCapacity.cfg.SLOExternalityNoCapacity = true
	noCapacity := withoutCapacity.jointSLOExternalityCandidateScore(ec, ds, nil)
	assertClose(t, "no-capacity externality", noCapacity.externality, base.externality)
	assertClose(t, "no-capacity own good", noCapacity.ownGood, base.ownGood)
	assertClose(t, "removed capacity", noCapacity.capacityTotal, 0)

	withoutOwnGood := *d
	withoutOwnGood.cfg.SLOExternalityNoOwnGood = true
	noOwnGood := withoutOwnGood.jointSLOExternalityCandidateScore(ec, ds, nil)
	assertClose(t, "no-own-good externality", noOwnGood.externality, base.externality)
	assertClose(t, "removed own good", noOwnGood.ownGood, 0)
	assertClose(t, "no-own-good capacity", noOwnGood.capacityTotal, base.capacityTotal)
}

func TestSLOCapacityPriceAvoidsSacrificeDecoder(t *testing.T) {
	state := &RouterState{
		Clock: 10_000, SelectedInstance: "D0",
		Snapshots: []RoutingSnapshot{
			{ID: "D0", BatchSize: 1, RunningDecode: []RunningReqState{{
				StepsDone: 0, ArrivalUs: -1_000_000_000, FirstTokenUs: 1_000, TTFTSet: true,
			}}},
			{ID: "D1", BatchSize: 1, RunningDecode: []RunningReqState{{
				StepsDone: 0, ArrivalUs: 0, FirstTokenUs: 1_000, TTFTSet: true,
			}}},
		},
	}
	req := reqBatch("foreground", 200)

	noCapacity := newSLOExternalityTestDecider(t, func(cfg *EDPPConfig) {
		cfg.SLOExternalityNoCapacity = true
	}, coldCacheQuery("D0", "D1"), nil)
	withoutPrice := noCapacity.Decide(req, state)
	if withoutPrice.DecodePodOverride != "D0" {
		t.Fatalf("no-capacity control chose %q, want the low-value sacrifice decoder D0", withoutPrice.DecodePodOverride)
	}

	full := newSLOExternalityTestDecider(t, nil, coldCacheQuery("D0", "D1"), nil)
	full.refreshSLOCapacity(state.Clock, state.Snapshots, nil)
	full.bookSLOCapacityWork(reqBatch("background", 200_000), false, "D0", "")
	withPrice := full.Decide(req, state)
	if withPrice.DecodePodOverride != "D1" {
		t.Fatalf("capacity-aware policy chose %q, want D1 rather than the loaded sacrifice decoder D0", withPrice.DecodePodOverride)
	}
}
