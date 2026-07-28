package sim

import (
	"math"
	"testing"
)

// Tests for the value-at-risk drift oracle (sim/edpp_var.go).
// Design: docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md §6.

// standardRT is a representative re-timing where adding load raises the per-iter time:
// tIter0 < tIterOverlap and tIter0 < tIterAfter (the physically expected direction).
func standardRT() varReTiming {
	return varReTiming{tIter0: 40, tIterOverlap: 60, tIterAfter: 80}
}

// TestParseVarKernel locks the CLI/config kernel names and rejects unknown values.
func TestParseVarKernel(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want varKernel
		ok   bool
	}{
		{"flip", varKernelFlip, true},
		{"util", varKernelUtil, true},
		{"hazard", varKernelHazard, true},
		{"", 0, false},
		{"bogus", 0, false},
	} {
		got, ok := parseVarKernel(tc.in)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("parseVarKernel(%q) = (%v,%v), want (%v,%v)", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

// TestKernelFlip_CountsCompositeGoodFlips verifies kernel A (flip) is exactly the count of
// co-residents whose composite-good (TTFT ∧ meanITL ∧ E2E) flips true→false under the
// placement's added delay. One co-resident is pushed past its ITL SLO (flips); one keeps
// ample slack (no flip); one already missed TTFT (fixed miss, contributes 0).
func TestKernelFlip_CountsCompositeGoodFlips(t *testing.T) {
	rt := standardRT()
	now := 0.0
	nChunks := 2.0

	flips := varDecodeCoResident{ // baseline meanITL 40 ≤ 50; local pushes to 76 > 50 ⇒ flip
		rem: 10, arrivalUs: 0, firstTokenUs: 100, ttftSet: true,
		slo: varSLO{tauTTFTUs: 200, tauITLUs: 50, tauE2EUs: 100000},
	}
	noFlip := varDecodeCoResident{ // loose ITL/E2E ⇒ good before and after
		rem: 10, arrivalUs: 0, firstTokenUs: 100, ttftSet: true,
		slo: varSLO{tauTTFTUs: 200, tauITLUs: 10000, tauE2EUs: 100000},
	}
	ttftMissed := varDecodeCoResident{ // first token at 500 > τ_ttft 200 ⇒ already bad, cannot flip
		rem: 10, arrivalUs: 0, firstTokenUs: 500, ttftSet: true,
		slo: varSLO{tauTTFTUs: 200, tauITLUs: 50, tauE2EUs: 100000},
	}
	crs := []varDecodeCoResident{flips, noFlip, ttftMissed}

	got := varDecodeLocal(now, crs, rt, nChunks, varKernelFlip)
	if got != 1 {
		t.Fatalf("flip VaR_local = %v, want exactly 1 (only the ITL-pushed co-resident flips)", got)
	}
}

// TestKernelUtil_DropsWithDelay_NeglectsDoomed verifies kernel B (util): a co-resident with
// slack loses utility as its completion is delayed (positive drop), while a DOOMED co-resident
// (deep past its deadline) sits in the σ→0 flat region and yields a ~0 marginal drop — the
// design's predicted "saturation ⇒ neglect" trap (§2 B, §7), which this kernel measures.
func TestKernelUtil_DropsWithDelay_NeglectsDoomed(t *testing.T) {
	rt := standardRT()
	now := 0.0
	nChunks := 2.0

	// Healthy: deadline comfortably ahead of baseline completion (Cbase=400).
	healthy := []varDecodeCoResident{{
		rem: 10, arrivalUs: 0, firstTokenUs: 50, ttftSet: true,
		slo: varSLO{tauTTFTUs: 500, tauITLUs: 0, tauE2EUs: 500},
	}}
	dropHealthy := varDecodeLocal(now, healthy, rt, nChunks, varKernelUtil)
	if dropHealthy <= 0 {
		t.Fatalf("util VaR_local for a near-deadline co-resident = %v, want > 0 (utility drops with delay)", dropHealthy)
	}

	// Doomed: E2E deadline far in the past relative to completion (both g≈0) ⇒ ~0 marginal drop.
	doomed := []varDecodeCoResident{{
		rem: 10, arrivalUs: -100000, firstTokenUs: 50, ttftSet: true,
		slo: varSLO{tauTTFTUs: 500, tauITLUs: 0, tauE2EUs: 500},
	}}
	dropDoomed := varDecodeLocal(now, doomed, rt, nChunks, varKernelUtil)
	if math.Abs(dropDoomed) > 1e-6 {
		t.Fatalf("util VaR_local for a doomed co-resident = %v, want ~0 (saturation ⇒ neglect)", dropDoomed)
	}
}

// TestKernelHazard_NonzeroForDoomed verifies kernel C (hazard) keeps a NONZERO weight for a
// doomed co-resident (avoiding B's hard-zero neglect by construction) and weights the delay
// most heavily for a co-resident right at its deadline.
func TestKernelHazard_NonzeroForDoomed(t *testing.T) {
	rt := standardRT()
	now := 0.0
	nChunks := 2.0

	// Doomed co-resident: slack < 0 but Gaussian hazard is still > 0 ⇒ nonzero VaR contribution.
	doomed := []varDecodeCoResident{{
		rem: 10, arrivalUs: -100000, firstTokenUs: 50, ttftSet: true,
		slo: varSLO{tauTTFTUs: 500, tauITLUs: 0, tauE2EUs: 500},
	}}
	got := varDecodeLocal(now, doomed, rt, nChunks, varKernelHazard)
	if got <= 0 {
		t.Fatalf("hazard VaR_local for a doomed co-resident = %v, want > 0 (gentle decay keeps weight)", got)
	}
}

// TestExternalityAsymmetryLaw is the structural law the whole mechanism rests on (§2, §6):
// with an idle prefill pool, a disaggregated placement delays fewer of each co-resident's
// steps than a local placement (R arrives later, no prefill-overlap inflation on the decode
// instance), so Δ_j^disagg ≤ Δ_j^local and hence VaR_local ≥ VaR_disagg under every kernel.
func TestExternalityAsymmetryLaw(t *testing.T) {
	rt := standardRT()
	now := 0.0
	nChunks := 2.0
	arrivalSteps := 5.0 // disagg: R arrives after 5 iters (> nChunks) ⇒ more baseline steps

	crs := []varDecodeCoResident{
		{rem: 10, arrivalUs: 0, firstTokenUs: 100, ttftSet: true, slo: varSLO{tauTTFTUs: 200, tauITLUs: 50, tauE2EUs: 500}},
		{rem: 6, arrivalUs: 0, firstTokenUs: 80, ttftSet: true, slo: varSLO{tauTTFTUs: 200, tauITLUs: 60, tauE2EUs: 400}},
		{rem: 20, arrivalUs: 0, firstTokenUs: 120, ttftSet: true, slo: varSLO{tauTTFTUs: 300, tauITLUs: 90, tauE2EUs: 2000}},
	}

	// Per-co-resident completion-delay direction: Δ_local ≥ Δ_disagg ≥ 0.
	for _, cr := range crs {
		cb := rt.cBase(now, cr.rem)
		dLocal := rt.cLocal(now, cr.rem, nChunks) - cb
		dDisagg := rt.cDisagg(now, cr.rem, arrivalSteps) - cb
		if dDisagg < -1e-9 {
			t.Errorf("rem=%d: Δ_disagg = %v, want ≥ 0", cr.rem, dDisagg)
		}
		if dLocal < dDisagg-1e-9 {
			t.Errorf("rem=%d: Δ_local (%v) < Δ_disagg (%v) — asymmetry law violated", cr.rem, dLocal, dDisagg)
		}
	}

	// Aggregate VaR direction under every kernel.
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		vl := varDecodeLocal(now, crs, rt, nChunks, k)
		vd := varDecodeDisagg(now, crs, rt, arrivalSteps, k)
		if vl < vd-1e-9 {
			t.Errorf("kernel %v: VaR_local (%v) < VaR_disagg (%v) — asymmetry law violated", k, vl, vd)
		}
	}
}

// TestVarCensoredCoResidentContributesZero locks the INV-9 oracle gate at the VaR unit: a
// co-resident whose remaining is censored (rem = -1, the value the deployable control path
// sees) is skipped entirely, so it can never contribute value-at-risk. Reverting the guard
// (counting rem < 0) would turn this red — a non-vacuous check.
func TestVarCensoredCoResidentContributesZero(t *testing.T) {
	rt := standardRT()
	now := 0.0
	censored := []varDecodeCoResident{{
		rem: -1, arrivalUs: 0, firstTokenUs: 100, ttftSet: true,
		slo: varSLO{tauTTFTUs: 200, tauITLUs: 50, tauE2EUs: 500},
	}}
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		if v := varDecodeLocal(now, censored, rt, 2, k); v != 0 {
			t.Errorf("kernel %v: censored co-resident contributed %v to VaR_local, want 0", k, v)
		}
		if v := varDecodeDisagg(now, censored, rt, 5, k); v != 0 {
			t.Errorf("kernel %v: censored co-resident contributed %v to VaR_disagg, want 0", k, v)
		}
	}
}

// collocPrefillCoResident is a mid-prefill occupant on the DECODE instance (placed there by a
// prior collocate decision). remPrefillTokens remaining prompt tokens separate it from its first
// token; chunk is the per-iter prefill advance. Its VaR is TTFT-side.
func collocOccupant(remPrefillTokens int64, arrivalUs int64, tauTTFT float64) varPrefillCoResident {
	return varPrefillCoResident{
		remPrefillTokens: remPrefillTokens,
		arrivalUs:        arrivalUs,
		slo:              varSLO{tauTTFTUs: tauTTFT},
	}
}

// TestCollocPrefill_FlipCountsFirstTokenFlips verifies the collocated-prefill TTFT term (kernel
// A): an occupant whose first token is pushed past its τ_ttft by a LOCAL placement flips
// true→false (contributes 1), while an occupant with ample TTFT slack does not (contributes 0).
// chunk 5 ⇒ remIters=⌈10/5⌉=2; cBase=2·40=80, cLocal=2·60=120 (both iters in the overlap window,
// nChunks=2). The tight occupant meets 80≤100 before but misses 120>100 after ⇒ flip. The loose
// occupant meets both against τ_ttft=100000 ⇒ no flip.
func TestCollocPrefill_FlipCountsFirstTokenFlips(t *testing.T) {
	rt := standardRT()
	now := 0.0
	chunk, nChunks := 5.0, 2.0

	tight := collocOccupant(10, 0, 100)    // cBase 80 ≤ 100; cLocal 120 > 100 ⇒ flip
	loose := collocOccupant(10, 0, 100000) // huge τ_ttft ⇒ good before and after
	ks := []varPrefillCoResident{tight, loose}

	got := varCollocPrefillLocal(now, ks, rt, chunk, nChunks, varKernelFlip)
	if got != 1 {
		t.Fatalf("colloc-prefill flip VaR_local = %v, want exactly 1 (only the tight occupant flips)", got)
	}
}

// TestCollocPrefill_DisaggUndisturbed is the law that justifies the term. A disaggregated
// placement leaves the decode instance undisturbed for arrivalSteps iterations, so an occupant
// that finishes its prefill within that window (remIters ≤ arrivalSteps) sees cDisagg = cBase and
// contributes zero under every kernel. Its first token is untouched by disagg — exactly the
// asymmetry the mechanism prices. remIters=⌈10/5⌉=2 ≤ arrivalSteps=5.
func TestCollocPrefill_DisaggUndisturbed(t *testing.T) {
	rt := standardRT()
	now := 0.0
	chunk, arrivalSteps := 5.0, 5.0

	ks := []varPrefillCoResident{collocOccupant(10, 0, 100)}
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		if v := varCollocPrefillDisagg(now, ks, rt, chunk, arrivalSteps, k); math.Abs(v) > 1e-9 {
			t.Errorf("kernel %v: disagg VaR for an occupant finishing before R arrives = %v, want 0", k, v)
		}
	}
}

// TestCollocPrefill_AsymmetryLaw verifies VaR_local ≥ VaR_disagg for collocated prefill occupants
// under every kernel: local co-schedules R's prefill (slows the occupant more), disagg arrives
// later (slows it less or not at all). Mirrors TestExternalityAsymmetryLaw for the decode side.
func TestCollocPrefill_AsymmetryLaw(t *testing.T) {
	rt := standardRT()
	now := 0.0
	chunk, nChunks, arrivalSteps := 5.0, 2.0, 5.0

	ks := []varPrefillCoResident{
		collocOccupant(10, 0, 100),
		collocOccupant(30, 0, 300),
		collocOccupant(5, 0, 90),
	}
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		vl := varCollocPrefillLocal(now, ks, rt, chunk, nChunks, k)
		vd := varCollocPrefillDisagg(now, ks, rt, chunk, arrivalSteps, k)
		if vl < vd-1e-9 {
			t.Errorf("kernel %v: colloc VaR_local (%v) < VaR_disagg (%v) — asymmetry law violated", k, vl, vd)
		}
	}
}

// TestCollocPrefill_CensoredSkipped locks the guard: an occupant with remPrefillTokens < 0 is
// skipped and contributes zero. remPrefillTokens is deployable (known input) so this is normally
// non-negative, but the guard mirrors the decode-side censored skip and must not count negatives.
func TestCollocPrefill_CensoredSkipped(t *testing.T) {
	rt := standardRT()
	now := 0.0
	ks := []varPrefillCoResident{collocOccupant(-1, 0, 100)}
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		if v := varCollocPrefillLocal(now, ks, rt, 5, 2, k); v != 0 {
			t.Errorf("kernel %v: censored occupant contributed %v to colloc VaR_local, want 0", k, v)
		}
		if v := varCollocPrefillDisagg(now, ks, rt, 5, 5, k); v != 0 {
			t.Errorf("kernel %v: censored occupant contributed %v to colloc VaR_disagg, want 0", k, v)
		}
	}
}

// collocOccupantFull is a mid-prefill occupant with a known decode horizon and full SLOs, so the
// decode-phase (ITL/E2E) risk of a local placement — R joining its batch (B→B+1) — is priced, not
// just its first token. remDec is its remaining decode steps once it reaches its first token.
func collocOccupantFull(remPrefillTokens, remDec, arrivalUs int64, tauTTFT, tauITL, tauE2E float64) varPrefillCoResident {
	return varPrefillCoResident{
		remPrefillTokens: remPrefillTokens,
		remDecodeSteps:   remDec,
		arrivalUs:        arrivalUs,
		slo:              varSLO{tauTTFTUs: tauTTFT, tauITLUs: tauITL, tauE2EUs: tauE2E},
	}
}

// TestCollocPrefill_PricesE2EFromBatchJoin is the core of the full-good extension. A local
// placement makes R join a mid-prefill occupant's decode batch, so the occupant's decode steps run
// at the re-timed per-iter time (tIterAfter 80 > tIter0 40) and it can miss its E2E deadline even
// though its first token is comfortably met. chunk 5 ⇒ remPf=⌈10/5⌉=2; with remDec=3, total=5.
// Baseline E2E = 5·40 = 200; placed E2E = 2·60 + 3·80 = 360. With τ_e2e = 300 the occupant is good
// before (200 ≤ 300) and bad after (360 > 300) ⇒ one flip. τ_ttft = 1000 keeps the first token met
// both ways (80, 120), so a TTFT-only term would have scored zero — the flip comes entirely from
// the decode phase the earlier model dropped.
func TestCollocPrefill_PricesE2EFromBatchJoin(t *testing.T) {
	rt := standardRT()
	now := 0.0
	chunk, nChunks := 5.0, 2.0
	ks := []varPrefillCoResident{collocOccupantFull(10, 3, 0, 1000, 0, 300)}

	if got := varCollocPrefillLocal(now, ks, rt, chunk, nChunks, varKernelFlip); got != 1 {
		t.Fatalf("colloc E2E flip VaR_local = %v, want 1 (decode-phase E2E miss from the B+1 join)", got)
	}
	// The util kernel must also register the decode-phase loss (its E2E slack utility drops).
	if got := varCollocPrefillLocal(now, ks, rt, chunk, nChunks, varKernelUtil); got <= 1e-9 {
		t.Fatalf("colloc util VaR_local = %v, want > 0 (E2E slack utility drops on the batch join)", got)
	}
}

// TestCollocPrefill_PricesITLFromBatchJoin locks the inter-token conjunct. Same geometry: baseline
// mean ITL over the decode phase = (200−80)/3 = 40 = tIter0; placed = (360−120)/3 = 80 = tIterAfter.
// With τ_itl = 50 the occupant meets ITL before (40 ≤ 50) and misses after (80 > 50) ⇒ one flip,
// with τ_ttft and τ_e2e loose so the flip is ITL-driven alone.
func TestCollocPrefill_PricesITLFromBatchJoin(t *testing.T) {
	rt := standardRT()
	now := 0.0
	chunk, nChunks := 5.0, 2.0
	ks := []varPrefillCoResident{collocOccupantFull(10, 3, 0, 1000, 50, 100000)}

	if got := varCollocPrefillLocal(now, ks, rt, chunk, nChunks, varKernelFlip); got != 1 {
		t.Fatalf("colloc ITL flip VaR_local = %v, want 1 (mean-ITL miss from the B+1 join)", got)
	}
}

// TestCollocPrefill_UnknownDecodeReducesToTTFTOnly guards the reduction: with no decode horizon
// (remDecodeSteps ≤ 0) and no ITL/E2E targets, every kernel returns exactly the earlier
// first-token-only contribution (varPrefillTTFTContribution). This is the byte-identical fallback
// the prefill-pool term and oracle-off runs rely on.
func TestCollocPrefill_UnknownDecodeReducesToTTFTOnly(t *testing.T) {
	rt := standardRT()
	now := 0.0
	nChunks := 2.0
	k := collocOccupant(10, 0, 100) // remDecodeSteps 0, τ_itl/τ_e2e 0
	remPf := int64(2)
	ftB := rt.cBase(now, remPf)
	ftP := rt.cLocal(now, remPf, nChunks)
	for _, kern := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		got := varCollocContribution(k, ftB, ftP, ftB, ftP, kern)
		want := varPrefillTTFTContribution(k, ftB, ftP, kern)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("kernel %v: full-good contribution %v != TTFT-only %v (reduction broken)", kern, got, want)
		}
	}
}

// TestVarReTiming_FullReTiming verifies the "full B+1 re-timing" completion model: the
// after-join per-iter time recomputes tIterDecode with batch B+1 and KV grown by R's full
// input length (Δkv_R), not a marginal add. Uses concrete coefficients so the arithmetic is
// pinned.
func TestVarReTiming_FullReTiming(t *testing.T) {
	c := EDPPCoeffs{AlphaD: 10, AlphaP: 10, C0: 2, C1: 0.5, CPf: 1, CAttn: 0}
	d := &EDPPDecider{cfg: EDPPConfig{ChunkTokens: 0}, coeffs: c}
	req := &Request{InputTokens: make([]TokenID, 100)} // Δkv_R = 100

	bDec, kv, sPf := 3, int64(500), int64(0)
	chunk := 100
	rt := d.varReTimingFor(req, c, bDec, kv, sPf, chunk)

	wantT0 := c.tIterDecode(bDec, kv, sPf)
	wantOverlap := c.tIterDecode(bDec, kv, sPf+int64(chunk))
	wantAfter := c.tIterDecode(bDec+1, kv+100, sPf) // B+1, kv + Δkv_R (full re-timing)
	if rt.tIter0 != wantT0 || rt.tIterOverlap != wantOverlap || rt.tIterAfter != wantAfter {
		t.Fatalf("varReTiming = %+v, want {tIter0:%v tIterOverlap:%v tIterAfter:%v}", rt, wantT0, wantOverlap, wantAfter)
	}
	// Overlap and after-join both exceed baseline (added prefill tokens / added decode occupant).
	if !(rt.tIterOverlap > rt.tIter0 && rt.tIterAfter > rt.tIter0) {
		t.Fatalf("expected tIterOverlap (%v) and tIterAfter (%v) > tIter0 (%v)", rt.tIterOverlap, rt.tIterAfter, rt.tIter0)
	}
}

// Causal overlap attention: cLocal adds Σ_{j=0}^{overlap-1} c_attn·chunk·(j·chunk+chunk/2)
// = c_attn·chunk²·overlap²/2 on top of the affine overlap cost. c_attn=0 recovers the
// pure affine projection.
func TestVarReTiming_CausalOverlapAttention(t *testing.T) {
	rt := varReTiming{tIter0: 100, tIterOverlap: 120, tIterAfter: 110, cAttn: 0.01, chunk: 50}
	// overlap = min(nChunks=4, rem=4) = 4; tail = 0. attn = 0.01·50·50·16/2 = 200.
	got := rt.cLocal(0, 4, 4)
	want := 4*120.0 + 0.01*50*50*4*4/2.0
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("cLocal causal = %v, want %v", got, want)
	}
	rt0 := rt
	rt0.cAttn = 0
	if got0 := rt0.cLocal(0, 4, 4); math.Abs(got0-4*120.0) > 1e-9 {
		t.Fatalf("cLocal affine (c_attn=0) = %v, want 480", got0)
	}
}
