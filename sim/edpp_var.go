package sim

import "math"

// Value-at-risk (VaR) drift oracle for EDPP.
//
// Design: docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md.
//
// The reduced/joint EDPP rule prices the backlog externality of a placement in WORK
// (microseconds). This unit re-prices that externality in VALUE — the marginal goodput
// destroyed among the co-residents already running on the candidate decode instance (and,
// for a disaggregated placement, the prefill-pool co-residents delayed by the request's
// remote prefill). The rule then compares VaR_local − VaR_disagg (the value-currency
// externality) against the unchanged z-weighted TTFT/ITL/transfer self terms.
//
// It is a DIAGNOSTIC ORACLE (UPPER BOUND): the co-resident remaining-step counts it reads
// (varReTiming's baseline / the completion model) come from the un-censored oracle
// TrueRemaining (INV-9 violation, gated behind the same admissionDetailOracle switch as the
// other EDPP oracles). Everything else it reads — arrival, realized first-token, class,
// SLO targets — is deployable/input-derived.
//
// All functions here are pure (no hidden state): identical inputs ⇒ identical float output
// (INV-6). The decider (sim/edpp.go) assembles the inputs and calls varReducedLHS / the
// per-candidate joint helper.

// varKernel selects the g(·) scoring kernel (design §2). All three are exercised by the
// experiment: A (flip) is the hyperparameter-free ceiling; B (util) makes the predicted
// "saturation ⇒ neglect" trap measurable; C (hazard) is the smoothed deployable-target shape.
type varKernel int

const (
	varKernelFlip   varKernel = iota // A: binary composite-good flip count (true→false)
	varKernelUtil                    // B: saturating slack-utility drop
	varKernelHazard                  // C: deadline-slack hazard weight × completion delay
	// composite is reserved for the constrained causal-SLO-externality policy. It
	// uses one smooth TTFT×E2E routing value for the arriving request and every
	// decode-side resident charge. Mean ITL remains an evaluation-goodput gate,
	// but is deliberately not a direct factor in this routing surrogate.
	// Historical util/hazard behavior remains unchanged.
	varKernelComposite
)

// parseVarKernel maps the CLI/config string to a kernel. ok=false for an unknown value
// (the config boundary turns this into a panic, R3).
func parseVarKernel(s string) (varKernel, bool) {
	switch s {
	case "flip":
		return varKernelFlip, true
	case "util":
		return varKernelUtil, true
	case "hazard":
		return varKernelHazard, true
	}
	return 0, false
}

// varSLO bundles the per-co-resident SLO thresholds g() evaluates against (all µs). Resolved
// per co-resident CLASS (not the deciding request's class), because a co-resident meeting or
// missing its own SLO is what defines the goodput destroyed. A zero threshold disables that
// dimension's conjunct in g() (matching cluster.SLOAttainmentMultiDim, which skips a
// dimension whose target is 0).
type varSLO struct {
	tauTTFTUs float64
	tauITLUs  float64
	tauE2EUs  float64
}

// varDecodeCoResident is one decode co-resident's VaR inputs. rem is the oracle remaining
// decode steps (un-censored TrueRemaining); rem < 0 marks a censored/unknown co-resident,
// which contributes zero VaR (skipped). arrivalUs/firstTokenUs/ttftSet are deployable
// (input-derived) and fix the realized TTFT.
type varDecodeCoResident struct {
	rem          int64
	arrivalUs    int64
	firstTokenUs int64
	ttftSet      bool
	slo          varSLO
}

// varPrefillCoResident is one prefilling co-resident. remPrefillTokens is its remaining prompt
// tokens — deployable (known input length), never oracle — and sets its first-token horizon.
//
// remDecodeSteps is its remaining DECODE steps once it reaches its first token. It is the
// collocated occupant's decode-phase horizon: a local placement makes R join this occupant's
// decode batch (B→B+1), so every one of these steps runs at the re-timed per-iter time, which is
// the occupant's inter-token and end-to-end risk. Source mirrors the decode co-resident: the
// oracle output length (OracleOutputLen) or, deployable, the censored per-class N̂_out (INV-9-safe,
// no output-length read). It is ≤ 0 when unknown, and then only the first-token risk is priced —
// which recovers the earlier TTFT-only behavior. The prefill-POOL term (varPrefillDisagg) leaves
// it unset because R never joins a pool occupant's decode batch, so only its first token is at risk.
type varPrefillCoResident struct {
	remPrefillTokens int64
	remDecodeSteps   int64
	arrivalUs        int64
	slo              varSLO
}

// varReTiming holds the batch-level per-iteration decode times the completion model needs
// (design §2, "full B+1 re-timing"). Every co-resident in one decode batch shares one
// per-iteration time, so these are batch-level, not per-co-resident. All µs.
//
//   - tIter0:       current batch B per-iter time, tIterDecode(B, kv, sPf) — the baseline.
//   - tIterOverlap: local placement, while the deciding request R prefills co-scheduled on the
//     decode batch (chunked prefill adds `chunk` resident prefill tokens):
//     tIterDecode(B, kv, sPf+chunk) = tIter0 + C_pf·chunk = tIter0 + δ_pf-chunk.
//   - tIterAfter:   after R joins the decode batch, tIterDecode(B+1, kv+Δkv_R, sPf), where
//     Δkv_R = R's resident context tokens (its full input length; input-only, oracle-safe).
//     "Full re-timing" = recompute tIterDecode with B+1 and kv+Δkv_R (not a marginal add).
type varReTiming struct {
	tIter0       float64
	tIterOverlap float64
	tIterAfter   float64
	// cAttn and chunk parameterize the causal prefill-attention added to the overlap
	// window in cLocal: each of R's co-scheduled chunks attends to its causal prefix.
	cAttn float64
	chunk float64

	// Exact-prefill-overlap ablation. The legacy model assumes every overlapping
	// chunk is full and starts at prefix zero. The exact form uses the known
	// uncached span [ar-ap, ar), handles the partial last chunk, and charges only
	// marginal prefill work above the baseline decode iteration.
	exactPrefillOverlap bool
	cPf                 float64
	ap                  float64
	ar                  float64
}

// cBase is a decode co-resident's projected completion with rem steps left at the current
// (pre-R) batch per-iter time. now is the decision instant (µs).
func (rt varReTiming) cBase(nowUs float64, rem int64) float64 {
	return nowUs + float64(rem)*rt.tIter0
}

// cLocal is the zero-admission-delay compatibility form of cLocalAfter.
func (rt varReTiming) cLocal(nowUs float64, rem int64, nChunks float64) float64 {
	return rt.cLocalAfter(nowUs, rem, 0, nChunks)
}

// cLocalAfter is the co-resident's completion under LOCAL placement when R waits
// admissionSteps baseline iterations before joining the running batch. Co-residents
// first execute min(admissionSteps, rem) iterations undisturbed. Of the surviving
// tail, the first min(nChunks, remaining) iterations overlap R's prefill and the
// rest run at the B+1 re-timed decode rate.
func (rt varReTiming) cLocalAfter(nowUs float64, rem int64, admissionSteps, nChunks float64) float64 {
	pre := math.Min(math.Max(admissionSteps, 0), float64(rem))
	remaining := float64(rem) - pre
	overlap := math.Min(nChunks, remaining)
	if rt.exactPrefillOverlap {
		return nowUs + pre*rt.tIter0 + overlap*rt.tIter0 +
			prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
			(remaining-overlap)*rt.tIterAfter
	}
	// Causal prefill attention over R's co-scheduled chunks j=0..overlap-1, each charged
	// against causal prefix j·chunk + chunk/2 (start prefix 0; matches prefix_length:0
	// workloads). Σ = c_attn·chunk²·overlap²/2.
	attn := rt.cAttn * rt.chunk * rt.chunk * overlap * overlap / 2.0
	return nowUs + pre*rt.tIter0 + overlap*rt.tIterOverlap + attn + (remaining-overlap)*rt.tIterAfter
}

// prefillMarginalWork returns the exact E3 work added by the first `iterations`
// chunks of an arriving request's uncached prefill. The known cached prefix is
// ar-ap, processed=min(ap, iterations*chunk), and the integrated causal work is
//
//	CPf·processed + CAttn·processed·(cachedPrefix + processed/2).
//
// It excludes baseline iteration time: co-residents would pay that even if the
// arriving request were absent.
func prefillMarginalWork(cPf, cAttn, ap, ar, chunk, iterations float64) float64 {
	if ap <= 0 || chunk <= 0 || iterations <= 0 {
		return 0
	}
	processed := math.Min(ap, iterations*chunk)
	cachedPrefix := math.Max(ar-ap, 0)
	return cPf*processed + cAttn*processed*(cachedPrefix+processed/2.0)
}

// cDisagg is the co-resident's completion under DISAGG placement: R's prefill runs remotely,
// so the decode instance is undisturbed for arrivalSteps iterations (R's remote prefill + KV
// transfer window, in units of tIter0); only the tail max(rem−arrivalSteps, 0) steps run at
// the B+1 re-timed per-iter time. No prefill-overlap inflation on this instance.
func (rt varReTiming) cDisagg(nowUs float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(arrivalSteps, float64(rem))
	return nowUs + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

// ttftMet reports whether a decode co-resident met its TTFT SLO. TTFT is realized (a
// decoding co-resident has already produced its first token) and therefore fixed under this
// placement decision. A co-resident that has NOT produced a first token (ttftSet false) or
// that missed its TTFT contributes zero to the flip/util VaR (g stays 0 before and after).
func (cr varDecodeCoResident) ttftMet() bool {
	if !cr.ttftSet {
		return false
	}
	if cr.slo.tauTTFTUs <= 0 {
		return true // no TTFT target configured ⇒ trivially met
	}
	return float64(cr.firstTokenUs-cr.arrivalUs) <= cr.slo.tauTTFTUs
}

// gDecodeFlip is kernel A's composite-good indicator for a decode co-resident completing at
// cUs (µs): 1 iff TTFT met (fixed) ∧ mean-ITL met ∧ E2E ≤ deadline, else 0. The mean ITL over
// the remaining steps is (cUs − now)/rem, the average per-iter time the co-resident would
// experience under the placement (baseline uses tIter0, which meets τ_itl iff the batch was
// already SLO-feasible). deadline = arrival + τ_e2e.
func gDecodeFlip(cr varDecodeCoResident, cUs, nowUs float64) float64 {
	if !cr.ttftMet() {
		return 0
	}
	if cr.rem <= 0 {
		return 1 // no remaining steps: ITL/E2E already realized; treat as good (no flip)
	}
	if cr.slo.tauE2EUs > 0 && cUs > float64(cr.arrivalUs)+cr.slo.tauE2EUs {
		return 0
	}
	if cr.slo.tauITLUs > 0 {
		meanITL := (cUs - nowUs) / float64(cr.rem)
		if meanITL > cr.slo.tauITLUs {
			return 0
		}
	}
	return 1
}

// gDecodeUtil is kernel B's saturating slack utility for a decode co-resident completing at
// cUs: σ((deadline − cUs)/scale). It saturates to ~1 with comfortable E2E slack and decays to
// ~0 past the deadline. scale is the E2E deadline budget's τ_ttft (a natural latency unit); a
// doomed co-resident sits deep in the σ→0 flat region, so its marginal utility drop is ~0 —
// the design's predicted "saturation ⇒ neglect" trap, which this kernel exists to MEASURE.
func gDecodeUtil(cr varDecodeCoResident, cUs float64) float64 {
	deadline := float64(cr.arrivalUs) + cr.slo.tauE2EUs
	scale := cr.slo.tauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return 1.0 / (1.0 + math.Exp(-(deadline-cUs)/scale))
}

// sloCompositeValue is the constrained policy's bounded routing value. It is a
// smooth TTFT×E2E surrogate; final goodput still gates TTFT, mean ITL, and E2E.
// Each enabled routing dimension uses its own fixed SLO scale as the transition
// bandwidth; disabled targets contribute one.
func sloCompositeValue(slo varSLO, ttftUs, e2eUs float64) float64 {
	u := 1.0
	if slo.tauTTFTUs > 0 {
		u *= sigmoid((slo.tauTTFTUs - ttftUs) / slo.tauTTFTUs)
	}
	if slo.tauE2EUs > 0 {
		u *= sigmoid((slo.tauE2EUs - e2eUs) / slo.tauE2EUs)
	}
	return u
}

func gDecodeComposite(cr varDecodeCoResident, cUs float64) float64 {
	if !cr.ttftSet {
		return 0
	}
	ttft := float64(cr.firstTokenUs - cr.arrivalUs)
	return sloCompositeValue(cr.slo, ttft, cUs-float64(cr.arrivalUs))
}

// hazardWeight is kernel C's deadline-slack hazard for a decode co-resident whose BASELINE
// completion is cBaseUs: a heavy-tailed (Cauchy-like) bump 1/(1+x²) peaking at slack 0 (right
// at the E2E deadline) and decaying GENTLY (polynomially, not Gaussian-fast) on both sides, so
// even a deeply doomed co-resident (large negative slack) keeps a small NONZERO weight —
// avoiding kernel B's hard-zero neglect by construction (design §2 C). band is the τ_ttft (an
// SLO-derived latency scale). The kernel C VaR contribution is hazardWeight · Δ (the completion
// delay caused by the placement).
func hazardWeight(cr varDecodeCoResident, cBaseUs float64) float64 {
	slack := float64(cr.arrivalUs) + cr.slo.tauE2EUs - cBaseUs
	band := cr.slo.tauTTFTUs
	if band <= 0 {
		band = 1
	}
	x := slack / band
	return 1.0 / (1.0 + x*x)
}

// varDecodeLocal sums, over the decode co-residents, the value-at-risk of the LOCAL
// placement: Σ_j contribution(j) where the co-resident is delayed from cBase to cLocal.
// For flip/util the contribution is g(before) − g(after); for hazard it is
// hazardWeight(slack) · (cLocal − cBase). Censored co-residents (rem < 0) are skipped.
func varDecodeLocal(nowUs float64, crs []varDecodeCoResident, rt varReTiming, nChunks float64, kernel varKernel) float64 {
	return varDecodeLocalAfter(nowUs, crs, rt, nChunks, 0, kernel)
}

// varDecodeLocalAfter is varDecodeLocal with an explicit local-admission
// window. R cannot delay a co-resident until that window has elapsed.
func varDecodeLocalAfter(nowUs float64, crs []varDecodeCoResident, rt varReTiming, nChunks, admissionSteps float64, kernel varKernel) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(nowUs, cr.rem)
		cp := rt.cLocalAfter(nowUs, cr.rem, admissionSteps, nChunks)
		sum += varDecodeContribution(cr, cb, cp, nowUs, kernel)
	}
	return sum
}

// varDecodeDisagg is the DISAGG mirror of varDecodeLocal: the co-resident is delayed from
// cBase to cDisagg (only its tail steps re-timed, R arriving after arrivalSteps iterations).
func varDecodeDisagg(nowUs float64, crs []varDecodeCoResident, rt varReTiming, arrivalSteps float64, kernel varKernel) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(nowUs, cr.rem)
		cp := rt.cDisagg(nowUs, cr.rem, arrivalSteps)
		sum += varDecodeContribution(cr, cb, cp, nowUs, kernel)
	}
	return sum
}

// varDecodeContribution is one decode co-resident's VaR contribution under the active kernel,
// given its baseline completion cb and placed completion cp (both µs). Shared by the local and
// disagg sums so the two paths use byte-identical arithmetic (INV-6).
func varDecodeContribution(cr varDecodeCoResident, cb, cp, nowUs float64, kernel varKernel) float64 {
	switch kernel {
	case varKernelComposite:
		return gDecodeComposite(cr, cb) - gDecodeComposite(cr, cp)
	case varKernelUtil:
		return gDecodeUtil(cr, cb) - gDecodeUtil(cr, cp)
	case varKernelHazard:
		return hazardWeight(cr, cb) * (cp - cb)
	default: // varKernelFlip
		return gDecodeFlip(cr, cb, nowUs) - gDecodeFlip(cr, cp, nowUs)
	}
}

// varPrefillDisagg sums the value-at-risk imposed on the prefill-pool co-residents by the
// deciding request's remote prefill (disaggregated placement only). Each prefill co-resident's
// first-token completion is pushed by rPrefillUs (the deciding request's added prefill duration
// on the pool). Its flip is TTFT-side: deadline = arrival + τ_ttft. tIterP is the prefill
// pool's per-iter time and chunkP its per-step token budget, used to project the co-resident's
// own remaining first-token time. This is a first-order contention model (design §2/§9): the
// decode-side asymmetry is the dominant mechanism; the asymmetry-law test isolates it by
// requiring an idle prefill pool (no prefill co-residents ⇒ this term is 0).
func varPrefillDisagg(nowUs float64, ks []varPrefillCoResident, tIterP, chunkP, rPrefillUs float64, kernel varKernel) float64 {
	return varPrefillDisaggAfter(nowUs, ks, tIterP, chunkP, 0, rPrefillUs, kernel)
}

// varPrefillDisaggAfter applies the remote request's prefill-pool externality
// only after it is admitted. A running occupant that finishes within
// admissionSteps baseline iterations is unaffected.
func varPrefillDisaggAfter(nowUs float64, ks []varPrefillCoResident, tIterP, chunkP, admissionSteps, rPrefillUs float64, kernel varKernel) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		cb := nowUs + remIters*tIterP
		cp := cb
		if remIters > admissionSteps {
			cp += rPrefillUs
		}
		sum += varPrefillTTFTContribution(k, cb, cp, kernel)
	}
	return sum
}

// varPrefillDisaggExactAfter is the marginal-overlap correction for the remote
// prefill-pool externality. After R's admission, an occupant is delayed only by
// R's prefill chunks that execute before that occupant reaches its first token.
// Baseline prefill iterations are not charged again, and an occupant with one
// remaining iteration is not charged R's entire multi-chunk prompt.
func varPrefillDisaggExactAfter(
	nowUs float64,
	ks []varPrefillCoResident,
	tIterP, chunkP, admissionSteps float64,
	rAp, rAr int,
	coeffs EDPPCoeffs,
	kernel varKernel,
) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	rChunks := math.Ceil(float64(maxInt(rAp, 0)) / chunkP)
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		remainingAfterAdmission := math.Max(remIters-admissionSteps, 0)
		overlap := math.Min(rChunks, remainingAfterAdmission)
		cb := nowUs + remIters*tIterP
		cp := cb + prefillMarginalWork(
			coeffs.CPf,
			coeffs.CAttn,
			float64(rAp),
			float64(rAr),
			chunkP,
			overlap,
		)
		sum += varPrefillTTFTContribution(k, cb, cp, kernel)
	}
	return sum
}

// varPrefillTTFTContribution scores one prefill co-resident's first-token value-at-risk under the
// active kernel, given its baseline first-token completion cb and placed completion cp (both µs).
// The flip is TTFT-side: deadline = arrival + τ_ttft. Shared by the disagg prefill-pool term
// (varPrefillDisagg) and the collocated decode-instance term (varCollocPrefill*) so both score
// first-token risk with byte-identical arithmetic (INV-6).
func varPrefillTTFTContribution(k varPrefillCoResident, cb, cp float64, kernel varKernel) float64 {
	deadline := float64(k.arrivalUs) + k.slo.tauTTFTUs
	switch kernel {
	case varKernelUtil, varKernelComposite:
		scale := k.slo.tauTTFTUs
		if scale <= 0 {
			scale = 1
		}
		return sigmoid((deadline-cb)/scale) - sigmoid((deadline-cp)/scale)
	case varKernelHazard:
		band := k.slo.tauTTFTUs
		if band <= 0 {
			band = 1
		}
		x := (deadline - cb) / band
		return (1.0 / (1.0 + x*x)) * (cp - cb)
	default: // varKernelFlip
		return b2f(k.slo.tauTTFTUs <= 0 || cb <= deadline) - b2f(k.slo.tauTTFTUs <= 0 || cp <= deadline)
	}
}

// varCollocPrefillLocal sums the value-at-risk imposed on the DECODE instance's collocated prefill
// occupants by a LOCAL placement. Such an occupant (placed here by a prior collocate decision) has
// not produced its first token yet, so RunningDecodeState skips it and the decode-side VaR terms
// miss it. A local placement harms it twice. It needs remPf = ⌈remPrefillTokens / chunk⌉ more
// batch iterations to reach its first token, and R's co-scheduled prefill slows those (first-token
// risk). Then R joins the decode batch (B→B+1) and slows its remDecodeSteps decode steps too
// (inter-token and end-to-end risk). We project the occupant's first-token completion over remPf
// iterations and its full completion over remPf + remDecodeSteps iterations, both with the same
// cBase→cLocal model, and price the full good — first token, inter-token, and end-to-end. When the
// decode horizon is unknown (remDecodeSteps ≤ 0) only the first-token risk is priced. Deployable:
// remPrefillTokens is known input length and remDecodeSteps is a censored estimate (INV-9-safe).
func varCollocPrefillLocal(nowUs float64, ks []varPrefillCoResident, rt varReTiming, chunk, nChunks float64, kernel varKernel) float64 {
	return varCollocPrefillLocalAfter(nowUs, ks, rt, chunk, nChunks, 0, kernel)
}

// varCollocPrefillLocalAfter is the admission-causal form of
// varCollocPrefillLocal. R's overlap and B+1 join begin only after the local
// decode admission window.
func varCollocPrefillLocalAfter(nowUs float64, ks []varPrefillCoResident, rt varReTiming, chunk, nChunks, admissionSteps float64, kernel varKernel) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(nowUs, remPf)
		ftP := rt.cLocalAfter(nowUs, remPf, admissionSteps, nChunks)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(nowUs, total)
			eP = rt.cLocalAfter(nowUs, total, admissionSteps, nChunks)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP, kernel)
	}
	return sum
}

// varCollocPrefillDisagg is the DISAGG mirror of varCollocPrefillLocal: R prefills remotely, so
// the decode instance is undisturbed for arrivalSteps iterations. An occupant that reaches its
// first token within that window (remIters ≤ arrivalSteps) sees cDisagg = cBase and contributes
// zero — disagg does not delay it. Only the tail beyond arrivalSteps is re-timed.
func varCollocPrefillDisagg(nowUs float64, ks []varPrefillCoResident, rt varReTiming, chunk, arrivalSteps float64, kernel varKernel) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(nowUs, remPf)
		ftP := rt.cDisagg(nowUs, remPf, arrivalSteps)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(nowUs, total)
			eP = rt.cDisagg(nowUs, total, arrivalSteps)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP, kernel)
	}
	return sum
}

// varCollocContribution scores one collocated occupant's composite value-at-risk under the active
// kernel, given its baseline/placed first-token completions (ftB, ftP) and baseline/placed full
// completions (eB, eP), all µs. It generalizes varPrefillTTFTContribution from a first-token-only
// flip to the full good: when the occupant has no known decode horizon (eB==ftB, eP==ftP and
// tauE2E/tauITL disabled) every kernel reduces to the earlier TTFT-only arithmetic exactly.
func varCollocContribution(k varPrefillCoResident, ftB, ftP, eB, eP float64, kernel varKernel) float64 {
	switch kernel {
	case varKernelComposite:
		return gCollocComposite(k, ftB, eB) - gCollocComposite(k, ftP, eP)
	case varKernelUtil:
		return gCollocUtil(k, ftB, eB) - gCollocUtil(k, ftP, eP)
	case varKernelHazard:
		return collocHazard(k, ftB, ftP, eB, eP)
	default: // varKernelFlip
		return gCollocFlip(k, ftB, eB) - gCollocFlip(k, ftP, eP)
	}
}

// gCollocFlip is kernel A's composite-good indicator for a collocated occupant reaching its first
// token at ftUs and completing at eUs (both µs): 1 iff its first token meets τ_ttft (LIVE — the
// placement can still move it, unlike a decoding co-resident) AND its mean inter-token time over
// the decode phase meets τ_itl AND it completes within τ_e2e, else 0. A zero threshold disables
// that conjunct. The mean ITL is the decode duration (eUs − ftUs) over remDecodeSteps.
func gCollocFlip(k varPrefillCoResident, ftUs, eUs float64) float64 {
	if k.slo.tauTTFTUs > 0 && ftUs > float64(k.arrivalUs)+k.slo.tauTTFTUs {
		return 0
	}
	if k.slo.tauE2EUs > 0 && eUs > float64(k.arrivalUs)+k.slo.tauE2EUs {
		return 0
	}
	if k.slo.tauITLUs > 0 && k.remDecodeSteps > 0 {
		meanITL := (eUs - ftUs) / float64(k.remDecodeSteps)
		if meanITL > k.slo.tauITLUs {
			return 0
		}
	}
	return 1
}

// gCollocUtil is kernel B's saturating slack utility for a collocated occupant: the product of its
// first-token slack utility σ((arrival+τ_ttft − ftUs)/scale) and its end-to-end slack utility
// σ((arrival+τ_e2e − eUs)/scale). Both live for a prefilling occupant. The product mirrors
// gDecodeUtil (the E2E factor) with the extra first-token factor a decoding co-resident lacks
// because its first token is already realized; when τ_e2e is disabled it reduces to the first-token
// utility alone. scale is τ_ttft, the same natural latency unit gDecodeUtil uses.
func gCollocUtil(k varPrefillCoResident, ftUs, eUs float64) float64 {
	scale := k.slo.tauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	u := 1.0
	if k.slo.tauTTFTUs > 0 {
		u *= sigmoid((float64(k.arrivalUs) + k.slo.tauTTFTUs - ftUs) / scale)
	}
	if k.slo.tauE2EUs > 0 {
		u *= sigmoid((float64(k.arrivalUs) + k.slo.tauE2EUs - eUs) / scale)
	}
	return u
}

func gCollocComposite(k varPrefillCoResident, ftUs, _ float64) float64 {
	// A resident that is still prefilling has no assigned decoder state in the
	// routing snapshot. Its declared phase value is therefore TTFT-only; adding
	// a synthetic decode horizon here would make the one-step potential depend
	// on state the router does not observe.
	if k.slo.tauTTFTUs <= 0 {
		return 1
	}
	return sigmoid((k.slo.tauTTFTUs - (ftUs - float64(k.arrivalUs))) / k.slo.tauTTFTUs)
}

// collocHazard is kernel C's deadline-slack hazard for a collocated occupant: the first-token
// hazard (slack at ftB, band τ_ttft) times the first-token delay ftP−ftB, plus the end-to-end
// hazard (slack at eB) times the end-to-end delay eP−eB when a decode horizon is known. Mirrors
// hazardWeight on both the first-token and end-to-end targets. Both delays are non-negative and
// local ≥ disagg, so the asymmetry law is preserved.
func collocHazard(k varPrefillCoResident, ftB, ftP, eB, eP float64) float64 {
	h := collocHazardTerm(float64(k.arrivalUs)+k.slo.tauTTFTUs, ftB, ftP, k.slo.tauTTFTUs)
	if k.slo.tauE2EUs > 0 && k.remDecodeSteps > 0 {
		h += collocHazardTerm(float64(k.arrivalUs)+k.slo.tauE2EUs, eB, eP, k.slo.tauTTFTUs)
	}
	return h
}

// collocHazardTerm is a Cauchy-like hazard 1/(1+x²) at slack (deadline−cB)/band, times the delay
// cP−cB. Shared by the first-token and end-to-end hazards so both use byte-identical arithmetic.
func collocHazardTerm(deadline, cB, cP, band float64) float64 {
	if band <= 0 {
		band = 1
	}
	x := (deadline - cB) / band
	return (1.0 / (1.0 + x*x)) * (cP - cB)
}

// --- decider-bound assembly (reduced + joint) -------------------------------------------

// varSLOFor resolves a co-resident class's SLO thresholds (µs) from the decider config.
func (d *EDPPDecider) varSLOFor(class string) varSLO {
	tt, itl := d.targetsFor(class)
	return varSLO{
		tauTTFTUs: float64(tt),
		tauITLUs:  float64(itl),
		tauE2EUs:  float64(d.e2eFor(class)),
	}
}

// varDecodeInputs converts a snapshot's decode running-request slice into VaR co-resident
// inputs. Each co-resident's remaining decode steps come from one of two sources:
//
//   - ORACLE (default): the un-censored true remaining, TrueRemaining (gated behind the
//     admissionDetailOracle switch). This is a diagnostic upper bound; when the oracle is off
//     TrueRemaining is -1 and the co-resident is skipped (VaR → 0, rule falls to its rhs sign).
//   - DEPLOYABLE (VarDeployable): the censored per-class output-length estimate,
//     max(N̂_out(class) − StepsDone, 1), which reads no hidden output length (INV-9-safe). This
//     mirrors decodeRemStepsEst's per-request censored form and turns the ceiling into a policy.
func (d *EDPPDecider) varDecodeInputs(running []RunningReqState) []varDecodeCoResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]varDecodeCoResident, 0, len(running))
	for _, r := range running {
		rem := r.TrueRemaining
		if d.varDeployable {
			// Censored estimate: a co-resident that has produced StepsDone tokens has output
			// length ≥ StepsDone, so N̂_out is floored by StepsDone before subtracting; the
			// remaining is then floored at 1 (it is still decoding).
			nHat := d.nHatFor(r.SLOClass).mean()
			rem = int64(math.Max(math.Max(nHat, float64(r.StepsDone))-float64(r.StepsDone), 1))
		}
		out = append(out, varDecodeCoResident{
			rem:          rem,
			arrivalUs:    r.ArrivalUs,
			firstTokenUs: r.FirstTokenUs,
			ttftSet:      r.TTFTSet,
			slo:          d.varSLOFor(r.SLOClass),
		})
	}
	return out
}

// varPrefillInputs converts a snapshot's prefill running-request slice into VaR prefill
// co-resident inputs. remPrefillTokens (TrueRemaining on the prefill slice) is deployable —
// remaining prompt tokens, known input, never oracle-gated (see Simulator.RunningPrefillState).
func (d *EDPPDecider) varPrefillInputs(running []RunningReqState) []varPrefillCoResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]varPrefillCoResident, 0, len(running))
	for _, r := range running {
		// Decode-phase horizon for the collocated-occupant term: how many decode steps this
		// occupant will run once it reaches its first token. It has produced no output yet, so
		// the deployable estimate is the full censored per-class N̂_out (no StepsDone to subtract,
		// INV-9-safe); the oracle uses its true output length. ≤ 0 leaves the decode phase unpriced
		// (first-token risk only) and is what the prefill-POOL term relies on. The oracle path is
		// gated to admissionDetailOracle-populated states (OracleOutputLen ≥ 0 only then).
		remDec := int64(0)
		if d.varDeployable {
			remDec = int64(math.Max(d.nHatFor(r.SLOClass).mean(), 1))
		} else if r.OracleOutputLen > 0 {
			remDec = r.OracleOutputLen
		}
		out = append(out, varPrefillCoResident{
			remPrefillTokens: r.TrueRemaining,
			remDecodeSteps:   remDec,
			arrivalUs:        r.ArrivalUs,
			slo:              d.varSLOFor(r.SLOClass),
		})
	}
	return out
}

// varReTimingFor builds the batch-level per-iter decode times for the completion model under
// decode node physics thetaD at batch state (bDec, kv, sPf). chunk is the deciding request's
// prefill token budget (for the local overlap term); Δkv_R = R's full input length (its
// resident context once it joins the decode batch; input-only, oracle-safe).
func (d *EDPPDecider) varReTimingFor(req *Request, thetaD EDPPCoeffs, bDec int, kv, sPf int64, chunk int) varReTiming {
	dkv := req.InputLen()
	return varReTiming{
		tIter0:       thetaD.tIterDecode(bDec, kv, sPf),
		tIterOverlap: thetaD.tIterDecode(bDec, kv, sPf+int64(chunk)),
		tIterAfter:   thetaD.tIterDecode(bDec+1, kv+dkv, sPf),
		cAttn:        thetaD.CAttn,
		chunk:        float64(chunk),
	}
}

// varPathBreakdown separates the co-resident populations a placement can harm.
// Keeping these components explicit makes the aggregate auditable in decision traces.
type varPathBreakdown struct {
	decode        float64
	collocPrefill float64
	prefillPool   float64
}

func (v varPathBreakdown) total() float64 {
	return v.decode + v.collocPrefill + v.prefillPool
}

// varReducedBreakdown computes both reduced-rule placement externalities for
// the deciding request R on its selected decode instance. It mirrors Decide's
// already-computed operands so no physics is recomputed differently (INV-6).
//
//   - VaR_local:  goodput destroyed among the decode co-residents delayed by R's prefill-on-
//     decode then B+1 join.
//   - VaR_disagg: goodput destroyed among the decode co-residents (only their tail steps
//     re-timed, R arriving after ⌈decodeJoinUs/tIter0⌉ iterations) PLUS the prefill-pool co-residents
//     whose first token is pushed by R's remote prefill.
func (d *EDPPDecider) varReducedBreakdown(
	req *Request, nowUs float64,
	decSnap RoutingSnapshot, prefillSnaps []RoutingSnapshot,
	thetaD EDPPCoeffs, bDec int, kv, sPf int64,
	chunk int, nChunks float64, apP, chunkP int, nChunksP float64,
	tAdmD, tAdmP, decodeJoinUs float64, sPfPrefill int64,
) (local, disagg varPathBreakdown) {
	rt := d.varReTimingFor(req, thetaD, bDec, kv, sPf, chunk)
	if d.cfg.VarExactPrefillOverlap {
		apLocal := d.apForInstance(req, decSnap.ID)
		rt.exactPrefillOverlap = true
		rt.cPf = thetaD.CPf
		rt.ap = float64(maxInt(apLocal, 0))
		rt.ar = float64(int(req.InputLen()))
	}
	decode := d.varDecodeInputs(decSnap.RunningDecode)
	kernel := d.varMetric

	localAdmissionSteps := math.Ceil(tAdmD / math.Max(rt.tIter0, 1))
	local.decode = varDecodeLocalAfter(nowUs, decode, rt, nChunks, localAdmissionSteps, kernel)

	arrivalSteps := math.Ceil(decodeJoinUs / math.Max(rt.tIter0, 1))
	disagg.decode = varDecodeDisagg(nowUs, decode, rt, arrivalSteps, kernel)

	// Collocated prefill occupants on the DECODE instance (placed by a prior collocate decision,
	// still pre-first-token so RunningDecode skips them): local co-schedules R's prefill and
	// delays their first token; disagg leaves those finishing before R arrives untouched. Same
	// re-timing as decode co-residents, TTFT-side flip. Deployable (remaining prompt tokens).
	if d.varCollocPrefill && len(decSnap.RunningPrefill) > 0 {
		colloc := d.varPrefillInputs(decSnap.RunningPrefill)
		local.collocPrefill = varCollocPrefillLocalAfter(nowUs, colloc, rt, float64(chunk), nChunks, localAdmissionSteps, kernel)
		disagg.collocPrefill = varCollocPrefillDisagg(nowUs, colloc, rt, float64(chunk), arrivalSteps, kernel)
	}

	// Prefill-side externality of disagg: R's remote prefill delays the prefill-pool occupants.
	// R's added prefill duration on the pool = nChunks·tIterPrefill + Wp (compute injected; the
	// transfer window is not pool contention). Aggregate occupants across all prefill snapshots.
	if len(prefillSnaps) > 0 {
		tIterP := d.coeffs.tIterPrefill(sPfPrefill)
		if d.cfg.VarExactPrefillOverlap && len(prefillSnaps) == 1 {
			// The reduced rule does not select a prefill node. In the evaluated
			// 1P topology the sole pool member is nevertheless known exactly,
			// so use its observable prefix-cache state.
			apP = d.apForInstance(req, prefillSnaps[0].ID)
			chunkP = maxInt(apP, 1)
			if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunkP {
				chunkP = d.cfg.ChunkTokens
			}
			nChunksP = 0
			if apP > 0 {
				nChunksP = math.Ceil(float64(apP) / float64(chunkP))
			}
		}
		rPrefillUs := nChunksP*tIterP + d.coeffs.Wp(maxInt(apP, 0), int(req.InputLen()))
		chunkPFloat := float64(chunkP)
		prefillAdmissionSteps := math.Ceil(tAdmP / math.Max(tIterP, 1))
		var prefill []varPrefillCoResident
		for _, ps := range prefillSnaps {
			prefill = append(prefill, d.varPrefillInputs(ps.RunningPrefill)...)
		}
		if d.cfg.VarExactPrefillOverlap {
			disagg.prefillPool = varPrefillDisaggExactAfter(
				nowUs,
				prefill,
				tIterP,
				chunkPFloat,
				prefillAdmissionSteps,
				maxInt(apP, 0),
				int(req.InputLen()),
				d.coeffs,
				kernel,
			)
		} else {
			disagg.prefillPool = varPrefillDisaggAfter(nowUs, prefill, tIterP, chunkPFloat, prefillAdmissionSteps, rPrefillUs, kernel)
		}
	}

	return local, disagg
}

// varReducedLHS is the reduced-rule value-currency benefit:
// VaR_local − VaR_disagg.
func (d *EDPPDecider) varReducedLHS(
	req *Request, nowUs float64,
	decSnap RoutingSnapshot, prefillSnaps []RoutingSnapshot,
	thetaD EDPPCoeffs, bDec int, kv, sPf int64,
	chunk int, nChunks float64, apP, chunkP int, nChunksP float64,
	tAdmD, tAdmP, decodeJoinUs float64, sPfPrefill int64,
) float64 {
	local, disagg := d.varReducedBreakdown(
		req, nowUs, decSnap, prefillSnaps, thetaD, bDec, kv, sPf,
		chunk, nChunks, apP, chunkP, nChunksP,
		tAdmD, tAdmP, decodeJoinUs, sPfPrefill,
	)
	return local.total() - disagg.total()
}

// varJointCandidateExternality computes the value-currency externality term for ONE joint
// candidate, replacing the work-currency backlog contribution (jDecodeBacklog + qd·(Wp/W*_d)
// locally, or + qp·(Wp/W*_p) for disagg). For a local candidate (ps == nil) it is VaR_local
// on ds's decode co-residents; for a disagg candidate it is VaR_disagg on ds's decode
// co-residents plus the prefill-side VaR on *ps's occupants. The decode node physics use ds's
// θ_i; the prefill physics use *ps's θ_i. Mirrors jointCandidateCost's operands (INV-6).
func (d *EDPPDecider) varJointCandidateExternality(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot, decodeAdmissionUs, prefillAdmissionUs float64,
) float64 {
	return d.varJointCandidateBreakdown(req, nowUs, ds, ps, decodeAdmissionUs, prefillAdmissionUs).total()
}

// varJointCandidateBreakdown is the diagnostic-preserving form of
// varJointCandidateExternality. Keeping the three causal populations separate lets
// candidate tracing attribute router-induced VaR without changing the scalar used by
// the routing objective.
func (d *EDPPDecider) varJointCandidateBreakdown(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot, decodeAdmissionUs, prefillAdmissionUs float64,
) varPathBreakdown {
	return d.varJointCandidateBreakdownWithKernel(
		req, nowUs, ds, ps, decodeAdmissionUs, prefillAdmissionUs, d.varMetric,
	)
}

func (d *EDPPDecider) varJointCandidateBreakdownWithKernel(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot,
	decodeAdmissionUs, prefillAdmissionUs float64, kernel varKernel,
) varPathBreakdown {
	return d.varJointCandidateBreakdownCore(
		req, nowUs, ds, ps, decodeAdmissionUs, prefillAdmissionUs,
		math.NaN(), kernel,
	)
}

// varJointCandidateBreakdownAtJoinWithKernel is the remote-path form used by
// the causal policy after applying the scheduler-rollout TTFT model. The caller
// supplies the already composed concurrent decode-join time max(P+xfer,TadmD),
// avoiding reconstruction with the obsolete serial closed form.
func (d *EDPPDecider) varJointCandidateBreakdownAtJoinWithKernel(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot,
	decodeJoinUs, prefillAdmissionUs float64, kernel varKernel,
) varPathBreakdown {
	return d.varJointCandidateBreakdownCore(
		req, nowUs, ds, ps, 0, prefillAdmissionUs, decodeJoinUs, kernel,
	)
}

func (d *EDPPDecider) varJointCandidateBreakdownCore(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot,
	decodeAdmissionUs, prefillAdmissionUs, decodeJoinOverrideUs float64, kernel varKernel,
) varPathBreakdown {
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := d.varDecodeInputs(ds.RunningDecode)

	if ps == nil {
		apLoc := d.apForInstance(req, ds.ID)
		chunkLoc := apLoc
		if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunkLoc {
			chunkLoc = d.cfg.ChunkTokens
		}
		rt := d.varReTimingFor(req, thetaD, bDec, kv, sPfD, chunkLoc)
		if d.cfg.VarExactPrefillOverlap {
			rt.exactPrefillOverlap = true
			rt.cPf = thetaD.CPf
			rt.ap = float64(maxInt(apLoc, 0))
			rt.ar = float64(int(req.InputLen()))
		}
		nChunksLoc, _ := d.chunkTerms(thetaD, apLoc)
		admissionSteps := math.Ceil(decodeAdmissionUs / math.Max(rt.tIter0, 1))
		v := varPathBreakdown{decode: varDecodeLocalAfter(nowUs, decode, rt, nChunksLoc, admissionSteps, kernel)}
		if d.varCollocPrefill && len(ds.RunningPrefill) > 0 {
			colloc := d.varPrefillInputs(ds.RunningPrefill)
			v.collocPrefill = varCollocPrefillLocalAfter(nowUs, colloc, rt, float64(chunkLoc), nChunksLoc, admissionSteps, kernel)
		}
		return v
	}

	// Disagg: decode co-residents on ds (tail re-timed after R arrives from the prefill node),
	// plus prefill-pool co-residents on *ps delayed by R's remote prefill.
	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(req, ps.ID)
	chunkP := apP
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunkP {
		chunkP = d.cfg.ChunkTokens
	}
	// Δkv_R / decode re-timing still use ds's θ (decode happens on ds in both placements).
	rt := d.varReTimingFor(req, thetaD, bDec, kv, sPfD, chunkP)
	if d.cfg.VarExactPrefillOverlap {
		rt.exactPrefillOverlap = true
		rt.cPf = thetaD.CPf
		rt.ap = float64(maxInt(apP, 0))
		rt.ar = float64(int(req.InputLen()))
	}
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)
	cXferUs := d.cXferUsFor(req)
	// Decode interference starts when R is admitted, not when its first token
	// completes. Keep this join-time clock separate from client-visible TTFT.
	decodeJoinUs := nChunksP*tIterP +
		thetaP.Wp(maxInt(apP, 0), int(req.InputLen())) +
		prefillAdmissionUs + cXferUs + decodeAdmissionUs
	if !math.IsNaN(decodeJoinOverrideUs) {
		decodeJoinUs = decodeJoinOverrideUs
	}
	arrivalSteps := math.Ceil(decodeJoinUs / math.Max(rt.tIter0, 1))
	v := varPathBreakdown{decode: varDecodeDisagg(nowUs, decode, rt, arrivalSteps, kernel)}

	// Collocated prefill occupants on the decode node ds are undisturbed until R arrives from the
	// pool; only those still prefilling past arrivalSteps have their first token delayed.
	if d.varCollocPrefill && len(ds.RunningPrefill) > 0 {
		colloc := d.varPrefillInputs(ds.RunningPrefill)
		v.collocPrefill = varCollocPrefillDisagg(nowUs, colloc, rt, float64(chunkP), arrivalSteps, kernel)
	}

	rPrefillUs := nChunksP*tIterP + thetaP.Wp(maxInt(apP, 0), int(req.InputLen()))
	prefill := d.varPrefillInputs(ps.RunningPrefill)
	prefillAdmissionSteps := math.Ceil(prefillAdmissionUs / math.Max(tIterP, 1))
	if d.cfg.VarExactPrefillOverlap {
		v.prefillPool = varPrefillDisaggExactAfter(
			nowUs,
			prefill,
			tIterP,
			float64(chunkP),
			prefillAdmissionSteps,
			maxInt(apP, 0),
			int(req.InputLen()),
			thetaP,
			kernel,
		)
	} else {
		v.prefillPool = varPrefillDisaggAfter(nowUs, prefill, tIterP, float64(chunkP), prefillAdmissionSteps, rPrefillUs, kernel)
	}
	return v
}

// goodSelf scores the ARRIVING request R's OWN smoothed goodput under a candidate placement. It is
// the reward half of the goodput-objective diagnostic (VarGoodputObjective). The rule then charges
// VaR − good: the goodput this placement DESTROYS among co-residents minus the goodput it EARNS for
// R, which is −Δgood, the negative per-slot goodput change the drift-plus-penalty rule minimizes.
//
// R arrives at the decision instant, so its TTFT measured from arrival equals tHatFromNow, its
// projected time-to-first-token. R then decodes nOut steps, each at tIterAfter (the B+1 re-timed
// decode per-iter time it experiences once it joins the batch), so its mean inter-token time is
// tIterAfter and its end-to-end latency from arrival is tHatFromNow + nOut·tIterAfter. nOut is R's
// decode length — the censored N̂_out, or R's TRUE output length under the --edpp-oracle-output-len
// upper-bound flag (an INV-9 violation confined to the diagnostic).
//
// The kernels mirror the co-resident good indicators. flip is the hard composite of gDecodeFlip /
// gCollocFlip (1 iff TTFT, ITL, and E2E all meet target, a zero threshold disabling its conjunct).
// util is the slack-utility product of gCollocUtil (first-token × end-to-end, no ITL factor, matching
// the co-resident util kernel). hazard has no natural absolute-good level, so its reward half reuses
// the util product. Pure (INV-6): identical inputs yield identical output.
func goodSelf(slo varSLO, tHatFromNow, tIterAfter, nOut float64, kernel varKernel) float64 {
	ttft := tHatFromNow
	e2e := tHatFromNow + nOut*tIterAfter
	meanITL := tIterAfter
	switch kernel {
	case varKernelComposite:
		return sloCompositeValue(slo, ttft, e2e)
	case varKernelUtil, varKernelHazard:
		scale := slo.tauTTFTUs
		if scale <= 0 {
			scale = 1
		}
		u := 1.0
		if slo.tauTTFTUs > 0 {
			u *= sigmoid((slo.tauTTFTUs - ttft) / scale)
		}
		if slo.tauE2EUs > 0 {
			u *= sigmoid((slo.tauE2EUs - e2e) / scale)
		}
		return u
	default: // varKernelFlip
		if slo.tauTTFTUs > 0 && ttft > slo.tauTTFTUs {
			return 0
		}
		if slo.tauITLUs > 0 && meanITL > slo.tauITLUs {
			return 0
		}
		if slo.tauE2EUs > 0 && e2e > slo.tauE2EUs {
			return 0
		}
		return 1
	}
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
