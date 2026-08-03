package sim

import "math"

// Kairos load-aware prefill deflection for disaggregated LLM serving
// ("Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving",
// arXiv:2607.02043).
//
// This file deliberately exposes two policy identities:
//
//   - kairos-paper follows the printed decision rule: discrete chunk candidates,
//     alpha=1.3 by default, the arriving request's TTFT-SLO gate, and no extra
//     decode-admission or KV-transfer term in the printed TTFT equations.
//   - kairos-adapted preserves the study's earlier continuous chunk relaxation,
//     admission-aware decode estimate, transfer-aware prefill estimate, and
//     deployable queue approximation. "kairos" is a compatibility alias for it.
//
// Both modes use this simulator's trained-physics coefficients. Paper mode also
// uses the strictest TBT target among current decode residents when SLO classes
// differ, rather than borrowing the arriving request's class.

// The paper profiles this discrete set of chunk sizes and Algorithm 1 consumes a
// descending candidate list. Candidates above the engine's token cap are removed.
var kairosPaperChunkCandidates = []float64{2048, 1024, 512, 256, 128}

// kairosStepPrefill is the prefill-node step time for a chunk of chi tokens
// attending over a resident context of k tokens.
func kairosStepPrefill(c EDPPCoeffs, chi, k float64) float64 {
	if chi <= 0 {
		return c.AlphaP
	}
	return c.AlphaP + c.CPf*chi + c.CAttn*chi*(k+chi/2)
}

// kairosMaxSafeChunk is the continuous relaxation retained for kairos-adapted.
func kairosMaxSafeChunk(c EDPPCoeffs, base, ctx, tbt float64) float64 {
	slack := tbt - base
	if slack <= 0 {
		return 0
	}
	a := c.CAttn / 2
	b := c.CPf + c.CAttn*ctx
	if a <= 0 {
		if b <= 0 {
			return 0
		}
		return slack / b
	}
	disc := b*b + 4*a*slack
	if disc <= 0 {
		return 0
	}
	return (-b + math.Sqrt(disc)) / (2 * a)
}

func kairosContinuousDeflectTTFT(c EDPPCoeffs, bDec int, kv int64, tokens, tbt, chunkCap float64, maxSteps int) (float64, []float64, bool) {
	if tokens <= 0 {
		return 0, nil, true
	}
	var elapsed, done float64
	schedule := make([]float64, 0, int(math.Ceil(tokens/math.Max(chunkCap, 1))))
	for step := 0; step < maxSteps && done < tokens; step++ {
		base := c.tIterDecode(bDec, kv+int64(done), 0)
		chi := kairosMaxSafeChunk(c, base, done, tbt)
		if chi <= 0 {
			return 0, nil, false
		}
		if chunkCap > 0 && chi > chunkCap {
			chi = chunkCap
		}
		if chi > tokens-done {
			chi = tokens - done
		}
		elapsed += base + c.CPf*chi + c.CAttn*chi*(done+chi/2)
		done += chi
		schedule = append(schedule, chi)
	}
	if done < tokens {
		return 0, nil, false
	}
	return elapsed, schedule, true
}

// kairosDeflectTTFT retains the original helper signature for callers and tests.
func kairosDeflectTTFT(c EDPPCoeffs, bDec int, kv int64, tokens, tbt, chunkCap float64, maxSteps int) (float64, bool) {
	ttft, _, ok := kairosContinuousDeflectTTFT(c, bDec, kv, tokens, tbt, chunkCap, maxSteps)
	return ttft, ok
}

func kairosDiscreteCandidates(chunkCap float64) []float64 {
	out := make([]float64, 0, len(kairosPaperChunkCandidates))
	for _, candidate := range kairosPaperChunkCandidates {
		if chunkCap <= 0 || candidate <= chunkCap {
			out = append(out, candidate)
		}
	}
	// Engines configured below the paper's smallest profiled point still need a
	// physically executable candidate. This fallback is explicit and deterministic.
	if len(out) == 0 && chunkCap > 0 {
		out = append(out, chunkCap)
	}
	return out
}

// kairosDiscreteDeflectTTFT implements Algorithm 1's greedy LargestSafe search.
// The final chunk is shortened to the remaining prompt tokens, as the executor
// cannot process padding beyond the request's prompt.
func kairosDiscreteDeflectTTFT(c EDPPCoeffs, bDec int, kv int64, tokens, tbt float64, candidates []float64, maxSteps int) (float64, []float64, bool) {
	if tokens <= 0 {
		return 0, nil, true
	}
	if len(candidates) == 0 {
		return 0, nil, false
	}
	var elapsed, done float64
	schedule := make([]float64, 0, int(math.Ceil(tokens/candidates[len(candidates)-1])))
	for step := 0; step < maxSteps && done < tokens; step++ {
		base := c.tIterDecode(bDec, kv+int64(done), 0)
		remaining := tokens - done
		chosen := 0.0
		for _, candidate := range candidates {
			chi := math.Min(candidate, remaining)
			stepTime := base + c.CPf*chi + c.CAttn*chi*(done+chi/2)
			if stepTime <= tbt {
				chosen = chi
				break
			}
		}
		if chosen <= 0 {
			return 0, nil, false
		}
		elapsed += base + c.CPf*chosen + c.CAttn*chosen*(done+chosen/2)
		done += chosen
		schedule = append(schedule, chosen)
	}
	if done < tokens {
		return 0, nil, false
	}
	return elapsed, schedule, true
}

func kairosScheduleSummary(schedule []float64) (first, minimum float64, steps int) {
	if len(schedule) == 0 {
		return 0, 0, 0
	}
	first, minimum = schedule[0], schedule[0]
	for _, chi := range schedule[1:] {
		if chi < minimum {
			minimum = chi
		}
	}
	return first, minimum, len(schedule)
}

func kairosExecutableSchedule(schedule []float64) []int {
	if len(schedule) == 0 {
		return nil
	}
	out := make([]int, len(schedule))
	for i, chi := range schedule {
		out[i] = int(math.Round(chi))
	}
	return out
}

// kairosResidentTBTTarget returns the strictest TBT target of the decode
// residents whose latency the safety constraint protects. The arriving class is
// used only when the node has no decode residents (or a hand-built snapshot omits
// resident detail).
func (d *EDPPDecider) kairosResidentTBTTarget(ds RoutingSnapshot, arrivingClass string) float64 {
	if len(ds.RunningDecode) == 0 {
		_, tau := d.targetsFor(arrivingClass)
		return float64(tau)
	}
	strictest := math.Inf(1)
	for _, resident := range ds.RunningDecode {
		_, tau := d.targetsFor(resident.SLOClass)
		if float64(tau) < strictest {
			strictest = float64(tau)
		}
	}
	return strictest
}

func (d *EDPPDecider) kairosAdaptedPrefillTTFT(req *Request, chunkCap float64) (float64, string) {
	if d.prefillSnapshots == nil {
		return math.Inf(1), ""
	}
	snaps := sortedSnapshotsByID(d.prefillSnapshots())
	bestTTFT := math.Inf(1)
	bestPrefillID := ""
	for _, ps := range snaps {
		theta := d.coeffsFor(ps.GPUType)
		tokens := float64(maxInt(d.apForInstance(req, ps.ID), 0))
		chi := chunkCap
		if chi <= 0 || chi > tokens {
			chi = tokens
		}
		if chi <= 0 {
			continue
		}
		sumL := float64(ps.ResidentPrefillTokens) + float64(ps.QueueDepth)*float64(len(req.InputTokens))
		queueWait := 0.0
		if sumL > 0 {
			ctxQ := math.Min(sumL/2, float64(len(req.InputTokens)))
			queueWait = (sumL / chi) * kairosStepPrefill(theta, chi, ctxQ)
		}
		exec := 0.0
		for done := 0.0; done < tokens; done += chi {
			stepChi := math.Min(chi, tokens-done)
			exec += kairosStepPrefill(theta, stepChi, done)
		}
		ttft := queueWait + exec + d.cXferUsFor(req)
		// snaps is ID-sorted and strict comparison preserves deterministic ID
		// tie-breaking when two prefill paths have equal predicted TTFT.
		if ttft < bestTTFT {
			bestTTFT, bestPrefillID = ttft, ps.ID
		}
	}
	return bestTTFT, bestPrefillID
}

func (d *EDPPDecider) kairosPaperPrefillTTFT(req *Request, chunkCap float64) (float64, string) {
	if d.prefillSnapshots == nil {
		return math.Inf(1), ""
	}
	snaps := sortedSnapshotsByID(d.prefillSnapshots())
	// Kairos does not account for prefix-cache residency; Algorithm 1 takes the
	// request's full prompt length as input.
	tokens := float64(len(req.InputTokens))
	if tokens <= 0 {
		return math.Inf(1), ""
	}
	bestTTFT := math.Inf(1)
	bestPrefillID := ""
	// The published algorithm has one prefill path. In a multi-prefill topology,
	// apply the same printed estimator to every eligible path and choose the
	// minimum; the returned hint makes execution use the path that was scored.
	for _, ps := range snaps {
		theta := d.coeffsFor(ps.GPUType)
		chi := chunkCap
		if chi <= 0 || chi > tokens {
			chi = tokens
		}
		// The snapshot exposes the actual remaining prompt-token total, avoiding the
		// previous QueueDepth×current-prompt approximation. Equation 2's context is
		// intentionally left literal rather than capped.
		sumL := float64(ps.PrefillTokensAhead)
		queueWait := 0.0
		if sumL > 0 {
			queueWait = (sumL / chi) * kairosStepPrefill(theta, chi, sumL/2)
		}
		exec := 0.0
		for done := 0.0; done < tokens; done += chi {
			stepChi := math.Min(chi, tokens-done)
			exec += kairosStepPrefill(theta, stepChi, done)
		}
		ttft := queueWait + exec
		if ttft < bestTTFT {
			bestTTFT, bestPrefillID = ttft, ps.ID
		}
	}
	// Printed Equation 1 is queue wait + prefill execution. The adapted mode
	// separately retains the evaluation extension that adds KV-transfer time.
	return bestTTFT, bestPrefillID
}

func (d *EDPPDecider) kairosTrace(req *Request, mode string, alpha, tauTTFT, tauITL, ttftPrefill, bestTTFT, residentTau, tbtBudget float64, schedule []float64, gateRequired, disaggregate bool, skip string) *EDPPDecisionTrace {
	if !d.cfg.TraceEnabled {
		return nil
	}
	first, minimum, steps := kairosScheduleSummary(schedule)
	return &EDPPDecisionTrace{
		Class: req.SLOClass, SkipReason: skip, Ap: len(req.InputTokens),
		TauTTFT: tauTTFT, TauITL: tauITL, TTFTP: ttftPrefill, TTFTD: bestTTFT,
		KairosMode: mode, KairosAlpha: alpha, KairosAlphaThreshold: alpha * ttftPrefill,
		KairosTTFTGateRequired: gateRequired, KairosTTFTGatePassed: bestTTFT <= tauTTFT,
		KairosResidentTauITL: residentTau, KairosTBTBudget: tbtBudget,
		KairosFirstChunk: first, KairosMinChunk: minimum, KairosChunkSteps: steps,
		LHS: alpha * ttftPrefill, RHS: bestTTFT, Disaggregate: disaggregate,
	}
}

// decideKairos selects the explicitly requested Kairos identity. The unqualified
// historical name remains an adapted-mode alias so existing scripts do not
// silently acquire a different algorithm.
func (d *EDPPDecider) decideKairos(req *Request, state *RouterState) DisaggregationDecision {
	// A request can carry an executable paper-mode chunk schedule. Clear stale
	// policy metadata before evaluating a fresh routing decision.
	req.PrefillChunkSchedule = nil
	req.resetPrefillChunkSchedule()
	if d.rule == "kairos-paper" {
		return d.decideKairosPaper(req, state)
	}
	return d.decideKairosAdapted(req, state)
}

func (d *EDPPDecider) decideKairosAdapted(req *Request, state *RouterState) DisaggregationDecision {
	keepLocal := DisaggregationDecision{Disaggregate: false}
	tauTTFTUs, tauITLUs := d.targetsFor(req.SLOClass)
	if len(req.InputTokens) == 0 {
		keepLocal.EDPPTrace = d.kairosTrace(req, "adapted", 1, float64(tauTTFTUs), float64(tauITLUs), 0, 0, 0, 0, nil, false, false, "empty-prompt")
		return keepLocal
	}
	tbt := d.kairosBeta * float64(tauITLUs)
	chunkCap := float64(d.cfg.ChunkTokens)
	const maxSteps = 4096
	ttftPrefill, prefillID := d.kairosAdaptedPrefillTTFT(req, chunkCap)

	reqKVNeed := d.reqKVNeed(req)
	bestTTFT := ttftPrefill
	bestDecode := ""
	var bestSchedule []float64
	for _, ds := range sortedSnapshotsByID(stateSnapshots(state)) {
		if ds.ResidentPrefillTokens > 0 {
			continue
		}
		theta := d.coeffsFor(ds.GPUType)
		tokens := float64(maxInt(d.apForInstance(req, ds.ID), 0))
		t, schedule, ok := kairosContinuousDeflectTTFT(theta, ds.BatchSize, ds.KvTokensInUse, tokens, tbt, chunkCap, maxSteps)
		if !ok {
			continue
		}
		_, qd := d.instWorkRaw(ds.ID)
		tAdm := d.tadmEstimator.EstimateTAdm(AdmissionContext{
			QWork: qd, Mu: theta.muDecode(ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens),
			BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
			FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: reqKVNeed,
			TIter:      theta.tIterDecode(ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens),
			QueueDepth: ds.QueueDepth, AdmissionRate: admissionRateFromSnapshot(ds),
			RemainingStepsEst: d.decodeRemStepsEst(ds, req.SLOClass),
			Running:           censorOracleRemaining(ds.RunningDecode),
		})
		if t+tAdm < bestTTFT {
			bestTTFT, bestDecode, bestSchedule = t+tAdm, ds.ID, schedule
		}
	}
	if bestDecode != "" {
		dec := DisaggregationDecision{Disaggregate: false, DecodePodOverride: bestDecode}
		dec.EDPPTrace = d.kairosTrace(req, "adapted", 1, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, float64(tauITLUs), tbt, bestSchedule, false, true, "")
		return dec
	}
	if prefillID == "" || math.IsInf(ttftPrefill, 1) {
		keepLocal.EDPPTrace = d.kairosTrace(req, "adapted", 1, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, float64(tauITLUs), tbt, nil, false, false, "no-prefill-path")
		return keepLocal
	}
	dec := DisaggregationDecision{Disaggregate: true, PrefillPodHint: prefillID}
	dec.EDPPTrace = d.kairosTrace(req, "adapted", 1, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, float64(tauITLUs), tbt, nil, false, false, "")
	return dec
}

func (d *EDPPDecider) decideKairosPaper(req *Request, state *RouterState) DisaggregationDecision {
	keepLocal := DisaggregationDecision{Disaggregate: false}
	tauTTFTUs, tauITLUs := d.targetsFor(req.SLOClass)
	if len(req.InputTokens) == 0 {
		keepLocal.EDPPTrace = d.kairosTrace(req, "paper", d.kairosAlpha, float64(tauTTFTUs), float64(tauITLUs), 0, 0, 0, 0, nil, true, false, "empty-prompt")
		return keepLocal
	}
	chunkCap := float64(d.cfg.ChunkTokens)
	candidates := kairosDiscreteCandidates(chunkCap)
	const maxSteps = 4096
	ttftPrefill, prefillID := d.kairosPaperPrefillTTFT(req, chunkCap)

	bestTTFT := math.Inf(1)
	bestDecode := ""
	bestResidentTau := 0.0
	bestTBTBudget := 0.0
	var bestSchedule []float64
	for _, ds := range sortedSnapshotsByID(stateSnapshots(state)) {
		// At most one deflected prefill may be in flight on a decode node.
		if ds.ResidentPrefillTokens > 0 || ds.PrefillTokensAhead > 0 {
			continue
		}
		residentTau := d.kairosResidentTBTTarget(ds, req.SLOClass)
		tbtBudget := d.kairosBeta * residentTau
		theta := d.coeffsFor(ds.GPUType)
		t, schedule, ok := kairosDiscreteDeflectTTFT(theta, ds.BatchSize, ds.KvTokensInUse, float64(len(req.InputTokens)), tbtBudget, candidates, maxSteps)
		if !ok {
			continue
		}
		if t < bestTTFT {
			bestTTFT, bestDecode = t, ds.ID
			bestResidentTau, bestTBTBudget, bestSchedule = residentTau, tbtBudget, schedule
		}
	}

	marginPassed := bestDecode != "" && bestTTFT <= d.kairosAlpha*ttftPrefill
	gatePassed := bestTTFT <= float64(tauTTFTUs)
	if marginPassed && gatePassed {
		req.PrefillChunkSchedule = kairosExecutableSchedule(bestSchedule)
		dec := DisaggregationDecision{Disaggregate: false, DecodePodOverride: bestDecode}
		dec.EDPPTrace = d.kairosTrace(req, "paper", d.kairosAlpha, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, bestResidentTau, bestTBTBudget, bestSchedule, true, true, "")
		return dec
	}
	if prefillID == "" || math.IsInf(ttftPrefill, 1) {
		keepLocal.EDPPTrace = d.kairosTrace(req, "paper", d.kairosAlpha, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, bestResidentTau, bestTBTBudget, bestSchedule, true, false, "no-prefill-path")
		return keepLocal
	}
	dec := DisaggregationDecision{Disaggregate: true, PrefillPodHint: prefillID}
	dec.EDPPTrace = d.kairosTrace(req, "paper", d.kairosAlpha, float64(tauTTFTUs), float64(tauITLUs), ttftPrefill, bestTTFT, bestResidentTau, bestTBTBudget, bestSchedule, true, false, "")
	return dec
}
