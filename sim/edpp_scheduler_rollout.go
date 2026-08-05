package sim

import "math"

// schedulerRolloutReq is a mutable copy used by the deployable scheduler-step
// replay. outputRemaining counts only future decode grants. Prefill progress is
// represented exactly by prompt-computed, because token-budget contention can
// make an actual prefill grant smaller than the nominal chunk cap. Combining
// the two into one grant count would let extra prefill steps incorrectly consume
// the predicted output lifetime. Output lengths come only from censored
// per-class running means.
type schedulerRolloutReq struct {
	id              string
	prompt          int64
	computed        int64
	kvBlocks        int64
	outputRemaining int64
	target          bool
}

type schedulerGrant struct {
	req     *schedulerRolloutReq
	before  int64
	grant   int64
	prefill bool
}

type schedulerRolloutResult struct {
	admissionUs  float64
	firstTokenUs float64
	admitted     bool
	firstToken   bool
}

func ceilBlocks(tokens, blockSize int64) int64 {
	if tokens <= 0 {
		return 0
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	return (tokens + blockSize - 1) / blockSize
}

func schedulerGrantTime(grants []schedulerGrant, theta EDPPCoeffs, alpha float64) float64 {
	t := alpha
	for _, item := range grants {
		if item.prefill {
			grant := float64(item.grant)
			before := float64(item.before)
			t += theta.CPf*grant + theta.CAttn*grant*(before+grant/2)
		} else {
			t += theta.C0 + theta.C1*float64(item.before)
		}
	}
	return math.Max(t, 0)
}

func currentScheduledTime(states []SchedulerReqState, theta EDPPCoeffs, alpha float64) float64 {
	grants := make([]schedulerGrant, 0, len(states))
	for i := range states {
		state := states[i]
		if state.ScheduledTokens <= 0 {
			continue
		}
		req := &schedulerRolloutReq{}
		grants = append(grants, schedulerGrant{
			req: req, before: state.ComputedTokens,
			grant:   state.ScheduledTokens,
			prefill: state.ComputedTokens < state.PromptTokens,
		})
	}
	if len(grants) == 0 {
		return 0
	}
	return schedulerGrantTime(grants, theta, alpha)
}

func schedulerRollout(ctx schedulerRolloutContext) schedulerRolloutResult {
	result := schedulerRolloutResult{}
	elapsed := math.Max(
		currentScheduledTime(ctx.currentScheduled, ctx.theta, ctx.alpha)-
			math.Max(ctx.nowUs-float64(ctx.currentStepStartUs), 0),
		0,
	)
	running := append([]*schedulerRolloutReq(nil), ctx.running...)
	waiting := append([]*schedulerRolloutReq(nil), ctx.waiting...)
	waiting = append(waiting, ctx.target)
	freeKV := max(ctx.freeKVBlocks, int64(0))

	for step := 0; step < ctx.maxSteps; step++ {
		budget := ctx.tokenBudget
		grants := make([]schedulerGrant, 0, len(running)+len(waiting))
		preempted := make([]*schedulerRolloutReq, 0)

		for index := 0; index < len(running) && budget > 0; index++ {
			req := running[index]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1)
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = min(demand, ctx.prefillChunkCap)
				}
			}
			grant := min(demand, budget)
			if grant <= 0 {
				continue
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := max(newBlocks-req.kvBlocks, int64(0))
			canSchedule := true
			for deltaBlocks > freeKV {
				if len(running) == 0 {
					canSchedule = false
					break
				}
				victim := running[len(running)-1]
				running = running[:len(running)-1]
				freeKV += victim.kvBlocks
				victim.kvBlocks = 0
				resumeTokens := victim.prompt
				if victim.computed >= victim.prompt {
					resumeTokens = victim.computed + 1
				}
				victim.prompt = resumeTokens
				victim.computed = 0
				preempted = append([]*schedulerRolloutReq{victim}, preempted...)
				if victim == req {
					canSchedule = false
					break
				}
			}
			if !canSchedule {
				break
			}
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			grants = append(grants, schedulerGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		if len(preempted) > 0 {
			waiting = append(preempted, waiting...)
		}
		for len(preempted) == 0 && len(waiting) > 0 && budget > 0 && len(running) < ctx.maxBatch {
			req := waiting[0]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1)
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = min(demand, ctx.prefillChunkCap)
				}
			}
			grant := min(demand, budget)
			if grant <= 0 {
				break
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := max(newBlocks-req.kvBlocks, int64(0))
			if deltaBlocks > freeKV {
				break
			}
			if req.target {
				result.admissionUs = elapsed
				result.admitted = true
			}
			waiting = waiting[1:]
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			running = append(running, req)
			grants = append(grants, schedulerGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		if len(grants) == 0 && len(preempted) > 0 {
			continue
		}
		if len(grants) == 0 {
			return result
		}

		targetFirstToken := false
		for _, item := range grants {
			if item.req.target && ((!item.prefill) || item.before+item.grant >= item.req.prompt) {
				targetFirstToken = true
			}
		}
		elapsed += schedulerGrantTime(grants, ctx.theta, ctx.alpha)

		kept := running[:0]
		for _, req := range running {
			var grant *schedulerGrant
			for i := range grants {
				if grants[i].req == req {
					grant = &grants[i]
					break
				}
			}
			if grant == nil {
				kept = append(kept, req)
				continue
			}
			req.computed = grant.before + grant.grant
			if !grant.prefill {
				req.outputRemaining--
			}
			if grant.prefill || req.outputRemaining > 0 {
				kept = append(kept, req)
			} else {
				freeKV += req.kvBlocks
			}
		}
		running = kept

		if targetFirstToken {
			result.firstTokenUs = elapsed
			result.firstToken = true
			return result
		}
	}
	return result
}

type schedulerRolloutContext struct {
	running, waiting   []*schedulerRolloutReq
	target             *schedulerRolloutReq
	currentScheduled   []SchedulerReqState
	currentStepStartUs int64
	nowUs              float64
	freeKVBlocks       int64
	tokenBudget        int64
	prefillChunkCap    int64
	blockSize          int64
	maxBatch           int
	maxSteps           int
	theta              EDPPCoeffs
	alpha              float64
}

func (d *EDPPDecider) schedulerReqForRollout(state SchedulerReqState, chunkCap int64) *schedulerRolloutReq {
	computed := max(state.ComputedTokens, int64(0))
	prompt := max(state.PromptTokens, int64(0))
	outputEstimate := d.nHatFor(state.SLOClass).mean()
	decodeDone := max(computed-prompt, int64(0))
	totalOutput := math.Max(outputEstimate, float64(decodeDone))
	remainingOutput := max(int64(math.Ceil(totalOutput))-decodeDone, int64(1))
	return &schedulerRolloutReq{
		id: state.ID, prompt: prompt, computed: computed,
		kvBlocks:        max(state.KVBlocks, int64(0)),
		outputRemaining: remainingOutput,
	}
}

// schedulerRolloutTimes applies the paper's scheduler-step rollout to one
// candidate instance. cachedTokens is the target's known prefix on that
// instance. decodeOnly models a transferred request: its prompt is already
// computed, but its KV blocks still have to fit at the candidate decoder.
func (d *EDPPDecider) schedulerRolloutTimes(req *Request, snap RoutingSnapshot, theta EDPPCoeffs, cachedTokens int, decodeOnly, prefillPool bool, nHatOut float64, nowUs float64) (schedulerRolloutResult, bool) {
	if !snap.SchedulerStateObserved || snap.MaxScheduledTokens <= 0 || snap.MaxBatchSize <= 0 {
		return schedulerRolloutResult{}, false
	}
	blockSize := snap.BlockSizeTokens
	if blockSize <= 0 {
		blockSize = int64(max(d.cfg.BlockSize, 1))
	}
	chunkCap := snap.MaxScheduledTokens
	if snap.LongPrefillTokenThreshold > 0 {
		chunkCap = min(chunkCap, snap.LongPrefillTokenThreshold)
	}
	running := make([]*schedulerRolloutReq, 0, len(snap.SchedulerRunning))
	for _, state := range snap.SchedulerRunning {
		running = append(running, d.schedulerReqForRollout(state, chunkCap))
	}
	waiting := make([]*schedulerRolloutReq, 0, len(snap.SchedulerWaiting))
	for _, state := range snap.SchedulerWaiting {
		waiting = append(waiting, d.schedulerReqForRollout(state, chunkCap))
	}
	prompt := req.InputLen()
	computed := int64(max(cachedTokens, 0))
	targetKV := ceilBlocks(computed, blockSize)
	if decodeOnly {
		computed = prompt
		targetKV = 0
	}
	target := &schedulerRolloutReq{
		id: req.ID, prompt: prompt, computed: computed, kvBlocks: targetKV,
		outputRemaining: max(int64(math.Ceil(math.Max(nHatOut, 1))), int64(1)),
		target:          true,
	}
	alpha := theta.AlphaD
	if prefillPool {
		alpha = theta.AlphaP
	}
	result := schedulerRollout(schedulerRolloutContext{
		running: running, waiting: waiting, target: target,
		currentScheduled: snap.CurrentScheduled, currentStepStartUs: snap.CurrentStepStartUs,
		nowUs: nowUs, freeKVBlocks: snap.FreeKVBlocks,
		tokenBudget: snap.MaxScheduledTokens, prefillChunkCap: chunkCap,
		blockSize: blockSize, maxBatch: int(snap.MaxBatchSize), maxSteps: 100000,
		theta: theta, alpha: alpha,
	})
	return result, result.admitted
}

func (d *EDPPDecider) rolloutLocalTTFT(ec *jointEvalCtx, ds RoutingSnapshot, theta EDPPCoeffs) (tAdm, ttft float64, ok bool) {
	cached := int(ec.req.InputLen()) - max(d.apForInstance(ec.req, ds.ID), 0)
	result, ok := d.schedulerRolloutTimes(ec.req, ds, theta, cached, false, false, ec.nHatOut, ec.nowUs)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs + d.outputTokenProcessingUs(), true
}

func (d *EDPPDecider) rolloutDecodeAdmission(ec *jointEvalCtx, ds RoutingSnapshot, theta EDPPCoeffs) (float64, bool) {
	result, ok := d.schedulerRolloutTimes(ec.req, ds, theta, int(ec.req.InputLen()), true, false, ec.nHatOut, ec.nowUs)
	if !ok {
		return 0, false
	}
	return result.admissionUs, true
}

func (d *EDPPDecider) rolloutPrefillCompletion(ec *jointEvalCtx, ps RoutingSnapshot, theta EDPPCoeffs) (tAdm, completion float64, ok bool) {
	cached := int(ec.req.InputLen()) - max(d.apForInstance(ec.req, ps.ID), 0)
	result, ok := d.schedulerRolloutTimes(ec.req, ps, theta, cached, false, true, ec.nHatOut, ec.nowUs)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs, true
}
