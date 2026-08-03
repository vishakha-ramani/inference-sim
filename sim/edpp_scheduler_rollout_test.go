package sim

import "testing"

func rolloutTestContext(targetPrompt int64) schedulerRolloutContext {
	theta := EDPPCoeffs{AlphaD: 10, AlphaP: 10, CPf: 1}
	return schedulerRolloutContext{
		target: &schedulerRolloutReq{
			id: "target", prompt: targetPrompt,
			outputRemaining: 1, target: true,
		},
		currentScheduled: []SchedulerReqState{{
			PromptTokens: 1, ComputedTokens: 1, ScheduledTokens: 1,
		}},
		currentStepStartUs: 0, nowUs: 4,
		freeKVBlocks: 1000, tokenBudget: 4, prefillChunkCap: 4,
		blockSize: 16, maxBatch: 16, maxSteps: 100,
		theta: theta, alpha: theta.AlphaD,
	}
}

func TestSchedulerRolloutPredictsAdmissionAndFinalPrefillStep(t *testing.T) {
	result := schedulerRollout(rolloutTestContext(8))
	if !result.admitted || result.admissionUs != 6 {
		t.Fatalf("admission = (%v,%v), want (true,6)", result.admitted, result.admissionUs)
	}
	if !result.firstToken || result.firstTokenUs != 34 {
		t.Fatalf("first token = (%v,%v), want (true,34)", result.firstToken, result.firstTokenUs)
	}
}

func TestSchedulerRolloutDoesNotSkipFIFOQueueWithFreeSlots(t *testing.T) {
	ctx := rolloutTestContext(1)
	ahead := &schedulerRolloutReq{id: "ahead", prompt: 8, outputRemaining: 1}
	ctx.waiting = []*schedulerRolloutReq{ahead}
	result := schedulerRollout(ctx)
	if !result.admitted || result.admissionUs != 34 {
		t.Fatalf("target admission = (%v,%v), want (true,34) after two ahead chunks", result.admitted, result.admissionUs)
	}
}

func TestSchedulerRolloutPrefillContentionDoesNotConsumeDecodeLifetime(t *testing.T) {
	ctx := rolloutTestContext(1)
	ctx.currentScheduled = nil
	ctx.nowUs = 0
	ctx.maxBatch = 1
	ctx.prefillChunkCap = 8
	ctx.tokenBudget = 4
	ctx.running = []*schedulerRolloutReq{{
		id: "resident", prompt: 8, outputRemaining: 2,
	}}

	result := schedulerRollout(ctx)
	// The resident needs two 4-token prefill steps (14us each) and two
	// decode steps (10us each) before the target can be admitted.
	if !result.admitted || result.admissionUs != 48 {
		t.Fatalf("target admission = (%v,%v), want (true,48)", result.admitted, result.admissionUs)
	}
}

func TestSchedulerRolloutRecomputesAttentionPerChunk(t *testing.T) {
	ctx := rolloutTestContext(8)
	ctx.theta.CAttn = 0.1
	result := schedulerRollout(ctx)
	// Residual current iteration is 6. Target chunks cost
	// 10+4+0.1*4*(0+2)=14.8 and 10+4+0.1*4*(4+2)=16.4.
	want := 37.2
	if diff := result.firstTokenUs - want; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("first token = %v, want %v", result.firstTokenUs, want)
	}
}

func TestRolloutLocalTTFTUsesCandidateSchedulerState(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.ChunkTokens = 4
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
	req := makeReq("target", 8, "standard")
	ec := &jointEvalCtx{req: req, nHatOut: 1, nowUs: 4}
	snap := RoutingSnapshot{
		ID: "d0", MaxBatchSize: 16, FreeKVBlocks: 1000,
		SchedulerStateObserved: true, MaxScheduledTokens: 4,
		BlockSizeTokens: 16, CurrentStepStartUs: 0,
		CurrentScheduled: []SchedulerReqState{{
			PromptTokens: 1, ComputedTokens: 1, ScheduledTokens: 1,
		}},
	}
	tAdm, ttft, ok := d.rolloutLocalTTFT(ec, snap, EDPPCoeffs{AlphaD: 10, AlphaP: 10, CPf: 1})
	if !ok || tAdm != 6 || ttft != 34 {
		t.Fatalf("local rollout = (adm=%v, ttft=%v, ok=%v), want (6,34,true)", tAdm, ttft, ok)
	}
}
