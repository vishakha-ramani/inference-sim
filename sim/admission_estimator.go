package sim

import (
	"fmt"
	"math"
	"sort"
)

// RunningReqState is one running decode request's state for the roll-forward
// estimator. TrueRemaining is the oracle remaining step count (-1 when the
// oracle is not populated); StepsDone is decode steps completed; KVBlocks held.
//
// The SLOClass/ArrivalUs/FirstTokenUs/TTFTSet fields are DEPLOYABLE per-co-resident
// SLO-deadline inputs (input-derived, INV-9-safe) read by the value-at-risk drift
// oracle (sim/edpp_var.go): they let g() evaluate a co-resident's composite-good
// (TTFT met, ITL, E2E ≤ deadline) at a projected completion. They are zero-valued
// unless admission detail is enabled, and the admission-delay estimators
// (waiting/little/fluid/rollforward) never read them, so populating them is
// byte-identical for every existing path (INV-6).
type RunningReqState struct {
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64

	SLOClass     string // co-resident SLO class (drives its τ_ttft/τ_itl/τ_e2e resolution); "" = default
	ArrivalUs    int64  // co-resident arrival instant (µs); E2E deadline = ArrivalUs + τ_e2e(class)
	FirstTokenUs int64  // realized first-token instant (µs); 0 when TTFTSet is false
	TTFTSet      bool   // true once the co-resident has produced its first token (realized TTFT is fixed)

	// OracleOutputLen is the total output length of a still-prefilling co-resident, populated only
	// in oracle mode (len(OutputTokens)), -1 otherwise. The value-at-risk oracle uses it to project
	// a collocated occupant's decode phase (its ITL/E2E risk once the deciding request joins its
	// batch). The deployable rule ignores it and uses the censored per-class N̂_out instead
	// (INV-9-safe). Decode co-residents carry their remaining output in TrueRemaining and leave this -1.
	OracleOutputLen int64
}

// SchedulerReqState is the deployable request state needed to replay future
// scheduler steps for admission and first-token prediction. It contains no
// output length: the EDPP decider supplies its censored per-class running mean.
type SchedulerReqState struct {
	ID              string
	SLOClass        string
	PromptTokens    int64
	ComputedTokens  int64
	ScheduledTokens int64
	KVBlocks        int64
	Priority        float64
	ArrivalUs       int64
}

// AdmissionContext bundles everything an admission-delay estimator may read for
// one pool. The EDPPDecider assembles it from its backlog/rate state and the
// (possibly enriched) selected snapshot. Times/work in µs.
type AdmissionContext struct {
	QWork             float64 // waiting-backlog work (µs)
	Mu                float64 // occupancy-aware drain rate
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64   // KV blocks this request needs
	TIter             float64 // occupancy-aware per-iteration time (µs)
	QueueDepth        int
	AdmissionRate     float64           // req/µs admitted at this pool (for little)
	RemainingStepsEst float64           // mean estimated remaining decode steps (N̂_out-based, for fluid)
	Running           []RunningReqState // per-running-request state (for rollforward)
}

// AdmissionDelayEstimator predicts the admission delay (µs) a request would incur
// at a pool given its current state. Pure function of ctx (INV-6).
type AdmissionDelayEstimator interface {
	EstimateTAdm(ctx AdmissionContext) float64
	Name() string
}

// flooredTAdm lower-bounds an admission-delay estimate by one iteration: even with a
// free slot, a request waits for the current decode step to finish before the next
// FormBatch admits it (~T_iter). No-op when TIter is unavailable (<=0) or already exceeded.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
	if ctx.TIter > est {
		return ctx.TIter
	}
	return est
}

type waitingEstimator struct{}

func (waitingEstimator) Name() string { return "waiting" }
func (waitingEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.Mu <= 0 {
		return 0
	}
	return ctx.QWork / ctx.Mu
}

type littleEstimator struct{}

func (littleEstimator) Name() string { return "little" }
func (littleEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return flooredTAdm(0, ctx)
	}
	return flooredTAdm(float64(ctx.QueueDepth)/ctx.AdmissionRate, ctx)
}

type fluidEstimator struct{}

func (fluidEstimator) Name() string { return "fluid" }
func (fluidEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	if ctx.BatchSize <= 0 || ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 {
		return flooredTAdm(0, ctx)
	}
	// Synchronized batch: occupants finish ~R̄ steps together, so slots free in WAVES of
	// BatchSize every ~R̄ iterations. A request at queue position QueueDepth waits
	// ⌈(QueueDepth+1)/BatchSize⌉ waves. (Not the naive fluid-drain /BatchSize.)
	waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
	return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
}

type rollforwardEstimator struct{}

func (rollforwardEstimator) Name() string { return "rollforward" }
func (rollforwardEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	// Deterministic look-ahead: each running req departs after its remaining steps
	// (oracle TrueRemaining if ≥0, else the N̂_out estimate), freeing its KV. Accumulate
	// elapsed = departureStep·T_iter until a slot AND enough free KV exist.
	type dep struct{ step, kv int64 }
	deps := make([]dep, 0, len(ctx.Running))
	for _, r := range ctx.Running {
		rem := r.TrueRemaining
		if rem < 0 {
			rem = int64(ctx.RemainingStepsEst)
			if rem < 1 {
				rem = 1
			}
		}
		deps = append(deps, dep{step: rem, kv: r.KVBlocks})
	}
	// Sort by departure step ascending (soonest first); stable tie-break for determinism (INV-6).
	sort.SliceStable(deps, func(i, j int) bool { return deps[i].step < deps[j].step })
	// The request sits at queue position QueueDepth: the QueueDepth requests ahead fill
	// the first QueueDepth freed slots, ours takes the next — so we need QueueDepth+1
	// slots to free (plus our KV) before we admit.
	needSlots := int64(ctx.QueueDepth + 1)
	freeSlots := int64(ctx.MaxBatchSize - ctx.BatchSize)
	freeKV := ctx.FreeKVBlocks
	for _, d := range deps {
		freeSlots++
		freeKV += d.kv
		if freeSlots >= needSlots && freeKV >= ctx.ReqKVNeed {
			return flooredTAdm(float64(d.step)*ctx.TIter, ctx)
		}
	}
	// The current running set's departures were exhausted before freeing QueueDepth+1
	// slots (queue deeper than one batch-drain). Fall back to the fluid wave form for
	// the deep tail: slots free in waves of BatchSize every ~RemainingStepsEst iters.
	if ctx.BatchSize > 0 {
		waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
		return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
	}
	// BatchSize<=0: no wave cadence available — cap at the last known departure.
	if len(deps) > 0 {
		return flooredTAdm(float64(deps[len(deps)-1].step)*ctx.TIter, ctx)
	}
	return flooredTAdm(0, ctx)
}

// NewAdmissionEstimator returns the estimator by name. Little/fluid/rollforward
// and the oracle variants are added in later tasks.
func NewAdmissionEstimator(name string) (AdmissionDelayEstimator, error) {
	switch name {
	case "", "waiting":
		return waitingEstimator{}, nil
	case "little":
		return littleEstimator{}, nil
	case "fluid":
		return fluidEstimator{}, nil
	case "rollforward":
		return rollforwardEstimator{}, nil
	case "fluid_oracle":
		// Oracle variant: same impl, run against a true-remaining-populated context
		// (Task 3 oracle mode). Logging-only — rejected as a routing driver by INV-9.
		return fluidEstimator{}, nil
	case "rollforward_oracle":
		return rollforwardEstimator{}, nil
	default:
		return nil, fmt.Errorf("unknown admission estimator %q", name)
	}
}

// IsDeployableEstimator reports whether an estimator name may drive routing. The
// deployable estimators read only production-observable state; the oracle variants
// (and unknown names) read TRUE remaining output and so are logging-only (INV-9).
func IsDeployableEstimator(name string) bool {
	switch name {
	case "", "waiting", "little", "fluid", "rollforward":
		return true
	default:
		return false // oracle variants and unknowns
	}
}
