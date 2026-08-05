// sim/simulator.go
package sim

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

const MaxTokenID = 128000 // Max token ID in request input/output

// Compile-time guard: MaxTokenID must fit in TokenID's underlying int32 range.
// If MaxTokenID is ever raised above math.MaxInt32 (~2.1B), the int→TokenID
// cast in GenerateRandomTokenIDs would silently truncate. This expression
// evaluates to a negative integer constant when MaxTokenID > MaxInt32, which
// is not representable as uint and fails to compile.
const _ = uint(math.MaxInt32 - MaxTokenID)

// eventEntry wraps an Event with a sequence ID for deterministic ordering.
// The EventQueue orders by (Timestamp, Priority, seqID), matching the
// cluster event queue's scheme. seqID breaks ties within same-type same-timestamp events.
type eventEntry struct {
	event Event
	seqID int64
}

// EventQueue implements heap.Interface and orders events by (timestamp, priority, seqID).
// This ensures deterministic same-tick event ordering (INV-6 improvement).
type EventQueue []eventEntry

func (eq EventQueue) Len() int { return len(eq) }

func (eq EventQueue) Less(i, j int) bool {
	if eq[i].event.Timestamp() != eq[j].event.Timestamp() {
		return eq[i].event.Timestamp() < eq[j].event.Timestamp()
	}
	if eq[i].event.Priority() != eq[j].event.Priority() {
		return eq[i].event.Priority() < eq[j].event.Priority()
	}
	return eq[i].seqID < eq[j].seqID
}

func (eq EventQueue) Swap(i, j int) { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x any) {
	*eq = append(*eq, x.(eventEntry))
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	*eq = old[:n-1]
	return item
}

// SimConfig holds all configuration for creating a Simulator.
// Sub-configs are embedded so fields are accessible via promotion
// (e.g., cfg.TotalKVBlocks resolves to cfg.KVCacheConfig.TotalKVBlocks).
type SimConfig struct {
	// Simulation control (no sub-config — no factory uses only these)
	Horizon int64
	Seed    int64

	// Module-scoped sub-configs (R16)
	KVCacheConfig
	BatchConfig
	LatencyCoeffs
	ModelHardwareConfig
	PolicyConfig
	WorkloadConfig
	// LoRAConfig is the 7th module sub-config (LoRA control-plane subsystem). Its
	// zero value is inert: unset => the subsystem is a no-op and output is
	// byte-identical to a pre-feature build (INV-6). See sim/lora.
	LoRAConfig

	// SLO priority overrides for preemption victim selection (--preemption-policy priority).
	// nil = use GAIE defaults (critical=4, standard=3, batch=-1, sheddable=-2, background=-3).
	// Shared with admission: same overrides flow from policy bundle slo_priorities.
	// Set programmatically in cmd/root.go and cmd/replay.go from parsed bundle/CLI overrides — no YAML tag needed.
	SLOPriorityOverrides map[string]int
}

// Simulator is the core object that holds simulation time, system state, and the event loop.
type Simulator struct {
	Clock   int64
	Horizon int64
	// eventQueue has all the simulator events, like arrival and step events
	eventQueue EventQueue
	// WaitQ aka request waiting queue before it is scheduled
	WaitQ   *WaitQueue
	KVCache KVStore
	// Running batch contains the set of requests that go into the model for execution per Step.
	// In vLLM, running is a list (not queue) of requests, hence we don't call it RunningQ here.
	// Requests are ordered by First-Come-First-Served in WaitQ, and the same order is maintained
	// while adding requests to RunningBatch
	RunningBatch *Batch
	Metrics      *Metrics
	// max number of requests RunningBatch can hold
	maxRunningReqs int64
	// max total number of new tokens across all requests in RunningBatch
	maxScheduledTokens        int64
	longPrefillTokenThreshold int64
	stepEvent                 Event
	stepCount                 int
	// map of request IDs to total num computed tokens (including cached tokens)
	reqNumComputedTokens map[string]int64
	batchFormation       BatchFormation
	model                string
	gpu                  string
	maxModelLen          int64           // max total sequence length (0 = unlimited)
	rng                  *PartitionedRNG // partitioned RNG for deterministic multi-subsystem simulation
	sloMap               *SLOPriorityMap // vLLM-convention priority mapping for instance-level scheduling
	scheduler            InstanceScheduler
	latencyModel         LatencyModel
	// residentAdapters tracks this instance's finite resident LoRA adapter slots
	// (capacity-bounded LRU). nil when the LoRA subsystem is inert (no adapters /
	// capacity configured, or sim/lora not imported), in which case adapter handling
	// is a no-op and output is byte-identical to a pre-feature build (INV-6).
	residentAdapters ResidentAdapterSet
	// adapterCost derives the cold-load latency charged by the pre-admission gate
	// (#1466). Non-nil exactly when residentAdapters is (both wired together from
	// the same sim/lora registration); nil ⇒ no gating.
	adapterCost AdapterCost
	// loadingAdapter is the id of the adapter whose load is currently in flight on
	// this instance, or "" when none. Loads serialize per instance: the gate starts
	// a new load only when this is "" (§7 serialization).
	loadingAdapter string
	seqCounter     int64 // monotonic counter for event queue seqID (deterministic ordering)
	// OnRequestDone is an optional callback invoked when a request reaches a terminal
	// state (completed, length-capped, or timed out). Returns follow-up requests to inject.
	// Set by the caller (cmd/root.go or ClusterSimulator). Nil = no callback.
	OnRequestDone func(req *Request, tick int64) []*Request
	// OnAdmit is an optional callback invoked once when a request first transitions
	// from the wait queue into the running batch (i.e. for each BatchResult.NewlyScheduled
	// entry). Used by control-plane deciders that track a *waiting* backlog and must
	// remove a request's work when it is admitted (EDPP waiting-only backlog, design §6.2).
	// nil ⇒ no-op. Fired inside the deterministic event loop (INV-6 safe).
	// Note: a preempted-then-readmitted request fires OnAdmit again on re-admission, so consumers must be idempotent.
	OnAdmit func(req *Request, tick int64)

	// OnFirstToken is an optional callback invoked once when a request produces its first
	// token (prefill completes: ProgressIndex reaches len(InputTokens)), with the absolute
	// sim tick. Used by SLO-feedback deciders to true up a TTFT virtual queue from the
	// realized first-token time instead of waiting for full completion. nil ⇒ no-op. Fired
	// inside the deterministic event loop (INV-6 safe). A preempted-then-re-prefilled
	// request fires again (TTFTSet reset on preemption), so consumers must be idempotent.
	OnFirstToken func(req *Request, tick int64)

	// OnFirstDecodeToken is an optional callback invoked once when a request
	// completes its first decode step, with the absolute simulation tick at
	// which the output token is produced. Unlike OnFirstToken, this observes
	// the actual decode execution boundary and therefore includes any wait
	// between prefill/transfer completion and decode admission. nil ⇒ no-op.
	OnFirstDecodeToken func(req *Request, tick int64)

	progressHook               ProgressHook
	simClockProgressIntervalUs int64
	nextSnapshotClockUs        int64

	// stepRec, when non-nil (enabled via BLIS_STEP_CSV), streams per-step E3
	// latency-law regressors to a CSV for offline coefficient calibration.
	// nil by default — no effect on the deterministic result channel.
	stepRec *stepRecorder

	// Work-trace accumulator (off unless --edpp-work-trace). Sums each resident
	// request's per-step δ (active latency-model basis) into per-request totals,
	// so realized trajectory work can be compared to the closed-form W_p/W_d.
	recordWorkTrace bool
	workCoeffs      EDPPCoeffs
	workAcc         map[string]*reqWorkAccum

	// Admission-detail gate (off by default, zero-cost). When recordAdmissionDetail
	// is set, RunningDecodeState()/RemainingDecodeWork() walk the running batch to
	// populate the occupancy-aware AdmissionContext fields for fluid/rollforward
	// estimators. admissionDetailOracle additionally populates RunningReqState.TrueRemaining
	// from OutputTokens (measurement-only; INV-9 keeps this out of the deployable path).
	recordAdmissionDetail bool
	admissionDetailOracle bool
	currentScheduled      []SchedulerReqState
	currentStepStartUs    int64

	// onAdmitInternal is an internal admission observer fired for each newly
	// scheduled request alongside the public OnAdmit callback. Unlike OnAdmit
	// (caller-settable, wired only for SLO-feedback deciders), this hook is owned
	// by the wrapping InstanceSimulator so the windowed admission-rate counter is
	// always fed when admission detail is enabled. nil ⇒ no-op.
	onAdmitInternal func(req *Request, tick int64)
}

// SetOnAdmitInternal registers an internal admission observer that fires for each
// newly scheduled request. Distinct from the public OnAdmit callback so both can
// coexist. Used by InstanceSimulator to feed the windowed admission-rate counter.
func (sim *Simulator) SetOnAdmitInternal(fn func(req *Request, tick int64)) {
	sim.onAdmitInternal = fn
}

// SetAdmissionDetail enables population of the occupancy-aware admission-detail
// snapshot fields (per-running-decode-request state and aggregate remaining decode
// work). When oracle is true, RunningReqState.TrueRemaining is filled from
// OutputTokens (measurement only — never on the deployable control path, INV-9);
// otherwise it is left at -1. Off by default (zero-cost).
func (sim *Simulator) SetAdmissionDetail(oracle bool) {
	sim.recordAdmissionDetail = true
	sim.admissionDetailOracle = oracle
}

// AdmissionDetailEnabled reports whether admission-detail population is enabled
// (see SetAdmissionDetail). Used by the cluster snapshot-build path to decide
// whether to pay for extra occupancy signals (e.g. per-instance DispatchRate).
func (sim *Simulator) AdmissionDetailEnabled() bool { return sim.recordAdmissionDetail }

// RunningDecodeState returns per-running-decode-request state for the roll-forward
// admission estimator. Returns nil when admission detail is disabled (zero-cost
// default) or there is no running batch. StepsDone = ProgressIndex − len(InputTokens);
// KVBlocks ≈ ⌈ProgressIndex / blockSize⌉; TrueRemaining = len(OutputTokens) − StepsDone
// only in oracle mode, else −1.
func (sim *Simulator) RunningDecodeState() []RunningReqState {
	if !sim.recordAdmissionDetail || sim.RunningBatch == nil {
		return nil
	}
	blockSize := int64(1)
	if sim.KVCache != nil && sim.KVCache.BlockSize() > 0 {
		blockSize = sim.KVCache.BlockSize()
	}
	var out []RunningReqState
	for _, req := range sim.RunningBatch.Requests {
		inLen := int64(len(req.InputTokens))
		if req.ProgressIndex < inLen {
			continue // still in prefill phase — not a decode request
		}
		stepsDone := req.ProgressIndex - inLen
		trueRemaining := int64(-1)
		if sim.admissionDetailOracle {
			trueRemaining = int64(len(req.OutputTokens)) - stepsDone
		}
		firstTokenUs := req.FirstTokenTimestamp
		ttftSet := req.TTFTSet || firstTokenUs > 0
		// Compatibility for requests constructed before the explicit absolute
		// timestamp field: FirstTokenTime is an elapsed duration.
		if firstTokenUs == 0 && req.TTFTSet {
			firstTokenUs = req.ArrivalTime + req.FirstTokenTime
		}
		out = append(out, RunningReqState{
			StepsDone:     stepsDone,
			KVBlocks:      (req.ProgressIndex + blockSize - 1) / blockSize,
			TrueRemaining: trueRemaining,
			// Deployable per-co-resident SLO-deadline inputs (INV-9-safe) for the VaR oracle.
			SLOClass:        req.SLOClass,
			ArrivalUs:       req.ArrivalTime,
			FirstTokenUs:    firstTokenUs,
			TTFTSet:         ttftSet,
			OracleOutputLen: -1, // decode co-residents carry remaining output in TrueRemaining
		})
	}
	return out
}

// RunningPrefillState returns per-running-prefill-request state for the prefill-pool
// admission estimators (ttft_p). Returns nil when admission detail is disabled
// (zero-cost default) or there is no running batch. It is the prefill-phase mirror of
// RunningDecodeState: only requests still in prefill (ProgressIndex < len(InputTokens))
// are reported. StepsDone = prefill chunks (tokens) already processed = ProgressIndex;
// KVBlocks ≈ ⌈ProgressIndex / blockSize⌉; TrueRemaining = remaining prefill tokens
// (len(InputTokens) − ProgressIndex), populated unconditionally (INV-9 asymmetry:
// prefill remaining is known input, deployable — not oracle-gated like decode).
func (sim *Simulator) RunningPrefillState() []RunningReqState {
	if !sim.recordAdmissionDetail || sim.RunningBatch == nil {
		return nil
	}
	blockSize := int64(1)
	if sim.KVCache != nil && sim.KVCache.BlockSize() > 0 {
		blockSize = sim.KVCache.BlockSize()
	}
	var out []RunningReqState
	for _, req := range sim.RunningBatch.Requests {
		inLen := int64(len(req.InputTokens))
		if req.ProgressIndex >= inLen {
			continue // past prefill — a decode request, not a prefill occupant
		}
		// INV-9 asymmetry: prefill remaining = inLen − ProgressIndex = remaining prompt
		// tokens, which is KNOWN at routing (input length is known). Unlike decode's
		// remaining (depends on hidden o_r), this is deployable-legitimate, so it is NOT
		// oracle-gated: populated whenever admission detail is on, and never censored.
		trueRemaining := inLen - req.ProgressIndex
		// Oracle-only: the occupant's total output length, so the VaR oracle can project its
		// decode phase (the ITL/E2E risk once R joins its batch). -1 when the oracle is off; the
		// deployable rule uses the censored per-class N̂_out instead and never reads this.
		oracleOutputLen := int64(-1)
		if sim.admissionDetailOracle {
			oracleOutputLen = int64(len(req.OutputTokens))
		}
		out = append(out, RunningReqState{
			StepsDone:     req.ProgressIndex,
			KVBlocks:      (req.ProgressIndex + blockSize - 1) / blockSize,
			TrueRemaining: trueRemaining,
			// Deployable per-co-resident SLO-deadline inputs (INV-9-safe) for the VaR oracle.
			// A prefill occupant has not produced its first token (TTFTSet=false); its VaR
			// flip is TTFT-side, keyed on ArrivalUs + τ_ttft.
			SLOClass:        req.SLOClass,
			ArrivalUs:       req.ArrivalTime,
			OracleOutputLen: oracleOutputLen,
		})
	}
	return out
}

func (sim *Simulator) schedulerReqState(req *Request, includeCachedPrefix bool) SchedulerReqState {
	computed := req.ProgressIndex
	if includeCachedPrefix && computed < int64(len(req.InputTokens)) && sim.KVCache != nil {
		cached := int64(len(sim.KVCache.GetCachedBlocks(req.InputTokens))) * sim.KVCache.BlockSize()
		if cached > computed {
			computed = cached
		}
	}
	blockSize := int64(1)
	if sim.KVCache != nil && sim.KVCache.BlockSize() > 0 {
		blockSize = sim.KVCache.BlockSize()
	}
	return SchedulerReqState{
		ID: req.ID, SLOClass: req.SLOClass,
		PromptTokens: int64(len(req.InputTokens)), ComputedTokens: computed,
		KVBlocks: (computed + blockSize - 1) / blockSize,
		Priority: req.Priority, ArrivalUs: req.ArrivalTime,
	}
}

// SchedulerRolloutState returns ordered, deployable scheduler state for the
// paper's admission/TTFT rollout. The running state is post-current-step; the
// separately captured CurrentScheduled state describes the in-flight step used
// only to predict its remaining duration.
func (sim *Simulator) SchedulerRolloutState() (running, waiting, current []SchedulerReqState, currentStart, tokenBudget, longPrefill, blockSize int64, observed bool) {
	if !sim.recordAdmissionDetail {
		return nil, nil, nil, 0, 0, 0, 0, false
	}
	if sim.RunningBatch != nil {
		running = make([]SchedulerReqState, 0, len(sim.RunningBatch.Requests))
		for _, req := range sim.RunningBatch.Requests {
			running = append(running, sim.schedulerReqState(req, false))
		}
	}
	if sim.WaitQ != nil {
		waiting = make([]SchedulerReqState, 0, sim.WaitQ.Len())
		for _, req := range sim.WaitQ.Items() {
			waiting = append(waiting, sim.schedulerReqState(req, true))
		}
	}
	current = append([]SchedulerReqState(nil), sim.currentScheduled...)
	blockSize = 1
	if sim.KVCache != nil && sim.KVCache.BlockSize() > 0 {
		blockSize = sim.KVCache.BlockSize()
	}
	return running, waiting, current, sim.currentStepStartUs,
		sim.maxScheduledTokens, sim.longPrefillTokenThreshold, blockSize, true
}

// NewSimulator creates a Simulator from a SimConfig struct and pre-built dependencies.
// All workload generation now happens externally — callers inject requests via InjectArrival.
func NewSimulator(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*Simulator, error) {
	if kvStore == nil {
		return nil, fmt.Errorf("NewSimulator: kvStore must not be nil")
	}
	if latencyModel == nil {
		return nil, fmt.Errorf("NewSimulator: latencyModel must not be nil")
	}
	if cfg.MaxRunningReqs <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxRunningReqs must be > 0, got %d", cfg.MaxRunningReqs)
	}
	if cfg.MaxScheduledTokens <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxScheduledTokens must be > 0, got %d", cfg.MaxScheduledTokens)
	}
	if cfg.LongPrefillTokenThreshold < 0 {
		return nil, fmt.Errorf("NewSimulator: LongPrefillTokenThreshold must be >= 0, got %d", cfg.LongPrefillTokenThreshold)
	}
	if cfg.MaxModelLen < 0 {
		return nil, fmt.Errorf("NewSimulator: MaxModelLen must be >= 0, got %d", cfg.MaxModelLen)
	}
	if cfg.MaxModelLen > 0 {
		if cfg.BlockSizeTokens <= 0 {
			return nil, fmt.Errorf("NewSimulator: BlockSizeTokens must be > 0 when MaxModelLen is set, got %d", cfg.BlockSizeTokens)
		}
		// Ceiling division for MaxModelLen → block count (R11).
		blocksForMaxLen := cfg.MaxModelLen / cfg.BlockSizeTokens
		if cfg.MaxModelLen%cfg.BlockSizeTokens != 0 {
			blocksForMaxLen++
		}
		if blocksForMaxLen > cfg.TotalKVBlocks {
			return nil, fmt.Errorf("NewSimulator: KV cache too small for MaxModelLen: need %d blocks (ceil(%d/%d)) but TotalKVBlocks=%d",
				blocksForMaxLen, cfg.MaxModelLen, cfg.BlockSizeTokens, cfg.TotalKVBlocks)
		}
	}
	batchFormation := NewBatchFormation(cfg.PreemptionPolicy)

	s := &Simulator{
		Clock:                     0,
		Horizon:                   cfg.Horizon,
		eventQueue:                make(EventQueue, 0),
		WaitQ:                     &WaitQueue{},
		KVCache:                   kvStore,
		RunningBatch:              &Batch{},
		Metrics:                   NewMetrics(),
		maxRunningReqs:            cfg.MaxRunningReqs,
		maxScheduledTokens:        cfg.MaxScheduledTokens,
		longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
		stepEvent:                 nil,
		stepCount:                 0,
		reqNumComputedTokens:      make(map[string]int64),
		batchFormation:            batchFormation,
		model:                     cfg.Model,
		gpu:                       cfg.GPU,
		maxModelLen:               cfg.MaxModelLen,
		latencyModel:              latencyModel,
		sloMap:                    NewSLOPriorityMap(cfg.SLOPriorityOverrides),
	}
	s.rng = NewPartitionedRNG(NewSimulationKey(cfg.Seed))
	s.scheduler = NewScheduler(cfg.Scheduler)
	s.stepRec = newStepRecorderFromEnv()

	// Defense-in-depth: reject a non-positive adapter capacity here rather than
	// letting it reach newResidentSet as a panic. cmd/ validates via LoRAConfig.Validate,
	// but NewSimulator is library code and must return an error, not terminate, for a
	// caller that bypasses the CLI (R6). Mirrors the BatchConfig re-validation above.
	if cfg.HasAdapters() && cfg.AdapterCapacity != nil && *cfg.AdapterCapacity <= 0 {
		return nil, fmt.Errorf("NewSimulator: adapter_capacity must be > 0 when adapters are declared, got %d", *cfg.AdapterCapacity)
	}
	// Wire the per-instance resident-adapter set only when the LoRA subsystem is
	// active — adapters declared with a positive capacity — and sim/lora is linked
	// (NewResidentAdapterSetFunc registered). Otherwise it stays nil and adapter
	// handling is a no-op (INV-6).
	// Wire the resident set AND the cost model together (both from sim/lora's single
	// init, so both registration funcs are non-nil or both nil). Requiring both here
	// guarantees the invariant the gate relies on: whenever residentAdapters != nil,
	// adapterCost != nil too. If only the resident set were wired, FormBatch would
	// gate cold requests (AdapterResident predicate set) but maybeStartAdapterLoad
	// could never start a load (adapterCost nil) — stranding them. A malformed cost
	// config is a library-boundary error (R6), not a panic.
	// BuildAdapterCost centralizes the activation condition (R4) so NewSimulator and
	// the sim/cluster latency backend agree on exactly when adapter costs apply; it
	// returns (nil, nil) when the LoRA subsystem is inert (no adapters, no capacity,
	// or sim/lora unlinked). A non-nil ac therefore stands in for the full
	// HasAdapters && capacity != nil && factories-registered condition.
	ac, err := BuildAdapterCost(cfg)
	if err != nil {
		return nil, fmt.Errorf("NewSimulator: adapter cost model: %w", err)
	}
	if ac != nil {
		// Guard against a factory that returns a nil interface value: the concrete
		// sim/lora set never does (it returns a valid set or panics on bad capacity),
		// but a test double or future implementation might, and a typed-nil interface
		// would slip past the sim.residentAdapters == nil guard and panic on first use.
		rs := NewResidentAdapterSetFunc(*cfg.AdapterCapacity)
		if rs == nil {
			return nil, fmt.Errorf("NewSimulator: NewResidentAdapterSetFunc returned nil for capacity %d", *cfg.AdapterCapacity)
		}
		s.residentAdapters = rs
		s.adapterCost = ac
	} else if cfg.HasAdapters() && cfg.AdapterCapacity == nil {
		// Adapters declared but no capacity: the resident set stays inert and every
		// adapter metric reports zero. Warn rather than fail silently (R1) — a run
		// that completes with zeroed adapter counts is otherwise indistinguishable
		// from a working one.
		logrus.Warnf("adapters declared but adapter_capacity is not set; per-instance resident-adapter tracking is disabled and adapter metrics will be zero")
	} else if cfg.HasAdapters() && cfg.AdapterCapacity != nil && (NewResidentAdapterSetFunc == nil || NewAdapterCostFunc == nil) {
		// Adapters + capacity configured but sim/lora was never linked, so the seam
		// factories are unregistered. Only reachable from a non-CLI caller (cmd/root.go
		// imports sim/lora); warn so that path does not silently drop adapter tracking.
		// This branch is not unit-testable from within package sim: the blank import in
		// lora_import_test.go always registers the factories, so there is no way to
		// observe them nil in a sim test.
		logrus.Warnf("adapters and adapter_capacity are configured but sim/lora is not linked (NewResidentAdapterSetFunc/NewAdapterCostFunc unregistered); resident-adapter tracking is disabled and adapter metrics will be zero")
	}

	return s, nil
}

// WorkloadRNG returns the RNG for workload generation.
// This maintains backward compatibility with the original single-RNG implementation.
func (sim *Simulator) WorkloadRNG() *rand.Rand {
	return sim.rng.ForSubsystem(SubsystemWorkload)
}

// Schedule pushes an event into the simulator's EventQueue with a monotonic seqID.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	sim.seqCounter++
	heap.Push(&sim.eventQueue, eventEntry{event: ev, seqID: sim.seqCounter})
}

// HasPendingEvents returns true if the EventQueue is non-empty.
func (sim *Simulator) HasPendingEvents() bool {
	return len(sim.eventQueue) > 0
}

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
func (sim *Simulator) PeekNextEventTime() int64 {
	return sim.eventQueue[0].event.Timestamp()
}

// ProcessNextEvent pops the earliest event, advances Clock, executes it, and returns it.
// The returned Event lets callers react to what happened (e.g., detect QueuedEvent for
// pending-request tracking) without maintaining fragile before/after heuristics.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
// Does NOT check horizon — caller is responsible.
//
// Special case — lazy cancellation: if the popped event is a TimeoutEvent for a
// request that has already completed (State == StateCompleted), the event is
// returned immediately without advancing Clock or calling Execute(). This models
// real-world client behavior where a deadline timer is cancelled the moment a
// response arrives, preventing orphaned timeouts from inflating SimEndedTime.
func (sim *Simulator) ProcessNextEvent() Event {
	entry := heap.Pop(&sim.eventQueue).(eventEntry)
	ev := entry.event

	// Lazy cancellation: a TimeoutEvent whose request already completed is an
	// orphan — the real-world equivalent of a client cancelling its deadline
	// timer when the response arrives. Skip it before advancing the clock so
	// Finalize() captures SimEndedTime from the last real-work event, not from
	// an orphaned no-op timeout 300s in the future.
	if te, ok := ev.(*TimeoutEvent); ok && te.Request.State == StateCompleted {
		return ev
	}

	sim.Clock = ev.Timestamp()
	logrus.Debugf("[tick %07d] Executing %T", sim.Clock, ev)
	ev.Execute(sim)
	return ev
}

// Finalize records end-of-run state and sets SimEndedTime.
// Call once after the event loop ends. Called by both sim.Run() (single-instance)
// and ClusterSimulator.Run() (cluster mode via inst.Finalize()).
func (sim *Simulator) Finalize() {
	// Record conservation fields (BC-8, BC-9) — must happen in Finalize
	// because cluster mode drives events via ProcessNextEvent() directly
	// and never calls sim.Run().
	sim.Metrics.StillQueued = sim.WaitQ.Len()
	if sim.RunningBatch != nil {
		sim.Metrics.StillRunning = len(sim.RunningBatch.Requests)
	}
	sim.Metrics.SimEndedTime = min(sim.Clock, sim.Horizon)
	sim.stepRec.close() // nil-safe; flushes and closes the calibration CSV
	logrus.Infof("[tick %07d] Simulation ended", sim.Clock)
}

// InjectArrival schedules an ArrivalEvent for req and registers it in Metrics.Requests.
func (sim *Simulator) InjectArrival(req *Request) {
	if req.ArrivalTime > sim.Horizon {
		logrus.Warnf("InjectArrival: request %s has ArrivalTime %d > Horizon %d; "+
			"ArrivalEvent will not fire (INV-1 conservation may be affected)",
			req.ID, req.ArrivalTime, sim.Horizon)
	}
	sim.Schedule(&ArrivalEvent{time: req.ArrivalTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

// InjectArrivalAt schedules an ArrivalEvent at eventTime (not req.ArrivalTime).
// Metrics.Requests uses req.ArrivalTime for ArrivedAt to preserve original arrival time.
// Used by cluster-mode online routing where event time differs from original arrival.
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64) {
	sim.Schedule(&ArrivalEvent{time: eventTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

func (sim *Simulator) Run() {
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		if sim.Clock > sim.Horizon {
			break
		}
		sim.maybeDeliverProgressSnapshot(false)
	}
	sim.maybeDeliverProgressSnapshot(true)
	sim.Finalize()
}

// SetProgressHook registers an optional hook that receives periodic state
// snapshots during simulation execution. Must be called before Run().
// When hook is nil (default), there is zero behavioral or performance impact.
// simClockIntervalUs controls the minimum simulation-clock interval (microseconds)
// between periodic snapshots. If simClockIntervalUs <= 0, only the final snapshot
// is delivered.
func (sim *Simulator) SetProgressHook(hook ProgressHook, simClockIntervalUs int64) {
	sim.progressHook = hook
	if simClockIntervalUs > 0 {
		sim.simClockProgressIntervalUs = simClockIntervalUs
		sim.nextSnapshotClockUs = simClockIntervalUs
	}
}

func (sim *Simulator) maybeDeliverProgressSnapshot(isFinal bool) {
	if sim.progressHook == nil {
		return
	}
	if !isFinal && (sim.simClockProgressIntervalUs <= 0 || sim.Clock < sim.nextSnapshotClockUs) {
		return
	}
	clock := sim.Clock
	if isFinal {
		clock = min(sim.Clock, sim.Horizon)
	}
	snap := ProgressSnapshot{
		Clock:             clock,
		TotalCompleted:    sim.Metrics.CompletedRequests,
		TotalTimedOut:     sim.Metrics.TimedOutRequests,
		TotalDropped:      sim.Metrics.DroppedUnservable,
		TotalInputTokens:  sim.Metrics.TotalInputTokens,
		TotalOutputTokens: sim.Metrics.TotalOutputTokens,
		TotalPreemptions:  sim.Metrics.PreemptionCount,
		InstanceSnapshots: []InstanceSnapshot{sim.buildInstanceSnapshot()},
		TotalInstances:    1,
		ActiveInstances:   1,
		IsFinal:           isFinal,
	}
	sim.progressHook.OnProgress(snap)
	if !isFinal {
		sim.nextSnapshotClockUs += sim.simClockProgressIntervalUs
	}
}

func (sim *Simulator) buildInstanceSnapshot() InstanceSnapshot {
	return InstanceSnapshot{
		ID:                "instance-0",
		Model:             sim.model,
		State:             InstanceStateActive,
		QueueDepth:        sim.QueueDepth(),
		BatchSize:         sim.BatchSize(),
		KVUtilization:     float64(sim.KVCache.UsedBlocks()) / float64(max(sim.KVCache.TotalCapacity(), 1)),
		KVFreeBlocks:      sim.KVCache.TotalCapacity() - sim.KVCache.UsedBlocks(),
		KVTotalBlocks:     sim.KVCache.TotalCapacity(),
		CacheHitRate:      sim.KVCache.CacheHitRate(),
		PreemptionCount:   sim.Metrics.PreemptionCount,
		CompletedRequests: sim.Metrics.CompletedRequests,
		TimedOutRequests:  sim.Metrics.TimedOutRequests,
	}
}

// QueueDepth returns the number of requests in the wait queue.
func (sim *Simulator) QueueDepth() int { return sim.WaitQ.Len() }

// PrefillTokensAhead returns the exact number of prompt tokens still waiting or
// executing on this instance. Unlike QueueDepth, this preserves the queued
// requests' actual prompt lengths, which Kairos's FIFO prefill estimator needs.
// Decode-only requests contribute zero because their ProgressIndex is already at
// or beyond len(InputTokens).
func (sim *Simulator) PrefillTokensAhead() int64 {
	remaining := func(req *Request) int64 {
		if req == nil {
			return 0
		}
		n := req.InputLen() - req.ProgressIndex
		if n < 0 {
			return 0
		}
		return n
	}
	var total int64
	if sim.RunningBatch != nil {
		for _, req := range sim.RunningBatch.Requests {
			total += remaining(req)
		}
	}
	if sim.WaitQ != nil {
		for _, req := range sim.WaitQ.Items() {
			total += remaining(req)
		}
	}
	return total
}

// ResidentAdapterIDs returns the ids of LoRA adapters currently resident on this
// instance, in a deterministic order (INV-6). Returns nil when the LoRA subsystem
// is inert (no adapters/capacity configured, or sim/lora not imported), so the
// snapshot's ResidentAdapters stays nil and the lora-affinity scorer is neutral
// (INV-6). Read by the cluster snapshot provider to populate
// RoutingSnapshot.ResidentAdapters (#1469).
func (sim *Simulator) ResidentAdapterIDs() []string {
	if sim.residentAdapters == nil {
		return nil
	}
	return sim.residentAdapters.ResidentIDs()
}

// DrainWaitQueue removes and returns all requests currently in the wait queue.
// Used by DrainRedirect policy to re-inject queued requests into the cluster router.
// After this call, WaitQ.Len() == 0.
func (sim *Simulator) DrainWaitQueue() []*Request {
	items := sim.WaitQ.Items()
	sim.WaitQ = &WaitQueue{}
	return items
}

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (sim *Simulator) BatchSize() int {
	if sim.RunningBatch == nil {
		return 0
	}
	return len(sim.RunningBatch.Requests)
}

// CurrentClock returns the current simulation clock (in ticks).
func (sim *Simulator) CurrentClock() int64 { return sim.Clock }

// SimHorizon returns the simulation horizon (in ticks).
func (sim *Simulator) SimHorizon() int64 { return sim.Horizon }

// ScheduleStepIfIdle schedules a batch-formation step if the instance is idle
// with pending work. Used by gateway eviction to maintain INV-8 after removing
// the last request from RunningBatch.
func (sim *Simulator) ScheduleStepIfIdle(time int64) {
	if sim.stepEvent == nil && sim.WaitQ.Len() > 0 {
		step := &StepEvent{time: time}
		sim.stepEvent = step
		sim.Schedule(step)
	}
}

// PostDecodeFixedOverhead returns the latency model's fixed per-request post-decode
// overhead in microseconds. Used by the cluster layer to include overhead in
// parent.CompletionTime when disaggregated decode sub-requests complete.
// Returns 0 for all backends except trained-physics (BC-1, issue #846).
func (sim *Simulator) PostDecodeFixedOverhead() int64 {
	return sim.latencyModel.PostDecodeFixedOverhead()
}

// EnqueueRequest adds a newly arrived request to the waiting queue.
//
// Preprocessing: auto-fills MaxOutputLen when the client doesn't set a budget
// (MaxOutputLen == 0) and maxModelLen > 0. Sets MaxOutputLen = maxModelLen - len(InputTokens),
// mirroring vLLM's input_processor.py:554 (max_tokens = max_model_len - seq_len).
// Workload generators normally set MaxOutputLen = len(OutputTokens) (tight budget);
// this auto-fill is a safety net for requests that bypass generators.
//
// Three guards then prevent unservable requests from entering the queue:
//  0. MaxOutputLen validation (R3): drops requests with negative MaxOutputLen.
//  1. MaxModelLen guard (when maxModelLen > 0): validates the request fits within
//     the model's context window. First checks input >= maxModelLen (vLLM uses >=:
//     input filling the entire context leaves no room for output). Then, when
//     MaxOutputLen > 0 (client budget), checks input + budget <= maxModelLen.
//  2. KV capacity guard (defense-in-depth, always active): drops requests whose input
//     tokens alone require more KV blocks than total cache capacity (R19: livelock protection).
//
// All guards mirror real vLLM behavior where oversized requests are rejected
// before entering the engine. The control plane never peeks at len(OutputTokens) —
// respecting the oracle knowledge boundary (INV-9, #567).
func (sim *Simulator) EnqueueRequest(r *Request) {
	// Guard -1: Already timed out (race: TimeoutEvent fired before QueuedEvent).
	// Request was timed out during the queueing delay (alpha overhead) before the server
	// processed the input. TotalInputTokens is NOT counted for this path.
	// INV-1 holds: request is counted in timed_out bucket.
	if r.State == StateTimedOut {
		return
	}

	// Auto-fill: if client didn't set a budget, cap at remaining context window.
	if r.MaxOutputLen == 0 && sim.maxModelLen > 0 && r.InputLen() < sim.maxModelLen {
		r.MaxOutputLen = int(sim.maxModelLen) - int(r.InputLen())
	}

	// Guard 0: Negative MaxOutputLen check (R3)
	if r.MaxOutputLen < 0 {
		logrus.Warnf("dropping request %s: MaxOutputLen %d is negative",
			r.ID, r.MaxOutputLen)
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		// Callback for dropped requests (R1: don't silently discard, BC-17)
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	// Guard 1: MaxModelLen check
	if sim.maxModelLen > 0 {
		if r.InputLen() >= sim.maxModelLen {
			logrus.Warnf("dropping request %s: input length %d >= MaxModelLen %d (no room for output)",
				r.ID, r.InputLen(), sim.maxModelLen)
			sim.Metrics.DroppedUnservable++
			delete(sim.Metrics.Requests, r.ID)
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(r, sim.Clock) {
					sim.InjectArrival(next)
				}
			}
			return
		}
		if r.MaxOutputLen > 0 {
			totalSeqLen := r.InputLen() + int64(r.MaxOutputLen)
			if totalSeqLen > sim.maxModelLen {
				logrus.Warnf("dropping request %s: total sequence length %d (input=%d + budget=%d) exceeds MaxModelLen %d",
					r.ID, totalSeqLen, r.InputLen(), r.MaxOutputLen, sim.maxModelLen)
				sim.Metrics.DroppedUnservable++
				delete(sim.Metrics.Requests, r.ID)
				if sim.OnRequestDone != nil {
					for _, next := range sim.OnRequestDone(r, sim.Clock) {
						sim.InjectArrival(next)
					}
				}
				return
			}
		}
	}

	// Guard 2: KV capacity check (defense-in-depth, always active)
	blocksNeeded := (r.InputLen() + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	// Input tokens counted BEFORE past-due check (request was received)
	sim.Metrics.TotalInputTokens += int(r.InputLen())

	// Past-due guard (EC-2): check BEFORE enqueue to avoid enqueue-then-remove.
	// Request is counted as timed_out, not dropped_unservable.
	if r.Deadline > 0 && r.Deadline <= sim.Clock {
		r.State = StateTimedOut
		sim.Metrics.TimedOutRequests++
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	// Pre-processor: convert cluster-convention SLO priority to vLLM instance convention.
	// Mirrors the llm-d → vLLM dispatch boundary in production (lower = more urgent).
	// Overwrites any routing-hint Priority (which was always transient; see routing.go:59).
	if sim.sloMap == nil {
		// sloMap should always be set by NewSimulator; this path indicates manual struct construction.
		logrus.Warnf("Simulator.sloMap not initialized — using DefaultSLOPriorityMap; prefer NewSimulator()")
		sim.sloMap = DefaultSLOPriorityMap()
	}
	r.Priority = float64(sim.sloMap.InvertForVLLM(r.SLOClass))

	sim.WaitQ.Enqueue(r)

	// Schedule timeout event (after all guards + enqueue — BC-5)
	// Skip scheduling when deadline > horizon (perf: avoids orphaned events)
	if r.Deadline > 0 && r.Deadline <= sim.Horizon {
		sim.Schedule(&TimeoutEvent{time: r.Deadline, Request: r})
	}
}

// EnqueueDecodeSubRequest enqueues a decode sub-request that already has KV blocks
// pre-allocated (via PD disaggregation transfer). Bypasses the oversized-request guard
// (blocks already allocated, guard would leak them) and does NOT increment TotalInputTokens
// (input tokens were already counted by the prefill sub-request).
// clusterTime is the cluster-level clock when this request is injected (from
// KVTransferCompletedEvent.Execute()). The StepEvent is scheduled at
// max(sim.Clock, clusterTime) to prevent the instance from processing the
// decode sub-request at a stale internal time that precedes the request's arrival.
// Triggers StepEvent if the instance is idle (INV-8: work-conserving).
func (sim *Simulator) EnqueueDecodeSubRequest(r *Request, clusterTime int64) {
	// Pre-processor: decode sub-requests inherit SLOClass from parent (pd_events.go:215).
	// Apply the same vLLM-convention priority as EnqueueRequest.
	if sim.sloMap == nil {
		logrus.Warnf("Simulator.sloMap not initialized — using DefaultSLOPriorityMap; prefer NewSimulator()")
		sim.sloMap = DefaultSLOPriorityMap()
	}
	r.Priority = float64(sim.sloMap.InvertForVLLM(r.SLOClass))

	sim.WaitQ.Enqueue(r)
	// Do NOT add len(r.InputTokens) to TotalInputTokens — already counted by prefill sub-request.

	// Schedule timeout for decode sub-request (R23: parity with EnqueueRequest)
	if r.Deadline > 0 && r.Deadline <= sim.Horizon {
		sim.Schedule(&TimeoutEvent{time: r.Deadline, Request: r})
	}

	// Trigger StepEvent if idle (work-conserving: INV-8).
	// Use max(sim.Clock, clusterTime) so the decode sub-request is not processed
	// at a stale instance time that precedes the cluster time when it was injected.
	if (sim.RunningBatch == nil || len(sim.RunningBatch.Requests) == 0) && sim.stepEvent == nil {
		stepTime := sim.Clock
		if clusterTime > stepTime {
			stepTime = clusterTime
		}
		step := &StepEvent{time: stepTime}
		sim.stepEvent = step
		sim.Schedule(step)
	}
}

// recordQueueSnapshots records the wait queue and running batch sizes at this step.
// Called after batch formation, before execution.
func (sim *Simulator) recordQueueSnapshots() {
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, sim.WaitQ.Len())
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))
}

// recordKVUsageMetrics records peak and time-weighted KV block usage.
// Called after execution, before completion processing.
func (sim *Simulator) recordKVUsageMetrics(stepDuration int64) {
	used := sim.KVCache.UsedBlocks()
	if used > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = used
	}
	sim.Metrics.KVBlocksUsed += float64(used) * float64(stepDuration)
}

// recordRequestCompletion records per-request metrics for a completed request.
// Called after state transitions (req.State, req.ITL, req.FinishedStepIdx)
// and KV cleanup are done.
//
// NOTE: E2E (lat) includes PostDecodeFixedOverhead and OutputTokenProcessingTime, both of
// which model non-blocking CPU overhead (concurrent with GPU execution). These inflate
// E2E and RequestCompletionTimes beyond the RequestLeftEvent timestamp by the overhead
// amount. This is architecturally intentional: real vLLM's post-processing (detokenization,
// response serialization) is non-blocking but still contributes to client-perceived latency.
// For trained-physics, PostDecodeFixedOverhead adds ~777µs to E2E; for other backends it's 0.
func (sim *Simulator) recordRequestCompletion(req *Request) {
	// Release this request's adapter pin (cold-load gate, #1466): a completed
	// request no longer uses its adapter, so the slot becomes evictable. Covers the
	// normal and length-capped completion paths (both funnel through here); the
	// running-request timeout path releases separately. No-op when inert (INV-6).
	sim.releaseAdapterPin(req)

	// INV-1 conservation: Always increment CompletedRequests.
	// For redirected requests: the source instance drained the request from its WaitQ
	// (StillQueued=0 at end), so source contributes 0 to InjectedRequests.
	// The destination is the sole completion site. Skipping CompletedRequests++ here
	// would cause the request to vanish from conservation accounting entirely.
	sim.Metrics.CompletedRequests++
	sim.Metrics.TTFTSum += req.FirstTokenTime

	// Count output tokens at completion time (not inline per step) to avoid
	// double-counting under preemption (ProgressIndex reset to 0 on eviction).
	// PI - InputLen counts decode-step increments (= OutputLen - 1 for normal completion).
	// Add 1 for the prefill-generated first token (#1097) when decodeTokens falls short
	// of OutputLen. PD 1-output decode sub-requests are the exception: their PI_final
	// lands at InputLen+1 (one step past the InputLen threshold), so decodeTokens==OutputLen
	// already — the guard prevents double-counting in that case.
	decodeTokens := int(req.ProgressIndex) - int(req.InputLen())
	if decodeTokens < len(req.OutputTokens) {
		decodeTokens++ // prefill-generated first token (vLLM parity)
	}
	if decodeTokens > 0 {
		sim.Metrics.TotalOutputTokens += decodeTokens
	}

	var itlSum int64
	for _, v := range req.ITL {
		itlSum += v
	}
	// PostDecodeFixedOverhead: fixed per-request overhead at completion (e.g., response setup).
	// Only applied to requests that went through a decode phase. Zero-output-token requests
	// (prefill-only) skip this overhead since they never entered the post-decode path.
	var postDecodeOverhead int64
	if len(req.OutputTokens) > 0 {
		postDecodeOverhead = sim.latencyModel.PostDecodeFixedOverhead()
	}
	lat := req.FirstTokenTime + itlSum + postDecodeOverhead
	sim.Metrics.RequestE2Es[req.ID] = float64(lat)
	logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
	if len(req.OutputTokens) > 0 {
		// Compute average ITL from itlSum directly (not from lat - FirstTokenTime)
		// to avoid contaminating per-token ITL with the fixed post-decode overhead.
		reqTotalOutput := itlSum
		if req.LengthCapped {
			// #588: Use actual decode step count for length-capped requests.
			// len(req.OutputTokens) is the pre-determined count; len(req.ITL) is actual.
			// TPOT convention: exclude first generated token → denominator is len(ITL)-1.
			sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.ITL)-1, 1))
		} else {
			// TPOT calculation in vLLM excludes the first generated token.
			sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
		}
	} else {
		sim.Metrics.RequestITLs[req.ID] = 0
	}
	sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
	sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat + req.ArrivalTime)
	sim.Metrics.AllITLs = append(sim.Metrics.AllITLs, req.ITL...)
}

// Step simulates a single vllm step(): batch scheduling, model execution, mirroring, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (2.5) mirror to CPU, (3) process completions, (4) schedule next step.
//
// Orphaned StepEvent guard: when a TimeoutEvent empties the RunningBatch it leaves
// sim.stepEvent pointing to the already-scheduled StepEvent (preventing the cascade
// described in #1096). If that StepEvent fires and finds nothing to do, clearing
// sim.stepEvent here prevents future QueuedEvent INV-8 guards from seeing a stale
// non-nil pointer and skipping their step-scheduling.
func (sim *Simulator) Step(now int64) {
	if sim.RunningBatch == nil && sim.WaitQ.Len() == 0 {
		sim.stepEvent = nil
		return
	}
	sim.scheduleBatch(now)
	currStepAdvance := sim.executeBatchStep(now)
	// Mirror in-use blocks to CPU tier (no-op for single-tier KVCacheState).
	// Runs after execution (new full blocks exist) and before completions
	// (completing requests' blocks are still in-use and can be mirrored).
	sim.KVCache.MirrorToCPU(sim.RunningBatch.Requests)
	remaining := sim.processCompletions(now, currStepAdvance)
	sim.scheduleNextStep(now, currStepAdvance, remaining)
}

// scheduleBatch handles Phase 1: priority assignment, queue reordering, batch formation,
// and event scheduling for preemptions and newly scheduled requests.
func (sim *Simulator) scheduleBatch(now int64) {
	sim.stepCount += 1

	// Synchronize KV cache clock for thrashing detection (no-op for single-tier KVCacheState)
	sim.KVCache.SetClock(now)

	// Order queue per scheduler policy. Priorities are static — set once at
	// EnqueueRequest/EnqueueDecodeSubRequest via SLOPriorityMap.InvertForVLLM
	// (vLLM static priority model; per-step recomputation removed).
	sim.WaitQ.Reorder(func(reqs []*Request) {
		sim.scheduler.OrderQueue(reqs, now)
	})

	// Cold-load pre-admission gate (LoRA, #1466): if the wait-queue head is a cold
	// adapter request and no load is in flight, start a serialized per-instance
	// adapter load now (committing an eviction victim and scheduling the load
	// completion). FormBatch then holds any not-yet-resident adapter out of the
	// batch via the AdapterResident predicate below. No-op when the subsystem is
	// inert (INV-6).
	sim.maybeStartAdapterLoad(now)

	// Delegate batch composition to the pluggable BatchFormation strategy.
	// Event scheduling and metrics recording happen after FormBatch returns (kernel concerns).
	batchCtx := BatchContext{
		RunningBatch:          sim.RunningBatch,
		WaitQ:                 sim.WaitQ,
		KVCache:               sim.KVCache,
		MaxScheduledTokens:    sim.maxScheduledTokens,
		MaxRunningReqs:        sim.maxRunningReqs,
		PrefillTokenThreshold: sim.longPrefillTokenThreshold,
		MaxModelLen:           sim.maxModelLen,
		Now:                   now,
		StepCount:             sim.stepCount,
		ComputedTokens:        sim.reqNumComputedTokens,
	}
	if sim.residentAdapters != nil {
		batchCtx.AdapterResident = sim.residentAdapters.IsResident
	}
	batchResult := sim.batchFormation.FormBatch(batchCtx)

	// Apply result: update running batch
	sim.RunningBatch = batchResult.RunningBatch

	// Record preemption metrics and emit debug log for each preempted request
	for _, p := range batchResult.Preempted {
		logrus.Debugf("<< Preemption: %s at %d ticks", p.Request.ID, now)
		sim.Metrics.PreemptionCount++
	}

	// Schedule events for newly scheduled requests and record scheduling metrics
	for _, s := range batchResult.NewlyScheduled {
		if sim.onAdmitInternal != nil {
			sim.onAdmitInternal(s.Request, now)
		}
		if sim.OnAdmit != nil {
			sim.OnAdmit(s.Request, now)
		}
		sim.Schedule(&ScheduledEvent{
			time:    now,
			Request: s.Request,
		})
		sim.Metrics.RequestSchedulingDelays[s.Request.ID] = now - s.Request.ArrivalTime
		sim.recordAdapterResidency(s.Request)
	}

	// Record queue depth observations after batch formation
	sim.recordQueueSnapshots()
}

// recordAdapterResidency updates the per-instance resident-adapter set when a
// request enters the running batch. Under the cold-load gate (#1466) a request is
// admitted only once its adapter is resident (a cold request is held out of the
// batch until its load completes), so here the adapter is always resident: touch
// it to most-recently-used and pin it for the request's in-flight lifetime so it
// cannot be evicted while in use (INV-L5). The pin is taken exactly once per
// request (a preempted-then-re-admitted request keeps its existing pin) and
// released at a terminal state by releaseAdapterPin. No-op when the LoRA subsystem
// is inert or the request targets the base model, preserving byte-identity (INV-6).
func (sim *Simulator) recordAdapterResidency(req *Request) {
	if sim.residentAdapters == nil || req.Adapter == "" {
		return
	}
	sim.residentAdapters.Touch(req.Adapter)
	if !req.adapterPinned {
		sim.residentAdapters.Pin(req.Adapter)
		req.adapterPinned = true
	}
}

// ReleaseAdapterPin releases req's adapter pin if it holds one. Exported for the
// cluster layer's gateway-eviction / drain paths (InstanceSimulator.EvictRequest),
// which remove a running request from this instance outside the normal
// completion/timeout terminal paths and must still free its pinned adapter slot —
// otherwise the pin leaks and eventually blocks all future cold loads (INV-L5,
// #1466). Idempotent and a no-op for unpinned (e.g. queued) or base-model requests.
func (sim *Simulator) ReleaseAdapterPin(req *Request) {
	sim.releaseAdapterPin(req)
}

// releaseAdapterPin drops this request's pin on its adapter when it reaches a
// terminal state (completed, length-capped, or timed-out while running), letting
// the adapter become evictable once no in-flight request references it. Idempotent
// via the per-request adapterPinned flag; no-op when inert or base-model (INV-6).
func (sim *Simulator) releaseAdapterPin(req *Request) {
	if sim.residentAdapters == nil || req.Adapter == "" || !req.adapterPinned {
		return
	}
	sim.residentAdapters.Unpin(req.Adapter)
	req.adapterPinned = false
}

// maybeStartAdapterLoad begins a serialized cold-adapter load when the wait-queue
// head is a new prefill request whose adapter is not yet resident (§7). It runs
// before batch formation each step. Loads serialize per instance: it starts at
// most one at a time (guarded by loadingAdapter). At load-start it commits the
// eviction victim and reserves a slot (EvictLRU when at capacity), then schedules
// an AdapterLoadCompletionEvent at now + LoadLatency; residency is committed at
// completion, so the gate keeps holding the request until then. No-op when the
// subsystem is inert (INV-6).
func (sim *Simulator) maybeStartAdapterLoad(now int64) {
	if sim.residentAdapters == nil || sim.adapterCost == nil || sim.loadingAdapter != "" {
		return
	}
	head := sim.WaitQ.Peek()
	if head == nil || head.IsDecodeSubRequest || head.Adapter == "" || sim.residentAdapters.IsResident(head.Adapter) {
		return
	}
	// Cold head: reserve a slot by committing the LRU non-pinned victim now (§7).
	if sim.residentAdapters.AtCapacity() {
		evicted, ok := sim.residentAdapters.EvictLRU()
		if !ok {
			// Every slot is pinned by an in-flight request; cannot start a load this
			// step. A running request will complete and unpin, and the INV-8 guard
			// will re-form a step to retry. (Guaranteed reachable: pins come from
			// running requests, which make progress.)
			return
		}
		sim.Metrics.AdapterEvictionCounts[evicted]++
	}
	sim.loadingAdapter = head.Adapter
	loadTicks := max(1, int64(math.Ceil(sim.adapterCost.LoadLatency(head.Adapter))))
	sim.Schedule(&AdapterLoadCompletionEvent{time: now + loadTicks, Adapter: head.Adapter})
}

// completeAdapterLoad finishes a cold-adapter load: it makes the adapter resident
// (a slot was reserved at load-start, so Store adds without further eviction),
// charges the one-time load (INV-L3 — exactly once per cold (adapter, instance)
// transition), clears the in-flight marker so the next serialized load can begin,
// and ensures a step forms so the gated request is admitted this tick (INV-8; the
// completion is ordered ahead of a co-timed step by PriorityAdapterLoad).
func (sim *Simulator) completeAdapterLoad(now int64, adapter string) {
	if sim.residentAdapters == nil {
		return
	}
	// A slot was reserved at load-start (EvictLRU when at capacity), so Store adds
	// the adapter without further eviction and must succeed. A false result would
	// mean the set filled and fully pinned during the load — impossible under the
	// blocking model (no admissions occur mid-load) — so surface it loudly (R1)
	// rather than silently dropping the load accounting. Store's evicted-id return
	// is intentionally discarded: the eviction (and its AdapterEvictionCounts
	// increment) already happened at load-start, so it is always "" here; any future
	// path that calls Store at capacity would need to account for that eviction.
	if _, admitted := sim.residentAdapters.Store(adapter); admitted {
		sim.Metrics.AdapterLoadCounts[adapter]++ // charged once per cold transition (INV-L3)
	} else {
		logrus.Errorf("[tick %07d] adapter %q load completed but could not be made resident (set full and fully pinned) — resident-set accounting bug", now, adapter)
	}
	// Clear the in-flight marker and ensure a step forms REGARDLESS of the Store
	// outcome, so the gated request is retried and the simulator never wedges with
	// queued work and no pending step (INV-8) — even on the unreachable error path.
	sim.loadingAdapter = ""
	sim.ScheduleStepIfIdle(now)
}

// executeBatchStep handles Phase 2: model execution (prefill + decode) for all requests
// in the running batch. Returns the step time advance in ticks.
func (sim *Simulator) executeBatchStep(now int64) int64 {
	// Match vLLM's scheduled_running_reqs: only requests that were allocated
	// tokens by FormBatch participate in the forward pass latency computation.
	// Requests with NumNewTokens=0 (past Phase 1 break point, token budget
	// exhaustion, or MaxModelLen boundary) retain their KV blocks and remain
	// in RunningBatch for the next step, but do not contribute to this step's
	// compute time. See vllm/v1/core/sched/scheduler.py scheduled_running_reqs.
	// Note: scheduled may be empty when all requests are idle (e.g., after
	// Phase 1 preemption cascade). All StepTime backends handle empty batches
	// correctly (return >= 1), and the max(1, ...) floor below guarantees INV-3.
	scheduled := make([]*Request, 0, len(sim.RunningBatch.Requests))
	for _, req := range sim.RunningBatch.Requests {
		if req.NumNewTokens > 0 {
			scheduled = append(scheduled, req)
		}
	}
	if sim.recordAdmissionDetail {
		sim.currentStepStartUs = now
		sim.currentScheduled = make([]SchedulerReqState, 0, len(scheduled))
		for _, req := range scheduled {
			state := sim.schedulerReqState(req, false)
			// executeBatchStep has not advanced ProgressIndex yet, so this is the
			// exact pre-step context consumed by the latency law.
			state.ComputedTokens = req.ProgressIndex
			state.ScheduledTokens = int64(req.NumNewTokens)
			sim.currentScheduled = append(sim.currentScheduled, state)
		}
	}
	currStepAdvance := sim.latencyModel.StepTime(scheduled)

	// Calibration tap (off unless BLIS_STEP_CSV is set): record the E3
	// latency-law regressors for this step. We classify scheduled with the
	// same rule StepTime uses (ProgressIndex<InputLen → prefill, else decode
	// when output exists) so the recorded (B_dec, KV, S_pf) are exactly the
	// ones that produced currStepAdvance. Recorded before the CPU-transfer
	// addition below, so t_iter is the pure GPU step the E3 law models.
	if sim.stepRec != nil {
		var bDec int
		var kv, sPf, pfCtx int64
		for _, req := range scheduled {
			si := util.Len64(req.InputTokens)
			if req.ProgressIndex < si {
				nt := int64(req.NumNewTokens)
				sPf += nt
				pfCtx += nt * (req.ProgressIndex + nt/2) // causal prefix, mirrors StepTime
			} else if len(req.OutputTokens) > 0 {
				bDec++
				kv += req.ProgressIndex
			}
		}
		sim.stepRec.record(sim.stepCount, currStepAdvance, bDec, kv, sPf, pfCtx, len(scheduled))
	}

	if sim.recordWorkTrace {
		for _, req := range scheduled {
			sim.accumulateStepWork(req.ID, req.SLOClass, req)
		}
	}

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// INV-3 defense-in-depth: guarantee clock advancement regardless of backend.
	// All LatencyModel implementations must return >= 1 per interface contract;
	// this floor catches violations that would cause infinite livelock.
	currStepAdvance = max(1, currStepAdvance)

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	// Note: Per-request TTFT fields (FirstTokenTime, RequestTTFTs) are recorded inline
	// because they are tightly coupled to the prefill/decode state transitions in this
	// loop. Safety across preemption comes from the !req.TTFTSet guard below (TTFTSet is
	// reset to false on preemption by batch_formation.go), not from overwrite idempotency —
	// the guard ensures the block fires exactly once per prefill completion so FirstTokenTime
	// always reflects the final re-prefill. TTFTSum and TotalOutputTokens are computed at
	// completion time in recordRequestCompletion to avoid double-counting when a preempted
	// request re-runs from ProgressIndex=0.
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < req.InputLen() {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// Decode phase: only generate a token if FormBatch allocated one.
			// Without this guard, a request at the MaxModelLen boundary (NumNewTokens=0
			// from proactive cap) would get a phantom ProgressIndex increment.
			// Also prevents phantom tokens from token budget exhaustion (pre-existing edge case).
			if req.NumNewTokens > 0 {
				req.ProgressIndex++
				tokenLatency := currStepAdvance + sim.latencyModel.OutputTokenProcessingTime()
				req.ITL = append(req.ITL, tokenLatency)
				if len(req.ITL) == 1 && sim.OnFirstDecodeToken != nil {
					firstDecodeTokenTick := now + tokenLatency
					if req.IsDecodeSubRequest {
						req.FirstTokenTimestamp = firstDecodeTokenTick
					}
					sim.OnFirstDecodeToken(req, firstDecodeTokenTick)
				} else if len(req.ITL) == 1 && req.IsDecodeSubRequest {
					// Capture the state even when no external observer is wired;
					// VaR snapshots consume it directly from the running request.
					req.FirstTokenTimestamp = now + tokenLatency
				}
			}
		}
		// !req.TTFTSet guard: fires once per prefill completion (including re-prefill after
		// preemption). TTFTSet is reset to false on preemption (batch_formation.go) so this
		// block fires again on re-prefill, overwriting FirstTokenTime with the correct
		// post-preemption TTFT. req.FirstTokenTime is a scalar assignment (not an
		// accumulation), so overwriting it is safe. TTFTSum is not accumulated here;
		// it is accumulated exactly once at completion time in recordRequestCompletion.
		if req.ProgressIndex == req.InputLen() && !req.TTFTSet {
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			req.FirstTokenTimestamp = req.ArrivalTime + req.FirstTokenTime
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime)
			if sim.OnFirstToken != nil {
				sim.OnFirstToken(req, req.FirstTokenTimestamp)
			}
		}
	}

	// Record KV cache usage observations after execution
	sim.recordKVUsageMetrics(currStepAdvance)

	return currStepAdvance
}

// reqWorkAccum accumulates one request's realized trajectory work.
type reqWorkAccum struct {
	slo           string
	ar            int64
	apRealized    int64
	oRealized     int64
	prefillChunks int
	prefillWork   float64
	decodeWork    float64
}

// ReqWork is an exported snapshot of a request's accumulated work (for the cluster
// builder / --edpp-work-trace).
type ReqWork struct {
	SLOClass            string
	Ar                  int64
	ApRealized          int64
	ORealized           int64
	PrefillChunks       int
	RealizedPrefillWork float64
	RealizedDecodeWork  float64
}

// SetWorkTrace enables per-request work accumulation with the given coeffs.
func (sim *Simulator) SetWorkTrace(coeffs EDPPCoeffs) {
	sim.recordWorkTrace = true
	sim.workCoeffs = coeffs
	if sim.workAcc == nil {
		sim.workAcc = make(map[string]*reqWorkAccum)
	}
}

// accumulateStepWork adds one scheduled request's per-step δ to its accumulator,
// mirroring the active latency model's charge (prefill C_pf·s + C_attn·s·(a_r+s/2)
// with a_r = full input length; decode C0 + C1·ProgressIndex). No-op when disabled.
func (sim *Simulator) accumulateStepWork(id, slo string, req *Request) {
	if !sim.recordWorkTrace {
		return
	}
	a := sim.workAcc[id]
	if a == nil {
		a = &reqWorkAccum{slo: slo, ar: util.Len64(req.InputTokens)}
		sim.workAcc[id] = a
	}
	si := util.Len64(req.InputTokens)
	if req.ProgressIndex < si {
		s := float64(req.NumNewTokens)
		a.prefillWork += sim.workCoeffs.CPf*s + sim.workCoeffs.CAttn*s*(float64(si)+s/2.0)
		a.apRealized += int64(req.NumNewTokens)
		a.prefillChunks++
	} else if len(req.OutputTokens) > 0 {
		a.decodeWork += sim.workCoeffs.C0 + sim.workCoeffs.C1*float64(req.ProgressIndex)
		a.oRealized++
	}
}

// WorkAccumulators returns a snapshot of accumulated per-request work (empty when disabled).
func (sim *Simulator) WorkAccumulators() map[string]ReqWork {
	out := make(map[string]ReqWork, len(sim.workAcc))
	for id, a := range sim.workAcc {
		out[id] = ReqWork{
			SLOClass: a.slo, Ar: a.ar, ApRealized: a.apRealized, ORealized: a.oRealized,
			PrefillChunks:       a.prefillChunks,
			RealizedPrefillWork: a.prefillWork, RealizedDecodeWork: a.decodeWork,
		}
	}
	return out
}

// processCompletions handles Phase 3: identifies completed requests, performs state
// transitions, releases KV blocks, and records completion metrics.
// Returns the remaining (non-completed) requests.
//
// IMPORTANT: This MUST run as a separate pass after executeBatchStep (BC-5).
// For zero-output-token requests, both "prefill completed" and "request completed"
// conditions are true in the same step. The two-pass design ensures prefill metrics
// (TTFT) are recorded before completion metrics (E2E). If these were ever
// consolidated into a single pass, both branches would fire for the same request
// in the same step.
func (sim *Simulator) processCompletions(now, currStepAdvance int64) []*Request {
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex >= req.InputLen()+max(util.Len64(req.OutputTokens), 1)-1 {
			// State transitions
			req.State = StateCompleted
			// Zero-output requests complete at prefill end with no decode phase.
			// The guard below has two distinct roles depending on output length:
			//
			// 1-output-token PD decode sub-requests: FormBatch Phase 2 already
			// allocated the single decode token's KV block. After executeBatchStep
			// runs, ProgressIndex = inputLen+1, so the guard
			// (req.ProgressIndex < inputLen+outputLen) evaluates to
			// (inputLen+1) < (inputLen+1) = false — preventing a duplicate allocation.
			//
			// Requests with 2+ output tokens (PD or non-PD): after executeBatchStep
			// runs, ProgressIndex = inputLen+outputLen-1 on the final decode step,
			// so the guard evaluates to true — this is the first and only allocation
			// for the final token.
			// ITL is NOT appended here — executeBatchStep already recorded it
			// for this decode step (fix for #524 phantom ITL entry).
			if len(req.OutputTokens) > 0 && req.ProgressIndex < req.InputLen()+util.Len64(req.OutputTokens) {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					logrus.Errorf("[tick %07d] KV allocation failed for completing request %s (request will still complete) — this indicates a cache accounting bug", now, req.ID)
					sim.Metrics.KVAllocationFailures++
				}
			}
			// ReleaseKVBlocks is safe even when the final-token allocation failed:
			// the decode pre-check returns false before any state mutation (check-then-act
			// pattern, matching vLLM kv_cache_manager.py:334-336), so RequestMap is
			// preserved and Release frees all blocks from prior successful allocations.
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})

			// Record completion metrics
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(req, now+currStepAdvance) {
					sim.InjectArrival(next)
				}
			}
		} else if sim.maxModelLen > 0 && req.ProgressIndex >= sim.maxModelLen-1 {
			// BC-5: Proactive MaxModelLen cap — force-complete at boundary.
			// After the proactive cap in FormBatch prevents scheduling tokens beyond
			// maxModelLen-1, and the decode guard in executeBatchStep prevents phantom
			// ProgressIndex increments, the request reaches PI=maxModelLen-1 and needs
			// a completion path. This matches vLLM's effective behavior where the scheduler
			// cap at max_model_len-1-num_computed prevents further scheduling.
			// Note: vLLM completes length-capped requests via check_stop (num_tokens >= max_model_len)
			// which fires AFTER the model appends the generated token, producing maxModelLen-input
			// output tokens. BLIS completes at PI >= maxModelLen-1 (before the final token),
			// producing maxModelLen-1-input tokens (1 fewer). This is because BLIS lacks vLLM's
			// post-execution check_stop loop; processCompletions is the DES equivalent.
			//
			// NOTE (R23 exception): Final-token KV allocation is intentionally skipped here.
			// The normal completion path's AllocateKVBlocks for the last token is not useful
			// for a force-terminated request whose blocks are immediately released.
			logrus.Warnf("[tick %07d] force-completing request %s: ProgressIndex %d >= MaxModelLen-1 %d (length-capped)",
				now, req.ID, req.ProgressIndex, sim.maxModelLen-1)
			sim.Metrics.LengthCappedRequests++
			req.LengthCapped = true
			// Refresh Metrics.Requests: NewRequestMetrics was called at enqueue before
			// LengthCapped was known. Update so per-request JSON reflects the flag.
			if rm, ok := sim.Metrics.Requests[req.ID]; ok {
				rm.LengthCapped = true
				sim.Metrics.Requests[req.ID] = rm
			}
			req.State = StateCompleted
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management (length-capped)
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(req, now+currStepAdvance) {
					sim.InjectArrival(next)
				}
			}
		} else {
			remaining = append(remaining, req)
		}
	}
	return remaining
}

// scheduleNextStep handles Phase 4: schedules the next step event based on
// remaining requests, or starts a new batch if only WaitQ has pending work
// (work-conserving property, INV-8).
func (sim *Simulator) scheduleNextStep(now, currStepAdvance int64, remaining []*Request) {
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		// estimate queue overhead from LR (sim.features)
		//
		pbe := StepEvent{time: now + currStepAdvance}
		sim.Schedule(&pbe)
		sim.stepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.stepEvent = nil
		// Work-conserving: if WaitQ has pending requests, immediately
		// schedule a new step to form the next batch. Without this,
		// queued requests are stranded until the next arrival event
		// triggers a QueuedEvent — violating the work-conserving
		// property that real vLLM maintains.
		//
		// Exception (cold-load gate, #1466): when a per-instance adapter load is in
		// flight, the wait-queue head is gated (held out of the batch) for the load
		// duration and nothing else can be admitted (blocking model) — including any
		// warm-adapter or base-model requests queued behind it (intentional
		// head-of-line blocking, DT-faithful; design §7/D1). The scheduled
		// work IS the load, so INV-8 is satisfied by the pending
		// AdapterLoadCompletionEvent — which re-forms a step on completion via
		// ScheduleStepIfIdle. Scheduling an empty step here instead would spin one
		// step per tick for the whole load. Inert when no LoRA (loadingAdapter == "").
		if sim.WaitQ.Len() > 0 && sim.loadingAdapter == "" {
			pbe := StepEvent{time: now + currStepAdvance}
			sim.Schedule(&pbe)
			sim.stepEvent = &pbe
		}
	}
}
