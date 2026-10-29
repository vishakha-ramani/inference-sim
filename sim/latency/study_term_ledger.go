package latency

// Study-only: per-term step-time attribution ledger (config-attribution study branch).
//
// When the BLIS_STUDY_TERM_LEDGER environment variable names a file, every
// TrainedPhysicsModel.StepTime call records its named, already-β-weighted terms here,
// and FlushStudyTermLedger writes the per-term totals as JSON. StepTime has exactly one
// caller (Simulator step execution), so Σ terms over the run is the simulated
// GPU-busy time decomposed by physical cause. Unset ⇒ no recording, byte-identical output.

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/inference-sim/inference-sim/sim"
)

// Term names, in StepTime's summation order.
var studyTermNames = []string{
	"prefill_compute", "prefill_kv", "decode_compute", "decode_kv", "weight_load",
	"tp_allreduce", "moe_dispatch", "per_layer_overhead", "per_request_overhead",
	"per_step_overhead", "moe_interleave_overhead", "adapter_overhead",
}

type studyTermLedger struct {
	mu        sync.Mutex
	path      string
	Steps     int64              `json:"steps"`
	BusyUs    float64            `json:"busy_us"` // Σ returned StepTime (post clamp/adapter)
	TermUs    map[string]float64 `json:"term_us"`
	Dominant  map[string]int64   `json:"dominant_steps"` // steps where this term was largest
	BatchHist map[int]int64      `json:"batch_size_hist"`

	// Shadow pricing (BLIS_STUDY_SHADOW_TP / BLIS_STUDY_SHADOW_HARDWARE): the same executed
	// batches costed under an alternate TP and/or GPU. Σ shadow terms = what this run's exact
	// schedule would have cost there; ShadowPerRequest attributes it per request exactly like
	// PerRequest, so the direct effect can be split out of TTFT/ITL/E2E as well as cost.
	ShadowTP         int                       `json:"shadow_tp,omitempty"`
	ShadowHardware   string                    `json:"shadow_hardware,omitempty"`
	ShadowBusyUs     float64                   `json:"shadow_busy_us,omitempty"`
	ShadowTermUs     map[string]float64        `json:"shadow_term_us,omitempty"`
	ShadowPerRequest map[string]*studyReqTerms `json:"shadow_per_request,omitempty"`

	// Per-request attribution: Σ terms of the steps each request spent in prefill and in
	// decode (a step's full time is charged to every request in it — batching is shared
	// wall-clock time, which is what the request experiences).
	PerRequest map[string]*studyReqTerms `json:"per_request"`
}

type studyReqTerms struct {
	PrefillUs []float64 `json:"prefill_us"` // studyTermNames order
	DecodeUs  []float64 `json:"decode_us"`
	// DecodeShareUs: each decode step's terms divided by the number of requests in that step —
	// this request's even share of the shared step, which strips out the batch-size effect.
	DecodeShareUs []float64 `json:"decode_share_us"`
	// DiscardedUs: steps from attempts abandoned by preemption. Preemption fully resets a
	// request (batch_formation.go: ProgressIndex, ITL and TTFTSet return to zero), so BLIS's
	// TTFT is measured to the first token of the FINAL attempt and this work sits inside it.
	DiscardedUs  []float64 `json:"discarded_us"`
	Attempts     int       `json:"attempts"`
	lastProgress int64
	DecodeSteps  int `json:"decode_steps"`
}

var studyLedger = func() *studyTermLedger {
	p := os.Getenv("BLIS_STUDY_TERM_LEDGER")
	if p == "" {
		return nil
	}
	return &studyTermLedger{path: p, TermUs: map[string]float64{}, ShadowTermUs: map[string]float64{}, ShadowPerRequest: map[string]*studyReqTerms{}, PerRequest: map[string]*studyReqTerms{}, Dominant: map[string]int64{}, BatchHist: map[int]int64{}}
}()

// scaleTo rescales terms so they sum to the integer step time BLIS advanced the clock by.
func scaleTo(terms []float64, returned int64) []float64 {
	scaled := make([]float64, len(terms))
	var sum float64
	for _, v := range terms {
		sum += v
	}
	for i, v := range terms {
		if sum > 0 {
			scaled[i] = v * float64(returned) / sum
		}
	}
	return scaled
}

// attribute charges one step's (scaled) terms to every request in the batch: to its prefill
// or decode phase, and to discarded work when its progress went backward (preempted).
func attribute(per map[string]*studyReqTerms, scaled []float64, batch []*sim.Request) {
	for _, req := range batch {
		rt := per[req.ID]
		if rt == nil {
			rt = &studyReqTerms{PrefillUs: make([]float64, len(scaled)), DecodeUs: make([]float64, len(scaled)),
				DiscardedUs: make([]float64, len(scaled)), DecodeShareUs: make([]float64, len(scaled)), Attempts: 1}
			per[req.ID] = rt
		}
		// Progress went backward ⇒ the request was preempted and restarted: fold the
		// abandoned attempt into DiscardedUs and start counting the new attempt afresh.
		if req.ProgressIndex < rt.lastProgress {
			for i := range scaled {
				rt.DiscardedUs[i] += rt.PrefillUs[i] + rt.DecodeUs[i]
				rt.PrefillUs[i], rt.DecodeUs[i], rt.DecodeShareUs[i] = 0, 0, 0
			}
			rt.DecodeSteps = 0
			rt.Attempts++
		}
		rt.lastProgress = req.ProgressIndex
		// Same classification StepTime uses (ProgressIndex is pre-step here).
		dst := rt.DecodeUs
		if req.ProgressIndex < req.InputLen() {
			dst = rt.PrefillUs
		} else {
			rt.DecodeSteps++
			for i, v := range scaled {
				rt.DecodeShareUs[i] += v / float64(len(batch))
			}
		}
		for i, v := range scaled {
			dst[i] += v
		}
	}
}

// record takes terms in studyTermNames order and the step time actually returned.
func (l *studyTermLedger) record(terms []float64, returned int64, batch []*sim.Request) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Steps++
	l.BusyUs += float64(returned)
	l.BatchHist[len(batch)]++
	attribute(l.PerRequest, scaleTo(terms, returned), batch)
	best := 0
	for i, v := range terms {
		l.TermUs[studyTermNames[i]] += v
		if v > terms[best] {
			best = i
		}
	}
	l.Dominant[studyTermNames[best]]++
}

// recordShadow is record for the shadow model's pricing of the same executed batch.
func (l *studyTermLedger) recordShadow(terms []float64, returned int64, batch []*sim.Request) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ShadowBusyUs += float64(returned)
	attribute(l.ShadowPerRequest, scaleTo(terms, returned), batch)
	for i, v := range terms {
		l.ShadowTermUs[studyTermNames[i]] += v
	}
}

// FlushStudyTermLedger writes the ledger JSON if recording is enabled. No-op otherwise.
func FlushStudyTermLedger() error {
	if studyLedger == nil {
		return nil
	}
	studyLedger.mu.Lock()
	defer studyLedger.mu.Unlock()
	b, err := json.MarshalIndent(studyLedger, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(studyLedger.path, b, 0o644)
}

// attachStudyShadow builds the shadow model (same coefficients and model, alternate TP via
// BLIS_STUDY_SHADOW_TP and/or alternate GPU via BLIS_STUDY_SHADOW_HARDWARE, a key of the
// bundled hardware_config.json in the working directory). Requires BLIS_STUDY_TERM_LEDGER.
// No-op when neither is set.
func attachStudyShadow(m *TrainedPhysicsModel, coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) error {
	tpEnv, hwEnv := os.Getenv("BLIS_STUDY_SHADOW_TP"), os.Getenv("BLIS_STUDY_SHADOW_HARDWARE")
	if (tpEnv == "" && hwEnv == "") || studyLedger == nil {
		return nil
	}
	shw := hw
	if tpEnv != "" {
		tp, err := strconv.Atoi(tpEnv)
		if err != nil || tp <= 0 {
			return fmt.Errorf("BLIS_STUDY_SHADOW_TP: want positive int, got %q", tpEnv)
		}
		shw.TP = tp
	}
	if hwEnv != "" {
		data, err := os.ReadFile("hardware_config.json")
		if err != nil {
			return fmt.Errorf("BLIS_STUDY_SHADOW_HARDWARE: %w", err)
		}
		entries, err := ParseHardwareCalibEntries(data)
		if err != nil {
			return fmt.Errorf("BLIS_STUDY_SHADOW_HARDWARE: %w", err)
		}
		calib, ok := entries[hwEnv]
		if !ok {
			return fmt.Errorf("BLIS_STUDY_SHADOW_HARDWARE: %q is not in hardware_config.json", hwEnv)
		}
		shw.HWConfig = calib
	}
	sh, err := NewTrainedPhysicsModel(coeffs, shw)
	if err != nil {
		return fmt.Errorf("study shadow model (TP=%d, hardware=%q): %w", shw.TP, hwEnv, err)
	}
	sh.adapterCost, sh.specTokens, sh.studyIsShadow = m.adapterCost, m.specTokens, true
	m.studyShadow = sh
	studyLedger.ShadowTP, studyLedger.ShadowHardware = shw.TP, hwEnv
	return nil
}
