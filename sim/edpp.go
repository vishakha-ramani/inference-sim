package sim

import (
	"fmt"
	"math"
	"sort"
)

// EDPP (Lyapunov drift-plus-penalty) prefill/decode placement decider.
//
// EDPP realizes the design in docs/superpowers/specs/2026-06-18-edpp-dpp-routing-design.md
// as a DisaggregationDecider: the per-request choice x ∈ {P, D} maps onto
// DisaggregationDecision.Disaggregate (P = true = remote prefill, D = false =
// prefill-on-decode). The rule (E14) trades a transfer penalty against backlog
// balancing and SLO pressure, where the SLO weights are virtual-queue states
// (z_ttft, z_itl) rather than hand-tuned constants.
//
// # Physics source
//
// EDPP's iteration-time coefficients {α, α_p, c0, c1, c_pf, c_attn} are FROZEN,
// loaded from a calibration JSON (scripts/calibration/coeffs-*.json) via
// LoadEDPPCoeffs and carried in EDPPConfig.Coeffs. They feed the demand W_p (§3),
// the live drain rates μ (§4), the normalizers W* (§7), and the counterfactual
// predictors (§5). See docs/superpowers/specs/2026-06-24-edpp-rule-coeffs-wiring-design.md.
// The model parameter on NewEDPPDecider is retained for the deferred
// recalibration-drift watchdog (§4) and is otherwise unused.
//
// # Backlog accounting
//
// Q_p/Q_d are the WAITING backlog (design §1.2, §6.1): work of requests routed to a
// server but not yet admitted to its running batch. OnRoute adds a request's work;
// OnAdmit removes the admitted side's share (Q_p when the prefill sub-request enters
// the prefill batch, Q_d when the decode/normal request enters the decode batch);
// OnComplete updates only z and N̂_out (the running batch is captured by T(B−1)/μ in
// the predictors, not by Q). Forget releases any still-waiting share on terminal
// non-completion (timeout/drop). Decode work uses the per-class running-mean output
// length N̂_out (INV-9: realized length read only at completion). The TTFT predictors
// are symmetric co-residency: TTFT_x = Q_x^wait/μ_x + n·(T_x(B−1)+δ_pf-chunk) [+c_xfer].
//
// # Oracle safety (INV-9)
//
// Decide reads only input-side quantities: len(req.InputTokens) and the prefix-cache
// hit (via cacheQuery), never req.OutputTokens. Per-class N̂_out is updated only at
// completion from realized output length — not used for servability decisions, so
// INV-9 is not violated.
//
// # Modeling residuals (carried from design §10)
//
//   - δ_pf-chunk (ITL inflation from prefill-on-decode) is the marginal work of ONE
//     prefill chunk — min(a_p, ChunkTokens) tokens, since only one chunk co-schedules
//     per decode iteration (§3.2: δ_pf(s)=c_pf·s, s = chunk tokens). It is distinct
//     from the full prefill demand W_p, which lands on the backlog/TTFT terms instead.
//   - For MoE models, c0/c1 slightly under-count because weight-load grows weakly
//     with B via nEff; exact for dense models.
//   - Optimistic in-flight backlog increments (§8.1) are omitted in the base
//     implementation; the z-feedback half of the rule is immune to scrape staleness.

// EDPPConfig holds the controller's fixed knobs. All durations are microseconds.
//
// SLO targets are per-SLO-class. TauTTFTUs/TauITLUs are the defaults applied to any
// class without an explicit entry in the per-class override maps (and to the empty
// "" class). TauTTFTByClassUs/TauITLByClassUs override the defaults for named classes
// (e.g. "critical", "batch"). The decision rule is evaluated from the perspective of
// the request's own class: a stricter class reads the same server backlog as more
// threatening, and its realized-SLO feedback accumulates in its own virtual queue.
type EDPPConfig struct {
	TauTTFTUs              int64                 // default τ_ttft: time-average TTFT SLO target (µs)
	TauITLUs               int64                 // default τ_itl: time-average ITL SLO target (µs)
	TauRefUs               int64                 // fixed reference τ for the transfer-penalty normalization (µs); makes the penalty scale 1/τ_ttft² like the other terms. Independent of the operating τ_ttft.
	TauTTFTByClassUs       map[string]int64      // per-class τ_ttft overrides (µs); nil = use default for all
	TauITLByClassUs        map[string]int64      // per-class τ_itl overrides (µs); nil = use default for all
	TauE2EUs               int64                 // default τ_e2e: end-to-end SLO deadline budget (µs); used ONLY by the VaR drift oracle (Rule=="var") to evaluate a co-resident's E2E composite-good (deadline = arrival + τ_e2e). 0 ⇒ E2E conjunct disabled in g(). Not read by dpp/least-ttft.
	TauE2EByClassUs        map[string]int64      // per-class τ_e2e overrides (µs); nil = use default for all
	V                      float64               // penalty/stability tradeoff knob (Neely's V); larger ⇒ fewer offloads
	CXferUs                int64                 // c_xfer: KV-transfer cost paid when routing P (µs)
	NomPrefillTokens       int                   // S_nom: nominal prefill chunk for the fixed prefill normalizer
	NomDecodeCtx           int                   // L_nom: nominal decode context for the fixed decode normalizer
	BlockSize              int                   // token block size for the prefix-cache a_p computation
	ChunkTokens            int                   // per-step prefill token budget (max_num_batched_tokens); caps δ_pf-chunk. 0 = no cap (whole prefill counts as one chunk)
	Coeffs                 EDPPCoeffs            // frozen E3 latency-law coefficients (design §1.1); required
	CoeffsByGPU            map[string]EDPPCoeffs // per-GPU-type θ_i overrides; nil ⇒ use Coeffs for every candidate (homogeneous)
	TraceEnabled           bool                  // when true, Decide attaches an EDPPDecisionTrace (intermediate rule terms) to each decision. Off ⇒ zero allocation.
	TAdmEstimator          string                // admission-delay estimator name ("" ⇒ waiting, the current formula)
	Joint                  bool                  // when true, Decide enumerates all (decode, prefill) candidates and picks the drift-plus-penalty argmin (joint P/D routing, --edpp-joint); false ⇒ the reduced fixed-d local-vs-disagg rule.
	JointTraceEnabled      bool                  // when true (joint mode only), decideJoint attaches an EDPPJointDecisionTrace comparing the scorer's (d,p) pick to the joint argmin. Off ⇒ zero allocation, no shadow prefill scorer run.
	Rule                   string                // reduced-path decision rule: "" / "dpp" (drift-plus-penalty, default) | "least-ttft" (disaggregate iff ttftP < ttftD; bypasses the drift/z/V machinery, design 2026-07-15) | "var" (replace the work-currency balance term with a value-at-risk externality; DIAGNOSTIC ORACLE, design 2026-07-21).
	VarMetric              string                // VaR scoring kernel when Rule=="var": "flip" (A, binary composite-good flip count; default), "util" (B, saturating slack utility), "hazard" (C, deadline-slack hazard × delay). Ignored unless Rule=="var".
	VarKeepCongestion      bool                  // when Rule=="var": KEEP the Lyapunov work-congestion drift term and ADD the VaR externality (drift-plus-VaR), instead of replacing it. The congestion term feels a node saturating (capacity + heterogeneity); VaR supplies the SLO externality. false ⇒ pure VaR (externality replaces congestion). Ignored unless Rule=="var".
	VarCongestionWeight    float64               // drift-plus-VaR balance: cost = VarCongestionWeight·congestion + VaR (the two terms live on different scales, so this makes them commensurate). 0 ⇒ 1.0. Used only when Rule=="var" and VarKeepCongestion.
	VarNormalize           bool                  // drift-plus-VaR auto-normalization (joint path): per-decision min-max normalize congestion and VaR across candidates to [0,1] before combining, so VarCongestionWeight is a scale-free relative weight (default ≈1) instead of an absolute scale. The spread floor (VarNormalizeFloorScale) makes a symmetric congestion term (identical hardware) cancel automatically. Used only when Rule=="var" and VarKeepCongestion.
	VarNormalizeFloorScale float64               // drift-plus-VaR normalization spread floor (joint path, only with VarNormalize): the min-max denominator is max{spread, ε₀} with ε₀ = scale·(dwork/W*) — one arriving request's work on the nominal decode instance in reference units. A term whose cross-candidate spread falls below ε₀ is compressed toward zero in proportion instead of amplified to the full unit range (the noise-amplification fix). 0 ⇒ 1.0 (paper default); the sensitivity study varies this scale alongside VarCongestionWeight.
	KairosBeta             float64               // Kairos baseline (Rule=="kairos") TBT safety margin β: a deflected prefill chunk must keep the decode step within β·τ_itl. 0 ⇒ 1.0. See sim/edpp_kairos.go (arXiv:2607.02043).
	VarDeployable          bool                  // DEPLOYABLE VaR (Rule=="var"): estimate each decode co-resident's remaining steps from the censored per-class N̂_out (max(N̂_out − StepsDone, 1)) instead of the ORACLE true remaining. INV-9-safe (no output-length read). Turns the diagnostic ceiling into a runnable policy; measures the oracle→deployable gap.
	VarCollocPrefill       bool                  // DEPLOYABLE VaR extra (Rule=="var"): also price the first-token (TTFT) value-at-risk of collocated prefill occupants ON the decode instance. These are requests a prior collocate decision placed there that are still pre-first-token, which RunningDecode skips so the decode-side terms miss them. Reads only remaining prompt tokens (known input, INV-9-safe). On by default so the rule prices this externality; set false to ablate.
	VarGoodputObjective    bool                  // DIAGNOSTIC (Rule=="var" && VarKeepCongestion, --edpp-var-goodput): reframe the objective from "minimize transfer cost" to "maximize goodput". The rule charges VaR − good_r (the goodput destroyed among co-residents minus the goodput EARNED for the arriving request) and DROPS the standalone transfer penalty (its effect already flows through the request's own projected TTFT). good_r uses the request's decode length via reqNHatOut — the censored N̂_out, or the TRUE output length when OracleOutputLen is also set (an INV-9 upper bound). Off ⇒ byte-identical to the current rule (INV-6). Tests whether the goodput reframing beats the transfer-cost objective.
	OracleOutputLen        bool                  // DIAGNOSTIC / UPPER-BOUND ONLY (--edpp-oracle-output-len): substitute the routed request's TRUE output length (len(req.OutputTokens)) for the per-class N̂_out estimate when charging its OWN decode work (joint W_d and the qdWork backlog bookkeeping). Violates INV-9 by design; never a deployable policy. Co-resident remaining stays estimated/censored. Used to test whether output-length estimation error explains the overload collapse (control for the value-vs-work-currency hypothesis).
	CXferSizeAware         bool                  // --edpp-c-xfer-size-aware: compute c_xfer per request from KV size (XferBaseUs + ⌈a_r/blockSize⌉·blockSize·KVBytesPerTokenPerGPU / bandwidth), mirroring the DES KV-transfer executor, instead of the flat CXferUs. Off ⇒ byte-identical to the flat-c_xfer behavior. Deployable (input-only, oracle-safe).
	KVBytesPerTokenPerGPU  float64               // per-GPU KV bytes per token (from ModelConfig + prefill TP); used only when CXferSizeAware
	XferBandwidthGBps      float64               // inter-instance KV-transfer bandwidth (GB/s), matching the DES executor; used only when CXferSizeAware
	XferBaseUs             float64               // KV-transfer base latency (µs), matching the DES executor; used only when CXferSizeAware
}

// EDPPJointDecisionTrace records, for one joint (--edpp-joint) decision, the scorer's
// pick vs the joint argmin, so an analysis pass can quantify how often (and by how much)
// the joint objective overrides the composable scorer. It is attached to
// DisaggregationDecision.EDPPJointTrace only when the decider has JointTraceEnabled set,
// and is pure instrumentation — computing it does not change the routing decision (INV-6).
//
// ScorerD is the decode-routing policy's pick (state.SelectedInstance). ScorerP is the
// prefill-routing policy's pick, obtained by SHADOW-running the injected prefill scorer
// over the prefill snapshots (compute-only, not acted on); it is populated only on
// disaggregate decisions (JointP != ""). JScorer is J evaluated at the scorer's slice —
// disagg (ScorerD, ScorerP) on a disagg decision, else local on ScorerD — and JJoint is
// the argmin's J. Because the argmin ranges over a superset that always includes the
// scorer's slice, JJoint <= JScorer by construction (asserted as an internal invariant).
type EDPPJointDecisionTrace struct {
	Class        string  // request SLO class
	ScorerD      string  // decode-routing policy pick (state.SelectedInstance)
	JointD       string  // joint argmin decode node
	ScorerP      string  // shadow prefill-scorer pick (disagg only; else "")
	JointP       string  // joint argmin prefill node (disagg only; else "")
	AgreeD       bool    // ScorerD == JointD
	AgreeP       bool    // ScorerP == JointP (trivially true when both empty, i.e. local)
	JScorer      float64 // J at the scorer's slice
	JJoint       float64 // J at the joint argmin (== the committed decision's cost)
	Disaggregate bool    // the joint decision (disagg vs local)
}

// EDPPDecisionTrace records the intermediate terms of one E14 rule evaluation, for
// diagnostics. It is attached to DisaggregationDecision.EDPPTrace only when the decider
// has TraceEnabled set. Every field is dimensionless or in microseconds; LHS and RHS are
// the two sides of the (E14) inequality and decompose exactly into the listed components:
//
//	LHS = BalanceTermD − BalanceTermP
//	RHS = TransferTerm + TTFTTerm + ITLTerm
//	Disaggregate = LHS > RHS
//
// On early-return paths (empty prompt, fully prefix-cached) the rule is not evaluated;
// SkipReason names the path and the term fields are left zero.
type EDPPDecisionTrace struct {
	Class        string  // request SLO class (drives τ resolution)
	SkipReason   string  // "" = rule evaluated; else "empty-prompt" or "fully-cached"
	Ap           int     // a_p: uncached prompt tokens
	Wp           float64 // W_p: prefill demand (µs)
	DeltaPfChunk float64 // δ_pf-chunk: one-chunk ITL inflation (µs)
	QdRaw        float64 // Q_d: raw decode backlog (µs)
	QpRaw        float64 // Q_p: raw prefill backlog (µs)
	Qd           float64 // normalized decode backlog (Q_d / W*_d)
	Qp           float64 // normalized prefill backlog (Q_p / W*_p)
	MuDNom       float64 // μ_d^nom
	MuPNom       float64 // μ_p^nom
	WStarD       float64 // W*_d normalizer (µs)
	WStarP       float64 // W*_p normalizer (µs)
	TauTTFT      float64 // τ_ttft for this class (µs)
	TauITL       float64 // τ_itl for this class (µs)
	TTFTP        float64 // predicted TTFT under P (µs)
	TTFTD        float64 // predicted TTFT under D (µs)
	ITLP         float64 // retained for compatibility; always 0 — ITL pressure is now the collapsed ITLTerm (Task 7 design); Decide no longer sets this field
	ITLD         float64 // retained for compatibility; always 0 — ITL pressure is now the collapsed ITLTerm (Task 7 design); Decide no longer sets this field
	ZTTFT        float64 // normalized TTFT virtual queue (z_ttft = Z_ttft / τ_ttft)
	ZITL         float64 // normalized ITL virtual queue (z_itl = Z_itl / τ_itl)
	BalanceTermD float64 // q_d·(W_p/W*_d)
	BalanceTermP float64 // q_p·(W_p/W*_p)
	TransferTerm float64 // V·(c_xfer/τ_ttft)
	TTFTTerm     float64 // z_ttft·(TTFT_P−TTFT_D)/τ_ttft
	ITLTerm      float64 // z_itl·(ITL_P−ITL_D)/τ_itl
	LHS          float64 // backlog-balancing benefit
	RHS          float64 // transfer penalty + SLO pressure
	Disaggregate bool    // the decision (LHS > RHS)
}

func (c EDPPConfig) validate() {
	if err := c.Coeffs.validate(); err != nil {
		panic(fmt.Sprintf("EDPPConfig: %v", err))
	}
	if float64(c.TauITLUs) <= c.Coeffs.AlphaD {
		panic(fmt.Sprintf("EDPPConfig: TauITLUs (%d µs) must exceed decode α (%.1f µs); the ITL SLO is physically unachievable on this hardware", c.TauITLUs, c.Coeffs.AlphaD))
	}
	switch {
	case c.TauTTFTUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauTTFTUs must be > 0, got %d", c.TauTTFTUs))
	case c.TauITLUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauITLUs must be > 0, got %d", c.TauITLUs))
	case c.TauRefUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauRefUs must be > 0, got %d", c.TauRefUs))
	case c.V < 0:
		panic(fmt.Sprintf("EDPPConfig: V must be >= 0, got %v", c.V))
	case c.CXferUs < 0:
		panic(fmt.Sprintf("EDPPConfig: CXferUs must be >= 0, got %d", c.CXferUs))
	case c.NomPrefillTokens <= 0:
		panic(fmt.Sprintf("EDPPConfig: NomPrefillTokens must be > 0, got %d", c.NomPrefillTokens))
	case c.NomDecodeCtx <= 0:
		panic(fmt.Sprintf("EDPPConfig: NomDecodeCtx must be > 0, got %d", c.NomDecodeCtx))
	case c.BlockSize <= 0:
		panic(fmt.Sprintf("EDPPConfig: BlockSize must be > 0, got %d", c.BlockSize))
	}
	for cls, v := range c.TauTTFTByClassUs {
		if v <= 0 {
			panic(fmt.Sprintf("EDPPConfig: TauTTFTByClassUs[%q] must be > 0, got %d", cls, v))
		}
	}
	for cls, v := range c.TauITLByClassUs {
		if v <= 0 {
			panic(fmt.Sprintf("EDPPConfig: TauITLByClassUs[%q] must be > 0, got %d", cls, v))
		}
	}
	switch c.Rule {
	case "", "dpp", "least-ttft", "kairos":
	case "var":
		if _, ok := parseVarKernel(c.VarMetric); !ok {
			panic(fmt.Sprintf("EDPPConfig: Rule==\"var\" requires VarMetric in {flip,util,hazard}, got %q", c.VarMetric))
		}
	default:
		panic(fmt.Sprintf("EDPPConfig: Rule must be \"\", \"dpp\", \"least-ttft\", \"var\", or \"kairos\", got %q", c.Rule))
	}
	for cls, v := range c.TauE2EByClassUs {
		if v <= 0 {
			panic(fmt.Sprintf("EDPPConfig: TauE2EByClassUs[%q] must be > 0, got %d", cls, v))
		}
	}
}

// edppMinMu floors the nominal drain rate so the normalizers and predictor
// denominators never collapse to 0 when α ≥ T_iter (a degenerate operating point).
const edppMinMu = 1e-3

// SLOFeedbackDecider is the lifecycle hook. OnRoute fires once when a request is
// committed to a pool (increments the work backlog Q). OnAdmit fires when a routed
// request first enters a running batch (drains the admitted side's waiting backlog
// share). OnComplete fires at the request's terminal completion (bumps the virtual
// queues and updates the per-class output-length estimate N̂_out — backlog is already
// gone via OnAdmit). Forget releases any remaining backlog for a routed request that
// reaches a terminal state WITHOUT a normal completion (timeout/drop) — no z bump, no
// N̂_out update. Call sites discover it via a type assertion, so adding it does not
// break DisaggregationDecider.
//
// key is an explicit, stable correlation id chosen by the caller: it MUST be the
// same value at OnRoute and at the matching OnAdmit/OnComplete/Forget, so the
// conservation bookkeeping decrements exactly the work the route added. For
// PD-disaggregated requests the routed request and the completing decode sub-request
// have different Request.IDs, so the cluster passes the parent identity here (see the
// cluster's feedSLOFeedback / feedAdmission). The decider never parses the key; it
// only uses it as a map key.
type SLOFeedbackDecider interface {
	OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string) // increment Q at routing (per-pool + per-instance)
	OnAdmit(key string, prefillSide bool)                                                           // drain waiting-work share at admission
	OnFirstToken(key string, nowUs int64)                                                           // true up z_ttft from the realized first-token time
	OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64)                   // bump z_itl, update N̂_out (z_ttft owned by OnFirstToken)
	Forget(key string)                                                                              // release Q for a terminally non-completed routed request
}

// edppAwaiting tracks a routed request still awaiting its first token, so z_ttft can be
// credited continuously from its observed elapsed wait (a certain lower bound on its TTFT
// miss) instead of only once at completion. startUs is the request's arrival time (when
// the TTFT clock starts); credited is the cumulative amount already applied to z_ttft for
// this request, so each sweep applies only the increment (no double-count).
type edppAwaiting struct {
	startUs  int64
	class    string
	credited float64
}

// edppPendingWork records the remaining waiting-work shares of a routed request.
// wp/wd are the µs still counted in qp/qd respectively; OnAdmit zeroes the
// admitted side's share and removes the entry once both are zero. Forget drains
// whatever share remains when the request reaches a terminal non-completion state.
type edppPendingWork struct {
	toPrefill bool
	wp, wd    float64 // remaining µs in qp/qd (drained to 0 as each side is admitted)

	// Routed instances recorded at OnRoute so the drains (OnAdmit/Forget) can mirror
	// the pool-scalar decrement per-instance. decodeInst carries the wd share;
	// prefillInst carries the wp share (disagg only). "" is ignored (nil-safe).
	decodeInst, prefillInst string
}

// edppInstWork is the per-instance waiting-work accumulator (µs). It mirrors the
// pool-level qp/qd scalars split by the routed instance, for the future joint
// P/D routing reader; the reduced (pool-level) path never reads it.
type edppInstWork struct {
	wp, wd float64
}

// edppRunningMean is a per-class running mean of realized output lengths (N̂_out).
type edppRunningMean struct {
	n   int64
	sum float64
}

func (r *edppRunningMean) update(v float64) { r.n++; r.sum += v }
func (r *edppRunningMean) mean() float64 {
	if r.n == 0 {
		return 1 // no completions yet: 1-token decode estimate (conservative seed)
	}
	return r.sum / float64(r.n)
}

// edppClassState is the per-SLO-class virtual-queue state (accumulated SLO
// violation, µs). Bumped on each completion (E8).
type edppClassState struct {
	zTTFT, zITL float64
}

// edppNorm bundles the class-resolved normalizers used by a single Decide call.
// μ_p^nom is class-independent (it depends on the prefill iteration time, not the
// SLO); μ_d^nom and both W* depend on the class's τ targets.
type edppNorm struct {
	muDNom, muPNom  float64
	wStarD, wStarP  float64
	tauTTFT, tauITL float64
}

// EDPPDecider implements DisaggregationDecider and SLOFeedbackDecider.
type EDPPDecider struct {
	cfg              EDPPConfig
	model            LatencyModel
	cacheQuery       map[string]func([]TokenID) int // shared with precise-prefix-cache scorer; may be nil
	prefillSnapshots func() []RoutingSnapshot   // prefill-pool backlogs; may be nil (⇒ Q_p = 0)

	// Physics constants precomputed once at construction (class-independent).
	// μ_p^nom is fixed: a moving normalizer would break the Lyapunov drift telescoping
	// and invert the congestion signal (design §4.3).
	muPNom float64 // μ_p^nom = 1 − α_p/T_iter^nom (E11); does not depend on τ

	// Frozen calibrated coefficients stored for use by downstream tasks.
	coeffs EDPPCoeffs

	// Per-GPU-type θ_i (design 2026-07-14). nil ⇒ homogeneous: coeffsFor returns coeffs.
	coeffsByGPU map[string]EDPPCoeffs

	// joint selects the (decode, prefill) argmin enumeration path (--edpp-joint).
	// false ⇒ the reduced fixed-d local-vs-disagg rule (byte-identical legacy path).
	joint bool

	// rule selects the reduced-path decision: "" / "dpp" => lhs > rhs (drift-plus-penalty);
	// "least-ttft" => ttftP < ttftD; "var" => lhs_var (value-at-risk externality) > rhs.
	// Estimation is identical; only the LHS/comparison differs.
	rule string

	// varMetric is the VaR scoring kernel (A/B/C) used when rule == "var". Parsed once at
	// construction from cfg.VarMetric (validated). Meaningless (zero value) for other rules.
	varMetric varKernel

	// varKeepCongestion selects drift-plus-VaR when rule == "var": keep the work-congestion
	// drift term and ADD the VaR externality, rather than replacing congestion with VaR.
	varKeepCongestion bool

	// varCongestionWeight scales the congestion term in drift-plus-VaR so it is commensurate
	// with the VaR externality (they live on different scales). Defaults to 1.0.
	varCongestionWeight float64

	// varNormalize enables per-decision min-max normalization of the congestion and VaR terms
	// (joint path) so varCongestionWeight is a scale-free relative weight. See EDPPConfig.VarNormalize.
	varNormalize bool

	// varNormalizeFloorScale scales the spread floor ε₀ = scale·(dwork/W*) applied to the min-max
	// denominator on the normalized path (default 1.0). See EDPPConfig.VarNormalizeFloorScale.
	varNormalizeFloorScale float64

	// varDeployable estimates co-resident remaining steps from the censored per-class N̂_out
	// instead of the oracle true remaining (INV-9-safe). See EDPPConfig.VarDeployable.
	varDeployable bool

	// varCollocPrefill also prices the first-token VaR of collocated prefill occupants on the
	// decode instance (deployable, INV-9-safe). See EDPPConfig.VarCollocPrefill.
	varCollocPrefill bool

	// varGoodputObjective reframes the objective to goodput: charge VaR − good_r and drop the
	// standalone transfer penalty (diagnostic). See EDPPConfig.VarGoodputObjective.
	varGoodputObjective bool

	// deficit accumulates the per-decision occupancy of the two SLO-deficit virtual queues,
	// sampled at every Decide call in their NORMALIZED form (z = Z/τ, the form the rule
	// actually multiplies into its objective). It answers a question the goodput metric
	// cannot: do the time-average SLO constraints ever bind, or is the rule driven entirely
	// by the congestion and goodput terms? Pure instrumentation; never read by the rule.
	deficit edppDeficitAccum

	// kairosBeta is the TBT safety margin for the Kairos baseline (rule == "kairos").
	kairosBeta float64

	// Pluggable admission-delay estimator (default "waiting" reproduces QWork/Mu).
	tadmEstimator AdmissionDelayEstimator

	// Per-class controller state: virtual queues keyed by SLO class. Lazily created.
	zByClass map[string]*edppClassState

	// Conservation bookkeeping (design §6 conservation form).
	qpWork, qdWork float64                    // running work-µs backlogs
	pending        map[string]edppPendingWork // per-request work, keyed by Request.ID

	// Per-instance congestion queues Q_i: the same waiting-work the pool scalars
	// track, split by the routed instance. Populated always (in OnRoute/drained in
	// OnAdmit/Forget) but READ only by the future joint P/D routing path. Invariant:
	// sum over instances of wp == qpWork, and of wd == qdWork.
	qByInstance map[string]*edppInstWork
	nHatOut     map[string]*edppRunningMean // per-class realized output-length estimate

	// Requests routed but not yet at their first token, keyed by the same conservation
	// key as pending. Drives the continuous z_ttft credit (design: responsive z_ttft).
	awaitingFirstToken map[string]*edppAwaiting

	// captureAdmissionCtx, when set, makes Decide attach the assembled per-pool
	// AdmissionContext(s) to the returned DisaggregationDecision so the cluster's
	// --edpp-admission-trace companion trace can recompute all six estimator predictions
	// at end of run. Off by default (zero-cost). Logging-only path (INV-9).
	captureAdmissionCtx bool

	// prefillScorer, when set, is the injected prefill-routing scorer used ONLY to
	// shadow-compute EDPPJointDecisionTrace.ScorerP on disaggregate joint decisions.
	// It returns the instance ID the prefill routing policy would pick over the given
	// prefill snapshots. It MUST NOT perturb production routing state (the cluster wires
	// a dedicated-RNG policy instance) — the shadow pick is logged, never acted on
	// (INV-6). nil ⇒ ScorerP is left empty and J_scorer falls back to the local slice.
	prefillScorer func(*Request, []RoutingSnapshot) string
}

// SetPrefillScorer injects the shadow prefill-routing scorer used to populate
// EDPPJointDecisionTrace.ScorerP (joint divergence trace). Logging-only; the returned
// pick is never acted on and must not mutate production routing RNG (INV-6).
func (d *EDPPDecider) SetPrefillScorer(fn func(*Request, []RoutingSnapshot) string) {
	d.prefillScorer = fn
}

// SetCaptureAdmissionContext toggles attaching the assembled per-pool AdmissionContext
// to each DisaggregationDecision (for the --edpp-admission-trace companion trace).
// Off by default; enabling it does not affect the routing decision itself (INV-9).
func (d *EDPPDecider) SetCaptureAdmissionContext(v bool) { d.captureAdmissionCtx = v }

// NewEDPPDecider constructs the decider and initializes class-independent physics
// constants from cfg.Coeffs. cfg is validated (panics on invalid values, R3).
// model is retained for the deferred recalibration-drift watchdog (§4).
// cacheQuery and prefillSnapshots may be nil (e.g. unit tests, or no prefill pool).
// The per-class target maps are copied defensively.
func NewEDPPDecider(cfg EDPPConfig, model LatencyModel, cacheQuery map[string]func([]TokenID) int, prefillSnapshots func() []RoutingSnapshot) *EDPPDecider {
	cfg.validate()
	for gpu, c := range cfg.CoeffsByGPU {
		if err := c.validate(); err != nil {
			panic(fmt.Sprintf("NewEDPPDecider: coeffs_by_gpu[%q]: %v", gpu, err))
		}
	}
	if model == nil {
		panic("NewEDPPDecider: model must not be nil")
	}
	cfg.TauTTFTByClassUs = copyClassTargets(cfg.TauTTFTByClassUs)
	cfg.TauITLByClassUs = copyClassTargets(cfg.TauITLByClassUs)
	cfg.TauE2EByClassUs = copyClassTargets(cfg.TauE2EByClassUs)

	// VarMetric is validated in cfg.validate() only when Rule=="var"; parse defensively
	// (unknown ⇒ varKernelFlip, unreachable past validation) so the field is always well-defined.
	varMetric, _ := parseVarKernel(cfg.VarMetric)
	varCongestionWeight := cfg.VarCongestionWeight
	if varCongestionWeight <= 0 {
		varCongestionWeight = 1.0
	}
	kairosBeta := cfg.KairosBeta
	if kairosBeta <= 0 {
		kairosBeta = 1.0
	}
	varNormalizeFloorScale := cfg.VarNormalizeFloorScale
	if varNormalizeFloorScale <= 0 {
		varNormalizeFloorScale = 1.0
	}

	d := &EDPPDecider{
		cfg:                    cfg,
		model:                  model,
		cacheQuery:             cacheQuery,
		prefillSnapshots:       prefillSnapshots,
		joint:                  cfg.Joint,
		rule:                   cfg.Rule,
		varMetric:              varMetric,
		varKeepCongestion:      cfg.VarKeepCongestion,
		varCongestionWeight:    varCongestionWeight,
		varNormalize:           cfg.VarNormalize,
		varNormalizeFloorScale: varNormalizeFloorScale,
		varDeployable:          cfg.VarDeployable,
		varCollocPrefill:       cfg.VarCollocPrefill,
		varGoodputObjective:    cfg.VarGoodputObjective,
		kairosBeta:             kairosBeta,
		zByClass:               make(map[string]*edppClassState),
		coeffs:                 cfg.Coeffs,
		coeffsByGPU:            cfg.CoeffsByGPU,
		pending:                make(map[string]edppPendingWork),
		qByInstance:            make(map[string]*edppInstWork),
		nHatOut:                make(map[string]*edppRunningMean),

		awaitingFirstToken: make(map[string]*edppAwaiting),
	}

	// INV-9 guard: oracle admission estimators read TRUE remaining output, so they
	// may only be logged (via NewAdmissionEstimator directly), never drive routing.
	if !IsDeployableEstimator(cfg.TAdmEstimator) {
		panic(fmt.Sprintf("NewEDPPDecider: oracle admission estimators are logging-only, not routing drivers (TAdmEstimator=%q)", cfg.TAdmEstimator))
	}
	est, err := NewAdmissionEstimator(cfg.TAdmEstimator)
	if err != nil {
		// Library boundary: mirror cfg.validate()'s panic-on-invalid-config style (R3).
		panic(fmt.Sprintf("NewEDPPDecider: %v", err))
	}
	d.tadmEstimator = est

	// μ_p^nom = 1 − α_p/(α_p + c_pf·S_pf^nom) (design §7): from frozen coefficients.
	d.muPNom = d.coeffs.muPNom(cfg.NomPrefillTokens)
	return d
}

func copyClassTargets(m map[string]int64) map[string]int64 {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]int64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func clampMu(mu float64) float64 {
	if mu < edppMinMu {
		return edppMinMu
	}
	if mu > 1.0 {
		return 1.0
	}
	return mu
}

// targetsFor resolves the (τ_ttft, τ_itl) SLO targets for an SLO class: a per-class
// override if present, else the configured defaults (also used for the empty class).
func (d *EDPPDecider) targetsFor(class string) (tauTTFTUs, tauITLUs int64) {
	tauTTFTUs, tauITLUs = d.cfg.TauTTFTUs, d.cfg.TauITLUs
	if v, ok := d.cfg.TauTTFTByClassUs[class]; ok {
		tauTTFTUs = v
	}
	if v, ok := d.cfg.TauITLByClassUs[class]; ok {
		tauITLUs = v
	}
	return
}

// e2eFor resolves the E2E SLO deadline budget (µs) for a class, mirroring targetsFor:
// the per-class override if present, else the default TauE2EUs. Used only by the VaR
// oracle (Rule=="var") to compute a co-resident's absolute E2E deadline = arrival + τ_e2e.
// Returns 0 when no E2E target is configured (⇒ the E2E conjunct is disabled in g()).
func (d *EDPPDecider) e2eFor(class string) int64 {
	tauE2EUs := d.cfg.TauE2EUs
	if v, ok := d.cfg.TauE2EByClassUs[class]; ok {
		tauE2EUs = v
	}
	return tauE2EUs
}

// coeffsFor returns the per-GPU-type θ_i for gpuType, or the global coeffs when
// no override exists (nil map, unmapped type, or empty gpuType). This is the single
// selection point for per-instance heterogeneous coefficients; with no coeffs_by_gpu
// it returns d.coeffs for every candidate, preserving byte-identity (INV-6).
func (d *EDPPDecider) coeffsFor(gpuType string) EDPPCoeffs {
	if gpuType != "" {
		if c, ok := d.coeffsByGPU[gpuType]; ok {
			return c
		}
	}
	return d.coeffs
}

// normFor computes the class-resolved normalizers (E10–E12). μ_d^nom = 1 − α/τ_itl
// and W* = μ^nom · τ_ttft are constant for a given class (α and μ_p^nom are fixed),
// so the §11 signal-direction property holds within each class.
func (d *EDPPDecider) normFor(class string) edppNorm {
	tauTTFTUs, tauITLUs := d.targetsFor(class)
	muD := d.coeffs.muDNom(float64(tauITLUs))
	return edppNorm{
		muDNom:  muD,
		muPNom:  d.muPNom,
		wStarD:  muD * float64(tauTTFTUs),
		wStarP:  d.muPNom * float64(tauTTFTUs),
		tauTTFT: float64(tauTTFTUs),
		tauITL:  float64(tauITLUs),
	}
}

// Decide evaluates the E14 rule and returns Disaggregate=true (route P) when the
// backlog-balancing benefit of offloading exceeds the transfer penalty plus SLO
// pressure. Conservative fallbacks (Disaggregate=false): empty prompt, or a request
// whose prompt is fully prefix-cached on the selected decode pod (no prefill to offload).
func (d *EDPPDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	// Credit z_ttft from the observed elapsed wait of all in-flight requests still
	// awaiting their first token, so the SLO virtual queue reflects trouble happening
	// right now (not only past completions). Lazy: runs exactly when z is about to be
	// read. state.Clock is the current sim time; nil state (unit tests) ⇒ skip.
	if state != nil {
		d.creditAwaiting(state.Clock)
	}

	// Kairos baseline (--edpp-rule kairos, arXiv:2607.02043): load-aware prefill deflection.
	// It picks its own decode node (the deflection target), so it branches ahead of both the
	// joint and reduced paths. Baseline only — not our rule. See sim/edpp_kairos.go.
	if d.rule == "kairos" {
		return d.decideKairos(req, state)
	}

	// Joint P/D routing (--edpp-joint): enumerate all (decode, prefill) candidates and
	// pick the drift-plus-penalty argmin. The reduced fixed-d rule below is left untouched
	// and stays byte-identical (INV-6) — the guard is the only change to its entry.
	if d.joint {
		return d.decideJoint(req, state)
	}

	keepD := DisaggregationDecision{Disaggregate: false}
	if len(req.InputTokens) == 0 {
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "empty-prompt"}
		}
		return keepD
	}

	// a_p = uncached prompt tokens (oracle-safe; same source as PrefixThresholdDecider).
	sel := ""
	if state != nil {
		sel = state.SelectedInstance
	}
	ap := d.apForInstance(req, sel)
	if ap <= 0 {
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "fully-cached", Ap: ap}
		}
		return keepD // fully cached: no prefill work to disaggregate
	}

	// Selected decode instance's θ_i (design 2026-07-14): the decode-side terms below are
	// charged against the physics of the decode node this request would land on, so they use
	// thetaD, not the pool-aggregate d.coeffs. Homogeneous clusters (no CoeffsByGPU) fall back
	// to d.coeffs, so this is byte-identical to the pre-existing behavior (INV-6).
	decSnap, _ := d.selectedDecodeSnapshot(state)
	thetaD := d.coeffsFor(decSnap.GPUType)

	// W_p = full prefill demand of this request (E6), charged against the decode node's balance.
	wp := thetaD.Wp(ap, len(req.InputTokens))

	// Live decode-server state from the pre-selected decode snapshot (the pod this
	// request would land on); fall back to the first snapshot, else nominal.
	bDec, kv, sPf := d.selectedDecodeState(state)
	muDec := thetaD.muDecode(bDec, kv, sPf)
	tBminus1 := thetaD.tIterDecode(bDec, kv, sPf)

	// Prefill-server live state: S_pf summed over prefill snapshots.
	var sPfPrefill int64
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = d.prefillSnapshots()
		for _, s := range prefillSnaps {
			sPfPrefill += s.ResidentPrefillTokens
		}
	}
	// prefill side is pool-aggregate (no single p chosen in the reduced rule) → global coeffs
	muPf := d.coeffs.muPrefill(sPfPrefill)

	n := d.normFor(req.SLOClass)

	// Conservation-bookkept backlogs (Task 6), in work-µs.
	qP := d.qpWork
	qD := d.qdWork
	qd := qD / n.wStarD
	qp := qP / n.wStarP

	// Explicit-chunk predictors (§5.1, divergence §9.1). chunk = decode batched-token
	// budget; n_chunks = ⌈a_p/chunk⌉; δ_pf-chunk = c_pf·chunk.
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	nChunks := math.Ceil(float64(ap) / float64(chunk))
	// Occupancy inputs for the admission-delay estimators (fluid/rollforward, Tasks 4/5).
	// Decode side reads the selected decode snapshot; prefill side reads the first prefill snapshot.
	// ReqKVNeed: KV blocks this request needs ≈ ⌈a_r / blockSize⌉ (a_r = full input length; oracle-safe).
	reqKVNeed := int64(0)
	if d.cfg.BlockSize > 0 {
		reqKVNeed = int64((len(req.InputTokens) + d.cfg.BlockSize - 1) / d.cfg.BlockSize)
	}
	// RemainingStepsEst (deployable): per-running-request censored estimate, NOT a mean that
	// can go negative. A request that has produced StepsDone tokens has o_r ≥ StepsDone
	// (censored lower bound), so floor the class output estimate by the max in-flight elapsed.
	remStepsEst := d.decodeRemStepsEst(decSnap, req.SLOClass)
	var prefillSnap RoutingSnapshot
	if len(prefillSnaps) > 0 {
		prefillSnap = prefillSnaps[0]
	}
	// Prefill-pool RemainingStepsEst: symmetric to the decode censored estimate, but
	// derived from the prefill snapshot's own running-prefill occupants (RunningPrefill)
	// so the ttft_p estimators are live on the prefill pool. StepsDone/TrueRemaining here
	// are prefill-chunk counts (see Simulator.RunningPrefillState). Floors at 1 so fluid's
	// wave term is non-zero when the prefill batch is full.
	prefillRemStepsEst := d.prefillRemStepsEst(prefillSnap)
	// AdmissionRate (req/µs) for the little estimator: prefer the explicit field if a
	// source ever populates it, else fall back to the observed completion rate
	// (DispatchRate, req/s → req/µs). In steady state admission ≈ completion (§3.8),
	// so DispatchRate is the arrival-free observable for L̄_q/λ_adm.
	decAdmRate := admissionRateFromSnapshot(decSnap)
	prefillAdmRate := admissionRateFromSnapshot(prefillSnap)

	prefillCtx := AdmissionContext{
		QWork: qP, Mu: muPf,
		BatchSize: prefillSnap.BatchSize, MaxBatchSize: int(prefillSnap.MaxBatchSize),
		FreeKVBlocks: prefillSnap.FreeKVBlocks, ReqKVNeed: reqKVNeed,
		TIter: d.coeffs.tIterPrefill(sPfPrefill), QueueDepth: prefillSnap.QueueDepth,
		AdmissionRate: prefillAdmRate, RemainingStepsEst: prefillRemStepsEst,
		// INV-9 asymmetry: prefill remaining (inLen − ProgressIndex) is known input, so
		// it is deployable — NOT censored. Decode's Running below stays censored.
		Running: prefillSnap.RunningPrefill,
	}
	decodeCtx := AdmissionContext{
		QWork: qD, Mu: muDec,
		BatchSize: decSnap.BatchSize, MaxBatchSize: int(decSnap.MaxBatchSize),
		FreeKVBlocks: decSnap.FreeKVBlocks, ReqKVNeed: reqKVNeed,
		TIter: tBminus1, QueueDepth: decSnap.QueueDepth,
		AdmissionRate: decAdmRate, RemainingStepsEst: remStepsEst,
		Running: censorOracleRemaining(decSnap.RunningDecode),
	}
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)
	cXferUs := d.cXferUsFor(req) // flat CXferUs, or the size-aware transfer cost
	// Prefill time = admission delay + the batch-iteration overhead the request waits through
	// (nChunks iterations at the path's per-iteration time: the decode batch's load for the local
	// path, the prefill pool's for the disagg path) + the request's OWN prefill work Wp. Wp carries
	// both projection (C_pf·a_p) AND attention over context (C_attn·a_p·(a_r+a_p/2)), matching the
	// executor's per-prefill-step charge. (Earlier this charged only the projection term C_pf·chunk,
	// which under-modelled long-context prefill; the attention term is now included for fidelity.)
	ttftP := tAdmP + nChunks*d.coeffs.tIterPrefill(sPfPrefill) + d.coeffs.Wp(ap, len(req.InputTokens)) + cXferUs
	ttftD := tAdmD + nChunks*tBminus1 + thetaD.Wp(ap, len(req.InputTokens))

	// Per-class virtual queues.
	var zTTFT, zITL float64
	if z := d.zByClass[req.SLOClass]; z != nil {
		zTTFT = z.zTTFT / n.tauTTFT
		zITL = z.zITL / n.tauITL
	}
	d.sampleDeficit(zTTFT, zITL)

	// E14, with the ITL term in collapsed closed form (§5.2/§9.2):
	//   z_itl·(ITL_P − ITL_D)/τ_itl = − z_itl·(c_pf·chunk)/τ_itl
	balanceTermD := qd * (wp / n.wStarD)
	balanceTermP := qp * (wp / n.wStarP)
	lhs := balanceTermD - balanceTermP

	transferTerm := d.transferPenalty(n, cXferUs)
	ttftTerm := zTTFT * (ttftP - ttftD) / n.tauTTFT
	itlTerm := -zITL * (thetaD.CPf * float64(chunk)) / n.tauITL
	rhs := transferTerm + ttftTerm + itlTerm

	disagg := lhs > rhs
	switch d.rule {
	case "least-ttft":
		disagg = ttftP < ttftD // bypass drift/z/V; decide purely on predicted TTFT (ttftP already includes c_xfer)
	case "var":
		// Value-at-risk: replace ONLY the work-currency balance term with the value-currency
		// externality lhs_var = VaR_local − VaR_disagg (goodput destroyed among co-residents);
		// the transfer/TTFT/ITL self terms in rhs are unchanged. Reuse `lhs` so the trace's LHS
		// field reflects the value the decision actually used (BalanceTermD/P stay informational).
		var nowUs float64
		if state != nil {
			nowUs = float64(state.Clock)
		}
		varLHS := d.varReducedLHS(req, nowUs, decSnap, prefillSnaps, thetaD, bDec, kv, sPf, chunk, nChunks, ttftP, sPfPrefill)
		if d.varKeepCongestion {
			// drift-plus-VaR: the work-congestion drift (current lhs = balanceTermD − balanceTermP)
			// PLUS the value-currency externality. Congestion feels a node's backlog; VaR the SLO cost.
			// The weight makes the two commensurate (they live on different scales).
			lhs = d.varCongestionWeight*lhs + varLHS
		} else {
			lhs = varLHS // pure VaR: the externality replaces the work-currency balance term
		}
		disagg = lhs > rhs
	}
	dec := DisaggregationDecision{Disaggregate: disagg}
	if d.cfg.TraceEnabled {
		dec.EDPPTrace = &EDPPDecisionTrace{
			Class: req.SLOClass, Ap: ap, Wp: wp, DeltaPfChunk: thetaD.CPf * float64(chunk),
			QdRaw: qD, QpRaw: qP, Qd: qd, Qp: qp,
			MuDNom: n.muDNom, MuPNom: n.muPNom, WStarD: n.wStarD, WStarP: n.wStarP,
			TauTTFT: n.tauTTFT, TauITL: n.tauITL,
			TTFTP: ttftP, TTFTD: ttftD,
			ZTTFT: zTTFT, ZITL: zITL,
			BalanceTermD: balanceTermD, BalanceTermP: balanceTermP,
			TransferTerm: transferTerm, TTFTTerm: ttftTerm, ITLTerm: itlTerm,
			LHS: lhs, RHS: rhs, Disaggregate: disagg,
		}
	}
	if d.captureAdmissionCtx {
		dc := decodeCtx
		dec.AdmissionCtxDecode = &dc
		if len(prefillSnaps) > 0 {
			pc := prefillCtx
			dec.AdmissionCtxPrefill = &pc
		}
	}
	return dec
}

// apForInstance returns a_p, the uncached prompt tokens for req on instance instID
// (block-aligned prefix-cache query; oracle-safe, input-only per INV-9). instID == ""
// or an absent/nil cacheQuery entry yields the full prompt length (cold). Shared by the
// reduced path (with instID = state.SelectedInstance) and the per-candidate joint path.
func (d *EDPPDecider) apForInstance(req *Request, instID string) int {
	ap := len(req.InputTokens)
	if instID != "" && d.cacheQuery != nil {
		if fn, ok := d.cacheQuery[instID]; ok && fn != nil {
			ap = len(req.InputTokens) - fn(req.InputTokens)*d.cfg.BlockSize
		}
	}
	return ap
}

// decodeRemStepsEst is the deployable per-running-request censored remaining-steps
// estimate for a decode snapshot (design: NOT a mean that can go negative). A request
// that has produced StepsDone tokens has o_r ≥ StepsDone (censored lower bound), so the
// class output estimate is floored by the max in-flight elapsed. Returns 1 when the
// snapshot has no running decode occupants. Shared by the reduced and joint paths.
func (d *EDPPDecider) decodeRemStepsEst(snap RoutingSnapshot, class string) float64 {
	n := len(snap.RunningDecode)
	if n == 0 {
		return 1.0
	}
	nHatOut := d.nHatFor(class).mean()
	var maxSteps int64
	for _, r := range snap.RunningDecode {
		if r.StepsDone > maxSteps {
			maxSteps = r.StepsDone
		}
	}
	nHatEff := math.Max(nHatOut, float64(maxSteps)) // censored: N̂_out ≥ longest in-flight elapsed
	var sum float64
	for _, r := range snap.RunningDecode {
		sum += math.Max(nHatEff-float64(r.StepsDone), 1) // per-request remaining, floored at 1
	}
	return sum / float64(n)
}

// prefillRemStepsEst is the symmetric censored remaining-steps estimate for a prefill
// snapshot, derived from its running-prefill occupants' TrueRemaining (prefill-chunk
// counts, floored at 1). Returns 1 when the snapshot has no running prefill occupants.
// Shared by the reduced and joint paths.
func (d *EDPPDecider) prefillRemStepsEst(snap RoutingSnapshot) float64 {
	n := len(snap.RunningPrefill)
	if n == 0 {
		return 1.0
	}
	var sum float64
	for _, r := range snap.RunningPrefill {
		rem := r.TrueRemaining
		if rem < 1 {
			rem = 1
		}
		sum += float64(rem)
	}
	return sum / float64(n)
}

// transferPenalty is the normalized KV-transfer cost paid when offloading prefill:
// V·(c_xfer/τ_ttft)·(τ_ref/τ_ttft) (design §3; scales like 1/τ_ttft² as the other terms).
// Shared by the reduced path (RHS) and the joint path (added to disagg candidates).
// cXferUs is the transfer cost for THIS request (flat CXferUs, or the size-aware value
// from cXferUsFor) — the same quantity added to ttftP, so the penalty and the TTFT
// prediction stay consistent about what a disaggregation costs.
func (d *EDPPDecider) transferPenalty(n edppNorm, cXferUs float64) float64 {
	return d.cfg.V * (cXferUs / n.tauTTFT) * (float64(d.cfg.TauRefUs) / n.tauTTFT)
}

// cXferUsFor returns the assumed KV-transfer cost (µs) for routing this request to a
// prefill node. Default: the flat CXferUs. With CXferSizeAware it mirrors the DES
// executor's sizing (sim/cluster/pd_events.go): base + blocks·blockSize·kvBytes / bandwidth,
// where blocks = ⌈a_r/blockSize⌉ = the request's KV footprint (input-only, oracle-safe).
// This removes the decision/execution mismatch where a flat 5ms mis-priced disaggregation
// by ~10× on long prompts and ~4.5× on short ones.
func (d *EDPPDecider) cXferUsFor(req *Request) float64 {
	if !d.cfg.CXferSizeAware {
		return float64(d.cfg.CXferUs)
	}
	bwBytesPerUs := d.cfg.XferBandwidthGBps * 1000.0 // GB/s → bytes/µs
	if bwBytesPerUs <= 0 || d.cfg.BlockSize <= 0 {
		return d.cfg.XferBaseUs
	}
	transferBytes := float64(d.reqKVNeed(req)) * float64(d.cfg.BlockSize) * d.cfg.KVBytesPerTokenPerGPU
	return d.cfg.XferBaseUs + transferBytes/bwBytesPerUs
}

// instWorkRaw reads the per-instance waiting-work backlog (µs) for id WITHOUT creating a
// map entry (unlike instWork). Absent id ⇒ (0, 0). Used by the joint path to read the
// per-candidate normalized congestion q_i = W_i / W*.
func (d *EDPPDecider) instWorkRaw(id string) (wp, wd float64) {
	if w, ok := d.qByInstance[id]; ok {
		return w.wp, w.wd
	}
	return 0, 0
}

// chunkTerms returns (n_chunks, δ_pf-chunk) for a_p uncached prefill tokens under the
// decode batched-token budget (ChunkTokens), where δ_pf-chunk = theta.CPf·chunk is charged
// on the pool the prefill runs on (local ⇒ decode θ, disagg ⇒ prefill θ). a_p ≤ 0 (fully
// cached / empty) ⇒ (0, 0): no prefill work, no per-chunk ITL inflation.
func (d *EDPPDecider) chunkTerms(theta EDPPCoeffs, ap int) (nChunks, deltaPfChunk float64) {
	if ap <= 0 {
		return 0, 0
	}
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	return math.Ceil(float64(ap) / float64(chunk)), theta.CPf * float64(chunk)
}

// reqKVNeed is ⌈a_r / blockSize⌉ (a_r = full input length; oracle-safe). 0 when BlockSize ≤ 0.
func (d *EDPPDecider) reqKVNeed(req *Request) int64 {
	if d.cfg.BlockSize <= 0 {
		return 0
	}
	return int64((len(req.InputTokens) + d.cfg.BlockSize - 1) / d.cfg.BlockSize)
}

// decideJoint implements the joint P/D routing rule (--edpp-joint): it enumerates every
// (decode d, placement) candidate — local (prefill+decode on d) and disagg (decode on d,
// prefill on each prefill node p) — computes the NORMALIZED absolute objective J(d,·) for
// each, and returns the argmin. Unlike the reduced rule (which differences ttft_p−ttft_d
// for a single fixed d), J carries every term with its divisor and uses the ABSOLUTE
// per-candidate admission-delay predictor T̂(a), so the argmin correctly ranks distinct
// decode nodes. Restricted to the scorer's single d it reproduces the reduced local-vs-
// disagg decision (§5.5; see TestJoint_ReducesToScorerSliceMatchesReduced).
//
// edppDeficitAccum tallies normalized deficit-queue occupancy across decisions.
type edppDeficitAccum struct {
	n                    int
	sumZT, sumZI         float64
	nonzeroZT, nonzeroZI int
	maxZT, maxZI         float64
}

// EDPPDeficitStats reports how much the two SLO-deficit virtual queues actually bound the
// rule over a run. All z values are normalized (z = Z/τ), matching the form the objective
// multiplies. MeanZT/MeanZI are time averages over decisions; FracActiveZT/FracActiveZI are
// the fractions of decisions on which the queue was non-empty. A run with FracActive ≈ 0 had
// a dormant time-average constraint: the placement was decided by the other terms.
type EDPPDeficitStats struct {
	Decisions                  int
	MeanZT, MeanZI             float64
	FracActiveZT, FracActiveZI float64
	MaxZT, MaxZI               float64
	// AwaitingAtEnd is how many requests were still registered as awaiting a first token
	// when the run ended. A value far above the in-flight count means the awaiting map is
	// leaking records, each of which keeps accruing lateness into z_ttft forever.
	AwaitingAtEnd int
}

// DeficitQueueStats returns the accumulated deficit-queue occupancy for this run.
func (d *EDPPDecider) DeficitQueueStats() EDPPDeficitStats {
	a := d.deficit
	st := EDPPDeficitStats{Decisions: a.n, MaxZT: a.maxZT, MaxZI: a.maxZI, AwaitingAtEnd: len(d.awaitingFirstToken)}
	if a.n > 0 {
		st.MeanZT = a.sumZT / float64(a.n)
		st.MeanZI = a.sumZI / float64(a.n)
		st.FracActiveZT = float64(a.nonzeroZT) / float64(a.n)
		st.FracActiveZI = float64(a.nonzeroZI) / float64(a.n)
	}
	return st
}

// sampleDeficit records one decision's normalized deficit-queue occupancy.
func (d *EDPPDecider) sampleDeficit(zT, zI float64) {
	a := &d.deficit
	a.n++
	a.sumZT += zT
	a.sumZI += zI
	if zT > 0 {
		a.nonzeroZT++
		if zT > a.maxZT {
			a.maxZT = zT
		}
	}
	if zI > 0 {
		a.nonzeroZI++
		if zI > a.maxZI {
			a.maxZI = zI
		}
	}
}

// Determinism (INV-6): decode and prefill snapshots are iterated in ascending instance-ID
// order, local is considered before disagg, and a strictly-lower J (by more than 1e-12)
// is required to replace the incumbent — so ties resolve to the lowest-index, local-first
// candidate, matching the reduced rule's strict Disaggregate = LHS > RHS tie handling.
func (d *EDPPDecider) decideJoint(req *Request, state *RouterState) DisaggregationDecision {
	class := req.SLOClass
	n := d.normFor(class)

	// Per-class virtual queues, normalized by their τ (as in the reduced rule).
	var zTTFT, zITL float64
	if z := d.zByClass[class]; z != nil {
		zTTFT = z.zTTFT / n.tauTTFT
		zITL = z.zITL / n.tauITL
	}
	d.sampleDeficit(zTTFT, zITL)

	// Deterministic candidate ordering (INV-6): sort snapshots by instance ID.
	decodeSnaps := sortedSnapshotsByID(stateSnapshots(state))
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = sortedSnapshotsByID(d.prefillSnapshots())
	}

	reqKVNeed := d.reqKVNeed(req)
	nHatOut := d.reqNHatOut(req) // deployable N̂_out, or TRUE o_r under the diagnostic oracle flag

	// Candidate-invariant terms shared by every (d,·) cost evaluation this decision.
	// wd (W_d) and mDec (m_dec = δ̄_dec) are now computed per candidate in
	// jointCandidateCost under θ_i (they discriminate fast vs slow decode nodes).
	var nowUs float64
	if state != nil {
		nowUs = float64(state.Clock)
	}
	ec := &jointEvalCtx{
		req: req, n: n, zTTFT: zTTFT, zITL: zITL,
		reqKVNeed: reqKVNeed, nHatOut: nHatOut, nowUs: nowUs,
	}

	var best *cand
	consider := func(c cand) {
		if best == nil || c.J < best.J-1e-12 {
			cc := c
			best = &cc
		}
	}

	// drift-plus-VaR with auto-normalization: the congestion and VaR terms live on different
	// scales, so a fixed weight is an absolute scale (see VarNormalize). The normalized path
	// min-max normalizes both across the candidate set (two passes) so the weight is relative.
	if d.rule == "var" && d.varKeepCongestion && d.varNormalize {
		for _, c := range d.jointNormalizedCandidates(ec, decodeSnaps, prefillSnaps) {
			consider(c)
		}
	} else {
		// least-TTFT-joint scores each candidate's own forward TTFT (hardware-aware, no drift/z/
		// VaR); every other rule keeps the drift-plus-penalty objective. The argmin loop, local-
		// before-disagg ordering, and tie-break are shared.
		costFn := d.jointCandidateCost
		if d.rule == "least-ttft" {
			costFn = d.jointCandidateTTFT
		}
		for _, ds := range decodeSnaps {
			// --- local: prefill+decode co-resident on d ---
			consider(cand{dID: ds.ID, local: true, J: costFn(ec, ds, nil)})
			// --- disagg: decode on d, prefill on each prefill node p ---
			for _, ps := range prefillSnaps {
				psCopy := ps
				consider(cand{dID: ds.ID, pID: ps.ID, local: false, J: costFn(ec, ds, &psCopy)})
			}
		}
	}

	if best == nil {
		return DisaggregationDecision{Disaggregate: false} // no candidates (empty snapshots)
	}
	var dec DisaggregationDecision
	if best.local {
		dec = DisaggregationDecision{Disaggregate: false, DecodePodOverride: best.dID}
	} else {
		dec = DisaggregationDecision{Disaggregate: true, DecodePodOverride: best.dID, PrefillPodHint: best.pID}
	}
	// Stage C admission capture (pure instrumentation, gated; INV-6): snapshot the
	// AdmissionContext(s) of the WINNING candidate so --edpp-admission-trace can recompute
	// every estimator's prediction at end of run and compare it against the realized t_adm.
	// Rebuilt from the same helpers the scoring pass used, after the argmin is committed, so
	// it cannot influence the decision. Without this the joint rule emitted no admission
	// records at all and the estimator's bias was unmeasurable under the policies we report.
	if d.captureAdmissionCtx {
		for _, ds := range decodeSnaps {
			if ds.ID == best.dID {
				dc := d.jointDecodeAdmissionCtx(ec, ds)
				dec.AdmissionCtxDecode = &dc
				break
			}
		}
		if !best.local {
			for _, ps := range prefillSnaps {
				if ps.ID == best.pID {
					pc := d.jointPrefillAdmissionCtx(ec, ps)
					dec.AdmissionCtxPrefill = &pc
					break
				}
			}
		}
	}

	// Scorer-vs-joint divergence trace (pure instrumentation, gated; INV-6): compute only
	// when enabled, after the decision is committed, so it can never influence the argmin.
	if d.cfg.JointTraceEnabled {
		dec.EDPPJointTrace = d.buildJointTrace(ec, state, decodeSnaps, prefillSnaps, best)
	}
	return dec
}

// jointDecodeAdmissionCtx builds the decode-pool AdmissionContext the joint rule feeds to the
// t_adm estimator for candidate decode instance ds. Factored out of jointCandidateCost /
// jointVaRComponents so the scoring paths and the --edpp-admission-trace capture below the
// argmin cannot drift apart; the field values are unchanged (INV-6).
func (d *EDPPDecider) jointDecodeAdmissionCtx(ec *jointEvalCtx, ds RoutingSnapshot) AdmissionContext {
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	_, qdRaw := d.instWorkRaw(ds.ID)
	return AdmissionContext{
		QWork: qdRaw, Mu: thetaD.muDecode(bDec, kv, sPfD),
		BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
		FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: thetaD.tIterDecode(bDec, kv, sPfD), QueueDepth: ds.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(ds), RemainingStepsEst: d.decodeRemStepsEst(ds, ec.req.SLOClass),
		Running: censorOracleRemaining(ds.RunningDecode),
	}
}

// jointPrefillAdmissionCtx is the prefill-pool counterpart of jointDecodeAdmissionCtx.
func (d *EDPPDecider) jointPrefillAdmissionCtx(ec *jointEvalCtx, ps RoutingSnapshot) AdmissionContext {
	thetaP := d.coeffsFor(ps.GPUType)
	qpRaw, _ := d.instWorkRaw(ps.ID)
	sPfP := ps.ResidentPrefillTokens
	return AdmissionContext{
		QWork: qpRaw, Mu: thetaP.muPrefill(sPfP),
		BatchSize: ps.BatchSize, MaxBatchSize: int(ps.MaxBatchSize),
		FreeKVBlocks: ps.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: thetaP.tIterPrefill(sPfP), QueueDepth: ps.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(ps), RemainingStepsEst: d.prefillRemStepsEst(ps),
		Running: ps.RunningPrefill,
	}
}

// cand is one enumerated joint (decode, placement) candidate: local (prefill co-resident
// on the decode node) or disagg (decode on dID, prefill on pID), with its objective J.
type cand struct {
	dID, pID string
	local    bool
	J        float64
}

// jointEvalCtx holds the per-decision, candidate-invariant terms shared by every (d,·)
// cost evaluation in one joint decision, so jointCandidateCost can be called both from the
// argmin enumeration and from the scorer-slice shadow evaluation (divergence trace) without
// recomputing them — and so both paths produce byte-identical arithmetic (INV-6).
type jointEvalCtx struct {
	req       *Request
	n         edppNorm
	zTTFT     float64
	zITL      float64
	reqKVNeed int64
	nHatOut   float64 // per-class realized output-length estimate; wd/mDec are now per-candidate (θ_i)
	nowUs     float64 // decision instant (µs), state.Clock; used only by the VaR rule's completion model
}

// jointCandidateCost evaluates the normalized joint objective J(d, ·) for one candidate:
// local (ps == nil, prefill co-resident on the decode node ds) or disagg (decode on ds,
// prefill on *ps). It reproduces exactly the arithmetic the argmin enumeration uses, so the
// enumeration and the scorer-slice shadow evaluation share one code path. The decode-side
// terms depend only on ds and are recomputed per call with identical operands (byte-identical
// float result, INV-6).
// jointSelfGood returns the arriving request's OWN smoothed goodput under a candidate
// (goodput-objective diagnostic, VarGoodputObjective). tHat is the candidate's projected
// time-to-first-token (tHatLocal or tHatDisagg — the same value the self term divides by τ_ttft),
// and tIterAfter is the B+1 re-timed decode per-iter time the VaR completion model uses for the
// batch the request joins. The prefill chunk does not affect tIterAfter (only the overlap term),
// so it is passed as 0. Decode always happens on ds, so ds's θ_i sets tIterAfter for both local
// and disagg. Mirrors jointCandidateCost's operands (byte-identical arithmetic, INV-6).
func (d *EDPPDecider) jointSelfGood(ec *jointEvalCtx, thetaD EDPPCoeffs, ds RoutingSnapshot, tHat float64) float64 {
	rt := d.varReTimingFor(ec.req, thetaD, ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens, 0)
	return goodSelf(d.varSLOFor(ec.req.SLOClass), tHat, rt.tIterAfter, ec.nHatOut, d.varMetric)
}

func (d *EDPPDecider) jointCandidateCost(ec *jointEvalCtx, ds RoutingSnapshot, ps *RoutingSnapshot) float64 {
	n := ec.n
	// Decode-side θ_i: the physics of the decode node ds (fast vs slow ranks correctly).
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	// Per-candidate decode work W_d and base decode-step ITL marginal m_dec = δ̄_dec at mean
	// context (design §3 z_itl term), both under this candidate's θ_i. Decode happens on d in
	// both local and disagg, so jDecodeITL is added to both.
	wd := thetaD.Wd(len(ec.req.InputTokens), ec.nHatOut)
	mDec := thetaD.deltaBarDecode(float64(len(ec.req.InputTokens)) + ec.nHatOut/2)
	jDecodeITL := ec.zITL * (mDec / n.tauITL)
	_, qdRaw := d.instWorkRaw(ds.ID)
	// W*_i is PER-INSTANCE (paper: W*_i = mu_nom,i * tau_ttft). The placed work wd is already
	// evaluated under this candidate's theta_i, so dividing every candidate by one shared W* would
	// charge a slow instance ~N x more for an identical request even when it is idle, and the rule
	// herds onto the fast instance. Both the backlog and the placed work use this instance's W*_i.
	wStarD := thetaD.muDNom(n.tauITL) * n.tauTTFT
	qd := qdRaw / wStarD
	decodeCtx := d.jointDecodeAdmissionCtx(ec, ds)
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)

	// Decode backlog term (same for local and disagg on this d — cancels within a d,
	// distinguishes across d): q_d·(W_d/W*_d). The per-candidate W_d makes the fast node's
	// smaller demand lower its J.
	jDecodeBacklog := qd * (wd / wStarD)

	if ps == nil {
		// --- local: prefill+decode co-resident on d ⇒ prefill uses the decode θ_i ---
		apLoc := d.apForInstance(ec.req, ds.ID)
		nChunksLoc, deltaPfLoc := d.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), len(ec.req.InputTokens))
		// T̂_local: admission + batch-iteration overhead (nChunks·tIter) + the request's OWN
		// prefill work Wp (projection AND attention over context). Wp replaces the projection-only
		// nChunks·deltaPf so the estimate keeps the quadratic attention cost (matches the executor).
		tHatLocal := tAdmD + nChunksLoc*tIterD + wpLoc // ABSOLUTE T̂_local(d)
		// balance = the work-currency backlog contribution, or (Rule=="var") the value-currency
		// externality VaR_local on ds's decode co-residents. Self terms below are unchanged.
		balance := jDecodeBacklog + qd*(wpLoc/wStarD)
		if d.rule == "var" {
			v := d.varJointCandidateExternality(ec.req, ec.nowUs, ds, nil)
			if d.varGoodputObjective {
				v -= d.jointSelfGood(ec, thetaD, ds, tHatLocal) // goodput objective: VaR − good_r (−Δgood)
			}
			if d.varKeepCongestion {
				balance = d.varCongestionWeight*balance + v // drift-plus-VaR: weighted congestion + value externality
			} else {
				balance = v // pure VaR: the externality replaces the work-currency backlog
			}
		}
		return balance +
			ec.zTTFT*(tHatLocal/n.tauTTFT) +
			jDecodeITL + // base decode-step ITL marginal m_dec (per-candidate θ_i)
			ec.zITL*(deltaPfLoc/n.tauITL) // prefill-on-decode ITL inflation lands on local only
	}

	// --- disagg: decode on d, prefill on node *ps ⇒ prefill uses the prefill node's θ_i ---
	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(ec.req, ps.ID)
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), len(ec.req.InputTokens))
	qpRaw, _ := d.instWorkRaw(ps.ID)
	wStarP := thetaP.muPNom(d.cfg.NomPrefillTokens) * n.tauTTFT // per-instance W*_i, prefill side
	qp := qpRaw / wStarP
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)
	prefillCtx := d.jointPrefillAdmissionCtx(ec, *ps)
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	cXferUs := d.cXferUsFor(ec.req) // flat CXferUs, or the size-aware transfer cost
	// T̂_disagg: admission + prefill-pool iteration overhead + the request's OWN prefill work Wp
	// (projection AND attention, replacing the projection-only nChunks·deltaPf) + transfer time.
	tHatDisagg := tAdmP + nChunksP*tIterP + wpP + cXferUs // ABSOLUTE T̂_disagg(d,p)
	// balance = the work-currency backlog contribution, or (Rule=="var") the value-currency
	// externality VaR_disagg on ds's decode co-residents + *ps's prefill co-residents.
	balance := jDecodeBacklog + qp*(wpP/wStarP)
	if d.rule == "var" {
		v := d.varJointCandidateExternality(ec.req, ec.nowUs, ds, ps)
		if d.varGoodputObjective {
			v -= d.jointSelfGood(ec, thetaD, ds, tHatDisagg) // goodput objective: VaR − good_r (−Δgood)
		}
		if d.varKeepCongestion {
			balance = d.varCongestionWeight*balance + v // drift-plus-VaR: weighted congestion + value externality
		} else {
			balance = v // pure VaR: the externality replaces the work-currency backlog
		}
	}
	// The KV-transfer penalty is dropped under the goodput objective: its effect already flows
	// through the request's own projected TTFT (cXfer is inside tHatDisagg above), so charging it
	// again as a standalone term double-prices the transfer. Pricing it separately is the eq. (16)
	// "minimize transfer cost" residue the goodput reframing removes.
	xfer := 0.0
	if !(d.rule == "var" && d.varGoodputObjective) {
		xfer = d.transferPenalty(n, cXferUs) // disagg pays the KV-transfer penalty; no local ITL inflation
	}
	return balance +
		ec.zTTFT*(tHatDisagg/n.tauTTFT) +
		jDecodeITL + // base decode-step ITL marginal m_dec (same as local: decode is on d)
		xfer
}

// jointCandidateTTFT is the least-TTFT-joint objective (Rule=="least-ttft" && Joint): the
// deciding request's OWN forward time-to-first-token for one candidate, local (ps == nil) or
// disagg (decode on ds, prefill on *ps). It carries no backlog/balance drift, no SLO virtual
// queues, no VaR externality, and no transfer penalty beyond the transfer latency already inside
// the disagg TTFT. Each side uses its candidate's θ_i, so the arm is hardware-aware by
// construction (the fair least-TTFT the reviewer asks for). The arithmetic reproduces exactly
// jointCandidateCost's tHatLocal/tHatDisagg and the reduced rule's ttftD/ttftP (INV-6), so
// restricted to a single decode instance it matches the reduced least-ttft decision.
func (d *EDPPDecider) jointCandidateTTFT(ec *jointEvalCtx, ds RoutingSnapshot, ps *RoutingSnapshot) float64 {
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)

	if ps == nil {
		// --- local: prefill+decode co-resident on d ⇒ prefill uses the decode θ_i ---
		_, qdRaw := d.instWorkRaw(ds.ID)
		decodeCtx := AdmissionContext{
			QWork: qdRaw, Mu: thetaD.muDecode(bDec, kv, sPfD),
			BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
			FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
			TIter: tIterD, QueueDepth: ds.QueueDepth,
			AdmissionRate: admissionRateFromSnapshot(ds), RemainingStepsEst: d.decodeRemStepsEst(ds, ec.req.SLOClass),
			Running: censorOracleRemaining(ds.RunningDecode),
		}
		tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)
		apLoc := d.apForInstance(ec.req, ds.ID)
		nChunksLoc, _ := d.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), len(ec.req.InputTokens))
		return tAdmD + nChunksLoc*tIterD + wpLoc // T̂_local(d)
	}

	// --- disagg: decode on d, prefill on node *ps ⇒ prefill uses the prefill node's θ_i ---
	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(ec.req, ps.ID)
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), len(ec.req.InputTokens))
	tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
	tAdmP := d.tadmEstimator.EstimateTAdm(d.jointPrefillAdmissionCtx(ec, *ps))
	cXferUs := d.cXferUsFor(ec.req)
	return tAdmP + nChunksP*tIterP + wpP + cXferUs // T̂_disagg(d,p)
}

// jointVaRComponents evaluates one joint candidate for the AUTO-NORMALIZED drift-plus-VaR path,
// returning the three pieces separately so decideJoint's first pass can min-max normalize the
// congestion and VaR terms across candidates before combining them:
//
//   - cong: the work-currency congestion drift (decode backlog + prefill-work load)
//   - vv:   the value-at-risk externality (goodput destroyed among co-residents)
//   - self: the z-weighted TTFT/ITL self terms and the transfer penalty (kept as-is)
//
// It mirrors jointCandidateCost's arithmetic. It runs ONLY on the normalized dpVaR path, so it
// has no golden to match; the un-normalized paths are untouched (byte-identical, INV-6).
func (d *EDPPDecider) jointVaRComponents(ec *jointEvalCtx, ds RoutingSnapshot, ps *RoutingSnapshot) (cong, vv, self float64) {
	n := ec.n
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	wd := thetaD.Wd(len(ec.req.InputTokens), ec.nHatOut)
	mDec := thetaD.deltaBarDecode(float64(len(ec.req.InputTokens)) + ec.nHatOut/2)
	jDecodeITL := ec.zITL * (mDec / n.tauITL)
	_, qdRaw := d.instWorkRaw(ds.ID)
	// W*_i is PER-INSTANCE (paper: W*_i = mu_nom,i * tau_ttft). The placed work wd is already
	// evaluated under this candidate's theta_i, so dividing every candidate by one shared W* would
	// charge a slow instance ~N x more for an identical request even when it is idle, and the rule
	// herds onto the fast instance. Both the backlog and the placed work use this instance's W*_i.
	wStarD := thetaD.muDNom(n.tauITL) * n.tauTTFT
	qd := qdRaw / wStarD
	decodeCtx := d.jointDecodeAdmissionCtx(ec, ds)
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)
	jDecodeBacklog := qd * (wd / wStarD)

	if ps == nil {
		apLoc := d.apForInstance(ec.req, ds.ID)
		nChunksLoc, deltaPfLoc := d.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), len(ec.req.InputTokens))
		tHatLocal := tAdmD + nChunksLoc*tIterD + wpLoc
		cong = jDecodeBacklog + qd*(wpLoc/wStarD)
		vv = d.varJointCandidateExternality(ec.req, ec.nowUs, ds, nil)
		if d.varGoodputObjective {
			vv -= d.jointSelfGood(ec, thetaD, ds, tHatLocal) // normalize VaR − good_r as one penalty
		}
		self = ec.zTTFT*(tHatLocal/n.tauTTFT) + jDecodeITL + ec.zITL*(deltaPfLoc/n.tauITL)
		return cong, vv, self
	}

	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(ec.req, ps.ID)
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), len(ec.req.InputTokens))
	qpRaw, _ := d.instWorkRaw(ps.ID)
	wStarP := thetaP.muPNom(d.cfg.NomPrefillTokens) * n.tauTTFT // per-instance W*_i, prefill side
	qp := qpRaw / wStarP
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)
	prefillCtx := d.jointPrefillAdmissionCtx(ec, *ps)
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	cXferUs := d.cXferUsFor(ec.req)
	tHatDisagg := tAdmP + nChunksP*tIterP + wpP + cXferUs
	cong = jDecodeBacklog + qp*(wpP/wStarP)
	vv = d.varJointCandidateExternality(ec.req, ec.nowUs, ds, ps)
	xfer := d.transferPenalty(n, cXferUs)
	if d.varGoodputObjective {
		vv -= d.jointSelfGood(ec, thetaD, ds, tHatDisagg) // normalize VaR − good_r as one penalty
		xfer = 0                                          // goodput objective: transfer cost flows through tHatDisagg's z_TTFT term only
	}
	self = ec.zTTFT*(tHatDisagg/n.tauTTFT) + jDecodeITL + xfer
	return cong, vv, self
}

// varNormFloor returns the spread floor ε₀ for the per-decision min-max normalization. It is one
// arriving request's work on the nominal decode instance (the first decode snapshot) in reference
// units, ε₀ = scale·(Wp+Wd)/W*_d, so the congestion term binds only once backlogs differ by more
// than a request's worth of work; below that the spread is treated as noise and compressed out. The
// scale (VarNormalizeFloorScale, default 1) is the knob the sensitivity study sweeps. Always
// positive so it doubles as the division guard. See jointNormalizedCandidates.
func (d *EDPPDecider) varNormFloor(ec *jointEvalCtx, decodeSnaps []RoutingSnapshot) float64 {
	const tiny = 1e-12
	if len(decodeSnaps) == 0 || ec.n.wStarD <= 0 {
		return tiny
	}
	nom := decodeSnaps[0]
	theta := d.coeffsFor(nom.GPUType)
	wd := theta.Wd(len(ec.req.InputTokens), ec.nHatOut)
	ap := d.apForInstance(ec.req, nom.ID)
	wp := theta.Wp(maxInt(ap, 0), len(ec.req.InputTokens))
	eps0 := d.varNormalizeFloorScale * (wd + wp) / ec.n.wStarD
	if eps0 < tiny {
		return tiny
	}
	return eps0
}

// jointNormalizedCandidates enumerates the joint candidates for the auto-normalized dpVaR path
// (two passes): first it computes each candidate's (congestion, VaR, self) via jointVaRComponents
// and finds the min/max of the congestion and VaR terms across the set; then it builds each
// candidate's objective J = w·norm(congestion) + norm(VaR) + self, where
// norm(x) = (x−min)/max{max−min, ε₀} floors the min-max denominator at ε₀ (varNormFloor). On
// identical hardware the congestion spread sits below ε₀, so the term is compressed out and VaR
// decides; on heterogeneous hardware it clears ε₀, so it reins in over-routing. Enumeration order
// (decode snapshots ascending, local before disagg) matches the un-normalized path (INV-6).
func (d *EDPPDecider) jointNormalizedCandidates(ec *jointEvalCtx, decodeSnaps, prefillSnaps []RoutingSnapshot) []cand {
	type comp struct {
		dID, pID       string
		local          bool
		cong, vv, self float64
	}
	comps := make([]comp, 0, len(decodeSnaps)*(len(prefillSnaps)+1))
	for _, ds := range decodeSnaps {
		c, v, s := d.jointVaRComponents(ec, ds, nil)
		comps = append(comps, comp{dID: ds.ID, local: true, cong: c, vv: v, self: s})
		for _, ps := range prefillSnaps {
			psCopy := ps
			c, v, s := d.jointVaRComponents(ec, ds, &psCopy)
			comps = append(comps, comp{dID: ds.ID, pID: ps.ID, local: false, cong: c, vv: v, self: s})
		}
	}
	if len(comps) == 0 {
		return nil
	}
	congMin, congMax := comps[0].cong, comps[0].cong
	varMin, varMax := comps[0].vv, comps[0].vv
	for _, c := range comps[1:] {
		congMin, congMax = math.Min(congMin, c.cong), math.Max(congMax, c.cong)
		varMin, varMax = math.Min(varMin, c.vv), math.Max(varMax, c.vv)
	}
	// norm maps x to [0,1] across the candidate set by min-max, dividing by max{spread, ε₀}. The
	// floor ε₀ = scale·(dwork/W*) is one arriving request's work on the nominal decode instance in
	// reference units (dwork = Wp+Wd, the currency of the congestion term). When a term's spread
	// clears the floor the map is the plain min-max; when it falls below (identical hardware under a
	// balanced scorer, so the candidates carry near-equal backlog) the term is compressed toward zero
	// in proportion instead of amplified to the full unit range, so it fades out and the VaR term
	// decides. A zero spread maps every candidate to 0. The floor is always positive, so it also
	// guards the division. See EDPPConfig.VarNormalize / VarNormalizeFloorScale.
	eps0 := d.varNormFloor(ec, decodeSnaps)
	norm := func(x, lo, hi float64) float64 {
		return (x - lo) / math.Max(hi-lo, eps0)
	}
	out := make([]cand, 0, len(comps))
	for _, c := range comps {
		j := d.varCongestionWeight*norm(c.cong, congMin, congMax) + norm(c.vv, varMin, varMax) + c.self
		out = append(out, cand{dID: c.dID, pID: c.pID, local: c.local, J: j})
	}
	return out
}

// buildJointTrace assembles the scorer-vs-joint divergence record for a committed joint
// decision (best). It reads the decode scorer's pick from state.SelectedInstance and, on a
// disaggregate decision, SHADOW-runs the injected prefill scorer for scorer_p — compute-only,
// never acted on. J_scorer is J at the scorer's slice (disagg (scorer_d, scorer_p) on a
// disagg decision, else local on scorer_d); since the argmin ranges over a superset that
// includes that slice, J_joint <= J_scorer always (asserted in tests as an internal invariant).
func (d *EDPPDecider) buildJointTrace(ec *jointEvalCtx, state *RouterState, decodeSnaps, prefillSnaps []RoutingSnapshot, best *cand) *EDPPJointDecisionTrace {
	tr := &EDPPJointDecisionTrace{
		Class:        ec.req.SLOClass,
		JointD:       best.dID,
		JJoint:       best.J,
		Disaggregate: !best.local,
	}
	if !best.local {
		tr.JointP = best.pID
	}

	// Scorer's decode pick (state.SelectedInstance); resolve its snapshot for the J eval,
	// falling back to the first candidate when the pre-selection is absent/unknown.
	if state != nil {
		tr.ScorerD = state.SelectedInstance
	}
	tr.AgreeD = tr.ScorerD == tr.JointD
	scorerDSnap, ok := findSnapshotByID(decodeSnaps, tr.ScorerD)
	if !ok {
		scorerDSnap = decodeSnaps[0] // decodeSnaps is non-empty here (best != nil)
	}

	// Score the scorer's slice with the SAME objective the argmin used, so J_joint (best.J) and
	// J_scorer share a scale and the J_joint <= J_scorer invariant holds under least-TTFT too.
	costFn := d.jointCandidateCost
	if d.rule == "least-ttft" {
		costFn = d.jointCandidateTTFT
	}

	if best.local {
		tr.AgreeP = true // both prefill nodes empty ⇒ trivially agree
		tr.JScorer = costFn(ec, scorerDSnap, nil)
		return tr
	}

	// Disagg decision: shadow-run the prefill scorer for scorer_p (logging-only, INV-6).
	if d.prefillScorer != nil {
		tr.ScorerP = d.prefillScorer(ec.req, prefillSnaps)
	}
	tr.AgreeP = tr.ScorerP == tr.JointP
	if sp, ok := findSnapshotByID(prefillSnaps, tr.ScorerP); ok {
		tr.JScorer = costFn(ec, scorerDSnap, &sp)
	} else {
		// No usable shadow prefill pick ⇒ score the scorer's local slice (still an
		// enumerated candidate, so the J_joint <= J_scorer invariant is preserved).
		tr.JScorer = costFn(ec, scorerDSnap, nil)
	}
	return tr
}

// findSnapshotByID returns the snapshot with the given ID (ok=false when id is empty or absent).
func findSnapshotByID(snaps []RoutingSnapshot, id string) (RoutingSnapshot, bool) {
	if id == "" {
		return RoutingSnapshot{}, false
	}
	for _, s := range snaps {
		if s.ID == id {
			return s, true
		}
	}
	return RoutingSnapshot{}, false
}

// stateSnapshots returns state.Snapshots (nil-safe).
func stateSnapshots(state *RouterState) []RoutingSnapshot {
	if state == nil {
		return nil
	}
	return state.Snapshots
}

// sortedSnapshotsByID returns a copy of snaps sorted ascending by instance ID, so the
// joint enumeration order (and its deterministic tie-break) is stable across runs (INV-6).
func sortedSnapshotsByID(snaps []RoutingSnapshot) []RoutingSnapshot {
	if len(snaps) == 0 {
		return nil
	}
	out := make([]RoutingSnapshot, len(snaps))
	copy(out, snaps)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// selectedDecodeSnapshot returns the decode snapshot this request would land on
// (state.SelectedInstance), falling back to the first snapshot; ok is false when
// no snapshot is available (nominal-state path).
func (d *EDPPDecider) selectedDecodeSnapshot(state *RouterState) (snap RoutingSnapshot, ok bool) {
	if state == nil {
		return RoutingSnapshot{}, false
	}
	snaps := state.Snapshots
	if state.SelectedInstance != "" {
		for _, s := range snaps {
			if s.ID == state.SelectedInstance {
				return s, true
			}
		}
	}
	if len(snaps) > 0 {
		return snaps[0], true
	}
	return RoutingSnapshot{}, false
}

// censorOracleRemaining returns a deep copy of the running-request slice with every
// TrueRemaining censored to -1, so the runtime routing driver (the deployable
// admission-delay estimator) can never consume oracle output-length info via
// Running[].TrueRemaining (INV-9 on the control path). This mirrors the logging-path
// stripOracle in cluster.BuildAdmissionRecords. The deployable RemainingStepsEst is the
// only remaining-steps signal the routing estimators may read. The copy is required
// because the snapshot slice (e.g. decSnap.RunningDecode) is shared and read elsewhere;
// mutating it in place would corrupt the shared state. Returns nil for an empty input.
func censorOracleRemaining(running []RunningReqState) []RunningReqState {
	if len(running) == 0 {
		return nil
	}
	out := make([]RunningReqState, len(running))
	copy(out, running)
	for i := range out {
		out[i].TrueRemaining = -1
	}
	return out
}

// admissionRateFromSnapshot returns the admission rate (req/µs) for the little
// estimator. It prefers the explicit AdmissionRate field when populated, else
// derives it from the observed completion rate DispatchRate (req/s → req/µs);
// admission ≈ completion in steady state (§3.8). Returns 0 when neither is available.
func admissionRateFromSnapshot(snap RoutingSnapshot) float64 {
	if snap.AdmissionRate > 0 {
		return snap.AdmissionRate
	}
	if snap.DispatchRate > 0 {
		return snap.DispatchRate / 1e6
	}
	return 0
}

// selectedDecodeState returns (B_dec, KV, S_pf) for the decode pod this request
// would land on (state.SelectedInstance), falling back to the first decode
// snapshot, then to a nominal single-request decode batch.
func (d *EDPPDecider) selectedDecodeState(state *RouterState) (bDec int, kv, sPf int64) {
	if state != nil {
		snaps := state.Snapshots
		if state.SelectedInstance != "" {
			for _, s := range snaps {
				if s.ID == state.SelectedInstance {
					return s.BatchSize, s.KvTokensInUse, s.ResidentPrefillTokens
				}
			}
		}
		if len(snaps) > 0 {
			s := snaps[0]
			return s.BatchSize, s.KvTokensInUse, s.ResidentPrefillTokens
		}
	}
	return 1, int64(d.cfg.NomDecodeCtx), 0
}

// ensureZ returns the per-class virtual-queue state, lazily creating it.
func (d *EDPPDecider) ensureZ(class string) *edppClassState {
	z := d.zByClass[class]
	if z == nil {
		z = &edppClassState{}
		d.zByClass[class] = z
	}
	return z
}

// creditAwaiting credits z_ttft with the certain lower-bound TTFT miss of every in-flight
// request still awaiting its first token. For a request that arrived at startUs and has no
// first token by nowUs, max(nowUs−startUs−τ, 0) is a guaranteed lower bound on its eventual
// miss; we apply only the increment over what was already credited (so repeated sweeps do
// not double-count). Iteration is in sorted-key order so the floating-point accumulation is
// byte-identical across runs (INV-6). The full realized miss is reconciled at first token
// (OnFirstToken) or, as a fallback, at completion (OnComplete) — see the design's
// faithfulness invariant: same total contribution, credited earlier.
func (d *EDPPDecider) creditAwaiting(nowUs int64) {
	if len(d.awaitingFirstToken) == 0 {
		return
	}
	keys := make([]string, 0, len(d.awaitingFirstToken))
	for k := range d.awaitingFirstToken {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		rec := d.awaitingFirstToken[k]
		tauTTFTUs, _ := d.targetsFor(rec.class)
		lb := float64(nowUs - rec.startUs - tauTTFTUs)
		if lb < 0 {
			lb = 0
		}
		delta := lb - rec.credited
		if delta == 0 {
			continue
		}
		z := d.ensureZ(rec.class)
		z.zTTFT = math.Max(z.zTTFT+delta, 0)
		rec.credited = lb
	}
}

// OnFirstToken trues up z_ttft to the realized TTFT (nowUs − startUs) for a request that
// has just produced its first token, applying the signed delta over what creditAwaiting
// already credited (which may be negative when the SLO was met), then stops tracking it.
// Idempotent: a second call for the same key (e.g. the prefill sub-request of a PD request
// after the decode sub-request already fired) finds no record and no-ops.
func (d *EDPPDecider) OnFirstToken(key string, nowUs int64) {
	rec, ok := d.awaitingFirstToken[key]
	if !ok {
		return
	}
	tauTTFTUs, _ := d.targetsFor(rec.class)
	target := float64(nowUs-rec.startUs) - float64(tauTTFTUs)
	z := d.ensureZ(rec.class)
	z.zTTFT = math.Max(z.zTTFT+(target-rec.credited), 0)
	delete(d.awaitingFirstToken, key)
}

// nHatFor returns the per-class running-mean output length, lazily created.
func (d *EDPPDecider) nHatFor(class string) *edppRunningMean {
	m := d.nHatOut[class]
	if m == nil {
		m = &edppRunningMean{}
		d.nHatOut[class] = m
	}
	return m
}

// reqNHatOut returns the output-length estimate used to charge THIS request's own
// decode work (its joint W_d and its qdWork backlog contribution). Deployable path:
// the per-class running-mean N̂_out. DIAGNOSTIC oracle path (cfg.OracleOutputLen,
// --edpp-oracle-output-len): the request's TRUE output length. This reads
// req.OutputTokens in the control plane — a deliberate INV-9 violation, gated behind
// the oracle flag and never a deployable policy (results are an upper bound). It is
// applied only to the request being routed/decided, never to co-residents.
func (d *EDPPDecider) reqNHatOut(req *Request) float64 {
	if d.cfg.OracleOutputLen {
		return float64(len(req.OutputTokens))
	}
	return d.nHatFor(req.SLOClass).mean()
}

// OnRoute increments the work backlog for a committed request (design §6.1,
// conservation form). apTokens is the uncached prompt token count (input-only; INV-9
// safe). W_d uses the class N̂_out estimate at the nominal decode context.
func (d *EDPPDecider) OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string) {
	// Track every routed request for the continuous z_ttft credit, regardless of apTokens
	// (a fully-cached request still has a TTFT and can still wait). The TTFT clock starts
	// at arrival.
	d.awaitingFirstToken[key] = &edppAwaiting{startUs: req.ArrivalTime, class: req.SLOClass}

	if apTokens <= 0 {
		return
	}
	wp := d.coeffs.Wp(apTokens, len(req.InputTokens))
	// W_d now uses the exact discrete decode sum Wd(a_r, N̂_out); it no longer uses NomDecodeCtx.
	// reqNHatOut yields the deployable N̂_out, or the TRUE o_r under the diagnostic oracle flag.
	wd := d.coeffs.Wd(len(req.InputTokens), d.reqNHatOut(req))
	pw := edppPendingWork{toPrefill: toPrefill, decodeInst: decodeInst, prefillInst: prefillInst}
	if toPrefill {
		pw.wp = wp // prefill work lands on the prefill pool
		pw.wd = wd // decode-only work lands on the decode pool
		d.qpWork += wp
		d.qdWork += wd
		// Mirror per-instance: prefill share → prefillInst, decode share → decodeInst.
		d.instWork(prefillInst).wp += wp
		d.instWork(decodeInst).wd += wd
	} else {
		pw.wd = wp + wd // mixed prefill+decode on the decode pool
		d.qdWork += wp + wd
		d.instWork(decodeInst).wd += wp + wd
	}
	d.pending[key] = pw
}

// instWork returns (creating if needed) the per-instance work accumulator for id.
// id == "" is ignored and returns a throwaway accumulator (nil-safe), so an unknown
// instance never panics and never pollutes the map with a "" key.
func (d *EDPPDecider) instWork(id string) *edppInstWork {
	if id == "" {
		return &edppInstWork{}
	}
	w, ok := d.qByInstance[id]
	if !ok {
		w = &edppInstWork{}
		d.qByInstance[id] = w
	}
	return w
}

// drainInst decrements the per-instance accumulator by (wp, wd), mirroring the
// pool-scalar drain, and 0-clamps like the pool path. id == "" or an unknown id is
// a no-op (the work was never attributed to a live instance). It never resurrects a
// deleted key beyond what instWork already created at OnRoute.
func (d *EDPPDecider) drainInst(id string, wp, wd float64) {
	if id == "" {
		return
	}
	w, ok := d.qByInstance[id]
	if !ok {
		return
	}
	w.wp -= wp
	w.wd -= wd
	if w.wp < 0 {
		w.wp = 0
	}
	if w.wd < 0 {
		w.wd = 0
	}
}

// QByInstance returns a snapshot copy of the per-instance congestion queues Q_i
// (waiting work µs, split into prefill Wp and decode Wd). For tests and the future
// joint P/D routing reader; the reduced (pool-level) path does not consume it.
func (d *EDPPDecider) QByInstance() map[string]struct{ Wp, Wd float64 } {
	out := make(map[string]struct{ Wp, Wd float64 }, len(d.qByInstance))
	for id, w := range d.qByInstance {
		out[id] = struct{ Wp, Wd float64 }{Wp: w.wp, Wd: w.wd}
	}
	return out
}

// OnAdmit removes the waiting-work share of a routed request that has just been
// admitted to a running batch (design §6.2, event-exact drain). prefillSide=true
// drains the Q_p share (prefill sub-request admitted to the prefill server);
// prefillSide=false drains the Q_d share (decode/normal request admitted to the
// decode server). pending[key] is deleted once both shares are gone. Idempotent:
// a re-admission (e.g. after preemption) finds the share already 0 and no-ops.
func (d *EDPPDecider) OnAdmit(key string, prefillSide bool) {
	pw, ok := d.pending[key]
	if !ok {
		return
	}
	if prefillSide {
		d.qpWork -= pw.wp
		d.drainInst(pw.prefillInst, pw.wp, 0) // mirror the wp decrement per-instance
		pw.wp = 0
	} else {
		d.qdWork -= pw.wd
		d.drainInst(pw.decodeInst, 0, pw.wd) // mirror the wd decrement per-instance
		pw.wd = 0
	}
	if d.qpWork < 0 {
		d.qpWork = 0
	}
	if d.qdWork < 0 {
		d.qdWork = 0
	}
	if pw.wp == 0 && pw.wd == 0 {
		delete(d.pending, key)
	} else {
		d.pending[key] = pw
	}
}

// OnComplete bumps the per-class virtual queues from realized latencies (E8) and
// updates N̂_out. It does NOT drain the waiting backlog — that work left q_p/q_d at
// admission (OnAdmit, §6.2). Reading realized output length here is post-completion
// and INV-9-permitted (§6.3).
func (d *EDPPDecider) OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64) {
	d.nHatFor(req.SLOClass).update(float64(len(req.OutputTokens)))
	tauTTFTUs, tauITLUs := d.targetsFor(req.SLOClass)
	z := d.ensureZ(req.SLOClass)
	z.zITL = math.Max(z.zITL+float64(realizedMeanITLUs-tauITLUs), 0)

	// z_ttft is normally owned by the continuous credit + OnFirstToken true-up. If the
	// first-token hook never fired for this request (record still present), fall back to
	// truing up from the realized TTFT here so the signal is never lost — same total
	// contribution as the pre-change completion-time bump.
	if rec, ok := d.awaitingFirstToken[key]; ok {
		target := float64(realizedTTFTUs) - float64(tauTTFTUs)
		z.zTTFT = math.Max(z.zTTFT+(target-rec.credited), 0)
		delete(d.awaitingFirstToken, key)
	}
}

// Forget releases the conservation backlog a routed request contributed when that
// request reaches a terminal state WITHOUT a normal completion (timeout/drop): it
// removes pending[key] and decrements qp/qd by the stored amounts (0-clamped). It
// deliberately does NOT bump the virtual queues z or touch N̂_out — a request that
// never completed carries no realized SLO signal and no realized output length. It
// is idempotent: forgetting an unknown key is a no-op. This is the conservation
// counterpart to OnComplete for the non-completion terminal paths (design §6).
func (d *EDPPDecider) Forget(key string) {
	pw, ok := d.pending[key]
	if !ok {
		return
	}
	d.qpWork -= pw.wp
	d.qdWork -= pw.wd
	d.drainInst(pw.prefillInst, pw.wp, 0) // mirror wp per-instance
	d.drainInst(pw.decodeInst, 0, pw.wd)  // mirror wd per-instance
	if d.qpWork < 0 {
		d.qpWork = 0
	}
	if d.qdWork < 0 {
		d.qdWork = 0
	}
	delete(d.pending, key)
	// Keep any z_ttft credit already applied: a request that waited past its target and
	// then dropped/timed-out is a real SLO failure (a censored observation), not a
	// non-event. Just stop tracking it for further credit.
	delete(d.awaitingFirstToken, key)
}

// BacklogForTest exposes the conservation bookkeeping state for tests only: the
// running prefill/decode work backlogs and the number of still-pending routed
// requests. It lets cluster-package tests assert observable conservation (Q→0,
// pending empty) without reaching into unexported fields. Not part of any contract;
// do not use outside tests.
func (d *EDPPDecider) BacklogForTest() (qp, qd float64, pendingLen int) {
	return d.qpWork, d.qdWork, len(d.pending)
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)
