"""Shared accounting for the config-attribution study.

Runs BLIS with the study ledger on (BLIS_STUDY_TERM_LEDGER) and turns one run into
per-metric breakdowns whose pieces sum exactly to the metric BLIS reports.
"""
import json, os, re, statistics as st, subprocess

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLIS = os.path.join(WORKTREE, "blis")
CATALOG = os.environ.get("BLIS_CATALOG", os.path.abspath(os.path.join(WORKTREE, "..", "..", "..", "blis-catalog")))

# StepTime's named terms, in the ledger's order, and the component each one rolls up into.
NAMES = ["prefill_compute", "prefill_kv", "decode_compute", "decode_kv", "weight_load",
         "tp_allreduce", "moe_dispatch", "per_layer_overhead", "per_request_overhead",
         "per_step_overhead", "moe_interleave_overhead", "adapter_overhead"]
GROUP = {"prefill_compute": "compute", "decode_compute": "compute", "weight_load": "weights",
         "prefill_kv": "kv", "decode_kv": "kv", "tp_allreduce": "comm", "moe_dispatch": "comm"}

# (key, label, colour) in stacking order (bottom first). Colours are the validated categorical order.
COMPONENTS = [
    ("weights", "Weight reads", "#2a78d6"),
    ("kv", "KV-cache reads/writes", "#eb6834"),
    ("compute", "Compute", "#1baf7a"),
    ("comm", "TP communication", "#eda100"),
    ("overhead", "Fixed per-step overhead", "#e87ba4"),
    ("api", "API overhead", "#008300"),
    ("wait", "Waiting for a batch slot", "#4a3aa7"),
    ("preempt", "Work discarded by preemption", "#e34948"),
    # Cost only: GPUs allocated but with no step to run. Neutral grey + hatch (no ninth hue).
    ("idle", "GPU idle (nothing to run)", "#c9ced6"),
]
CATS = [k for k, _, _ in COMPONENTS]

METRICS = {  # key -> (title, unit)
    "TTFT": ("Time to first token", "ms"),
    "ITL": ("Inter-token latency", "ms"),
    "E2E": ("End-to-end latency", "ms"),
    "COST": ("Paid GPU-time per request", "GPU-ms"),
}


def alphas_ms():
    """α₀, α₁, α₂ from defaults.yaml, truncated to whole µs as BLIS applies them (clampToInt64)."""
    txt = open(os.path.join(WORKTREE, "defaults.yaml")).read()
    vals = re.search(r"alpha_coeffs:\s*\[([^\]]+)\]", txt).group(1).split(",")
    return [int(float(v)) / 1e3 for v in vals[:3]]


def _grp(vec, scale):
    out = {c: 0.0 for c in CATS}
    for n, v in zip(NAMES, vec):
        out[GROUP.get(n, "overhead")] += v * scale
    return out


def run(flags, tag, outdir, shadow=None):
    """shadow: optional {"tp": n} and/or {"hardware": name}: also price every executed batch
    under that config. Returns the breakdown, plus out["counterfactual"] when shadow is set."""
    mns = int(flags[flags.index("--max-num-seqs") + 1]) if "--max-num-seqs" in flags else 256  # BLIS default
    """Run `blis run <flags>` with the ledger on; return the per-metric breakdown dict."""
    os.makedirs(outdir, exist_ok=True)
    led, met = os.path.join(outdir, f"{tag}.ledger.json"), os.path.join(outdir, f"{tag}.metrics.json")
    env = dict(os.environ, BLIS_CATALOG=CATALOG, BLIS_STUDY_TERM_LEDGER=led)
    for k, var in (("tp", "BLIS_STUDY_SHADOW_TP"), ("hardware", "BLIS_STUDY_SHADOW_HARDWARE")):
        env.pop(var, None)
        if shadow and k in shadow:
            env[var] = str(shadow[k])
    p = subprocess.run([BLIS, "run", *flags, "--metrics-path", met], cwd=WORKTREE, env=env,
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"blis run failed for {tag}:\n" + "\n".join(p.stderr.strip().splitlines()[-5:]))
    L, M = json.load(open(led)), json.load(open(met))
    out = breakdown(L, M, int(flags[flags.index("--tp") + 1]))
    if shadow:
        out["counterfactual"] = counterfactual(L, M, int(flags[flags.index("--tp") + 1]), L["shadow_tp"])
    out["meta"]["max_num_seqs"] = mns
    out["meta"]["batch_fill"] = out["meta"]["batch"] / mns
    return out


def breakdown(L, M, tp):
    A0, A1, A2 = alphas_ms()
    reqs = [r for r in M["requests"] if r["e2e_ms"] > 0 and r["num_decode_tokens"] > 1]
    if not reqs:
        raise SystemExit("no completed multi-token requests in this run")
    T, I, E, S = [], [], [], []
    for r in reqs:
        pr = L["per_request"][r["requestID"]]
        n = r["num_decode_tokens"]
        # TTFT (to the first token of the final attempt) = α0 + waiting + work discarded by
        # preemption + final-attempt prefill steps + α2 for the first token.
        t = _grp(pr["prefill_us"], 1e-3)
        t["preempt"] = sum(pr["discarded_us"]) * 1e-3
        t["api"] = A0 + A2
        t["wait"] = r["ttft_ms"] - sum(t[c] for c in CATS if c != "wait")
        # Decode phase = final-attempt decode steps + α2 per later token.
        d = _grp(pr["decode_us"], 1e-3)
        d["api"] = A2 * (n - 1)
        resid = r["e2e_ms"] - r["ttft_ms"] - A1 - sum(d.values())
        if abs(resid) > 0.01:
            raise SystemExit(f"decode-phase pieces miss E2E by {resid:.3f} ms for {r['requestID']}")
        e = {c: t[c] + d[c] for c in CATS}
        e["api"] += A1
        T.append(t); I.append({c: d[c] / (n - 1) for c in CATS}); E.append(e)
        # Per request in the batch: this request's even share of each decode step it was in.
        sh = _grp(pr["decode_share_us"], 1e-3)
        sh["api"] = A2
        S.append({c: sh[c] / (n - 1) if c != "api" else sh[c] for c in CATS})
    mean = lambda rows: {c: st.mean(x[c] for x in rows) for c in CATS}
    # GPU-time per request = GPU-time per output token × output tokens per request
    #                      = GPU-busy time × GPUs ÷ completed requests (GPU-ms; compares with E2E).
    # Paid GPU-time also counts the time the GPUs sat allocated with nothing to run:
    # wall-clock (sim end) minus busy time, × GPUs ÷ requests.
    done = M["completed_requests"]
    wall_us = M["vllm_estimated_duration_s"] * 1e6
    cost = _grp([L["term_us"][k] for k in NAMES], tp / done / 1e3)
    cost["idle"] = max(0.0, wall_us - L["busy_us"]) * tp / done / 1e3
    out = {"TTFT": mean(T), "ITL": mean(I), "E2E": mean(E), "COST": cost, "ITL_SHARE": mean(S)}
    reported = {"TTFT": st.mean(r["ttft_ms"] for r in reqs), "ITL": st.mean(r["itl_ms"] for r in reqs),
                "E2E": st.mean(r["e2e_ms"] for r in reqs), "COST": max(wall_us, L["busy_us"]) * tp / done / 1e3}
    for m, v in reported.items():
        s = sum(out[m].values())
        if abs(s - v) > max(0.01, 1e-3 * abs(v)):
            raise SystemExit(f"{m} pieces sum to {s:.3f}, BLIS reports {v:.3f}")
    out["meta"] = {"batch": sum(int(k) * v for k, v in L["batch_size_hist"].items()) / L["steps"],
                   "preemptions": M.get("preemption_count", 0), "tok_per_s": M["tokens_per_sec"],
                   "output_tokens": M["total_output_tokens"], "completed": done,
                   "injected": M["injected_requests"], "busy_frac": min(1.0, L["busy_us"] / wall_us)}
    return out


def counterfactual(L, M, tp, shadow_tp):
    """This run's exact schedule priced under the shadow config.

    Waiting (queueing) is the schedule's own and stays as it was; only the step pieces are
    re-priced. Paid cost keeps this run's idle wall-time, on the shadow config's GPU count."""
    A0, A1, A2 = alphas_ms()
    reqs = [r for r in M["requests"] if r["e2e_ms"] > 0 and r["num_decode_tokens"] > 1]
    T, I, E = [], [], []
    for r in reqs:
        real, sh, n = L["per_request"][r["requestID"]], L["shadow_per_request"][r["requestID"]], r["num_decode_tokens"]
        real_steps = (sum(real["prefill_us"]) + sum(real["discarded_us"])) * 1e-3
        wait = r["ttft_ms"] - (A0 + A2) - real_steps        # the schedule's queueing, unchanged
        t = _grp(sh["prefill_us"], 1e-3)
        t["preempt"] = sum(sh["discarded_us"]) * 1e-3
        t["api"], t["wait"] = A0 + A2, wait
        d = _grp(sh["decode_us"], 1e-3)
        d["api"] = A2 * (n - 1)
        e = {c: t[c] + d[c] for c in CATS}
        e["api"] += A1
        T.append(t); I.append({c: d[c] / (n - 1) for c in CATS}); E.append(e)
    mean = lambda rows: {c: st.mean(x[c] for x in rows) for c in CATS}
    done = M["completed_requests"]
    wall_us = M["vllm_estimated_duration_s"] * 1e6
    cost = _grp([L["shadow_term_us"][k] for k in NAMES], shadow_tp / done / 1e3)
    cost["idle"] = max(0.0, wall_us - L["busy_us"]) * shadow_tp / done / 1e3
    return {"TTFT": mean(T), "ITL": mean(I), "E2E": mean(E), "COST": cost}
