"""Layer 2 prototype: vary one BLIS knob and see which latency component moved.

Example (run from anywhere; BLIS runs from the worktree root):
  python3 study/config-attribution/scripts/compare.py \
    --model meta-llama/llama-3.1-70b-instruct --hardware H100 --tp 4 \
    --workload chatbot --rate 32 --vary tp=2,4,8

--vary takes any `blis run` flag name and a comma list of values; the first value is the
baseline. Extra BLIS flags go after `--`. Writes one PNG per metric plus summary.md to
charts/compare-<name>/ (raw run files go to runs/<name>/).
"""
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attrib

STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INK, INK2, INK3, GRID, BG = "#14161a", "#4b5059", "#737a85", "#dfe2e7", "#fcfcfb"
LABEL = {k: n for k, n, _ in attrib.COMPONENTS}
COLOR = {k: c for k, _, c in attrib.COMPONENTS}
COLOR["other"] = "#b8bdc6"
HATCH = {"idle": "///"}
REL_MIN_SHARE = 0.02  # relative change shown only for pieces >= 2% of the larger total
MIN_CHANGE_SHARE = 0.05  # pieces contributing < 5% of the change are folded into "Other"


def fmt(v, unit, signed=False):
    s = "+" if signed and v > 0 else ("−" if v < 0 else "")
    a = abs(v)
    if unit in ("ms", "GPU-ms") and a >= 1000:
        return f"{s}{a / 1000:.2f} {'s' if unit == 'ms' else 'GPU-s'}"
    return f"{s}{a:.0f} {unit}" if a >= 100 else f"{s}{a:.1f} {unit}" if a >= 1 else f"{s}{a:.2f} {unit}"


def style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9aa0a8"); ax.spines["bottom"].set_color("#9aa0a8")
    ax.tick_params(colors=INK2)
    ax.set_axisbelow(True)


def compare(base, other, unit):
    """Per-component change of `other` vs `base`: (rows sorted by |Δ|, total Δ, total %)."""
    tb, to = sum(base.values()), sum(other.values())
    rows = []
    for c in attrib.CATS:
        b, o = base[c], other[c]
        d = o - b
        if abs(b) < 1e-9 and abs(o) < 1e-9:
            continue
        eligible = max(abs(b), abs(o)) >= REL_MIN_SHARE * max(tb, to)
        # A piece that was negligible in the baseline has no meaningful % change: call it "new".
        base_ok = abs(b) >= REL_MIN_SHARE * tb
        pct = (d / b * 100) if (eligible and base_ok) else None
        rows.append(dict(c=c, b=b, o=o, d=d, pct=pct, new=eligible and not base_ok))
    rows.sort(key=lambda r: -abs(r["d"]))
    # Denoise: fold pieces that account for < 5% of the change into one "Other" row, so the
    # bars still sum to the total. Reference is the larger of |total Δ| and the biggest piece Δ,
    # so pieces that cancel each other out don't all count as "large".
    ref = max([abs(to - tb)] + [abs(r["d"]) for r in rows])
    small = [r for r in rows if abs(r["d"]) < MIN_CHANGE_SHARE * ref]
    rows = [r for r in rows if r not in small]
    if small:
        rows.append(dict(c="other", n=len(small), members=[r["c"] for r in small], b=sum(r["b"] for r in small),
                         o=sum(r["o"] for r in small), d=sum(r["d"] for r in small), pct=None, new=False))
    return rows, to - tb, ((to - tb) / tb * 100 if tb else 0.0)


CAUSE_KNOBS = ("tp", "hardware")  # knobs that change what a step costs; others only change scheduling


def cause_split(real_b, real_v, cf_b_at_v, cf_v_at_b):
    """Split real_v − real_b per component into direct (same batches, new costs) and knock-on
    (different batches and queueing), averaged over both orders so neither config is privileged.

    Order 1 re-prices the baseline's batches at v's costs; order 2 re-prices v's batches at the
    baseline's costs. direct + knock-on = the layer-2 change exactly, by construction."""
    d1 = {c: cf_b_at_v[c] - real_b[c] for c in attrib.CATS}
    d2 = {c: real_v[c] - cf_v_at_b[c] for c in attrib.CATS}
    direct = {c: (d1[c] + d2[c]) / 2 for c in attrib.CATS}
    knock = {c: (real_v[c] - real_b[c]) - direct[c] for c in attrib.CATS}
    return direct, knock, sum(d1.values()), sum(d2.values())


def plot_metric(metric, values, results, knob, title_ctx, path, causes=None):
    title, unit = attrib.METRICS[metric]
    others = list(range(1, len(values)))
    ncol = 3 if causes else 2
    fig = plt.figure(figsize=(14 if ncol == 2 else 20, 4.6 + 3.3 * len(others)), facecolor=BG)
    gs = fig.add_gridspec(1 + len(others), ncol, height_ratios=[1.45] + [1] * len(others),
                          hspace=0.95, wspace=0.42 if ncol == 2 else 0.55)
    # Panel 1: absolute stacked breakdown, one bar per knob value. ITL also gets the
    # per-request-in-the-batch view beside it, which strips out the batch-size effect.
    tops = [(metric, f"{title}: breakdown by {knob}", f"{title} ({unit})")]
    if metric == "ITL":
        tops.append(("ITL_SHARE", "Per request in the batch (each step ÷ requests in it)", f"ms per request per token"))
    used = []
    for col, (key, ttl, ylab) in enumerate(tops):
        if len(tops) == 1:
            ax = fig.add_subplot(gs[0, :])
        else:
            ax = fig.add_subplot(gs[0, :ncol // 2 + (ncol % 2 and col == 0)] if col == 0 else gs[0, ncol // 2 + (ncol % 2):])
        style(ax)
        totals = [sum(results[i][key].values()) for i in range(len(values))]
        for i in range(len(values)):
            acc = 0.0
            for c in attrib.CATS:
                v = max(0.0, results[i][key][c])
                if v <= 0:
                    continue
                if c not in used:
                    used.append(c)
                ax.bar(i, v, bottom=acc, width=0.55, color=COLOR[c], edgecolor=BG, linewidth=1.5,
                       hatch=HATCH.get(c))
                acc += v
            if key == "COST":
                paid = totals[i]
                busy = paid - results[i][key].get("idle", 0.0)
                ax.text(i, paid * 1.02, f"busy {fmt(busy, unit)}\npaid {fmt(paid, unit)}\nutilization {busy / paid * 100:.0f}%",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK, linespacing=1.3)
            else:
                ax.text(i, totals[i] * 1.02, fmt(totals[i], unit), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks(range(len(values)), [f"{knob}={v}" + ("\n(baseline)" if i == 0 else "")
                                           for i, v in enumerate(values)], fontweight="bold", color=INK)
        ax.set_ylim(0, max(totals) * (1.38 if key == "COST" else 1.15))
        ax.set_ylabel(ylab, color=INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for i in range(len(values)):
            m = results[i]["meta"]
            pre = f"{m['preemptions']} preempt." if m["preemptions"] else "no preempt."
            ax.annotate(f"avg {m['batch']:.0f} req/step\n{pre}\n{m['completed']}/{m['injected']} requests done", (i, 0),
                        xytext=(0, -38), textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK3)
        ax.set_title(ttl, loc="left", fontsize=12.5 if col == 0 else 11, fontweight="bold", color=INK, pad=10)
    # Listed top-down in the same order the pieces stack, so legend and bars read alike.
    stacked = [c for c in attrib.CATS if c in used][::-1]
    ax.legend(handles=[Patch(facecolor=COLOR[c], edgecolor=BG, hatch=HATCH.get(c), label=LABEL[c]) for c in stacked], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9.5, borderaxespad=0)

    # Panels 2-3 per compared value: impact (ms) and relative change (%).
    lines = []
    for row, i in enumerate(others, start=1):
        rows, dt, dpct = compare(results[0][metric], results[i][metric], unit)
        name = f"{knob}={values[i]} vs {knob}={values[0]}"
        labels = ["Total"] + [f"Other ({r['n']} small)" if r["c"] == "other" else LABEL[r["c"]] for r in rows]
        ypos = range(len(labels))[::-1]
        # Impact in absolute units.
        ax = fig.add_subplot(gs[row, 0]); style(ax)
        dv = [dt] + [r["d"] for r in rows]
        cols = [INK2] + [COLOR[r["c"]] for r in rows]
        hat = [None] + [HATCH.get(r["c"]) for r in rows]
        for y, v, col, h in zip(ypos, dv, cols, hat):
            ax.barh(y, v, color=col, height=0.62, hatch=h, edgecolor=BG)
        span = max(abs(x) for x in dv) or 1
        for y, v in zip(ypos, dv):
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, fmt(v, unit, signed=True),
                    va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        ax.set_yticks(list(ypos), labels, fontsize=9.5, color=INK)
        ax.axvline(0, color="#9aa0a8", linewidth=1)
        ax.set_xlim(min(0, min(dv)) - (0.55 if min(dv) < 0 else 0.1) * span, max(0, max(dv)) + 0.35 * span)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        mb, mo = results[0]["meta"]["batch"], results[i]["meta"]["batch"]
        ax.set_title(f"Impact on total ({unit})", loc="left", fontsize=11, fontweight="bold", color=INK, pad=20)
        ax.text(0, 1.24, name, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
        ax.text(0, 1.03, f"avg requests per step: {mb:.0f} → {mo:.0f} ({(mo - mb) / mb * 100:+.0f}%)",
                transform=ax.transAxes, fontsize=9.5, color=INK2)
        # Relative change of each piece.
        ax = fig.add_subplot(gs[row, 1]); style(ax)
        pv = [dpct] + [r["pct"] if r["pct"] is not None else 0.0 for r in rows]
        for y, v, col, h in zip(ypos, pv, cols, hat):
            ax.barh(y, v, color=col, height=0.62, hatch=h, edgecolor=BG)
        pspan = max(abs(x) for x in pv) or 1
        for y, v, r in zip(ypos, pv, [None] + rows):
            if r is not None and r["pct"] is None:
                txt = "–" if r["c"] == "other" else "new" if r["new"] else "too small to compare"
                ax.text(0.02 * pspan, y, txt, va="center", ha="left", fontsize=8.5, color=INK3, style="italic")
                continue
            ax.text(v + (0.02 * pspan if v >= 0 else -0.02 * pspan), y, f"{v:+.0f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        ax.set_yticks(list(ypos), labels, fontsize=9.5, color=INK)
        ax.axvline(0, color="#9aa0a8", linewidth=1)
        ax.set_xlim(min(0, min(pv)) - (0.55 if min(pv) < 0 else 0.1) * pspan, max(0, max(pv)) + 0.35 * pspan)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_title("Change in each piece (%)", loc="left", fontsize=11, fontweight="bold", color=INK, pad=20)
        if causes:
            direct, knock, o1, o2 = causes[i][metric]
            pick = lambda dd, r: sum(dd[m] for m in r["members"]) if r["c"] == "other" else dd[r["c"]]
            dvals = [sum(direct.values())] + [pick(direct, r) for r in rows]
            kvals = [sum(knock.values())] + [pick(knock, r) for r in rows]
            ax = fig.add_subplot(gs[row, 2]); style(ax)
            cspan = max([abs(x) for x in dvals + kvals] + [1e-9])
            for y, dv_, kv_, col in zip(ypos, dvals, kvals, cols):
                ax.barh(y + 0.18, dv_, color=col, height=0.34, edgecolor=BG)
                ax.barh(y - 0.18, kv_, color=col, height=0.34, hatch="///", edgecolor=BG, alpha=0.55)
                for off, v in ((0.18, dv_), (-0.18, kv_)):
                    if abs(v) >= 0.005 * cspan:
                        ax.text(v + (0.02 * cspan if v >= 0 else -0.02 * cspan), y + off, fmt(v, unit, signed=True),
                                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
            ax.set_yticks(list(ypos), labels, fontsize=9.5, color=INK)
            ax.axvline(0, color="#9aa0a8", linewidth=1)
            lo = min(0, min(dvals + kvals))
            ax.set_xlim(lo - (0.6 if lo < 0 else 0.1) * cspan, max(0, max(dvals + kvals)) + 0.4 * cspan)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_title(f"Direct vs knock-on ({unit})", loc="left", fontsize=11, fontweight="bold", color=INK, pad=20)
            ax.text(0, 1.03, "solid = direct (same batches, new costs)   hatched = knock-on (different batches, queueing)",
                    transform=ax.transAxes, fontsize=8.5, color=INK2)
            td, tk = dvals[0], kvals[0]
            cause_line = (f"{name}, {title}, cause: direct {fmt(td, unit, signed=True)} "
                          f"({(td / dt * 100) if abs(dt) > 1e-9 else 0:.0f}% of the change), knock-on {fmt(tk, unit, signed=True)} "
                          f"({(tk / dt * 100) if abs(dt) > 1e-9 else 0:.0f}%).")
            if abs(dt) > 1e-9 and abs(o1 - o2) > 0.2 * abs(dt):
                cause_line += (f" The two orders disagree (direct {fmt(o1, unit, signed=True)} vs {fmt(o2, unit, signed=True)}):"
                               " cost and scheduling interact, so the split is an average.")
        # Text summary.
        top = next(r for r in rows if r["c"] != "other")
        rel = [r for r in rows if r["pct"] is not None]
        most = max(rel, key=lambda r: abs(r["pct"])) if rel else None
        share = (top["d"] / dt * 100) if abs(dt) > 1e-9 else 0.0
        line = (f"{name}, {title}: {fmt(dt, unit, signed=True)} ({dpct:+.0f}%). "
                f"Biggest impact: {LABEL[top['c']].lower()}, {fmt(top['d'], unit, signed=True)} ({share:.0f}% of the change).")
        if most:
            line += f" Changed most: {LABEL[most['c']].lower()}, {most['pct']:+.0f}%."
        lines.append(line)
        if causes:
            lines.append(cause_line)
    fig.text(0.01, 0.004, "req/step = average requests processed together per step.  Other = pieces that are each < 5% of the change.",
             fontsize=8.5, color=INK3)
    fig.suptitle(title_ctx, x=0.01, ha="left", fontsize=11.5, color=INK2, y=0.995)
    fig.subplots_adjust(left=0.16 if ncol == 2 else 0.12, right=0.80 if ncol == 2 else 0.86, top=1 - 0.55 / fig.get_figheight(), bottom=0.35 / fig.get_figheight() + 0.03)
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return lines


def plot_utilization(values, results, knob, title_ctx, path):
    """GPU utilization = busy GPU-time ÷ paid GPU-time (busy + idle), per knob value."""
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=BG)
    style(ax)
    lines = []
    for i, v in enumerate(values):
        cost = results[i]["COST"]
        paid = sum(cost.values())
        busy = paid - cost.get("idle", 0.0)
        util = busy / paid * 100
        ax.bar(i, util, width=0.55, color="#2a78d6")
        ax.text(i, util + 2, f"{util:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"busy {fmt(busy, 'GPU-ms')}\npaid {fmt(paid, 'GPU-ms')}", (i, 0), xytext=(0, -38),
                    textcoords="offset points", ha="center", va="top", fontsize=9, color=INK3)
        lines.append(f"{knob}={v}: busy {fmt(busy, 'GPU-ms')}, paid {fmt(paid, 'GPU-ms')} per request, "
                     f"GPU utilization {util:.0f}%.")
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(len(values)), [f"{knob}={v}" + ("\n(baseline)" if i == 0 else "")
                                       for i, v in enumerate(values)], fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title("GPU utilization per request (busy ÷ paid GPU-time)", loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.text(0.01, 0.01, "busy = GPU-time running steps (compare with E2E latency).  paid = busy + idle (cost per request).",
             fontsize=8.5, color=INK3)
    fig.suptitle(title_ctx, x=0.01, ha="left", fontsize=10, color=INK2, y=0.99)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.27)
    fig.savefig(path, dpi=150, facecolor=BG); plt.close(fig)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--tp", required=True)
    ap.add_argument("--workload", default="chatbot")
    ap.add_argument("--rate")
    ap.add_argument("--concurrency")
    ap.add_argument("--num-requests", default="800")
    ap.add_argument("--seed", default="7")
    ap.add_argument("--vary", required=True, help="flag=v1,v2,... (first value is the baseline)")
    ap.add_argument("--metrics", default="TTFT,ITL,E2E,COST")
    ap.add_argument("--name", help="output folder name (default: from model and knob)")
    ap.add_argument("--cause", action="store_true",
                    help="layer 3: split each change into direct vs knock-on (tp and hardware only)")
    ap.add_argument("extra", nargs="*", help="extra blis flags, after --")
    a = ap.parse_args()
    knob, vals = a.vary.split("=", 1)
    knob = knob.lstrip("-")
    values = [v.strip() for v in vals.split(",") if v.strip()]
    if len(values) < 2:
        raise SystemExit("--vary needs at least two values")
    base = {"model": a.model, "hardware": a.hardware, "tp": a.tp, "workload": a.workload,
            "num-requests": a.num_requests, "seed": a.seed}
    if a.rate: base["rate"] = a.rate
    if a.concurrency: base["concurrency"] = a.concurrency
    name = a.name or f"{a.model.split('/')[-1]}-{knob}"
    outdir, rundir = os.path.join(STUDY, "charts", f"compare-{name}"), os.path.join(STUDY, "runs", name)
    os.makedirs(outdir, exist_ok=True)
    flags_for = lambda v: [x for k, val in dict(base, **{knob: v}).items() for x in (f"--{k}", val)] + a.extra
    cause = a.cause and knob in CAUSE_KNOBS
    if a.cause and not cause:
        print(f"note: {knob} only changes scheduling, not what a step costs, so its whole effect is "
              f"knock-on by definition. Skipping the direct vs knock-on split.")
    results, causes = [], {}
    for i, v in enumerate(values):
        print(f"running {knob}={v} ...", flush=True)
        if not cause or i == 0:
            results.append(attrib.run(flags_for(v), f"{knob}-{v}", rundir))
            continue
        # Each config's run also prices its batches under the other config (both orders).
        b = values[0]
        base_run = attrib.run(flags_for(b), f"{knob}-{b}.shadow-{v}", rundir, shadow={knob: v})
        this_run = attrib.run(flags_for(v), f"{knob}-{v}.shadow-{b}", rundir, shadow={knob: b})
        results.append(this_run)
        causes[i] = {}
        for m in attrib.METRICS:
            direct, knock, o1, o2 = cause_split(base_run[m], this_run[m], base_run["counterfactual"][m],
                                                this_run["counterfactual"][m])
            moved = sum(this_run[m].values()) - sum(base_run[m].values())
            if abs(sum(direct.values()) + sum(knock.values()) - moved) > max(0.01, 1e-6 * abs(moved)):
                raise SystemExit(f"{m}: direct + knock-on does not add up to the change")
            causes[i][m] = (direct, knock, o1, o2)
    load = f"{a.rate} req/s" if a.rate else f"{a.concurrency} users" if a.concurrency else "default load"
    if knob in ("rate", "concurrency"):
        load = "load varies"
    fixed = ", ".join(f"{k}={v}" for k, v in base.items() if k not in (knob, "num-requests", "seed", "model", "rate", "concurrency"))
    ctx = f"{a.model} · {fixed} · {load} · varying {knob} · BLIS trained-physics (modelled, not GPU-measured)"
    summary = [f"# {a.model}: varying `{knob}` ({', '.join(values)}; baseline {knob}={values[0]})", "",
               f"Fixed: {fixed}, {load}, {a.num_requests} requests, seed {a.seed}.", ""]
    for m in [x.strip().upper() for x in a.metrics.split(",")]:
        path = os.path.join(outdir, f"{m.lower()}.png")
        summary += [f"- {l}" for l in plot_metric(m, values, results, knob, ctx, path,
                                                     causes={i: c for i, c in causes.items()} if causes else None)]
        print(path)
    path = os.path.join(outdir, "utilization.png")
    summary += ["", "GPU-time per request (busy, paid) and utilization:"] + [f"- {l}" for l in plot_utilization(values, results, knob, ctx, path)]
    print(path)
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(summary) + "\n")
    print(os.path.join(outdir, "summary.md"))
    print("\n".join(summary[4:]))


if __name__ == "__main__":
    main()
