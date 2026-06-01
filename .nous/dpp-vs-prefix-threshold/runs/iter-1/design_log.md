All artifacts written and validation passes. Here's the summary:

**Iter-1 Design Complete** — characterizes 4 disaggregation policies (Never, Always, PrefixThreshold, DPP) across a rate sweep (5–50 req/s).

**Key finding from exploration:** DPP(V=100, SLO=50ms) is mathematically equivalent to AlwaysDisaggregate because the penalty term V·c_D/2 = 650,000μs overwhelms all queue-feedback terms. The experiment will empirically confirm this equivalence and characterize the ITL–TTFT tradeoff at each load point, establishing the baseline for iter-2's V/SLO parameter sweep where DPP's adaptive behavior actually engages.