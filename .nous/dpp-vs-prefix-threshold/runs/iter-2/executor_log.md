Validation passed. Here's a summary of the experiment:

---

## Experiment Complete — Iteration 2: DPP V Sweep

**All 3 arms validated. `nous validate` returned `{"status": "pass"}`.**

### Key Findings

**h-main (CONFIRMED):** DPP partial disaggregation (V=1–50, SLO=20ms) is a **dominated strategy** for ITL at rate=30. Every V value produces higher mean ITL than PrefixThreshold (28.65ms):
- V=1: 33.69ms (+17.5%), V=5: 36.16ms (+26.2% — worst), V=50: 30.34ms (+5.9% — best partial)
- Disaggregation fractions match probe predictions closely (3–73%)
- Z-feedback permanently suppresses disaggregation after just 65–1462 initial requests

**h-robustness (CONFIRMED):** At rate=40, DPP V≤10 achieves **100% completion** vs PT's 86.25% (275 drops). The tradeoff is catastrophic TTFT (2894–4870ms vs PT's 64ms) — DPP acts as a stability controller by keeping traffic local and avoiding decode KV exhaustion. V=20 starts dropping (3.1%), V=50 resembles PT.

**h-control-negative (PARTIALLY_CONFIRMED, regime error):** The binary Z-feedback switch is confirmed, but the transition threshold is **between SLO=20ms and SLO=22ms** (not 22–25ms as predicted):
- SLO=20ms → 10.2% disagg, ITL=36.16ms
- SLO=22ms → 90.6% disagg, ITL=29.19ms (sharp jump!)
- SLO=25ms → 100% disagg, ITL=28.65ms

The actual critical threshold (~21ms) equals the **minimum achievable disagg TTFT at low queue** (prefill scheduling ~15ms + KV transfer ~6ms). This explains the bistable behavior: at SLO=22ms, the first requests at an empty queue see TTFT<SLO keeping Z=0, enabling sustained disaggregation.

**3 new principles extracted (RP-5, RP-6, RP-7):** Partial disaggregation dominated for ITL; DPP as saturation controller; Z-feedback binary phase transition anchored to minimum TTFT.