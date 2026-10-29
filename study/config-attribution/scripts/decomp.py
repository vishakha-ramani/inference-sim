import json
rows={(r["tp"],r["rate"]):r for r in json.load(open("/tmp/attrib/sweep.json"))}
def per_tok(r, gpu=False):
    l=r["ledger"]; k=r["tp"] if gpu else 1
    return {t:v*k/r["out_tok"] for t,v in l["term_us"].items()}, l["busy_us"]*k/r["out_tok"]
for rate in (32,96):
  a,b=rows[(4,rate)],rows[(8,rate)]
  for gpu,label in ((False,"replica busy µs per output token (latency view)"),(True,"GPU-µs per output token (cost view)")):
    ta,ba=per_tok(a,gpu); tb,bb=per_tok(b,gpu)
    print(f"\nrate {rate} — {label}:  TP4 {ba:.1f}  TP8 {bb:.1f}  Δ(TP8−TP4) {bb-ba:+.1f}")
    for t in sorted(ta,key=lambda t:-abs(tb[t]-ta[t])):
      d=tb[t]-ta[t]
      if abs(d)>0.05 or ta[t]>0.05: print(f"   {t:<22}{ta[t]:>8.2f}{tb[t]:>8.2f}{d:>+9.2f}")
