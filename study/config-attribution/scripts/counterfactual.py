import json, subprocess, os
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"
M="meta-llama/llama-3.1-70b-instruct"; O="/tmp/attrib"
base=dict(os.environ, BLIS_CATALOG="/tmp/blis-catalog")
def run(tp, rate, shadow=None, ledger=True):
    env=dict(base)
    if ledger: env["BLIS_STUDY_TERM_LEDGER"]=f"{O}/cf_tp{tp}_r{rate}.json"
    if shadow: env["BLIS_STUDY_SHADOW_TP"]=str(shadow)
    out=subprocess.run([W,"run","--model",M,"--hardware","H100","--tp",str(tp),"--workload","chatbot",
        "--rate",str(rate),"--num-requests","800","--seed","7"],env=env,capture_output=True,text=True,check=True).stdout
    return out, (json.load(open(env["BLIS_STUDY_TERM_LEDGER"])) if ledger else None)
# parity: shadow pricing must not perturb the simulation
assert run(4,32,ledger=False)[0]==run(4,32,shadow=8)[0], "stdout changed with shadow on"
print("parity OK: stdout byte-identical with shadow pricing on\n")
A,B=4,8
out_tok=None
print("GPU-µs per output token.  A=TP4, B=TP8.  total gap = direct (same batches, B's costs) + feedback (B's own schedule)")
print(f"{'rate':>5}{'A|A':>8}{'B|A':>8}{'B|B':>8}{'gap':>8}{'direct':>9}{'feedback':>10}   top terms: direct | feedback")
res={}
for rate in (8,16,32,48,64,96):
    oa,la=run(A,rate,shadow=B); ob,lb=run(B,rate,shadow=A)
    ntok=json.loads(oa[oa.index("{"):oa.index("\n}")+2])["total_output_tokens"]
    g=lambda us,tp: us*tp/ntok
    AA=g(la["busy_us"],A); BA=g(la["shadow_busy_us"],B); BB=g(lb["busy_us"],B)
    dterm={t:g(la["shadow_term_us"][t],B)-g(la["term_us"][t],A) for t in la["term_us"]}
    fterm={t:g(lb["term_us"][t],B)-g(la["shadow_term_us"][t],B) for t in la["term_us"]}
    top=lambda d:", ".join(f"{k} {v:+.0f}" for k,v in sorted(d.items(),key=lambda kv:-abs(kv[1]))[:2])
    print(f"{rate:>5}{AA:>8.0f}{BA:>8.0f}{BB:>8.0f}{BB-AA:>+8.0f}{BA-AA:>+9.0f}{BB-BA:>+10.0f}   {top(dterm)} | {top(fterm)}")
    res[rate]=dict(AA=AA,BA=BA,BB=BB,direct=dterm,feedback=fterm)
json.dump(res,open(f"{O}/counterfactual.json","w"),indent=1)
