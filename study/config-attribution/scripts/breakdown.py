import json, subprocess, os, statistics as st
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"
M="meta-llama/llama-3.1-70b-instruct"; O="/tmp/attrib/bd"; os.makedirs(O,exist_ok=True)
A0,A1,A2=15.563,0.777,0.045   # ms; BLIS applies alpha_coeffs truncated to whole µs (clampToInt64)
NAMES=["prefill_compute","prefill_kv","decode_compute","decode_kv","weight_load","tp_allreduce","moe_dispatch",
       "per_layer_overhead","per_request_overhead","per_step_overhead","moe_interleave_overhead","adapter_overhead"]
G={"prefill_compute":"compute","decode_compute":"compute","weight_load":"weights","prefill_kv":"kv","decode_kv":"kv",
   "tp_allreduce":"comm","moe_dispatch":"comm"}
CATS=["wait","api","compute","weights","kv","comm","overhead","preempt"]
def grp(vec,scale=1e-3):
    out={c:0.0 for c in CATS}
    for n,v in zip(NAMES,vec): out[G.get(n,"overhead")]+=v*scale
    return out
def add(a,b): return {c:a[c]+b[c] for c in CATS}
def run(tp, rate):
    led,mp=f"{O}/led_tp{tp}_r{rate}.json",f"{O}/met_tp{tp}_r{rate}.json"
    env=dict(os.environ,BLIS_CATALOG="/tmp/blis-catalog",BLIS_STUDY_TERM_LEDGER=led)
    subprocess.run([W,"run","--model",M,"--hardware","H100","--tp",str(tp),"--workload","chatbot","--rate",str(rate),
        "--num-requests","800","--seed","7","--metrics-path",mp],env=env,capture_output=True,text=True,check=True)
    L=json.load(open(led)); Mt=json.load(open(mp))
    reqs=[r for r in Mt["requests"] if r["e2e_ms"]>0 and r["num_decode_tokens"]>1]
    T,I,E=[],[],[]; maxres=0; minwait=1e9
    for r in reqs:
        pr=L["per_request"][r["requestID"]]; n=r["num_decode_tokens"]
        # TTFT (to the first token of the FINAL attempt) = α0 + waiting + discarded work from
        # preempted attempts + final-attempt prefill steps + α2 for the first token
        t=grp(pr["prefill_us"]); t["preempt"]=sum(pr["discarded_us"])*1e-3; t["api"]=A0+A2
        t["wait"]=r["ttft_ms"]-sum(t[c] for c in CATS if c!="wait")
        # decode phase = final-attempt decode steps + α2 per later token; remainder should be ~0
        d=grp(pr["decode_us"]); d["api"]=A2*(n-1)
        resid=r["e2e_ms"]-r["ttft_ms"]-A1-sum(d.values())
        maxres=max(maxres,abs(resid)); minwait=min(minwait,t["wait"])
        e=add(t,d); e["api"]+=A1
        i={c:d[c]/(n-1) for c in CATS}
        T.append(t);I.append(i);E.append(e)
    mean=lambda rows:{c:st.mean(x[c] for x in rows) for c in CATS}
    ntok=Mt["total_output_tokens"]
    cost=grp([L["term_us"][k] for k in NAMES],scale=tp/ntok)
    pre=Mt.get("preemption_count",0)
    return dict(tp=tp,rate=rate,TTFT=mean(T),ITL=mean(I),E2E=mean(E),COST=cost,preemptions=pre,
        reported=dict(TTFT=st.mean(r["ttft_ms"] for r in reqs),ITL=st.mean(r["itl_ms"] for r in reqs),
                      E2E=st.mean(r["e2e_ms"] for r in reqs),COST=L["busy_us"]*tp/ntok,tps=Mt["tokens_per_sec"]),
        batch=round(sum(int(k)*v for k,v in L["batch_size_hist"].items())/L["steps"],1),itl_resid=maxres,minwait=minwait,n=len(reqs))
out=[]
for rate in (8,16,32,48,64):
    for tp in (2,4,8):
        r=run(tp,rate); out.append(r)
        chk=" ".join(f"{m}:{sum(r[m].values()):.2f}/{r['reported'][m]:.2f}" for m in ("TTFT","ITL","E2E","COST"))
        print(f"r{rate:<3}tp{tp} preempt={r['preemptions']:<4} batch={r['batch']:<6} sum/reported  {chk}   max decode-phase remainder={r['itl_resid']:.4f}ms  min wait={r['minwait']:.2f}ms")
json.dump(out,open("/tmp/attrib/breakdown.json","w"),indent=1)
