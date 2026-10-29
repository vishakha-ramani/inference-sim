import json, subprocess, os, statistics as st
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"; O="/tmp/attrib/lb"; os.makedirs(O,exist_ok=True)
A0,A1,A2=15.563,0.777,0.045
NAMES=["prefill_compute","prefill_kv","decode_compute","decode_kv","weight_load","tp_allreduce","moe_dispatch",
       "per_layer_overhead","per_request_overhead","per_step_overhead","moe_interleave_overhead","adapter_overhead"]
G={"prefill_compute":"compute","decode_compute":"compute","weight_load":"weights","prefill_kv":"kv","decode_kv":"kv","tp_allreduce":"comm","moe_dispatch":"comm"}
CATS=["wait","api","compute","weights","kv","comm","overhead","preempt"]
def grp(vec,scale=1e-3):
    out={c:0.0 for c in CATS}
    for n,v in zip(NAMES,vec): out[G.get(n,"overhead")]+=v*scale
    return out
LB=json.load(open("/tmp/attrib/lb_results.json"))
out=[]
for r in LB:
    if r["group_id"]!=LB[0]["group_id"] or r["status"].get("disqualifications"): continue
    rid=r["run_id"]; led,mp=f"{O}/{rid}_led.json",f"{O}/{rid}_met.json"
    argv=[W]+r["provenance"]["argv"][1:]; argv[argv.index("--metrics-path")+1]=mp
    env=dict(os.environ,BLIS_CATALOG="/tmp/blis-catalog",BLIS_STUDY_TERM_LEDGER=led)
    subprocess.run(argv,env=env,capture_output=True,text=True,check=True)
    L=json.load(open(led)); M=json.load(open(mp)); tp=r["deployment"]["tp"]
    reqs=[q for q in M["requests"] if q["e2e_ms"]>0 and q["num_decode_tokens"]>1]
    T,I=[],[]
    for q in reqs:
        pr=L["per_request"][q["requestID"]]; n=q["num_decode_tokens"]
        t=grp(pr["prefill_us"]); t["preempt"]=sum(pr["discarded_us"])*1e-3; t["api"]=A0+A2
        t["wait"]=q["ttft_ms"]-sum(t[c] for c in CATS if c!="wait")
        dd=grp(pr["decode_us"]); dd["api"]=A2*(n-1)
        T.append(t); I.append({c:dd[c]/(n-1) for c in CATS})
    mean=lambda rows:{c:st.mean(x[c] for x in rows) for c in CATS}
    cost=grp([L["term_us"][k] for k in NAMES],scale=tp/M["total_output_tokens"])
    pub=r["metrics"]
    out.append(dict(run=rid,hw=r["deployment"]["hardware"],tp=tp,mns=r["deployment"]["max_num_seqs"],
        TTFT=mean(T),ITL=mean(I),COST=cost,batch=sum(int(k)*v for k,v in L["batch_size_hist"].items())/L["steps"],
        now=dict(ttft=M["ttft_mean_ms"],itl=M["itl_mean_ms"],tps=M["tokens_per_sec"],ttft99=M["ttft_p99_ms"]),
        pub=dict(ttft=pub["ttft_mean_ms"],itl=pub["itl_mean_ms"],tps=pub["tokens_per_sec"],ttft99=pub["ttft_p99_ms"])))
json.dump(out,open("/tmp/attrib/lb_explain.json","w"),indent=1)
print(f"{'run':<16}{'ITL now/pub':>14}{'TTFT now/pub':>15}{'TTFTp99 now/pub':>17}{'batch':>7}")
for x in out:
    print(f"{x['run']:<16}{x['now']['itl']:>6.2f}/{x['pub']['itl']:<7.2f}{x['now']['ttft']:>7.1f}/{x['pub']['ttft']:<7.1f}{x['now']['ttft99']:>8.0f}/{x['pub']['ttft99']:<8.0f}{x['batch']:>7.1f}")
