import json, subprocess, os, statistics as st
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"
M="meta-llama/llama-3.1-70b-instruct"; O="/tmp/attrib"
def run(tp, rate):
    mp=f"{O}/lat_tp{tp}_r{rate}.json"
    env=dict(os.environ, BLIS_CATALOG="/tmp/blis-catalog")
    subprocess.run([W,"run","--model",M,"--hardware","H100","--tp",str(tp),"--workload","chatbot","--rate",str(rate),
        "--num-requests","800","--seed","7","--metrics-path",mp],env=env,capture_output=True,text=True,check=True)
    d=json.load(open(mp)); rs=[r for r in d["requests"] if r["e2e_ms"]>0]
    for r in rs:
        r["queue"]=r["scheduling_delay_ms"]; r["prefill"]=r["ttft_ms"]-r["queue"]; r["decode"]=r["e2e_ms"]-r["ttft_ms"]
    return rs, d.get("preemption_count",0)
def mean(rs,k): return st.mean(r[k] for r in rs)
def tail(rs,k="ttft_ms",q=0.99):
    s=sorted(rs,key=lambda r:r[k]); return s[int(q*len(s)):]
print("Mean per request (ms).  TTFT = queue + prefill ; E2E = TTFT + decode\n")
print(f"{'rate':>5}{'tp':>4}{'TTFT':>8}{'=queue':>8}{'+prefill':>9}{'E2E':>8}{'decode':>8}{'preempt':>8}   p99-TTFT requests: queue share")
for rate in (16,32,48,64,96):
    for tp in (4,8):
        rs,pc=run(tp,rate); t=tail(rs)
        qshare=sum(r["queue"] for r in t)/sum(r["ttft_ms"] for r in t)
        print(f"{rate:>5}{tp:>4}{mean(rs,'ttft_ms'):>8.0f}{mean(rs,'queue'):>8.0f}{mean(rs,'prefill'):>9.0f}{mean(rs,'e2e_ms'):>8.0f}{mean(rs,'decode'):>8.0f}{pc:>8}   {qshare:>5.0%}")
