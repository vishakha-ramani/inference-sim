import json, subprocess, os, re, sys
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"
M="meta-llama/llama-3.1-70b-instruct"; O="/tmp/attrib"
env=dict(os.environ, BLIS_CATALOG="/tmp/blis-catalog")
def first_json(s):
    i=s.index("{"); dep=0
    for j,c in enumerate(s[i:],i):
        dep+= (c=="{")-(c=="}")
        if dep==0: return json.loads(s[i:j+1])
rows=[]
for tp in (4,8):
  for rate in (4,8,16,24,32,48,64,96):
    led=f"{O}/led_tp{tp}_r{rate}.json"
    out=subprocess.run([W,"run","--model",M,"--hardware","H100","--tp",str(tp),"--workload","chatbot",
        "--rate",str(rate),"--num-requests","800","--seed","7"],env=dict(env,BLIS_STUDY_TERM_LEDGER=led),
        capture_output=True,text=True,check=True).stdout
    m=first_json(out); l=json.load(open(led))
    rows.append(dict(tp=tp,rate=rate,tps=m["tokens_per_sec"],ttft99=m["ttft_p99_ms"],e2e99=m["e2e_p99_ms"],
        itl=m.get("itl_mean_ms"),out_tok=m["total_output_tokens"],ledger=l))
json.dump(rows,open(f"{O}/sweep.json","w"),indent=1)
print(f"{'tp':>3}{'rate':>5}{'tok/s':>9}{'tok/s/GPU':>10}{'TTFTp99':>9}{'E2Ep99':>9}{'ITL':>7}{'meanB':>7}  dominant")
for r in rows:
    l=r["ledger"]; hb=l["batch_size_hist"]; mb=sum(int(k)*v for k,v in hb.items())/l["steps"]
    dom=max(l["dominant_steps"],key=l["dominant_steps"].get)
    print(f"{r['tp']:>3}{r['rate']:>5}{r['tps']:>9.0f}{r['tps']/r['tp']:>10.0f}{r['ttft99']:>9.0f}{r['e2e99']:>9.0f}{r['itl'] or 0:>7.1f}{mb:>7.1f}  {dom}")
