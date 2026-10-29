import json, subprocess, os
W="/Users/vishakha/git-repos/llm-git-repos/inference-sim/.worktrees/config-attribution/blis"
M="meta-llama/llama-3.1-70b-instruct"; O="/tmp/attrib"
def run(tp, conc, shadow):
    led=f"{O}/san_tp{tp}_c{conc}.json"
    env=dict(os.environ, BLIS_CATALOG="/tmp/blis-catalog", BLIS_STUDY_TERM_LEDGER=led, BLIS_STUDY_SHADOW_TP=str(shadow))
    p=subprocess.run([W,"run","--model",M,"--hardware","H100","--tp",str(tp),"--workload","chatbot",
        "--concurrency",str(conc),"--num-requests","800","--seed","7"],env=env,capture_output=True,text=True)
    if p.returncode: raise SystemExit(p.stderr[-800:])
    o=p.stdout; m=json.loads(o[o.index("{"):o.index("\n}")+2]); l=json.load(open(led))
    mb=sum(int(k)*v for k,v in l["batch_size_hist"].items())/l["steps"]
    return m, l, mb
print("Closed loop: N users, each sends its next request when the last finishes -> both configs hold ~N requests.")
print("Expectation: batch sizes match, so feedback ~ 0 and gap ~ direct.\n")
print(f"{'users':>6}{'batch TP4':>10}{'batch TP8':>10}{'TP4':>7}{'TP8':>7}{'gap':>7}{'direct':>8}{'feedback':>9}{'fb share':>9}")
for conc in (16,64,128):
    ma,la,ba=run(4,conc,8); mb_,lb,bb=run(8,conc,4)
    assert ma["total_output_tokens"]==mb_["total_output_tokens"], "workloads differ"
    n=ma["total_output_tokens"]
    AA=la["busy_us"]*4/n; BA=la["shadow_busy_us"]*8/n; BB=lb["busy_us"]*8/n
    gap=BB-AA; d=BA-AA; f=BB-BA
    print(f"{conc:>6}{ba:>10.1f}{bb:>10.1f}{AA:>7.0f}{BB:>7.0f}{gap:>+7.0f}{d:>+8.0f}{f:>+9.0f}{f/gap:>9.0%}")
