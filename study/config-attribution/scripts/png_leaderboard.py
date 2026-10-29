# Charts for the leaderboard example: qwen3-14b @ 6 req/s (inference-sim/leaderboard prototypes group).
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE=os.path.dirname(os.path.abspath(__file__))
X={x["run"]:x for x in json.load(open(f"{HERE}/../data/lb_explain.json"))}
CATS=[("weights","Weight reads","#2a78d6"),("kv","KV-cache reads/writes","#eb6834"),("compute","Compute","#1baf7a"),
      ("comm","TP communication","#eda100"),("overhead","Fixed per-step overhead","#e87ba4"),("api","API overhead","#008300"),
      ("wait","Waiting for a batch slot","#4a3aa7"),("preempt","Work discarded by preemption","#e34948")]
PANELS=[
 ("Why H100 > A100 > L40S\non ITL (TP4)","ITL",[("h100-tp4","H100"),("a100-tp4","A100"),("l40s-tp4","L40S")],"ms",
  "Gap is weight reads; ratios match\nmemory bandwidth (1.64×, 3.9×)"),
 ("Why more TP gives diminishing\nITL returns (H100)","ITL",[("h100-tp1","TP1"),("h100-tp2","TP2"),("h100-tp4","TP4")],"ms",
  "Weight reads halve each step;\nfixed overhead (~1.45 ms) does not"),
 ("Why max_num_seqs=8 loses\non TTFT (H100 TP2)","TTFT",[("h100-tp2","seqs 256"),("h100-tp2-seqs8","seqs 8")],"ms",
  "Whole gap is waiting for a slot;\nITL unchanged (7.19 vs 7.16 ms)"),
 ("Why the slower A100 is almost\nas cheap per token (TP1)","COST",[("h100-tp1","H100"),("a100-tp1","A100")],"µs",
  "Bigger batches (21 vs 13.5) share\neach weight read across more tokens"),
]
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#9aa0a8","xtick.color":"#14161a","ytick.color":"#4b5059"})
fig,axs=plt.subplots(1,4,figsize=(16,5.9),facecolor="#fcfcfb")
used=set()
for ax,(title,m,runs,u,note) in zip(axs,PANELS):
    ax.set_facecolor("#fcfcfb"); tops=[]
    for i,(rid,lab) in enumerate(runs):
        acc=0
        for k,_,c in CATS:
            v=max(0,X[rid][m][k])
            if v<=0: continue
            used.add(k); ax.bar(i,v,bottom=acc,width=0.55,color=c,edgecolor="#fcfcfb",linewidth=1.5); acc+=v
        tops.append(acc)
    ymax=max(tops)*1.18
    for i,t in enumerate(tops):
        ax.text(i,t+ymax*0.015,(f"{t:.0f} {u}" if t>=100 else f"{t:.1f} {u}"),ha="center",va="bottom",fontsize=10.5,fontweight="bold",color="#14161a")
    ax.set_ylim(0,ymax); ax.set_xticks(range(len(runs)),[l for _,l in runs],fontweight="bold")
    if m=="COST":
        for i,(rid,_) in enumerate(runs): ax.text(i,-0.12,f"batch {X[rid]['batch']:.1f}",ha="center",va="top",fontsize=8.5,color="#737a85",transform=ax.get_xaxis_transform())
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#dfe2e7",linewidth=0.8); ax.set_axisbelow(True)
    ax.set_ylabel({"ITL":"mean ITL (ms)","TTFT":"mean TTFT (ms)","COST":"GPU-µs per output token"}[m],color="#4b5059")
    ax.set_title(title,fontsize=11.2,fontweight="bold",loc="left",color="#14161a",pad=10)
    ax.text(0.0,-0.24,note,transform=ax.transAxes,fontsize=9.5,color="#4b5059",va="top")
fig.legend(handles=[Patch(color=c,label=n) for k,n,c in CATS if k in used],loc="lower center",ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(0.5,0.0))
fig.suptitle("Explaining the leaderboard: qwen3-14b, 6 req/s, prompts ~512 / outputs ~128 tokens   (BLIS trained-physics; reproduces published numbers exactly)",
             x=0.01,ha="left",fontsize=12,color="#4b5059",y=0.99)
fig.subplots_adjust(left=0.045,right=0.995,top=0.83,bottom=0.30,wspace=0.32)
out=f"{HERE}/../charts/leaderboard-why-qwen3-14b.png"; fig.savefig(out,dpi=160,facecolor="#fcfcfb"); print(out)
