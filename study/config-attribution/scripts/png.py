import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
D=json.load(open("/tmp/attrib/bd_slim.json"))
CATS=[("weights","Weight reads","#2a78d6"),("kv","KV-cache reads/writes","#eb6834"),("compute","Compute","#1baf7a"),
      ("comm","TP communication","#eda100"),("overhead","Fixed per-step overhead","#e87ba4"),("api","API overhead","#008300"),
      ("wait","Waiting for a batch slot","#4a3aa7"),("preempt","Work discarded by preemption","#e34948")]
MET=[("TTFT","Time to first token","ms"),("ITL","Inter-token latency","ms"),("E2E","End-to-end latency","ms"),("COST","GPU-time per token (µs)","GPU-µs")]
def fmt(x,u):
    if u!="ms": return f"{x:.0f} µs"
    return f"{x/1000:.1f} s" if x>=10000 else (f"{x:.0f} ms" if x>=100 else f"{x:.1f} ms")
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#9aa0a8","axes.labelcolor":"#4b5059","xtick.color":"#14161a","ytick.color":"#4b5059"})
for rate in (32,64):
    fig,axs=plt.subplots(1,4,figsize=(15,5.8),facecolor="#fcfcfb")
    for ax,(m,title,u) in zip(axs,MET):
        ax.set_facecolor("#fcfcfb")
        rows=[next(d for d in D if d["tp"]==tp and d["rate"]==rate) for tp in (2,4,8)]
        for i,r in enumerate(rows):
            tot=sum(max(0,r[m][k]) for k,_,_ in CATS); acc=0
            for k,_,c in CATS:
                v=max(0,r[m][k])/tot
                if v<=0: continue
                ax.bar(i,v,bottom=acc,width=0.55,color=c,edgecolor="#fcfcfb",linewidth=1.5); acc+=v
            ax.text(i,1.02,fmt(tot,u),ha="center",va="bottom",fontsize=10.5,fontweight="bold",color="#14161a")
            ax.text(i,-0.13,f"batch {round(r['batch'])}\n{str(r['pre'])+' preempt.' if r['pre'] else 'no preempt.'}",ha="center",va="top",fontsize=8.5,color="#737a85",transform=ax.get_xaxis_transform() if False else ax.transData)
        ax.set_xticks(range(3),["TP2","TP4","TP8"],fontweight="bold"); ax.set_ylim(0,1.12)
        ax.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y",color="#dfe2e7",linewidth=0.8); ax.set_axisbelow(True)
        ax.set_title(title,fontsize=12.5,fontweight="bold",loc="left",color="#14161a",pad=18)
    fig.legend(handles=[Patch(color=c,label=n) for _,n,c in CATS],loc="lower center",ncol=4,frameon=False,fontsize=10,bbox_to_anchor=(0.5,0.01))
    fig.suptitle(f"Where the time goes — llama-3.1-70b on H100, chatbot preset, {rate} req/s   (share of total; label = total; BLIS-modelled, not GPU-measured)",
                 x=0.01,ha="left",fontsize=12,color="#4b5059",y=0.99)
    fig.subplots_adjust(left=0.04,right=0.995,top=0.84,bottom=0.29,wspace=0.28)
    fig.savefig(f"/tmp/attrib/where-time-goes-{rate}rps.png",dpi=160,facecolor="#fcfcfb")
    print(f"/tmp/attrib/where-time-goes-{rate}rps.png")
