"""前進測試：把所有事先凍結的規則套到 2026-10-20 以後的新資料，規則一行都不改。
需要分支 claude/dazzling-euler-endxzx 的 research/lib 和 research/agents（共用回測引擎與候選規則）。
做法：在該分支的 worktree 裡，把 data/ 主檔換成「主檔＋data/*_1m_yahoo_20261020-20261110.csv.gz」合併後的
{NQ,ES}_1m_continuous_20240102-20261110.csv.gz，再把本檔放到 research/ 執行。
輸出 fwd_out/summary.csv、fwd_out/trades.csv（本資料夾的 summary.csv、trades.csv 就是這次的輸出）。"""
import sys, os, importlib.util
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import numpy as np, pandas as pd
from lib import data, features, sim, strategies
FWD='2026-10-20'
G={s:data.load_grid(s,include_holdout=True) for s in ("NQ","ES")}
LV={s:features.daily_levels(G[s]) for s in G}
g,lv=G["NQ"],LV["NQ"]
def load(agent):
    p=os.path.join(HERE,"agents",agent,"candidates.py"); sys.path.insert(0,os.path.dirname(p))
    spec=importlib.util.spec_from_file_location(f"c_{agent}",p); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); sys.path.pop(0); return m
T={}
T['I2 v1（正式）']=sim.run_signals(g,strategies.i2_v1_signals(g,lv))
T['I2 v2（11:00前、+80）']=sim.run_signals(g,strategies.i2_v1_signals(g,lv,last_entry_idx=90,tp=80.0))
T['I2 v1＋H1']=sim.run_signals(g,strategies.i2_v1_signals(g,lv,h1=True))
for a in ["A1","A2","A3","B1","B2","B3","C1","C2","C3"]:
    m=load(a)
    for n,fn in m.CANDIDATES.items():
        try:
            if a=="C3" and n in getattr(m,"PORTFOLIOS",set()): tr=m.run_portfolio(n,include_holdout=True)
            elif a=="C3" and n=="C3_ES_I2_40_15": tr=sim.run_signals(G["ES"],fn(G["ES"],LV["ES"]))
            else: tr=sim.run_signals(g,fn(g,lv))
            T[f"{a}/{n}"]=tr
        except Exception as e: print('ERR',a,n,e)
sys.path.insert(0,os.path.join(HERE,"agents","D"))
spec=importlib.util.spec_from_file_location("books",os.path.join(HERE,"agents","D","portfolios.py"))
books=importlib.util.module_from_spec(spec); spec.loader.exec_module(books)
for n,fn in books.BOOKS.items():
    try: T["D/"+n]=fn(include_holdout=True)
    except Exception as e: print('ERR D',n,e)
rows=[]; allf=[]
for k,tr in T.items():
    if len(tr)==0: continue
    f=tr[tr.date>=FWD]; h=tr[(tr.date>='2026-01-01')&(tr.date<FWD)]; p=tr[tr.date<'2026-01-01']
    rows.append(dict(rule=k,n_fwd=len(f),fwd_total=f.pnl.sum(),fwd_avg=f.pnl.mean() if len(f) else np.nan,
        fwd_tp=(f.reason=='tp').sum(),fwd_sl=(f.reason.isin(['sl','stop'])).sum(),
        avg_2026pre=h.pnl.mean(),avg_2024_25=p.pnl.mean()))
    f=f.copy(); f['rule']=k; allf.append(f)
R=pd.DataFrame(rows).round(0)
os.makedirs('fwd_out',exist_ok=True)
R.to_csv('fwd_out/summary.csv',index=False)
pd.concat(allf).to_csv('fwd_out/trades.csv',index=False)
with pd.option_context('display.width',250,'display.max_rows',100): print(R.to_string(index=False))
