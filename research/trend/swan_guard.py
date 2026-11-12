import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
import numpy as np, pandas as pd
from lib import data
g = data.load_grid("NQ", include_holdout=True)
F = pd.read_csv("wr_table.csv.gz"); F["key"] = F.rule + "|" + F.side
def recent_shock(t, look=30):
    i = g.dindex[t.date]; C = g.C[i]; a = max(0, t.entry_slot - look); w = C[a:t.entry_slot]; w = w[~np.isnan(w)]
    if len(w) < 6: return 0.0
    return float(np.max(np.abs(w[5:] - w[:-5])))
F["shock"] = [recent_shock(t) for t in F.itertuples()]
F["P"] = np.where(F.date >= "2026-10-01", "近期", "過去")
HW = (F.key == "I2|多") | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True)) | ((F.key == "ASIA-VA-k1|多") & (F.es_ok == True)) | ((F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2))
REC = F.key.str.startswith("I2") | (F.key == "AM-PD-k1|空") | ((F.key == "AM-PD-k1|多") & (F.i2 != "逆")) | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True))
for name, m in [("所有點", F.key.notna()), ("推薦組合的點", REC), ("高勝率組合的點", HW)]:
    x = F[m]
    for th in [None, 150, 200, 250, 300]:
        y = x if th is None else x[x.shock < th]; z = x if th is None else x[x.shock >= th]
        s = "｜".join(f"{p}: {len(y[y.P==p])} 筆 {(y[y.P==p].pnl55>0).mean()*100:.1f}% {y[y.P==p].pnl55.mean():+.0f}" for p in ["過去", "近期"])
        sz = "" if th is None else f"｜被擋掉 {len(z)} 筆：勝率 {(z.pnl55>0).mean()*100:.1f}% 每筆 {z.pnl55.mean():+.0f}"
        print(f"{name:8s} {'不擋' if th is None else f'擋 ≥{th} 點':8s} {s}{sz}")
F[["date", "rule", "side", "entry_slot", "shock", "pnl55"]].to_csv("swan_shock.csv", index=False)
