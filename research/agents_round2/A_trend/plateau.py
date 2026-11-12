"""第二步：從 explore_dv.csv 挑「探索、驗證都賺」且參數高原（相鄰參數也賺）的變化。只用前兩段。"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILIES
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_dv.csv"), low_memory=False)
df = df[df.fam != "對照每天做多"].copy()
df["ok"] = (df.d_tot > 0) & (df.v_tot > 0)
EXN = {"移40": ["移70"], "移70": ["移40", "移100"], "移100": ["移70", "移150"], "移150": ["移100"],
       "移A.15": ["移A.25"], "移A.25": ["移A.15"], "2R": ["3R"], "3R": ["2R"], "15根低": ["30根低"], "30根低": ["15根低"],
       "收盤": ["移150", "保本1R"], "保本1R": ["收盤"], "VWAP": ["VWAP+移100"], "VWAP+移100": ["VWAP", "移100"], "中午": ["收盤"]}
rows = []
for fam, (fn, G) in FAMILIES.items():
    x = df[df.fam == fam]
    keys = list(G)
    idx = {tuple(json.loads(p).values()) + (s, e): ok for p, s, e, ok in zip(x.params, x.stop, x.exit, x.ok)}
    for r in x[x.ok & (x.d_n >= 60) & (x.v_n >= 20)].itertuples():
        p = json.loads(r.params); nb = []
        for j, k in enumerate(keys):
            vals = G[k]; pos = vals.index(p[k])
            for q in (pos - 1, pos + 1):
                if 0 <= q < len(vals):
                    pp = dict(p); pp[k] = vals[q]
                    nb.append(idx.get(tuple(pp.values()) + (r.stop, r.exit), False))
        for ex in EXN.get(r.exit, []):
            nb.append(idx.get(tuple(p.values()) + (r.stop, ex), False))
        rows.append(dict(fam=fam, params=r.params, stop=r.stop, exit=r.exit, d_n=r.d_n, d_avg=r.d_avg, d_tot=r.d_tot, d_mdd=r.d_mdd,
                         v_n=r.v_n, v_avg=r.v_avg, v_tot=r.v_tot, v_mdd=r.v_mdd, nb_n=len(nb), nb_ok=np.mean(nb)))
R = pd.DataFrame(rows)
R["score"] = np.minimum(R.d_avg, R.v_avg)
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plateau_dv.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 200)
print("探索＋驗證都賺、筆數夠的變化：", len(R), "／", len(df))
P = R[R.nb_ok >= 0.8]
print("其中相鄰參數 ≥80% 也都賺（高原）：", len(P))
print(P.groupby("fam").size())
for fam in FAMILIES:
    y = P[P.fam == fam].sort_values("score", ascending=False).head(6)
    if len(y):
        print("==", fam); print(y[["params", "stop", "exit", "d_n", "d_avg", "d_tot", "d_mdd", "v_n", "v_avg", "v_tot", "v_mdd", "nb_ok"]].to_string(index=False))
