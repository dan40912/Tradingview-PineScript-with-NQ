import pickle, numpy as np, pandas as pd, grid, strat
pd.set_option("display.width", 250)
R = pd.read_pickle("grid2.pkl")
def show(mask, title):
    x = R[mask]; print(f"{title}：{len(x)} 個變化｜選擇期每筆中位數 {x['選擇期 每筆'].median():+.0f}｜三段都賺的比例 {x['過驗證'].mean():.0%}｜報酬回撤比中位數 {x['選擇期 報酬回撤比'].median():.1f}")
print("— 參數高原：同一家族換參數會不會壞 —")
show(R.family == "順勢突破", "順勢突破（全部 576）")
show((R.family == "順勢突破") & R.name.str.contains("IB30") & R.name.str.contains("出場179"), "順勢 IB30、12:29 出場")
show((R.family == "順勢突破") & R.name.str.contains("IB15"), "順勢 IB15")
show((R.family == "順勢突破") & R.name.str.contains(" all "), "順勢：所有開在區外的日子")
show((R.family == "順勢突破") & R.name.str.contains("es=1"), "順勢＋ES 同向")
show(R.family == "I2", "I2（全部 288）")
show((R.family == "I2") & R.name.str.contains(" es$"), "I2＋ES 同向")
show((R.family == "I2") & R.name.str.contains("inside_pd"), "I2＋只做前日範圍內")
show((R.family == "I2") & R.name.str.contains(" none$") & R.name.str.contains("最晚89"), "I2 11:00 截止、無過濾")
show((R.family == "I2") & R.name.str.contains(" none$") & R.name.str.contains("最晚120"), "I2 11:30 截止（原本）、無過濾")
print(R[R.name == "I2 最晚120 h1=0 +55/-145 出場179 ha=0 es"][["選擇期 筆", "選擇期 每筆", "開發 2024 每筆", "開發 2025 每筆", "驗證 2026-01~09 每筆"]].round(0).to_string(index=False), "← 對照研究文件的 H3（ES 同步偏離）")

# ES 交叉驗證：同樣參數直接套在 ES（價格 ×3.7 換成 NQ 等值點數）；ES 的「同向確認」改用 NQ
nq = pickle.load(open("NQ_days2.pkl", "rb")); es = pickle.load(open("ES_days2.pkl", "rb"))
NQd = {d["date"]: (-1 if d["O"][0] > d["vah"] else 1 if d["O"][0] < d["val"] else 0) for d in nq if not np.isnan(d["O"][0])}
short = ["I2 最晚120 h1=0 +55/-145 出場179 ha=0 none", "I2 最晚89 h1=1 +55/-145 出場179 ha=0 none", "I2 最晚89 h1=0 +55/-145 出場379 ha=0 es",
         "I2 最晚59 h1=1 +100/-145 出場379 ha=0 inside_pd", "順勢 IB30 SL30-80 2.0R 出場179 outpd be=1 es=0", "順勢 IB30 SL30-80 1.0R 出場179 outpd be=1 es=0",
         "順勢 IB60 SL20-60 2.0R 出場379 outpd be=0 es=0", "回踩 IB30 2.0R 出場179", "ORB15 2.0R gap 出場179", "前低多 +55/-145"]
V = {name: (fam, kw) for fam, name, kw in grid.variants()}
print("\n— 短名單：NQ vs ES、滑價、信賴區間 —")
rng = np.random.default_rng(0)
for nm in short:
    fam, kw = V[nm]; out = {}
    for lab, days, conf in (("NQ", nq, grid.ES), ("ES", es, NQd)):
        k = dict(kw); usees = k.pop("es", False)
        if usees: k["es"] = conf
        tr = [t | {"date": d["date"]} for d in days for t in [grid.FN[fam](d, **k)] if t]
        df = pd.DataFrame(tr); out[lab] = df
    n = out["NQ"]; p = n[n.date <= "2026-09-30"].pnl.values
    bs = rng.choice(p, (10000, len(p))).mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    e = out["ES"]; ey = e.groupby(e.date.str[:4]).pnl.mean() if len(e) else pd.Series(dtype=float)
    print(f"{nm:<48} NQ 每筆 {p.mean():+5.0f} 95%[{lo:+5.0f},{hi:+5.0f}] 滑2點 {p.mean()-80:+5.0f}｜ES 每筆 {e.pnl.mean() if len(e) else 0:+5.0f} "
          f"({' '.join(f'{y}:{v:+.0f}' for y, v in ey.items())})")
