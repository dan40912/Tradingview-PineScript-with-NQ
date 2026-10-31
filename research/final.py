import pickle, numpy as np, pandas as pd, grid
pd.set_option("display.width", 250)
R = pd.read_pickle("grid2.pkl")
x = R[(R.family == "順勢突破") & R.name.str.contains("IB60") & R.name.str.contains("出場379")]
print(f"順勢 IB60、15:49 出場：{len(x)} 個變化，三段都賺 {x['過驗證'].mean():.0%}，報酬回撤比中位數 {x['選擇期 報酬回撤比'].median():.1f}")
nq = pickle.load(open("NQ_days2.pkl", "rb")); es = pickle.load(open("ES_days2.pkl", "rb"))
NQd = {d["date"]: (-1 if d["O"][0] > d["vah"] else 1 if d["O"][0] < d["val"] else 0) for d in nq if not np.isnan(d["O"][0])}
V = {name: (fam, kw) for fam, name, kw in grid.variants()}
I2v1, I2imp, I2es, I2pd = "I2 最晚120 h1=0 +55/-145 出場179 ha=0 none", "I2 最晚89 h1=1 +55/-145 出場179 ha=0 none", "I2 最晚89 h1=0 +55/-145 出場379 ha=0 es", "I2 最晚59 h1=1 +100/-145 出場379 ha=0 inside_pd"
T1, T2, PB = "順勢 IB30 SL30-80 2.0R 出場179 outpd be=1 es=0", "順勢 IB60 SL20-60 2.0R 出場379 outpd be=0 es=0", "回踩 IB30 2.0R 出場179"
PH, PL = "前高空 +55/-145", "前低多 +55/-145"
combos = {"0 I2 v1（原規則）": [I2v1], "0 現在指標（I2＋前高空＋前低多）": [I2v1, PH, PL],
          "A I2改良＋順勢30": [I2imp, T1], "B I2改良＋順勢60": [I2imp, T2], "C I2改良＋順勢30＋順勢60": [I2imp, T1, T2],
          "D I2(ES同向)＋順勢30": [I2es, T1], "E I2(ES同向)＋順勢60": [I2es, T2], "F I2(前日範圍內)＋順勢60＋回踩": [I2pd, T2, PB]}
def trades(nm, days, conf):
    fam, kw = V[nm]; k = dict(kw)
    if k.pop("es", False): k["es"] = conf
    return {d["date"]: t for d in days for t in [grid.FN[fam](d, **k)] if t}
def port(names, days, conf, slip=0.0):
    T = [trades(n, days, conf) for n in names]; out = []
    for d in sorted(set().union(*[t.keys() for t in T])):
        busy, n, pl, losses = -1, 0, 0.0, 0
        for t in sorted([t[d] for t in T if d in t], key=lambda t: t["ei"]):
            if t["ei"] <= busy or n >= 3 or pl <= -3000 or losses >= 2: continue
            t = t | {"date": d, "pnl": t["pnl"] - slip * 40}; out.append(t); busy = t["xi"]; n += 1; pl += t["pnl"]; losses = losses + 1 if t["pnl"] < 0 else 0
    return pd.DataFrame(out)
def mstats(df, a, b):
    x = df[(df.date >= a) & (df.date <= b)]
    if len(x) == 0: return dict(n=0, avg=0, tot=0, dd=0, lm=0, wm=0, wd=0)
    c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    mo = x.groupby(x.date.str[:7]).pnl.sum()
    return dict(n=len(x), avg=x.pnl.mean(), tot=x.pnl.sum(), dd=dd, lm=(mo < 0).sum(), wm=mo.min(), wd=x.groupby("date").pnl.sum().min())
rows = []
for k, names in combos.items():
    n = port(names, nq, grid.ES); e = port(names, es, NQd); n2 = port(names, nq, grid.ES, 2.0)
    s = mstats(n, "2024", "2026-09-30"); o = mstats(n, "2026-10", "2026-10-31"); ee = mstats(e, "2024", "2026-12-31"); s2 = mstats(n2, "2024", "2026-09-30")
    y = n.groupby(n.date.str[:4]).pnl.mean(); ey = e.groupby(e.date.str[:4]).pnl.mean()
    rows.append({"組合": k, "筆": s["n"], "每筆": s["avg"], "合計": s["tot"], "回撤": s["dd"], "報酬/回撤": s["tot"] / s["dd"], "虧錢月/33": s["lm"],
                 "最差月": s["wm"], "最差日": s["wd"], "24": y.get("2024", 0), "25": y.get("2025", 0), "26": y.get("2026", 0),
                 "滑2點每筆": s2["avg"], "ES每筆": ee["avg"], "ES 24/25/26": "/".join(f"{ey.get(z, 0):+.0f}" for z in ("2024", "2025", "2026")),
                 "10月筆": o["n"], "10月合計": o["tot"]})
    pickle.dump(n, open(f"port_{k[0]}{len(names)}.pkl", "wb"))
T = pd.DataFrame(rows); print(T.round(1).to_string(index=False)); T.to_pickle("final.pkl")
