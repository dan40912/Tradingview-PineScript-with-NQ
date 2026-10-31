import pickle, itertools, numpy as np, pandas as pd, strat
from multiprocessing import Pool
ds = pickle.load(open("NQ_days2.pkl", "rb")); es_days = pickle.load(open("ES_days2.pkl", "rb"))
ES = {}
for d in es_days:
    o = d["O"][0]
    if not np.isnan(o): ES[d["date"]] = -1 if o > d["vah"] else 1 if o < d["val"] else 0
P = {"開發 2024": ("2024-01-01", "2024-12-31"), "開發 2025": ("2025-01-01", "2025-12-31"),
     "驗證 2026-01~09": ("2026-01-01", "2026-09-30"), "測試 2026-10": ("2026-10-01", "2026-10-31")}

def variants():
    V = []
    for last, h1, (tp, sl), xi, ha, f in itertools.product((59, 89, 120), (False, True), ((55, 145), (80, 145), (55, 100), (100, 145)),
                                                          (179, 379), (None, (60, 20)), ("none", "inside_pd", "es")):
        V.append(("I2", f"I2 最晚{last} h1={int(h1)} +{tp}/-{sl} 出場{xi} ha={int(bool(ha))} {f}",
                  dict(last=last, h1=h1, tp=tp, sl=sl, xi=xi, ha=ha, inside_pd=f == "inside_pd", es=f == "es")))
    for ib, (a, b), rr, xi, cond, be, es in itertools.product((14, 29, 59), ((20, 60), (30, 80)), (1.0, 1.5, 2.0, 3.0), (179, 379),
                                                            ("any", "outpd", "all"), (None, 1.0), (False, True)):
        V.append(("順勢突破", f"順勢 IB{ib+1} SL{a}-{b} {rr}R 出場{xi} {cond} be={int(bool(be))} es={int(es)}",
                  dict(ib=ib, smin=a, smax=b, rr=rr, xi=xi, cond=cond, be=be, es=es)))
    for ib, rr, xi in itertools.product((29, 59), (1.5, 2.0), (179, 379)):
        V.append(("順勢回踩", f"回踩 IB{ib+1} {rr}R 出場{xi}", dict(ib=ib, rr=rr, xi=xi)))
    for orl, rr, filt, xi in itertools.product((15, 30), (1.0, 1.5, 2.0), ("none", "gap", "vwap", "va"), (179, 379)):
        V.append(("開盤區間突破", f"ORB{orl} {rr}R {filt} 出場{xi}", dict(orl=orl, rr=rr, filt=filt, xi=xi)))
    for ext, rr, xi in itertools.product((5, 15), (1.0, 1.5, 2.0), (179, 379)):
        V.append(("IB假突破", f"IB假突破 ext{ext} {rr}R 出場{xi}", dict(ext=ext, rr=rr, xi=xi)))
    for rr, xi in itertools.product((1.5, 2.0, 3.0), (179, 379)):
        V.append(("VWAP回踩", f"VWAP回踩 {rr}R 出場{xi}", dict(rr=rr, xi=xi)))
    for w, (tp, sl) in itertools.product((0, 1), ((55, 145), (55, 100), (80, 145))):
        V.append(("前日高低假突破", f"{'前高空' if w == 0 else '前低多'} +{tp}/-{sl}", dict(which=w, tp=tp, sl=sl)))
    return V

FN = {"I2": strat.i2, "順勢突破": strat.trend, "順勢回踩": strat.pullback, "開盤區間突破": strat.orb,
      "IB假突破": strat.ibfade, "VWAP回踩": strat.vwap_pb, "前日高低假突破": strat.pdfade}

def trades(fam, kw):
    kw = dict(kw)
    if kw.pop("es", False): kw["es"] = ES
    elif "es" in kw: kw.pop("es")
    f = FN[fam]; out = []
    for d in ds:
        t = f(d, **kw)
        if t: out.append(t | {"date": d["date"]})
    return out

def stats(p):
    if len(p) == 0: return 0, 0.0, 0.0, 0.0
    cum = np.cumsum(p); dd = (np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()
    return len(p), p.mean(), p.sum(), dd

def evaluate(v):
    fam, name, kw = v; tr = trades(fam, kw)
    df = pd.DataFrame(tr) if tr else pd.DataFrame(columns=["date", "pnl"])
    row = dict(family=fam, name=name)
    for k, (a, b) in P.items():
        n, m, s, dd = stats(df[(df.date >= a) & (df.date <= b)].pnl.values if len(df) else np.array([]))
        row[f"{k} 筆"] = n; row[f"{k} 每筆"] = m; row[f"{k} 合計"] = s; row[f"{k} 回撤"] = dd
    sel = df[df.date <= "2026-09-30"].pnl.values if len(df) else np.array([])
    n, m, s, dd = stats(sel); row.update({"選擇期 筆": n, "選擇期 每筆": m, "選擇期 合計": s, "選擇期 回撤": dd,
                                          "選擇期 報酬回撤比": s / dd if dd > 0 else np.nan,
                                          "選擇期 t": m / (sel.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan})
    return row

if __name__ == "__main__":
    V = variants(); print("假設數", len(V))
    with Pool(4) as pool: rows = pool.map(evaluate, V, chunksize=8)
    R = pd.DataFrame(rows)
    R["過開發"] = (R["開發 2024 每筆"] > 0) & (R["開發 2025 每筆"] > 0) & ((R["開發 2024 筆"] + R["開發 2025 筆"]) >= 30)
    R["過驗證"] = R["過開發"] & (R["驗證 2026-01~09 每筆"] > 0)
    R.to_pickle("grid2.pkl"); print("完成")
