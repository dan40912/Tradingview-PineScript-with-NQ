import pickle, itertools, numpy as np, pandas as pd, grid, strat
R = pd.read_pickle("grid2.pkl"); V = {name: (fam, kw) for fam, name, kw in grid.variants()}
S = R[R["過驗證"]].sort_values("選擇期 報酬回撤比", ascending=False).drop_duplicates(subset=["選擇期 合計", "選擇期 筆", "family"])
pick = {f: list(S[S.family == f].name.head(k)) for f, k in (("I2", 6), ("順勢突破", 6), ("順勢回踩", 2), ("開盤區間突破", 2), ("前日高低假突破", 1))}
cache = {}
def tr(name):
    if name not in cache:
        fam, kw = V[name]; cache[name] = {t["date"]: t for t in grid.trades(fam, kw)}
    return cache[name]
def portfolio(names, max_n=3, day_stop=-3000):
    out = []
    dates = sorted(set().union(*[tr(n).keys() for n in names]))
    for d in dates:
        cand = sorted([tr(n)[d] for n in names if d in tr(n)], key=lambda t: t["ei"])
        busy, n, pl, losses = -1, 0, 0.0, 0
        for t in cand:
            if t["ei"] <= busy or n >= max_n or pl <= day_stop or losses >= 2: continue
            out.append(t | {"date": d}); busy = t["xi"]; n += 1; pl += t["pnl"]; losses = losses + 1 if t["pnl"] < 0 else 0
    return pd.DataFrame(out)
def st(df, a, b):
    x = df[(df.date >= a) & (df.date <= b)].pnl.values
    if len(x) == 0: return 0, 0, 0, 0
    c = np.cumsum(x); dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max(); return len(x), x.mean(), x.sum(), dd
rows = []
fams = [f for f in pick if pick[f]]
for k in (1, 2, 3):
    for fs in itertools.combinations(fams, k):
        for names in itertools.product(*[pick[f] for f in fs]):
            df = portfolio(list(names))
            r = dict(combo=" ＋ ".join(names), k=k)
            for lab, (a, b) in {"24": ("2024", "2024-12-31"), "25": ("2025", "2025-12-31"), "26驗證": ("2026", "2026-09-30"), "選擇期": ("2024", "2026-09-30"), "10月": ("2026-10", "2026-10-31")}.items():
                n, m, s, dd = st(df, a, b); r[f"{lab}筆"] = n; r[f"{lab}每筆"] = m; r[f"{lab}合計"] = s; r[f"{lab}回撤"] = dd
            sel = df[df.date <= "2026-09-30"]; mo = sel.groupby(sel.date.str[:7]).pnl.sum().reindex(pd.period_range("2024-01", "2026-09", freq="M").astype(str), fill_value=0)
            r["虧錢月"] = int((mo < 0).sum()); r["最差月"] = mo.min(); r["報酬回撤比"] = r["選擇期合計"] / max(r["選擇期回撤"], 1)
            r["過"] = r["24每筆"] > 0 and r["25每筆"] > 0 and r["26驗證每筆"] > 0
            rows.append(r)
C = pd.DataFrame(rows); C.to_pickle("combo.pkl")
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 140)
C = C[C["過"]].sort_values("報酬回撤比", ascending=False)
print("組合數", len(rows), "通過", len(C))
for k in (1, 2, 3):
    print(f"\n== {k} 個策略的組合，前 5 名 ==")
    print(C[C.k == k].head(5)[["combo", "選擇期筆", "選擇期每筆", "選擇期合計", "選擇期回撤", "報酬回撤比", "虧錢月", "最差月", "24每筆", "25每筆", "26驗證每筆", "10月筆", "10月合計"]].round(0).to_string(index=False))
