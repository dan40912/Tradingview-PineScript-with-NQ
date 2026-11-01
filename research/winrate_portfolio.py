import pickle, numpy as np, pandas as pd, winrate_opt as W
R = pd.read_pickle("winrate_grid.pkl")
print("=== 單一方法的效果（其他參數都一樣、只改這一項，所有配對的中位數變化）===")
def pair_effect(fam, a, b, label):
    x = R[R.family == fam]; m = x[x.name.str.contains(a, regex=False)].copy(); m["key"] = m.name.str.replace(a, "§", regex=False)
    n = x[x.name.str.contains(b, regex=False)].copy(); n["key"] = n.name.str.replace(b, "§", regex=False)
    j = m.merge(n, on="key", suffixes=("_a", "_b"))
    print(f"  {label:<36} 配對 {len(j):4d}｜勝率 {np.nanmedian(j['選擇期勝率_b'] - j['選擇期勝率_a']):+.1%}｜每筆 {np.nanmedian(j['選擇期每筆_b'] - j['選擇期每筆_a']):+5.0f}｜筆數 {np.nanmedian(j['選擇期筆_b'] - j['選擇期筆_a']):+4.0f}")
pair_effect("I2", "rvol≥0 ", "rvol≥1.0 ", "I2 加成交量確認（RVOL≥1）")
pair_effect("順勢突破", "rvol≥0 ", "rvol≥1.0 ", "順勢加成交量確認（RVOL≥1）")
pair_effect("順勢突破", "rvol≥0 ", "rvol≥1.5 ", "順勢加成交量確認（RVOL≥1.5）")
pair_effect("參考點", "rvol≥0 ", "rvol≥1.0 ", "參考點加成交量確認（RVOL≥1）")
pair_effect("順勢突破", "停損ib ", "停損atr1.5 ", "順勢停損 IB 中點 → ATR×1.5")
pair_effect("順勢突破", "停損ib ", "停損atr2.5 ", "順勢停損 IB 中點 → ATR×2.5")
pair_effect("I2", "es=0 ", "es=1 ", "I2 加 ES 同向")
pair_effect("I2", "orb=0 ", "orb=1 ", "I2 加 ORB15 不反向")
pair_effect("I2", "vwap=0 ", "vwap=1 ", "I2 加 VWAP 同方向")
pair_effect("I2", "h1=0 ", "h1=1 ", "I2 加 H1")
pair_effect("參考點", "vwap=0 ", "vwap=1 ", "參考點加 VWAP 同方向")
pair_effect("I2", "+55/-145", "+30/-145", "I2 止盈 55 → 30")
pair_effect("I2", "出場179", "出場379", "I2 出場 12:29 → 15:49")

cfg = pickle.load(open("winrate_cfg.pkl", "rb")); nq = W.nq
REFS = dict(which=0, vw=True, vol=0, tp=30, sl=145, be=None)
def gen(fam, kw): return {d["date"]: t | {"date": d["date"]} for d in nq for t in [W.FN[fam](d, **kw)] if t}
G = {"I2 高勝率": gen("I2", cfg["I2H"]), "I2 D-1": gen("I2", cfg["I2D"]), "順勢 0.5R": gen("順勢突破", cfg["TRH"]), "順勢 2R": gen("順勢突破", cfg["TRD"]),
     "前低多 VWAP": gen("參考點", cfg["REF"]), "前高空 VWAP": gen("參考點", REFS)}
def port(names):
    out = []
    for d in sorted(set().union(*[G[n].keys() for n in names])):
        busy, k, pl, losses = -1, 0, 0.0, 0
        for t in sorted([G[n][d] for n in names if d in G[n]], key=lambda t: t["ei"]):
            if t["ei"] <= busy or k >= 3 or pl <= -3000 or losses >= 2: continue
            out.append(t); busy = t["xi"]; k += 1; pl += t["pnl"]; losses += t["pnl"] < 0
    return pd.DataFrame(out)
print("\n=== 組合（同時只持 1 筆、每天最多 3 筆、連 2 虧或 −$3,000 停）===")
rows = []
for lab, names in (("D（目前建議）", ["I2 D-1", "順勢 2R"]), ("H1 只做 I2 高勝率", ["I2 高勝率"]), ("H2 I2 高勝率＋順勢 2R", ["I2 高勝率", "順勢 2R"]),
                   ("H3 I2 高勝率＋前低多 VWAP", ["I2 高勝率", "前低多 VWAP"]), ("H4 I2 高勝率＋順勢 0.5R", ["I2 高勝率", "順勢 0.5R"]),
                   ("H5 I2 高勝率＋前低多 VWAP＋順勢 0.5R", ["I2 高勝率", "前低多 VWAP", "順勢 0.5R"]),
                   ("H6 I2 高勝率＋前低多＋前高空 VWAP＋順勢 0.5R", ["I2 高勝率", "前低多 VWAP", "前高空 VWAP", "順勢 0.5R"])):
    df = port(names); x = df[df.date <= "2026-09-30"]; c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    yw = x.groupby(x.date.str[:4]).pnl.apply(lambda s: (s > 0).mean()); ye = x.groupby(x.date.str[:4]).pnl.mean(); o = df[df.date >= "2026-10"]
    mo = x.groupby(x.date.str[:7]).pnl.sum(); streak = max((len(list(g)) for v, g in __import__("itertools").groupby(x.pnl < 0) if v), default=0)
    rows.append({"組合": lab, "筆": len(x), "每月": round(len(x) / 33, 1), "勝率": f"{(x.pnl > 0).mean():.1%}", "每筆": round(x.pnl.mean()), "合計": round(x.sum().pnl if False else x.pnl.sum()),
                 "回撤": round(dd), "報酬/回撤": round(x.pnl.sum() / dd, 1), "最多連虧": streak, "虧錢月": int((mo < 0).sum()),
                 "勝率 24/25/26": "/".join(f"{v:.0%}" for v in yw), "每筆 24/25/26": "/".join(f"{v:+.0f}" for v in ye),
                 "10月": f"{len(o)} 筆 {int((o.pnl > 0).sum())} 勝 {o.pnl.sum():+,.0f}"})
pd.set_option("display.width", 260); print(pd.DataFrame(rows).to_string(index=False))
print("\n10 月逐筆（H3、H5）：")
for lab, names in (("H3", ["I2 高勝率", "前低多 VWAP"]), ("H5", ["I2 高勝率", "前低多 VWAP", "順勢 0.5R"])):
    o = port(names); o = o[o.date >= "2026-10"]
    for _, t in o.iterrows(): print(f"  {lab} {t.date} {t.setup} {'多' if t.side == 1 else '空'} {9 + (30 + t.ei) // 60:02d}:{(30 + t.ei) % 60:02d} {t.res} {t.pnl:+,.0f}")

print("\n=== D 加上單獨有效的小改良 ===")
G["I2 D-1＋ORB"] = gen("I2", dict(cfg["I2D"], orbf=True))
G["順勢 2R ATR2.5"] = gen("順勢突破", dict(cfg["TRD"], stop="atr2.5"))
rows = []
for lab, names in (("D（目前建議）", ["I2 D-1", "順勢 2R"]), ("D＋I2 加 ORB15 不反向", ["I2 D-1＋ORB", "順勢 2R"]),
                   ("D＋順勢停損改 ATR×2.5", ["I2 D-1", "順勢 2R ATR2.5"]), ("D＋兩個都改", ["I2 D-1＋ORB", "順勢 2R ATR2.5"])):
    df = port(names); x = df[df.date <= "2026-09-30"]; c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    yw = x.groupby(x.date.str[:4]).pnl.apply(lambda s: (s > 0).mean()); ye = x.groupby(x.date.str[:4]).pnl.mean(); o = df[df.date >= "2026-10"]
    rows.append({"組合": lab, "筆": len(x), "勝率": f"{(x.pnl > 0).mean():.1%}", "每筆": round(x.pnl.mean()), "合計": round(x.pnl.sum()), "回撤": round(dd),
                 "報酬/回撤": round(x.pnl.sum() / dd, 1), "勝率 24/25/26": "/".join(f"{v:.0%}" for v in yw), "每筆 24/25/26": "/".join(f"{v:+.0f}" for v in ye),
                 "10月": f"{len(o)} 筆 {int((o.pnl > 0).sum())} 勝 {o.pnl.sum():+,.0f}"})
print(pd.DataFrame(rows).to_string(index=False))
print("\n=== 示範：如果直接用 10 月挑「勝率最高」（10 月 ≥ 3 筆）===")
o = R[R["10月筆"] >= 3].sort_values(["10月勝率", "10月合計"], ascending=False).drop_duplicates(subset=["10月筆", "10月合計", "選擇期合計"]).head(5)
print(o[["name", "10月筆", "10月勝率", "10月合計", "2024每筆", "2025每筆", "2026驗證每筆", "選擇期勝率"]].round(2).to_string(index=False))
