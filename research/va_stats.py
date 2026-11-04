"""價值區的行為統計（2024-01～2026-10-19，主檔 690 天）：開盤位置、回到價值區的機率、80% 法則、區寬、POC。"""
import pickle, numpy as np, pandas as pd
nq = pickle.load(open("NQ_days2.pkl", "rb")); rows = []
for d in nq:
    O, H, L, C = d["O"], d["H"], d["L"], d["C"]
    if np.isnan(O[0]): continue
    o, val, vah, poc = O[0], d["val"], d["vah"], d["poc"]; w = vah - val
    side = 1 if o > vah else -1 if o < val else 0; dist = (o - vah) if side == 1 else (val - o) if side == -1 else 0
    closes5 = [(i, C[i]) for i in range(4, 390, 5) if not np.isnan(C[i])]
    inside5 = [val <= c <= vah for _, c in closes5]
    trig = next((closes5[k][0] for k in range(1, len(closes5)) if inside5[k] and inside5[k - 1]), None)   # 連續 2 根 5 分K 收在區內
    touch = (np.nanmax(H) >= val) if side == -1 else (np.nanmin(L) <= vah) if side == 1 else True
    trav = None
    if side != 0 and trig is not None and trig < 389:
        after = slice(trig + 1, 390)
        trav = (np.nanmax(H[after]) >= vah) if side == -1 else (np.nanmin(L[after]) <= val)     # 進區後走到另一邊
        to_poc = (np.nanmax(H[after]) >= poc) if side == -1 else (np.nanmin(L[after]) <= poc)
    else: to_poc = None
    last = C[~np.isnan(C)][-1]; rng = np.nanmax(H) - np.nanmin(L)
    out5 = [c for _, c in closes5 if not (val <= c <= vah)]
    rows.append(dict(date=d["date"], side=side, w=w, wpct=w / o, dist_w=dist / w if w else np.nan, touch=touch, trig=trig is not None, trav=trav, to_poc=to_poc,
                     poc_touch=(np.nanmin(L) <= poc <= np.nanmax(H)), close_in=val <= last <= vah, broke=len(out5) > 0,
                     rng_w=rng / w, rng=rng, o=o))
X = pd.DataFrame(rows); n = len(X)
print(f"共 {n} 天")
print("\n1) 開盤位置：", {k: f"{v:.0%}" for k, v in X.side.map({1: "VAH 之上", 0: "區內", -1: "VAL 之下"}).value_counts(normalize=True).items()})
out = X[X.side != 0]
print(f"\n2) 開在區外（{len(out)} 天）：當天碰到價值區 {out.touch.mean():.0%}｜連續 2 根 5 分K 收回區內 {out.trig.mean():.0%}")
t = out[out.trig]
print(f"   收回區內之後：走到另一邊（80% 法則）{t.trav.astype(float).mean():.0%}｜碰到 POC {t.to_poc.astype(float).mean():.0%}")
print("   依「開盤離區距離 ÷ 區寬」分組：")
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 99)):
    g = out[(out.dist_w > lo) & (out.dist_w <= hi)]
    tt = g[g.trig]
    print(f"     {lo}～{hi if hi < 99 else '∞'} 倍：{len(g):3d} 天｜碰到價值區 {g.touch.mean():.0%}｜收回區內 {g.trig.mean():.0%}｜收回後走到另一邊 {tt.trav.astype(float).mean() if len(tt) else np.nan:.0%}")
ins = X[X.side == 0]
print(f"\n3) 開在區內（{len(ins)} 天）：碰到 POC {ins.poc_touch.mean():.0%}｜盤中有 5 分K 收到區外 {ins.broke.mean():.0%}｜收盤仍在區內 {ins.close_in.mean():.0%}")
print("\n4) 區寬（佔價格 %）分四組，當天全天振幅是區寬的幾倍、開在區外的比例：")
X["q"] = pd.qcut(X.wpct, 4, labels=["最窄 25%", "偏窄", "偏寬", "最寬 25%"])
for q, g in X.groupby("q", observed=True):
    print(f"   {q}：區寬 {g.w.min():.0f}～{g.w.max():.0f} 點（{g.wpct.min():.2%}～{g.wpct.max():.2%}）｜全天振幅中位數 {g.rng.median():.0f} 點＝區寬 {g.rng_w.median():.1f} 倍｜開在區外 {(g.side != 0).mean():.0%}｜開在區內時收盤仍在區內 {g[g.side == 0].close_in.mean():.0%}")
print(f"\n區寬分位數：10% {X.w.quantile(.1):.0f}、25% {X.w.quantile(.25):.0f}、50% {X.w.quantile(.5):.0f}、75% {X.w.quantile(.75):.0f}、90% {X.w.quantile(.9):.0f} 點")
pickle.dump(X, open("va_stats.pkl", "wb"))
