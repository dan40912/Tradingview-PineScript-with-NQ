"""NQ 趨勢 scanner：開盤前只用「前一天以前」的日線收盤，判斷大方向，再測它能不能改善組合 D。
做法和 docs/11 相同：2024–2025 開發、2026-01～09 驗證，選完才看 2026-10。"""
import pickle, numpy as np, pandas as pd, grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb"))
closes = np.array([d["C"][~np.isnan(d["C"])][-1] for d in nq])
def rsi(x, n=14):
    d = np.diff(x); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0); a = up[:n].mean(); b = dn[:n].mean(); out = [np.nan] * (n + 1)
    for u, v in zip(up[n:], dn[n:]): a = (a * (n - 1) + u) / n; b = (b * (n - 1) + v) / n; out.append(100 - 100 / (1 + a / b) if b > 0 else 100)
    return np.array(out)
R14 = rsi(closes)
F = {}
for k, d in enumerate(nq):
    if k < 51: continue
    c = closes[:k]                                     # 只用前一天以前的收盤（換月價差會讓均線有一點誤差）
    er = abs(c[-1] - c[-11]) / np.abs(np.diff(c[-11:])).sum()
    F[d["date"]] = dict(ma20=1 if c[-1] > c[-20:].mean() else -1, ma50=1 if c[-1] > c[-50:].mean() else -1,
                        mom5=1 if c[-1] > c[-6] else -1, rsi=R14[k - 1], er=er,
                        hi20=1 if c[-1] >= c[-20:].max() else -1 if c[-1] <= c[-20:].min() else 0)
I2k = dict(last=89, h1=False, tp=55, sl=145, xi=379, es=grid.ES)
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
A = {d["date"]: t | {"date": d["date"]} for d in nq for t in [strat.i2(d, **I2k)] if t and d["date"] in F}
B = {d["date"]: t | {"date": d["date"]} for d in nq for t in [strat.trend(d, **T1)] if t and d["date"] in F}
def port(fa, fb):
    out = []
    for d in sorted(set(A) | set(B)):
        busy, n, pl, losses = -1, 0, 0.0, 0
        for t, ok in sorted([(A[d], fa(A[d], F[d])) for _ in [0] if d in A] + [(B[d], fb(B[d], F[d])) for _ in [0] if d in B], key=lambda z: z[0]["ei"]):
            if not ok or t["ei"] <= busy or n >= 3 or pl <= -3000 or losses >= 2: continue
            out.append(t); busy = t["xi"]; n += 1; pl += t["pnl"]; losses += t["pnl"] < 0
    return pd.DataFrame(out)
def per(df, a, b):
    x = df[(df.date >= a) & (df.date <= b)]
    if len(x) == 0: return 0, 0.0, 0.0, 0.0
    c = x.pnl.cumsum().values; return len(x), x.pnl.mean(), x.pnl.sum(), (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
yes = lambda t, f: True
H = {
    "D 基準（沒有 scanner）": (yes, yes),
    "S1 順勢只做 20 日均線方向": (yes, lambda t, f: t["side"] == f["ma20"]),
    "S2 順勢只做 50 日均線方向": (yes, lambda t, f: t["side"] == f["ma50"]),
    "S3 順勢只做 5 日動能方向": (yes, lambda t, f: t["side"] == f["mom5"]),
    "S4 I2 只做 50 日均線方向": (lambda t, f: t["side"] == f["ma50"], yes),
    "S5 I2 和順勢都只做 50 日均線方向": (lambda t, f: t["side"] == f["ma50"], lambda t, f: t["side"] == f["ma50"]),
    "S6 RSI 過熱不追（多單 RSI>70、空單 RSI<30 不做順勢）": (yes, lambda t, f: not ((t["side"] == 1 and f["rsi"] > 70) or (t["side"] == -1 and f["rsi"] < 30))),
    "S7 RSI 過熱才做 I2（多 RSI<40、空 RSI>60）": (lambda t, f: (t["side"] == 1 and f["rsi"] < 40) or (t["side"] == -1 and f["rsi"] > 60), yes),
    "S8 趨勢盤（效率比 ER≥0.3）才做順勢": (yes, lambda t, f: f["er"] >= 0.3),
    "S9 盤整盤（ER<0.3）才做 I2、趨勢盤才做順勢": (lambda t, f: f["er"] < 0.3, lambda t, f: f["er"] >= 0.3),
    "S10 順勢只在 20 日新高（做多）／新低（做空）": (yes, lambda t, f: f["hi20"] == t["side"]),
    "S11 順勢不做逆 20 日新高新低的方向": (yes, lambda t, f: f["hi20"] != -t["side"]),
}
rows = []
for name, (fa, fb) in H.items():
    df = port(fa, fb); r = {"假設": name}
    for lab, (a, b) in {"24": ("2024", "2024-12-31"), "25": ("2025", "2025-12-31"), "26驗證": ("2026", "2026-09-30"), "選擇期": ("2024", "2026-09-30"), "10月": ("2026-10", "2026-10-31")}.items():
        n, m, s, dd = per(df, a, b); r[f"{lab}筆"] = n; r[f"{lab}每筆"] = round(m); r[f"{lab}合計"] = round(s); r[f"{lab}回撤"] = round(dd)
    r["報酬/回撤"] = round(r["選擇期合計"] / max(r["選擇期回撤"], 1), 1)
    r["過"] = "✓" if r["24每筆"] > 0 and r["25每筆"] > 0 and r["26驗證每筆"] > 0 and r["24合計"] + r["25合計"] >= per(port(yes, yes), "2024", "2025-12-31")[2] * 0 else "✗"
    rows.append(r)
T = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(T[["假設", "選擇期筆", "選擇期每筆", "選擇期合計", "選擇期回撤", "報酬/回撤", "24每筆", "25每筆", "26驗證每筆", "10月筆", "10月合計"]].to_string(index=False))
base = T.iloc[0]
print("\n開發期（2024–2025）比基準好、驗證期也比基準好的：")
for _, r in T.iloc[1:].iterrows():
    dev_b = (r["24合計"] + r["25合計"]) / max(1, 1) - (base["24合計"] + base["25合計"]); dev_r = ((r["24合計"] + r["25合計"]) / max(per(port(*H[r['假設']]), "2024", "2025-12-31")[3], 1)) - ((base["24合計"] + base["25合計"]) / max(per(port(yes, yes), "2024", "2025-12-31")[3], 1))
    val_r = r["26驗證合計"] / max(r["26驗證回撤"], 1) - base["26驗證合計"] / max(base["26驗證回撤"], 1)
    print(f"  {r['假設']:<46} 開發期報酬回撤比 {dev_r:+5.1f}｜驗證期報酬回撤比 {val_r:+5.1f} {'← 兩段都改善' if dev_r > 0 and val_r > 0 else ''}")
print("\n10 月每天 scanner 的判斷：")
for d in nq:
    if d["date"] >= "2026-10-01":
        f = F[d["date"]]; print(f"  {d['date']} 20日均線 {'上' if f['ma20'] > 0 else '下'}｜50日均線 {'上' if f['ma50'] > 0 else '下'}｜5日動能 {'+' if f['mom5'] > 0 else '−'}｜RSI {f['rsi']:4.0f}｜ER {f['er']:.2f}｜20日 {'新高' if f['hi20'] > 0 else '新低' if f['hi20'] < 0 else '—'}")
