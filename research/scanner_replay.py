"""每一次進場訊號出現時，跑 scanner（day-trader-skill 的型態掃描＋Time Machine 勝算判斷），
畫出當下的圖，再對照實際結果。2026-10 逐筆畫圖；2024-01～2026-10-19 全部交易統計 scanner 判斷準不準。"""
import os, pickle, numpy as np, pandas as pd, grid, strat
import logging, matplotlib; matplotlib.use("Agg"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
from eng import REPO
nq = pickle.load(open("NQ_days2.pkl", "rb")); IDX = {d["date"]: k for k, d in enumerate(nq)}
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
RULES = {  # 名稱: (函式, 出場上限分鐘, TP 點數規則)
    "I2 v1（現在指標）": (lambda d: strat.i2(d), 179, lambda t: 55),
    "前高空（現在指標）": (lambda d: strat.pdfade(d, which=0), 179, lambda t: 55),
    "前低多（現在指標）": (lambda d: strat.pdfade(d, which=1), 179, lambda t: 55),
    "D-1 I2（ES 同向）": (lambda d: strat.i2(d, last=89, tp=55, sl=145, xi=379, es=grid.ES), 379, lambda t: 55),
    "D-2 順勢突破": (lambda d: strat.trend(d, **T1), 179, lambda t: 2 * t["risk"]),
}
# ---------- 5 分K ----------
def bars5(d):
    out = np.full((78, 4), np.nan)
    for b in range(78):
        s = slice(b * 5, b * 5 + 5); o, h, l, c = d["O"][s], d["H"][s], d["L"][s], d["C"][s]
        ok = ~np.isnan(c)
        if ok.any(): out[b] = [o[ok][0], np.nanmax(h), np.nanmin(l), c[ok][-1]]
    return out
B5 = [bars5(d) for d in nq]; FULL = [k for k, b in enumerate(B5) if not np.isnan(b).any()]
CL = np.array([d["C"][~np.isnan(d["C"])][-1] for d in nq])
# ---------- scanner 1：day-trader-skill 的日線型態（只用前一天以前的收盤） ----------
def rsi_series(x, n=14):
    d = np.diff(x); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0); a, b = up[:n].mean(), dn[:n].mean(); out = [np.nan] * (n + 1)
    for u, v in zip(up[n:], dn[n:]): a = (a * (n - 1) + u) / n; b = (b * (n - 1) + v) / n; out.append(100 - 100 / (1 + a / b) if b > 0 else 100.0)
    return np.array(out)
RSI = rsi_series(CL)
def daily_scan(k):
    if k < 22: return [], 0
    c = CL[:k]; ma = c[-20:].mean(); ma_prev = c[-21:-1].mean(); r, r0 = RSI[k - 1], RSI[k - 2]; hits = []
    if c[-1] > c[-21:-1].max(): hits.append(("20日收盤新高", 1))
    if c[-1] < c[-21:-1].min(): hits.append(("20日收盤新低", -1))
    if c[-2] < ma_prev and c[-1] > ma: hits.append(("站回20日均線", 1))
    if c[-2] > ma_prev and c[-1] < ma: hits.append(("跌破20日均線", -1))
    if k >= 16 and c[-6] / c[-16] > 1.10 and c[-5:].max() / c[-5:].min() < 1.05: hits.append(("多頭旗形", 1))
    if r < 30: hits.append((f"超賣 RSI {r:.0f}", 1))
    if r0 < 30 and c[-1] > c[-2]: hits.append(("超賣反彈", 1))
    if r > 70: hits.append((f"超買 RSI {r:.0f}", -1))
    return hits, int(np.sign(sum(s for _, s in hits)))
# ---------- scanner 2：15 分開盤區間突破（trader-scanner orb），只看到訊號那根為止 ----------
def orb_scan(d, upto):
    H, L, C = d["H"], d["L"], d["C"]; orh, orl = np.nanmax(H[:15]), np.nanmin(L[:15])
    for i in range(19, upto + 1, 5):
        if np.isnan(C[i]): continue
        if C[i] > orh: return 1, f"{9 + (35 + i) // 60:02d}:{(35 + i) % 60:02d} 向上突破", orh, orl
        if C[i] < orl: return -1, f"{9 + (35 + i) // 60:02d}:{(35 + i) % 60:02d} 向下跌破", orh, orl
    return 0, "還在區間內", orh, orl
# ---------- scanner 3：Time Machine（前 60 個完整交易日、同時段 ±2 根 5 分K 隨機重組 2,000 次） ----------
def time_machine(k, ei, side, tp, sl, cap, n=2000, seed=19):
    pool = [j for j in FULL if j < k][-60:]
    if len(pool) < 20: return None
    P = np.stack([B5[j] for j in pool]); prevc = np.concatenate([np.full((len(pool), 1, 1), np.nan), P[:, :-1, 3:4]], 1); R = P / prevc
    rng = np.random.default_rng(seed); e = nq[k]["O"][ei]; price = np.full(n, e); done = np.zeros(n, bool); res = np.zeros(n)
    for b in range(ei // 5, min(cap // 5, 77) + 1):
        j = np.clip(b + rng.integers(-2, 3, n), 1, 77); r = R[rng.integers(0, len(pool), n), j]
        h, l, c = price * r[:, 1], price * r[:, 2], price * r[:, 3]
        hs = ~done & ((l <= e - sl) if side == 1 else (h >= e + sl)); res[hs] = -sl; done |= hs
        ht = ~done & ((h >= e + tp) if side == 1 else (l <= e - tp)); res[ht] = tp; done |= ht; price = c
    res[~done] = side * (price[~done] - e)
    ptp, psl = (res == tp).mean(), (res == -sl).mean(); share = ptp / (ptp + psl) if ptp + psl > 0 else np.nan
    diff = (share - 1 / (1 + tp / sl)) * 100     # 和損益兩平所需勝率差幾個百分點（day-trader-skill 的判斷方式）
    verdict = "接近損益兩平" if abs(diff) < 5 else ("偏向目標" if diff < 15 else "強烈偏向目標") if diff > 0 else ("偏向停損" if diff > -15 else "強烈偏向停損")
    return dict(tm_tp=ptp, tm_sl=psl, tm_ev=res.mean() * 20 - 48, tm_diff=diff, tm_verdict=verdict)
# ---------- 每一筆進場 ----------
hh = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"
rows = []
for name, (f, cap, tpf) in RULES.items():
    for d in nq:
        t = f(d)
        if not t: continue
        k = IDX[d["date"]]; tp = tpf(t); hits, dbias = daily_scan(k); od, otxt, orh, orl = orb_scan(d, t["ei"] - 1)
        tm = time_machine(k, t["ei"], t["side"], tp, t["risk"], cap) or {}
        vw = d["vwap"][t["ei"] - 1]
        rows.append(dict(rule=name, date=d["date"], side=t["side"], ei=t["ei"], xi=t["xi"], entry=d["O"][t["ei"]], tp=tp, sl=t["risk"], res=t["res"], pnl=t["pnl"],
                         daily=" ".join(h for h, _ in hits) or "沒有型態", daily_bias=dbias, orb=od, orb_txt=otxt, orh=orh, orl=orl,
                         vwap_side=int(np.sign(d["C"][t["ei"] - 1] - vw)), **tm))
X = pd.DataFrame(rows); X.to_pickle("scanner_replay.pkl")
X["方向一致_ORB"] = np.where(X.orb == 0, "ORB 未突破", np.where(X.orb == X.side, "同方向", "反方向"))
X["方向一致_日線"] = np.where(X.daily_bias == 0, "日線沒有偏向", np.where(X.daily_bias == X.side, "同方向", "反方向"))
X["方向一致_VWAP"] = np.where(X.vwap_side == X.side, "同方向", "反方向")
# ---------- 2026-10 逐筆 ----------
O = X[X.date >= "2026-10-01"].sort_values(["date", "ei"])
print(f"2026-10（到 {nq[-1]['date']}）每一次進場訊號：scanner 當下的判斷 vs 實際結果\n")
for _, r in O.iterrows():
    print(f"{r.date} {hh(r.ei)} {r.rule:<14} {'多' if r.side == 1 else '空'}｜日線型態：{r.daily}｜ORB15：{r.orb_txt}｜VWAP：{'上' if r.vwap_side > 0 else '下'}"
          f"｜Time Machine：先碰TP {r.tm_tp:.0%}、先碰SL {r.tm_sl:.0%}、期望值 {r.tm_ev:+,.0f} → {r.tm_verdict}｜實際：{r.res} {r.pnl:+,.0f}")
# ---------- 全部歷史：scanner 判斷 vs 結局 ----------
H = X[X.date <= "2026-10-19"]
def tab(col):
    g = H.groupby(["rule", col]).agg(筆=("pnl", "size"), 每筆=("pnl", "mean"), 止盈率=("res", lambda s: (s == "TP").mean())).round(2)
    return g
pd.set_option("display.width", 200)
for col, title in (("tm_verdict", "Time Machine 判斷"), ("方向一致_ORB", "和 15 分開盤區間突破的方向"), ("方向一致_日線", "和日線型態的偏向"), ("方向一致_VWAP", "和 VWAP 的位置")):
    print(f"\n=== 全部歷史（2024-01～2026-10-19）：依「{title}」分組 ===\n{tab(col).to_string()}")
# ---------- 畫圖 ----------
out = f"{REPO}/images/scanner_202610"; os.makedirs(out, exist_ok=True)
for f_ in os.listdir(out): os.remove(os.path.join(out, f_))
files = []
for n_, (_, r) in enumerate(O.iterrows(), 1):
    d = nq[IDX[r.date]]; b = B5[IDX[r.date]]; fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    for i, (o, h, l, c) in enumerate(b):
        if np.isnan(o): continue
        col = "#2e9e5b" if c >= o else "#d64545"; ax.plot([i, i], [l, h], color=col, lw=0.8)
        ax.add_patch(Rectangle((i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.25), color=col))
    ax.axhline(d["vah"], color="#e08a1e", lw=1.6); ax.axhline(d["val"], color="#1a9c9c", lw=1.6); ax.axhline(d["poc"], color="gray", lw=0.8)
    ax.axhline(d["pdh"], color="#d64545", lw=0.8, ls=":"); ax.axhline(d["pdl"], color="#2e9e5b", lw=0.8, ls=":")
    ax.axhspan(d["val"], d["vah"], color="gray", alpha=0.07)
    ibh, ibl = np.nanmax(d["H"][:30]), np.nanmin(d["L"][:30]); ax.add_patch(Rectangle((-0.5, ibl), 6, ibh - ibl, fill=False, ls="--", ec="#7a5cc7", lw=1))
    ax.plot([-0.5, 2.5], [r.orh, r.orh], color="#7a5cc7", lw=2); ax.plot([-0.5, 2.5], [r.orl, r.orl], color="#7a5cc7", lw=2)
    ax.plot(np.arange(78), [d["vwap"][min(i * 5 + 4, 389)] for i in range(78)], color="#3b6fd6", lw=1, alpha=0.8)
    e, x0, x1 = r.entry, r.ei / 5, r.xi / 5; tpx, slx = e + r.side * r.tp, e - r.side * r.sl
    ax.add_patch(Rectangle((x0, min(e, tpx)), max(x1 - x0, 0.6), abs(tpx - e), color="#2e9e5b", alpha=0.18))
    ax.add_patch(Rectangle((x0, min(e, slx)), max(x1 - x0, 0.6), abs(slx - e), color="#d64545", alpha=0.12))
    ax.plot([x0], [e], marker="^" if r.side == 1 else "v", ms=12, color="#1d4ed8")
    for lv, txt, c_ in ((d["vah"], "VAH", "#e08a1e"), (d["val"], "VAL", "#1a9c9c"), (d["poc"], "POC", "gray"), (d["pdh"], "前高", "#d64545"), (d["pdl"], "前低", "#2e9e5b"), (tpx, "TP", "#2e9e5b"), (slx, "SL", "#d64545")):
        ax.text(78.3, lv, f"{txt} {lv:,.0f}", va="center", fontsize=8, color=c_)
    win = r.pnl > 0
    ax.set_title(f"{r.date}  {r.rule}  {'做多' if r.side == 1 else '做空'} {hh(r.ei)} @ {e:,.2f}  →  {'✓ 賺' if win else '✗ 賠'} {r.res} {r.pnl:+,.0f}（{hh(min(r.xi, 389))}）",
                 fontsize=12, color="#15803d" if win else "#b91c1c")
    note = (f"scanner 當下（{hh(r.ei - 1)} 收盤）\n日線型態：{r.daily}\nORB15：{r.orb_txt}\nVWAP：價格在{'上' if r.vwap_side > 0 else '下'}方\n"
            f"Time Machine（2,000 次）：先碰 TP {r.tm_tp:.0%}／先碰 SL {r.tm_sl:.0%}\n  期望值 {r.tm_ev:+,.0f} → {r.tm_verdict}")
    ax.text(0.01, 0.02 if r.side == -1 else 0.98, note, transform=ax.transAxes, fontsize=9, va="bottom" if r.side == -1 else "top",
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.92))
    ax.set_xlim(-1, 86); ticks = range(0, 79, 6); ax.set_xticks(list(ticks)); ax.set_xticklabels([hh(i * 5) for i in ticks], fontsize=8)
    lo = np.nanmin([np.nanmin(b[:, 2]), slx, tpx, d["val"], d["pdl"]]); hi_ = np.nanmax([np.nanmax(b[:, 1]), slx, tpx, d["vah"], d["pdh"]]); pad = (hi_ - lo) * 0.06; ax.set_ylim(lo - pad, hi_ + pad)
    ax.grid(alpha=0.2); ax.set_ylabel("NQ"); fig.tight_layout()
    fn = f"{out}/{n_:02d}_{r.date}_{r.rule.split('（')[0].replace(' ', '')}.png"; fig.savefig(fn); plt.close(fig); files.append(fn)
print("\n圖：", len(files), "張 →", out)
