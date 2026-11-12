"""R2_backtest 共用函式（研究用途，不是投資建議）。

依 Q2_critic/strategies_v2.json 的 common_rules：
- 『hh:mm 的價格』＝slot(hh:mm)−1 收盤；NaN 往前找最多 3 分鐘，否則不做。
- 進場價＝entry_slot 開盤；NaN 時用之後 3 分鐘內第一根有值的開盤，否則不做。
- 停損：1 分 K 高低點觸價即出；進場那一分鐘也檢查（成交價＝停損價）；之後的分鐘跳空穿越用開盤價（較差）。
  條件出場（追蹤、時間）只在規定檢查點用前一根收盤判斷、該檢查點開盤出場。時間出場＝exit_slot 收盤。
- 每口 $48；1 口＝20 美元／點。
- 每日 $3,000：今日已實現（平倉即記帳：實際出場 slot < 本筆進場 slot 才算）− 本筆最壞 − 其他持倉最壞 < −3000 → 不下。
資料：lib.data 的網格（不修改 lib）；唯一修正 2026-10-30 15:03（slot 1263）一根 Yahoo 壞價（NQ、ES 各一根），
開盤改前一根收盤、高低改成開收範圍（與 A_trend、RT_redteam 相同的偵測規則；slot 0 不套用）。
"""
from __future__ import annotations
import os, sys
sys.dont_write_bytecode = True
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
AG = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/agents"
OUT = os.path.join(AG, "R2_backtest")
sys.path.insert(0, RES)
from lib import data, features  # noqa: E402

PT, COST, LIM = 20.0, 48.0, 3000.0
R0, EOD = 930, 1319
SEGS = ["探索", "驗證", "測試", "近期"]


def slot(hhmm):
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m - 1080) % 1440


def hhmm(s):
    t = (int(s) + 1080) % 1440
    return f"{t // 60:02d}:{t % 60:02d}"


def seg_of(d):
    d = str(d)[:10]
    if d <= "2025-06-30":
        return "探索"
    if d <= "2025-12-31":
        return "驗證"
    if d <= "2026-09-30":
        return "測試"
    if d <= "2026-11-11":
        return "近期"
    return "無"


def _clean(g, thr):
    O, H, L, C, V = g.O.copy(), g.H.copy(), g.L.copy(), g.C.copy(), g.V.copy()
    pc = pd.DataFrame(C).ffill(axis=1).shift(1, axis=1).values
    bo = (np.abs(O - pc) > thr) & (np.abs(O - C) > thr)
    bh = (H - np.fmax(pc, C)) > thr
    bl = (np.fmin(pc, C) - L) > thr
    bad = (bo | bh | bl)
    bad[:, 0] = False
    O = np.where(bad & bo, pc, O)
    H = np.where(bad, np.fmax(O, C), H)
    L = np.where(bad, np.fmin(O, C), L)
    return O, H, L, C, V, np.argwhere(bad)


g = data.load_grid("NQ", include_holdout=True)
ge = data.load_grid("ES", include_holdout=True)
assert (g.dates == ge.dates).all()
N = len(g)
DATES = g.dates
SEG = np.array([seg_of(d) for d in DATES])
O, H, L, C, V, BAD_NQ = _clean(g, 250)
EO, EH, EL, EC, EV, BAD_ES = _clean(ge, 80)
TRAD = g.tradable.copy() & (SEG != "無")
PREV_SAME = np.array([bool(g.tradable[i]) and g.prev_same_contract(i) for i in range(N)])
LV = features.daily_levels(g).set_index("i")
PDC = LV.pdc.values      # NaN on roll days
PDH = LV.pdh.values
PDL = LV.pdl.values

with np.errstate(all="ignore"):
    RNG = np.nanmax(H[:, R0:R0 + 390], 1) - np.nanmin(L[:, R0:R0 + 390], 1)
    RTH_HI = np.nanmax(H[:, R0:R0 + 390], 1)
    RTH_LO = np.nanmin(L[:, R0:R0 + 390], 1)


def _last_valid(a, lo, hi):
    """a[lo..hi] 的最後一個有值（含）。"""
    for k in range(hi, lo - 1, -1):
        if np.isfinite(a[k]):
            return a[k], k
    return np.nan, -1


RTH_CLOSE = np.array([_last_valid(C[i], R0, EOD)[0] for i in range(N)])
RTH_OPEN = np.array([O[i, R0] for i in range(N)])
ES_CLOSE = np.array([_last_valid(EC[i], R0, EOD)[0] for i in range(N)])
ES_OPEN = np.array([EO[i, R0] for i in range(N)])


def prior_mean(x, n, valid, minfrac=1.0):
    """前 n 個 valid 日（不含當天）x 的平均；有效值 < minfrac×n → NaN。"""
    out = np.full(N, np.nan)
    hist = []
    for i in range(N):
        if len(hist) >= n:
            w = np.array(hist[-n:])
            if np.isfinite(w).sum() >= minfrac * n:
                out[i] = np.nanmean(w)
        if valid[i]:
            hist.append(x[i])
    return out


# ADR10：前 10 個真正 RTH 日（tradable）高低差平均，不含當天
ADR10 = prior_mean(RNG, 10, g.tradable)


def price_at(arr, i, s, back=3):
    """『hh:mm 的價格』：arr[i, s-1]，NaN 往前找最多 back 分鐘。回傳 (價格, 用到的 slot)。"""
    for k in range(s - 1, s - 1 - back - 1, -1):
        if k >= 0 and np.isfinite(arr[i, k]):
            return arr[i, k], k
    return np.nan, -1


def entry_slot_valid(i, s, fwd=3):
    for k in range(s, s + fwd + 1):
        if k < 1440 and np.isfinite(O[i, k]):
            return k
    return -1


# ---------------------------------------------------------------- 模擬器
def sim(i, e, d, sl, xs, cond_exit=None, entry="open", slip=0.0, cost=COST, size=1.0, fwd=3):
    """i 日、entry_slot e、方向 d、停損點數 sl、時間出場 xs（該根收盤）。
    cond_exit：set of slots；在該 slot 開盤出場（決策用的是 slot−1 收盤，呼叫者負責）。只在 > 實際進場 slot 時有效。
    entry='open'｜'adverse'（多單用進場那根 high、空單用 low）。slip：進、出各差幾點（不利方向）。"""
    e = entry_slot_valid(i, e, fwd)
    if e < 0 or e > xs:
        return None
    px = O[i, e] if entry == "open" else (H[i, e] if d == 1 else L[i, e])
    px = px + d * slip
    stop = px - d * sl
    why, xp, xslot = "time", np.nan, -1
    ce = cond_exit or ()
    s = e
    while s <= xs:
        if s in ce and s > e:
            k = entry_slot_valid(i, s, 3)
            if 0 <= k <= xs:
                # 檢查點開盤（或 3 分內第一根有值開盤）出場；若開盤已穿越停損，以開盤價成交（同樣的價）
                xp, xslot, why = O[i, k], k, "cond"
                break
        h, l, o = H[i, s], L[i, s], O[i, s]
        if np.isfinite(h):
            if (d == 1 and l <= stop) or (d == -1 and h >= stop):
                f = stop
                if s > e and np.isfinite(o) and ((d == 1 and o < stop) or (d == -1 and o > stop)):
                    f = o
                xp, xslot, why = f, s, "sl"
                break
        s += 1
    if xslot < 0:
        xp, xslot = _last_valid(C[i], e, xs)
        why = "time"
    xp = xp - d * slip
    pts = d * (xp - px)
    return dict(date=DATES[i], i=i, seg=SEG[i], d=d, e=e, x=xslot, et=hhmm(e), xt=hhmm(xslot), entry=px, exit=xp,
                why=why, sl=sl, pts=pts, pnl=(pts * PT - cost) * size, worst=(min(sl, 1e5) * PT + COST) * size, size=size)


# ---------------------------------------------------------------- 記帳
def book(T, lim=LIM, prio_col="prio"):
    """每日 $3,000 檢查（平倉即記帳）。T 需 date,e,x,pnl,worst；同一分鐘依 prio 小者先。回傳實際下單的交易。"""
    if T is None or len(T) == 0:
        return T
    T = T.copy()
    if prio_col not in T:
        T[prio_col] = 0
    T = T.sort_values(["date", "e", prio_col], kind="stable").reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for d, idx in T.groupby("date", sort=False).groups.items():
        acc = []  # (x, pnl, worst)
        for k in idx:
            e = T.at[k, "e"]
            real = sum(p for x, p, w in acc if x < e)
            openw = sum(w for x, p, w in acc if x >= e)
            if real - openw - T.at[k, "worst"] < -lim:
                continue
            keep[k] = True
            acc.append((T.at[k, "x"], T.at[k, "pnl"], T.at[k, "worst"]))
    return T[keep].reset_index(drop=True)


# ---------------------------------------------------------------- 統計
def mdd(p):
    c = np.cumsum(np.asarray(p, float))
    if len(c) == 0:
        return 0.0
    pk = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - pk).min())


def stats(T):
    if T is None or len(T) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0, worst=0.0)
    T = T.sort_values(["date", "e"], kind="stable")
    p = T.pnl.values
    dy = T.groupby("date").pnl.sum()
    return dict(n=len(p), win=100 * (p > 0).mean(), avg=p.mean(), tot=p.sum(), mdd=mdd(p), worst=dy.min())


def seg_stats(T):
    out = {s: stats(T[T.seg == s]) if T is not None and len(T) else stats(None) for s in SEGS}
    out["全期"] = stats(T)
    a = out["全期"]
    out["全期"]["rdd"] = a["tot"] / -a["mdd"] if a["mdd"] < 0 else np.inf
    return out


def fmt_seg(st, segs=SEGS + ["全期"]):
    lines = []
    for s in segs:
        v = st[s]
        if v["n"] == 0:
            lines.append(f"  {s}: 0 筆")
            continue
        extra = f"  報酬/回撤 {v['rdd']:.2f}" if s == "全期" else ""
        lines.append(f"  {s}: {v['n']:4d} 筆 勝率 {v['win']:5.1f}% 每筆 {v['avg']:+7.0f} 合計 {v['tot']:+10,.0f} "
                     f"最大回撤 {v['mdd']:+9,.0f} 最差一天 {v['worst']:+8,.0f}{extra}")
    return "\n".join(lines)


def oneline(st):
    return " | ".join(f"{s} {st[s]['n']}筆 {st[s]['avg']:+.0f}/{st[s]['tot']:+,.0f}" if st[s]['n'] else f"{s} 0筆" for s in SEGS) + \
        f" || 全期 {st['全期']['n']}筆 {st['全期']['tot']:+,.0f} DD {st['全期']['mdd']:+,.0f} 比 {st['全期']['rdd']:.2f}"


def tstat_p(p):
    """單尾 t 檢定 H1: 平均 > 0。"""
    from scipy import stats as ss
    p = np.asarray(p, float)
    if len(p) < 3:
        return np.nan, np.nan
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if p.std(ddof=1) > 0 else 0.0
    return t, float(ss.t.sf(t, len(p) - 1))


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[k]))
        adj[k] = run
    return adj


def drop_top_days(T, k=5):
    dy = T.groupby("date").pnl.sum().sort_values(ascending=False)
    top = list(dy.index[:k])
    return T[~T.date.isin(top)], top


# ---------------------------------------------------------------- 分組變數（只報告）
def _load_cal():
    import json
    j = json.load(open(os.path.join(AG, "Q2_critic", "strategies_v2.json")))
    return j["event_calendar"]


CAL = _load_cal()
FOMC = set(CAL["FOMC"]); CPI = set(CAL["CPI"]); NFP = set(CAL["NFP"])
DATA_DAYS = CPI | NFP


def group_labels():
    dt = pd.to_datetime(pd.Series(DATES))
    lab = pd.DataFrame({"date": DATES})
    lab["FOMC日"] = lab.date.isin(FOMC)
    lab["CPI/NFP日"] = lab.date.isin(DATA_DAYS)
    lab["OPEX"] = (dt.dt.weekday == 4) & (dt.dt.day >= 15) & (dt.dt.day <= 21)
    roll = np.array([i > 0 and g.contract[i] != g.contract[i - 1] for i in range(N)])
    wk = dt.dt.isocalendar()
    wkkey = (wk.year.astype(str) + "-" + wk.week.astype(str)).values
    rollwk = set(wkkey[roll])
    lab["換月週"] = pd.Series(wkkey).isin(rollwk).values
    lab["債市休市股市開"] = lab.date.isin({"2024-10-14", "2024-11-11", "2025-10-13", "2025-11-11", "2026-10-12", "2026-11-11"})
    # TOM：月末最後 1 個交易日＋月初 3 個交易日（只看真正 RTH 日）
    tr = pd.Series(DATES[g.tradable])
    ym = tr.str[:7]
    tom = set()
    for m, x in tr.groupby(ym):
        x = list(x)
        tom.update(x[:3]); tom.add(x[-1])
    lab["TOM"] = lab.date.isin(tom)
    # IBS（前一 RTH 日）
    ibs = (PDC - PDL) / (PDH - PDL)
    lab["IBS"] = pd.cut(ibs, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["低<1/3", "中", "高>2/3"]).astype(str)
    # NR7 / Inside day（前一個完整 RTH 日）
    nr = np.zeros(N, bool); idd = np.zeros(N, bool)
    tidx = np.where(g.tradable)[0]
    pos = {d: k for k, d in enumerate(tidx)}
    for i in range(N):
        prev = [j for j in tidx if j < i][-7:] if i > 0 else []
        if len(prev) == 7:
            r = RNG[prev]
            nr[i] = r[-1] <= np.nanmin(r)
            j1, j2 = prev[-1], prev[-2]
            idd[i] = RTH_HI[j1] < RTH_HI[j2] and RTH_LO[j1] > RTH_LO[j2]
    lab["NR7/ID"] = nr | idd
    rv5 = prior_mean(RNG, 5, g.tradable); rv60 = prior_mean(RNG, 60, g.tradable, 0.75)
    rvr = rv5 / rv60
    lab["RVR"] = pd.cut(rvr, [0, 0.8, 1.2, 99], labels=["<0.8", "0.8-1.2", ">1.2"]).astype(str)
    return lab


def group_report(T, lab, extra=None):
    """T：逐筆交易；lab：group_labels()；extra：dict 名稱→布林 Series(index=date)。"""
    x = T.merge(lab, on="date", how="left")
    if extra:
        for k, v in extra.items():
            x[k] = x.date.map(v).fillna(False)
    lines = []
    cols = [c for c in lab.columns if c != "date"] + (list(extra) if extra else [])
    for c in cols:
        parts = []
        for val, y in x.groupby(c):
            parts.append(f"{val}: {len(y)}筆 每筆 {y.pnl.mean():+.0f}")
        lines.append(f"    {c:14s} " + "；".join(parts))
    return "\n".join(lines)


def perm_test(ev_pnl, pool_pnl, n=10000, seed=0):
    """事件日平均 vs 從非事件日隨機抽同樣天數的平均。回傳 p＝P(隨機平均 ≥ 事件平均)。"""
    ev = np.asarray(ev_pnl, float); pool = np.asarray(pool_pnl, float)
    if len(ev) == 0 or len(pool) < len(ev):
        return np.nan
    rng = np.random.default_rng(seed)
    obs = ev.mean()
    sims = np.array([pool[rng.choice(len(pool), len(ev), replace=False)].mean() for _ in range(n)])
    return float((np.sum(sims >= obs) + 1) / (n + 1))
