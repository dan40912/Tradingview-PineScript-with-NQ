"""B_exits 共用核心：固定進場點（順勢多點 6 個點），可換出場／加碼／部位／停損規則。研究用途，不是投資建議。

- 進場點：wr_table.csv.gz（I2、AM-PD-k1 空且順 VWAP、AM-PD-k1 多且順 VWAP、ASIA-PD-k1 多＋ES、ASIA-VA-k1 多＋ES、
  AM-ON-k1 多＋ES＋vol≥1.2），最近 30 分鐘 5 分鐘波動 ≥250 點不進場，亞洲做多每晚最多一筆（同 books17.py）。
- 進場價＝entry_slot 那一分鐘開盤價（訊號用 entry_slot-1 以前收盤的資料，wr_table 已保證）。
- 模擬（numba）：每一分鐘先處理加碼成交 → 停損 → 止盈 → 用該分鐘高低更新最大浮盈／保本／移動停損 → 時間停損。
  同一分鐘止盈止損都碰到算止損；加碼（逆勢）成交那一分鐘不准止盈；跳空越過停損用開盤價（較差者）成交。
- 每日風險檢查：今天已實現（實際出場時間早於這筆進場）− 這筆最壞虧損 − 其他未平倉最壞虧損 < −3000 就不下。
"""
import os, sys
RESEARCH = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
sys.path.insert(0, RESEARCH)
import numpy as np, pandas as pd
from numba import njit
from lib import data

PT, COST, DAYLIM = 20.0, 48.0, 3000.0
g = data.load_grid("NQ", include_holdout=True)
O, H, L, C = g.O, g.H, g.L, g.C
EXT_AM, EXT_ASIA = 1319, (8 * 60 + 29 - 18 * 60) % 1440   # 15:59、08:29

# ---- ATR（前 10 個交易日美股盤中振幅平均，不含當天）----
_rng = pd.Series(np.nanmax(g.rth(g.H), 1) - np.nanmin(g.rth(g.L), 1), index=g.dates)
ATR = _rng[g.tradable].rolling(10, min_periods=1).mean().shift(1).reindex(g.dates).ffill()   # 前幾天不足 10 天就用已有的天數


def seg_of(d):
    return np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-09-30"], ["探索", "驗證", "測試"], "近期")


SEGS = ["探索", "驗證", "測試", "近期"]


def candidates():
    F = pd.read_csv(os.path.join(RESEARCH, "wr_table.csv.gz")); F["key"] = F.rule + "|" + F.side
    S = pd.read_csv(os.path.join(RESEARCH, "swan_shock.csv"))[["date", "rule", "side", "entry_slot", "shock"]]
    F = F.merge(S, on=["date", "rule", "side", "entry_slot"], how="left")
    withI = F.to_vwap == False
    P = {"I2": F.key.str.startswith("I2"),
         "前高空": (F.key == "AM-PD-k1|空") & withI,
         "前低多": (F.key == "AM-PD-k1|多") & withI,
         "亞洲多ES": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True),
         "亞洲VAL多ES": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),
         "早盤夜低多ES": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2)}
    F["pt"] = None
    for n, m in P.items():
        F.loc[m.fillna(False).values & F.pt.isna().values, "pt"] = n
    T = F[F.pt.notna()].copy()
    T = T[~(T.shock >= 250)].sort_values(["date", "entry_slot"])
    a = T.key.str.startswith("ASIA") & (T.dir == 1)
    T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"], kind="stable")
    T["i"] = T.date.map(g.dindex).astype(int)
    T["xs_ext"] = np.where(T.entry_slot >= 930, EXT_AM, EXT_ASIA)
    T["atr"] = T.date.map(ATR).fillna(250.0).values   # 只有 2024-01-02 沒有前一天，用 250
    T["seg"] = seg_of(T.date.values)
    return T.reset_index(drop=True)


CAND = candidates()

# 參數向量索引
(SL, TP1, F1, TP2, BET, BEO, TRT, TRD, TSM, TSF, BE1, A1D, A1S, A2D, A2S, TPAVG, PYO, TPAFTER) = range(18)
NP = 18


def plan(sl=145., tp1=55., f1=1.0, tp2=np.inf, be_trig=np.inf, be_off=0., tr_trig=np.inf, tr_dist=np.inf,
         ts_min=0, ts_mfe=0., be_after_tp1=0, a1d=0., a1s=0., a2d=0., a2s=0., tp_avg=0, pyr_off=np.nan, tp_after_add=np.nan):
    p = np.zeros(NP)
    p[[SL, TP1, F1, TP2, BET, BEO, TRT, TRD, TSM, TSF, BE1, A1D, A1S, A2D, A2S, TPAVG, PYO, TPAFTER]] = \
        [sl, tp1, f1, tp2, be_trig, be_off, tr_trig, tr_dist, ts_min, ts_mfe, be_after_tp1, a1d, a1s, a2d, a2s, tp_avg, pyr_off, tp_after_add]
    return p


@njit(cache=True)
def _sim(Or, Hr, Lr, Cr, e, xs, d, p):
    """回傳 (損益點數×口數 合計, 實際成交總口數, 出場 slot, 原因碼 0時間 1止損 2止盈 3時間停損)。基本 1 口。"""
    sl, tp1, f1, tp2 = p[0], p[1], p[2], p[3]
    bet, beo, trt, trd = p[4], p[5], p[6], p[7]
    tsm, tsf, be1 = int(p[8]), p[9], p[10]
    a1d, a1s, a2d, a2s, tpavg, pyo, tpafter = p[11], p[12], p[13], p[14], p[15], p[16], p[17]
    while e <= xs and np.isnan(Or[e]):
        e += 1
    if e > xs:
        return 0.0, 0.0, -1, -1
    px = Or[e]
    size = 1.0; cost = px; filled = 1.0       # cost = Σ 成交價×口數
    real = 0.0
    stop = px - d * sl
    a1 = a1s > 0; a2 = a2s > 0
    lv1 = px + d * a1d; lv2 = px + d * a2d
    tp1_done = False
    best = 0.0
    reason = 0; xslot = xs
    tp1_dist = tp1
    for s in range(e, xs + 1):
        h = Hr[s]; l = Lr[s]; o = Or[s]; c = Cr[s]
        if np.isnan(h):
            continue
        adv_fill = False
        # 1) 加碼單
        for k in range(2):
            act = a1 if k == 0 else a2
            if not act:
                continue
            dist = a1d if k == 0 else a2d; lvl = lv1 if k == 0 else lv2; sz = a1s if k == 0 else a2s
            if dist < 0:   # 逆勢限價
                hit = (l <= lvl) if d == 1 else (h >= lvl)
                fp = min(o, lvl) if d == 1 else max(o, lvl)
            else:          # 順勢停損單
                hit = (h >= lvl) if d == 1 else (l <= lvl)
                fp = max(o, lvl) if d == 1 else min(o, lvl)
            if s == e:
                fp = lvl
            if hit:
                size += sz; cost += fp * sz; filled += sz
                if k == 0: a1 = False
                else: a2 = False
                if dist < 0:
                    adv_fill = True
                    if not np.isnan(tpafter):
                        tp1_dist = tpafter
                elif not np.isnan(pyo):
                    ns = px + d * pyo
                    stop = max(stop, ns) if d == 1 else min(stop, ns)
        # 2) 停損
        if (d == 1 and l <= stop) or (d == -1 and h >= stop):
            fp = stop
            if s > e:
                fp = min(o, stop) if d == 1 else max(o, stop)
            real += d * (fp * size - cost); size = 0.0
            reason = 1; xslot = s
            break
        # 3) 止盈
        if not adv_fill:
            ref = cost / size if tpavg > 0 else px
            if not tp1_done:
                lvl = ref + d * tp1_dist
                if (d == 1 and h >= lvl) or (d == -1 and l <= lvl):
                    q = size * f1
                    avg = cost / size
                    real += d * (lvl - avg) * q; cost -= avg * q; size -= q
                    tp1_done = True
                    a1 = False; a2 = False      # 止盈後不再加碼
                    if size <= 1e-9:
                        reason = 2; xslot = s
                        break
                    if be1 > 0:
                        stop = max(stop, px) if d == 1 else min(stop, px)
            if tp1_done and size > 1e-9 and np.isfinite(tp2):
                lvl = px + d * tp2
                if (d == 1 and h >= lvl) or (d == -1 and l <= lvl):
                    avg = cost / size
                    real += d * (lvl - avg) * size; size = 0.0
                    reason = 2; xslot = s
                    break
        # 4) 更新浮盈、保本、移動停損（本分鐘之後才生效）
        fav = (h - px) if d == 1 else (px - l)
        if fav > best:
            best = fav
        if best >= bet:
            ns = px + d * beo
            stop = max(stop, ns) if d == 1 else min(stop, ns)
        if best >= trt:
            ns = px + d * (best - trd)
            stop = max(stop, ns) if d == 1 else min(stop, ns)
        # 5) 時間停損：進場後第 tsm 分鐘收盤時最大浮盈仍 < tsf → 出場
        if tsm > 0 and s >= e + tsm and best < tsf and not tp1_done:
            real += d * (c * size - cost); size = 0.0
            reason = 3; xslot = s
            break
    if size > 1e-9:
        k = xs
        while k >= e and np.isnan(Cr[k]):
            k -= 1
        real += d * (Cr[k] * size - cost); size = 0.0
        xslot = k
    return real, filled, xslot, reason


@njit(cache=True)
def _sim_all(O, H, L, C, I, E, XS, D, SC, P, M):
    n = len(I)
    out = np.zeros((n, 4))
    q = np.empty(len(P))
    for t in range(n):
        for j in range(len(P)):
            q[j] = P[j]
        # 距離類參數乘上比例（ATR 模式用；固定點數時 SC=1）
        for j in (0, 1, 3, 4, 5, 6, 7, 9, 11, 13, 16, 17):
            if M[j] > 0:
                q[j] = q[j] * SC[t]
        i = I[t]
        r = _sim(O[i], H[i], L[i], C[i], E[t], XS[t], D[t], q)
        out[t, 0], out[t, 1], out[t, 2], out[t, 3] = r
    return out


def worst_points(p):
    """這個出場計畫在 1 口基本單、比例 1 下的最壞虧損（點×口）與最大口數。加碼全部成交、停損打到。"""
    sl = p[SL]; legs = [(0.0, 1.0)]
    for dd, ss in ((p[A1D], p[A1S]), (p[A2D], p[A2S])):
        if ss > 0:
            legs.append((dd, ss))
    adv = [(dd, ss) for dd, ss in legs if dd <= 0]
    w_adv = sum(ss * (sl + dd) for dd, ss in adv)            # 逆勢：全部在停損出場（dd<=0）
    n_adv = sum(ss for dd, ss in adv)
    pyr = [(dd, ss) for dd, ss in legs if dd > 0]
    if pyr:
        stop_after = -sl if np.isnan(p[PYO]) else p[PYO]       # 加碼後停損相對原進場
        w_pyr = sum(ss * (dd - stop_after) for dd, ss in pyr) + n_adv * (-stop_after)
        return max(w_adv, w_pyr), n_adv + sum(ss for dd, ss in pyr)
    return w_adv, n_adv


def simulate(p, ext=False, scale=None, cand=None, only_sl=False):
    """對全部候選點跑一個出場計畫（1 口基本單）。scale：每筆距離比例（ATR 模式）。"""
    T = CAND if cand is None else cand
    sc = np.ones(len(T)) if scale is None else np.asarray(scale, float)
    xs = (T.xs_ext if ext else T.exit_slot).values.astype(np.int64)
    M = np.ones(NP)
    if only_sl:          # 只有停損（含加碼距離）跟著 ATR 放大，止盈維持固定點數
        M[:] = 0; M[[SL, A1D, A2D]] = 1
    r = _sim_all(O, H, L, C, T.i.values.astype(np.int64), T.entry_slot.values.astype(np.int64), xs,
                 T.dir.values.astype(np.int64), sc, np.asarray(p, float), M)
    wp, nmax = worst_points(p)
    out = T[["date", "pt", "key", "dir", "entry_slot", "seg", "atr"]].copy()
    out["pnl1"] = r[:, 0] * PT - COST * r[:, 1]          # 1 口基本單的損益（含加碼成交的成本）
    out["filled"] = r[:, 1]; out["xslot"] = r[:, 2].astype(int); out["why"] = r[:, 3].astype(int)
    out["worst1"] = wp * sc * PT + COST * nmax              # 1 口基本單的最壞虧損（$）
    out["xsched"] = xs
    return out


def book(R, size=None, day_stop=None, week_stop=None, pause_after=None, pause_days=1, guard=True,
         perf=None, lim=DAYLIM, sched=False, fill=False):
    """R：simulate() 的結果。size：每筆基本口數（陣列或 None=1）。
    day_stop：今天已實現 ≤ −day_stop 就停；week_stop：本週已實現 ≤ −week_stop 就停到下週；
    pause_after：連虧 N 筆後暫停 pause_days 個交易日。perf：函式(過去已實現損益序列)→口數倍數。
    fill=True：額度不夠下原口數時，改用剩下額度能下的口數（≥0.1 口才下）。
    sched=True：已實現損益要等到「預定出場時間」才算（books17.py 的簡化寫法，基準 445 筆用這個）。
    回傳實際成交的交易（含 pnl、口數）。"""
    sz = np.ones(len(R)) if size is None else np.asarray(size, float)
    rows = []
    hist = []
    streak = 0; paused_until = None
    days = list(R.groupby("date", sort=True))
    alld = sorted(R.date.unique())
    week_pnl = {}
    for di, (d, x) in enumerate(days):
        wk = pd.Timestamp(d).isocalendar()[:2]
        if paused_until is not None and d < paused_until:
            continue
        open_ = []   # (exit_slot, worst$, pnl$)
        n0 = len(rows)
        closed = []
        for idx, t in zip(x.index, x.itertuples()):
            if t.xslot < 0:
                continue
            c = sz[idx]
            if perf is not None:
                c = perf(hist, c)
            c = round(c * 10) / 10
            if c <= 0:
                continue
            realized = sum(p for xs, p in closed if xs < t.entry_slot) + sum(p for xs, w, p in open_ if xs < t.entry_slot)
            still = [w for xs, w, p in open_ if xs >= t.entry_slot]
            if day_stop is not None and realized <= -day_stop:
                continue
            if week_stop is not None and week_pnl.get(wk, 0) + realized <= -week_stop:
                continue
            if guard and realized - sum(still) - t.worst1 * c < -lim:
                if not fill:
                    continue
                c = np.floor((lim + realized - sum(still)) / t.worst1 * 10 + 1e-9) / 10   # 用剩下的風險額度下小口數
                if c < 0.1:
                    continue
            closed += [(xs, p) for xs, w, p in open_ if xs < t.entry_slot]
            open_ = [o for o in open_ if o[0] >= t.entry_slot]
            pnl = t.pnl1 * c
            open_.append((t.xsched if sched else t.xslot, t.worst1 * c, pnl))
            rows.append((idx, c, pnl))
        dp = sum(p for xs, p in closed) + sum(p for xs, w, p in open_)
        week_pnl[wk] = week_pnl.get(wk, 0) + dp
        for idx, c, pnl in rows[n0:]:
            hist.append(pnl)
            streak = streak + 1 if pnl < 0 else 0
        if pause_after is not None and streak >= pause_after:
            j = alld.index(d)
            paused_until = alld[min(j + 1 + pause_days, len(alld) - 1)] if j + 1 + pause_days < len(alld) else "9999"
            streak = 0
    if not rows:
        return R.iloc[:0].assign(c=[], pnl=[])
    idx, cs, pn = zip(*rows)
    B = R.loc[list(idx)].copy(); B["c"] = cs; B["pnl"] = pn
    return B


def dd(p):
    c = np.cumsum(p); return float(np.min(c - np.maximum.accumulate(np.concatenate([[0.0], c]))[1:])) if len(c) else 0.0


def stats(B):
    """四段 + 全期。"""
    out = {}
    for s in SEGS + ["全期"]:
        x = B if s == "全期" else B[B.seg == s]
        p = x.pnl.values
        if len(p) == 0:
            out[s] = dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0, worst=0.0, rdd=np.nan); continue
        m = dd(p)
        out[s] = dict(n=len(p), win=100 * (p > 0).mean(), avg=p.mean(), tot=p.sum(), mdd=m,
                      worst=x.groupby("date").pnl.sum().min(), rdd=p.sum() / -m if m < 0 else np.inf)
    return out


def flat(st, name):
    r = {"name": name}
    for s, v in st.items():
        for k, val in v.items():
            r[f"{s}_{k}"] = val
    return r


def table(rows, segs=SEGS + ["全期"]):
    """Markdown 四段表。rows: [(名稱, stats)]"""
    hdr = "| 方案 | 段 | 筆數 | 勝率 | 每筆 | 合計 | 最大回撤 | 最差一天 | 報酬/回撤 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    s = ""
    for name, st in rows:
        for sg in segs:
            v = st[sg]
            s += f"| {name} | {sg} | {v['n']} | {v['win']:.1f}% | {v['avg']:+,.0f} | {v['tot']:+,.0f} | {v['mdd']:+,.0f} | {v['worst']:+,.0f} | {v['rdd']:.2f} |\n"
    return hdr + s


ATR_REF = float(ATR[ATR.index <= "2025-06-30"].dropna().median())   # 只用探索期


def sizes(R, mode=None, cap=1.0, lo=0.5, hi=1.5, risk=2948.0):
    """部位大小：None=1 口；'risk'：每筆最壞虧損 ≈ risk（不超過 cap 口）；'vol'：ATR_REF/ATR 夾在 [lo,hi]。"""
    if mode is None:
        return np.ones(len(R))
    if mode == "risk":
        return np.minimum(cap, np.floor(risk / R.worst1.values * 10 + 1e-9) / 10)
    if mode == "vol":
        return np.clip(ATR_REF / R.atr.values, lo, hi)
    raise ValueError(mode)
