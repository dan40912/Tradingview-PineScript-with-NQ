"""A_trend 共用函式：資料、盤中特徵、出場模擬器、績效、基準組合。研究用途，不是投資建議。

規則（BRIEF.md）：
  * 進場只用進場那一分鐘之前已收盤的資料；進場價＝訊號後下一分鐘開盤價。
  * 每口 NQ 來回 $48；同一分鐘止盈止損都碰到算止損；跳空穿過止損用較差的開盤價成交。
  * 不抱過 15:59（預設 15:55 收盤出場）。
  * 1 口 NQ；每筆最壞虧損 = 止損點數×20+48，止損最多 145 點（最壞 $2,948）。
索引：RTH 陣列第 k 欄 = 09:30 後第 k 分鐘（k=0 → 09:30，k=385 → 15:55）。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RES)
from lib import data, features  # noqa: E402

try:
    from numba import njit
except Exception:  # 沒有 numba 也能跑，只是慢
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

PT, COST, MAXSTOP = 20.0, 48.0, 145.0
EOD = 385   # 15:55
PERIODS = ["探索", "驗證", "測試", "近期"]


def period_of(d):
    if d <= "2025-06-30":
        return "探索"
    if d <= "2025-12-31":
        return "驗證"
    if d <= "2026-09-30":
        return "測試"
    if d <= "2026-11-11":
        return "近期"
    return "無"


class Data:
    def __init__(self):
        g = data.load_grid("NQ", include_holdout=True)
        e = data.load_grid("ES", include_holdout=True)
        assert (g.dates == e.dates).all()
        self.g, self.e = g, e
        self.lv = features.daily_levels(g)
        R = data.RTH_OFF
        sl = slice(R, R + 390)
        self.O, self.H, self.L, self.C, self.V = (a[:, sl].copy() for a in (g.O, g.H, g.L, g.C, g.V))
        # 壞資料修正（凍結候選之後才發現，只影響 2026-10-30 15:03 一根 Yahoo 壞價：開/高 29,562，前後都在 28,2xx）：
        # 一根 K 的高（低）比「前一根收盤與本根收盤」高（低）超過 250 點、或開盤離兩者都超過 250 點 → 視為壞價，
        # 開盤改成前一根收盤，高低改成開收的範圍。全資料只有這一根被改。
        pc = pd.DataFrame(self.C).ffill(axis=1).shift(1, axis=1).values
        bad_o = (np.abs(self.O - pc) > 250) & (np.abs(self.O - self.C) > 250)
        bad_h = (self.H - np.fmax(pc, self.C) > 250); bad_l = (np.fmin(pc, self.C) - self.L > 250)
        self.O = np.where(bad_o, pc, self.O)
        self.H = np.where(bad_h, np.fmax(self.O, self.C), self.H)
        self.L = np.where(bad_l, np.fmin(self.O, self.C), self.L)
        self.n_bad_bars = int((bad_o | bad_h | bad_l).sum())
        self.eC = e.C[:, sl].copy(); self.eO = e.O[:, sl].copy(); self.eH = e.H[:, sl].copy(); self.eL = e.L[:, sl].copy()
        self.dates = g.dates
        self.n = len(g)
        self.P = np.array([period_of(d) for d in g.dates])
        # 盤中 VWAP（09:30 起；第 k 欄含第 k 根）
        tp = (self.H + self.L + self.C) / 3
        m = np.isfinite(tp) & np.isfinite(self.V)
        pv = np.cumsum(np.where(m, tp * self.V, 0), 1); vv = np.cumsum(np.where(m, self.V, 0), 1)
        self.vwap = np.where(vv > 0, pv / np.where(vv > 0, vv, 1), np.nan)
        etp = (self.eH + self.eL + self.eC) / 3
        # 前 10 個交易日的 RTH 振幅平均（只用前一天以前）
        rng = np.nanmax(self.H, 1) - np.nanmin(self.L, 1)
        self.rng = rng
        trad = g.tradable.copy()
        adr = np.full(self.n, np.nan)
        hist = []
        for i in range(self.n):
            if len(hist) >= 10:
                adr[i] = np.mean(hist[-10:])
            if trad[i] and np.isfinite(rng[i]):
                hist.append(rng[i])
        self.adr10 = adr
        # 前一天 RTH 收盤（同合約時才有）
        L_ = self.lv.set_index("i")
        self.pdc = L_.pdc.values; self.pdh = L_.pdh.values; self.pdl = L_.pdl.values
        self.vah = L_.vah.values; self.val = L_.val.values; self.has_prev = L_.has_prev.values.astype(bool)
        self.onh = L_.onh.values; self.onl = L_.onl.values
        # 夜盤最後價（09:29 收盤）、夜盤開盤（18:00 第一根）
        on = g.C[:, :R]
        self.on_last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in on])
        ono = g.O[:, :R]
        self.on_first = np.array([r[np.isfinite(r)][0] if np.isfinite(r).any() else np.nan for r in ono])
        eon = e.C[:, :R]
        self.e_on_last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in eon])
        # 可交易：真正 RTH、有 09:30 開盤、有 ADR、日期在 2026-11-11 以前、前一天同合約
        self.ok = trad & np.isfinite(self.O[:, 0]) & np.isfinite(adr) & (self.P != "無") & self.has_prev
        # 平均開盤前 N 分鐘振幅（前 10 天）
        self._orcache = {}

    def or_avg(self, N):
        """前 10 個可交易日、開盤前 N 分鐘振幅的平均（只用以前的日子）。"""
        if N in self._orcache:
            return self._orcache[N]
        r = np.nanmax(self.H[:, :N], 1) - np.nanmin(self.L[:, :N], 1)
        out = np.full(self.n, np.nan); hist = []
        for i in range(self.n):
            if len(hist) >= 10:
                out[i] = np.mean(hist[-10:])
            if self.g.tradable[i] and np.isfinite(r[i]):
                hist.append(r[i])
        self._orcache[N] = out
        return out


@njit(cache=True)
def sim1(O, H, L, C, VW, e, d, stop, tp, xidx, trail, ttrig, be, vwbuf, tbars, adrtrail):
    """單筆模擬。回傳 (出場索引, 出場價, 原因碼 0時間 1止損 2止盈 3VWAP 4移動停損)。
    stop/tp 是價位（nan＝不用）。trail＝移動停損距離（點，nan＝不用），ttrig＝啟動所需浮盈；
    be＝浮盈到這麼多點後把止損移到成本；vwbuf＝收盤跌破 VWAP 超過這麼多點就在下一根開盤出場；
    tbars＝用最近 tbars 根的最低（做空最高）當移動停損（0＝不用）。"""
    px = O[e]
    best = 0.0
    trailed = False
    for t in range(e, xidx + 1):
        h = H[t]; l = L[t]
        if h != h:
            continue
        if stop == stop:
            if d == 1 and l <= stop:
                xp = stop if O[t] != O[t] or O[t] > stop or t == e else O[t]
                return t, min(xp, stop), (4 if trailed else 1)
            if d == -1 and h >= stop:
                xp = stop if O[t] != O[t] or O[t] < stop or t == e else O[t]
                return t, max(xp, stop), (4 if trailed else 1)
        if tp == tp:
            if d == 1 and h >= tp:
                return t, tp, 2
            if d == -1 and l <= tp:
                return t, tp, 2
        if t == xidx:
            return t, C[t], 0
        fav = (h - px) if d == 1 else (px - l)
        if fav > best:
            best = fav
        if be == be and best >= be:
            if stop != stop or (d == 1 and px > stop) or (d == -1 and px < stop):
                stop = px; trailed = True
        if trail == trail and best >= ttrig:
            ns = px + d * (best - trail)
            if stop != stop or (d == 1 and ns > stop) or (d == -1 and ns < stop):
                stop = ns; trailed = True
        if tbars > 0 and best >= ttrig and t - tbars + 1 >= e:
            if d == 1:
                ns = 1e18
                for k in range(t - tbars + 1, t + 1):
                    if L[k] == L[k] and L[k] < ns:
                        ns = L[k]
                if ns < 1e17 and (stop != stop or ns > stop):
                    stop = ns; trailed = True
            else:
                ns = -1e18
                for k in range(t - tbars + 1, t + 1):
                    if H[k] == H[k] and H[k] > ns:
                        ns = H[k]
                if ns > -1e17 and (stop != stop or ns < stop):
                    stop = ns; trailed = True
        if vwbuf == vwbuf and C[t] == C[t] and VW[t] == VW[t]:
            if (d == 1 and C[t] < VW[t] - vwbuf) or (d == -1 and C[t] > VW[t] + vwbuf):
                # 下一根開盤出場
                for k in range(t + 1, xidx + 1):
                    if O[k] == O[k]:
                        return k, O[k], 3
                return t, C[t], 3
    # 收盤前資料缺：用最後有值的收盤
    for k in range(xidx, e - 1, -1):
        if C[k] == C[k]:
            return k, C[k], 0
    return e, px, 0


def last_valid(C, xidx):
    for k in range(min(xidx, len(C) - 1), -1, -1):
        if C[k] == C[k]:
            return k
    return -1


def run(D: Data, sigs, exit_cfg, cap="cap"):
    """sigs: list of dict(i, e, d, stop_px[可 nan], [stop_dist]) 依日期、時間排序；
    exit_cfg: dict(stop_pts=固定點數或 None, stop_mode='px'|'pts'|'adr', stop_adr=倍數,
                   tp_R=止盈倍數(相對止損距離) or tp_pts, trail, ttrig, be, vwbuf, tbars, xidx)
    cap='cap'：結構止損 > 145 點就縮成 145；'skip'：直接不做。止損 < 10 點時放大到 10。
    同一天多筆時套用每日 $3,000 檢查。回傳 DataFrame。"""
    c = exit_cfg
    rows = []
    day_real = {}
    day_open = {}
    for s in sigs:
        i, e, d = s["i"], s["e"], s["d"]
        O, H, L, C, VW = D.O[i], D.H[i], D.L[i], D.C[i], D.vwap[i]
        while e < EOD and O[e] != O[e]:
            e += 1
        if e >= EOD:
            continue
        px = O[e]
        mode = c.get("stop_mode", "pts")
        if mode == "px":
            dist = d * (px - s["stop_px"]) + c.get("stop_buf", 0.0)
        elif mode == "adr":
            dist = c["stop_adr"] * D.adr10[i]
        else:
            dist = c["stop_pts"]
        if not np.isfinite(dist):
            continue
        dist = max(dist, 10.0)
        if dist > MAXSTOP:
            if cap == "skip":
                continue
            dist = MAXSTOP
        stop = px - d * dist
        tp = np.nan
        if c.get("tp_R") is not None:
            tp = px + d * c["tp_R"] * dist
        elif c.get("tp_pts") is not None:
            tp = px + d * c["tp_pts"]
        tr = c.get("trail"); tr = np.nan if tr is None else (tr * D.adr10[i] if c.get("trail_adr") else tr)
        tt = c.get("ttrig", 0.0) or 0.0
        if tt == "same":
            tt = tr if tr == tr else 0.0
        if c.get("ttrig_R") is not None:
            tt = c["ttrig_R"] * dist
        be = c.get("be"); be = np.nan if be is None else be
        if c.get("be_R") is not None:
            be = c["be_R"] * dist
        vb = c.get("vwbuf"); vb = np.nan if vb is None else vb
        tb = int(c.get("tbars", 0) or 0)
        xi = int(c.get("xidx", EOD))
        if e > xi:
            continue
        worst = dist * PT + COST
        # 每日風險檢查
        real = sum(p for xs, es, p in day_real.get(i, []) if xs <= e)
        nopen = [w for xs, w in day_open.get(i, []) if xs > e]
        if real - sum(nopen) - worst < -3000:
            continue
        xs, xp, why = sim1(O, H, L, C, VW, e, d, stop, tp, xi, tr, tt, be, vb, tb, 0.0)
        pts = d * (xp - px)
        pnl = pts * PT - COST
        day_real.setdefault(i, []).append((xs, e, pnl))
        day_open.setdefault(i, []).append((xs, worst))
        r = {"date": D.dates[i], "i": i, "P": D.P[i], "d": d, "e": e, "x": xs, "entry": px, "exit": xp,
             "stop0": stop, "dist": dist, "why": int(why), "pts": pts, "pnl": pnl, "worst": worst}
        for k, v in s.items():
            if k not in r and k not in ("stop_px",):
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def dd(p):
    c = np.cumsum(np.asarray(p, float))
    if len(c) == 0:
        return 0.0
    pk = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - pk).min())


def stats(T, col="pnl"):
    if T is None or len(T) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0, worst=0.0)
    p = T[col].values
    dy = T.groupby("date")[col].sum()
    return dict(n=len(p), win=round(100 * (p > 0).mean(), 1), avg=round(p.mean(), 0), tot=round(p.sum(), 0),
                mdd=round(dd(p), 0), worst=round(dy.min(), 0))


def by_period(T, col="pnl", periods=PERIODS):
    out = {}
    for P in periods:
        out[P] = stats(T[T.P == P] if len(T) else T, col)
    out["全期"] = stats(T, col)
    return out


def fmt_table(T, col="pnl", periods=PERIODS):
    bp = by_period(T, col, periods)
    lines = ["| 期間 | 筆數 | 勝率 | 每筆 | 合計 | 最大回撤 | 最差一天 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, s in bp.items():
        if s["n"] == 0:
            lines.append(f"| {k} | 0 | – | – | 0 | 0 | 0 |"); continue
        lines.append(f"| {k} | {s['n']} | {s['win']:.1f}% | {s['avg']:+,.0f} | {s['tot']:+,.0f} | {s['mdd']:+,.0f} | {s['worst']:+,.0f} |")
    return "\n".join(lines)


# ---------------- 基準組合「順勢多點」（照 books17.py 重建） ----------------
def baseline_book():
    F = pd.read_csv(os.path.join(RES, "wr_table.csv.gz")); F["key"] = F.rule + "|" + F.side
    S = pd.read_csv(os.path.join(RES, "swan_shock.csv"))[["date", "rule", "side", "entry_slot", "shock"]]
    F = F.merge(S, on=["date", "rule", "side", "entry_slot"], how="left")
    withI = F.to_vwap == False  # noqa: E712
    P = {"I2": F.key.str.startswith("I2"),
         "前高空順": (F.key == "AM-PD-k1|空") & withI, "前低多順": (F.key == "AM-PD-k1|多") & withI,
         "亞洲多ES": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True),  # noqa: E712
         "亞洲VAL多ES": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),  # noqa: E712
         "美盤早夜盤低多ES": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2)}  # noqa: E712
    m = np.zeros(len(F), bool)
    for n in P:
        m |= P[n].fillna(False).values
    T = F[m].copy()
    T = T[~(T.shock >= 250)]
    T = T.sort_values(["date", "entry_slot"])
    a = T.key.str.startswith("ASIA") & (T.dir == 1)
    T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
    keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []; opened = []
        for t in x.itertuples():
            real = sum(p for xs, p in closed if xs <= t.entry_slot); n_open = sum(1 for xs in opened if xs > t.entry_slot)
            if real - 2948 * (n_open + 1) < -3000:
                keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl55)); opened.append(t.exit_slot)
    T = T[keep].copy()
    T["pnl"] = T.pnl55; T["P"] = [period_of(d) for d in T.date]; T["worst"] = 2948.0
    T["src"] = "base"
    return T[["date", "P", "dir", "entry_slot", "exit_slot", "pnl", "worst", "src", "key"]]


def combine(base, mine):
    """把我的交易（RTH 索引 e/x）轉成 18:00 起的 slot，與基準合併後重新套每日 $3,000 檢查。"""
    m = mine.copy()
    m["entry_slot"] = m.e + data.RTH_OFF; m["exit_slot"] = m.x + data.RTH_OFF; m["dir"] = m.d; m["src"] = "mine"
    m["key"] = "mine"
    A = pd.concat([base, m[["date", "P", "dir", "entry_slot", "exit_slot", "pnl", "worst", "src", "key"]]])
    A = A.sort_values(["date", "entry_slot", "src"]).reset_index(drop=True)
    keep = []
    for d, x in A.groupby("date", sort=False):
        closed = []; opened = []
        for t in x.itertuples():
            real = sum(p for xs, p in closed if xs <= t.entry_slot)
            ow = sum(w for xs, w in opened if xs > t.entry_slot)
            if real - ow - t.worst < -3000:
                keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl)); opened.append((t.exit_slot, t.worst))
    return A[keep].copy()


def daily_corr(D, base, mine):
    days = [D.dates[i] for i in range(D.n) if D.ok[i]]
    a = base.groupby("date").pnl.sum().reindex(days).fillna(0)
    b = mine.groupby("date").pnl.sum().reindex(days).fillna(0)
    both = (a != 0) & (b != 0)
    return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 5 else np.nan, int(both.sum())
