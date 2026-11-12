"""R1_backtest 共用函式（研究用途，不是投資建議）。

依 Q1_critic/strategies_v2.json：
- 時間：slot 0＝前一天 18:00；930＝09:30；1319＝15:59。N 分 K 從 18:00 對齊（09:30 剛好是 5/15 分 K 開頭）；
  60 分 K 夜盤從 18:00 對齊、RTH 從 09:30 對齊（09:00～09:29 一根 30 分）。有效分鐘 < N/2 的 K 無效（直接丟掉）。
- 市價：訊號那根最後一分鐘收盤後，下一分鐘開盤進場（QC3：落在 958～962 → 963，前提是 e..962 沒碰停損或止盈）。
- 限價（QC1）：L[s] <= P−0.25 才成交，成交價 min(P, O[s])；成交那分鐘 L<=停損 → 停損；O 已在停損外 → 完整虧損 P−停損；
  成交那分鐘不止盈；958～962 不算觸價。
- QC2：停損 <15 → 15；>145 → 不做。
- 每口 $48、20 美元／點；同分鐘止盈止損都碰到算止損；跳空穿停損用開盤價（較差）；止盈用限價價位（跳空不給好價）。
資料：lib.data 網格；唯一修正 2026-10-30 15:03 一根 Yahoo 壞價（與 A／RT／R2 同規則：NQ 門檻 250 點、ES 80 點）。
"""
from __future__ import annotations
import os, sys
sys.dont_write_bytecode = True
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
AG = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/agents"
OUT = os.path.join(AG, "R1_backtest")
sys.path.insert(0, RES)
from lib import data, features  # noqa: E402

PT, COST, LIM = 20.0, 48.0, 3000.0
R0, EOD = 930, 1319
NEWS_A, NEWS_B = 958, 962         # 09:58～10:02
SEGS = ["探索", "驗證", "測試", "近期"]
PEN = 0.25


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
CONTRACT = g.contract
ECONTRACT = ge.contract
TRAD = g.tradable & (SEG != "無") & np.isfinite(O[:, R0])
ETRAD = ge.tradable & np.isfinite(EO[:, R0])
PREV_SAME = np.array([bool(g.tradable[i]) and g.prev_same_contract(i) for i in range(N)])
LV = features.daily_levels(g).set_index("i")
ELV = features.daily_levels(ge).set_index("i")

with np.errstate(all="ignore"):
    RNG = np.nanmax(H[:, R0:R0 + 390], 1) - np.nanmin(L[:, R0:R0 + 390], 1)
ADR10 = np.full(N, np.nan)
_h = []
for _i in range(N):
    if len(_h) >= 10:
        ADR10[_i] = np.mean(_h[-10:])
    if g.tradable[_i] and np.isfinite(RNG[_i]):
        _h.append(RNG[_i])


def vwap_row(Hr, Lr, Cr, Vr):
    """09:30 起 (H+L+C)/3 量加權，算到該分鐘（含）。slot < 930 為 NaN。"""
    out = np.full(1440, np.nan)
    tp = (Hr[R0:] + Lr[R0:] + Cr[R0:]) / 3
    vv = Vr[R0:]
    m = np.isfinite(tp) & np.isfinite(vv)
    cv = np.cumsum(np.where(m, tp * vv, 0.0)); cw = np.cumsum(np.where(m, vv, 0.0))
    with np.errstate(all="ignore"):
        out[R0:] = np.where(cw > 0, cv / cw, np.nan)
    return out


_VW = {}


def VW(i):
    if i not in _VW:
        _VW[i] = vwap_row(H[i], L[i], C[i], V[i])
    return _VW[i]


# ---------------------------------------------------------------- K 棒
def bars(o, h, l, c, v, t, tf):
    """把一段分鐘序列（長度為 tf 倍數，起點已對齊）聚成 tf 分 K，丟掉無效 K。
    回傳 dict：O,H,L,C,V,ts(第一分鐘全域時間),te(最後一分鐘全域時間)。"""
    n = len(o) // tf
    O_ = o[:n * tf].reshape(n, tf); H_ = h[:n * tf].reshape(n, tf); L_ = l[:n * tf].reshape(n, tf)
    C_ = c[:n * tf].reshape(n, tf); V_ = v[:n * tf].reshape(n, tf); T_ = t[:n * tf].reshape(n, tf)
    fin = np.isfinite(C_)
    cnt = fin.sum(1)
    ok = cnt >= tf / 2.0
    first = np.argmax(fin, 1)
    last = tf - 1 - np.argmax(fin[:, ::-1], 1)
    r = np.arange(n)
    with np.errstate(all="ignore"):
        out = dict(O=O_[r, first], H=np.nanmax(H_, 1), L=np.nanmin(L_, 1), C=C_[r, last], V=np.nansum(V_, 1),
                   ts=T_[:, 0], te=T_[:, -1])
    return {k: a[ok] for k, a in out.items()}


def bars_custom(o, h, l, c, v, t, edges):
    """自訂邊界（edges＝每根第一個 index 的串列，最後一個是結尾）。用在 60 分 K。"""
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        cc = c[a:b]; fin = np.isfinite(cc)
        if fin.sum() < (b - a) / 2.0 or fin.sum() == 0:
            continue
        idx = np.where(fin)[0]
        rows.append((o[a + idx[0]], np.nanmax(h[a:b]), np.nanmin(l[a:b]), cc[idx[-1]], np.nansum(v[a:b]), t[a], t[b - 1]))
    if not rows:
        return dict(O=np.array([]), H=np.array([]), L=np.array([]), C=np.array([]), V=np.array([]), ts=np.array([]), te=np.array([]))
    A = np.array(rows)
    return dict(O=A[:, 0], H=A[:, 1], L=A[:, 2], C=A[:, 3], V=A[:, 4], ts=A[:, 5].astype(int), te=A[:, 6].astype(int))


def day_bars(i, tf, es=False, a=0, b=1440):
    """單日 slot a..b-1（a 對齊 tf）的 tf 分 K；全域時間＝slot。"""
    A = (EO, EH, EL, EC, EV) if es else (O, H, L, C, V)
    t = np.arange(a, b)
    return bars(*(x[i, a:b] for x in A), t, tf)


def seq(rows, start_slot):
    """多日分鐘序列：rows[0] 從 start_slot 開始，其他列整天；全域時間 t＝(列序)*1440＋slot，今天（最後一列）的 t 偏移回傳。"""
    parts = {k: [] for k in "ohlcvt"}
    for k_, r in enumerate(rows):
        a = start_slot if k_ == 0 else 0
        for nm, arr in zip("ohlcv", (O, H, L, C, V)):
            parts[nm].append(arr[r, a:])
        parts["t"].append(np.arange(a, 1440) + k_ * 1440)
    s = {k: np.concatenate(v) for k, v in parts.items()}
    return s, (len(rows) - 1) * 1440


def pivots(hi, lo, n=2):
    """C3：轉折高 H[k] > max(H[k-n..k-1]) 且 H[k] >= max(H[k+1..k+n])；低點相反。回傳 (ph, pl) bool 陣列（第 k 根是轉折，k+n 收盤確定）。"""
    m = len(hi)
    ph = np.zeros(m, bool); pl = np.zeros(m, bool)
    for k in range(n, m - n):
        if hi[k] > hi[k - n:k].max() and hi[k] >= hi[k + 1:k + n + 1].max():
            ph[k] = True
        if lo[k] < lo[k - n:k].min() and lo[k] <= lo[k + 1:k + n + 1].min():
            pl[k] = True
    return ph, pl


def wilder_atr(h, l, c, n=14):
    """Wilder ATR，前 n 根暖身（之前 NaN）。輸入為已去掉 NaN 的 K 棒序列。第 k 個值＝第 k 根收盤時。"""
    m = len(h)
    out = np.full(m, np.nan)
    if m == 0:
        return out
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.where(np.isnan(pc), h - l, np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))))
    if m < n:
        return out
    a = tr[:n].mean(); out[n - 1] = a
    for k in range(n, m):
        a = (a * (n - 1) + tr[k]) / n; out[k] = a
    return out


# ---------------------------------------------------------------- 進場處理
def qc2(px, stop, d):
    """回傳調整後停損價與 sl 點數；>145 或方向錯 → None。"""
    sl = d * (px - stop)
    if not np.isfinite(sl) or sl <= 0:
        return None
    if sl > 145:
        return None
    if sl < 15:
        sl = 15.0; stop = px - d * 15.0
    return stop, sl


def tgt_px(px, stop, d, tgt, rr=2.0):
    """結構版止盈價：tgt 為 None → rr×R；否則用 tgt，若 RR<1（含方向錯）→ 2R。"""
    sl = d * (px - stop)
    if tgt is not None and np.isfinite(tgt):
        tp = d * (tgt - px)
        if tp >= sl:
            return tgt
    return px + d * rr * sl


def mkt_signal(i, d, e, stop_raw, tgt=None, no_tp=False):
    """市價訊號：e＝原進場 slot。QC3 處理、QC2 停損。回傳 dict(e, px, stop, sl, tp) 或 None（含取消原因）。"""
    while e <= EOD and np.isnan(O[i, e]):
        e += 1
    if e > EOD:
        return None, "無價"
    moved = False
    if NEWS_A <= e <= NEWS_B:
        px0 = O[i, e]
        q = qc2(px0, stop_raw, d)
        if q is None:
            return None, ">145"
        st0, sl0 = q
        tp0 = None if no_tp else tgt_px(px0, st0, d, tgt)
        for s in range(e, NEWS_B + 1):
            if np.isnan(H[i, s]):
                continue
            if (d == 1 and L[i, s] <= st0) or (d == -1 and H[i, s] >= st0):
                return None, "QC3取消"
            if tp0 is not None and ((d == 1 and H[i, s] >= tp0) or (d == -1 and L[i, s] <= tp0)):
                return None, "QC3取消"
        e = NEWS_B + 1
        while e <= EOD and np.isnan(O[i, e]):
            e += 1
        moved = True
    px = O[i, e]
    q = qc2(px, stop_raw, d)
    if q is None:
        return None, ">145"
    stop, sl = q
    return dict(e=e, px=px, stop=stop, sl=sl, tp=None if no_tp else tgt_px(px, stop, d, tgt), moved=moved,
                floor15=d * (px - stop_raw) < 15), "ok"


def lim_fill(i, d, P, a, b, pen=PEN, extra_ok=None):
    """限價 P 在 slot a..b（含）有效：第一個 L<=P-pen（空：H>=P+pen）的 slot（跳過 958～962、NaN）。
    extra_ok(s)：該分鐘是否仍有效（例如失效條件），False → 回傳 -1。"""
    for s in range(a, b + 1):
        if NEWS_A <= s <= NEWS_B or np.isnan(H[i, s]):
            continue
        if extra_ok is not None and not extra_ok(s):
            return -1
        if (d == 1 and L[i, s] <= P - pen) or (d == -1 and H[i, s] >= P + pen):
            return s
    return -1


# ---------------------------------------------------------------- 模擬
def sim(i, d, e, stop, tp, xs, limit_P=None, be_R=None, cost=COST):
    """e＝進場（或限價成交）slot。stop/tp 為價位（tp None＝不設）。limit_P：限價單價位（成交那分鐘規則）。
    be_R：浮盈 >= be_R 點後停損移到進場價（下一分鐘生效）。回傳 dict。"""
    if limit_P is not None:
        px = min(limit_P, O[i, e]) if d == 1 else max(limit_P, O[i, e])
        if (d == 1 and O[i, e] <= stop) or (d == -1 and O[i, e] >= stop):
            pts = -abs(limit_P - stop)          # QC1：開盤已在停損外 → 記完整虧損 P−停損
            return dict(e=e, px=limit_P, x=e, xp=stop, why="sl_gap", pts=pts, pnl=pts * PT - cost)
        if (d == 1 and L[i, e] <= stop) or (d == -1 and H[i, e] >= stop):
            pts = d * (stop - px)
            return dict(e=e, px=px, x=e, xp=stop, why="sl", pts=pts, pnl=pts * PT - cost)
        s0 = e + 1
    else:
        px = O[i, e]
        s0 = e
    best = 0.0
    st = stop
    for s in range(s0, xs + 1):
        h, l, o = H[i, s], L[i, s], O[i, s]
        if np.isnan(h):
            continue
        if (d == 1 and l <= st) or (d == -1 and h >= st):
            f = st
            if s > e and ((d == 1 and o < st) or (d == -1 and o > st)):
                f = o
            pts = d * (f - px)
            return dict(e=e, px=px, x=s, xp=f, why="be" if abs(st - px) < 1e-9 else "sl", pts=pts, pnl=pts * PT - cost)
        if tp is not None and ((d == 1 and h >= tp) or (d == -1 and l <= tp)):
            pts = d * (tp - px)
            return dict(e=e, px=px, x=s, xp=tp, why="tp", pts=pts, pnl=pts * PT - cost)
        fav = (h - px) if d == 1 else (px - l)
        best = max(best, fav)
        if be_R is not None and best >= be_R:
            st = max(st, px) if d == 1 else min(st, px)
    k = xs
    while k >= e and np.isnan(C[i, k]):
        k -= 1
    pts = d * (C[i, k] - px)
    return dict(e=e, px=px, x=k, xp=C[i, k], why="time", pts=pts, pnl=pts * PT - cost)


def b55_xs(e):
    return slot("12:29") if e < slot("12:00") else slot("15:49")


# ---------------------------------------------------------------- 統計
def dd(p):
    c = np.cumsum(np.asarray(p, float))
    if len(c) == 0:
        return 0.0
    pk = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - pk).min())


def stats(T, col="pnl"):
    if T is None or len(T) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0, worst=0.0)
    T = T.sort_values(["date", "e"])
    p = T[col].values
    dy = T.groupby("date")[col].sum()
    return dict(n=len(p), win=100 * (p > 0).mean(), avg=p.mean(), tot=p.sum(), mdd=dd(p), worst=dy.min())


def seg_table(T, col="pnl"):
    return {s: stats(T[T.seg == s] if len(T) else T, col) for s in SEGS} | {"全期": stats(T, col)}


def fmt_seg(st):
    s = ""
    for k in SEGS + ["全期"]:
        v = st[k]
        if v["n"] == 0:
            s += f"  {k}: 0筆\n"; continue
        s += (f"  {k}: {v['n']:4d}筆 勝率{v['win']:5.1f}% 每筆{v['avg']:+7.0f} 合計{v['tot']:+9,.0f} "
              f"回撤{v['mdd']:+8,.0f} 最差一天{v['worst']:+7,.0f}\n")
    a = st["全期"]
    s += f"  全期 報酬/回撤 {a['tot'] / -a['mdd'] if a['mdd'] < 0 else float('inf'):.2f}\n"
    return s
