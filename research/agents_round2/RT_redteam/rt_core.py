"""RT_redteam 獨立實作（不 import A_trend／C_indicators／B_exits 的任何程式）。研究用途，不是投資建議。
只用共用資料載入 lib.data.load_grid（資料本身），其餘：模擬器、ADR10、C1 訊號、TD10 五個指標、基準組合、
每日 $3,000 風控，全部自己寫。

時間：slot 930＝09:30，959＝09:59，960＝10:00，1315＝15:55，1319＝15:59（18:00 起算的分鐘格）。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RES)
from lib import data  # noqa: E402  只用資料載入

PT = 20.0
COST = 48.0
R0 = 930
X1555 = 930 + 385
PER = ["探索", "驗證", "測試", "近期"]


def period(d):
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


g = data.load_grid("NQ", include_holdout=True)
ge = data.load_grid("ES", include_holdout=True)
N = len(g)
DATES = g.dates
PERIOD = np.array([period(d) for d in DATES])
O, H, L, C, V = g.O.copy(), g.H.copy(), g.L.copy(), g.C.copy(), g.V.copy()

# ---- 壞價偵測（獨立檢查）：一根 K 的高/低離前後收盤 >250 點 ----
def bad_bars():
    out = []
    for i in range(N):
        c = pd.Series(C[i]).ffill().values
        pc = np.concatenate([[np.nan], c[:-1]])
        for s in range(R0, R0 + 390):
            if np.isnan(H[i, s]):
                continue
            ref_hi = np.nanmax([pc[s], C[i, s]]); ref_lo = np.nanmin([pc[s], C[i, s]])
            if H[i, s] - ref_hi > 250 or ref_lo - L[i, s] > 250:
                out.append((DATES[i], s, O[i, s], H[i, s], L[i, s], C[i, s], pc[s]))
    return out


BAD = bad_bars()
for (d, s, o, h, l, c, p) in BAD:   # 修正：開盤改成前一根收盤，高低改成開收範圍
    i = g.dindex[d]
    if abs(o - p) > 250:
        O[i, s] = p
    H[i, s] = max(O[i, s], C[i, s]); L[i, s] = min(O[i, s], C[i, s])

# ---- 可交易日：真正 RTH、前一個 RTH 日同合約 ----
PREV_SAME = np.array([bool(g.tradable[i]) and g.prev_same_contract(i) for i in range(N)])
OKDAY = PREV_SAME & np.isfinite(O[:, R0]) & (PERIOD != "無")

# ---- ADR10：前 10 個「可交易日」RTH 振幅平均，不含當天 ----
with np.errstate(all="ignore"):
    RNG = np.nanmax(H[:, R0:R0 + 390], 1) - np.nanmin(L[:, R0:R0 + 390], 1)
ADR10 = np.full(N, np.nan)
_h = []
for i in range(N):
    if len(_h) >= 10:
        ADR10[i] = np.mean(_h[-10:])
    if g.tradable[i] and np.isfinite(RNG[i]):
        _h.append(RNG[i])


# ---------------------------------------------------------------- 模擬器
def sim(i, e, d, sl, trail=None, trig=None, tp=None, xs=X1555, entry="open", slip_stop=0.0,
        cost=COST, pess_trail=False, size=1.0):
    """回傳 dict 或 None。
    entry='open'：進場價＝O[e]；'adverse'：多單用 H[e]、空單用 L[e]。
    停損／移動停損：價位在下一分鐘才生效（本分鐘先檢查舊停損，再用本分鐘高低更新）；跳空穿越用開盤價。
    pess_trail=True：同一分鐘若新移動停損也被本分鐘低（高）碰到，就當作先高後低，以新停損出場（另一種假設）。
    slip_stop：停損成交再差幾點。"""
    while e <= xs and np.isnan(O[i, e]):
        e += 1
    if e > xs:
        return None
    if entry == "open":
        px = O[i, e]
    else:
        px = H[i, e] if d == 1 else L[i, e]
    stop = px - d * sl
    tpp = None if tp is None else px + d * tp
    best = 0.0
    xp = np.nan; xslot = -1; why = "time"
    for s in range(e, xs + 1):
        h, l, o = H[i, s], L[i, s], O[i, s]
        if np.isnan(h):
            continue
        if (d == 1 and l <= stop) or (d == -1 and h >= stop):
            f = stop
            if s > e and ((d == 1 and o < stop) or (d == -1 and o > stop)):
                f = o
            xp = f - d * slip_stop; xslot = s; why = "trail" if (d * (stop - px) > -sl + 1e-9) else "sl"
            break
        if tpp is not None and ((d == 1 and h >= tpp) or (d == -1 and l <= tpp)):
            xp = tpp; xslot = s; why = "tp"; break
        if s == xs:
            xp = C[i, s]; xslot = s; why = "time"; break
        fav = (h - px) if d == 1 else (px - l)
        if fav > best:
            best = fav
        if trail is not None and best >= trig:
            ns = px + d * (best - trail)
            if (d == 1 and ns > stop) or (d == -1 and ns < stop):
                stop = ns
                if pess_trail and ((d == 1 and l <= stop) or (d == -1 and h >= stop)):
                    xp = stop - d * slip_stop; xslot = s; why = "trail"; break
    if xslot < 0:
        k = xs
        while k >= e and np.isnan(C[i, k]):
            k -= 1
        xp = C[i, k]; xslot = k; why = "time"
    pts = d * (xp - px)
    return dict(date=DATES[i], i=i, P=PERIOD[i], d=d, e=e, x=xslot, entry=px, exit=xp, why=why, pts=pts,
                pnl=(pts * PT - cost) * size, worst=(sl * PT + cost) * size)


def run(sigs, **kw):
    rows = []
    for s in sigs:
        kk = dict(kw); kk.update({k: v for k, v in s.items() if k not in ("i", "e", "d")})
        r = sim(s["i"], s["e"], s["d"], **kk)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C1 開盤 5 分鐘驅動
def c1_signals(N_=5, k=0.15, stop_adr=0.25, cap=145.0, trail=100.0):
    out = []
    for i in range(N):
        if not OKDAY[i] or not np.isfinite(ADR10[i]):
            continue
        c = C[i, R0 + N_ - 1]
        if np.isnan(c):
            continue
        m = c - O[i, R0]
        if m == 0 or abs(m) < k * ADR10[i]:
            continue
        sl = min(max(stop_adr * ADR10[i], 10.0), cap)
        out.append(dict(i=i, e=R0 + N_, d=1 if m > 0 else -1, sl=sl, trail=trail, trig=trail))
    return out


# ---------------------------------------------------------------- TD10 五個指標
def five_min_indicators(adjust=True):
    """連續 5 分 K（含夜盤）。adjust=True：換月那天把之前的價格平移，消掉換月跳空（只用當時已知的差值，
    指標都是平移不變的，所以等同只用過去資料）。回傳 dict 指標 → (N, 288) 陣列（第 b 根＝slot 5b..5b+4）。"""
    nb = 288
    o5 = O.reshape(N, nb, 5); h5 = H.reshape(N, nb, 5); l5 = L.reshape(N, nb, 5); c5 = C.reshape(N, nb, 5)
    ok = np.isfinite(c5).any(2)
    with np.errstate(all="ignore"):
        bh = np.nanmax(h5, 2); bl = np.nanmin(l5, 2)
    lastk = np.where(np.isfinite(c5), np.arange(5)[None, None, :], -1).max(2)
    bc = np.take_along_axis(c5, np.clip(lastk, 0, 4)[..., None], 2)[..., 0]
    off = np.zeros(N)
    if adjust:
        for i in range(1, N):
            if g.contract[i] != g.contract[i - 1]:
                pc = C[i - 1][np.isfinite(C[i - 1])]; no = O[i][np.isfinite(O[i])]
                if len(pc) and len(no):
                    off[i] = no[0] - pc[-1]
        cum = np.cumsum(off)       # 第 i 天起要減掉的累計換月差（往前不需要知道未來）
    else:
        cum = np.zeros(N)
    di, bi = np.where(ok)
    h = bh[di, bi] - cum[di]; l = bl[di, bi] - cum[di]; c = bc[di, bi] - cum[di]
    h = pd.Series(h); l = pd.Series(l); c = pd.Series(c)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    slope = (e20 - e20.shift(6)) / atr
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    mh = (macd - macd.ewm(span=9, adjust=False).mean()) / atr
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0)); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    out = {}
    for nm, s in [("slope", slope), ("di", pdi - ndi), ("macdh", mh)]:
        a = np.full((N, nb), np.nan); a[di, bi] = s.values
        # 往前補（缺 K 時用最後一根已完成的）
        a = pd.DataFrame(a.reshape(-1)).ffill().values.reshape(N, nb)
        out[nm] = a
    return out


def intraday_10(t=960):
    """10:00 決策：只用 slot 930..t-1 的 1 分 K。"""
    eff = np.full(N, np.nan); avw = np.full(N, np.nan)
    for i in range(N):
        c = C[i, R0:t]; okm = np.isfinite(c)
        if okm.sum() < 10:
            continue
        o = O[i, R0] if np.isfinite(O[i, R0]) else c[okm][0]
        cc = c[okm]
        path = np.abs(np.diff(np.concatenate([[o], cc]))).sum()
        eff[i] = (cc[-1] - o) / path if path > 0 else 0.0
        tp = (H[i, R0:t] + L[i, R0:t] + C[i, R0:t]) / 3; vv = V[i, R0:t]
        mm = np.isfinite(tp) & np.isfinite(vv)
        cv = np.cumsum(np.where(mm, tp * vv, 0)); cw = np.cumsum(np.where(mm, vv, 0))
        vw = np.where(cw > 0, cv / np.where(cw > 0, cw, 1), np.nan)
        avw[i] = np.mean((c[okm] > vw[okm]).astype(float)) * 2 - 1
    return eff, avw


def td_features(t=960, adjust=True):
    eff, avw = intraday_10(t)
    k = five_min_indicators(adjust)
    b = t // 5 - 1          # 最後一根「已完成」的 5 分 K（10:00 → 09:55–09:59）
    return pd.DataFrame({"effi": eff, "above_vw": avw, "slope": k["slope"][:, b], "di": k["di"][:, b], "macdh": k["macdh"][:, b]})


def td_signal(F, q=0.8, k=3, feats=("effi", "above_vw", "slope", "di", "macdh"), thr=None, train=("探索",)):
    okd = OKDAY & np.isin(PERIOD, train)
    up = np.zeros(N); dn = np.zeros(N); used = {}
    for f in feats:
        x = F[f].values
        th = thr[f] if thr is not None else np.nanquantile(np.abs(x[okd]), q)
        used[f] = th
        up += np.nan_to_num(x) > th; dn += np.nan_to_num(x) < -th
    dv = np.where((up >= k) & (dn == 0), 1, np.where((dn >= k) & (up == 0), -1, 0))
    return np.where(OKDAY, dv, 0), used


def td_sigs(dv, t=960, sl=100.0):
    return [dict(i=int(i), e=t, d=int(dv[i]), sl=sl) for i in np.where(dv != 0)[0]]


# ---------------------------------------------------------------- 基準「順勢多點」（自己從 wr_table 重建）
def base_candidates():
    F = pd.read_csv(os.path.join(RES, "wr_table.csv.gz")); F["key"] = F.rule + "|" + F.side
    S = pd.read_csv(os.path.join(RES, "swan_shock.csv"))[["date", "rule", "side", "entry_slot", "shock"]]
    F = F.merge(S, on=["date", "rule", "side", "entry_slot"], how="left")
    noI = F.to_vwap == False  # noqa: E712
    es = F.es_ok == True  # noqa: E712
    m = (F.key.str.startswith("I2") | ((F.key == "AM-PD-k1|空") & noI) | ((F.key == "AM-PD-k1|多") & noI)
         | ((F.key == "ASIA-PD-k1|多") & es) | ((F.key == "ASIA-VA-k1|多") & es)
         | ((F.key == "AM-ON-k1|多") & es & (F.vol >= 1.2)))
    T = F[m.fillna(False)].copy()
    T = T[~(T.shock >= 250)].sort_values(["date", "entry_slot"])
    a = T.key.str.startswith("ASIA") & (T.dir == 1)
    T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"], kind="stable")
    return T.reset_index(drop=True)


def base_trades(tp=55.0, sl=145.0, size=1.0, entry="open", slip_stop=0.0, cost=COST):
    """每個候選點自己模擬 +tp／−sl 到預定出場時間，得到真實出場分鐘。"""
    T = base_candidates()
    rows = []
    for t in T.itertuples():
        i = g.dindex[t.date]
        r = sim(i, int(t.entry_slot), int(t.dir), sl, tp=tp, xs=int(t.exit_slot), entry=entry, slip_stop=slip_stop,
                cost=cost, size=size)
        if r is None:
            continue
        r.update(src="base", key=t.key, xsched=int(t.exit_slot), pnl55=t.pnl55)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 每日 $3,000 風控
def book(T, mode="real", lim=3000.0, single=False):
    """T 需要 date, e(進場 slot), x(實際出場 slot), xsched(預定出場；沒有就用 x), pnl, worst。
    mode='real'：已實現＝實際出場時間 ≤ 本筆進場（平倉就算）；'sched'：等到預定出場時間才算（books17 寫法），
    而且預定出場前都當作持倉、占用最壞虧損。
    single=True：同一時間只允許一個部位（真實帳戶單一淨部位；有部位時新訊號直接跳過）。"""
    T = T.copy()
    if "xsched" not in T or T.xsched.isna().any():
        T["xsched"] = T.get("xsched", pd.Series(index=T.index, dtype=float)).fillna(T.x)
    T["src_o"] = T.src.map({"base": 0}).fillna(1)
    T = T.sort_values(["date", "e", "src_o"], kind="stable").reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for d, idx in T.groupby("date", sort=False).groups.items():
        hist = []   # (realize_slot, hold_until, pnl, worst)
        for k in idx:
            t = T.loc[k]
            ru = (lambda h: h[1]) if mode == "sched" else (lambda h: h[0])
            real = sum(h[2] for h in hist if ru(h) <= t.e)
            openw = sum(h[3] for h in hist if ru(h) > t.e)
            if single and any(h[0] >= t.e for h in hist):
                continue
            if real - openw - t.worst < -lim:
                continue
            keep[k] = True
            hist.append((t.x, max(t.xsched, t.x), t.pnl, t.worst))
    return T[keep].copy()


# ---------------------------------------------------------------- 統計
def dd(p):
    c = np.cumsum(np.asarray(p, float))
    if len(c) == 0:
        return 0.0
    pk = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - pk).min())


def stats(T):
    if T is None or len(T) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, tot=0.0, mdd=0.0, worst=0.0)
    T = T.sort_values(["date", "e"])
    p = T.pnl.values
    dy = T.groupby("date").pnl.sum()
    return dict(n=len(p), win=round(100 * (p > 0).mean(), 1), avg=round(p.mean()), tot=round(p.sum()),
                mdd=round(dd(dy.values)), worst=round(dy.min()))


def table(T, name=""):
    rows = []
    for P in PER + ["全期"]:
        x = T if P == "全期" else T[T.P == P]
        s = stats(x); s["期間"] = P; s["名稱"] = name
        rows.append(s)
    r = pd.DataFrame(rows)[["名稱", "期間", "n", "win", "avg", "tot", "mdd", "worst"]]
    return r


def line(T, name):
    s = {P: stats(T[T.P == P]) for P in PER}
    a = stats(T)
    rr = a["tot"] / -a["mdd"] if a["mdd"] < 0 else np.inf
    return (f"{name:34s} " + " | ".join(f"{P} {s[P]['n']:3d}筆 {s[P]['avg']:+6.0f} {s[P]['tot']:+8,.0f}" if s[P]['n'] else f"{P}   0筆" for P in PER)
            + f" || 全期 {a['n']}筆 {a['tot']:+,.0f} 回撤 {a['mdd']:+,.0f} 比 {rr:.1f}")
