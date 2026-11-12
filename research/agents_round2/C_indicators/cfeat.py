"""C_indicators 特徵建構（研究用途，不是投資建議）。
全部特徵只用「決策時間之前已收盤」的資料：
  - 日線特徵：只用前一個 RTH 交易日（含）以前的資料，以及當天 09:29 前的夜盤。
  - 盤中特徵（10:00／10:30）：只用 09:30 到決策分鐘前一分鐘（959／989 格）收盤的 K 棒。
  - 候選進場的 5 分 K 指標：只用進場分鐘前最後一根「已完成」的 5 分 K（entry_slot//5 - 1）。
輸出：days.pkl（每日特徵＋目標）、cand.pkl（wr_table 7,517 筆＋進場當下指標）。
執行：python cfeat.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
sys.path.insert(0, RES)
import numpy as np, pandas as pd
from lib import data, features
from lib.data import RTH_OFF as R0

g = data.load_grid("NQ", include_holdout=True)
ge = data.load_grid("ES", include_holdout=True)
lv = features.daily_levels(g)
N = len(g)


def period(d):
    d = np.asarray(d, dtype=str)
    return np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-09-30"], ["探索", "驗證", "測試"], "近期")


# ---------- 換月後復權（差值），只用來算跨日指標 ----------
def adj_offsets(G):
    off = np.zeros(len(G))
    for i in range(1, len(G)):
        if G.contract[i] != G.contract[i - 1]:
            c_prev = G.C[i - 1][np.isfinite(G.C[i - 1])]
            o_new = G.O[i][np.isfinite(G.O[i])]
            if len(c_prev) and len(o_new):
                off[i] = o_new[0] - c_prev[-1]
    # 早期的價格要加上之後所有換月差
    cum = np.cumsum(off[::-1])[::-1]
    return np.concatenate([cum[1:], [0.0]])   # 第 i 天要加的值＝i 之後的換月差總和


# ---------- 5 分 K 連續序列 ----------
def five_min(G):
    adj = adj_offsets(G)
    nb = 288
    O = G.O.reshape(N, nb, 5); H = G.H.reshape(N, nb, 5); L = G.L.reshape(N, nb, 5)
    C = G.C.reshape(N, nb, 5); V = G.V.reshape(N, nb, 5)
    ok = np.isfinite(C).any(2)
    with np.errstate(all="ignore"):
        bh = np.nanmax(H, 2); bl = np.nanmin(L, 2); bv = np.nansum(V, 2)
    # 最後一根有效收盤
    idx_last = np.where(np.isfinite(C), np.arange(5)[None, None, :], -1).max(2)
    bc = np.take_along_axis(C, np.clip(idx_last, 0, 4)[..., None], 2)[..., 0]
    bc = np.where(ok, bc, np.nan)
    bh = bh + adj[:, None]; bl = bl + adj[:, None]; bc = bc + adj[:, None]
    di, bi = np.where(ok)
    df = pd.DataFrame({"i": di, "b": bi, "h": bh[di, bi], "l": bl[di, bi], "c": bc[di, bi], "v": bv[di, bi]})
    c = df.c; h = df.h; l = df.l
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macdh"] = (m - m.ewm(span=9, adjust=False).mean()) / atr
    e20 = c.ewm(span=20, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    df["ema_x"] = (e20 - e50) / atr
    df["ema_slope"] = (e20 - e20.shift(6)) / atr
    upm = h.diff(); dnm = -l.diff()
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0); ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    pdi = 100 * pd.Series(pdm).ewm(alpha=1 / 14, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    df["di"] = pdi - ndi
    sd = c.rolling(20).std()
    bbw = 4 * sd / c
    df["bbw_rel"] = bbw / bbw.rolling(288 * 5, min_periods=288).median()
    df["bb_pos"] = (c - c.rolling(20).mean()) / (2 * sd)
    df["atr5_rel"] = atr / atr.rolling(288 * 20, min_periods=288 * 3).median()
    # RVOL：近 30 分鐘量 / 前 20 天同一時段平均
    VB = np.where(ok, bv, np.nan)
    s30 = pd.DataFrame(VB).T.rolling(6, min_periods=3).sum().T.values   # 同一天內
    hist = pd.DataFrame(s30).rolling(20, min_periods=5).mean().shift(1).values
    rv = s30 / hist
    df["rvol30"] = rv[df.i.values, df.b.values]
    return df


# ---------- 日線特徵 ----------
def daily(G):
    rH, rL, rC, rO, rV = G.rth(G.H), G.rth(G.L), G.rth(G.C), G.rth(G.O), G.rth(G.V)
    adj = adj_offsets(G)
    out = pd.DataFrame({"date": G.dates})
    with np.errstate(all="ignore"):
        hi = np.nanmax(rH, 1); lo = np.nanmin(rL, 1)
    last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in rC])
    first = rO[:, 0]
    out["hi"], out["lo"], out["cl"], out["op"] = hi, lo, last, first
    out["has_rth"] = G.has_rth; out["tradable"] = G.tradable
    return out, adj


def build():
    D, adj = daily(g)
    De, adje = daily(ge)
    D = D.merge(lv[["date", "pdh", "pdl", "pdc", "vah", "val", "poc", "onh", "onl", "i2_dir", "has_prev", "open_0930"]], on="date")
    rth = D.has_rth.values
    # 前一個 RTH 日的序列（用 shift 前都只取 has_rth 日）
    Dr = D[rth].copy()
    Dr["acl"] = Dr.cl + adj[rth]
    Dr["rng"] = Dr.hi - Dr.lo
    Dr["atr10"] = Dr.rng.rolling(10).mean()
    pc = Dr.acl.shift(1)
    tr = pd.concat([Dr.hi - Dr.lo, (Dr.hi + adj[rth] - pc).abs(), (Dr.lo + adj[rth] - pc).abs()], axis=1).max(1)
    Dr["atr14d"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    Dr["eff"] = (Dr.cl - Dr.op).abs() / Dr.rng                  # 當天方向效率
    Dr["clv"] = (Dr.cl - Dr.lo) / Dr.rng                        # 收盤位置
    Dr["dret"] = (Dr.cl - Dr.op)
    dd = Dr.acl.diff(); up = dd.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dd.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    Dr["rsi_d"] = 100 - 100 / (1 + up / dn)
    m = Dr.acl.ewm(span=12, adjust=False).mean() - Dr.acl.ewm(span=26, adjust=False).mean()
    Dr["macdh_d"] = (m - m.ewm(span=9, adjust=False).mean()) / Dr.atr14d
    e20 = Dr.acl.ewm(span=20, adjust=False).mean(); e50 = Dr.acl.ewm(span=50, adjust=False).mean()
    Dr["emax_d"] = (e20 - e50) / Dr.atr14d
    Dr["emaslope_d"] = (e20 - e20.shift(5)) / Dr.atr14d
    Dr["ret5_d"] = (Dr.acl - Dr.acl.shift(5)) / Dr.atr14d
    Dr["ret20_d"] = (Dr.acl - Dr.acl.shift(20)) / Dr.atr14d
    sd = Dr.acl.rolling(20).std(); bbw = 4 * sd / Dr.acl
    Dr["bbw_d"] = bbw / bbw.rolling(120, min_periods=40).median()
    Dr["atr_rel"] = Dr.atr10 / Dr.atr10.rolling(120, min_periods=40).median()
    Dr["vol_d"] = np.nansum(g.rth(g.V)[rth], 1)
    Dr["rvol_d"] = Dr.vol_d / Dr.vol_d.rolling(20).mean()
    hi_a = Dr.hi + adj[rth]; lo_a = Dr.lo + adj[rth]
    upm = hi_a.diff(); dnm = -lo_a.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=Dr.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=Dr.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / Dr.atr14d; ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / Dr.atr14d
    Dr["adx_d"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()
    Dr["di_d"] = pdi - ndi
    Dr["eff3"] = Dr.eff.rolling(3).mean()
    prevcols = ["atr10", "atr14d", "eff", "clv", "dret", "rsi_d", "macdh_d", "emax_d", "emaslope_d", "ret5_d", "ret20_d",
                "bbw_d", "atr_rel", "rvol_d", "adx_d", "di_d", "eff3", "rng"]
    P = Dr[prevcols].shift(1)   # 前一天收盤後才知道 → 今天可以用
    P.columns = ["p_" + c for c in prevcols]
    P["date"] = Dr.date.values
    D = D.merge(P, on="date", how="left")
    # 非 RTH 日（無）不交易；向前補
    D[P.columns.drop("date")] = D[P.columns.drop("date")].ffill()
    A = D.p_atr10
    D["gap"] = (D.open_0930 - D.pdc) / A
    D["on_rng"] = (D.onh - D.onl) / A
    D["on_ret"] = np.nan
    D["wd"] = pd.to_datetime(D.date).dt.weekday
    D["month"] = pd.to_datetime(D.date).dt.month
    D["open_va"] = D.i2_dir      # +1 開在 VAL 下（I2 做多）、-1 開在 VAH 上
    D["open_vs_pdhl"] = np.where(D.open_0930 > D.pdh, 1, np.where(D.open_0930 < D.pdl, -1, 0))
    # ---- 盤中（10:00、10:30）----
    esd = {d: k for k, d in enumerate(ge.dates)}
    for tag, t in [("10", 960), ("1030", 990)]:
        rows = []
        for i in range(N):
            O, H, L, C, V = g.O[i], g.H[i], g.L[i], g.C[i], g.V[i]
            seg = slice(R0, t)
            c = C[seg]; ok = np.isfinite(c)
            r = {}
            if ok.sum() < 10 or not g.tradable[i]:
                rows.append(r); continue
            o = O[R0] if np.isfinite(O[R0]) else c[ok][0]
            cl = c[ok][-1]; hh = np.nanmax(H[seg]); ll = np.nanmin(L[seg])
            path = np.nansum(np.abs(np.diff(np.concatenate([[o], c[ok]]))))
            a = A.iloc[i]
            r["ret"] = (cl - o) / a
            r["effi"] = (cl - o) / path if path > 0 else 0.0          # 帶正負號的方向效率
            r["ibw"] = (hh - ll) / a
            r["clv"] = (cl - ll) / (hh - ll) if hh > ll else 0.5
            tp = (H[seg] + L[seg] + C[seg]) / 3; vv = V[seg]; mm = np.isfinite(tp) & np.isfinite(vv)
            cv = np.cumsum(np.where(mm, tp * vv, 0)); cw = np.cumsum(np.where(mm, vv, 0))
            vw = np.where(cw > 0, cv / np.where(cw > 0, cw, 1), np.nan)
            r["vwap_d"] = (cl - vw[-1]) / a
            r["above_vw"] = np.nanmean(np.where(ok, (c > vw).astype(float), np.nan)) * 2 - 1   # -1..+1
            r["rvol"] = np.nansum(vv)
            # 最後 30 分鐘（10:30 時＝10:00～10:30；10:00 時＝09:45～10:00 用 15 分）
            k = 30 if tag == "1030" else 15
            c2 = C[t - k:t]; c2 = c2[np.isfinite(c2)]
            r["last_ret"] = (c2[-1] - O[t - k]) / a if len(c2) and np.isfinite(O[t - k]) else np.nan
            r["break_pdh"] = float(hh > D.pdh.iloc[i]) - float(ll < D.pdl.iloc[i])
            r["break_on"] = float(hh > D.onh.iloc[i]) - float(ll < D.onl.iloc[i])
            # ES
            j = esd.get(g.dates[i])
            if j is not None:
                ce = ge.C[j][seg]; oke = np.isfinite(ce); oe = ge.O[j][R0]
                if oke.sum() > 10 and np.isfinite(oe):
                    ae = (ce[oke][-1] - oe) / oe; an = (cl - o) / o
                    r["es_ret"] = ae * 1e4; r["nq_es_rs"] = (an - ae) * 1e4
                    r["same_dir"] = float(np.sign(ae) == np.sign(an)) * np.sign(an)
            rows.append(r)
        X = pd.DataFrame(rows, index=range(N))
        # RVOL：開盤到決策時間累積量 / 前 20 個交易日同時段平均
        vol = X.rvol.copy()
        X["rvol"] = vol / vol.where(D.tradable).rolling(20, min_periods=5).mean().shift(1)
        X.columns = [f"{c}_{tag}" for c in X.columns]
        D = pd.concat([D, X], axis=1)
    # 夜盤 ES／NQ（到 09:29）
    on_n = []
    for i in range(N):
        c = g.C[i][:R0]; okk = np.isfinite(c)
        on_n.append((c[okk][-1] - c[okk][0]) / D.p_atr10.iloc[i] if okk.sum() > 30 else np.nan)
    D["on_ret"] = on_n
    # ---- 目標（只用來評估，不能當特徵）----
    s1555 = R0 + 385
    def px(i, s, col):
        a = col[i][s:s + 1]
        return a[0] if np.isfinite(a[0]) else np.nan
    def last_le(i, s):
        cc = g.C[i][:s + 1]; k = np.where(np.isfinite(cc))[0]
        return cc[k[-1]] if len(k) else np.nan
    for tag, t in [("10", 960), ("1030", 990)]:
        D[f"y_{tag}"] = [last_le(i, s1555) - g.O[i][t] if g.tradable[i] else np.nan for i in range(N)]
    D["y_day_eff"] = [(last_le(i, R0 + 389) - g.O[i][R0]) / (np.nanmax(g.rth(g.H)[i]) - np.nanmin(g.rth(g.L)[i])) if g.tradable[i] else np.nan for i in range(N)]
    D["P"] = period(D.date)
    return D


def cand(D):
    F = pd.read_csv(os.path.join(RES, "wr_table.csv.gz")); F["key"] = F.rule + "|" + F.side
    S = pd.read_csv(os.path.join(RES, "swan_shock.csv"))[["date", "rule", "side", "entry_slot", "shock"]]
    F = F.merge(S, on=["date", "rule", "side", "entry_slot"], how="left")
    fm = five_min(g)
    fm["seq"] = np.arange(len(fm))
    key = fm.i.values * 288 + fm.b.values
    ii = F.date.map(g.dindex).values
    want = ii * 288 + (F.entry_slot.values // 5 - 1)
    pos = np.searchsorted(key, want, side="right") - 1      # 最後一根 <= 需要的 5 分 K
    cols = ["rsi", "macdh", "ema_x", "ema_slope", "adx", "di", "bbw_rel", "bb_pos", "atr5_rel", "rvol30"]
    for c in cols:
        F["k_" + c] = fm[c].values[pos]
    F["k_age"] = want - key[pos]     # 應該是 0（沒有缺 K）
    F["i"] = ii
    F["P"] = period(F.date)
    dcols = [c for c in D.columns if c.startswith("p_")] + ["gap", "on_rng", "on_ret", "wd", "month", "open_va", "open_vs_pdhl"] + \
            [c for c in D.columns if c.endswith("_10") or c.endswith("_1030")]
    F = F.merge(D[["date"] + dcols], on="date", how="left")
    return F, fm


if __name__ == "__main__":
    D = build()
    D.to_pickle(os.path.join(HERE, "days.pkl"))
    F, fm = cand(D)
    F.to_pickle(os.path.join(HERE, "cand.pkl"))
    fm.to_pickle(os.path.join(HERE, "fivemin.pkl"))
    print(D.shape, F.shape, (F.k_age > 0).mean())
    print(D[D.date == "2026-11-11"].T.to_string())
