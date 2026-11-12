"""R1_backtest 主程式：76 組全部回測、挑參數（只用探索＋驗證）、Holm、合併、壓力、11/11。研究用途，不是投資建議。"""
from __future__ import annotations
import os, sys, json, itertools, random
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy import stats as sst
import r1lib as R
import strat as S

OUT = R.OUT
sys.path.insert(0, os.path.join(R.AG, "B_exits"))
import core as Bc  # noqa: E402  只用來重現基準 520 筆（已驗證）

# ------------------------------------------------------------------ 基準、C1、TD10
BR = Bc.simulate(Bc.plan())
BASE_BOOK = Bc.book(BR)
_bs = Bc.stats(BASE_BOOK)["全期"]
assert len(BASE_BOOK) == 520 and abs(_bs["tot"] - 175110) < 1 and abs(_bs["mdd"] + 11851) < 1, _bs
BASEC = pd.DataFrame(dict(date=BR.date, d=BR.dir, e=BR.entry_slot, x=BR.xslot, pnl=BR.pnl1, worst=BR.worst1, src="base", pri=0))
C1 = pd.read_csv(os.path.join(R.AG, "RT_redteam", "rt_trades_C1.csv")).assign(src="C1", pri=1)
TD = pd.read_csv(os.path.join(R.AG, "RT_redteam", "rt_trades_TD10.csv")).assign(src="TD10", pri=2)
BASE_DAYDIR = set(zip(BASE_BOOK.date, BASE_BOOK.dir))
BASE_DAYS = set(BASE_BOOK.date)
CT_DAYDIR = set(zip(C1.date, C1.d)) | set(zip(TD.date, TD.d))


def book(T, lim=3000.0):
    """B_exits.book 同語意（Pine 平倉即記帳）：已實現＝出場 slot < 本筆進場 slot；持倉最壞＝出場 slot >= 本筆進場。
    排序：日期、進場 slot、pri（基準 0 ＞ C1 1 ＞ TD10 2 ＞ 本策略 3）。"""
    T = T.sort_values(["date", "e", "pri"], kind="stable").reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for d, idx in T.groupby("date", sort=False).groups.items():
        hist = []
        for k in idx:
            e = T.at[k, "e"]; w = T.at[k, "worst"]
            if T.at[k, "x"] < 0:
                continue
            real = sum(p for x, ww, p in hist if x < e)
            still = sum(ww for x, ww, p in hist if x >= e)
            if real - still - w < -lim:
                continue
            keep[k] = True
            hist.append((T.at[k, "x"], w, T.at[k, "pnl"]))
    return T[keep].copy()


_chk = book(BASEC)
assert len(_chk) == 520 and abs(_chk.pnl.sum() - 175110) < 1, (len(_chk), _chk.pnl.sum())
_chk["seg"] = [R.seg_of(d) for d in _chk.date]
BASE_SEG = R.seg_table(_chk)

# ------------------------------------------------------------------ 組合定義
SPEC = json.load(open(os.path.join(R.AG, "Q1_critic", "strategies_v2.json")))
STRUCT = {"S11": ["2R"], "S06": ["2R"], "S03": ["2R", "EOD"], "S02": ["2R", "pool"], "S10": ["range"], "S01": ["2R"],
          "S07": ["range"], "S09": ["2R"]}


def sig_sets(sid, pen=R.PEN):
    """回傳 [(訊號參數 dict, 產生器呼叫)]。"""
    if sid == "S11":
        return [({"N": n, "es": e}, lambda n=n, e=e: S.gen_S11(n, e == "on")) for n in (5, 10, 20) for e in ("off", "on")]
    if sid == "S06":
        return [({"breaker": b, "window": w}, lambda b=b, w=w: S.gen_S06(b, w)) for b in ("NQ", "ES") for w in ("AM", "DAY")]
    if sid == "S03":
        return [({"anchor": a}, lambda a=a: S.gen_S03(a)) for a in ("mid", "0830")]
    if sid == "S02":
        return [({"entry": en, "es": e}, lambda en=en, e=e: S.gen_S02(en, "pool", e == "on", pen)) for en in ("top", "ce", "ote") for e in ("off", "on")]
    if sid == "S10":
        return [({"confirm": c, "vwap": v}, lambda c=c, v=v: S.gen_S10(c, v)) for c in ("now", "test") for v in ("off", "on")]
    if sid == "S01":
        return [({"window": w, "filter": f}, lambda w=w, f=f: S.gen_S01(w, f, pen)) for w in ("AM", "PM") for f in ("any", "sweep", "vwap")]
    if sid == "S07":
        return [({"vwap": v}, lambda v=v: S.gen_S07(v, pen)) for v in ("off", "on")]
    if sid == "S09":
        return [({"level_type": lt, "entry": en}, lambda lt=lt, en=en: S.gen_S09(lt, en, pen)) for lt in ("AV", "GAP") for en in ("limit", "reject")]


def exit_names(sid):
    return STRUCT[sid] + ["B55"]


def exit_param_name(sid):
    return {"S02": "target", "S07": "target"}.get(sid, "exit")


# ------------------------------------------------------------------ 實現（出場）
def realize(sid, sg, ex, cost=R.COST, stop_shift=0.0, late=0):
    i, d = sg["i"], sg["d"]
    stop_raw = sg["stop_raw"] - d * stop_shift      # stop_shift>0 → 停損放寬
    tg = sg.get("tgt")
    xs_struct = sg.get("xs_struct", R.EOD)
    if sg["kind"] == "mkt":
        m, why = R.mkt_signal(i, d, sg["e0"] + late, stop_raw, tgt=tg)
        if m is None:
            return None, why
        e, px = m["e"], m["px"]
        if ex == "B55":
            r = R.sim(i, d, e, px - 145 * d, px + 55 * d, R.b55_xs(e), cost=cost); w = 145
        elif ex == "EOD":
            r = R.sim(i, d, e, m["stop"], None, R.EOD, be_R=m["sl"], cost=cost); w = m["sl"]
        elif ex in ("range", "pool"):
            r = R.sim(i, d, e, m["stop"], m["tp"], R.EOD, cost=cost); w = m["sl"]
        else:
            r = R.sim(i, d, e, m["stop"], px + 2 * d * m["sl"], xs_struct, cost=cost); w = m["sl"]
        r.update(sl=145.0 if ex == "B55" else m["sl"], stop=px - 145 * d if ex == "B55" else m["stop"], stop_struct=m["stop"],
                 moved=m["moved"], floor15=m["floor15"])
    else:
        if late:
            return None, "限價不適用"
        P = sg["P"]; e = sg["e"]
        q = R.qc2(P, stop_raw, d)
        if q is None:
            return None, ">145"
        stop, sl = q
        if ex == "B55":
            fpx = min(P, R.O[i, e]) if d == 1 else max(P, R.O[i, e])
            r = R.sim(i, d, e, fpx - 145 * d, fpx + 55 * d, R.b55_xs(e), limit_P=P, cost=cost); w = 145
        else:
            tp = R.tgt_px(P, stop, d, tg if ex in ("range", "pool") else None)
            r = R.sim(i, d, e, stop, tp, xs_struct, limit_P=P, cost=cost); w = sl
        fpx = min(P, R.O[i, e]) if d == 1 else max(P, R.O[i, e])
        r.update(sl=145.0 if ex == "B55" else sl, stop=fpx - 145 * d if ex == "B55" else stop, stop_struct=stop,
                 moved=False, floor15=d * (P - stop_raw) < 15)
    r["worst"] = w * R.PT + cost
    # 壓力：該分鐘不利價（市價單）
    if sg["kind"] == "mkt":
        adv = R.H[i, r["e"]] if d == 1 else R.L[i, r["e"]]
        r["pnl_adv"] = r["pnl"] - d * (adv - r["px"]) * R.PT
    else:
        r["pnl_adv"] = np.nan
    return r, "ok"


def run_combo(sid, sigs, ex, **kw):
    rows = []; skips = {}
    for sg in sigs:
        r, why = realize(sid, sg, ex, **kw)
        if r is None:
            skips[why] = skips.get(why, 0) + 1
            continue
        dt = R.DATES[sg["i"]]
        r.update(date=dt, i=sg["i"], seg=R.SEG[sg["i"]], d=sg["d"], kind=sg["kind"], sig=R.hhmm(sg["sig"]),
                 known=sg["known"], info=json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                                                          [float(z) for z in v] if isinstance(v, tuple) else v)
                                                      for k, v in sg["info"].items()}, ensure_ascii=False),
                 base_same_day_dir=(dt, sg["d"]) in BASE_DAYDIR, c1_td10_same_day_dir=(dt, sg["d"]) in CT_DAYDIR,
                 base_day=dt in BASE_DAYS)
        rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T = T[T.seg != "無"].reset_index(drop=True)
        T["c"] = 1.0
    return T, skips


def merged(T, size=1.0, with_ct=False):
    X = [BASEC]
    if with_ct:
        X += [C1[["date", "d", "e", "x", "pnl", "worst", "src", "pri"]], TD[["date", "d", "e", "x", "pnl", "worst", "src", "pri"]]]
    if T is not None and len(T):
        X.append(pd.DataFrame(dict(date=T.date, d=T.d, e=T.e, x=T.x, pnl=T.pnl * size, worst=T.worst * size, src="S", pri=3)))
    B = book(pd.concat(X, ignore_index=True))
    B["seg"] = [R.seg_of(d) for d in B.date]
    return B


def tstat(p):
    p = np.asarray(p, float)
    if len(p) < 3 or p.std(ddof=1) == 0:
        return np.nan, 1.0
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    return t, 1 - sst.t.cdf(t, len(p) - 1)


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, k in enumerate(order):
        run = max(run, (m - r) * p[k]); adj[k] = min(1.0, run)
    return adj


def top5_removed(T, col="pnl"):
    dy = T.groupby("date")[col].sum().sort_values(ascending=False)
    return dy.iloc[5:].sum()


# ------------------------------------------------------------------ 主流程
if __name__ == "__main__":
    log = []
    def P(s=""):
        print(s); log.append(str(s))

    ORDER = SPEC["priority_order"]
    ALL = {}          # (sid, combo_name) -> dict
    SIGS = {}
    for sid in ORDER:
        for sp, fn in sig_sets(sid):
            sigs = fn()
            SIGS[(sid, tuple(sp.items()))] = sigs
            for ex in exit_names(sid):
                T, sk = run_combo(sid, sigs, ex)
                prm = dict(sp); prm[exit_param_name(sid)] = ex
                name = sid + "_" + "_".join(f"{v}" for v in prm.values())
                ALL[(sid, name)] = dict(sid=sid, name=name, prm=prm, sp=tuple(sp.items()), ex=ex, T=T, skips=sk, nsig=len(sigs))
        P(f"{sid} 完成")
    pd.to_pickle({k: {kk: vv for kk, vv in v.items()} for k, v in ALL.items()}, os.path.join(OUT, "all_combos.pkl"))
    pd.to_pickle(SIGS, os.path.join(OUT, "all_sigs.pkl"))
    pd.to_pickle(S.SKIP, os.path.join(OUT, "skip.pkl"))
