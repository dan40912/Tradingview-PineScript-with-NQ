"""(3) 用判斷器分配 ＋ (4) walk-forward 模型。研究用途，不是投資建議。
判斷器（參數只用探索／驗證挑）：TD10 = a3_trend.td_signal("10", q, k)。
組合：
  基準           ＝ 順勢多點（books17）
  A 基準＋TD單    ＝ 趨勢日 10:00 順勢進場、停損 SL、15:55 出場（最壞虧損 SL×20+48 算進每日風控）
  B A＋砍逆勢     ＝ 趨勢日把 10:00 以後、方向和趨勢相反的基準單拿掉
  C 基準＋T1~T3   ＝ 趨勢日只做和趨勢同向、10:00 以後的 T1/T2/T3（trend.py，+55/−145，12:29 出場）
  D B＋T1~T3
WF 模型：邏輯迴歸（L2）／決策樹（深度 2），10:00 特徵，目標＝10:00→15:55 漲跌；
每月月初只用之前的資料重訓（擴張視窗，至少 120 天）；|p−0.5|>δ 才交易，δ 只用探索段的 WF 結果挑。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/15bba16f-f33f-5548-9c6a-1eb415eb2457/scratchpad/wt/research"
sys.path.insert(0, RES)
import numpy as np, pandas as pd
import a3_trend as A
import cbook as B
from lib import sim
g, D = A.g, A.D
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)

# ---------- T1~T3（trend.py 的訊號函式，原封不動執行，不寫檔）----------
src = open(os.path.join(RES, "trend.py")).read().split("S = signals()")[0]
G = {"__file__": os.path.join(RES, "trend.py"), "__name__": "trend_src"}
exec(src, G)
TS = pd.DataFrame(G["signals"]())
TS["date"] = g.dates[TS.i.values]
TT = sim.run_signals(g, TS.to_dict("records"), tp_pts=55.0, sl_pts=145.0)
TT["P"] = A.period(TT.date)


def td_trades(dv, sl=100.0, tag="10"):
    T = A.run_direct(dv, tag, sl)
    T["P"] = A.period(T.date); T["pt"] = "TD順勢"; T["worst"] = sl * 20 + 48
    return T


def tsel(dv, rules=("T1", "T2", "T3"), after=960):
    m = TT.rule.isin(rules) & (TT.entry_slot >= after)
    dd = dv[TT.i.values]
    T = TT[m & (dd != 0) & (TT.dir == dd)].copy()
    T["pt"] = "T" ; T["worst"] = 2948.0
    return T


def drop_counter(dv, after=960):
    dmap = dict(zip(g.dates, dv))
    def f(T):
        d = T.date.map(dmap).fillna(0).values
        bad = (T.pt != "TD順勢") & (T.pt != "T") & (T.entry_slot >= after) & (d != 0) & (T.dir.values == -d)
        return T[~bad]
    return f


def all_books(dv, sl=100.0):
    TD = td_trades(dv, sl)
    Tx = tsel(dv)
    return {
        "基準（順勢多點）": B.book(),
        "A 基準＋TD單": B.book(extra=TD),
        "B 基準＋TD單＋砍逆勢": B.book(extra=TD, drop_fn=drop_counter(dv)),
        "C 基準＋趨勢日T1~T3": B.book(extra=Tx),
        "D 基準＋TD單＋T1~T3＋砍逆勢": B.book(extra=pd.concat([TD, Tx]), drop_fn=drop_counter(dv)),
        "E 只做TD單": B.risk(TD[["date", "P", "pt", "dir", "entry_slot", "exit_slot", "pnl", "worst"]]),
    }


# ---------- WF 模型 ----------
def wf_model(kind="logit", feats=None, min_train=120):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    X = D[feats].astype(float)
    y = D.y_10.values
    ok = A.OK & np.isfinite(y) & X.notna().all(1).values
    month = D.date.str[:7].values
    prob = np.full(len(D), np.nan)
    for mth in sorted(set(month[ok])):
        test = ok & (month == mth)
        train = ok & (D.date.values < mth + "-01")
        if train.sum() < min_train:
            continue
        Xt, yt = X.values[train], (y[train] > 0).astype(int)
        if kind == "logit":
            sc = StandardScaler().fit(Xt)
            m = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(Xt), yt)
            prob[test] = m.predict_proba(sc.transform(X.values[test]))[:, 1]
        else:
            m = DecisionTreeClassifier(max_depth=2, min_samples_leaf=30, random_state=0).fit(Xt, yt)
            prob[test] = m.predict_proba(X.values[test])[:, 1]
    return prob


if __name__ == "__main__":
    out = []
    # ---- T1~T3 本身（不分局勢 vs 趨勢日同向）----
    print("== T1~T3 本身，每筆（探索/驗證/測試/近期）")
    for q, k in [(0.67, 3)]:
        dv = A.td_signal("10", q, k)
        for rule in ["T1", "T2", "T3"]:
            x = TT[(TT.rule == rule) & (TT.entry_slot >= 960)]
            dd = dv[x.i.values]
            for nm, mm in [("全部", np.ones(len(x), bool)), ("趨勢日同向", (dd != 0) & (x.dir.values == dd)),
                           ("趨勢日逆向", (dd != 0) & (x.dir.values == -dd)), ("非趨勢日", dd == 0)]:
                y = x[mm]
                print(f"  {rule} {nm:6s}", " ".join(f"{p}:{(y.P == p).sum():3d}筆 {y[y.P == p].pnl.mean():+6.0f}" for p in B.PER))
    # ---- 組合：判斷器參數格 ----
    rows = []
    for q in [0.5, 0.67, 0.8]:
        for k in [2, 3, 4]:
            dv = A.td_signal("10", q, k)
            for sl in [100.0, 145.0]:
                for name, T in all_books(dv, sl).items():
                    b = B.brief(T)
                    r = dict(q=q, k=k, sl=sl, book=name)
                    for p in B.PER + ["全期"]:
                        r[f"{p}_tot"] = round(b[p][0]); r[f"{p}_dd"] = round(b[p][1]); r[f"{p}_n"] = b[p][2]
                    rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(HERE, "out_a4_books.csv"), index=False)
    print("\n== 組合（參數格）")
    print(R[["q", "k", "sl", "book"] + [f"{p}_{m}" for p in B.PER + ["全期"] for m in ["tot", "dd"]]].to_string(index=False))
    # ---- WF 模型 ----
    F10 = [c for c in D.columns if c.endswith("_10") and not c.startswith("y_")] + \
          ["gap", "on_rng", "on_ret", "p_eff", "p_clv", "p_dret", "p_rsi_d", "p_macdh_d", "p_emax_d", "p_ret5_d", "p_ret20_d", "p_adx_d", "p_di_d", "p_atr_rel", "open_va", "wd"]
    print("\n== WF 模型（10:00 決策，SL100，15:55 出場）特徵數", len(F10))
    wrows = []
    for kind, feats in [("logit", F10), ("tree", F10), ("logit", [f"{v}_10" for v in A.VOTE])]:
        p = wf_model(kind, feats)
        D["prob"] = p
        for dlt in [0.0, 0.05, 0.1, 0.15]:
            dv = np.where(np.isfinite(p) & (np.abs(p - 0.5) > dlt), np.sign(p - 0.5), 0).astype(int)
            T = td_trades(dv, 100.0)
            yv = D.y_10.values
            r = dict(model=kind, nfeat=len(feats), delta=dlt)
            for per in ["探索", "驗證", "測試", "近期"]:
                m = (D.P == per).values & (dv != 0) & np.isfinite(yv)
                r[f"{per}_acc"] = round(100 * (np.sign(yv[m]) == dv[m]).mean(), 1) if m.sum() else np.nan
                x = T[T.P == per]
                r[f"{per}_n"] = len(x); r[f"{per}_avg"] = round(x.pnl.mean()) if len(x) else np.nan
            wrows.append(r)
    W = pd.DataFrame(wrows); W.to_csv(os.path.join(HERE, "out_a4_wf.csv"), index=False)
    print(W.to_string(index=False))
