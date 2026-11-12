"""組合（book）工具：重現 books17.py 的「順勢多點」，可加過濾遮罩或外加交易。研究用途，不是投資建議。
每日風控：今天已實現 − 2,948×(持倉數+1) < −3,000 就不下（同 books17）。外加交易用自己的最壞虧損。
"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
PER = ["探索", "驗證", "測試", "近期"]
F = pd.read_pickle(os.path.join(HERE, "cand.pkl"))
withI = F.to_vwap == False
PTS = {"I2": F.key.str.startswith("I2"),
       "前高空順": (F.key == "AM-PD-k1|空") & withI,
       "前低多順": (F.key == "AM-PD-k1|多") & withI,
       "亞洲多ES": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True),
       "亞洲VAL多ES": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),
       "美盤早夜盤低多ES": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2)}
BASE = list(PTS)
F["pt"] = None
for k, m in PTS.items():
    F.loc[m.fillna(False).values & F.pt.isna().values, "pt"] = k
WORST = 2948.0


def select(names=BASE, keep=None, guard=True):
    m = np.zeros(len(F), bool)
    for n in names:
        m |= PTS[n].fillna(False).values
    if keep is not None:
        m &= np.asarray(keep, bool)
    T = F[m].copy()
    if guard:
        T = T[~(T.shock >= 250)]
    T = T.sort_values(["date", "entry_slot"])
    a = T.key.str.startswith("ASIA") & (T.dir == 1)
    T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
    T["pnl"] = T.pnl55
    T["worst"] = WORST
    return T[["date", "P", "pt", "dir", "entry_slot", "exit_slot", "pnl", "worst"]]


def risk(T):
    """依時間順序套每日 $3,000 風控。T 需含 date, entry_slot, exit_slot, pnl, worst。"""
    T = T.sort_values(["date", "entry_slot"]).reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for d, idx in T.groupby("date", sort=False).groups.items():
        closed = []; opened = []
        for k in idx:
            t = T.loc[k]
            real = sum(p for xs, p in closed if xs <= t.entry_slot)
            openw = sum(w for xs, w in opened if xs > t.entry_slot)
            if real - openw - t.worst < -3000:
                continue
            keep[k] = True; closed.append((t.exit_slot, t.pnl)); opened.append((t.exit_slot, t.worst))
    return T[keep]


def book(names=BASE, keep=None, extra=None, drop_fn=None):
    T = select(names, keep)
    if extra is not None and len(extra):
        T = pd.concat([T, extra[["date", "P", "pt", "dir", "entry_slot", "exit_slot", "pnl", "worst"]]])
    if drop_fn is not None:
        T = drop_fn(T)
    return risk(T)


def dd(p):
    c = np.cumsum(p); return float((c - np.maximum.accumulate(np.concatenate([[0], c]))[1:]).min()) if len(c) else 0.0


def stats(T, label=""):
    rows = []
    for p in PER + ["全期"]:
        x = T if p == "全期" else T[T.P == p]
        dy = x.groupby("date").pnl.sum().sort_index()
        rows.append(dict(組合=label, 期間=p, 筆數=len(x), 勝率=round(100 * (x.pnl > 0).mean(), 1) if len(x) else np.nan,
                         每筆=round(x.pnl.mean(), 0) if len(x) else np.nan, 合計=round(x.pnl.sum(), 0),
                         最大回撤=round(dd(dy.values), 0), 最差一天=round(dy.min(), 0) if len(dy) else np.nan))
    r = pd.DataFrame(rows)
    full = r[r.期間 == "全期"].iloc[0]
    r["報酬回撤比"] = np.where(r.期間 == "全期", round(full.合計 / -full.最大回撤, 2) if full.最大回撤 < 0 else np.inf, np.nan)
    return r


def brief(T):
    """{期間: (合計, 回撤)}"""
    out = {}
    for p in PER + ["全期"]:
        x = T if p == "全期" else T[T.P == p]
        dy = x.groupby("date").pnl.sum().sort_index()
        out[p] = (x.pnl.sum(), dd(dy.values), len(x))
    return out


def md(r):
    cols = ["組合", "期間", "筆數", "勝率", "每筆", "合計", "最大回撤", "最差一天", "報酬回撤比"]
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, x in r.iterrows():
        s += "| " + " | ".join("" if (isinstance(x[c], float) and np.isnan(x[c])) else (f"{x[c]:,.0f}" if c in ("每筆", "合計", "最大回撤", "最差一天") else str(x[c])) for c in cols) + " |\n"
    return s
