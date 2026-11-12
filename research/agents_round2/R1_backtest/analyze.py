"""R1_backtest 分析：summary.csv、trades_<策略>.csv、REPORT 用的各段落（analysis.txt）。研究用途，不是投資建議。"""
import sys, json, os
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r1lib as R
import strat as S
import run as RU

OUT = R.OUT
A = pd.read_pickle(os.path.join(OUT, "all_combos.pkl"))
SIGS = pd.read_pickle(os.path.join(OUT, "all_sigs.pkl"))
SKIP = pd.read_pickle(os.path.join(OUT, "skip.pkl"))
ORDER = RU.SPEC["priority_order"]
SEGS = R.SEGS
ORDERED = {"S11": ("N", [5, 10, 20]), "S02": ("entry", ["top", "ce", "ote"])}
lines = []


def P(s=""):
    print(s); lines.append(str(s))


def segtot(T, seg):
    if T is None or len(T) == 0:
        return 0.0
    return float(T[T.seg == seg].pnl.sum())


BASE_M = RU.merged(None)
BASE_CT = RU.merged(None, with_ct=True)
BT = {s: segtot(BASE_M, s) for s in SEGS}
BT_CT = {s: segtot(BASE_CT, s) for s in SEGS}

# ------------------------------------------------------------------ 每組基本統計
rows = []
for (sid, name), v in A.items():
    T = v["T"]
    st = R.seg_table(T)
    ev = T[T.seg.isin(["探索", "驗證"])].pnl.values if len(T) else np.array([])
    t_ev, p_ev = RU.tstat(ev)
    tt = T[T.seg == "測試"].pnl.values if len(T) else np.array([])
    t_t, p_t = RU.tstat(tt)
    r = dict(sid=sid, name=name, **{k: str(x) for k, x in v["prm"].items()})
    for sg in SEGS + ["全期"]:
        for k in ("n", "win", "avg", "tot", "mdd", "worst"):
            r[f"{sg}_{k}"] = st[sg][k]
    a = st["全期"]
    r["全期_報酬回撤"] = a["tot"] / -a["mdd"] if a["mdd"] < 0 else np.inf
    r["t_EV"] = t_ev; r["p_EV"] = p_ev; r["t_測試"] = t_t; r["p_測試"] = p_t
    r["停損下限15筆數"] = int(T.floor15.sum()) if len(T) else 0
    r["停損>145不做"] = v["skips"].get(">145", 0)
    r["QC3改10:03"] = int(T.moved.sum()) if len(T) else 0
    r["QC3取消"] = v["skips"].get("QC3取消", 0)
    r["訊號數"] = v["nsig"]
    if len(T):
        r["同天同向基準%"] = 100 * T.base_same_day_dir.mean()
        r["同天同向C1TD10%"] = 100 * T.c1_td10_same_day_dir.mean()
        nb = T[~T.base_day]
        for sg in SEGS:
            r[f"無基準日_{sg}_n"] = int((nb.seg == sg).sum()); r[f"無基準日_{sg}_tot"] = float(nb[nb.seg == sg].pnl.sum())
        r["無基準日_EVT_tot"] = float(nb[nb.seg != "近期"].pnl.sum())
        M = RU.merged(T)
        for sg in SEGS:
            r[f"合併_{sg}_增量"] = segtot(M, sg) - BT[sg]
        r["合併_全期_回撤"] = R.dd(M.sort_values(["date", "e"]).pnl.values)
        r["合併_擋掉基準筆數"] = 520 - int((M.src == "base").sum())
        r["合併_策略成交筆數"] = int((M.src == "S").sum())
    rows.append(r)
SUM = pd.DataFrame(rows)
SUM["holm_EV"] = RU.holm(SUM.p_EV.fillna(1).values)


def get(sid, **kw):
    m = SUM.sid == sid
    for k, x in kw.items():
        m &= SUM[k] == str(x)
    return SUM[m]


# ------------------------------------------------------------------ 選參數（只用探索＋驗證）
SUM["合格EV"] = (SUM["探索_tot"] > 0) & (SUM["驗證_tot"] > 0) & ((SUM["探索_n"] + SUM["驗證_n"]) >= 15)
SUM["高原"] = "不適用"; SUM["鄰居"] = ""
# 高原：同策略、其他參數相同、有序參數相鄰值的 探索＋驗證 合計 > 0
PCOLS = {sid: list(A[next(kk for kk in A if kk[0] == sid)]["prm"].keys()) for sid in ORDER}
for k, r in SUM.iterrows():
    if r.sid not in ORDERED:
        continue
    pn, vals = ORDERED[r.sid]
    cur = [str(x) for x in vals]
    j = cur.index(r[pn])
    res = []
    for x in (j - 1, j + 1):
        if 0 <= x < len(vals):
            m = SUM.sid == r.sid
            for c in PCOLS[r.sid]:
                m &= SUM[c] == (cur[x] if c == pn else r[c])
            nr = SUM[m].iloc[0]
            res.append((cur[x], nr["探索_tot"] + nr["驗證_tot"], nr["測試_tot"]))
    SUM.at[k, "高原"] = "是" if all(e > 0 for _, e, _ in res) else "否"
    SUM.at[k, "鄰居"] = "; ".join(f"{a}: EV{e:+,.0f} 測{t:+,.0f}" for a, e, t in res)

SEL = {}
for sid in ORDER:
    c = SUM[(SUM.sid == sid) & SUM["合格EV"] & (SUM["高原"] != "否")]
    if len(c):
        SEL[sid] = c.sort_values("t_EV", ascending=False).iloc[0]["name"]; how = "合格"
    else:
        c = SUM[SUM.sid == sid]
        SEL[sid] = c.sort_values("t_EV", ascending=False).iloc[0]["name"]; how = "無合格組，取探索＋驗證 t 最高者僅供報告"
    SUM.loc[SUM.name == SEL[sid], "選中"] = how
SUM["選中"] = SUM["選中"].fillna("")

# ------------------------------------------------------------------ 選中組合的詳細
DET = {}
for sid in ORDER:
    name = SEL[sid]; v = A[(sid, name)]; T = v["T"]
    r = SUM[SUM.name == name].iloc[0]
    sp = v["sp"]; ex = v["ex"]
    sigs = SIGS[(sid, sp)]
    d = dict(name=name, prm=v["prm"], T=T, row=r)
    # 同一批訊號的另一種出場
    sib = [n for (s2, n), vv in A.items() if s2 == sid and vv["sp"] == sp and n != name]
    d["sib"] = [(n, R.seg_table(A[(sid, n)]["T"])) for n in sib]
    # 壓力
    Tc, _ = RU.run_combo(sid, sigs, ex, cost=96.0); d["cost2"] = R.seg_table(Tc)
    if all(z["kind"] == "mkt" for z in sigs):
        d["adv"] = R.seg_table(T, "pnl_adv"); d["adv_note"] = "市價單：進場改用該分鐘不利價（多 high／空 low）"
    else:
        fn = [f for spp, f in RU.sig_sets(sid, pen=2.0) if tuple(spp.items()) == sp][0]
        s2 = fn(); Ta, _ = RU.run_combo(sid, s2, ex)
        Ta["pnl_x"] = np.where(Ta.kind == "mkt", Ta.pnl_adv, Ta.pnl)
        d["adv"] = R.seg_table(Ta, "pnl_x"); d["adv_note"] = "限價單：要穿過掛單價 2 點才算成交（不能比掛單價差）；市價單用該分鐘不利價"
        # P 的 C9 近似（對照）
        f0 = [f for spp, f in RU.sig_sets(sid, pen=0.0) if tuple(spp.items()) == sp][0]
        s0 = []
        for z in f0():
            if z["kind"] == "lim":
                z = dict(z); z["kind"] = "mkt"; z["e0"] = z["e"] + 1
            s0.append(z)
        T9, _ = RU.run_combo(sid, s0, ex); d["c9"] = R.seg_table(T9)
    dy = T.groupby("date").pnl.sum().sort_values(ascending=False)
    d["top5"] = dy.head(5); d["no_top5"] = dy.iloc[5:].sum()
    d["no_top5_seg"] = {s: T[(T.seg == s) & ~T.date.isin(dy.head(5).index)].pnl.sum() for s in SEGS}
    # QC9 穩健性（只報告）
    rb = {}
    for nm, kw in [("停損緩衝+2", dict(stop_shift=2.0)), ("停損緩衝-2", dict(stop_shift=-2.0)), ("進場晚1分", dict(late=1))]:
        Tx, _ = RU.run_combo(sid, sigs, ex, **kw)
        rb[nm] = R.seg_table(Tx) if len(Tx) else None
    Ts = T.copy(); Ts["pnl"] = Ts.pnl - 4 * R.PT; rb["進出各滑2點"] = R.seg_table(Ts)
    d["robust"] = rb
    # 只在基準沒單的日子
    d["nobase"] = R.seg_table(T[~T.base_day])
    d["withbase"] = R.seg_table(T[T.base_day])
    # 合併
    M1 = RU.merged(T); M5 = RU.merged(T, size=0.5); MC = RU.merged(T, with_ct=True); MC5 = RU.merged(T, size=0.5, with_ct=True)
    d["merged"] = {"基準單獨": BASE_M, "基準＋本策略 1 口": M1, "基準＋本策略 0.5 口": M5,
                   "基準＋C1＋TD10": BASE_CT, "基準＋C1＋TD10＋本策略 1 口": MC, "基準＋C1＋TD10＋本策略 0.5 口": MC5}
    DET[sid] = d

# 測試期 Holm（8 個選中組合）
pt = np.array([DET[s]["row"]["p_測試"] for s in ORDER])
HT = dict(zip(ORDER, RU.holm(np.nan_to_num(pt, nan=1.0))))


# ------------------------------------------------------------------ 分級
def pos3(st):
    return all(st[s]["tot"] > 0 for s in ("探索", "驗證", "測試"))


GR = {}
for sid in ORDER:
    d = DET[sid]; r = d["row"]
    st = R.seg_table(d["T"])
    c = {}
    c["A 三段都賺"] = pos3(st)
    c["B 高原"] = r["高原"] if r["高原"] != "否" else "否"
    c["C 結構或B55通過"] = c["A 三段都賺"] or any(pos3(s) for _, s in d["sib"])
    nb = d["nobase"]
    c["D 無基準日賺(探索+驗證+測試)"] = sum(nb[s]["tot"] for s in ("探索", "驗證", "測試")) > 0
    M1 = d["merged"]["基準＋本策略 1 口"]
    c["E 合併測試期不少於基準"] = segtot(M1, "測試") >= BT["測試"]
    rob = all(x is None or all(x[s]["tot"] > 0 for s in ("探索", "驗證", "測試")) for x in d["robust"].values()) and \
        all(d["cost2"][s]["tot"] > 0 for s in ("探索", "驗證", "測試"))
    c["F 穩健(成本×2、±2停損、晚1分、滑2點)"] = rob
    c["G Holm(探索+驗證)<0.05"] = r["holm_EV"] < 0.05
    core = c["A 三段都賺"] and c["B 高原"] != "否" and c["C 結構或B55通過"] and c["D 無基準日賺(探索+驗證+測試)"] and c["E 合併測試期不少於基準"]
    if not c["A 三段都賺"] or r["選中"] != "合格":
        g = "否決"
    elif core and rob and c["G Holm(探索+驗證)<0.05"]:
        g = "通過（可小量實盤）"
    else:
        g = "只紙上記錄"
    GR[sid] = (g, c)

# ------------------------------------------------------------------ 輸出 CSV
SUM.to_csv(os.path.join(OUT, "summary.csv"), index=False, encoding="utf-8-sig")
for sid in ORDER:
    parts = []
    for (s2, name), v in A.items():
        if s2 != sid or len(v["T"]) == 0:
            continue
        T = v["T"].copy(); T.insert(0, "combo", name); T.insert(1, "選中", name == SEL[sid])
        parts.append(T)
    X = pd.concat(parts, ignore_index=True)
    X["entry_time"] = X.e.map(R.hhmm); X["exit_time"] = X.x.map(R.hhmm)
    X["dir"] = np.where(X.d == 1, "多", "空")
    cols = ["combo", "選中", "date", "seg", "dir", "kind", "sig", "entry_time", "px", "stop", "stop_struct", "sl", "exit_time", "xp", "why", "pts", "pnl",
            "worst", "pnl_adv", "moved", "floor15", "base_same_day_dir", "c1_td10_same_day_dir", "base_day", "known", "info"]
    X[cols].to_csv(os.path.join(OUT, f"trades_{sid}.csv"), index=False, encoding="utf-8-sig")

pd.to_pickle(dict(SUM=SUM, SEL=SEL, DET=DET, GR=GR, HT=HT, BT=BT, BT_CT=BT_CT), os.path.join(OUT, "analysis.pkl"))
print("done")
