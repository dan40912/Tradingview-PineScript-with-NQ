"""R2_backtest 主程式：依 priority 逐一回測 Q03v2、Q13v2、Q02v2、Q08v2、EVT_FOMC。研究用途，不是投資建議。

事先登記（看任何結果之前寫好）：
  挑選：每個策略在探索期、單獨 1 口（含每日 $3,000 檢查）跑完整網格；
        『高原』＝探索期合計 > 0 且數值軸上所有相鄰參數（同一組類別參數下）探索期合計也 > 0；
        類別參數（Q03 的 trail/start、Q13 的 decision_time、Q08 的 entry_time）各自成一條線，不互當鄰居（只報告另一類別的結果）。
        在符合高原的組合中挑探索期 t 值（每筆平均／標準誤）最高的一組；沒有任何組合符合 → 標『無高原』，仍以 t 值最高者報告。
  多重檢定：5 個假設（Q03v2、Q13v2、Q02v2、Q08v2、EVT_FOMC 規則 A）用驗證期單尾 p 值做 Holm；
        非事件型用 t 檢定，事件型（Q08v2、EVT）用排列檢定（同分割隨機抽同天數非事件日，10,000 次）。
  分級：
    通過（可小量實盤）＝高原 ＋ 探索、驗證、測試三段合計皆 > 0 ＋ Holm 校正後驗證 p < 0.10
                       ＋ 成本加倍與不利進場價兩種壓力下全期合計仍 > 0
                       ＋ 0.5 口併入基準後全期合計更高且最大回撤不變或更小（絕對值 ≤ 基準）
                       ＋（Q08v2 另需：探索期排列 p < 0.10，且探索、驗證都贏非數據日對照）。
    只紙上記錄＝三段皆 > 0 但未達上面其他條件；或 EVT_FOMC（規格規定只紙上記錄）。
    否決＝三段中任一段 ≤ 0，或無高原。
  組合：基準＝B_exits 的 +55／−145、平倉即記帳（重現 520 筆 +175,110 −11,851）；策略的『原始訊號』（未經單獨記帳）
        以 1 口或 0.5 口加入，所有單依 (日期, 進場 slot, 優先序) 排序後一筆一筆過每日 $3,000 檢查（先到先做；
        同一分鐘基準＞Q03v2＞Q13v2＞Q02v2＞Q08v2＞EVT）；不同策略可以同時持倉（含反向，另報筆數）。
        Q03v2 的第二筆是在『第一筆有做』的前提下產生的；組合中若第一筆被擋，第二筆仍照原樣當候選（近似，寫在報告）。
"""
import sys
sys.dont_write_bytecode = True
import os, json, itertools, time
import numpy as np, pandas as pd
import r2lib as R
import strats as S

t0 = time.time()
OUT = R.OUT
LOG = []


def P(s=""):
    print(s); LOG.append(str(s))


# ------------------------------------------------------------------ 基準
import b_core as B
_bc = B.simulate(B.plan())
BASE = pd.DataFrame(dict(date=_bc.date.values, e=_bc.entry_slot.values, x=_bc.xslot.values, pnl=_bc.pnl1.values,
                         worst=_bc.worst1.values, d=_bc.dir.values, src="base", prio=0))
BASE = BASE[BASE.x >= 0].copy()
BASE["seg"] = [R.seg_of(d) for d in BASE.date]
BASE_B = R.book(BASE)
BASE_ST = R.seg_stats(BASE_B)
P("基準（B_exits 重現、平倉即記帳）：" + R.oneline(BASE_ST))

# ------------------------------------------------------------------ C1、TD10 當天（RT 獨立實作）
import rt_core as RT
_c1 = {RT.DATES[s["i"]]: s["d"] for s in RT.c1_signals()}
_F = RT.td_features(960); _dv, _ = RT.td_signal(_F, 0.8, 3)
_td = {RT.DATES[i]: int(_dv[i]) for i in np.where(_dv != 0)[0]}
LAB = R.group_labels()


def overlap_report(T):
    x = T.copy()
    x["c1"] = x.date.map(_c1); x["td"] = x.date.map(_td)
    has = x.c1.notna() | x.td.notna()
    same = ((x.c1 == x.d) | (x.td == x.d))
    s = (f"    和 C1/TD10 同日 {int(has.sum())} 筆（方向至少和一個一致 {int((has & same).sum())} 筆）："
         f"同日每筆 {x[has].pnl.mean():+.0f}（{int(has.sum())}筆）／不同日每筆 {x[~has].pnl.mean():+.0f}（{int((~has).sum())}筆）")
    for sg in R.SEGS:
        y = x[x.seg == sg]; h = has[x.seg == sg]
        s += f"\n      {sg}: 同日 {int(h.sum())}筆 {y[h].pnl.mean() if h.any() else np.nan:+.0f} ／ 不同日 {int((~h).sum())}筆 {y[~h].pnl.mean() if (~h).any() else np.nan:+.0f}"
    return s


def combo(T, prio, size):
    """把策略原始訊號（size 口）加進基準，一起過每日 $3,000 檢查。"""
    X = T.copy()
    X["pnl"] = X.pnl * size; X["worst"] = X.worst * size; X["src"] = "new"; X["prio"] = prio
    A = pd.concat([BASE, X[["date", "e", "x", "pnl", "worst", "d", "src", "prio", "seg"]]], ignore_index=True)
    Bk = R.book(A)
    # 反向同時持倉筆數
    nopp = 0
    for d, y in Bk.groupby("date"):
        y = y.sort_values("e")
        for a in range(len(y)):
            for b in range(a):
                if y.iloc[b].x >= y.iloc[a].e and y.iloc[b].d != y.iloc[a].d:
                    nopp += 1
    return Bk, nopp


def combo_report(T, prio, name):
    lines = []
    res = {}
    for size in [1.0, 0.5]:
        Bk, nopp = combo(T, prio, size)
        st = R.seg_stats(Bk)
        nn = int((Bk.src == "new").sum()); nb = int((Bk.src == "base").sum())
        dtot = st["全期"]["tot"] - BASE_ST["全期"]["tot"]; ddd = st["全期"]["mdd"] - BASE_ST["全期"]["mdd"]
        better = (dtot > 0) and (st["全期"]["mdd"] >= BASE_ST["全期"]["mdd"] - 1e-6)
        lines.append(f"  基準＋{name} {size}口：策略成交 {nn} 筆、基準 {nb} 筆（基準原 {len(BASE_B)}）；反向同時持倉 {nopp} 筆\n    " + R.oneline(st) +
                     f"\n    相對基準：全期合計 {dtot:+,.0f}，最大回撤 {ddd:+,.0f}（正＝變小），報酬/回撤 {BASE_ST['全期']['rdd']:.2f}→{st['全期']['rdd']:.2f}"
                     f"；各段合計差 " + " ".join(f"{s} {st[s]['tot'] - BASE_ST[s]['tot']:+,.0f}" for s in R.SEGS) +
                     f"；最差一天 {st['全期']['worst']:+,.0f}；判定：{'獲利更高且回撤不變或更小' if better else '未達（獲利沒更高或回撤變大）'}")
        res[size] = dict(st=st, better=better, dtot=dtot, ddd=ddd, Bk=Bk)
    return "\n".join(lines), res


def seg_line(T):
    return R.oneline(R.seg_stats(T))


def check_signals(T, name, cols, k=5, seed=1):
    bad = T[T.last_data_slot >= T.e]
    P(f"  訊號正確性：全部 {len(T)} 筆中『用到的最後一根 K ≥ 進場 slot』的筆數 = {len(bad)}（應為 0）")
    smp = T.sample(min(k, len(T)), random_state=seed)
    for _, r in smp.iterrows():
        vals = "，".join(f"{c.replace('chk_', '')}={r[c]:.4f}" if isinstance(r[c], float) else f"{c.replace('chk_', '')}={r[c]}" for c in cols if c in r)
        P(f"    {r.date} {'多' if r.d == 1 else '空'} 進場 {r.et}（slot {r.e}）@{r.entry:.2f}；用到最後一根 K {R.hhmm(r.last_data_slot)}（slot {r.last_data_slot}）；{vals}；"
          f"出場 {r.xt} @{r.exit:.2f}（{r.why}）損益 {r.pnl:+.0f}")
    assert len(bad) == 0


def stress(gen, params, name):
    out = {}
    out["原版"] = R.book(gen(**params))
    out["成本加倍 $96"] = R.book(gen(**params, cost=96.0))
    out["進場用不利價（多 high／空 low）"] = R.book(gen(**params, entry="adverse"))
    out["進出各差 2 點＋$48"] = R.book(gen(**params, slip=2.0))
    out["晚 2 分鐘進場"] = R.book(gen(**params, late=2))
    base = out["原版"]
    dr, top = R.drop_top_days(base, 5)
    out["拿掉最大 5 天"] = dr
    lines = []
    for k, T in out.items():
        lines.append(f"    {k:22s} " + seg_line(T))
    lines.append(f"    （最大 5 天：{', '.join(top)}）")
    return out, "\n".join(lines)


def run_grid(name, gen, grid, num_axes, cat_axes):
    """回傳 DataFrame（每個組合的四段結果）與 dict key→booked trades。"""
    rows = []; keep = {}
    keys = list(grid)
    for combo_ in itertools.product(*[grid[k] for k in keys]):
        prm = dict(zip(keys, combo_))
        Tb = R.book(gen(**prm))
        st = R.seg_stats(Tb)
        ex = Tb[Tb.seg == "探索"].pnl.values
        t, p = R.tstat_p(ex)
        r = dict(strategy=name, params=json.dumps(prm, ensure_ascii=False), t_explore=t)
        for s in R.SEGS + ["全期"]:
            for kk in ["n", "win", "avg", "tot", "mdd", "worst"]:
                r[f"{s}_{kk}"] = st[s][kk]
        r["全期_rdd"] = st["全期"]["rdd"]
        rows.append(r); keep[r["params"]] = (prm, Tb)
    D = pd.DataFrame(rows)
    # 高原
    pl = []
    for _, r in D.iterrows():
        prm = json.loads(r.params)
        ok = r["探索_tot"] > 0
        for ax in num_axes:
            vals = grid[ax]; j = vals.index(prm[ax])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(vals):
                    q = dict(prm); q[ax] = vals[jj]
                    rr = D[D.params == json.dumps(q, ensure_ascii=False)].iloc[0]
                    ok = ok and rr["探索_tot"] > 0
        pl.append(ok)
    D["plateau"] = pl
    return D, keep


def show_grid(D):
    P("  探索期網格（單獨 1 口，含每日檢查）：參數 | 探索 筆/每筆/合計 | t | 高原")
    for _, r in D.iterrows():
        P(f"    {r.params:55s} | {int(r['探索_n']):4d} {r['探索_avg']:+6.0f} {r['探索_tot']:+9,.0f} | t={r.t_explore:+.2f} | {'高原' if r.plateau else '—'}")


def choose(D):
    c = D[D.plateau]
    if len(c):
        return c.sort_values("t_explore", ascending=False).iloc[0], True
    return D.sort_values("t_explore", ascending=False).iloc[0], False


def day1111(name, keep, chosen_params):
    P(f"  2026-11-11（所有組合；★＝選中）：")
    for k, (prm, Tb) in keep.items():
        x = Tb[Tb.date == "2026-11-11"]
        star = "★" if k == chosen_params else " "
        if len(x) == 0:
            P(f"   {star}{k}: 不交易")
        for _, r in x.iterrows():
            P(f"   {star}{k}: {'做多' if r.d == 1 else '做空'} {r.et} @{r.entry:,.2f} → {r.xt} @{r.exit:,.2f}（{ {'sl': '停損', 'cond': '追蹤出場', 'time': '時間出場'}[r.why] }）{r.pts:+.2f} 點 {r.pnl:+,.0f}")


SUMMARY = []
RESULTS = {}


def summarize_row(name, label, T, extra=None):
    st = R.seg_stats(T)
    r = dict(strategy=name, variant=label)
    for s in R.SEGS + ["全期"]:
        for kk in ["n", "win", "avg", "tot", "mdd", "worst"]:
            r[f"{s}_{kk}"] = round(st[s][kk], 2) if isinstance(st[s][kk], float) else st[s][kk]
    r["全期_rdd"] = round(st["全期"]["rdd"], 2)
    if extra:
        r.update(extra)
    SUMMARY.append(r)


def full_strategy(name, prio, gen, grid, num_axes, cat_axes, chk_cols, controls=None, event=None):
    P("\n" + "=" * 100 + f"\n{name}\n" + "=" * 100)
    D, keep = run_grid(name, gen, grid, num_axes, cat_axes)
    show_grid(D)
    for _, r in D.iterrows():
        summarize_row(name, "網格 " + r.params, keep[r.params][1], dict(plateau=r.plateau, t_explore=round(r.t_explore, 2)))
    ch, has_pl = choose(D)
    prm = json.loads(ch.params)
    P(f"  ★ 選中：{ch.params}（{'高原' if has_pl else '無高原，僅報告 t 最高者'}；探索 t={ch.t_explore:.2f}）")
    T = keep[ch.params][1]
    P("  （選定之後才看）全部組合的四段合計／每筆，只報告：")
    for _, r in D.iterrows():
        P(f"    {r.params:55s} " + " ".join(f"{s} {int(r[s + '_n'])}筆 {r[s + '_avg']:+.0f}/{r[s + '_tot']:+,.0f}" for s in R.SEGS))
    raw = gen(**prm)
    P("  單獨 1 口（含每日 $3,000 檢查）四段：")
    P(R.fmt_seg(R.seg_stats(T)))
    for dd_, nm in [(1, "多"), (-1, "空")]:
        x = T[T.d == dd_]
        if len(x):
            P(f"    只看{nm}單：" + seg_line(x))
    tv, pv = R.tstat_p(T[T.seg == "驗證"].pnl.values)
    tt, pt = R.tstat_p(T[T.seg == "測試"].pnl.values)
    P(f"  驗證期單尾 t={tv:.2f} p={pv:.4f}；（參考）測試期 t={tt:.2f} p={pt:.4f}")
    check_signals(raw, name, chk_cols)
    stv, stxt = stress(gen, prm, name)
    P("  壓力測試（不可用來挑參數）：\n" + stxt)
    for k, X in stv.items():
        summarize_row(name, f"選中 {ch.params} ／{k}", X)
    P("  分組（只報告）：")
    P(R.group_report(T, LAB))
    P(overlap_report(T))
    ctxt = []
    cres = {}
    if controls:
        P("  對照（只報告，不參與挑選）：")
        for lab_, f in controls.items():
            X = f(prm)
            P(f"    {lab_}: " + seg_line(X))
            summarize_row(name, "對照 " + lab_, X)
            cres[lab_] = X
    ctext, cr = combo_report(raw, prio, name)
    P("  組合效果（基準＋本策略原始訊號，先到先做、每日 $3,000）：\n" + ctext)
    summarize_row(name, f"基準＋選中 1口", cr[1.0]["Bk"]); summarize_row(name, f"基準＋選中 0.5口", cr[0.5]["Bk"])
    day1111(name, keep, ch.params)
    T.to_csv(os.path.join(OUT, f"trades_{name}.csv"), index=False)
    cr[0.5]["Bk"].to_csv(os.path.join(OUT, f"trades_combo_base+{name}_0.5lot.csv"), index=False)
    RESULTS[name] = dict(params=prm, plateau=has_pl, T=T, stress=stv, p_valid=pv, combo=cr, D=D, controls=cres)
    return RESULTS[name]


# ================================================================== Q03v2
def q03_controls():
    return {
        "hard_only 對照（同 m、start）": lambda prm: R.book(S.gen_q03(prm["m"], "hard_only", prm["start"])),
        "10:02 開盤進場（start=10:00 版）": lambda prm: R.book(S.gen_q03(prm["m"], prm["trail"], "10:00", late=2)),
    }


full_strategy("Q03v2", 3, S.gen_q03, dict(m=[1.0, 1.5, 2.0], trail=["vwap_band", "hard_only"], start=["10:00", "10:30"]),
              ["m"], ["trail", "start"], ["T", "chk_p", "chk_p_slot", "chk_O930", "chk_PDC", "chk_sigma", "chk_UB", "chk_LB", "chk_vwap", "chk_ADR10", "sl"],
              controls=q03_controls())
# RVR 分組已含在分組報告中
P(f"  （{time.time() - t0:.0f}s）")

# ================================================================== Q13v2
full_strategy("Q13v2", 4, S.gen_q13, dict(decision_time=["09:45", "13:00"], z0=[0.0, 1.0, 1.5]), ["z0"], ["decision_time"],
              ["chk_beta", "chk_nq_now", "chk_nq_60", "chk_rel_bp", "chk_sd_bp", "chk_z", "chk_ADR10", "sl"],
              controls={"B3 原版（09:45、正負號、無停損、11:44 出）": lambda prm: S.gen_q13("09:45", 0.0, nostop=True),
                        "另一個決策時刻（同 z0）": lambda prm: R.book(S.gen_q13("13:00" if prm["decision_time"] == "09:45" else "09:45", prm["z0"]))})
x = RESULTS["Q13v2"]["T"]
P(f"  09:45 版與基準 I2 同日：{len(set(x[x.et.str.startswith('09:4')].date) & set(BASE_B[(BASE_B.e >= 930) & (BASE_B.e < 1110)].date))} 天（基準早盤有單的日子）")

# ================================================================== Q02v2
full_strategy("Q02v2", 5, S.gen_q02, dict(k=[0.5, 1.0, 1.5], SL_pts=[60, 120]), ["k", "SL_pts"], [],
              ["chk_p1529", "chk_PDC", "chk_ROD_pct", "chk_sigd20_pct", "chk_ratio", "sl"],
              controls={"k=0（每天順 ROD，涵蓋 Q01 無條件版）": lambda prm: R.book(S.gen_q02(0.0, prm["SL_pts"])),
                        "反向版（同 k、SL）": lambda prm: R.book(S.gen_q02(prm["k"], prm["SL_pts"], reverse=True)),
                        "反向 k=0（≈A2 尾盤反轉）": lambda prm: R.book(S.gen_q02(0.0, prm["SL_pts"], reverse=True))})

# ================================================================== Q08v2
full_strategy("Q08v2", 6, S.gen_q08, dict(entry_time=["03:00", "06:00"], SL_pts=[80, 120]), ["SL_pts"], ["entry_time"],
              ["chk_event", "sl"],
              controls={"非數據日同規則（隔夜漂移對照）": lambda prm: S.gen_q08(prm["entry_time"], prm["SL_pts"], days="nonevent"),
                        "只看 CPI": lambda prm: (lambda T: T[T.chk_event.str.contains("CPI")])(S.gen_q08(**prm)),
                        "只看 NFP": lambda prm: (lambda T: T[T.chk_event.str.contains("NFP")])(S.gen_q08(**prm))})
q8 = RESULTS["Q08v2"]
ev = q8["T"]; non = q8["controls"]["非數據日同規則（隔夜漂移對照）"]
perm8 = {}
for sg in R.SEGS + ["全期"]:
    a = ev if sg == "全期" else ev[ev.seg == sg]; b = non if sg == "全期" else non[non.seg == sg]
    perm8[sg] = R.perm_test(a.pnl.values, b.pnl.values)
    P(f"  Q08v2 排列檢定 {sg}: 數據日 {len(a)} 筆 每筆 {a.pnl.mean():+.0f} vs 非數據日 {len(b)} 筆 每筆 {b.pnl.mean():+.0f} → p={perm8[sg]:.4f}")
q8["perm"] = perm8
q8["p_valid"] = perm8["驗證"]

# ================================================================== EVT_FOMC（固定規則、不挑參數）
P("\n" + "=" * 100 + "\nEVT_FOMC（只紙上記錄；固定規則）\n" + "=" * 100)
evt = {}
for rule in ["A", "B"]:
    T = R.book(S.gen_evt(rule))
    non = R.book(S.gen_evt(rule, days="non"))
    P(f"規則 {rule}（{'09:31 做多→13:55，SL120' if rule == 'A' else '|Δ(13:59→14:59)|≥0.15×ADR10 時 15:00 逆向，SL100→15:59'}）")
    P(R.fmt_seg(R.seg_stats(T)))
    P("  非 FOMC 日同規則：" + seg_line(non))
    pp = {}
    for sg in R.SEGS + ["全期"]:
        a = T if sg == "全期" else T[T.seg == sg]; b = non if sg == "全期" else non[non.seg == sg]
        pp[sg] = R.perm_test(a.pnl.values, b.pnl.values) if len(a) else np.nan
        P(f"    排列 {sg}: FOMC {len(a)} 筆 每筆 {a.pnl.mean() if len(a) else np.nan:+.0f} vs 非 FOMC {len(b)} 筆 每筆 {b.pnl.mean():+.0f} → p={pp[sg]:.4f}")
    raw = S.gen_evt(rule)
    check_signals(raw, "EVT" + rule, ["chk_c1359", "chk_c1459", "chk_delta", "chk_thr"])
    stv = {"成本加倍": R.book(S.gen_evt(rule, cost=96.0)), "不利進場價": R.book(S.gen_evt(rule, entry="adverse")),
           "晚 2 分鐘": R.book(S.gen_evt(rule, late=2)), "進出各差 2 點": R.book(S.gen_evt(rule, slip=2.0))}
    if rule == "B":
        stv["進出各差 4 點"] = R.book(S.gen_evt(rule, slip=4.0))
    stv["拿掉最大 5 天"] = R.drop_top_days(T, 5)[0]
    for k, X in stv.items():
        P(f"    {k:14s} " + seg_line(X))
        summarize_row("EVT_FOMC", f"規則{rule} ／{k}", X)
    summarize_row("EVT_FOMC", f"規則{rule}", T); summarize_row("EVT_FOMC", f"規則{rule} 非FOMC對照", non)
    ctext, cr = combo_report(raw, 7, f"EVT_{rule}")
    P(ctext)
    x = T[T.date == "2026-11-11"]
    P(f"  2026-11-11：{'不是 FOMC 日 → 不交易' if len(x) == 0 else x.to_string()}")
    T.assign(rule=rule).to_csv(os.path.join(OUT, f"trades_EVT_FOMC_{rule}.csv"), index=False)
    evt[rule] = dict(T=T, perm=pp, combo=cr, stress=stv)
RESULTS["EVT_FOMC"] = evt

# ================================================================== Holm、分級
P("\n" + "=" * 100 + "\nHolm 校正（5 個假設，驗證期單尾 p）與分級\n" + "=" * 100)
names = ["Q03v2", "Q13v2", "Q02v2", "Q08v2", "EVT_FOMC"]
pv = [RESULTS[n]["p_valid"] for n in names[:4]] + [evt["A"]["perm"]["驗證"]]
adj = R.holm(pv)
grades = {}
for n, p_, a_ in zip(names, pv, adj):
    if n == "EVT_FOMC":
        grades[n] = "只紙上記錄（規格規定）"
        P(f"  {n}: 驗證 p={p_:.4f} Holm={a_:.4f} → {grades[n]}")
        continue
    r = RESULTS[n]; st = R.seg_stats(r["T"])
    three = all(st[s]["tot"] > 0 for s in ["探索", "驗證", "測試"])
    s2 = r["stress"]
    stress_ok = R.seg_stats(s2["成本加倍 $96"])["全期"]["tot"] > 0 and R.seg_stats(s2["進場用不利價（多 high／空 low）"])["全期"]["tot"] > 0
    combo_ok = r["combo"][0.5]["better"]
    extra_ok = True
    if n == "Q08v2":
        non = r["controls"]["非數據日同規則（隔夜漂移對照）"]
        beat = all(r["T"][r["T"].seg == s].pnl.mean() > non[non.seg == s].pnl.mean() for s in ["探索", "驗證"])
        extra_ok = (r["perm"]["探索"] < 0.10) and beat
    if not (r["plateau"] and three):
        g_ = "否決"
    elif a_ < 0.10 and stress_ok and combo_ok and extra_ok:
        g_ = "通過（可小量實盤）"
    else:
        g_ = "只紙上記錄"
    grades[n] = g_
    P(f"  {n}: 參數 {r['params']}；高原 {r['plateau']}；三段皆賺 {three}；驗證 p={p_:.4f} Holm={a_:.4f}；壓力 OK {stress_ok}；"
      f"0.5 口組合改善 {combo_ok}；額外條件 {extra_ok} → {g_}")
P("  補充（不改變分級）：驗證＋測試合併的單尾 p（Q03v2、Q13v2 09:45 的測試期依規格不算獨立證據）")
pv2 = []
for n in names[:4]:
    x = RESULTS[n]["T"]; x = x[x.seg.isin(["驗證", "測試"])]
    if n == "Q08v2":
        non = RESULTS[n]["controls"]["非數據日同規則（隔夜漂移對照）"]
        pv2.append(R.perm_test(x.pnl.values, non[non.seg.isin(["驗證", "測試"])].pnl.values))
    else:
        pv2.append(R.tstat_p(x.pnl.values)[1])
xa = evt["A"]["T"]; xa = xa[xa.seg.isin(["驗證", "測試"])]
na = R.book(S.gen_evt("A", days="non")); na = na[na.seg.isin(["驗證", "測試"])]
pv2.append(R.perm_test(xa.pnl.values, na.pnl.values))
for n, p_, a_ in zip(names, pv2, R.holm(pv2)):
    P(f"    {n}: 驗證＋測試 p={p_:.4f} Holm={a_:.4f}")
json.dump(dict(grades=grades, holm=dict(zip(names, map(float, adj))), p=dict(zip(names, map(float, pv))),
               params={n: RESULTS[n]["params"] for n in names[:4]}), open(os.path.join(OUT, "grades.json"), "w"), ensure_ascii=False, indent=1)
pd.DataFrame(SUMMARY).to_csv(os.path.join(OUT, "summary.csv"), index=False)
open(os.path.join(OUT, "results_detail.txt"), "w").write("\n".join(LOG))
P(f"完成 {time.time() - t0:.0f}s")
