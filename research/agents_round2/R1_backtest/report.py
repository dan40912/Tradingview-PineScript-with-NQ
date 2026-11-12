"""產生 REPORT.txt 的數字段落（report_body.txt）。研究用途，不是投資建議。"""
import sys, os, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r1lib as R
import run as RU

OUT = R.OUT
X = pd.read_pickle(os.path.join(OUT, "analysis.pkl"))
A = pd.read_pickle(os.path.join(OUT, "all_combos.pkl"))
SKIP = pd.read_pickle(os.path.join(OUT, "skip.pkl"))
SUM, SEL, DET, GR, HT, BT, BT_CT = (X[k] for k in ("SUM", "SEL", "DET", "GR", "HT", "BT", "BT_CT"))
ORDER = RU.SPEC["priority_order"]
SEGS = R.SEGS
L = []


def P(s=""):
    L.append(str(s))


def segline(st, label=""):
    out = []
    for k in SEGS:
        v = st[k]
        if v["n"] == 0:
            out.append(f"{k} 0筆"); continue
        out.append(f"{k} {v['n']}筆 勝{v['win']:.0f}% 每筆{v['avg']:+,.0f} 合計{v['tot']:+,.0f} 回撤{v['mdd']:+,.0f} 最差日{v['worst']:+,.0f}")
    a = st["全期"]
    rr = a["tot"] / -a["mdd"] if a["mdd"] < 0 else float("inf")
    return label + "｜".join(out) + f"｜全期 {a['n']}筆 合計{a['tot']:+,.0f} 回撤{a['mdd']:+,.0f} 報酬/回撤 {rr:.2f}"


def short(st):
    return " / ".join(f"{st[k]['avg']:+,.0f}" if st[k]["n"] else "—" for k in SEGS) + f"（合計 {st['全期']['tot']:+,.0f}）"


# ------------------------------------------------------------------ 76 組總表
P("【附表 1】76 組全部結果（1 口 NQ，每筆扣 $48；每筆＝平均損益；p＝探索＋驗證每筆>0 單尾 t 檢定；Holm 對 76 組校正）")
P("組合 | 探索 筆/每筆 | 驗證 筆/每筆 | 測試 筆/每筆 | 近期 筆/每筆 | 全期合計 | 全期回撤 | p(EV) | Holm | 同天同向基準% | 合格EV/高原 | 選中")
for _, r in SUM.iterrows():
    seg = " | ".join(f"{int(r[f'{k}_n'])}/{r[f'{k}_avg']:+.0f}" if r[f"{k}_n"] else "0/—" for k in SEGS)
    P(f"{r['name']} | {seg} | {r['全期_tot']:+,.0f} | {r['全期_mdd']:+,.0f} | {r.p_EV:.3f} | {r.holm_EV:.3f} | "
      f"{r.get('同天同向基準%', np.nan):.0f} | {'合格' if r['合格EV'] else '—'}/{r['高原']} | {r['選中']}")
P()
P("Holm（探索＋驗證，76 組）顯著 <0.05 的：" + ", ".join(f"{r['name']}({r.holm_EV:.3f})" for _, r in SUM[SUM.holm_EV < 0.05].iterrows()))
P()
P("QC2／QC3／QC7 計數：")
for sid in ORDER:
    s = SUM[SUM.sid == sid]
    P(f"  {sid}: 停損<15改15 共 {int(s['停損下限15筆數'].sum())} 筆（各組合加總）；停損>145 不做 {int(s['停損>145不做'].sum())} 次；"
      f"QC3 改 10:03 進場 {int(s['QC3改10:03'].sum())} 筆、取消 {int(s['QC3取消'].sum())} 次")
P(f"  QC7 換月刪除天數：S11 N=5/10/20 → {SKIP.get('S11_N5_roll')}/{SKIP.get('S11_N10_roll')}/{SKIP.get('S11_N20_roll')} 天（回看期跨合約）；"
  f"S09 → {SKIP.get('S09_roll')} 天（前 2 日不同合約）；S07、S09 另要求前一 RTH 日同合約（同 features 的 has_prev）。")
P(f"  S02 停損>145 不掛單 {SKIP.get('S02_gt145', 0)} 次；S09 limit 觸價時停損>145 跳過 {SKIP.get('S09_gt145', 0)} 次。")
P()

# ------------------------------------------------------------------ 逐策略
for sid in ORDER:
    d = DET[sid]; r = d["row"]; g, c = GR[sid]
    T = d["T"]
    st = R.seg_table(T)
    P("=" * 100)
    P(f"{sid} {[s for s in RU.SPEC['strategies'] if s['id'] == sid][0]['name']} → 機械分級：{g}")
    P(f"選中：{d['name']}（{r['選中']}；探索＋驗證 t={r.t_EV:.2f}、p={r.p_EV:.3f}、Holm={r.holm_EV:.3f}；高原：{r['高原']} {r.get('鄰居', '')}）")
    P("四段（單獨，1 口）：")
    P("  " + segline(st))
    P(f"  多/空：多 {int((T.d == 1).sum())} 筆 {T[T.d == 1].pnl.sum():+,.0f}；空 {int((T.d == -1).sum())} 筆 {T[T.d == -1].pnl.sum():+,.0f}；"
      f"出場 {T.why.value_counts().to_dict()}")
    P("  成本加倍（$96）：" + short(d["cost2"]))
    P(f"  不利價（{d['adv_note']}）：" + short(d["adv"]))
    if "c9" in d:
        P("  對照 P 的 C9 近似（觸價後下一分鐘開盤進場、開盤在停損外取消）：" + short(d["c9"]))
    P(f"  拿掉最大 5 天（{', '.join(f'{k} {v:+,.0f}' for k, v in d['top5'].items())}）：全期 {d['no_top5']:+,.0f}；"
      + "／".join(f"{k} {v:+,.0f}" for k, v in d["no_top5_seg"].items()))
    P("  穩健性（只報告）：" + "；".join(f"{k} " + (short(v) if v is not None else "不適用（限價單）") for k, v in d["robust"].items()))
    for n2, s2 in d["sib"]:
        P(f"  同一批訊號另一種出場 {n2}：" + short(s2))
    P(f"  同天同向：基準 {100 * T.base_same_day_dir.mean():.0f}%、C1/TD10 {100 * T.c1_td10_same_day_dir.mean():.0f}%；"
      f"在基準有單的日子 {int(T.base_day.sum())} 筆")
    P("  只在基準沒單的日子：" + segline(d["nobase"]))
    P("  基準有單的日子：      " + short(d["withbase"]))
    P("  合併（平倉即記帳、每日 $3,000、同分鐘 基準＞C1＞TD10＞本策略）：")
    for nm, M in d["merged"].items():
        tt = {s: M[M.seg == s].pnl.sum() for s in SEGS}
        mdd = R.dd(M.sort_values(["date", "e"]).pnl.values); tot = M.pnl.sum()
        nb = int((M.src == "base").sum()); ns = int((M.src == "S").sum())
        P(f"    {nm:22s} " + "  ".join(f"{s} {tt[s]:+9,.0f}" for s in SEGS) + f"  全期 {tot:+10,.0f} 回撤 {mdd:+8,.0f} 比 {tot / -mdd:5.2f}"
          f"  （基準 {nb} 筆、本策略 {ns} 筆）")
    P("  通過條件檢查：" + "；".join(f"{k}={'是' if v is True or v == '是' or v is np.True_ else ('不適用' if v == '不適用' else '否')}" for k, v in c.items()))
    P(f"  測試期 p={r.p_測試:.3f}，8 個選中組合 Holm 後 {HT[sid]:.3f}")
    # 11/11
    P("  2026-11-11 各組合動作：")
    seen = set()
    for (s2, n2), v in A.items():
        if s2 != sid:
            continue
        TT = v["T"]
        x = TT[TT.date == "2026-11-11"] if len(TT) else TT
        if len(x) == 0:
            P(f"    {n2}: 不交易")
            continue
        t = x.iloc[0]
        P(f"    {n2}: {'多' if t.d == 1 else '空'} {t.sig} 訊號／{R.hhmm(t.e)} 進場 {t.px:,.2f}，停損 {t.stop:,.2f}，"
          f"{R.hhmm(t.x)} {({'tp': '止盈', 'sl': '停損', 'time': '時間出場', 'be': '保本', 'sl_gap': '跳空停損'})[t.why]} {t.xp:,.2f}，"
          f"{t.pts:+.2f} 點 ${t.pnl:+,.0f}")
    P()

# ------------------------------------------------------------------ 基準
P("基準（Pine 平倉即記帳，520 筆）各段合計：" + "  ".join(f"{s} {BT[s]:+,.0f}" for s in SEGS))
P("基準＋C1＋TD10 各段合計：" + "  ".join(f"{s} {BT_CT[s]:+,.0f}" for s in SEGS))
open(os.path.join(OUT, "report_body.txt"), "w").write("\n".join(L))
print("\n".join(L))
