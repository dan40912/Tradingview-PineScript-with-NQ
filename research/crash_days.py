"""劇烈跳空日（像 2026-10-30：開盤離價值區很遠、在前日範圍外）歷史上怎麼走、各種做法的結果。"""
import pickle, numpy as np, pandas as pd, strat
from eng import sim
nq = pickle.load(open("NQ_days2.pkl", "rb"))
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
rows = []
for d in nq:
    O, H, L, C = d["O"], d["H"], d["L"], d["C"]
    if np.isnan(O[0]): continue
    o = O[0]; w = d["vah"] - d["val"]; g = 1 if o > d["vah"] else -1 if o < d["val"] else 0
    if g == 0: continue
    dist = (o - d["vah"]) if g == 1 else (d["val"] - o); pct = dist / o
    if not (dist > w and (o > d["pdh"] or o < d["pdl"]) and pct >= 0.01): continue   # 開太遠（H1 跳過）、前日範圍外、離價值區 ≥ 1%
    ibh, ibl = np.nanmax(H[:30]), np.nanmin(L[:30]); last = C[~np.isnan(C)][-1]
    back = any(d["val"] <= C[i] <= d["vah"] for i in range(4, 390, 5) if not np.isnan(C[i]))
    t = strat.trend(d, **T1)
    # 對照做法：10:05 開盤直接順著跳空方向 / 逆著跳空方向，SL 80、TP 160、12:29 出場
    f_with = sim(d, 35, -1 if g == -1 else 1, 160, 80, 179) if not np.isnan(O[35]) else None
    f_fade = sim(d, 35, 1 if g == -1 else -1, 160, 80, 179) if not np.isnan(O[35]) else None
    rows.append(dict(date=d["date"], 方向="跳空跌" if g == -1 else "跳空漲", 離價值區=round(dist), 離區百分比=f"{pct:.1%}", 區寬=round(w),
                     開到收=round(last - o), IB寬=round(ibh - ibl), 收回價值區="是" if back else "否",
                     順勢規則=(f"{t['res']} {t['pnl']:+.0f}" if t else "沒觸發"), 順勢pnl=t["pnl"] if t else 0.0,
                     _10點05順跳空=f_with["pnl"] if f_with else np.nan, _10點05逆跳空=f_fade["pnl"] if f_fade else np.nan))
X = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"2024-01～2026-10-19 共 {len(X)} 天符合（開盤離價值區 ≥ 1%、超過區寬、在前日範圍外）\n")
print(X.drop(columns=["順勢pnl"]).to_string(index=False))
for lab, x in (("全部", X), ("跳空跌", X[X.方向 == "跳空跌"]), ("跳空漲", X[X.方向 == "跳空漲"])):
    trig = x[x.順勢規則 != "沒觸發"]
    print(f"\n【{lab}】{len(x)} 天｜收盤和開盤同方向（續跌／續漲）{((x.開到收 * np.where(x.方向 == '跳空跌', -1, 1)) > 0).mean():.0%}｜當天收回價值區 {(x.收回價值區 == '是').mean():.0%}")
    print(f"   順勢規則觸發 {len(trig)} 天，每筆 {trig.順勢pnl.mean():+,.0f}，勝率 {(trig.順勢pnl > 0).mean():.0%}，合計 {trig.順勢pnl.sum():+,.0f}")
    print(f"   10:05 直接順著跳空做（SL 80／TP 160）：每筆 {x._10點05順跳空.mean():+,.0f}，勝率 {(x._10點05順跳空 > 0).mean():.0%}")
    print(f"   10:05 逆著跳空做（接刀子，SL 80／TP 160）：每筆 {x._10點05逆跳空.mean():+,.0f}，勝率 {(x._10點05逆跳空 > 0).mean():.0%}")
