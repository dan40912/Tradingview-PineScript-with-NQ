"""檢查 3（A）：用 rt_selA_all.csv 算基準率。"""
import os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
Z = pd.read_csv(os.path.join(H, "rt_selA_all.csv"))
A0 = pd.read_csv(os.path.join(H, "..", "A_trend", "explore_dv.csv"))
out = []
def P(s): print(s); out.append(s)
# 和 A 的探索／驗證一致性檢查
m = Z.merge(A0[["fam", "params", "stop", "exit", "d_avg", "v_avg"]], on=["fam", "params", "stop", "exit"], suffixes=("", "_A"))
P(f"和 A 的 explore_dv 對上 {len(m)} 列；探索每筆差 >1 的 {int(((m.d_avg - m.d_avg_A).abs() > 1).sum())}，驗證差 >1 的 {int(((m.v_avg - m.v_avg_A).abs() > 1).sum())}")
Z = Z[Z.fam != "對照每天做多"]
okn = (Z.d_n >= 30) & (Z.v_n >= 10) & (Z.t_n >= 10)
Zs = Z[okn]
ev = Zs[(Zs.d_avg > 0) & (Zs.v_avg > 0)]
P(f"A 全部 {len(Z)} 種（筆數夠的 {len(Zs)}）：探索＋驗證都賺 {len(ev)}；其中測試也賺 {int((ev.t_avg > 0).sum())}（{100*(ev.t_avg > 0).mean():.1f}%）；全部變化測試賺的比例 {100*(Zs.t_avg > 0).mean():.1f}%")
P(f"  探索＋驗證都賺的變化：測試每筆中位 {ev.t_avg.median():+.0f}，平均 {ev.t_avg.mean():+.0f}；近期也賺 {100*(ev.r_avg > 0).mean():.0f}%")
for f, x in ev.groupby("fam"):
    allf = Zs[Zs.fam == f]
    P(f"  {f}: 兩段都賺 {len(x)}／{len(allf)}，測試也賺 {100*(x.t_avg > 0).mean():.0f}%（該家族全部 {100*(allf.t_avg > 0).mean():.0f}%），測試每筆中位 {x.t_avg.median():+.0f}")
# 加上「兩段每筆都 ≥ +200」等更嚴格篩選
for th in [100, 200, 300, 400]:
    x = Zs[(Zs.d_avg >= th) & (Zs.v_avg >= th)]
    P(f"  探索、驗證每筆都 ≥ +{th}：{len(x)} 種，測試也賺 {100*(x.t_avg > 0).mean():.0f}%，測試每筆中位 {x.t_avg.median():+.0f}")
c1 = Z[(Z.fam == "F1開盤驅動") & (Z.params == json.dumps(dict(N=5, k=0.15, cl=0.0, es=0, side="both"), ensure_ascii=False)) & (Z.stop == "0.25ADR") & (Z.exit == "移100")].iloc[0]
P(f"C1 本身：探索 {c1.d_avg:+.0f}／驗證 {c1.v_avg:+.0f}／測試 {c1.t_avg:+.0f}／近期 {c1.r_avg:+.0f}")
x = ev[(ev.d_avg >= 400) & (ev.v_avg >= 400)]
P(f"  和 C1 同級（探索、驗證每筆都 ≥ +400）的 {len(x)} 種：測試也賺 {100*(x.t_avg > 0).mean():.0f}%，測試每筆中位 {x.t_avg.median():+.0f}；C1 的測試 {c1.t_avg:+.0f} 在其中的百分位 {100*(x.t_avg < c1.t_avg).mean():.0f}%")
f1 = ev[ev.fam == "F1開盤驅動"]
P(f"  F1 家族兩段都賺的 {len(f1)} 種：C1 測試每筆的百分位 {100*(f1.t_avg < c1.t_avg).mean():.0f}%")
# 以「探索＋驗證取小者」排名前 20 的變化，測試表現（模擬『挑最好的』）
ev2 = ev.assign(s=np.minimum(ev.d_avg, ev.v_avg)).sort_values("s", ascending=False)
P("  探索＋驗證(取小者)最高的 20 種的測試每筆：" + ", ".join(f"{v:+.0f}" for v in ev2.head(20).t_avg) + f"；平均 {ev2.head(20).t_avg.mean():+.0f}，賺的比例 {100*(ev2.head(20).t_avg > 0).mean():.0f}%")
for f in ["F1開盤驅動"]:
    y = ev2[ev2.fam == f]
    P(f"  {f} 最高 20 種的測試每筆平均 {y.head(20).t_avg.mean():+.0f}，賺的比例 {100*(y.head(20).t_avg>0).mean():.0f}%")
# 出場維度：同進場（C1）下 45 種止損×出場的測試
y = Z[(Z.fam == "F1開盤驅動") & (Z.params == c1.params)]
P("  C1 進場配 45 種止損×出場：探索＋驗證都賺 " + str(int(((y.d_avg > 0) & (y.v_avg > 0)).sum())) + f"，測試賺 {int((y.t_avg > 0).sum())}／45，測試每筆中位 {y.t_avg.median():+.0f}")
open(os.path.join(H, "rt_selA.txt"), "w").write("\n".join(out))
