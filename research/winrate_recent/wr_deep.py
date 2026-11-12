import numpy as np, pandas as pd, itertools
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib 和 research/multi_entry 的 family.py、f_family.py；放到該分支 research/ 下執行（見 docs/22）。
exec(open("wr_search.py").read().split("def stat")[0])
F["Y"] = np.select([F.date >= "2026-10-01", F.date >= "2026-01-01", F.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
def yr(x, tp=55):
    g = x.groupby("Y")[f"pnl{tp}"]; w = g.apply(lambda p: (p > 0).mean() * 100).round(0); n = g.size(); a = g.mean().round(0)
    return " ｜".join(f"{y}:{int(n.get(y,0))}筆 {w.get(y,np.nan):.0f}% {a.get(y,np.nan):+.0f}" for y in ["2024", "2025", "2026-1~9", "近期"])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
print("=== D. 候選逐年（筆數、勝率、每筆，止盈 55）")
CAND = {"美盤早 掃夜盤高收回 空｜高波動": (F.key == "AM-ON-k1|空") & (F.volb == "高>1.8"),
        "亞洲 掃前日低收回 多｜高波動": (F.key == "ASIA-PD-k1|多") & (F.volb == "高>1.8"),
        "亞洲 掃前日低收回 多｜中＋高波動": (F.key == "ASIA-PD-k1|多") & (F.volb != "低波動<1.2"),
        "亞洲 掃 VAL 收回 多｜價格已在前收之上": (F.key == "ASIA-VA-k1|多") & (F.moveb == "已往這邊走"),
        "亞洲 掃 VAL 收回 多｜ES 同步": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),
        "亞洲 掃前日低收回 多｜ES 同步（指標的亞洲多 ES）": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True),
        "美盤早 掃夜盤低收回 多｜中波動": (F.key == "AM-ON-k1|多") & (F.volb == "中1.2-1.8"),
        "美盤早 掃夜盤低收回 多｜ES 同步": (F.key == "AM-ON-k1|多") & (F.es_ok == True),
        "I2 多": (F.key == "I2|多"), "I2 空": (F.key == "I2|空"),
        "前高空": (F.key == "AM-PD-k1|空"), "前低多｜順 I2": (F.key == "AM-PD-k1|多") & (F.i2 != "逆")}
for k, m in CAND.items(): print(f"{k:34s} {yr(F[m])}")
print("\n=== E. 止盈大小對勝率（停損 145 不變）：推薦組合的 4 個點")
REC4 = (F.key == "I2|多") | (F.key == "I2|空") | (F.key == "AM-PD-k1|空") | ((F.key == "AM-PD-k1|多") & (F.i2 != "逆")) | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True))
for tp in (20, 30, 40, 55, 80):
    x = F[REC4]
    print(f"止盈 {tp:>2}：", "｜".join(f"{w}: 勝率 {(x[x.W==w][f'pnl{tp}']>0).mean()*100:.1f}% 每筆 {x[x.W==w][f'pnl{tp}'].mean():+.0f} 合計 {x[x.W==w][f'pnl{tp}'].sum():+,.0f}" for w in ["HIST", "REC"]))
print("\n=== F. 兩個條件疊加（所有規則，止盈 55）：兩段勝率都 ≥ 70%、兩段都賺、過去 ≥ 30 筆、近期 ≥ 4 筆")
fl = {"ES同步": F.es_ok == True, "順I2(美盤)": F.i2 == "順", "中高波動": F.volb != "低波動<1.2", "高波動": F.volb == "高>1.8",
      "不往VWAP": F.to_vwap == False, "已往這邊走": F.moveb == "已往這邊走", "夜盤寬(美盤)": F.onb == "夜盤寬"}
out = []
for k, x in F.groupby("key"):
    for a, b in itertools.combinations(fl, 2):
        m = fl[a].loc[x.index] & fl[b].loc[x.index]; y = x[m]
        h, r = y[y.W == "HIST"].pnl55, y[y.W == "REC"].pnl55
        if len(h) >= 30 and len(r) >= 4:
            out.append(dict(rule=k, f=a + "＋" + b, hist_n=len(h), hist_win=(h > 0).mean() * 100, hist_avg=h.mean(), rec_n=len(r), rec_win=(r > 0).mean() * 100, rec_avg=r.mean()))
O = pd.DataFrame(out); O = O[(O.hist_win >= 70) & (O.rec_win >= 70) & (O.hist_avg > 0) & (O.rec_avg > 0)]
print(O.sort_values("hist_win", ascending=False).round(1).to_string(index=False))
