import numpy as np, pandas as pd
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
exec(open("trend.py").read().split("def vwap_row")[0])
F = pd.read_csv("wr_table.csv.gz"); F["key"] = F.rule + "|" + F.side
F["D"] = F.date.map(Dser)
F["P"] = np.select([F.date >= "2026-10-01", F.date >= "2026-01-01", F.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
rth = F.entry_slot >= 930
# 盤中／夜盤順勢：美盤＝進場價在 VWAP 同一邊（to_vwap False）；開盤前＝價格已經往交易方向走（相對前一天收盤）
F["I_al"] = np.where(rth, F.to_vwap == False, F.move > 0)
F["D_al"] = F.D == F.dir
F["D_ctr"] = F.D == -F.dir
P = {"I2": F.key.str.startswith("I2"), "前高空": F.key == "AM-PD-k1|空", "前低多": F.key == "AM-PD-k1|多",
     "亞洲多 ES": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True), "亞洲 VAL 多 ES": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),
     "美盤早掃夜盤低多 ES": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2), "亞洲空": F.key == "ASIA-PD-k1|空"}
pd.set_option("display.width", 250)
def line(x):
    h = x[x.P != "近期"].pnl55; r = x[x.P == "近期"].pnl55
    yrs = x[x.P != "近期"].groupby("P").pnl55.mean().round(0).to_dict()
    return f"過去 {len(h):3d} 筆 {(h>0).mean()*100:5.1f}% {h.mean():+5.0f}（{yrs}）｜近期 {len(r):2d} 筆 {(r>0).mean()*100 if len(r) else float('nan'):5.1f}% {r.mean() if len(r) else float('nan'):+5.0f}"
for k, m in P.items():
    x = F[m]; print("==", k)
    print("   全部        ", line(x))
    print("   順日線      ", line(x[x.D_al]))
    print("   逆日線      ", line(x[x.D_ctr]))
    print("   日線盤整    ", line(x[x.D == 0]))
    print("   順盤中/夜盤 ", line(x[x.I_al == True]))
    print("   逆盤中/夜盤 ", line(x[x.I_al == False]))
print("== 所有點合併（7,517 筆）")
for nm, m in [("順日線", F.D_al), ("逆日線", F.D_ctr), ("日線盤整", F.D == 0), ("順盤中/夜盤", F.I_al == True), ("逆盤中/夜盤", F.I_al == False)]:
    print(f"   {nm:10s}", line(F[m]))
print("今天 11/12 的日線趨勢 D =", Dser.get("2026-11-12", Dser.iloc[-1]), "｜11/11 =", Dser.get("2026-11-11"), "｜11/05 =", Dser.get("2026-11-05"), "｜11/06 =", Dser.get("2026-11-06"))
print(Dser["2026-10-01":].to_string())
