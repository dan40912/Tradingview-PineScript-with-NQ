import numpy as np, pandas as pd
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib 和 research/multi_entry 的 family.py、f_family.py；放到該分支 research/ 下執行（見 docs/22）。
exec(open("wr_deep.py").read().split('pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)')[0])
REC4 = (F.key == "I2|多") | (F.key == "I2|空") | (F.key == "AM-PD-k1|空") | ((F.key == "AM-PD-k1|多") & (F.i2 != "逆")) | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True))
def book(mask, one_asia_long=False):
    T = F[mask].sort_values(["date", "entry_slot"]).copy()
    if one_asia_long:   # 亞洲盤做多每晚最多一筆（VAL 和前日低常常同一晚都觸發）
        a = T.key.str.startswith("ASIA") & (T.dir == 1)
        T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
    keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []
        for t in x.itertuples():
            if sum(p for xs, p in closed if xs <= t.entry_slot) <= -2900: keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl55))
    return T[keep]
def dd(p): c = p.cumsum(); return (c - c.cummax()).min()
ndays = F.groupby("Y").date.nunique()
BOOKS = {
 "推薦組合（I2＋前高空＋前低多順I2＋亞洲多ES）": (REC4, False),
 "高勝率組合（I2 多＋亞洲多 ES＋亞洲 VAL 多 ES＋美盤早掃夜盤低多 ES 中高波動）": (
    (F.key == "I2|多") | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True)) | ((F.key == "ASIA-VA-k1|多") & (F.es_ok == True)) |
    ((F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.volb != "低波動<1.2")), True),
 "高勝率組合＋I2 空": (
    (F.key == "I2|多") | (F.key == "I2|空") | ((F.key == "ASIA-PD-k1|多") & (F.es_ok == True)) | ((F.key == "ASIA-VA-k1|多") & (F.es_ok == True)) |
    ((F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.volb != "低波動<1.2")), True),
}
for k, (m, one) in BOOKS.items():
    T = book(m, one); print("==", k)
    for y in ["2024", "2025", "2026-1~9", "近期"]:
        x = T[T.Y == y].pnl55
        print(f"   {y:9s} {len(x):4d} 筆｜每天 {len(x)/ndays[y]:.2f}｜勝率 {(x>0).mean()*100:5.1f}%｜每筆 {x.mean():+5.0f}｜合計 {x.sum():+8,.0f}｜最大回撤 {dd(x):+,.0f}")
    x = T.pnl55; print(f"   全期      {len(x):4d} 筆｜勝率 {(x>0).mean()*100:5.1f}%｜合計 {x.sum():+,.0f}｜最大回撤 {dd(x):+,.0f}")
for name, m in {"AM-ON-k1 多 ES 中高波動": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.volb != "低波動<1.2"),
                "ASIA-VA-k1 多 ES 中高波動": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True) & (F.volb != "低波動<1.2")}.items():
    print(f"{name:26s} {yr(F[m])}")
B = book(BOOKS["高勝率組合（I2 多＋亞洲多 ES＋亞洲 VAL 多 ES＋美盤早掃夜盤低多 ES 中高波動）"][0], True)
print(B[B.Y == "近期"][["date", "key", "entry_slot", "why55", "pnl55"]].to_string(index=False))
