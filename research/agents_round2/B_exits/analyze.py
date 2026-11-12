"""讀 results.csv，和基準比較。只用探索＋驗證排序；測試只看有沒有一起變好。研究用途，不是投資建議。"""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)
R = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
b = R.iloc[0]
for s in ["探索", "驗證", "測試", "近期", "全期"]:
    R[f"{s}_賺"] = R[f"{s}_tot"] > b[f"{s}_tot"]
    R[f"{s}_穩"] = R[f"{s}_rdd"] > b[f"{s}_rdd"]
    R[f"{s}_好"] = R[f"{s}_賺"] & R[f"{s}_穩"]
R["選"] = R.探索_好 & R.驗證_好                     # 只用前兩段
R["通過"] = R.選 & R.測試_好 & (R.測試_tot > 0)
R["穩選"] = R.探索_穩 & R.驗證_穩 & (R.探索_tot > 0) & (R.驗證_tot > 0)   # 風險調整版（部位類用）
R["穩通過"] = R.穩選 & R.測試_穩 & (R.測試_tot > 0)
# 排序分數：探索、驗證兩段「合計/基準」與「報酬回撤/基準」的最小值
R["分數"] = np.minimum.reduce([R.探索_tot / b.探索_tot, R.驗證_tot / b.驗證_tot, R.探索_rdd / b.探索_rdd, R.驗證_rdd / b.驗證_rdd])
cols = ["fam", "name", "全期_n", "探索_tot", "探索_rdd", "驗證_tot", "驗證_rdd", "測試_tot", "測試_rdd", "近期_tot", "全期_tot", "全期_mdd", "全期_worst", "全期_rdd", "分數", "選", "通過", "穩選", "穩通過"]
if __name__ == "__main__":
    print("總變化數（不含基準 2 列）:", len(R) - 2)
    print(R.groupby("fam").agg(n=("name", "size"), 選=("選", "sum"), 通過=("通過", "sum"), 穩選=("穩選", "sum"), 穩通過=("穩通過", "sum")).to_string())
    print("\n基準\n", R.iloc[:2][cols].round(2).to_string())
    for fam, x in R.iloc[2:].groupby("fam"):
        print("\n==", fam)
        print(x.sort_values("分數", ascending=False).head(8)[cols].round(2).to_string(index=False))
    R.to_csv((sys.argv[1] if len(sys.argv) > 1 else "results.csv").replace(".csv", "_scored.csv"), index=False)
