"""第五批：只有停損跟 ATR 走（止盈固定），口數讓最壞虧損 ≈ $2,948（最多 1 口）。寫 results5.csv。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import exp
from core import *
exp.run("0基準", "基準 +55/−145", plan())
exp.run("0基準", "基準（books17 記帳）", plan(), book_kw=dict(sched=True))
for k, tp, cap in itertools.product([0.5, 0.56, 0.65, 0.75, 0.85, 1.0], [45, 55, 65, 75], [1.0, 1.5]):
    R = simulate(plan(sl=k * ATR_REF, tp1=tp), scale=CAND.atr.values / ATR_REF, only_sl=True)
    exp.run("B_ATR停損", f"停損 {k}×ATR、TP{tp}、同風險≤{cap}口", R=R, size=sizes(R, "risk", cap=cap))
pd.DataFrame(exp.ROWS).to_csv("results5.csv", index=False)
