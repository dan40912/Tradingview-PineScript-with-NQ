"""第三批：寬停損＋小口數（同風險）細網格，檢查高原。寫 results3.csv。研究用途，不是投資建議。"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import exp
from core import *
exp.run("0基準", "基準 +55/−145", plan())
exp.run("0基準", "基準（books17 記帳）", plan(), book_kw=dict(sched=True))
for sl, tp in itertools.product([145, 160, 175, 190, 200, 225, 250, 300], [40, 45, 55, 65, 75, 85]):
    exp.run("A寬停損同風險", f"SL{sl}/TP{tp} 同風險", plan(sl=sl, tp1=tp), size_mode="risk", size_kw=dict(cap=3.0))
pd.DataFrame(exp.ROWS).to_csv("results3.csv", index=False)
