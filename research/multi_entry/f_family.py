"""第三批（PREREG 第三批）：前高空／前低多／亞洲空／亞洲多 × ES 同步（F1）、順著 I2（F2）。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 9 節）。
import sys, os, importlib.util; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import sim
def load(sym):
    sys.argv = ["x", sym]
    spec = importlib.util.spec_from_file_location("fam_" + sym, os.path.join(HERE, "family.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
NQ, ES = load("NQ"), load("ES")
RULES = {"前高空": ("AM-PD-k1", "空"), "前低多": ("AM-PD-k1", "多"), "亞洲空": ("ASIA-PD-k1", "空"), "亞洲多": ("ASIA-PD-k1", "多")}
def sigs(m):
    s = pd.DataFrame(m.signals()); s["date"] = m.g.dates[s.i.values]; return s
SN, SE = sigs(NQ), sigs(ES)
I2DIR = NQ.lv.set_index("date").i2_dir
def build(name, f1, f2):
    rule, side = RULES[name]
    a = SN[(SN.rule == rule) & (SN.side == side)].copy()
    if f1:
        e = SE[(SE.rule == rule) & (SE.side == side)][["date", "entry_slot"]].rename(columns={"entry_slot": "es_slot"})
        a = a.merge(e, on="date", how="inner"); a = a[a.es_slot <= a.entry_slot]
    if f2 and rule.startswith("AM"):
        dd = a.date.map(I2DIR).fillna(0).values
        want = -1 if side == "空" else 1
        a = a[(dd == 0) | (dd == want)]
    return [dict(i=r.i, entry_slot=r.entry_slot, dir=r.dir, exit_slot=r.exit_slot) for r in a.itertuples()]
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["disc", "val", "test"], "fwd")
if __name__ == "__main__":
    rows = []; keep = {}
    for name in RULES:
        for f1, f2 in [(False, False), (True, False), (False, True), (True, True)]:
            if f2 and name.startswith("亞洲"): continue
            tag = name + ("" if not (f1 or f2) else "＋" + "＋".join(x for x, y in [("ES同步", f1), ("順I2", f2)] if y))
            T = sim.run_signals(NQ.g, build(name, f1, f2), tp_pts=55.0, sl_pts=145.0); T["sp"] = sp(T.date.values)
            keep[tag] = T; row = {"rule": tag}
            for s in ["disc", "val", "test", "fwd"]:
                y = T[T.sp == s].pnl; row[f"{s}_n"] = len(y); row[f"{s}_avg"] = round(y.mean()) if len(y) else np.nan
                row[f"{s}_win"] = round((y > 0).mean() * 100, 1) if len(y) else np.nan
                if s == "disc" and len(y) > 2: row["disc_t"] = round(y.mean() / (y.std(ddof=1) / np.sqrt(len(y))), 2)
            row["PASS"] = bool(row["disc_n"] >= 20 and row["disc_avg"] >= 50 and row.get("disc_t", 0) >= 1.0 and row["val_avg"] > 0)
            rows.append(row)
    R = pd.DataFrame(rows); R.to_csv("f_family_summary.csv", index=False)
    pd.concat([t.assign(rule=k) for k, t in keep.items()]).to_csv("f_family_trades.csv", index=False)
    pd.set_option("display.width", 250); print(R.to_string(index=False))
