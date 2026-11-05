"""C3 step 2 (discovery only): grid edge extension. POC floor reached the edge (F=10) in step 1,
so extend F to 12/15/20; plus a small fixed TP/SL grid per bin to see whether ANY exit works with
finer (scale-equivalent) bins."""
import itertools, numpy as np, pandas as pd
from c3lib import data, levels, i2_exit_sigs, run, stats, tradable_days, half_avgs

gE = data.load_grid("ES")
D = tradable_days(gE, "discovery")
rows = []
for b in [1.25, 2.5, 5.0]:
    lv = levels(gE, b)
    ex = {}
    for F, S, NP in itertools.product([12, 15, 20], [15, 20], [None, 4]):
        ex[f"poc_f{F}_s{S}_np{NP}"] = dict(poc_floor=F, sl=S, np_n=(30 if NP else None), np_x=NP)
    for T, S in itertools.product([22, 40, 55], [15, 22, 58]):
        ex[f"fix_{T}_{S}"] = dict(tp=T, sl=S)
    for name, kw in ex.items():
        tr = run(gE, i2_exit_sigs(gE, lv, **kw))
        td = tr[tr.split == "discovery"]
        rows.append({"bin": b, "exit": name, **stats(td, D), **half_avgs(td)})
df = pd.DataFrame(rows)
df["all3pos"] = (df[["2024H1", "2024H2", "2025H1"]] > 0).all(axis=1)
df.to_csv("out_02_es_extend_discovery.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("variants:", len(df))
print(df.to_string(index=False))
