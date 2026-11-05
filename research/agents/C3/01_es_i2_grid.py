"""C3 step 1: ES I2 (frozen VA algorithm on ES, bins 1.25/2.5/5) x exits. DISCOVERY ONLY for choosing.
Exits: original ES shadow 22/58 (contaminated: chosen on data thru 2026), scaled 55/145 (11/29),
A3-analog family (POC target w/ floor F, stop S, optional no-progress N=30 min / X pts)."""
import itertools, numpy as np, pandas as pd
from c3lib import data, levels, i2_exit_sigs, run, stats, tradable_days, half_avgs

gE = data.load_grid("ES")
D = tradable_days(gE, "discovery")
rows = []
for b in [1.25, 2.5, 5.0]:
    lv = levels(gE, b)
    exits = {"shadow_22_58": dict(tp=22, sl=58), "scaled_11_29": dict(tp=11, sl=29)}
    for F, S, NP in itertools.product([4, 6, 8, 10], [10, 15, 20, 30], [None, 4, 6]):
        exits[f"poc_f{F}_s{S}_np{NP}"] = dict(poc_floor=F, sl=S, np_n=(30 if NP else None), np_x=NP)
    for name, kw in exits.items():
        tr = run(gE, i2_exit_sigs(gE, lv, **kw))
        td = tr[tr.split == "discovery"] if len(tr) else tr
        st = stats(td, D)
        rows.append({"bin": b, "exit": name, **st, **half_avgs(td)})
df = pd.DataFrame(rows)
df["all3pos"] = (df[["2024H1", "2024H2", "2025H1"]] > 0).all(1)
df.to_csv("out_01_es_grid_discovery.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("variants:", len(df))
for b in [1.25, 2.5, 5.0]:
    x = df[df.bin == b]
    print(f"\n=== bin {b}: n={x.n.iloc[0]}  positive {int((x.avg>0).sum())}/{len(x)}  all3pos {int(x.all3pos.sum())}")
    print(x.sort_values("sharpe", ascending=False).head(12).to_string(index=False))
    print(x[x.exit.isin(["shadow_22_58", "scaled_11_29"])].to_string(index=False))
# pivot of avg by F,S for np variants per bin
for b in [1.25, 2.5, 5.0]:
    for NP in ["None", "4", "6"]:
        x = df[(df.bin == b) & df.exit.str.endswith("np" + NP)].copy()
        x["F"] = x.exit.str.extract(r"f(\d+)").astype(int); x["S"] = x.exit.str.extract(r"_s(\d+)").astype(int)
        print(f"\nbin {b} NP {NP}: avg $/trade (rows F, cols S)")
        print(x.pivot(index="F", columns="S", values="avg"))
