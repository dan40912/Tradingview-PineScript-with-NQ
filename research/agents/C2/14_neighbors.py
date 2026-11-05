"""14: Neighbour table (each parameter one step up/down) for C2_TE_OUTSIDE and the contaminated NOI2 variant.
Discovery decides; validation column is information only (1 contract)."""
import pandas as pd
import c2lib as L
from lib import sim
import sig
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
grid = [("base", {}), ("k=0.6", {"k": 0.6}), ("k=0.9", {"k": 0.9}), ("s=0.35", {"s": 0.35}), ("s=0.7", {"s": 0.7}),
        ("start=0", {"start": 0}), ("start=60", {"start": 60}), ("end=265", {"end": 265}), ("end=385", {"end": 385}),
        ("exit 14:59", {"exit_idx": 329}), ("exit 15:59", {"exit_idx": 389})]
rows = []
for day in ["outside", "out_noI2"]:
    for lab, kw in grid + [("ADR5", "a5"), ("ADR15", "a15")]:
        l2 = lv
        if isinstance(kw, str):
            l2 = lv.copy(); l2["adr10"] = L.adr(g, int(kw[1:])); kw = {}
        tr = sim.run_signals(g, sig.trend_ext(g, l2, day=day, i2map=im, **kw))
        d, v = L.summ(g, tr, "discovery"), L.summ(g, tr, "validation")
        rows.append({"rule": day, "variant": lab, "d_n": d["n"], "d_avg": d["avg"], "d_sh": d["sharpe_ann"],
                     **{h: L.summ(g, tr, hal=h).get("avg") for h in L.HALVES}, "v_n": v["n"], "v_avg": v["avg"], "v_sh": v["sharpe_ann"]})
T = pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv("out_14_neighbors.csv", index=False)
