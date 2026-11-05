"""C3 step 5 (DISCOVERY ONLY): risk-parity mini-portfolios in micros (0.1 granularity).
Sizing rule (frozen here): each sleeve's size  w_k ∝ 1 / sd_k, sd_k = discovery daily P&L std of 1 contract
(all tradable days, flat=0); then the whole portfolio is scaled so its discovery daily std equals that of
1 NQ I2 v1 ("same risk budget"); each w rounded to 0.1 (MNQ / MES). Micro cost assumed = 1/10 of full:
$4.80 per MNQ, $10.80 per MES round trip (engine: pnl = (pts*point - cost) * contracts)."""
import sys, os, numpy as np, pandas as pd
from c3lib import data, features, levels, i2_exit_sigs, run, stats, tradable_days, daily, strategies, half_avgs
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "A3")); from candidates import c3_poc_sl60_np30
import importlib.util
def _load(agent):
    spec = importlib.util.spec_from_file_location(f"cand_{agent}", os.path.join(HERE, "..", agent, "candidates.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
B1 = _load("B1"); B2 = _load("B2")

gN = data.load_grid("NQ"); lvN = features.daily_levels(gN)
gE = data.load_grid("ES"); lvE = levels(gE, 5.0)
SL = {
    "NQ_I2v1": run(gN, strategies.i2_v1_signals(gN, lvN)),
    "NQ_A3C3": run(gN, c3_poc_sl60_np30(gN, lvN)),
    "NQ_B1volpoor": run(gN, B1.CANDIDATES["B1_I2_volpoor"](gN, lvN)),
    "NQ_B2trend": run(gN, B2.CANDIDATES["B2_I2_TrendSize"](gN, lvN)),
    "ES_40_15": run(gE, i2_exit_sigs(gE, lvE, tp=40, sl=15)),
    "ES_22_58(contam)": run(gE, i2_exit_sigs(gE, lvE, tp=22, sl=58)),
}
for k, v in SL.items():
    v["symbol"] = "ES" if k.startswith("ES") else "NQ"
D = sorted(set(tradable_days(gN, "discovery")) & set(tradable_days(gE, "discovery")))
sd = {k: daily(v, D).std(ddof=1) for k, v in SL.items()}
print("discovery daily sd per 1 contract:", {k: round(x) for k, x in sd.items()})
PORTS = {
    "P0 1xNQ_I2v1": ["NQ_I2v1"],
    "P0b 1xNQ_A3C3": ["NQ_A3C3"],
    "P1 NQ_I2v1+ES_40_15": ["NQ_I2v1", "ES_40_15"],
    "P2 NQ_A3C3+ES_40_15": ["NQ_A3C3", "ES_40_15"],
    "P2c NQ_A3C3+ES_22_58(contam)": ["NQ_A3C3", "ES_22_58(contam)"],
    "P3 NQ_A3C3+NQ_I2v1+ES_40_15": ["NQ_A3C3", "NQ_I2v1", "ES_40_15"],
    "P4 NQ_B1volpoor+ES_40_15": ["NQ_B1volpoor", "ES_40_15"],
    "P5 NQ_B2trend+ES_40_15": ["NQ_B2trend", "ES_40_15"],
}
target = sd["NQ_I2v1"]


def weights(names):
    raw = {k: 1.0 / sd[k] for k in names}
    comb = pd.concat([daily(SL[k], D) * raw[k] for k in names], axis=1).sum(1)
    s = target / comb.std(ddof=1)
    return {k: max(0.1, round(raw[k] * s, 1)) for k in names}


rows, W = [], {}
for p, names in PORTS.items():
    w = {names[0]: 1.0} if len(names) == 1 else weights(names)
    W[p] = w
    tr = pd.concat([SL[k].assign(pnl=SL[k].pnl * w[k]) for k in names])
    td = tr[tr.split == "discovery"]
    rows.append({"portfolio": p, "weights": " ".join(f"{k}:{v}" for k, v in w.items()), **stats(td, D), **half_avgs(td)})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.to_string(index=False))
df.to_csv("out_05_portfolio_discovery.csv", index=False)
pd.Series({p: w for p, w in W.items()}).to_json("out_05_weights.json")
