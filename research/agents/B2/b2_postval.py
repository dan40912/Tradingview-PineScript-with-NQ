"""B2: post-validation diagnostics (no parameter changes). Random-halving permutation for TrendSize."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
o = pd.read_pickle(SP + "val_B2_ORB30_830.pkl")
print(o[o.split == "validation"][["date", "dir", "entry_time", "exit_time", "reason", "pts", "pnl"]].to_string())
ts = attach(pd.read_pickle(SP + "val_B2_I2_TrendSize.pkl"), rf, ev)
ts["pnl1"] = ts.pts * 20 - 48; ts["contracts"] = (ts.pnl / ts.pnl1).round(2); ts["against"] = ts.contracts < 1
print(ts.groupby(["split", "against"]).pnl1.agg(["count", "mean"]).round(0))
rng = np.random.default_rng(7)
for sp in ["discovery", "validation"]:
    t = ts[ts.split == sp].reset_index(drop=True); nd = ndays(g, sp)
    obs = metrics.summary(t.assign(pnl=t.pnl), n_days=nd)
    k = int(t.against.sum()); R = []; S = []; D = []
    for _ in range(3000):
        w = np.ones(len(t)); w[rng.choice(len(t), k, replace=False)] = 0.5
        s = metrics.summary(t.assign(pnl=t.pnl1 * w), n_days=nd); R.append(s["ret/dd"]); S.append(s["sharpe_ann"]); D.append(s["maxdd"])
    R, S, D = map(np.array, (R, S, D))
    print(sp, "obs ret/dd", obs["ret/dd"], "p", round((np.sum(R >= obs["ret/dd"]) + 1) / 3001, 3),
          "| sharpe", obs["sharpe_ann"], "p", round((np.sum(S >= obs["sharpe_ann"]) + 1) / 3001, 3),
          "| maxdd", obs["maxdd"], "p", round((np.sum(D >= obs["maxdd"]) + 1) / 3001, 3), "random median dd", np.median(D))
# post-hoc (NOT a candidate): gap fade split by open outside prior VA (known at 09:30)
gf = attach(pd.read_pickle(SP + "val_B2_GAPF_NoNews.pkl"), rf, ev)
gf["outVA"] = lv.i2_dir.values[gf.i] != 0
print(gf.groupby(["split", "outVA"]).pnl.agg(["count", "mean"]).round(0))
