"""Phase D analysis of the frozen books on 2024-01..2025-12 ONLY (never loads 2026).
Run: cd research/agents/D && python analyze.py   -> out_*.csv, out_analyze.txt"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import portfolios as P

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
ctx = P.get_ctx(False)
days = ctx.trading_days()
assert days.max() < "2026-01-01"
LOG = open("out_analyze.txt", "w")
def say(*a):
    print(*a); print(*a, file=LOG)

BENCH = {"I2v1": ("I2v1", 2900.0)}          # 1 NQ I2 v1 (stop 145 pts = $2,900)
def run(spec, heat=P.HEAT):
    if spec is BENCH:                       # benchmark: plain 1 NQ I2 v1, no book rules
        return P.assemble(ctx, spec, heat_mult=1.0, risk_ref=2900.0)
    return P.assemble(ctx, spec, heat_mult=heat, risk_ref=P.R)

def full_stats(tr, label):
    dv = P.daily_pnl(tr, days)
    s = P.stats(dv, tr)
    b = P.boot_maxdd(dv, n=10000)
    s.update({"bootDD50": round(np.quantile(b, .5)), "bootDD95": round(np.quantile(b, .05)),
              "bootDD99": round(np.quantile(b, .01)), "P(DD1y<-8224)": round(float((b < -8224).mean()), 3)})
    s["book"] = label
    return s, dv

# ---------------------------------------------------------------- 1. sleeves: stats + correlation (at R=$1,000)
SLEEVES = {"I2v1": "I2v1", "A3_C3": P.A3, "C1_I2_A3X_TS": "C1/C1_I2_A3X_TS", "C1_ORB15_POC": "C1/C1_ORB15_POC_A3X_TS",
           "C1_ORB15_TP110": P.ORBT, "B1_volpoor": "B1/B1_I2_volpoor", "B2_TrendSize": "B2/B2_I2_TrendSize",
           "C2_TE_OUTSIDE": P.TE, "C2_TE_NOI2*": "C2/C2_TE_OUTSIDE_NOI2", "A2_TREND_EXT": "A2/A2_TREND_EXT_FOLLOW",
           "B3_NQES_RS": "B3/B3_NQES_RS_0945_T120", "A1_OOV_ORB15": "A1/OOV_ORB15_EOD", "C3_ES_40_15": P.ES}
rows, D = [], {}
for lab, key in SLEEVES.items():
    tr = P.size(ctx.raw(key), P.R)
    dv = P.daily_pnl(tr, days); D[lab] = dv
    s = P.stats(dv, tr); s["sleeve"] = lab; s["avg_contracts"] = round(tr.contracts.mean(), 2)
    rows.append(s)
st = pd.DataFrame(rows).set_index("sleeve")
st.to_csv("out_sleeves_R1000.csv")
say("== sleeves sized at $1,000 stop-risk per trade, 2024-01..2025-12 ==\n", st.to_string())
corr = pd.DataFrame(D).corr().round(2)
corr.to_csv("out_corr.csv")
say("\n== daily P&L correlation (flat days = 0) ==\n", corr.to_string())

# ---------------------------------------------------------------- 2. books vs benchmark
books = {"I2v1 (1 NQ)": BENCH, **P.BOOK_SPECS}
rows, DV, TR = [], {}, {}
for name, spec in books.items():
    tr = run(spec); s, dv = full_stats(tr, name); rows.append(s); DV[name] = dv; TR[name] = tr
bt = pd.DataFrame(rows).set_index("book")
bt.to_csv("out_books.csv")
say("\n== books 2024-01..2025-12 (account $50k, R=$1,000) ==\n", bt.to_string())

# ---------------------------------------------------------------- 3. per half-year
rows = []
for name, dv in DV.items():
    for h, x in dv.groupby(dv.index.map(P.half_of)):
        s = P.stats(x); s.update({"book": name, "half": h,
                                  "n": int((TR[name].date.map(P.half_of) == h).sum())}); rows.append(s)
hy = pd.DataFrame(rows)
hy.to_csv("out_halfyear.csv", index=False)
say("\n== per half-year ==\n", hy.pivot(index="book", columns="half", values="total").to_string())
say(hy.pivot(index="book", columns="half", values="maxdd").to_string())
say(hy.pivot(index="book", columns="half", values="sharpe").to_string())
# split (discovery / validation)
rows = []
for name, dv in DV.items():
    for sp, x in [("discovery", dv[dv.index <= "2025-06-30"]), ("validation", dv[dv.index > "2025-06-30"])]:
        s = P.stats(x); s.update({"book": name, "split": sp}); rows.append(s)
say("\n== by split ==\n", pd.DataFrame(rows).set_index(["book", "split"]).to_string())

# ---------------------------------------------------------------- 4. sleeve contributions, contracts, conflicts
rows = []
for name, spec in P.BOOK_SPECS.items():
    tr = TR[name]
    rawn = sum(len(P.size(ctx.raw(k), r)) for k, r in spec.values())
    for lab, g in tr.groupby("sleeve"):
        key, r = spec[lab]
        nraw = len(P.size(ctx.raw(key), r))
        rows.append({"book": name, "sleeve": lab, "risk_usd": r, "trades": len(g), "skipped_or_cut":
                     nraw - len(g) + int((g.contracts < P.size(ctx.raw(key), r).set_index("date").loc[g.date, "contracts"].values).sum()),
                     "contracts_min": g.contracts.min(), "contracts_med": g.contracts.median(),
                     "contracts_max": g.contracts.max(), "total": round(g.pnl.sum()),
                     "share%": round(100 * g.pnl.sum() / tr.pnl.sum(), 1)})
ct = pd.DataFrame(rows)
ct.to_csv("out_contrib.csv", index=False)
say("\n== sleeve contributions / sizes / conflicts ==\n", ct.to_string(index=False))

# ---------------------------------------------------------------- 5. weight perturbation
def beats(s, b):
    return (s["total"] > b["total"] and s["maxdd"] > b["maxdd"] and s["ret/dd"] > b["ret/dd"] and
            s["sharpe"] > b["sharpe"] and s["worst_day"] > b["worst_day"] and s["worst_month"] > b["worst_month"])
b0 = bt.loc["I2v1 (1 NQ)"]
rows = []
for name, spec in P.BOOK_SPECS.items():
    labs = list(spec)
    grids = [(m,) * 0 for m in []]
    # (a) one sleeve at a time x0.5 / x1.5 ; (b) full grid {0.75,1,1.25}^k ; (c) global R scale
    combos = []
    for lab in labs:
        for m in (0.5, 1.5):
            combos.append(("1at", {l: (m if l == lab else 1.0) for l in labs}))
    for ms in itertools.product((0.75, 1.0, 1.25), repeat=len(labs)):
        combos.append(("grid", dict(zip(labs, ms))))
    for m in (0.5, 0.75, 1.25, 1.5):
        combos.append(("global", {l: m for l in labs}))
    for kind, mult in combos:
        sp = {l: (spec[l][0], spec[l][1] * mult[l]) for l in labs}
        tr = run(sp); dv = P.daily_pnl(tr, days); s = P.stats(dv, tr)
        s.update({"book": name, "kind": kind, "mult": " ".join(f"{l}x{mult[l]}" for l in labs), "beats_I2_all": beats(s, b0)})
        rows.append(s)
pt = pd.DataFrame(rows)
pt.to_csv("out_perturb.csv", index=False)
say("\n== weight perturbation: one-at-a-time and global ==\n",
    pt[pt.kind != "grid"][["book", "kind", "mult", "total", "maxdd", "ret/dd", "sharpe", "worst_day", "worst_month", "beats_I2_all"]].to_string(index=False))
agg = pt[pt.kind == "grid"].groupby("book").agg(n=("total", "size"), total_min=("total", "min"), total_med=("total", "median"),
      total_max=("total", "max"), dd_min=("maxdd", "min"), dd_med=("maxdd", "median"), sharpe_min=("sharpe", "min"),
      sharpe_med=("sharpe", "median"), sharpe_max=("sharpe", "max"), wm_min=("worst_month", "min"), beats_all=("beats_I2_all", "sum"))
say("\n== weight grid {0.75,1,1.25}^k per sleeve ==\n", agg.to_string())

# ---------------------------------------------------------------- 6. stress / ablations
rows = []
def add(name, label, tr):
    dv = P.daily_pnl(tr, days); s = P.stats(dv, tr); s.update({"book": name, "test": label}); rows.append(s)
for name, spec in P.BOOK_SPECS.items():
    tr = TR[name]
    add(name, "as frozen", tr)
    t2 = tr.copy(); t2["pnl"] = t2.pnl - t2.contracts * np.where(t2.symbol == "ES", 108.0, 48.0); add(name, "cost x2", t2)
    t3 = tr.copy(); t3["pnl"] = t3.pnl - t3.contracts * np.where(t3.symbol == "ES", 2 * 12.5, 2 * 5.0); add(name, "+2 ticks slip/side", t3)
    add(name, "no conflict rules", run(spec, heat=99))
    if "ES" in spec:
        add(name, "drop ES leg", run({k: v for k, v in spec.items() if k != "ES"}))
        t4 = tr[~((tr.sleeve == "ES") & (tr.date > "2025-06-30"))]; add(name, "ES leg flat in 2025H2", t4)
    if "TE" in spec:
        add(name, "drop TE leg", run({k: v for k, v in spec.items() if k != "TE"}))
    if "A3" in spec:
        add(name, "A3 -> I2v1 exits (same R)", run({("I2" if k == "A3" else k): (("I2v1", v[1]) if k == "A3" else v) for k, v in spec.items()}))
    m = tr.assign(m=tr.date.str[:7]).groupby("m").pnl.sum(); top = m.idxmax()
    add(name, f"drop best month ({top})", tr[tr.date.str[:7] != top])
    t5 = tr.sort_values("pnl"); add(name, "drop best 5 trades", t5.iloc[:-5])
stt = pd.DataFrame(rows)
stt.to_csv("out_stress.csv", index=False)
say("\n== stress / ablations ==\n", stt[["book", "test", "n", "total", "maxdd", "ret/dd", "sharpe", "worst_day", "worst_month"]].to_string(index=False))

# ---------------------------------------------------------------- 7. bootstrap detail
rows = []
for name, dv in DV.items():
    for blk in (1, 5, 20):
        b = P.boot_maxdd(dv, n=10000, block=blk, seed=1)
        rows.append({"book": name, "block_days": blk, "DD50": round(np.quantile(b, .5)), "DD95": round(np.quantile(b, .05)),
                     "DD99": round(np.quantile(b, .01)), "P(DD<-8224)": round(float((b < -8224).mean()), 3),
                     "P(DD<-5000)": round(float((b < -5000).mean()), 3)})
    # 1-year P&L distribution
bs = pd.DataFrame(rows); bs.to_csv("out_boot.csv", index=False)
say("\n== 1-year (252 d) block-bootstrap maxDD ==\n", bs.to_string(index=False))
rng = np.random.default_rng(2)
rows = []
for name, dv in DV.items():
    x = dv.values; idx = rng.integers(0, len(x), size=(10000, 252)); tot = x[idx].sum(1)
    rows.append({"book": name, "1y_P5": round(np.quantile(tot, .05)), "1y_P50": round(np.quantile(tot, .5)),
                 "P(1y<0)": round(float((tot < 0).mean()), 3)})
say("\n== 1-year P&L (iid bootstrap) ==\n", pd.DataFrame(rows).to_string(index=False))
monthly = pd.DataFrame({k: v.groupby(v.index.str[:7]).sum().round() for k, v in DV.items()})
monthly.to_csv("out_monthly.csv")
say("\n== monthly ==\n", monthly.to_string())
for name, tr in TR.items():
    tr.to_csv(f"trades_{name.split(' ')[0]}.csv", index=False)
LOG.close()
