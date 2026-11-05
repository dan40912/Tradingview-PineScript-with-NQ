"""MAE/MFE anatomy of I2 v1 entries (discovery only). Descriptive; no parameters chosen here."""
import numpy as np, pandas as pd
from a3lib import load, i2_entries, entry_px, run, stats, RTH, EXIT_1229

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
base = run(g, sigs)
rows = []
for s, (_, t) in zip(sigs, base.iterrows()):
    i, d = s["i"], s["dir"]
    e = t.entry_slot
    px = t.entry
    H, L, C = g.H[i], g.L[i], g.C[i]
    fav = (H[e:EXIT_1229 + 1] - px) if d == 1 else (px - L[e:EXIT_1229 + 1])
    adv = (px - L[e:EXIT_1229 + 1]) if d == 1 else (H[e:EXIT_1229 + 1] - px)
    favc, advc = np.fmax.accumulate(np.nan_to_num(fav, nan=-1e9)), np.fmax.accumulate(np.nan_to_num(adv, nan=-1e9))
    def first(arr, x):
        k = np.where(arr >= x)[0]
        return int(k[0]) if len(k) else 10**6
    close1229 = d * (C[EXIT_1229] - px) if not np.isnan(C[EXIT_1229]) else np.nan
    eod = C[RTH + 389] if not np.isnan(C[RTH + 389]) else C[RTH + 385]
    rows.append({"date": t.date, "dir": d, "entry_time": t.entry_time, "reason": t.reason, "pnl": t.pnl,
                 "mfe": np.nanmax(fav), "mae": np.nanmax(adv), "t55": first(favc, 55), "t145": first(advc, 145),
                 "mae_before_tp": np.nanmax(adv[:first(favc, 55) + 1]) if first(favc, 55) < 10**6 else np.nan,
                 "mfe30": np.nanmax(fav[:30]), "mfe60": np.nanmax(fav[:60]), "mae30": np.nanmax(adv[:30]),
                 "close1229": close1229, "eod": d * (eod - px), "va_w": lv.va_width[i], "atr": lv.atr14[i],
                 "dist_opp": (lv.vah[i] - px) if d == 1 else (px - lv.val[i]), "dist_poc": d * (lv.poc[i] - px)})
df = pd.DataFrame(rows)
df.to_csv("out_mae_mfe_discovery.csv", index=False)
pd.set_option("display.width", 200)
print("n", len(df))
print(df.groupby("reason")[["pnl", "mfe", "mae", "mfe30", "mfe60", "mae30", "close1229"]].agg(["count", "median", "mean"]).round(1).T)
print("\nMFE quantiles (pts):", df.mfe.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
print("MAE quantiles (pts):", df.mae.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
print("\nP(MFE>=x) :", {x: round((df.mfe >= x).mean(), 2) for x in [20, 30, 40, 55, 70, 85, 100, 130, 160, 200]})
print("P(MAE>=x) :", {x: round((df.mae >= x).mean(), 2) for x in [20, 30, 45, 60, 80, 100, 120, 145, 175]})
w = df[df.t55 < 10**6]
print("\nTP55 trades: MAE before TP quantiles", w.mae_before_tp.quantile([.5, .75, .9, .95, 1]).round(1).to_dict())
print("time-exit trades (12:29): close1229 quantiles", df[df.reason == "time"].close1229.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
# MFE60 as predictor of outcome
for X in [10, 20, 30]:
    a = df[df.mfe30 < X]; b = df[df.mfe30 >= X]
    print(f"mfe in first 30min <{X}: n={len(a)} avg pnl={a.pnl.mean():.0f} | >= : n={len(b)} avg={b.pnl.mean():.0f}")
for X in [10, 20, 30, 40]:
    a = df[df.mfe60 < X]; b = df[df.mfe60 >= X]
    print(f"mfe in first 60min <{X}: n={len(a)} avg pnl={a.pnl.mean():.0f} | >= : n={len(b)} avg={b.pnl.mean():.0f}")
print("\nby entry time bucket")
df["bucket"] = df.entry_time.str[:4]
print(df.groupby(df.entry_time.str[:2] + ":" + (df.entry_time.str[3].astype(int) >= 3).map({True: "30", False: "00"}))
      .agg(n=("pnl", "size"), avg=("pnl", "mean"), mfe=("mfe", "median"), mae=("mae", "median")).round(0))
print("\ndist to opposite VA edge (pts) quantiles", df.dist_opp.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
print("dist to POC quantiles", df.dist_poc.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
print("VA width quantiles", df.va_w.quantile([.1, .25, .5, .75, .9]).round(0).to_dict(), "ATR14", df.atr.quantile([.1, .5, .9]).round(0).to_dict())
print("corr mfe vs va_w", df[["mfe", "mae", "va_w", "atr"]].corr().round(2))
