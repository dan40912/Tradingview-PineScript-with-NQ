"""Stage A (discovery only): B1 indicators as filters on I2 v1 trades.
Each feature is oriented so that larger = 'supports reversion' is NOT assumed; we test both
'drop low tercile' and 'drop high tercile' (thresholds from discovery day/trade distribution)."""
from common import *
import json

g, lv = load("NQ")
ges, lves = load("ES")
pf = ind.profile_features(g, lv)
esf = ind.es_features(g, ges, lves)
ia = ind.intraday_arrays(g)
tr = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
rV = g.rth(g.V)

def trade_feats(tr):
    rows = []
    for t in tr.itertuples():
        i, d, es = t.i, t.dir, t.entry_slot
        p = pf.loc[i]; e = esf.loc[i]; r = lv.loc[i]
        f = {}
        o = r.open_0930
        # --- day-level (pre-open) oriented by trade direction: 'toward open side' = -d
        f["va_w_atr"] = p.va_w_atr
        f["rng_atr"] = p.rng_atr
        f["poc_toward_open"] = p.poc_pos if d == -1 else 1 - p.poc_pos
        f["close_toward_open"] = p.pd_close_pos if d == -1 else 1 - p.pd_close_pos
        f["poor_on_open_side"] = p.poor_high if d == -1 else p.poor_low
        f["tail_open_side_pts"] = p.tail_hi_pts if d == -1 else p.tail_lo_pts
        f["single_prints_pts"] = p.single_prints_pts
        f["mig_with_open"] = p.va_migration * (-d)
        f["poc_shift_with_open"] = p.poc_shift_atr * (-d)
        f["onva_with_open"] = p.on_va_rel * (-d)
        f["open_out_onva_same"] = int(p.open_vs_onva == -d)
        f["on_range_atr"] = p.on_range_atr
        f["rvol_on"] = p.rvol_on
        f["rvol_prev_rth"] = p.rvol_prev_rth
        f["on_delta_dir"] = p.on_delta_frac * d
        f["pd_delta_dir"] = p.pd_delta_frac * d
        for k in (2, 3, 5):
            pos = p[f"c{k}_open_pos"]
            f[f"open_in_c{k}"] = int(0 <= pos <= 1) if pos == pos else np.nan
        f["prev_poc_naked"] = p.prev_poc_naked
        nd = p.naked_up_dist if d == 1 else p.naked_dn_dist
        f["naked_poc_ahead_300"] = int(nd == nd and nd <= 300)
        f["open_dist_w"] = ((o - r.vah) if d == -1 else (r.val - o)) / r.va_width
        f["es_same_dir"] = int(e.es_dir == r.i2_dir)
        f["rs_on_dir"] = e.rs_on * (-d)     # NQ stronger in open direction
        # --- entry-time (bars < es)
        s = es - 1
        f["or_rvol5"] = p.or_rvol5
        f["px_vs_vwap18_sd"] = (g.C[i, s] - ia["vwap18"][i, s]) * d / ia["sd18"][i, s]
        f["px_vs_vwap930_sd"] = (g.C[i, s] - ia["vwap930"][i, s]) * d / max(ia["sd930"][i, s], 1e-9)
        f["cd930_dir"] = ia["cd930"][i, s] * d / max(np.nansum(g.V[i, RTH_OFF:es]), 1)
        f["cd18_dir"] = ia["cd18"][i, s] * d / max(np.nansum(g.V[i, :es]), 1)
        # delta divergence: price moved against d since open but delta proxy with d
        f["dlt_div"] = int(((g.C[i, s] - o) * d < 0) and f["cd930_dir"] > 0)
        dv = ind.developing_va(g, i, es)
        f["dpoc_ahead"] = ((dv[1] + 2.5) - g.O[i, es]) * d
        f["entry_in_dva"] = int(dv[0] <= g.O[i, es] <= dv[2])
        # volume so far vs 20d avg of same window
        prevs = []; j = i
        while len(prevs) < 20:
            j = g.prev_rth_day(j)
            if j is None: break
            if g.tradable[j]: prevs.append(j)
        w = es - RTH_OFF
        f["rvol_to_entry"] = np.nansum(rV[i, :w]) / np.nanmean(np.nansum(rV[prevs, :w], 1)) if len(prevs) >= 10 else np.nan
        # climax: max 1m volume in opening window / median of window
        vv = rV[i, :w]
        f["climax_ratio"] = np.nanmax(vv) / np.nanmedian(vv) if np.isfinite(vv).sum() > 3 else np.nan
        f["entry_min"] = w
        rows.append(f)
    return pd.DataFrame(rows, index=tr.index)

if __name__ == "__main__":
    F = trade_feats(tr)
    T = pd.concat([tr, F], axis=1)
    T.to_pickle(os.path.join(HERE, "_i2_trades_feats.pkl"))
    D = T[T.split == "discovery"].reset_index(drop=True)
    pnl = D.pnl.values
    base = pnl.mean()
    res = []
    for c in F.columns:
        if c == "entry_min": continue
        x = D[c].values.astype(float)
        vals = np.unique(x[np.isfinite(x)])
        if len(vals) <= 2:
            rules = [(f"{c}==0", x == 0), (f"{c}==1", x == 1)] if set(vals) <= {0, 1} else []
            if not rules:  # e.g. migration -2..2
                pass
        else:
            q1, q2 = np.nanquantile(x, [1/3, 2/3])
            rules = [(f"{c}>={q1:.3g}", x >= q1), (f"{c}<={q2:.3g}", x <= q2)]
        if c in ("mig_with_open", "onva_with_open"):
            rules = [(f"{c}<=0", x <= 0), (f"{c}>=0", x >= 0), (f"{c}<2", x < 2), (f"{c}>-2", x > -2)]
        for name, m in rules:
            m = np.nan_to_num(m.astype(float)).astype(bool)
            k = m.sum()
            if k < 10 or k == len(m): continue
            res.append({"rule": name, "kept": k, "keep%": round(100 * k / len(m)), "avg_kept": round(pnl[m].mean()),
                        "avg_drop": round(pnl[~m].mean()), "win_kept": round(100*(pnl[m]>0).mean(),1),
                        "p": perm_p(pnl, m)})
    R = pd.DataFrame(res).sort_values("p")
    print("discovery I2 trades", len(D), "base avg", round(base))
    print("n rules tested:", len(R), " expected p<0.05 by luck:", round(0.05 * len(R), 1))
    print(R.to_string(index=False))
    R.to_csv(os.path.join(HERE, "filter_stageA.csv"), index=False)
