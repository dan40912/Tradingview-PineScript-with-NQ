"""Reference strategy implementations (baseline). Each returns a list of signal dicts for sim.run_signals."""
from __future__ import annotations
import numpy as np
from .data import RTH_OFF


def i2_v1_signals(g, lv, last_entry_idx=120, tp=55.0, sl=145.0, exit_idx=179, h1=False):
    """Frozen I2 v1. lv = features.daily_levels(g)."""
    sigs = []
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir == 0 or np.isnan(r.open_0930):
            continue
        if h1 and ((r.i2_dir == -1 and r.open_0930 - r.vah > r.va_width) or (r.i2_dir == 1 and r.val - r.open_0930 > r.va_width)):
            continue
        i = r.i
        C = g.C[i]
        cnt = 0
        for idx in range(4, 120, 5):
            c = C[RTH_OFF + idx]
            if np.isnan(c):
                continue
            cnt = cnt + 1 if (r.val <= c <= r.vah) else 0
            if cnt >= 2:
                ei = idx + 1
                if not ((28 <= ei <= 32) or ei > last_entry_idx):
                    sigs.append({"i": i, "entry_slot": RTH_OFF + ei, "dir": int(r.i2_dir), "tp_pts": tp, "sl_pts": sl,
                                 "exit_slot": RTH_OFF + exit_idx, "tag": "I2v1"})
                break
    return sigs
