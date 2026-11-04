"""畫一天的成交量分布（價值區怎麼算出來）＋隔天到目前的走勢。用法：python3 va_chart.py Yahoo合併CSV 日期 輸出PNG"""
import sys, math, logging, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
from eng import BIN, VA_FRAC
src, day, out = sys.argv[1:4]
y = pd.read_csv(src); et = pd.to_datetime(y.timestamp_utc).dt.tz_convert("America/New_York"); y["et"] = et; y["m"] = et.dt.hour * 60 + et.dt.minute
r = y[(y.trade_date == day) & (y.m >= 570) & (y.m <= 959)]
prof = {}
for l, h, v in zip(r.low, r.high, r.volume):
    a, b = int(math.floor(l / BIN)), int(math.floor(h / BIN))
    for k in range(a, b + 1): prof[k] = prof.get(k, 0.0) + v / (b - a + 1)
keys = sorted(prof); total = sum(prof.values()); poc = max(keys, key=lambda k: (prof[k], -k)); lo = hi = poc; acc = prof[poc]
while acc < VA_FRAC * total and (lo > keys[0] or hi < keys[-1]):
    lv, hv = prof.get(lo - 1, -1.0), prof.get(hi + 1, -1.0)
    if hv > lv: hi += 1; acc += max(0, hv)
    else: lo -= 1; acc += max(0, lv)
val, vah, pocp = lo * BIN, (hi + 1) * BIN, poc * BIN
after = y[(y.et > r.et.iloc[-1])]
seg = pd.concat([r, after]).set_index("et").resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
fig = plt.figure(figsize=(13, 7), dpi=110); ax = fig.add_axes([0.06, 0.1, 0.62, 0.8]); ap = fig.add_axes([0.70, 0.1, 0.18, 0.8], sharey=ax)
for i, (t, row) in enumerate(seg.iterrows()):
    rth = t <= r.et.iloc[-1]; c = ("#2e9e5b" if row.close >= row.open else "#d64545") if rth else ("#7fc79c" if row.close >= row.open else "#e99a9a")
    ax.plot([i, i], [row.low, row.high], color=c, lw=0.8); ax.add_patch(Rectangle((i - 0.35, min(row.open, row.close)), 0.7, max(abs(row.close - row.open), 0.5), color=c))
n_rth = (seg.index <= r.et.iloc[-1]).sum(); ax.axvline(n_rth - 0.5, color="#888", ls="--", lw=1)
ax.text(n_rth / 2, 0.98, f"{day} 美股盤中（用來算價值區）", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
ax.text(n_rth + (len(seg) - n_rth) / 2, 0.98, "收盤後～現在（夜盤）", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, color="#555")
for lv, txt, c in ((vah, "VAH", "#e08a1e"), (pocp, "POC", "#555"), (val, "VAL", "#1a9c9c"), (r.high.max(), "前日高", "#d64545"), (r.low.min(), "前日低", "#2e9e5b")):
    ax.axhline(lv, color=c, lw=1.6 if txt in ("VAH", "VAL") else 1, ls="-" if txt in ("VAH", "VAL", "POC") else ":")
    ax.text(len(seg) + 0.3, lv, f"{txt} {lv:,.0f}", va="center", fontsize=9, color=c)
ax.axhspan(val, vah, color="#f2c94c", alpha=0.18)
ticks = list(range(0, len(seg), 8)); ax.set_xticks(ticks); ax.set_xticklabels([seg.index[i].strftime("%m/%d %H:%M") for i in ticks], fontsize=8, rotation=30)
ax.set_xlim(-1, len(seg) + 6); ax.set_ylabel("NQZ26"); ax.grid(alpha=0.2)
ax.set_title(f"價值區怎麼算：{day} 美股盤中每分鐘的成交量，平均分到 5 點一格 → 成交最多的格＝POC → 往上下擴張到包住 70% 成交量", fontsize=11)
ks = np.array(keys) * BIN; vs = np.array([prof[k] for k in keys]); cols = ["#f2994a" if val <= p < vah else "#bbb" for p in ks]; cols[keys.index(poc)] = "#333"
ap.barh(ks + BIN / 2, vs, height=BIN * 0.9, color=cols); ap.set_xlabel("成交量（口）"); ap.tick_params(labelleft=False); ap.grid(alpha=0.2)
ap.set_title(f"成交量分布\n橘色＝價值區（70%）\n黑色＝POC", fontsize=9)
lo_ = min(seg.low.min(), val) - 30; hi_ = max(seg.high.max(), vah) + 30; ax.set_ylim(lo_, hi_)
fig.savefig(out); print("VA", val, vah, "POC", pocp, "→", out)
