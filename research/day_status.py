"""某一天（可以是盤中）每條規則的狀態：開盤位置、IB、5 分K、訊號、進場、TP／SL、結果或持倉中。
用法（在有 NQ_days2.pkl、ES_days2.pkl 的資料夾執行）：python3 day_status.py 2026-11-03"""
import sys, pickle, numpy as np, grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb")); date = sys.argv[1]
d = [x for x in nq if x["date"] == date][0]; O, H, L, C = d["O"], d["H"], d["L"], d["C"]
hh = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"; last = int(np.where(~np.isnan(C))[0].max())
w = d["vah"] - d["val"]; o = O[0]; pos = "VAH 之上" if o > d["vah"] else "VAL 之下" if o < d["val"] else "價值區內"
dist = (o - d["vah"]) if o > d["vah"] else (d["val"] - o) if o < d["val"] else 0
print(f"{date}（資料到 {hh(last)} 美東）｜VA {d['val']:,.0f}–{d['vah']:,.0f}（寬 {w:.0f}）POC {d['poc']:,.0f}｜前日高低 {d['pdh']:,.2f}／{d['pdl']:,.2f}")
print(f"開盤 {o:,.2f} → {pos}" + (f"，離區 {dist:.0f} 點（{'≤' if dist <= w else '>'} 區寬 → H1 {'會做' if dist <= w else '跳過'}）" if dist else "")
      + f"｜{'在' if d['pdl'] <= o <= d['pdh'] else '不在'}前日範圍內｜ES 開盤方向 {grid.ES.get(date, 0):+d}（+1＝ES 開在區下方、−1＝上方）")
if last >= 29: print(f"IB（09:30–09:59）高 {np.nanmax(H[:30]):,.2f} 低 {np.nanmin(L[:30]):,.2f}")
print(f"目前 {C[last]:,.2f}｜盤中高 {np.nanmax(H):,.2f} 低 {np.nanmin(L):,.2f}｜VWAP {d['vwap'][last]:,.2f}")
print("5 分K 收盤：" + "  ".join(f"{hh(i)} {C[i]:,.0f}" for i in range(4, last + 1, 5) if not np.isnan(C[i])))
R = {"I2 v1（現在指標）": (lambda: strat.i2(d), 179), "前高空（只看）": (lambda: strat.pdfade(d, 0), 179), "前低多（只看）": (lambda: strat.pdfade(d, 1), 179),
     "D-1 I2（ES 同向）": (lambda: strat.i2(d, last=89, tp=55, sl=145, xi=379, es=grid.ES), 379),
     "D-2 順勢突破": (lambda: strat.trend(d, ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0), 179)}
print()
for nm, (f, cap) in R.items():
    t = f()
    if not t: print(f"  {nm:<14} 沒有訊號"); continue
    e = O[t["ei"]]; s = t["side"]; tp = e + s * (55 if "順勢" not in nm else 2 * t["risk"]); sl = e - s * t["risk"]
    live = t["res"] == "TIME" and t["xi"] < min(cap, 389) and t["xi"] >= last
    out = f"持倉中（目前 {C[last]:,.2f}，浮動 {s * (C[last] - e) * 20 - 48:+,.0f}）" if live else f"{t['res']} {hh(min(t['xi'], 389))} {t['pnl']:+,.0f}"
    print(f"  {nm:<14} {'多' if s == 1 else '空'} {hh(t['ei'])} 進 {e:,.2f}｜TP {tp:,.2f}｜SL {sl:,.2f}｜{out}")
