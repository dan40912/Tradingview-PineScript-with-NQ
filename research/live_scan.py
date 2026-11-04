"""盤前／盤中即時掃描：抓 Yahoo 最近 5 天 NQZ26、ESZ26 的 1 分K，判斷今天照組合 D 有沒有進場點。
用法：python3 live_scan.py [--offline DIR]
研究用途，不是投資建議；Yahoo 是免費、非官方資料，大約延遲 10 分鐘。"""
import sys, numpy as np, pandas as pd
import strat
from eng import value_area
from today_scan import load, NQ, ES
hh = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"
def tw(i):   # 美東 RTH 第 i 分鐘 → 台灣時間（用當天的時差）
    return (TODAY_ET + pd.Timedelta(minutes=570 + i)).tz_convert("Asia/Taipei").strftime("%H:%M")
def days(df):
    out, prev = {}, None
    for td in sorted(df.td.unique()):
        g = df[df.td == td]; r = g[(g.m >= 570) & (g.m <= 959)]
        if not len(r): continue
        A = np.full((5, 390), np.nan); i = (r.m.values - 570)
        for k, c in enumerate(("open", "high", "low", "close", "volume")): A[k, i] = r[c].values
        cur = dict(date=td, O=A[0], H=A[1], L=A[2], C=A[3], V=A[4])
        if prev is not None:
            val, vah, poc = value_area(prev["L"], prev["H"], prev["V"]); vv = np.nan_to_num(A[4]); tp = np.nan_to_num((A[1] + A[2] + A[3]) / 3)
            cur.update(val=val, vah=vah, poc=poc, pdh=np.nanmax(prev["H"]), pdl=np.nanmin(prev["L"]), vwap=np.cumsum(tp * vv) / np.maximum(np.cumsum(vv), 1), pr_rel=np.nan)
        out[td] = cur; prev = cur
    return out
off = sys.argv[sys.argv.index("--offline") + 1] if "--offline" in sys.argv else None
(nq, _), (es, _) = load(NQ, off), load(ES, off)
last_et = nq.et.iloc[-1]; today = nq.td.iloc[-1]
TODAY_ET = pd.Timestamp(today + " 00:00", tz="America/New_York")
print(f"資料最新：美東 {last_et:%m/%d %H:%M}（台灣 {last_et.tz_convert('Asia/Taipei'):%H:%M}），NQZ26 {nq.close.iloc[-1]:,.2f}")
N, E = days(nq), days(es)
d = N.get(today)
if d is None or np.isnan(d["O"][0]):
    print("\n美股還沒開盤 → 盤前掃描：")
    import subprocess; subprocess.run([sys.executable, "today_scan.py"] + (["--offline", off] if off else []))
    sys.exit()
O, H, L, C = d["O"], d["H"], d["L"], d["C"]; last = int(np.where(~np.isnan(C))[0].max()); o = O[0]; w = d["vah"] - d["val"]
side = -1 if o > d["vah"] else 1 if o < d["val"] else 0; dist = (o - d["vah"]) if side == -1 else (d["val"] - o) if side == 1 else 0
e = E.get(today); es_dir = 0
if e is not None and "val" in e and not np.isnan(e["O"][0]): es_dir = -1 if e["O"][0] > e["vah"] else 1 if e["O"][0] < e["val"] else 0
outpd = o > d["pdh"] or o < d["pdl"]
print(f"今天價值區 {d['val']:,.0f}～{d['vah']:,.0f}（寬 {w:.0f}）POC {d['poc']:,.0f}｜前日高 {d['pdh']:,.2f} 低 {d['pdl']:,.2f}")
print(f"09:30（台灣 {tw(0)}）開盤 {o:,.2f} → {'VAH 之上' if side == -1 else 'VAL 之下' if side == 1 else '價值區內'}"
      + (f"，離區 {dist:.0f} 點＝{dist / w:.2f} 倍區寬（H1 {'會做' if dist <= w else '跳過'}）" if side else "")
      + f"｜{'不在' if outpd else '在'}前日範圍內｜ES {'同向' if es_dir == side and side else '不同向' if side else '—'}")
print(f"目前（{hh(last)}，台灣 {tw(last)}）{C[last]:,.2f}｜盤中高 {np.nanmax(H):,.2f} 低 {np.nanmin(L):,.2f}｜VWAP {d['vwap'][last]:,.2f}")
if last >= 29: print(f"IB（09:30–09:59）高 {np.nanmax(H[:30]):,.2f} 低 {np.nanmin(L[:30]):,.2f} 中點 {(np.nanmax(H[:30]) + np.nanmin(L[:30])) / 2:,.2f}")
c5 = [(i, C[i]) for i in range(4, last + 1, 5) if not np.isnan(C[i])]
print("5 分K 收盤：" + "  ".join(f"{hh(i)} {c:,.0f}" for i, c in c5))
ESD = {today: es_dir}
R = [("D-1 I2（ES 同向）", lambda: strat.i2(d, last=89, tp=55, sl=145, xi=379, es=ESD), 379, lambda t: 55),
     ("D-2 順勢突破", lambda: strat.trend(d, ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0), 179, lambda t: 2 * t["risk"]),
     ("I2 v1（原規則，對照）", lambda: strat.i2(d), 179, lambda t: 55)]
print("\n■ 規則狀態")
for nm, f, cap, tpf in R:
    t = f()
    if t:
        en = O[t["ei"]]; s = t["side"]; tp = en + s * tpf(t); sl = en - s * t["risk"]
        live = t["res"] in ("TIME",) and t["xi"] >= last and t["xi"] < min(cap, 389)
        st = f"持倉中，目前浮動 {s * (C[last] - en) * 20 - 48:+,.0f}" if live else f"已出場：{t['res']} {hh(min(t['xi'], 389))}（台灣 {tw(min(t['xi'], 389))}）{t['pnl']:+,.0f}"
        print(f"  ✅ {nm}：{'做多' if s == 1 else '做空'} {hh(t['ei'])}（台灣 {tw(t['ei'])}）進 {en:,.2f}｜TP {tp:,.2f}｜SL {sl:,.2f}｜{st}")
        continue
    # 沒有訊號：說明在等什麼，或為什麼今天不適用
    if "I2" in nm:
        if side == 0: why = "開在價值區內 → 今天不做"
        elif "ES" in nm and dist > w: why = "離區超過 1 倍區寬（H1）→ 今天不做"
        elif "ES" in nm and es_dir != side: why = "ES 沒有同向偏離 → 今天不做"
        elif last >= (89 if "ES" in nm else 120): why = "已過最晚進場時間 → 今天沒有訊號"
        else:
            cnt = 0
            for _, c in c5: cnt = cnt + 1 if d["val"] <= c <= d["vah"] else 0
            why = f"等待：連續 2 根 5 分K 收在 {d['val']:,.0f}～{d['vah']:,.0f} 就{'做多' if side == 1 else '做空'}（目前連續 {cnt} 根；現價離區 {abs(C[last] - (d['val'] if side == 1 else d['vah'])):,.0f} 點）"
    else:
        g = -side
        if side == 0 or not outpd: why = "開在價值區內或前日範圍內 → 今天不適用"
        elif any(d["val"] <= c <= d["vah"] for _, c in c5 if _ < 150): why = "有 5 分K 收回價值區 → 今天作廢"
        elif last < 34: why = f"等 IB 完成（10:00，台灣 {tw(30)}），10:04 後 5 分K 收{'過 IB 高點' if g == 1 else '破 IB 低點'}就{'做多' if g == 1 else '做空'}"
        elif last >= 150: why = "已過 12:00 → 今天沒有訊號"
        else:
            lvl = np.nanmax(H[:30]) if g == 1 else np.nanmin(L[:30])
            why = f"等待：5 分K 收{'過' if g == 1 else '破'} {lvl:,.2f} 就{'做多' if g == 1 else '做空'}（現價離 {abs(C[last] - lvl):,.0f} 點）；SL＝IB 中點（30～80 點）、TP＝2 倍 SL"
    print(f"  ⏳ {nm}：{why}")
print("\n風控：同時只持 1 筆、最多 3 筆、連 2 虧或 −$3,000 停。I2：TP ±55／SL ∓145、最晚 10:59 進場、15:49 平倉；順勢：12:29 平倉。")
