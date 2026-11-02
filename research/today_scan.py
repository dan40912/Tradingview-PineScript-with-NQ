"""今天的 NQ 盤前／盤中掃描：價值區、前日高低、夜盤位置、ES 同向、開盤後會走哪條規則（附 TP／SL）。
用法：python3 today_scan.py               # 直接抓 Yahoo（需要能連 query1.finance.yahoo.com）
      python3 today_scan.py --offline DIR  # 讀已下載的 DIR/NQZ26.CME_*.json、DIR/ESZ26.CME_*.json
研究用途，不是投資建議；Yahoo 是免費、非官方資料，可能延遲或有錯誤報價。"""
import sys, json, glob, urllib.request, numpy as np, pandas as pd
from eng import value_area
NQ, ES = "NQZ26.CME", "ESZ26.CME"
def load(sym, offline=None):
    if offline:
        raws = [json.load(open(f)) for f in sorted(glob.glob(f"{offline}/{sym}_*.json"))]
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&range=5d&includePrePost=true"
        raws = [json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))]
    parts = []
    for raw in raws:
        r = raw["chart"]["result"][0]; q = r["indicators"]["quote"][0]
        parts.append(pd.DataFrame({"t": r["timestamp"], **{k: q[k] for k in ("open", "high", "low", "close", "volume")}}))
    df = pd.concat(parts).drop_duplicates("t").sort_values("t").dropna(subset=["open", "high", "low", "close"])
    df = df[df.volume.fillna(0) > 0].reset_index(drop=True); pc = df.close.shift(1)
    bad = ((df.open - pc).abs() > 0.02 * pc) & ((df.close - pc).abs() < 0.005 * pc)      # 錯誤報價（像 10/30 15:03 那根）
    df = df[~bad]
    et = pd.to_datetime(df.t, unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(et=et, m=et.dt.hour * 60 + et.dt.minute,
                   td=(et + pd.to_timedelta(np.where(et.dt.hour >= 18, 1, 0), unit="D")).dt.date.astype(str))
    return df, int(bad.sum())
def levels(df):
    days = sorted(df.td.unique()); today = days[-1]
    rth = df[(df.m >= 570) & (df.m <= 959)]
    prev = rth[rth.td < today].td.max(); p = rth[rth.td == prev]
    val, vah, poc = value_area(p.low.values, p.high.values, p.volume.values)
    t = df[df.td == today]; on = t[t.m < 570] if len(t) else t
    return dict(today=today, prev=prev, prev_bars=len(p), val=val, vah=vah, poc=poc, pdh=p.high.max(), pdl=p.low.min(), pclose=p.close.iloc[-1],
                on_hi=on.high.max() if len(on) else np.nan, on_lo=on.low.min() if len(on) else np.nan,
                last=df.close.iloc[-1], last_et=df.et.iloc[-1], rth_today=t[(t.m >= 570) & (t.m <= 959)])
def where(px, L):
    return "VAH 之上" if px > L["vah"] else "VAL 之下" if px < L["val"] else "價值區內"
if __name__ == "__main__":
    off = sys.argv[sys.argv.index("--offline") + 1] if "--offline" in sys.argv else None
    (nq, bn), (es, be) = load(NQ, off), load(ES, off); L, E = levels(nq), levels(es)
    w = L["vah"] - L["val"]; px = L["last"]; tw = L["last_et"].tz_convert("Asia/Taipei")
    print(f"NQZ26 掃描｜交易日 {L['today']}｜最新 {L['last_et']:%m/%d %H:%M} 美東（台灣 {tw:%H:%M}）{px:,.2f}｜移除錯誤報價 NQ {bn}、ES {be} 根")
    print(f"\n■ 今天的價位（來自 {L['prev']} 美股盤中，{L['prev_bars']} 根 1 分K）")
    print(f"  VAH {L['vah']:,.0f}｜POC {L['poc']:,.0f}｜VAL {L['val']:,.0f}（寬 {w:.0f}）｜前日高 {L['pdh']:,.2f}｜前日低 {L['pdl']:,.2f}｜前日收 {L['pclose']:,.2f}")
    print(f"  夜盤 高 {L['on_hi']:,.2f}／低 {L['on_lo']:,.2f}｜ES：VAL {E['val']:,.2f}–VAH {E['vah']:,.2f}，現價 {E['last']:,.2f}（{where(E['last'], E)}）")
    d_val, d_vah = px - L["val"], px - L["vah"]
    print(f"\n■ 現在的位置：{where(px, L)}（離 VAH {d_vah:+,.0f}、離 VAL {d_val:+,.0f}）；{'在' if L['pdl'] <= px <= L['pdh'] else '不在'}前日高低範圍內；比前日收盤 {px - L['pclose']:+,.0f}（{(px / L['pclose'] - 1):+.2%}）")
    em = L["last_et"].hour * 60 + L["last_et"].minute
    acc = "（美東 09:00 後的歷史準確率：在區下方 97%、上方 90%、區內 86%）" if 540 <= em < 570 else "（美東 08:00 前判斷只能參考；09:00 時的歷史準確率才有 86～97%）" if em < 480 or em >= 960 else "（美東 08:00–09:00 準確率約 73～84%）"
    if em < 570 or em >= 960:
        print(f"  盤前預判：若現在開盤，會開在{where(px, L)} {acc}")
    print("\n■ 09:30 開盤後的劇本（照組合 D；TP／SL 都是事先定好的）")
    print(f"  A. 開在 VAH {L['vah']:,.0f} 與 VAL {L['val']:,.0f} 之間 → 今天不做（區內開盤）")
    for side, edge, lab in ((-1, L["vah"], "VAH 之上"), (1, L["val"], "VAL 之下")):
        far = edge + side * -1 * w
        pd_edge = L["pdh"] if side == -1 else L["pdl"]
        print(f"  {'B' if side == -1 else 'C'}. 開在{lab}（{'>' if side == -1 else '<'} {edge:,.0f}）：")
        print(f"     • 開盤離區 ≤ {w:.0f} 點（{'≤' if side == -1 else '≥'} {far:,.0f}）且 ES 也開在 ES 價值區{'上方' if side == -1 else '下方'} → **I2 等回區**：連續 2 根 5 分K 收回 {L['val']:,.0f}～{L['vah']:,.0f} → "
              f"下一分鐘{'做空' if side == -1 else '做多'}，TP {'−' if side == -1 else '+'}55 點、SL {'+' if side == -1 else '−'}145 點，最晚 10:59 進場，15:49 平倉")
        print(f"     • 開在前日{'高' if side == -1 else '低'} {pd_edge:,.2f} {'之上' if side == -1 else '之下'} → **順勢觀察**：09:30–10:00 沒收回價值區，10:04 後 5 分K 收{'過 IB 高點' if side == -1 else '破 IB 低點'} → "
              f"{'做多' if side == -1 else '做空'}；SL＝IB 中點（30～80 點），TP＝2 倍 SL，浮盈 1 倍 SL 時停損移到進場價，12:29 平倉")
    print("  每天：同時只持 1 筆、最多 3 筆、連 2 虧或 −$3,000 停。前高空／前低多只看不做。")
    r = L["rth_today"]
    if len(r):
        o = r.open.iloc[0]; ib = r[r.m < 600]
        print(f"\n■ 盤中：09:30 開盤 {o:,.2f}（{where(o, L)}）｜IB 高 {ib.high.max():,.2f} 低 {ib.low.min():,.2f}｜目前 {px:,.2f}")
