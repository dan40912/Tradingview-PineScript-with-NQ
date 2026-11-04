"""頭肩頂／頭肩底的客觀規則回測（5 分K、美股盤中）。參數先寫死，看整組參數的表現，不挑最好的。
- 轉折點：一根 5 分K 的高點高於左右各 k 根 → 波段高點（低點同理）；要等右邊 k 根收完才確認，不偷看。
- 頭肩頂：連續三個波段高點 左肩 < 頭 > 右肩；頭至少比兩肩高 0.2×（頭 − 頸線）；兩肩差距 ≤ 0.5×（頭 − 頸線）。
  頸線＝兩個低點中「靠近右邊那個」（near）或「比較低的那個」（low）。
- 進場：右肩確認之後，5 分K 收在頸線之下 → 下一分鐘開盤做空。頭肩底反過來做多。
- SL＝右肩外 5 點（最多 150 點，超過就不做）；TP＝1R／2R／等幅（頭到頸線的距離）；15:49 平倉。
- 背景過濾：none（不過濾）／ctx（頭肩頂的頭要在 VAH 和前日高之上；頭肩底的頭要在 VAL 和前日低之下＝假突破）。
- 每天每個方向只做第一個；頭要在 09:45 之後，進場在 14:00 之前。"""
import sys, pickle, itertools, numpy as np, pandas as pd
from eng import sim
nq = pickle.load(open("NQ_days2.pkl", "rb"))
def b5(d):
    O, H, L, C = d["O"], d["H"], d["L"], d["C"]; out = np.full((78, 4), np.nan)
    for b in range(78):
        s = slice(b * 5, b * 5 + 5); ok = ~np.isnan(C[s])
        if ok.any(): out[b] = [O[s][ok][0], np.nanmax(H[s]), np.nanmin(L[s]), C[s][ok][-1]]
    return out
def pivots(x, k, hi=True):
    p = []
    for i in range(k, len(x) - k):
        w = x[i - k:i + k + 1]
        if np.isnan(w).any(): continue
        if (hi and x[i] == w.max() and (w[:k] < x[i]).all()) or (not hi and x[i] == w.min() and (w[:k] > x[i]).all()): p.append(i)
    return p
def hs(d, k=2, neck="near", tgt="1R", ctx="none", side=-1, maxrisk=150):
    B = b5(d); H, L, C = B[:, 1], B[:, 2], B[:, 3]
    if side == 1: H, L = -L, -H                                          # 頭肩底：把價格翻過來，用同一套頭肩頂邏輯
    C2 = C if side == -1 else -C
    ph = pivots(H, k, True)
    for a, b, c in zip(ph, ph[1:], ph[2:]):
        if b < 3: continue
        ls, hd, rs = H[a], H[b], H[c]
        t1, t2 = np.nanmin(L[a:b + 1]), np.nanmin(L[b:c + 1])
        nl = t2 if neck == "near" else min(t1, t2); depth = hd - nl
        if depth <= 0 or hd - max(ls, rs) < 0.2 * depth or abs(ls - rs) > 0.5 * depth: continue
        if ctx == "ctx":
            real_hd = hd if side == -1 else -hd
            if side == -1 and not (real_hd > d["vah"] and real_hd > d["pdh"]): continue
            if side == 1 and not (real_hd < d["val"] and real_hd < d["pdl"]): continue
        for j in range(c + k, 78):                                          # 右肩確認之後才開始找跌破
            if np.isnan(C2[j]): continue
            if H[j] > hd: break                                              # 創新高 → 型態失敗
            if C2[j] < nl:
                ei = j * 5 + 5
                if ei > 270 or ei > 389: break                               # 14:00 之後不進場
                real_sl = (rs + 5) if side == -1 else -(rs + 5)
                e = d["O"][ei]
                if np.isnan(e): break
                risk = abs(real_sl - e)
                if risk > maxrisk or risk < 5: break
                tp = {"1R": risk, "2R": 2 * risk, "MM": depth}[tgt]
                t = sim(d, ei, side, tp, risk, 379, None, None, "頭肩")
                return t
            if j > c + 20: break
        return None
    return None
if __name__ == "__main__":
    rows = []
    for k, neck, tgt, ctx, side in itertools.product((1, 2), ("near", "low"), ("1R", "2R", "MM"), ("none", "ctx"), (-1, 1)):
        tr = [t | {"date": d["date"]} for d in nq for t in [hs(d, k, neck, tgt, ctx, side)] if t]
        df = pd.DataFrame(tr)
        r = dict(型態="頭肩頂（空）" if side == -1 else "頭肩底（多）", k=k, 頸線=neck, 目標=tgt, 背景=ctx, 筆=len(df))
        for lab, (a, b) in {"2024": ("2024", "2024-12-31"), "2025": ("2025", "2025-12-31"), "2026驗證": ("2026", "2026-09-30"), "10月起": ("2026-10", "2026-12-31")}.items():
            x = df[(df.date >= a) & (df.date <= b)] if len(df) else df
            r[f"{lab}"] = f"{len(x)}筆 {x.pnl.mean():+.0f}" if len(x) else "—"
        x = df[df.date <= "2026-09-30"] if len(df) else df
        r["全期每筆"] = round(x.pnl.mean()) if len(x) else np.nan; r["勝率"] = f"{(x.pnl > 0).mean():.0%}" if len(x) else "—"
        rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.to_string(index=False))
    for s in ("頭肩頂（空）", "頭肩底（多）"):
        x = R[R.型態 == s]; print(f"\n{s}：{len(x)} 組參數，全期每筆為正 {(x.全期每筆 > 0).sum()} 組，中位數 {x.全期每筆.median():+.0f}")
