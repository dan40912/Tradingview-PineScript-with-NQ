"""凍結的候選（只用探索＋驗證挑出，2026-11-12 在看測試期之前寫下）。研究用途，不是投資建議。
C1 是事先指定的主要候選；其他是各家族最好的高原點，用來對照。"""
from families import f1_od, f5_rs, f6_reclaim, f3_vwap, f6_trendday, f6_gap, EXITS, STOPS

CANDIDATES = {
    "C1_開盤驅動5分": (f1_od, dict(N=5, k=0.15, cl=0.0, es=0, side="both"), "0.25ADR", "移100"),
    "C2_相對ES強弱15分": (f5_rs, dict(N=15, th=0.05, need_es=0, side="both"), "0.25ADR", "移100"),
    "C3_收復開盤": (f6_reclaim, dict(T=30, k=0.1, vw=0, last=180, side="both"), "0.25ADR", "收盤"),
    "C4_VWAP單邊60分": (f3_vwap, dict(start=30, K=60, slope=0.0, mode="first", last=150, side="both"), "0.25ADR", "3R"),
    "C5_趨勢日11:30": (f6_trendday, dict(at=120, k=0.1, frac=0.95, side="both"), "0.25ADR", "移100"),
    "C6_跳空延續5分": (f6_gap, dict(N=5, gk=0.1, hold=1, side="both"), "0.25ADR", "移100"),
}
PRIMARY = "C1_開盤驅動5分"


def cfg(name):
    fn, p, s, x = CANDIDATES[name]
    return fn, p, {**STOPS[s], **EXITS[x]}
