// NQ 順勢多點（Tradovate 自訂指標版，JavaScript）
// 研究用途，不是投資建議。只畫訊號，不下單。
//
// 對應 TradingView 指標 NQ_I2_價值區助手_H1研究版.pine v18 的邏輯（docs/24；說明與回測見 tradovate/README.md）。
// Tradovate 的限制與差異：
//   1. 只能讀圖表本身的商品 → 需要 ES 同步確認的三個做多點（亞洲前低多 ES、亞洲 VAL 多 ES、早盤夜低多 ES）
//      在 NQ 圖上只畫成灰色「觀察」。另開 ES 1 分鐘圖加同一個指標，會畫「ES 同步確認」標籤；
//      ES 標籤在 NQ 那根 5 分K 的同一根或更早出現，NQ 的點才算計畫內。
//   2. 只能用圖表本身的週期 → 請用 1 分鐘圖（不是 1 分鐘會在最新一根上方顯示提醒，訊號不畫）。
//   3. 時間：程式自己換算美東時間（含夏令時間），不需要設定時區。
//
// 圖上：綠 BUY／紅 SELL＝第一層計畫內（I2、前高空·順、前低多·順）；藍＝第二層紙上（開盤動能 C1／趨勢日 TD10）；
//       灰＝觀察（逆勢、或缺 ES 確認）。每個點都寫進場價、TP、SL 和原因；出場時在出場那根標結果。
//
// 安裝：Tradovate → Code Explorer → 新增 Indicator → 貼上整份 → 存檔 → 1 分鐘圖加指標 nqTrendPlan（NQ 順勢多點）。

// ---------- 圖形座標（和 Tradovate ./tools/graphics 的 du/px/op 相同格式）----------
const du = (v) => ({ unit: "du", value: v });
const px = (v) => ({ unit: "px", value: v });
const op = (a, o, b) => ({ a: a, op: o, b: b });

// ---------- 固定參數（凍結）----------
const BIN = 5.0, VA_FRAC = 0.70, TP = 55.0, SL = 145.0;
const TD_EFF = 0.3215553652062646, TD_VW = 0.8, TD_EMA = 1.3001621620137531, TD_DI = 21.74334553752221, TD_MAC = 0.3815786106438511;
const COL = { buy: "#26a69a", sell: "#ef5350", obs: "#9e9e9e", paper: "#42a5f5", text: "#e0e0e0", es: "#00bcd4" };
const ES_NOTE = "；要 ES 同步：ES 圖上同名「ES 同步確認」標籤在同一根 5 分K 或更早出現才做";

// ---------- 美東時間（含夏令時間：3 月第二個週日～11 月第一個週日）----------
function nthSunday(y, mon, n) {
    const first = new Date(Date.UTC(y, mon, 1)).getUTCDay();
    return 1 + ((7 - first) % 7) + 7 * (n - 1);
}
function etOffsetMin(ms) {
    const y = new Date(ms).getUTCFullYear();
    const start = Date.UTC(y, 2, nthSunday(y, 2, 2), 7);
    const end = Date.UTC(y, 10, nthSunday(y, 10, 1), 6);
    return (ms >= start && ms < end) ? -240 : -300;
}
function etOf(ms) {
    const t = new Date(ms + etOffsetMin(ms) * 60000);
    return { ymd: t.getUTCFullYear() * 10000 + (t.getUTCMonth() + 1) * 100 + t.getUTCDate(), m: t.getUTCHours() * 60 + t.getUTCMinutes() };
}
function tradeKey(ms) {   // CME 交易日：美東 18:00 起算
    const t = new Date(ms + etOffsetMin(ms) * 60000 + 6 * 3600000);
    return t.getUTCFullYear() * 10000 + (t.getUTCMonth() + 1) * 100 + t.getUTCDate();
}
// 美股休市日（期貨只開到 13:00，研究裡不交易）：元旦、金恩、總統、耶穌受難、陣亡將士、六月節、獨立、勞動、感恩、聖誕（含補假）
function easterMD(y) {
    const a = y % 19, b = Math.floor(y / 100), c = y % 100, d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25), g = Math.floor((b - f + 1) / 3);
    const h = (19 * a + b - d - g + 15) % 30, i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451);
    return { mon: Math.floor((h + l - 7 * m + 114) / 31), day: ((h + l - 7 * m + 114) % 31) + 1 };
}
const SPECIAL_CLOSED = [20250109];   // 臨時休市（卡特國葬日）
function usHoliday(ymd) {
    if (SPECIAL_CLOSED.indexOf(ymd) >= 0) return true;
    const y = Math.floor(ymd / 10000), mo = Math.floor(ymd / 100) % 100, d = ymd % 100;
    const dow = new Date(Date.UTC(y, mo - 1, d)).getUTCDay();
    const nthDow = (mon, wd, n) => { const f = new Date(Date.UTC(y, mon - 1, 1)).getUTCDay(); return 1 + ((wd - f + 7) % 7) + 7 * (n - 1); };
    const lastMon = (mon) => { const ld = new Date(Date.UTC(y, mon, 0)).getUTCDate(), w = new Date(Date.UTC(y, mon - 1, ld)).getUTCDay(); return ld - ((w - 1 + 7) % 7); };
    const fixed = (mon, day, allowSat) => { const w = new Date(Date.UTC(y, mon - 1, day)).getUTCDay(); return (mo === mon && d === day && w !== 0 && w !== 6) || (allowSat && w === 6 && mo === mon && d === day - 1) || (w === 0 && mo === mon && d === day + 1); };
    if (fixed(1, 1, false) || fixed(6, 19, true) || fixed(7, 4, true) || fixed(12, 25, true)) return true;
    if (dow === 1 && ((mo === 1 && d === nthDow(1, 1, 3)) || (mo === 2 && d === nthDow(2, 1, 3)) || (mo === 5 && d === lastMon(5)) || (mo === 9 && d === nthDow(9, 1, 1)))) return true;
    if (dow === 4 && mo === 11 && d === nthDow(11, 4, 4)) return true;
    const ea = easterMD(y), gf = new Date(Date.UTC(y, ea.mon - 1, ea.day - 2));
    return gf.getUTCMonth() + 1 === mo && gf.getUTCDate() === d;
}
function msOf(d) {
    const ts = typeof d.timestamp === "function" ? d.timestamp() : d.timestamp;
    return ts instanceof Date ? ts.getTime() : Number(ts);
}
function f2(x) { return (Math.round(x * 100) / 100).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
function f0(x) { return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
function usd(x) { return (x >= 0 ? "+$" : "−$") + f0(Math.abs(x)); }

// ---------- 價值區（和 I2 相同算法：5 點一格、70%、POC 同量取低、同量先往下擴）----------
function valueArea(prof) {
    const keys = Object.keys(prof).map(Number).sort((a, b) => a - b);
    if (!keys.length) return null;
    const kMin = keys[0], kMax = keys[keys.length - 1];
    let total = 0, poc = kMin, pocV = -1;
    for (const k of keys) { const v = prof[k]; total += v; if (v > pocV) { pocV = v; poc = k; } }
    let lo = poc, hi = poc, acc = pocV;
    while (acc < VA_FRAC * total && (lo > kMin || hi < kMax)) {
        const lv = prof[lo - 1] !== undefined ? prof[lo - 1] : -1.0;
        const hv = prof[hi + 1] !== undefined ? prof[hi + 1] : -1.0;
        if (hv > lv) { hi += 1; acc += Math.max(0, hv); } else { lo -= 1; acc += Math.max(0, lv); }
    }
    return { val: lo * BIN, poc: poc * BIN, vah: (hi + 1) * BIN };
}

class NQPlan {
    init() {
        this.done = -1; this.cur = 0;   // 已處理到第幾根（只處理已收盤的 K 棒）；目前呼叫的 index
        this.gfx = {};             // 每個 index 要畫的圖形
        this.log = [];             // 交易紀錄（測試與檢查用）
        // 價值區
        this.prof = {}; this.inRth = false; this.ses = null;
        this.lv = { val: NaN, poc: NaN, vah: NaN, pdh: NaN, pdl: NaN, pdc: NaN };
        // 局勢、波動
        this.ranges = []; this.pv = 0; this.vv = 0; this.vwPrev = NaN; this.recent = [];
        this.day = { key: 0, rth: null };
        this.onLo = NaN; this.onDay = 0; this.asHi = NaN; this.asLo = NaN; this.asDay = 0;
        // 5 分K 遞迴
        this.b5 = null; this.k = { e20: NaN, e12: NaN, e26: NaN, sig: NaN, atr: NaN, pdm: NaN, ndm: NaN, pc: NaN, ph: NaN, pl: NaN, hist: [], slope: NaN, di: NaN, mac: NaN };
        // 計畫（第一層）每日風險
        this.plan = { key: 0, pnl: 0, open: 0 };
        this.trades = [];          // 進行中的單
        this.chartOk = true;
    }

    // ---------- 主流程：Tradovate 對每根 K 棒呼叫，即時盤會對最後一根重複呼叫 ----------
    map(d, i, history) {
        const cd = this.chartDescription;
        if (cd && cd.underlyingType && (cd.underlyingType !== "MinuteBar" || cd.elementSize !== 1 || (cd.elementSizeUnit && cd.elementSizeUnit !== "UnderlyingUnits"))) this.chartOk = false;
        this.cur = i;              // 這次呼叫產生的圖形都掛在第 i 根的輸出（座標仍是各自的 K 棒）
        if (this.isES === undefined) { const ci = this.contractInfo || {}; this.isES = /^M?ES/.test(String(ci.product || ci.contract || "")); }
        if (!this.chartOk) {
            const last = i === 0 || (typeof d.isLast === "function" && d.isLast());
            return { graphics: last ? { items: [{ tag: "Text", key: "warn" + i, point: { x: du(i), y: op(du(d.high()), "-", px(40)) }, text: "NQ 順勢多點：請改用 1 分鐘圖", style: { fontSize: 16, fontWeight: "bold", fill: COL.sell }, textAlignment: "leftAbove" }] } : undefined };
        }
        // 先把之前已收盤的 K 棒依序處理
        while (this.done < i - 1) {
            this.done += 1;
            const bar = this.done === i ? d : history.get(this.done);
            this.step(bar, this.done);
        }
        // 目前這根（還沒收盤）：等進場的單用這根的開盤價進場
        this.enterPending(d, i);
        const items = (this.gfx[i] || []).slice();
        // 持倉中的單：TP／SL 線畫到目前這根
        const isLast = typeof d.isLast === "function" ? d.isLast() : true;
        if (isLast) for (const t of this.trades) if (t.state === 2) items.push(...this.levelLines(t, i, "live" + t.id));
        const out = {
            vah: this.lv.vah, poc: this.lv.poc, val: this.lv.val, pdh: this.lv.pdh, pdl: this.lv.pdl,
            vwap: this.inRthNow(d) && this.vv > 0 ? this.pv / this.vv : undefined
        };
        if (items.length) out.graphics = { items: items };
        return out;
    }

    inRthNow(d) { const e = this.et(d); return e.m >= 570 && e.m <= 959; }
    et(d) { return etOf(msOf(d) - (this.props.tsIsClose ? 60000 : 0)); }
    // 第 i 根收盤後才知道的東西（出場、訊號）會在下一根的 map 才算出來，所以掛在目前這次呼叫的輸出上
    add(i, item) { const j = Math.max(i, this.cur); (this.gfx[j] = this.gfx[j] || []).push(item); }

    // ---------- 處理一根已收盤的 1 分K ----------
    step(d, i) {
        const ms = msOf(d) - (this.props.tsIsClose ? 60000 : 0);
        const e = etOf(ms), m = e.m, o = d.open(), h = d.high(), l = d.low(), c = d.close(), v = d.volume ? d.volume() : 0;
        this.curM = m; this.curL = l; this.curMs = ms;
        const rth = m >= 570 && m <= 959, i1 = m - 570, tk = tradeKey(ms);

        // 前一個美股盤中時段結束 → 價值區、前日高低收、前 10 天振幅、日線
        if (this.inRth && (!rth || e.ymd !== this.ses.ymd)) {
            const va = valueArea(this.prof);
            if (va) { this.lv.val = va.val; this.lv.poc = va.poc; this.lv.vah = va.vah; }
            this.lv.pdh = this.ses.hi; this.lv.pdl = this.ses.lo; this.lv.pdc = this.ses.close;
            if (this.ses.lastI > 209) this.ranges.push(this.ses.hi - this.ses.lo);   // 只算正常交易日
            this.prof = {}; this.inRth = false;
        }
        // 計畫的交易日（美東 18:00 起算）
        if (tk !== this.plan.key) { this.plan = { key: tk, pnl: 0, open: 0, asiaLong: false }; }
        // 夜盤最低（18:00～09:29）、亞洲盤高低（18:00 起）
        if (m >= 1080 && this.onDay !== e.ymd) { this.onDay = e.ymd; this.onLo = l; this.asHi = h; this.asLo = l; this.asiaNew(); }
        else {
            if ((m >= 1080 || m < 570) && !isNaN(this.onLo)) this.onLo = Math.min(this.onLo, l);
            if ((m >= 1080 || m <= 179) && !isNaN(this.asHi)) { this.asHi = Math.max(this.asHi, h); this.asLo = Math.min(this.asLo, l); }
        }
        // VWAP（09:30 起，用到前一根為止）與暴衝幅度（前 30 根）
        if (m === 570) { this.pv = 0; this.vv = 0; }
        this.vwPrev = this.vv > 0 ? this.pv / this.vv : NaN;
        // 5 分K：新的 5 分鐘開始時先收掉上一根
        const bk = (e.ymd % 100000) * 1000 + Math.floor(m / 5);
        if (this.b5 && this.b5.key !== bk) { this.close5(this.b5); this.b5 = null; }
        // 09:30：新的一天
        if (m === 570) this.newRthDay(d, i, o, h, l, e.ymd);

        // 先處理進場與持倉（進場那一分鐘也檢查），再找新訊號
        this.enterPending(d, i);
        this.manage(d, i, m, e.ymd, ms, o, h, l, c);

        if (rth) {
            // 累積價值區、當天高低
            if (!this.inRth) { this.inRth = true; this.ses = { ymd: e.ymd, hi: h, lo: l, close: c, lastI: i1 }; }
            else { this.ses.hi = Math.max(this.ses.hi, h); this.ses.lo = Math.min(this.ses.lo, l); this.ses.close = c; this.ses.lastI = i1; }
            const a = Math.floor(l / BIN), b = Math.floor(h / BIN), nb = b - a + 1;
            for (let k = a; k <= b; k++) this.prof[k] = (this.prof[k] || 0) + v / nb;
            if (this.day.rth && i1 <= 179 && m !== 570) { this.day.rth.hi = Math.max(this.day.rth.hi, h); this.day.rth.lo = Math.min(this.day.rth.lo, l); }
            if (this.day.rth && i1 <= 29) this.tdAccumulate(h, l, c, v);
            this.signalsRth(d, i, m, i1, c);
        }
        if ((m >= 1084 || m <= 119) && m % 5 === 4) this.signalsAsia(d, i, m, c);

        // 收盤後才更新：VWAP、暴衝序列、5 分K
        if (rth) { this.pv += (h + l + c) / 3 * v; this.vv += v; }
        this.recent.push(c); if (this.recent.length > 30) this.recent.shift();
        if (!this.b5) this.b5 = { key: bk, h: h, l: l, c: c };
        else { this.b5.h = Math.max(this.b5.h, h); this.b5.l = Math.min(this.b5.l, l); this.b5.c = c; }
        if (m % 5 === 4) { this.close5(this.b5); this.b5 = null; }   // 這根 5 分K 最後一分鐘：馬上收掉（10:00 判斷 TD10 要用）
    }

    // ---------- 5 分K 指標（EMA20／12／26、MACD 訊號 9、Wilder ATR14、DI±14）----------
    close5(b) {
        const k = this.k;
        const tr = isNaN(k.pc) ? b.h - b.l : Math.max(b.h - b.l, Math.abs(b.h - k.pc), Math.abs(b.l - k.pc));
        const upm = isNaN(k.ph) ? NaN : b.h - k.ph, dnm = isNaN(k.pl) ? NaN : k.pl - b.l;
        const pdm = (!isNaN(upm) && upm > dnm && upm > 0) ? upm : 0, ndm = (!isNaN(dnm) && dnm > upm && dnm > 0) ? dnm : 0;
        k.atr = isNaN(k.atr) ? tr : k.atr + (tr - k.atr) / 14;
        k.pdm = isNaN(k.pdm) ? pdm : k.pdm + (pdm - k.pdm) / 14;
        k.ndm = isNaN(k.ndm) ? ndm : k.ndm + (ndm - k.ndm) / 14;
        k.e20 = isNaN(k.e20) ? b.c : k.e20 + (b.c - k.e20) * 2 / 21;
        k.e12 = isNaN(k.e12) ? b.c : k.e12 + (b.c - k.e12) * 2 / 13;
        k.e26 = isNaN(k.e26) ? b.c : k.e26 + (b.c - k.e26) * 2 / 27;
        const macd = k.e12 - k.e26;
        k.sig = isNaN(k.sig) ? macd : k.sig + (macd - k.sig) * 2 / 10;
        k.hist.push(k.e20); if (k.hist.length > 7) k.hist.shift();
        k.slope = k.hist.length === 7 && k.atr > 0 ? (k.e20 - k.hist[0]) / k.atr : NaN;
        k.di = k.atr > 0 ? 100 * (k.pdm - k.ndm) / k.atr : NaN;
        k.mac = k.atr > 0 ? (macd - k.sig) / k.atr : NaN;
        k.pc = b.c; k.ph = b.h; k.pl = b.l;
    }

    adr10() { const r = this.ranges; if (r.length < 10) return NaN; let s = 0; for (let j = r.length - 10; j < r.length; j++) s += r[j]; return s / 10; }
    shock() { const w = this.recent; let mx = 0; for (let j = 0; j + 5 < w.length; j++) mx = Math.max(mx, Math.abs(w[j + 5] - w[j])); return mx; }

    // ---------- 09:30：I2 方向、當天狀態、標頭文字 ----------
    newRthDay(d, i, o, h, l, ymd) {
        if (usHoliday(ymd)) {
            this.day = { rth: null };
            this.add(i, { tag: "Text", key: "day" + i, point: { x: du(i), y: op(du(h), "-", px(30)) }, text: "美股休市（期貨提早收盤），今天不做", style: { fontSize: 11, fill: COL.obs }, textAlignment: "centerAbove" });
            return;
        }
        const lv = this.lv, ok = !isNaN(lv.val);
        const dir = ok ? (o > lv.vah ? -1 : o < lv.val ? 1 : 0) : 0;
        this.day = { rth: { open: o, hi: h, lo: l }, i2: { dir: dir, cnt: 0, done: false }, ref: [0, 0], amOn: 0, c1Done: false, tdDone: false, td: { path: 0, prev: o, above: 0, n: 0, pv: 0, vv: 0 }, t2: false };
        let txt;
        if (!ok) txt = "I2：沒有前一日價值區，今天不做";
        else if (dir === 0) txt = "I2：開在價值區內（" + f0(lv.val) + "～" + f0(lv.vah) + "），今天不做";
        else {
            const dist = dir === 1 ? lv.val - o : o - lv.vah, width = lv.vah - lv.val;
            txt = "I2：" + (dir === 1 ? "開在 VAL " + f0(lv.val) + " 之下 → 等回區才多" : "開在 VAH " + f0(lv.vah) + " 之上 → 等回區才空") + (dist > width ? "（離太遠，H1 會跳過）" : "");
        }
        this.add(i, { tag: "Text", key: "day" + i, point: { x: du(i), y: op(du(h), "-", px(30)) }, text: txt, style: { fontSize: 11, fill: COL.obs }, textAlignment: "centerAbove" });
    }

    tdAccumulate(h, l, c, v) {
        const t = this.day.td;
        t.path += Math.abs(c - t.prev); t.prev = c; t.pv += (h + l + c) / 3 * v; t.vv += v; t.above += (t.vv > 0 && c > t.pv / t.vv) ? 1 : 0; t.n += 1;
    }

    // ---------- 美股盤中的訊號 ----------
    signalsRth(d, i, m, i1, c) {
        const day = this.day; if (!day.rth) return;
        const lv = this.lv;
        // I2：5 分K 收盤（09:34 … 11:29）連續 2 根收在價值區內 → 下一分鐘進場
        if (day.i2.dir !== 0 && !day.i2.done && i1 >= 4 && i1 <= 119 && i1 % 5 === 4) {
            const inside = c >= lv.val && c <= lv.vah;
            day.i2.cnt = inside ? day.i2.cnt + 1 : 0;
            if (day.i2.cnt >= 2) {
                day.i2.done = true;
                const ei = i1 + 1;
                if (!((ei >= 28 && ei <= 32) || ei > 120)) {
                    const dir = day.i2.dir;
                    this.queue({ kind: "I2", tier: 1, dir: dir, sigI: i, exitM: 749, name: "I2 回價值區",
                        why: dir === 1 ? "原因：開在價值區下方，2 根 5 分K 收回 VAL " + f0(lv.val) + " 之上" : "原因：開在價值區上方，2 根 5 分K 收回 VAH " + f0(lv.vah) + " 之下" });
                }
            }
        }
        if (i1 >= 4 && i1 <= 119 && i1 % 5 === 4) {
            const ei = i1 + 1, okE = ei >= 5 && ei <= 120 && !(ei >= 28 && ei <= 32);
            // 前高空／前低多：突破前日高（低）後 5 分K 收回；計畫內要順盤中趨勢（VWAP 同一邊，進場時判斷）
            if (okE && !isNaN(lv.pdh) && day.ref[0] === 0 && day.rth.hi > lv.pdh && c < lv.pdh) {
                day.ref[0] = 1;
                this.queue({ kind: "前高空", tier: 1, dir: -1, sigI: i, exitM: 749, name: "前高空", needTrend: true, level: lv.pdh,
                    why: "原因：強壓力 前日高 " + f0(lv.pdh) + " 假突破收回" });
            }
            if (okE && !isNaN(lv.pdl) && day.ref[1] === 0 && day.rth.lo < lv.pdl && c > lv.pdl) {
                day.ref[1] = 1;
                this.queue({ kind: "前低多", tier: 1, dir: 1, sigI: i, exitM: 749, name: "前低多", needTrend: true, level: lv.pdl,
                    why: "原因：強支撐 前日低 " + f0(lv.pdl) + " 破底翻" });
            }
            // 早盤夜低多（原本要 ES 同步、波動 ≥1.2 倍）：這裡沒有 ES，只畫觀察
            if (this.props.showObs && okE && day.amOn === 0 && !isNaN(this.onLo) && day.rth.lo < this.onLo && c > this.onLo) {
                day.amOn = 1;
                const a = this.adr10(), vol = isNaN(a) ? NaN : a / 255.2;
                this.queue({ kind: "早盤夜低多", tier: 0, dir: 1, sigI: i, exitM: 749, name: "早盤夜低多（未確認 ES）",
                    why: "觀察：夜盤低 " + f0(this.onLo) + " 破底翻" + ES_NOTE + "，而且波動 ≥ 1.2 倍（現在 " + (isNaN(vol) ? "—" : vol.toFixed(1)) + " 倍）" });
            }
        }
        // 第二層 C1：09:34 收盤 − 09:30 開盤 ≥ 0.15×前 10 天振幅 → 09:35 順勢
        if (this.props.showT2 && i1 === 4 && !day.c1Done) {
            day.c1Done = true;
            const a = this.adr10(), mv = c - day.rth.open;
            if (!isNaN(a) && Math.abs(mv) >= 0.15 * a && mv !== 0) {
                day.t2 = true;
                this.queue({ kind: "C1", tier: 2, dir: mv > 0 ? 1 : -1, sigI: i, exitM: 955, name: "開盤動能 C1", dist: Math.min(Math.max(0.25 * a, 10), 145), trail: true, noTP: true,
                    why: "原因：開盤 5 分鐘" + (mv > 0 ? "上攻 +" : "下殺 −") + f0(Math.abs(mv)) + " 點 ≥ 0.15×前 10 天振幅（" + f0(0.15 * a) + " 點）" });
            }
        }
    }

    // ---------- 10:00 第一根：TD10（09:55 那根 5 分K 已收；沒有 C1 才做）----------
    tdCheck(d, i, i1) {
        const day = this.day;
        if (!this.props.showT2 || !day.rth || day.tdDone || i1 < 30 || day.t2) return;
        day.tdDone = true;
        const t = day.td, k = this.k;
        const effi = t.path > 0 ? (t.prev - day.rth.open) / t.path : 0, avw = t.n > 0 ? t.above / t.n * 2 - 1 : NaN;
        const vals = [effi, avw, k.slope, k.di, k.mac], th = [TD_EFF, TD_VW, TD_EMA, TD_DI, TD_MAC];
        let up = 0, dn = 0;
        for (let j = 0; j < 5; j++) { if (vals[j] > th[j]) up++; if (vals[j] < -th[j]) dn++; }
        const dir = up >= 3 && dn === 0 ? 1 : dn >= 3 && up === 0 ? -1 : 0;
        if (dir !== 0) {
            day.t2 = true;
            this.queue({ kind: "TD10", tier: 2, dir: dir, sigI: i - 1, exitM: 955, name: "趨勢日 TD10", dist: 100, trail: false, noTP: true,
                why: "原因：10:00 趨勢日 " + (dir === 1 ? up : dn) + " 張" + (dir === 1 ? "多方" : "空方") + "強票（效率 " + effi.toFixed(2) + "、VWAP 時間 " + avw.toFixed(2) + "、EMA 斜率 " + (k.slope || 0).toFixed(1) + "、DI 差 " + Math.round(k.di || 0) + "、MACD 柱 " + (k.mac || 0).toFixed(2) + "）" });
        }
    }

    // ---------- 亞洲盤（美東 18:05～02:00 進場）：只畫觀察（原本要 ES 同步）----------
    asiaNew() { this.asia = { short: 0, long: 0, val: 0 }; }
    signalsAsia(d, i, m, c) {
        if (!this.props.showObs || !this.asia || isNaN(this.asHi) || usHoliday(this.plan.key)) return;
        const lv = this.lv, a = this.asia;
        if (a.short === 0 && !isNaN(lv.pdh) && this.asHi > lv.pdh && c < lv.pdh) {
            a.short = 1;
            this.queue({ kind: "亞洲空", tier: 0, dir: -1, sigI: i, exitM: 179, asia: true, name: "亞洲空", why: "觀察：前日高 " + f0(lv.pdh) + " 亞洲盤假突破收回（歷史上沒通過驗證）" });
        }
        if (a.long === 0 && !isNaN(lv.pdl) && this.asLo < lv.pdl && c > lv.pdl) {
            a.long = 1;
            this.queue({ kind: "亞洲前低多", tier: 0, dir: 1, sigI: i, exitM: 179, asia: true, name: "亞洲前低多（未確認 ES）", why: "觀察：前日低 " + f0(lv.pdl) + " 亞洲盤破底翻" + ES_NOTE + "（每晚最多一筆亞洲做多）" });
        }
        if (a.val === 0 && !isNaN(lv.val) && this.asLo < lv.val && c > lv.val) {
            a.val = 1;
            this.queue({ kind: "亞洲VAL多", tier: 0, dir: 1, sigI: i, exitM: 179, asia: true, name: "亞洲VAL多（未確認 ES）", why: "觀察：VAL " + f0(lv.val) + " 亞洲盤破底翻" + ES_NOTE + "（每晚最多一筆亞洲做多）" });
        }
    }

    // ---------- 單的生命週期 ----------
    queue(t) {
        // ES 圖：只畫三個 ES 同步確認點（給 NQ 圖上的灰色觀察點對照），不畫交易
        if (this.isES) {
            if (t.kind === "亞洲前低多" || t.kind === "亞洲VAL多" || t.kind === "早盤夜低多") {
                const mm = (this.curM + 1) % 1440, hm = String(Math.floor(mm / 60)).padStart(2, "0") + ":" + String(mm % 60).padStart(2, "0");
                this.add(t.sigI, { tag: "Text", key: "es" + t.kind + t.sigI, point: { x: du(t.sigI), y: op(du(this.curL), "+", px(14)) }, text: "ES 同步確認 " + t.kind + "（美東 " + hm + " 收盤）", style: { fontSize: 12, fontWeight: "bold", fill: COL.es }, textAlignment: "centerBelow" });
                this.log.push({ id: "es" + t.sigI, kind: "ES確認" + t.kind, tier: -1, dir: 1, inPlan: false, skipWhy: "", entryMs: this.curMs + 60000, entry: NaN });
            }
            return;
        }
        t.state = 1; t.id = this.log.length + this.trades.length + "_" + t.sigI; this.trades.push(t); }

    enterPending(d, i) {
        // 10:00 那根：先判斷 TD10
        const e = this.et(d), i1 = e.m - 570;
        if (e.m >= 570 && e.m <= 959) this.tdCheck(d, i, i1);
        for (const t of this.trades) {
            if (t.state !== 1 || i <= t.sigI) continue;
            const o = d.open();
            t.state = 2; t.entry = o; t.entryI = i; t.best = 0; t.trailed = false; t.ymd = e.ymd;
            t.lastC = o; t.lastI = i; t.lastMs = msOf(d) - (this.props.tsIsClose ? 60000 : 0); t.lastH = d.high(); t.lastL = d.low();
            const dist = t.dist || SL;
            t.stop = o - t.dir * dist; t.tp = t.noTP ? NaN : o + t.dir * TP;
            // 計畫內判斷（第一層）
            let why = "";
            if (t.tier === 1) {
                if (t.needTrend) {
                    const vw = this.vv > 0 ? this.pv / this.vv : NaN, withT = !isNaN(vw) && (t.dir === 1 ? o > vw : o < vw);
                    t.why += withT ? "，價格在 VWAP " + (t.dir === 1 ? "之上" : "之下") + "（順勢）" : "，但價格在 VWAP " + (t.dir === 1 ? "之下" : "之上") + "（逆勢）";
                    if (!withT) why = "逆盤中趨勢";
                }
                const sh = this.shock();
                if (!why && sh >= this.props.swanTh) why = "最近 30 分鐘暴衝 " + f0(sh) + " 點";
                if (!why && this.plan.pnl - 2948 * (this.plan.open + 1) < -3000) why = "今天再虧一筆會超過 −$3,000";
                t.inPlan = !why;
                if (t.inPlan) this.plan.open += 1;
            }
            t.skipWhy = why;
            const head = t.tier === 2 ? "紙上 " + (t.dir === 1 ? "BUY  " : "SELL  ") : t.tier === 1 && t.inPlan ? (t.dir === 1 ? "BUY  " : "SELL  ") : "觀察（不做）";
            const col = t.tier === 2 ? COL.paper : t.tier === 1 && t.inPlan ? (t.dir === 1 ? COL.buy : COL.sell) : COL.obs;
            const line2 = "進 " + f2(o) + (t.noTP ? "" : "｜TP " + f2(t.tp)) + "｜SL " + f2(t.stop) + (t.trail ? "｜+100 後移動停損 100" : "") + (t.exitM === 955 ? "｜15:55 平倉" : "");
            const line3 = t.why + (why ? "（" + why + "，不做）" : "") + (t.tier === 2 ? "（5 口 MNQ，只紙上記錄）" : "");
            const below = t.dir === 1, y0 = below ? op(du(d.low()), "+", px(14)) : op(du(d.high()), "-", px(14));
            const al = below ? "centerBelow" : "centerAbove", step = below ? "+" : "-";
            const lines = below ? [[head + t.name, 13, col, "bold"], [line2, 11, COL.text, "normal"], [line3, 10, COL.obs, "normal"]]
                                : [[line3, 10, COL.obs, "normal"], [line2, 11, COL.text, "normal"], [head + t.name, 13, col, "bold"]];
            let off = 0;
            lines.forEach((ln, j) => {
                this.add(i, { tag: "Text", key: "e" + t.id + "_" + j, point: { x: du(i), y: op(y0, step, px(off)) }, text: ln[0], style: { fontSize: ln[1], fontWeight: ln[3], fill: ln[2] }, textAlignment: al });
                off += ln[1] + 5;
            });
            this.log.push({ id: t.id, kind: t.kind, tier: t.tier, dir: t.dir, inPlan: !!t.inPlan, skipWhy: why, entryMs: msOf(d) - (this.props.tsIsClose ? 60000 : 0), entry: o, stop0: t.stop, tp: t.tp });
            t.logRow = this.log[this.log.length - 1];
        }
    }

    manage(d, i, m, ymd, ms, o, h, l, c) {
        for (const t of this.trades) {
            if (t.state !== 2) continue;
            let exit = NaN, why = "", xi = i, xms = ms, xh = h, xl = l;
            const hitS = t.dir === 1 ? l <= t.stop : h >= t.stop;
            const timeUp = t.asia ? (m >= t.exitM && m < 1080) : (m >= t.exitM && m <= 959);
            // 美股盤中的單：如果盤提早收（假日半天），下一根已經不是同一個盤中時段 → 用最後一根的收盤價出場
            if (!t.asia && (ymd !== t.ymd || m < 570 || m > 959)) { exit = t.lastC; why = "時間到"; xi = t.lastI; xms = t.lastMs; xh = t.lastH; xl = t.lastL; }
            else if (hitS) {
                const gap = i > t.entryI && (t.dir === 1 ? o < t.stop : o > t.stop);
                exit = t.trail && gap ? o : t.stop; why = t.trailed ? "移動停損" : "止損";
            } else if (!isNaN(t.tp) && (t.dir === 1 ? h >= t.tp : l <= t.tp)) { exit = t.tp; why = "止盈"; }
            else if (timeUp) { exit = c; why = "時間到"; }
            if (why) {
                t.state = 3;
                const lots = t.tier === 2 ? 0.5 : 1.0;
                const pnl = (t.dir * (exit - t.entry) * 20 - 48) * lots;
                if (t.tier === 1 && t.inPlan) { this.plan.pnl += pnl; this.plan.open -= 1; }
                Object.assign(t.logRow, { exitMs: xms, exit: exit, why: why, pnl: pnl });
                const mark = why === "止盈" ? "✓ 止盈 " : why === "止損" ? "✗ 止損 " : why === "移動停損" ? "⇢ 移動停損 " : "⏱ 時間到 ";
                const col = pnl >= 0 ? COL.buy : COL.sell;
                this.add(i, { tag: "Text", key: "x" + t.id, point: { x: du(xi), y: t.dir === 1 ? op(du(xh), "-", px(10)) : op(du(xl), "+", px(10)) }, text: mark + f2(exit) + "（" + usd(pnl) + (t.tier === 2 ? "，5 口 MNQ" : "") + "）", style: { fontSize: 11, fontWeight: "bold", fill: col }, textAlignment: t.dir === 1 ? "centerAbove" : "centerBelow" });
                for (const it of this.levelLines(t, xi, "done" + t.id)) this.add(i, it);
            } else {
                t.lastC = c; t.lastI = i; t.lastMs = ms; t.lastH = h; t.lastL = l;
                const fav = t.dir === 1 ? h - t.entry : t.entry - l;
                t.best = Math.max(t.best, fav);
                if (t.trail && t.best >= 100) {
                    const ns = t.entry + t.dir * (t.best - 100);
                    if ((t.dir === 1 && ns > t.stop) || (t.dir === -1 && ns < t.stop)) { t.stop = ns; t.trailed = true; }
                }
            }
        }
        this.trades = this.trades.filter((t) => t.state !== 3);
    }

    levelLines(t, i, key) {
        const seg = (y) => ({ tag: "Line", a: { x: du(t.entryI), y: du(y) }, b: { x: du(i), y: du(y) } });
        const items = [{ tag: "LineSegments", key: key + "sl", lines: [seg(t.stop)], lineStyle: { lineWidth: 1, color: COL.sell, lineStyle: 3 } },
                       { tag: "LineSegments", key: key + "en", lines: [seg(t.entry)], lineStyle: { lineWidth: 1, color: COL.obs, lineStyle: 3 } }];
        if (!isNaN(t.tp)) items.push({ tag: "LineSegments", key: key + "tp", lines: [seg(t.tp)], lineStyle: { lineWidth: 1, color: COL.buy, lineStyle: 3 } });
        return items;
    }
}

module.exports = {
    name: "nqTrendPlan",
    description: "NQ 順勢多點（I2／前高空／前低多＋紙上開盤動能）",
    calculator: NQPlan,
    inputType: "bars",
    areaChoice: "overlay",
    tags: ["NQ 研究"],
    params: {
        showObs: { type: "boolean", def: true },
        showT2: { type: "boolean", def: true },
        swanTh: { type: "number", def: 250, restrictions: { min: 50, max: 1000, step: 10 } },
        tsIsClose: { type: "boolean", def: false }
    },
    plots: {
        vah: { title: "VAH" }, poc: { title: "POC" }, val: { title: "VAL" },
        pdh: { title: "前日高" }, pdl: { title: "前日低" }, vwap: { title: "VWAP" }
    },
    plotter: [{ type: "multiline", fields: ["vah", "poc", "val", "pdh", "pdl", "vwap"] }],
    schemeStyles: {
        dark: {
            vah: { color: "#ff9800", lineWidth: 2 }, poc: { color: "#9e9e9e", lineWidth: 1 }, val: { color: "#26a69a", lineWidth: 2 },
            pdh: { color: "#ef5350", lineWidth: 1, lineStyle: 3 }, pdl: { color: "#66bb6a", lineWidth: 1, lineStyle: 3 }, vwap: { color: "#ab47bc", lineWidth: 1 }
        },
        light: {
            vah: { color: "#ef6c00", lineWidth: 2 }, poc: { color: "#757575", lineWidth: 1 }, val: { color: "#00897b", lineWidth: 2 },
            pdh: { color: "#e53935", lineWidth: 1, lineStyle: 3 }, pdl: { color: "#43a047", lineWidth: 1, lineStyle: 3 }, vwap: { color: "#8e24aa", lineWidth: 1 }
        }
    }
};
