// 自己輸入 K 線：解析 CSV／貼上的文字 → 每個交易日的價位與 09:30–12:29 1 分K（和 build_data.py 同一套算法）
// 研究用途，不是投資建議。
(function (root) {
  const BIN = 5, VA_FRAC = 0.7;

  // ---- 時區：把任何時間換成美東的「日期＋第幾分鐘」 ----
  const nyFmt = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  const offCache = new Map();
  function nyOffsetMin(ms) {            // 美東相對 UTC 的分鐘數（同一小時快取）
    const key = Math.floor(ms / 3600000);
    if (offCache.has(key)) return offCache.get(key);
    const p = {}; for (const x of nyFmt.formatToParts(new Date(key * 3600000))) p[x.type] = x.value;
    const asUtc = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute);
    const off = Math.round((asUtc - key * 3600000) / 60000);
    offCache.set(key, off); return off;
  }
  function toNY(ms) { const d = new Date(ms + nyOffsetMin(ms) * 60000); return { ymd: d.toISOString().slice(0, 10), mod: d.getUTCHours() * 60 + d.getUTCMinutes() }; }
  // 沒寫時區的時間字串，依使用者選的時區解讀
  function naiveToMs(y, mo, d, hh, mi, ss, tz) {
    const asUtc = Date.UTC(y, mo - 1, d, hh, mi, ss);
    if (tz === 'UTC') return asUtc;
    if (tz === 'Asia/Taipei') return asUtc - 8 * 3600000;
    let ms = asUtc - nyOffsetMin(asUtc) * 60000;          // 美東：猜一次再修正
    ms = asUtc - nyOffsetMin(ms) * 60000;
    return ms;
  }
  function parseTime(s, tz) {
    s = s.trim().replace(/^"|"$/g, '');
    if (/^\d{9,13}(\.\d+)?$/.test(s)) { const x = +s; return x > 1e12 ? x : x * 1000; }   // unix 秒或毫秒
    const m = s.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?$/i);
    if (!m) return NaN;
    const [, y, mo, d, hh, mi, ss, z] = m;
    if (z) { const t = Date.parse(`${y}-${mo.padStart(2, '0')}-${d.padStart(2, '0')}T${hh.padStart(2, '0')}:${mi}:${ss || '00'}${z.toUpperCase() === 'Z' ? 'Z' : z.length === 5 ? z.slice(0, 3) + ':' + z.slice(3) : z}`); return t; }
    return naiveToMs(+y, +mo, +d, +hh, +mi, +(ss || 0), tz);
  }

  // ---- CSV ----
  function parseCSV(text, tz) {
    const lines = text.replace(/^﻿/, '').split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) throw new Error('沒有內容');
    const sep = lines[0].includes('\t') ? '\t' : lines[0].includes(',') ? ',' : lines[0].includes(';') ? ';' : /\s+/;
    const split = l => l.split(sep).map(x => x.trim().replace(/^"|"$/g, ''));
    let head = split(lines[0]).map(x => x.toLowerCase()), start = 1;
    const isNum = x => x !== '' && !isNaN(+x);
    if (head.slice(1).some(isNum)) { start = 0; head = ['time', 'open', 'high', 'low', 'close', 'volume'].slice(0, head.length); if (split(lines[0]).length >= 7 && !isNum(split(lines[0])[1])) head = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']; }
    const find = (...names) => head.findIndex(h => names.some(n => h === n || h.startsWith(n)));
    const iT = find('timestamp_utc', 'timestamp', 'datetime', 'time', 'date'), iD = head.indexOf('date');
    const iTime = head.indexOf('time'), sepDate = iD >= 0 && iTime >= 0 && iD !== iTime;
    const iO = find('open', 'o'), iH = find('high', 'h'), iL = find('low', 'l'), iC = find('close', 'c', 'last'), iV = find('volume', 'vol', 'v');
    if (iT < 0 || iO < 0 || iH < 0 || iL < 0 || iC < 0) throw new Error('找不到欄位：需要 時間、open、high、low、close（volume 可省略）。目前的欄位：' + head.join(', '));
    const bars = []; let bad = 0;
    for (let k = start; k < lines.length; k++) {
      const c = split(lines[k]);
      const ts = sepDate ? `${c[iD]} ${c[iTime]}` : c[iT];
      const t = parseTime(ts, tz), o = +c[iO], h = +c[iH], l = +c[iL], cl = +c[iC], v = iV >= 0 ? +c[iV] || 0 : 0;
      if (!isFinite(t) || !isFinite(o) || !isFinite(h) || !isFinite(l) || !isFinite(cl) || h < l) { bad++; continue; }
      bars.push({ t, o, h, l, c: cl, v });
    }
    if (!bars.length) throw new Error('沒有任何一列能讀懂。時間請用 2026-10-19 09:30、2026-10-19T09:30:00-04:00 或 unix 秒。');
    bars.sort((a, b) => a.t - b.t);
    return { bars, bad };
  }

  // ---- 價值區（和 Pine、build_data.py 相同）----
  function valueArea(rth) {
    const prof = new Map(), noVol = rth.every(b => !b.v);
    for (const b of rth) {
      const a = Math.floor(b.l / BIN), z = Math.floor(b.h / BIN), n = z - a + 1, v = noVol ? 1 : b.v;
      for (let k = a; k <= z; k++) prof.set(k, (prof.get(k) || 0) + v / n);
    }
    if (!prof.size) return null;
    const keys = [...prof.keys()].sort((a, b) => a - b), kmin = keys[0], kmax = keys[keys.length - 1];
    let total = 0, poc = kmin, pv = -1;
    for (const k of keys) { const v = prof.get(k); total += v; if (v > pv) { pv = v; poc = k; } }
    let lo = poc, hi = poc, acc = pv;
    while (acc < VA_FRAC * total && (lo > kmin || hi < kmax)) {
      const lv = prof.has(lo - 1) ? prof.get(lo - 1) : -1, hv = prof.has(hi + 1) ? prof.get(hi + 1) : -1;
      if (hv > lv) { hi++; acc += Math.max(0, hv); } else { lo--; acc += Math.max(0, lv); }
    }
    return [lo * BIN, poc * BIN, (hi + 1) * BIN];
  }

  // ---- 分成交易日（美東 18:00 以後算下一個交易日）----
  function buildDays(bars, step) {
    const by = new Map();
    for (const b of bars) {
      const { ymd, mod } = toNY(b.t);
      let td = ymd;
      if (mod >= 1080) { const d = new Date(ymd + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + 1); td = d.toISOString().slice(0, 10); }
      if (!by.has(td)) by.set(td, []);
      by.get(td).push(Object.assign({ mod }, b));
    }
    const out = [], ranges = [];
    let prev = null;
    for (const td of [...by.keys()].sort()) {
      const g = by.get(td), rth = g.filter(b => b.mod >= 570 && b.mod <= 959);
      if (!rth.length) continue;
      const pre = g.filter(b => b.mod < 570 || b.mod >= 1080);
      const morning = new Map(rth.filter(b => b.mod <= 749).map(b => [b.mod - 570, b]));
      if (morning.size) {
        const lastI = Math.min(179, Math.max(...morning.keys()) + step - 1);
        const o = [], h = [], l = [], c = [], v = [];
        let last = rth[0].o;
        for (let i = 0; i <= lastI; i++) {
          const b = morning.get(i);
          if (b) {
            o.push(b.o); h.push(b.h); l.push(b.l); c.push(b.c); v.push(b.v); last = b.c;
          } else {   // 沒成交的分鐘（或 5 分K 資料的中間幾分鐘）：用前一根收盤補平，量 0
            o.push(last); h.push(last); l.push(last); c.push(last); v.push(0); }
        }
        const va = prev && prev.va;
        out.push({
          meta: {
            d: td, ct: '自己輸入',
            val: va ? va[0] : null, poc: va ? va[1] : null, vah: va ? va[2] : null,
            pdh: prev ? prev.hi : null, pdl: prev ? prev.lo : null, pdc: prev ? prev.close : null,
            onh: pre.length ? Math.max(...pre.map(b => b.h)) : null, onl: pre.length ? Math.min(...pre.map(b => b.l)) : null,
            atr: ranges.length >= 5 ? +(ranges.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, ranges.length)).toFixed(2) : null,
            atrN: Math.min(20, ranges.length),
          }, o: o.slice(0, 180), h: h.slice(0, 180), l: l.slice(0, 180), c: c.slice(0, 180), v: v.slice(0, 180),
          full: rth[rth.length - 1].mod >= 959,
        });
      }
      const hi = Math.max(...rth.map(b => b.h)), lo = Math.min(...rth.map(b => b.l));
      prev = { va: valueArea(rth), hi, lo, close: rth[rth.length - 1].c };
      if (rth[rth.length - 1].mod >= 959) ranges.push(hi - lo);
    }
    return out;
  }

  function importText(text, tz) {
    const { bars, bad } = parseCSV(text, tz);
    // 偵測週期：相鄰 K 棒最常見的間隔
    const gaps = new Map();
    for (let i = 1; i < Math.min(bars.length, 3000); i++) { const g = Math.round((bars[i].t - bars[i - 1].t) / 60000); if (g > 0) gaps.set(g, (gaps.get(g) || 0) + 1); }
    const step = [...gaps.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || 1;
    if (step > 5) throw new Error(`資料看起來是 ${step} 分K。這頁的規則用 1 分K（最多接受 5 分K），請匯出 1 分K。`);
    const days = buildDays(bars, step);
    if (!days.length) throw new Error('資料裡沒有美股盤中（美東 09:30 以後）的 K 棒。請確認時區設定。');
    return { days, bars: bars.length, bad, step };
  }

  const api = { importText, parseCSV, buildDays, valueArea, toNY, parseTime };
  if (typeof module !== 'undefined') module.exports = api; else root.Importer = api;
})(typeof window !== 'undefined' ? window : globalThis);
