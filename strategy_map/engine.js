// 策略進出場地圖：每種策略的進出場判斷（瀏覽器與 node 共用）
// 研究用途，不是投資建議。i = 09:30 起第幾分鐘（0 = 09:30，179 = 12:29）。
(function (root) {
  const PT = 20, COST = 48, LAST = 179;

  function pad(a, n) { const x = new Float64Array(180).fill(NaN); for (let i = 0; i < n; i++) x[i] = a[i]; return x; }
  function decode(r) {
    const n = r.o.length, f = a => pad(a.map(x => (r.b + x) / 4), n);
    return Object.assign({}, r, { O: f(r.o), H: f(r.h), L: f(r.l), C: f(r.c), V: pad(r.v, n), n });
  }
  // 自己輸入的資料：meta 是價位，o/h/l/c/v 是 09:30 起的 1 分K（最多 180 根，可以只到盤中某一分鐘）
  function fromBars(meta, o, h, l, c, v) {
    const n = Math.min(180, o.length);
    return Object.assign({}, meta, { O: pad(o, n), H: pad(h, n), L: pad(l, n), C: pad(c, n), V: pad(v, n), n });
  }

  const tm = i => { const m = 570 + i; return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
  const banned = ei => ei >= 28 && ei <= 32;                 // 09:58–10:02 不進場（經濟數據）
  const okE = (ei, last = 120) => ei >= 5 && ei <= last && !banned(ei);

  // 5 分K：第 k 根涵蓋第 5k～5k+4 分鐘，在第 5k+4 分鐘收盤
  function bars5(D) {
    const n = 36, o = [], h = [], l = [], c = [], v = [];
    for (let k = 0; k < n; k++) {
      let hi = -Infinity, lo = Infinity, vol = 0;
      for (let i = 5 * k; i < 5 * k + 5; i++) { hi = Math.max(hi, D.H[i]); lo = Math.min(lo, D.L[i]); vol += D.V[i]; }
      o.push(D.O[5 * k]); h.push(hi); l.push(lo); c.push(D.C[5 * k + 4]); v.push(vol);
    }
    return { o, h, l, c, v };
  }

  // 進場後逐分鐘追蹤：同一分鐘兩邊都碰到算止損（和凍結規則一樣保守）
  function sim(D, ei, dir, tp, sl, extra) {
    const n = D.n == null ? 180 : D.n;
    if (ei >= n) {   // 訊號剛出現在最後一根，下一分鐘才進場：用最後收盤估計
      const e = D.C[n - 1];
      return Object.assign({ dir, sigIdx: ei - 1, entryIdx: ei, entry: e, tp: e + dir * tp, sl: e - dir * sl, exitIdx: Infinity, exit: NaN, reason: 'pending', net: 0, est: true }, extra || {});
    }
    const e = D.O[ei], tpPx = e + dir * tp, slPx = e - dir * sl, end = Math.min(LAST, n - 1);
    let exitIdx = LAST, exit = D.C[LAST], reason = 'time';
    if (end < LAST) { exitIdx = Infinity; exit = D.C[end]; reason = 'open'; }
    for (let i = ei; i <= end; i++) {
      if (dir === 1 ? D.L[i] <= slPx : D.H[i] >= slPx) { exitIdx = i; exit = slPx; reason = 'sl'; break; }
      if (dir === 1 ? D.H[i] >= tpPx : D.L[i] <= tpPx) { exitIdx = i; exit = tpPx; reason = 'tp'; break; }
    }
    const pts = dir * (exit - e);
    if (reason === 'open') return Object.assign({ dir, sigIdx: ei - 1, entryIdx: ei, entry: e, tp: tpPx, sl: slPx, exitIdx, exit, reason, net: 0 }, extra || {});
    return Object.assign({ dir, sigIdx: ei - 1, entryIdx: ei, entry: e, tp: tpPx, sl: slPx, exitIdx, exit, reason, net: pts * PT - COST }, extra || {});
  }

  function dayHiLo(D, upto) { let hi = -Infinity, lo = Infinity; for (let i = 0; i <= upto; i++) { hi = Math.max(hi, D.H[i]); lo = Math.min(lo, D.L[i]); } return [hi, lo]; }
  const fmt = x => x == null ? '—' : x.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  // ---------- I2 價值區回歸 ----------
  function i2(D, ver) {
    if (D.val == null || D.vah == null) return { why: '沒有前一日價值區（資料要包含前一天的盤中，或在左側自己填 VAL／VAH）' };
    const dir = D.O[0] > D.vah ? -1 : D.O[0] < D.val ? 1 : 0;
    const width = D.vah - D.val, dist = dir === -1 ? D.O[0] - D.vah : dir === 1 ? D.val - D.O[0] : null;
    const h1 = dir !== 0 && dist <= width;
    if (dir === 0) return { why: `開盤 ${fmt(D.O[0])} 在價值區內（${fmt(D.val)}～${fmt(D.vah)}）→ 今天不做`, wait: null };
    const pre = { wait: dir === 1 ? `開在 VAL 下方，等 2 根 5 分K 收回區內才做多` : `開在 VAH 上方，等 2 根 5 分K 收回區內才做空`, side: dir };
    let cnt = 0;
    for (let k = 0; k < 24; k++) {
      const i1 = 5 * k + 4, c = D.C[i1];
      cnt = (c >= D.val && c <= D.vah) ? cnt + 1 : 0;
      if (cnt >= 2) {
        const ei = i1 + 1, last = ver === 2 ? 89 : 120;
        if (banned(ei)) return Object.assign(pre, { why: `${tm(i1)} 第 2 根收回區內，但進場時間落在 09:58–10:02 禁止時段 → 取消` });
        if (ei > last) return Object.assign(pre, { why: `${tm(i1)} 才收回區內，超過最晚進場 ${ver === 2 ? '10:59' : '11:30'} → 不做` });
        const t = sim(D, ei, dir, ver === 2 ? 80 : 55, 145, { note: `H1（只記錄）：開盤距離 ${fmt(dist)}／寬度 ${fmt(width)} → ${h1 ? '會做' : '跳過'}` });
        t.why = `開在${dir === 1 ? ' VAL 下方' : ' VAH 上方'}，${tm(i1 - 5)}、${tm(i1)} 兩根 5 分K 收在區內 → ${tm(ei)} 開盤${dir === 1 ? '做多' : '做空'}，目標往區的另一邊`;
        return Object.assign(pre, { trades: [t] });
      }
    }
    return Object.assign(pre, { why: '價格一直沒有連續 2 根 5 分K 收回價值區 → 沒有進場' });
  }

  // ---------- 突破共振（近似重建） ----------
  function subSignals(D) {
    const B = bars5(D), ev = [];
    for (let k = 6; k < 36; k++) {
      const hh = Math.max(...B.h.slice(k - 6, k)), ll = Math.min(...B.l.slice(k - 6, k));
      if (B.c[k] > hh) ev.push({ i: 5 * k + 4, dir: 1, type: 'ch' });
      else if (B.c[k] < ll) ev.push({ i: 5 * k + 4, dir: -1, type: 'ch' });
    }
    let orh = -Infinity, orl = Infinity;
    for (let i = 0; i < 30; i++) { orh = Math.max(orh, D.H[i]); orl = Math.min(orl, D.L[i]); }
    let up = false, dn = false;
    for (let i = 30; i <= LAST; i++) {
      if (!up && D.C[i] > orh) { up = true; ev.push({ i, dir: 1, type: 'orb' }); }
      if (!dn && D.C[i] < orl) { dn = true; ev.push({ i, dir: -1, type: 'orb' }); }
    }
    ev.sort((a, b) => a.i - b.i || (a.type === 'orb' ? 1 : -1));
    return { ev, orh, orl };
  }
  function resonance(D, ver) {
    if (ver === 2 && D.pdh == null) return { why: '沒有前一日高低（在左側填入）' };
    const outside = D.O[0] > D.pdh || D.O[0] < D.pdl;
    if (ver === 2 && !outside) return { why: `開盤 ${fmt(D.O[0])} 在前日高低範圍內（${fmt(D.pdl)}～${fmt(D.pdh)}）→ v2 今天不做` };
    const { ev, orh, orl } = subSignals(D);
    const wait = `等「前 6 根 5 分K 通道突破」和「開盤 30 分鐘區間（${fmt(orl)}～${fmt(orh)}）突破」15 分鐘內同方向出現`;
    for (let a = 0; a < ev.length; a++) {
      const x = ev[a];
      const win = ev.filter(y => y.i <= x.i && y.i >= x.i - 15);
      const mate = win.find(y => y.type !== x.type && y.dir === x.dir);
      if (!mate) continue;
      if (win.some(y => y.dir !== x.dir)) continue;          // 15 分鐘內有反方向的子訊號 → 不進場
      const ei = x.i + 1;
      if (!okE(ei, ver === 2 ? 89 : 120)) {
        if (ei > (ver === 2 ? 89 : 120)) return { wait, why: `共振在 ${tm(x.i)} 才出現，超過最晚進場 → 不做` };
        continue;
      }
      const t = sim(D, ei, x.dir, 55, 145);
      t.why = `${tm(mate.i)} ${mate.type === 'ch' ? '通道突破' : '開盤區間突破'}、${tm(x.i)} ${x.type === 'ch' ? '通道突破' : '開盤區間突破'}，同方向 → ${tm(ei)} ${x.dir === 1 ? '做多' : '做空'}`;
      return { wait, trades: [t] };
    }
    return { wait, why: '兩種突破沒有在 15 分鐘內同方向出現 → 沒有進場' };
  }

  // ---------- 參考點：前高／前低假突破 ----------
  function fakeBreak(D, side) {
    const lvl = side === -1 ? D.pdh : D.pdl;
    if (lvl == null) return { why: '沒有前一日高低（在左側填入）' };
    const wait = side === -1 ? `等價格衝過前日高 ${fmt(lvl)} 後，5 分K 收回下方` : `等價格跌破前日低 ${fmt(lvl)} 後，5 分K 收回上方`;
    for (let k = 0; k < 24; k++) {
      const i1 = 5 * k + 4, ei = i1 + 1;
      const [hi, lo] = dayHiLo(D, i1);
      const hit = side === -1 ? (hi > lvl && D.C[i1] < lvl) : (lo < lvl && D.C[i1] > lvl);
      if (hit && okE(ei)) {
        const t = sim(D, ei, side, 55, 145);
        t.why = side === -1 ? `今天高點 ${fmt(hi)} 衝過前日高 ${fmt(lvl)}，${tm(i1)} 5 分K 收回下方 ${fmt(D.C[i1])} → ${tm(ei)} 做空`
                            : `今天低點 ${fmt(lo)} 跌破前日低 ${fmt(lvl)}，${tm(i1)} 5 分K 收回上方 ${fmt(D.C[i1])} → ${tm(ei)} 做多`;
        return { wait, trades: [t] };
      }
    }
    return { wait, why: side === -1 ? '沒有出現「衝過前日高又收回」→ 沒有進場' : '沒有出現「跌破前日低又收回」→ 沒有進場' };
  }

  // ---------- 參考點：區邊回歸（開在區內、盤中衝出又收回） ----------
  function edgeReturn(D) {
    if (D.val == null || D.vah == null) return { why: '沒有前一日價值區' };
    if (D.O[0] > D.vah || D.O[0] < D.val) return { why: '只在開盤落在價值區內的日子做；今天開在區外' };
    let side = 0, cnt = 0;
    const wait = '等 5 分K 收到價值區外，再連續 2 根收回區內，往回做';
    for (let k = 0; k < 24; k++) {
      const i1 = 5 * k + 4, c = D.C[i1];
      if (c > D.vah) { side = -1; cnt = 0; }
      else if (c < D.val) { side = 1; cnt = 0; }
      else if (side !== 0 && ++cnt >= 2) {
        const ei = i1 + 1;
        if (!okE(ei)) return { wait, why: `${tm(i1)} 收回，但進場時間不允許 → 不做` };
        const t = sim(D, ei, side, 55, 145);
        t.why = `盤中 5 分K 收到${side === -1 ? ' VAH 上方' : ' VAL 下方'}，${tm(i1)} 連續 2 根收回區內 → ${tm(ei)} ${side === 1 ? '做多' : '做空'}`;
        return { wait, trades: [t] };
      }
    }
    return { wait, why: '沒有「衝出區外再收回」→ 沒有進場' };
  }

  // ---------- 單獨突破 ----------
  function channel6(D) {
    const B = bars5(D), wait = '等 5 分K 收盤突破前 6 根 5 分K 的最高／最低';
    for (let k = 6; k < 36; k++) {
      const hh = Math.max(...B.h.slice(k - 6, k)), ll = Math.min(...B.l.slice(k - 6, k));
      const dir = B.c[k] > hh ? 1 : B.c[k] < ll ? -1 : 0, ei = 5 * k + 5;
      if (!dir) continue;
      if (ei > 60) break;
      if (!okE(ei)) continue;
      const t = sim(D, ei, dir, 55, 145);
      t.why = `${tm(5 * k + 4)} 5 分K 收 ${fmt(B.c[k])}，${dir === 1 ? '高於前 6 根最高 ' + fmt(hh) : '低於前 6 根最低 ' + fmt(ll)} → ${tm(ei)} 順勢${dir === 1 ? '做多' : '做空'}`;
      return { wait, trades: [t] };
    }
    return { wait, why: '10:30 前沒有通道突破 → 沒有進場' };
  }
  function orb30(D) {
    const B = bars5(D);
    let orh = -Infinity, orl = Infinity;
    for (let i = 0; i < 30; i++) { orh = Math.max(orh, D.H[i]); orl = Math.min(orl, D.L[i]); }
    const wait = `等 5 分K 收盤突破開盤 30 分鐘區間 ${fmt(orl)}～${fmt(orh)}`;
    for (let k = 6; k < 12; k++) {
      const dir = B.c[k] > orh ? 1 : B.c[k] < orl ? -1 : 0, ei = 5 * k + 5;
      if (!dir || !okE(ei)) continue;
      const t = sim(D, ei, dir, 55, 145);
      t.why = `${tm(5 * k + 4)} 5 分K 收在開盤區間${dir === 1 ? '上方' : '下方'} → ${tm(ei)} ${dir === 1 ? '做多' : '做空'}`;
      return { wait, trades: [t] };
    }
    return { wait, why: '10:30 前沒有收盤突破開盤區間 → 沒有進場' };
  }

  // ---------- I1 隔夜庫存修正（近似） ----------
  function overnight(D) {
    if (D.onh == null) return { why: '沒有隔夜資料' };
    const rng = D.onh - D.onl, pos = (D.O[0] - D.onl) / rng;
    if (rng < 40) return { why: `隔夜區間只有 ${rng.toFixed(0)} 點，太窄 → 不做` };
    const side = pos >= 0.8 ? -1 : pos <= 0.2 ? 1 : 0;
    if (!side) return { why: `開盤在隔夜區間的 ${(pos * 100).toFixed(0)}% 位置，不在頂部或底部 20% → 不做` };
    const wait = side === -1 ? `開在隔夜區間頂部，等 10:03 前都沒再突破隔夜高 ${fmt(D.onh)}` : `開在隔夜區間底部，等 10:03 前都沒再跌破隔夜低 ${fmt(D.onl)}`;
    if ((D.n || 180) <= 33) return { wait, side };
    const [hi, lo] = dayHiLo(D, 32);
    if (side === -1 ? hi > D.onh : lo < D.onl) return { wait, side, why: side === -1 ? '開盤後又突破了隔夜高 → 庫存沒有修正，不做' : '開盤後又跌破了隔夜低 → 不做' };
    const t = sim(D, 33, side, 55, 145);
    t.why = `開盤在隔夜區間 ${(pos * 100).toFixed(0)}% 位置，開盤後 30 分鐘沒能${side === -1 ? '再創隔夜高' : '再創隔夜低'} → 10:03 ${side === 1 ? '做多' : '做空'}`;
    return { wait, side, trades: [t] };
  }

  // ---------- I4 窄初始平衡區突破 ----------
  function narrowIB(D) {
    if (!D.atr) return { why: '沒有 ATR20（資料不足）' };
    if ((D.n || 180) < 60) return { wait: '等第一小時（09:30–10:29）走完，看區間夠不夠窄' };
    const [ibh, ibl] = dayHiLo(D, 59), w = ibh - ibl, lim = 0.35 * D.atr;
    if (w > lim) return { why: `第一小時區間 ${w.toFixed(0)} 點，大於 0.35×ATR20＝${lim.toFixed(0)} 點，不算窄 → 不做` };
    const B = bars5(D), wait = `第一小時區間只有 ${w.toFixed(0)} 點（窄），等 10:30 後 5 分K 收盤突破 ${fmt(ibl)}～${fmt(ibh)}`;
    for (let k = 12; k < 24; k++) {
      const dir = B.c[k] > ibh ? 1 : B.c[k] < ibl ? -1 : 0, ei = 5 * k + 5;
      if (!dir || !okE(ei)) continue;
      const t = sim(D, ei, dir, 55, 145);
      t.why = `第一小時區間窄（${w.toFixed(0)} 點），${tm(5 * k + 4)} 收盤突破 → ${tm(ei)} ${dir === 1 ? '做多' : '做空'}`;
      return { wait, trades: [t] };
    }
    return { wait, why: '11:30 前沒有突破 → 沒有進場' };
  }

  // ---------- S2 急跌後反彈（已否定） ----------
  function dropBounce(D) {
    if (!D.atr) return { why: '沒有 ATR20' };
    const need = 0.5 * D.atr, wait = `等價格從今天高點跌超過 0.5×ATR20（${need.toFixed(0)} 點），再等 1 分K 收盤突破前 3 根高點`;
    let hi = -Infinity, armed = -1;
    for (let i = 0; i < 120; i++) {
      hi = Math.max(hi, D.H[i]);
      if (armed < 0 && hi - D.L[i] >= need) armed = i;
      if (armed >= 0 && i > armed && i >= 3 && D.C[i] > Math.max(D.H[i - 1], D.H[i - 2], D.H[i - 3])) {
        const ei = i + 1;
        if (!okE(ei)) continue;
        const t = sim(D, ei, 1, 55, 145);
        t.why = `${tm(armed)} 從高點 ${fmt(hi)} 跌超過 ${need.toFixed(0)} 點，${tm(i)} 1 分K 突破前 3 根高點 → ${tm(ei)} 接反彈做多`;
        return { wait, trades: [t] };
      }
    }
    return { wait, why: '沒有出現夠深的急跌 → 沒有進場' };
  }

  // ---------- S4 掃隔夜高低後收回（ICT 掃流動性） ----------
  function sweep(D) {
    if (D.onh == null) return { why: '沒有隔夜資料' };
    const B = bars5(D), wait = `等價格刺穿隔夜高 ${fmt(D.onh)} 或隔夜低 ${fmt(D.onl)}，3 根 5 分K 內收回`;
    let upK = -1, dnK = -1;
    for (let k = 0; k < 24; k++) {
      if (upK < 0 && B.h[k] > D.onh) upK = k;
      if (dnK < 0 && B.l[k] < D.onl) dnK = k;
      const ei = 5 * k + 5;
      if (upK >= 0 && k - upK <= 3 && B.c[k] < D.onh && okE(ei)) {
        const t = sim(D, ei, -1, 55, 145); t.why = `${tm(5 * upK)} 刺穿隔夜高，${tm(5 * k + 4)} 收回下方 → ${tm(ei)} 做空`; return { wait, trades: [t] };
      }
      if (dnK >= 0 && k - dnK <= 3 && B.c[k] > D.onl && okE(ei)) {
        const t = sim(D, ei, 1, 55, 145); t.why = `${tm(5 * dnK)} 刺穿隔夜低，${tm(5 * k + 4)} 收回上方 → ${tm(ei)} 做多`; return { wait, trades: [t] };
      }
    }
    return { wait, why: '沒有掃過隔夜高低又收回 → 沒有進場' };
  }

  // ---------- 委託簿（Orderbook）代理：由 1 分K 成交量估算 ----------
  // 估計主動買賣差（delta）：收在 K 棒上半部當成買方主動多、下半部當成賣方主動多
  function estDelta(D, i) { const r = D.H[i] - D.L[i]; return r > 0 ? D.V[i] * (2 * (D.C[i] - D.L[i]) / r - 1) : 0; }
  function keyLevels(D) {
    return [['VAH', D.vah], ['VAL', D.val], ['前日高', D.pdh], ['前日低', D.pdl], ['隔夜高', D.onh], ['隔夜低', D.onl]].filter(x => x[1] != null);
  }
  function absorption(D) {
    const lv = keyLevels(D), wait = '等價格碰到關鍵價位時爆出 2.5 倍以上的量，但被擋回（K 棒收在反方向那一半）';
    for (let i = 10; i < 120; i++) {
      let avg = 0; const n = Math.min(20, i); for (let j = i - n; j < i; j++) avg += D.V[j]; avg /= n;
      if (D.V[i] < 2.5 * avg) continue;
      const r = D.H[i] - D.L[i]; if (r <= 0) continue;
      const pos = (D.C[i] - D.L[i]) / r;
      for (const [name, p] of lv) {
        let side = 0;
        if (D.H[i] >= p && D.C[i] < p && pos <= 0.4) side = -1;        // 往上衝被賣方掛單吸收
        else if (D.L[i] <= p && D.C[i] > p && pos >= 0.6) side = 1;    // 往下砸被買方掛單吸收
        if (!side || !okE(i + 1)) continue;
        const t = sim(D, i + 1, side, 55, 145);
        t.why = `${tm(i)} 在${name} ${fmt(p)} 爆量 ${D.V[i].toLocaleString()} 口（前 20 分鐘平均的 ${(D.V[i] / avg).toFixed(1)} 倍），價格卻被擋回 → 視為大單吸收，${tm(i + 1)} ${side === 1 ? '做多' : '做空'}`;
        return { wait, trades: [t] };
      }
    }
    return { wait, why: '沒有在關鍵價位出現「爆量但被擋回」→ 沒有進場' };
  }
  function imbalance(D) {
    const B = bars5(D), wait = '等最近 15 分鐘估計買賣差超過總量的 ±35%，而且 5 分K 收盤突破前 3 根高低';
    for (let k = 3; k < 24; k++) {
      const i1 = 5 * k + 4; let dl = 0, vol = 0;
      for (let i = i1 - 14; i <= i1; i++) { dl += estDelta(D, i); vol += D.V[i]; }
      const ratio = vol ? dl / vol : 0;
      const hh = Math.max(B.h[k - 1], B.h[k - 2], B.h[k - 3]), ll = Math.min(B.l[k - 1], B.l[k - 2], B.l[k - 3]);
      const dir = ratio >= 0.35 && B.c[k] > hh ? 1 : ratio <= -0.35 && B.c[k] < ll ? -1 : 0;
      const ei = i1 + 1;
      if (!dir || !okE(ei)) continue;
      const t = sim(D, ei, dir, 55, 145);
      t.why = `${tm(i1)} 最近 15 分鐘估計${dir === 1 ? '買方' : '賣方'}主動量多出 ${(Math.abs(ratio) * 100).toFixed(0)}%，5 分K 同方向突破 → ${tm(ei)} ${dir === 1 ? '做多' : '做空'}`;
      return { wait, trades: [t] };
    }
    return { wait, why: '沒有出現明顯的單邊主動量＋突破 → 沒有進場' };
  }

  const STRATS = [
    { id: 'i2v1', name: 'I2 價值區回歸 v1', short: 'I2', group: 'rule', run: D => i2(D, 1) },
    { id: 'i2v2', name: 'I2 v2（11:00 前、止盈 80）', short: 'I2v2', group: 'rule', run: D => i2(D, 2) },
    { id: 'res1', name: '突破共振 v1', short: '共振', group: 'rule', run: D => resonance(D, 1) },
    { id: 'res2', name: '突破共振 v2（開在前日範圍外）', short: '共振v2', group: 'rule', run: D => resonance(D, 2) },
    { id: 'pdhF', name: '前高假突破 空', short: '前高空', group: 'ref', run: D => fakeBreak(D, -1) },
    { id: 'pdlF', name: '前低假突破 多', short: '前低多', group: 'ref', run: D => fakeBreak(D, 1) },
    { id: 'edge', name: '區邊回歸（開在區內）', short: '區邊', group: 'ref', run: edgeReturn },
    { id: 'ch6', name: '前 6 根通道突破', short: '通道', group: 'tested', run: channel6 },
    { id: 'orb', name: '開盤 30 分鐘區間突破', short: 'ORB', group: 'tested', run: orb30 },
    { id: 'i1', name: 'I1 隔夜庫存修正', short: 'I1', group: 'tested', run: overnight },
    { id: 'i4', name: 'I4 窄初始平衡區突破', short: 'I4', group: 'tested', run: narrowIB },
    { id: 's2', name: '急跌後接反彈', short: '急跌', group: 'tested', run: dropBounce },
    { id: 's4', name: '掃隔夜高低後收回', short: '掃蕩', group: 'tested', run: sweep },
    { id: 'obA', name: '委託簿：關鍵價位大單吸收', short: '吸收', group: 'ob', run: absorption },
    { id: 'obI', name: '委託簿：主動買賣失衡', short: '失衡', group: 'ob', run: imbalance },
  ];

  function runAll(D) {
    const out = {};
    for (const s of STRATS) { const r = s.run(D); r.trades = r.trades || []; out[s.id] = r; }
    return out;
  }

  // 價量階梯（給委託簿面板）：到第 upto 分鐘為止，每 5 點一格的估計買／賣量
  function ladder(D, upto, bin = 5) {
    const m = new Map();
    for (let i = 0; i <= upto; i++) {
      if (!(D.H[i] >= D.L[i])) continue;
      const a = Math.floor(D.L[i] / bin), b = Math.floor(D.H[i] / bin), n = b - a + 1;
      const r = D.H[i] - D.L[i], buyFrac = r > 0 ? (D.C[i] - D.L[i]) / r : 0.5;
      for (let k = a; k <= b; k++) {
        const x = m.get(k) || [0, 0];
        x[0] += D.V[i] * buyFrac / n; x[1] += D.V[i] * (1 - buyFrac) / n; m.set(k, x);
      }
    }
    return m;
  }

  const api = { decode, fromBars, runAll, STRATS, tm, fmt, bars5, ladder, estDelta, keyLevels, PT, COST };
  if (typeof module !== 'undefined') module.exports = api; else root.Engine = api;
})(typeof window !== 'undefined' ? window : globalThis);
