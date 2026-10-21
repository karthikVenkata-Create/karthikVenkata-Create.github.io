// Costco DCF model, ported from valuation_engine.py so it can run in the browser.
// Same math as the Python version: 5-year FCFF, Gordon-growth terminal value,
// reverse DCF four ways, and a Monte Carlo over the same bear/base/bull ranges.

(function (root) {
  "use strict";

  // FY2026 actuals, $ millions (Costco FY2026 earnings release)
  const BASE = {
    revenue: 303200,
    cash: 20200,
    totalDebt: 6162,
    shares: 443.27,
    taxRate: 0.24,
  };

  // Monte Carlo ranges, same as assumptions.json
  const SCENARIOS = {
    growthShift: [-0.04, 0.0, 0.04],
    marginShift: [-0.005, 0.0, 0.01],
    waccShift: [0.015, 0.0, -0.015],
    terminalG: [0.02, 0.03, 0.04],
  };

  // Straight line from first to last value over n years
  function fade(first, last, n) {
    const out = [];
    for (let t = 0; t < n; t++) out.push(n === 1 ? first : first + (last - first) * (t / (n - 1)));
    return out;
  }

  // a = { growth: [...], margin: [...], salesToCapital, terminalG, wacc }
  function dcf(a, opt) {
    opt = opt || {};
    const gShift = opt.growthShift || 0;
    const mShift = opt.marginShift || 0;
    const tg = opt.terminalG !== undefined ? opt.terminalG : a.terminalG;
    const wacc = opt.wacc !== undefined ? opt.wacc : a.wacc;
    const n = a.growth.length;

    let revPrev = BASE.revenue, pvFcff = 0;
    const rows = [];
    for (let t = 0; t < n; t++) {
      const rev = revPrev * (1 + a.growth[t] + gShift);
      const ebit = rev * (a.margin[t] + mShift);
      const nopat = ebit * (1 - BASE.taxRate);
      const reinvest = (rev - revPrev) / a.salesToCapital;
      const fcff = nopat - reinvest;
      pvFcff += fcff / Math.pow(1 + wacc, t + 1);
      rows.push({ year: "FY" + (2027 + t), rev, ebit, nopat, reinvest, fcff });
      revPrev = rev;
    }
    const revTv = revPrev * (1 + tg);
    const fcffTv = revTv * (a.margin[n - 1] + mShift) * (1 - BASE.taxRate)
                 - (revTv - revPrev) / a.salesToCapital;
    const tv = fcffTv / Math.max(wacc - tg, 0.005);          // guard against wacc <= g
    const pvTv = tv / Math.pow(1 + wacc, n);
    const ev = pvFcff + pvTv;
    const equity = ev + BASE.cash - BASE.totalDebt;
    return { value: equity / BASE.shares, rows, pvFcff, pvTv, ev, equity, tvShare: pvTv / ev };
  }

  function value(a, opt) { return dcf(a, opt).value; }

  function solve(f, lo, hi) {           // bisection, f increasing; NaN if not reachable
    if (f(hi) < 0) return NaN;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (f(mid) < 0) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  function avg(xs) { return xs.reduce((s, x) => s + x, 0) / xs.length; }

  function reverseDcf(a, price, runwayGrowth) {
    runwayGrowth = runwayGrowth || 0.10;
    const g5 = avg(a.growth) + solve(x => value(a, { growthShift: x }) - price, -0.10, 0.60);
    const gTerm = solve(g => value(a, { terminalG: g }) - price, 0.0, a.wacc - 0.0051);
    let years = null;
    const lastM = a.margin[a.margin.length - 1];
    for (let n = 1; n <= 60; n++) {
      const m = a.margin.concat(Array(n).fill(lastM)).slice(0, n);
      const r = Object.assign({}, a, { growth: Array(n).fill(runwayGrowth), margin: m });
      if (value(r) >= price) { years = n; break; }
    }
    const margin = lastM + solve(x => value(a, { marginShift: x }) - price, -0.03, 0.15);
    return { g5, gTerm, years, runwayGrowth, margin };
  }

  // Discount rate that makes the DCF equal the price: the yearly return a buyer
  // at that price should expect if the forecast plays out
  function impliedReturn(a, price) {
    return solve(w => price - value(a, { wacc: w, terminalG: Math.min(a.terminalG, w - 0.0051) }), 0.035, 0.20);
  }

  function sensitivity(a) {
    const waccs = [-0.01, -0.005, 0, 0.005, 0.01].map(d => a.wacc + d);
    const gs = [0.02, 0.025, 0.03, 0.035, 0.04];
    return { waccs, gs, grid: waccs.map(w => gs.map(g => value(a, { wacc: w, terminalG: g }))) };
  }

  // Seeded RNG (mulberry32) so the same inputs always give the same simulation
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function triangular(r, spec, center) {
    const [lo, mode, hi] = spec.slice().sort((x, y) => x - y).map(v => v + (center || 0));
    const u = r(), c = (mode - lo) / (hi - lo);
    return u < c ? lo + Math.sqrt(u * (hi - lo) * (mode - lo))
                 : hi - Math.sqrt((1 - u) * (hi - lo) * (hi - mode));
  }

  function percentile(sorted, p) {
    const idx = (sorted.length - 1) * p, lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  function monteCarlo(a, price, n, seed) {
    n = n || 10000;
    const r = rng(seed || 42), vals = new Array(n);
    for (let i = 0; i < n; i++) {
      vals[i] = value(a, {
        wacc: triangular(r, SCENARIOS.waccShift, a.wacc),
        growthShift: triangular(r, SCENARIOS.growthShift),
        marginShift: triangular(r, SCENARIOS.marginShift),
        terminalG: triangular(r, SCENARIOS.terminalG),
      });
    }
    const s = vals.slice().sort((x, y) => x - y);
    return {
      values: vals, p10: percentile(s, 0.1), p50: percentile(s, 0.5), p90: percentile(s, 0.9),
      shareBelow: vals.filter(v => v < price).length / n,
    };
  }

  function rating(target, price) {
    const up = target / price - 1;
    return { upside: up, rating: up > 0.10 ? "BUY" : up < -0.10 ? "SELL" : "HOLD" };
  }

  const api = { BASE, SCENARIOS, fade, dcf, value, reverseDcf, impliedReturn, sensitivity, monteCarlo, rating };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CostcoModel = api;
})(this);
