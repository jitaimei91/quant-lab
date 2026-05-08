(async () => {
  // ─── Utilities ────────────────────────────────────────────────────────────
  const fmt = (v, digits = 2) => v == null ? "-" : v.toFixed(digits);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  const PERIOD_CONTEXT = {
    "2008-crisis": "S&P 500 lost ~50% peak-to-trough between Oct 2007 and Mar 2009. Watch which bots stayed defensive. 1 walk-forward test window (2009).",
    "2020-covid":  "VIX spiked above 80 in March 2020. The V-shaped recovery followed within months. Watch how the regime kill-switch behaved. 2 walk-forward test windows (2020, 2021).",
    "2022-rates":  "Bonds and stocks fell together in 2022 — uncommon. The Fed hiked rates from 0.25% to 5.5%. Tested the diversification thesis and hit trend-following bots hard. 3 walk-forward test windows (2023).",
  };

  const palette = [
    "#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899",
    "#79c0ff", "#56d364", "#ffa657", "#d2a8ff", "#f78166",
    "#1f6feb", "#238636", "#9e6a03", "#6e40c9", "#bf4b8a",
    "#388bfd",
  ];

  // ─── Stats helpers ────────────────────────────────────────────────────────
  function currentDrawdown(navSeries) {
    let peak = -Infinity, maxDD = 0;
    for (const p of navSeries) {
      if (p.nav > peak) peak = p.nav;
      const dd = (peak - p.nav) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
  }

  function currentReturn(navSeries) {
    if (!navSeries || navSeries.length < 2) return 0;
    const first = navSeries[0].nav, last = navSeries[navSeries.length - 1].nav;
    return (last - first) / first;
  }

  // Annualised Sharpe from a series of NAVs (daily returns, rf=0)
  function rollingAnnualisedSharpe(navSeries) {
    if (!navSeries || navSeries.length < 5) return null;
    const rets = [];
    for (let i = 1; i < navSeries.length; i++) {
      rets.push((navSeries[i].nav - navSeries[i - 1].nav) / navSeries[i - 1].nav);
    }
    const n = rets.length;
    const mean = rets.reduce((s, r) => s + r, 0) / n;
    const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    if (std === 0) return null;
    return (mean / std) * Math.sqrt(252);
  }

  // ─── Chart management ─────────────────────────────────────────────────────
  const charts = [];   // { chart, labels, datasets, allLabels }

  function destroyCharts() {
    for (const c of charts) c.chart.destroy();
    charts.length = 0;
  }

  function buildChartScaffolding(curvesData, botColorMap) {
    destroyCharts();
    const container = document.getElementById("curves-container");
    container.innerHTML = "";

    for (const win of curvesData.windows) {
      const botMap = curvesData.curves[win];
      if (!botMap) continue;

      const block = document.createElement("div");
      block.dataset.window = win;
      block.innerHTML = `<h3 class="win-label">Window: ${win}</h3><canvas height="120"></canvas>`;
      container.appendChild(block);

      // Build master date axis for this window
      const allDates = new Set();
      for (const series of Object.values(botMap)) series.forEach(p => allDates.add(p.date));
      const allLabels = Array.from(allDates).sort();

      const sortedBots = Object.keys(botMap).sort();
      const datasets = sortedBots.map(bot => ({
        label:           bot,
        data:            new Array(allLabels.length).fill(null),   // filled progressively
        spanGaps:        true,
        borderColor:     botColorMap[bot] ?? "#58a6ff",
        backgroundColor: "transparent",
        pointRadius:     0,
        borderWidth:     1.5,
        borderDash:      [],
      }));

      const ctx = block.querySelector("canvas").getContext("2d");
      const chart = new Chart(ctx, {
        type: "line",
        data: { labels: allLabels, datasets },
        options: {
          responsive:  true,
          animation:   false,
          plugins: {
            legend: {
              labels: { color: "#e6edf3", font: { size: 11 } },
            },
            tooltip: {
              callbacks: {
                label: (c) => `${c.dataset.label}: $${c.parsed.y?.toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
              },
            },
          },
          scales: {
            x: {
              ticks: { color: "#8b949e", maxTicksLimit: 12 },
              grid:  { color: "rgba(48,54,61,0.6)" },
            },
            y: {
              ticks: { color: "#8b949e" },
              grid:  { color: "rgba(48,54,61,0.6)" },
            },
          },
        },
      });

      // Store everything we need for progressive updates
      charts.push({
        chart,
        win,
        botMap,
        sortedBots,
        allLabels,
        // per-bot lookup: date → index in allLabels
        dateIndex: Object.fromEntries(allLabels.map((d, i) => [d, i])),
      });
    }
  }

  // ─── Leaderboard ──────────────────────────────────────────────────────────
  function buildLeaderboardRows(bots) {
    // bots: [{bot_id, nav, sharpe, dd, ret, isLeader}]
    const tbody = document.querySelector("#period-leaderboard tbody");
    tbody.innerHTML = "";
    bots.forEach((b, idx) => {
      const tr = document.createElement("tr");
      tr.dataset.botId = b.bot_id;
      if (b.isLeader) tr.classList.add("leader-row");
      const ddPct  = b.dd  != null ? (b.dd * 100).toFixed(1) + "%" : "-";
      const retPct = b.ret != null ? (b.ret * 100).toFixed(2) + "%" : "-";
      const sharpe = b.sharpe != null ? fmt(b.sharpe) : "-";
      const sharpeClass = b.sharpe != null ? cls(b.sharpe) : "";
      const retClass    = b.ret    != null ? cls(b.ret) : "";

      // Drawdown bar: red intensity scales with depth, max at 50%
      const ddFrac   = Math.min(b.dd ?? 0, 0.50) / 0.50;       // 0–1
      const ddOpacity = 0.25 + ddFrac * 0.75;
      const ddBarWidth = (b.dd ?? 0) * 200;                     // px, capped elsewhere by CSS
      const ddBarHtml = `
        <div class="dd-bar-wrap">
          <div class="dd-bar" style="width:${Math.min(ddBarWidth, 100)}%;background:rgba(248,81,73,${ddOpacity.toFixed(2)})"></div>
          <span class="dd-bar-label neg">${ddPct}</span>
        </div>`;

      tr.innerHTML = `
        <td class="rank-cell">${idx + 1}</td>
        <td class="bot-cell${b.isLeader ? " leader-name" : ""}">${b.bot_id}${b.isLeader ? " <span class='leader-crown'>&#9733;</span>" : ""}</td>
        <td class="${sharpeClass}">${sharpe}</td>
        <td class="${retClass}">${retPct}</td>
        <td>${ddBarHtml}</td>
        <td>${b.days ?? "-"}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ─── Simulator class ───────────────────────────────────────────────────────
  class Simulator {
    constructor(curvesData, resultsData) {
      this.curvesData  = curvesData;
      this.resultsData = resultsData;

      // Flatten all windows into a sequential timeline
      // Each entry: { win, date, botMap }  (shared botMap ref)
      this._timeline = [];
      for (const win of curvesData.windows) {
        const botMap = curvesData.curves[win];
        if (!botMap) continue;
        const dates = new Set();
        for (const s of Object.values(botMap)) s.forEach(p => dates.add(p.date));
        const sortedDates = Array.from(dates).sort();
        for (const date of sortedDates) {
          this._timeline.push({ win, date, botMap });
        }
      }

      this._dayIdx    = 0;
      this._running   = false;
      this._speed     = 1;     // multiplier: 1 | 5 | 30
      this._lastTick  = 0;

      // Per-bot accumulated series across windows: bot_id → [{date, nav}]
      this._botSeries = {};
      // Per-bot: current window
      this._botWindow = {};

      // Build bots list from first window
      const firstWin    = curvesData.windows[0];
      this._botIds      = firstWin ? Object.keys(curvesData.curves[firstWin]).sort() : [];

      this._rafId = null;

      this._bindUI();
    }

    get totalDays() { return this._timeline.length; }

    _delay() {
      // ms per day: 1x=100ms, 5x=20ms, 30x=5ms
      const base = { 1: 100, 5: 20, 30: 5 };
      return base[this._speed] ?? 100;
    }

    _bindUI() {
      document.getElementById("btn-play").addEventListener("click",  () => this.play());
      document.getElementById("btn-pause").addEventListener("click", () => this.pause());
      document.getElementById("btn-skip").addEventListener("click",  () => this.skipToEnd());
      document.querySelectorAll(".sim-speed-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".sim-speed-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          this.setSpeed(parseInt(btn.dataset.speed, 10));
        });
      });
    }

    _updateStatusBar() {
      const entry = this._timeline[this._dayIdx] ?? this._timeline[this._timeline.length - 1];
      document.getElementById("sim-day-current").textContent = this._dayIdx;
      document.getElementById("sim-day-total").textContent   = this.totalDays;
      document.getElementById("sim-date").textContent        = entry?.date ?? "—";

      // Window label in leaderboard header
      const winLabel = document.getElementById("sim-window-label");
      if (entry) winLabel.textContent = entry.win;
    }

    // ── Core render for a given dayIdx ────────────────────────────────────
    _renderDay(dayIdx) {
      const entry = this._timeline[dayIdx];
      if (!entry) return;

      const { win, date, botMap } = entry;

      // Accumulate per-bot NAV up to this day
      for (const bot of this._botIds) {
        const series = botMap[bot];
        if (!series) continue;

        // Find the point for this date in the bot's series
        const pt = series.find(p => p.date === date);
        if (!pt) continue;

        if (!this._botSeries[bot]) this._botSeries[bot] = [];

        // If window changed, segment (don't carry over across windows)
        if (this._botWindow[bot] !== win) {
          this._botSeries[bot] = [];
          this._botWindow[bot] = win;
        }

        // Append if not already there
        const last = this._botSeries[bot].at(-1);
        if (!last || last.date !== date) {
          this._botSeries[bot].push(pt);
        }
      }

      // Update charts for the current window
      for (const c of charts) {
        if (c.win !== win) continue;
        const { chart, sortedBots, dateIndex } = c;

        for (let di = 0; di < sortedBots.length; di++) {
          const bot    = sortedBots[di];
          const botSer = this._botSeries[bot];
          if (!botSer) continue;

          // Write all accumulated points into the dataset data array
          const dataArr = chart.data.datasets[di].data;
          for (const pt of botSer) {
            const idx = dateIndex[pt.date];
            if (idx !== undefined) dataArr[idx] = pt.nav;
          }
        }
        chart.update("none");
      }

      // Compute live stats and re-rank leaderboard
      this._updateLeaderboard(win);
    }

    _updateLeaderboard(win) {
      const botStats = this._botIds.map(bot_id => {
        const series = this._botSeries[bot_id] ?? [];
        return {
          bot_id,
          sharpe: rollingAnnualisedSharpe(series),
          dd:     currentDrawdown(series),
          ret:    currentReturn(series),
          days:   series.length,
        };
      });

      // Sort by Sharpe descending (null last)
      botStats.sort((a, b) => {
        if (a.sharpe == null && b.sharpe == null) return 0;
        if (a.sharpe == null) return 1;
        if (b.sharpe == null) return -1;
        return b.sharpe - a.sharpe;
      });

      // Mark leader
      const leader = botStats[0]?.bot_id;
      botStats.forEach(b => b.isLeader = b.bot_id === leader);

      // Highlight leader's curve: thicker + brighter
      for (const c of charts) {
        if (c.win !== win) continue;
        c.sortedBots.forEach((bot, di) => {
          const ds = c.chart.data.datasets[di];
          ds.borderWidth = bot === leader ? 2.5 : 1.5;
          ds.borderColor = bot === leader
            ? lighten(botColorMap[bot] ?? "#58a6ff", 0.25)
            : (botColorMap[bot] ?? "#58a6ff");
        });
        c.chart.update("none");
      }

      buildLeaderboardRows(botStats);
    }

    // ── Playback loop ──────────────────────────────────────────────────────
    _loop(ts) {
      if (!this._running) return;
      if (this._dayIdx >= this.totalDays) {
        this._finish();
        return;
      }

      const elapsed = ts - this._lastTick;
      if (elapsed >= this._delay()) {
        this._renderDay(this._dayIdx);
        this._updateStatusBar();
        this._dayIdx++;
        this._lastTick = ts;
      }

      this._rafId = requestAnimationFrame(ts2 => this._loop(ts2));
    }

    _finish() {
      this._running = false;
      document.getElementById("btn-play").disabled  = true;
      document.getElementById("btn-pause").disabled = true;
      this._updateStatusBar();
    }

    // ── Public API ─────────────────────────────────────────────────────────
    play() {
      if (this._running) return;
      if (this._dayIdx >= this.totalDays) this._reset();
      this._running = true;
      document.getElementById("btn-play").disabled  = true;
      document.getElementById("btn-pause").disabled = false;
      this._lastTick = performance.now();
      this._rafId = requestAnimationFrame(ts => this._loop(ts));
    }

    pause() {
      if (!this._running) return;
      this._running = false;
      if (this._rafId) cancelAnimationFrame(this._rafId);
      document.getElementById("btn-play").disabled  = false;
      document.getElementById("btn-pause").disabled = true;
    }

    skipToEnd() {
      this.pause();
      // Fast-forward: clear charts and re-render final state statically
      this._reset();
      this._dayIdx = this.totalDays - 1;
      // Batch-fill all days up to the end
      for (let i = 0; i <= this._dayIdx; i++) this._renderDay(i);
      this._updateStatusBar();
      document.getElementById("btn-play").disabled  = true;
      document.getElementById("btn-pause").disabled = true;
    }

    setSpeed(multiplier) {
      this._speed = multiplier;
    }

    _reset() {
      this._dayIdx    = 0;
      this._botSeries = {};
      this._botWindow = {};
      // Clear chart data arrays
      for (const c of charts) {
        for (const ds of c.chart.data.datasets) {
          ds.data = new Array(c.allLabels.length).fill(null);
        }
        c.chart.update("none");
      }
      document.getElementById("btn-play").disabled  = false;
      document.getElementById("btn-pause").disabled = true;
      this._updateStatusBar();
    }
  }

  // ─── Color utilities ──────────────────────────────────────────────────────
  function lighten(hex, amount) {
    // Very simple: push RGB channels toward white by `amount` (0–1)
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const lerp = (v) => Math.round(v + (255 - v) * amount);
    return `rgb(${lerp(r)},${lerp(g)},${lerp(b)})`;
  }

  // ─── Global state ─────────────────────────────────────────────────────────
  let botColorMap = {};
  let simulator   = null;

  // ─── Period loader ────────────────────────────────────────────────────────
  async function loadPeriod(period) {
    // Stop any active simulation
    if (simulator) simulator.pause();
    simulator = null;

    let results, curves;
    try {
      [results, curves] = await Promise.all([
        fetch(`data/historical/${period}/results.json`).then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        }),
        fetch(`data/historical/${period}/curves.json`).then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        }),
      ]);
    } catch (err) {
      document.querySelector("#period-leaderboard tbody").innerHTML =
        `<tr><td colspan="6" style="color:var(--neg)">Failed to load data for "${period}": ${err.message}</td></tr>`;
      document.getElementById("curves-container").innerHTML = "";
      return;
    }

    document.getElementById("period-context").textContent = PERIOD_CONTEXT[period] ?? "";

    // Assign stable colors per bot (sorted alphabetically)
    const firstWin  = curves.windows[0];
    const botIds    = firstWin ? Object.keys(curves.curves[firstWin]).sort() : [];
    botColorMap     = Object.fromEntries(botIds.map((b, i) => [b, palette[i % palette.length]]));

    // Build chart scaffolding (empty, ready for progressive fill)
    buildChartScaffolding(curves, botColorMap);

    // Create simulator
    simulator = new Simulator(curves, results);

    // Wire up static leaderboard header (sorted by aggregate Sharpe)
    const strategies = [...(results.strategies || [])].sort(
      (a, b) => b.aggregate.sharpe - a.aggregate.sharpe
    );
    const staticRows = strategies.map((s, idx) => ({
      bot_id:   s.bot_id,
      sharpe:   s.aggregate.sharpe,
      dd:       null,   // no series yet
      ret:      null,
      days:     null,
      isLeader: idx === 0,
    }));
    buildLeaderboardRows(staticRows);

    // Reset status bar
    document.getElementById("sim-day-current").textContent = "0";
    document.getElementById("sim-day-total").textContent   = simulator.totalDays;
    document.getElementById("sim-date").textContent        = curves.curves[firstWin]
      ? Object.values(curves.curves[firstWin])[0]?.[0]?.date ?? "—"
      : "—";
    document.getElementById("sim-window-label").textContent = firstWin ?? "";
    document.getElementById("btn-play").disabled  = false;
    document.getElementById("btn-pause").disabled = true;
  }

  // ─── Init ──────────────────────────────────────────────────────────────────
  const sel = document.getElementById("period-select");
  sel.addEventListener("change", () => loadPeriod(sel.value));
  loadPeriod(sel.value);
})();
