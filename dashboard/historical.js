(async () => {
  const fmt = (v, digits = 2) => v == null ? "-" : v.toFixed(digits);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  const PERIOD_CONTEXT = {
    "2008-crisis": "S&P 500 lost ~50% peak-to-trough between Oct 2007 and Mar 2009. Watch which bots stayed defensive. 1 walk-forward test window (2009).",
    "2020-covid": "VIX spiked above 80 in March 2020. The V-shaped recovery followed within months. Watch how the regime kill-switch behaved. 2 walk-forward test windows (2020, 2021).",
    "2022-rates": "Bonds and stocks fell together in 2022 — uncommon. The Fed hiked rates from 0.25% to 5.5%. Tested the diversification thesis and hit trend-following bots hard. 3 walk-forward test windows (2023).",
  };

  const palette = [
    "#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899",
    "#79c0ff", "#56d364", "#ffa657", "#d2a8ff", "#f78166",
    "#1f6feb", "#238636", "#9e6a03", "#6e40c9", "#bf4b8a",
    "#388bfd",
  ];

  function maxDrawdown(navSeries) {
    // navSeries: [{date, nav}, ...]
    let peak = -Infinity;
    let maxDD = 0;
    for (const p of navSeries) {
      if (p.nav > peak) peak = p.nav;
      const dd = (peak - p.nav) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
  }

  let currentChart = null;

  async function loadPeriod(period) {
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
      document.getElementById("period-leaderboard").querySelector("tbody").innerHTML =
        `<tr><td colspan="6" style="color:var(--neg)">Failed to load data for "${period}": ${err.message}</td></tr>`;
      document.getElementById("curves-container").innerHTML = "";
      return;
    }

    // Period context
    document.getElementById("period-context").textContent =
      PERIOD_CONTEXT[period] ?? "";

    // Compute max drawdown per bot across all windows
    const ddByBot = {};
    for (const [win, botMap] of Object.entries(curves.curves)) {
      for (const [bot, series] of Object.entries(botMap)) {
        const dd = maxDrawdown(series);
        if (ddByBot[bot] == null || dd > ddByBot[bot]) ddByBot[bot] = dd;
      }
    }

    // Build leaderboard: sort by aggregate Sharpe descending
    const strategies = [...(results.strategies || [])].sort(
      (a, b) => b.aggregate.sharpe - a.aggregate.sharpe
    );

    const tbody = document.querySelector("#period-leaderboard tbody");
    tbody.innerHTML = "";
    strategies.forEach((s, idx) => {
      const a = s.aggregate;
      const dd = ddByBot[s.bot_id];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${s.bot_id}</td>
        <td class="${cls(a.sharpe)}">${fmt(a.sharpe)}</td>
        <td>[${fmt(a.sharpe_ci_lo)}, ${fmt(a.sharpe_ci_hi)}]</td>
        <td class="neg">${dd != null ? (dd * 100).toFixed(1) + "%" : "-"}</td>
        <td>${a.total_test_days}</td>
      `;
      tbody.appendChild(tr);
    });

    // Equity curves — one chart per window
    const container = document.getElementById("curves-container");
    container.innerHTML = "";
    if (currentChart) {
      currentChart.destroy();
      currentChart = null;
    }

    for (const win of curves.windows) {
      const botMap = curves.curves[win];
      if (!botMap) continue;

      const block = document.createElement("div");
      block.innerHTML = `<h3 style="color:var(--muted);font-size:.9rem;margin-bottom:.4rem;">Window: ${win}</h3><canvas height="120"></canvas>`;
      container.appendChild(block);

      const allDates = new Set();
      for (const series of Object.values(botMap)) {
        series.forEach(p => allDates.add(p.date));
      }
      const labels = Array.from(allDates).sort();

      // Sort bots by name for consistent color assignment
      const sortedBots = Object.keys(botMap).sort();
      let colorIdx = 0;
      const datasets = sortedBots.map(bot => {
        const map = Object.fromEntries(botMap[bot].map(p => [p.date, p.nav]));
        return {
          label: bot,
          data: labels.map(d => map[d] ?? null),
          spanGaps: true,
          borderColor: palette[colorIdx++ % palette.length],
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 1.5,
        };
      });

      const ctx = block.querySelector("canvas").getContext("2d");
      const chart = new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true,
          animation: false,
          plugins: {
            legend: { labels: { color: "#e6edf3", font: { size: 11 } } },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.y?.toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
              },
            },
          },
          scales: {
            x: { ticks: { color: "#8b949e", maxTicksLimit: 12 }, grid: { color: "#30363d" } },
            y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
          },
        },
      });
      // Track last chart for cleanup on next switch (only one window usually)
      currentChart = chart;
    }
  }

  const sel = document.getElementById("period-select");
  sel.addEventListener("change", () => loadPeriod(sel.value));
  loadPeriod(sel.value);
})();
