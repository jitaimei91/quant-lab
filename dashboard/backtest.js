(async () => {
  const fmt = (v, digits = 2) => v == null ? "-" : v.toFixed(digits);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  let results, curves;
  try {
    [results, curves] = await Promise.all([
      fetch("data/backtest/backtest_results.json").then(r => r.json()),
      fetch("data/backtest/backtest_curves.json").then(r => r.json()),
    ]);
  } catch {
    document.body.innerHTML = "<header><h1>No backtest data yet</h1><p>Run <code>quant-lab backtest</code> to generate.</p></header>";
    return;
  }

  const tbody = document.querySelector("#agg-table tbody");
  for (const s of results.strategies || []) {
    const a = s.aggregate;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.bot_id}</td>
      <td class="${cls(a.sharpe)}">${fmt(a.sharpe)}</td>
      <td>[${fmt(a.sharpe_ci_lo)}, ${fmt(a.sharpe_ci_hi)}]</td>
      <td>${fmt(a.median_alpha_t)}</td>
      <td>${fmt(a.significance_weight)}</td>
      <td>${a.total_test_days}</td>
    `;
    tbody.appendChild(tr);
  }

  const curvesEl = document.getElementById("curves");
  for (const window of curves.windows) {
    const block = document.createElement("div");
    block.innerHTML = `<h3>${window}</h3><canvas height="120"></canvas>`;
    curvesEl.appendChild(block);
    const ctx = block.querySelector("canvas").getContext("2d");
    const palette = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899"];
    let i = 0;
    const dataByBot = curves.curves[window];
    const allDates = new Set();
    for (const series of Object.values(dataByBot)) {
      series.forEach(p => allDates.add(p.date));
    }
    const labels = Array.from(allDates).sort();
    const datasets = [];
    for (const [bot, series] of Object.entries(dataByBot)) {
      const map = Object.fromEntries(series.map(p => [p.date, p.nav]));
      datasets.push({
        label: bot,
        data: labels.map(d => map[d] ?? null),
        spanGaps: true,
        borderColor: palette[i++ % palette.length],
        backgroundColor: "transparent",
        pointRadius: 0,
        borderWidth: 2,
      });
    }
    new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e6edf3" } } },
        scales: {
          x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
          y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        },
      },
    });
  }
})();
