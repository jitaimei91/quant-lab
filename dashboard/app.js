(async () => {
  const fmtPct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
  const cls = (v) => (v >= 0 ? "pos" : "neg");

  const [leaderboardRes, navRes] = await Promise.all([
    fetch("data/leaderboard.json"),
    fetch("data/nav_history.json"),
  ]);
  const leaderboard = await leaderboardRes.json();
  const navHistory = await navRes.json();

  // Market
  const marketEl = document.getElementById("market-stats");
  for (const sym of ["SPY", "QQQ"]) {
    const m = leaderboard.market[sym] || { change_pct: 0, ytd_pct: 0 };
    const card = document.createElement("div");
    card.className = "market-card";
    card.innerHTML = `
      <div class="sym">${sym}</div>
      <div class="chg ${m.change_pct >= 0 ? "pos" : "neg"}">${(m.change_pct >= 0 ? "+" : "") + m.change_pct.toFixed(2)}%</div>
      <div>YTD ${(m.ytd_pct >= 0 ? "+" : "") + m.ytd_pct.toFixed(2)}%</div>
    `;
    marketEl.appendChild(card);
  }

  // Leaderboard
  const tbody = document.querySelector("#leaderboard-table tbody");
  for (const bot of leaderboard.bots) {
    const row = document.createElement("tr");
    const weights = Object.entries(bot.current_weights || {})
      .filter(([, w]) => w > 0.01)
      .map(([s, w]) => `${s} ${(w * 100).toFixed(0)}%`)
      .join(", ") || "cash";
    row.innerHTML = `
      <td>${bot.bot_id}</td>
      <td class="${cls(bot.metrics.total_return)}">${fmtPct(bot.metrics.total_return)}</td>
      <td class="${cls(bot.metrics.annualized_return)}">${fmtPct(bot.metrics.annualized_return)}</td>
      <td>${bot.metrics.sharpe.toFixed(2)}</td>
      <td class="neg">${fmtPct(bot.metrics.max_drawdown)}</td>
      <td>${bot.metrics.days}</td>
      <td>${weights}</td>
    `;
    tbody.appendChild(row);
  }

  document.getElementById("generated-at").textContent = leaderboard.generated_at;

  // Equity chart
  const ctx = document.getElementById("equity-chart").getContext("2d");
  const datasets = [];
  const palette = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ec4899"];
  let i = 0;
  let allDates = new Set();
  for (const [bot, series] of Object.entries(navHistory)) {
    series.forEach(p => allDates.add(p.date));
  }
  const labels = Array.from(allDates).sort();
  for (const [bot, series] of Object.entries(navHistory)) {
    const map = Object.fromEntries(series.map(p => [p.date, p.nav]));
    datasets.push({
      label: bot,
      data: labels.map(d => map[d] ?? null),
      spanGaps: true,
      borderColor: palette[i % palette.length],
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2,
    });
    i++;
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
})();
