(async () => {
  const params = new URLSearchParams(location.search);
  const botId = params.get("id");
  if (!botId) {
    document.getElementById("bot-title").textContent = "No bot selected — use ?id=<bot_id>";
    return;
  }

  const fmtPct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
  const fmt2 = (v) => v == null ? "—" : v.toFixed(2);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  let data;
  try {
    const res = await fetch(`data/bots/${encodeURIComponent(botId)}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    document.getElementById("bot-title").textContent = `Bot not found: ${botId}`;
    return;
  }

  document.title = `Quant Lab — ${botId}`;
  document.getElementById("bot-title").textContent = botId;

  // Significance badge
  const badge = document.getElementById("sig-badge");
  const sw = data.metrics.significance_weight ?? 0;
  if (sw >= 0.7) {
    badge.textContent = "Significant";
    badge.className = "sig-badge sig-green";
  } else if (sw >= 0.3) {
    badge.textContent = "Marginal";
    badge.className = "sig-badge sig-yellow";
  } else {
    badge.textContent = "Not significant";
    badge.className = "sig-badge sig-gray";
  }

  // Metrics grid
  const m = data.metrics;
  const grid = document.getElementById("metrics-grid");
  const metricItems = [
    ["Total Return", fmtPct(m.total_return), cls(m.total_return)],
    ["Annualized Return", fmtPct(m.annualized_return), cls(m.annualized_return)],
    ["Sharpe", fmt2(m.sharpe), m.sharpe >= 1 ? "pos" : m.sharpe < 0 ? "neg" : ""],
    ["Sharpe 95% CI", `[${fmt2(m.sharpe_ci_lo)}, ${fmt2(m.sharpe_ci_hi)}]`, ""],
    ["Volatility", fmtPct(m.volatility), "neg"],
    ["Max Drawdown", fmtPct(m.max_drawdown), "neg"],
    ["Days", m.days, ""],
    ["α t-stat vs SPY", fmt2(m.alpha_t_stat_vs_spy), cls(m.alpha_t_stat_vs_spy)],
    ["α t-stat vs QQQ", fmt2(m.alpha_t_stat_vs_qqq), cls(m.alpha_t_stat_vs_qqq)],
    ["Significance Weight", fmt2(sw), sw >= 0.7 ? "pos" : ""],
  ];
  for (const [label, value, clsName] of metricItems) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<div class="metric-label">${label}</div><div class="metric-value ${clsName}">${value}</div>`;
    grid.appendChild(card);
  }

  // Equity curve
  const nav = data.nav_series || [];
  if (nav.length > 1) {
    const labels = nav.map(p => p.date);
    const values = nav.map(p => p.nav);
    const ctx = document.getElementById("equity-chart").getContext("2d");
    new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: botId,
          data: values,
          borderColor: "#58a6ff",
          backgroundColor: "rgba(88,166,255,0.07)",
          pointRadius: 0,
          borderWidth: 2,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e6edf3" } } },
        scales: {
          x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
          y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        },
      },
    });
  } else {
    document.getElementById("equity-chart").insertAdjacentHTML("afterend", "<p class='muted'>Not enough data for equity curve.</p>");
  }

  // Factor loadings
  const fl = m.factor_loadings;
  const factorEl = document.getElementById("factor-loadings");
  if (fl && Object.keys(fl).length > 0) {
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Factor</th><th>Loading</th></tr></thead><tbody>" +
      Object.entries(fl).map(([f, v]) =>
        `<tr><td>${f}</td><td class="${cls(v)}">${fmt2(v)}</td></tr>`
      ).join("") + "</tbody>";
    factorEl.appendChild(table);
  } else {
    factorEl.textContent = "Factor loadings not available.";
  }

  // Current weights
  const weightsEl = document.getElementById("weights");
  const weights = data.current_weights || {};
  const nonZero = Object.entries(weights).filter(([, w]) => w > 0.001);
  if (nonZero.length > 0) {
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Symbol</th><th>Weight</th></tr></thead><tbody>" +
      nonZero.map(([s, w]) =>
        `<tr><td>${s}</td><td>${(w * 100).toFixed(1)}%</td></tr>`
      ).join("") + "</tbody>";
    weightsEl.appendChild(table);
  } else {
    weightsEl.textContent = "Cash (no open positions)";
  }

  // Recent trades — if trades.jsonl doesn't exist, show placeholder
  const tradesEl = document.getElementById("recent-trades");
  try {
    const res = await fetch("data/trades.jsonl");
    if (!res.ok) throw new Error("no trades file");
    const text = await res.text();
    const lines = text.trim().split("\n").filter(Boolean);
    const botTrades = lines
      .map(l => { try { return JSON.parse(l); } catch { return null; } })
      .filter(t => t && t.bot_id === botId)
      .slice(-20)
      .reverse();
    if (botTrades.length === 0) {
      tradesEl.textContent = "No trades yet.";
    } else {
      const table = document.createElement("table");
      table.innerHTML = "<thead><tr><th>Date</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead><tbody>" +
        botTrades.map(t =>
          `<tr><td>${t.date || ""}</td><td>${t.symbol || ""}</td><td>${t.side || ""}</td><td>${t.qty ?? ""}</td><td>${t.price != null ? t.price.toFixed(2) : ""}</td></tr>`
        ).join("") + "</tbody>";
      tradesEl.appendChild(table);
    }
  } catch {
    tradesEl.textContent = "No trades yet.";
  }
})();
