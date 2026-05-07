(async () => {
  const DEFAULT_A = "codex-r1000";
  const DEFAULT_B = "meta-ensemble";

  const fmtPct = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%");
  const fmt2 = (v) => v == null ? "—" : v.toFixed(2);
  const cls = (v) => v >= 0 ? "pos" : "neg";

  // Load leaderboard for bot list
  let leaderboard;
  try {
    leaderboard = await fetch("data/leaderboard.json").then(r => r.json());
  } catch {
    document.body.innerHTML += "<p style='color:var(--bad);padding:1.5rem'>Could not load leaderboard.json</p>";
    return;
  }

  const bots = (leaderboard.bots || []).map(b => b.bot_id);

  // Populate dropdowns
  const selA = document.getElementById("select-a");
  const selB = document.getElementById("select-b");
  for (const id of bots) {
    selA.appendChild(new Option(id, id));
    selB.appendChild(new Option(id, id));
  }
  // Pre-select defaults (headline competition)
  const findOrFirst = (sel, target) => {
    const opt = Array.from(sel.options).find(o => o.value === target);
    if (opt) sel.value = target;
  };
  findOrFirst(selA, DEFAULT_A);
  findOrFirst(selB, DEFAULT_B);

  let overlayChart = null;
  let diffChart = null;

  async function loadBotData(id) {
    const res = await fetch(`data/bots/${encodeURIComponent(id)}.json`);
    if (!res.ok) throw new Error(`Bot data not found: ${id}`);
    return res.json();
  }

  function buildNavMap(data) {
    const map = {};
    for (const p of data.nav_series || []) map[p.date] = p.nav;
    return map;
  }

  function dailyRets(navMap, labels) {
    const rets = [];
    const keys = labels;
    for (let i = 1; i < keys.length; i++) {
      const prev = navMap[keys[i - 1]];
      const curr = navMap[keys[i]];
      rets.push(prev && curr ? curr / prev - 1.0 : null);
    }
    return rets;
  }

  async function runCompare() {
    const idA = selA.value;
    const idB = selB.value;

    let dataA, dataB;
    try {
      [dataA, dataB] = await Promise.all([loadBotData(idA), loadBotData(idB)]);
    } catch (e) {
      alert(`Could not load bot data: ${e.message}\nMake sure write_per_bot_files() has been run.`);
      return;
    }

    const mapA = buildNavMap(dataA);
    const mapB = buildNavMap(dataB);
    const allDates = Array.from(new Set([...Object.keys(mapA), ...Object.keys(mapB)])).sort();

    // Overlay chart
    if (overlayChart) overlayChart.destroy();
    const ctxOverlay = document.getElementById("overlay-chart").getContext("2d");
    overlayChart = new Chart(ctxOverlay, {
      type: "line",
      data: {
        labels: allDates,
        datasets: [
          {
            label: idA,
            data: allDates.map(d => mapA[d] ?? null),
            spanGaps: true,
            borderColor: "#58a6ff",
            backgroundColor: "rgba(88,166,255,0.07)",
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: idB,
            data: allDates.map(d => mapB[d] ?? null),
            spanGaps: true,
            borderColor: "#3fb950",
            backgroundColor: "rgba(63,185,80,0.07)",
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
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

    // Side-by-side stats
    const statsEl = document.getElementById("stats-compare");
    statsEl.innerHTML = "";
    const renderStats = (data, id, color) => {
      const m = data.metrics;
      const div = document.createElement("div");
      div.innerHTML = `
        <h3 style="color:${color}">${id}</h3>
        <table>
          <tbody>
            <tr><td>Total Return</td><td class="${cls(m.total_return)}">${fmtPct(m.total_return)}</td></tr>
            <tr><td>Annualized</td><td class="${cls(m.annualized_return)}">${fmtPct(m.annualized_return)}</td></tr>
            <tr><td>Sharpe</td><td>${fmt2(m.sharpe)}</td></tr>
            <tr><td>Sharpe CI</td><td>[${fmt2(m.sharpe_ci_lo)}, ${fmt2(m.sharpe_ci_hi)}]</td></tr>
            <tr><td>Volatility</td><td>${fmtPct(m.volatility)}</td></tr>
            <tr><td>Max Drawdown</td><td class="neg">${fmtPct(m.max_drawdown)}</td></tr>
            <tr><td>Days</td><td>${m.days}</td></tr>
            <tr><td>α t-stat vs SPY</td><td class="${cls(m.alpha_t_stat_vs_spy)}">${fmt2(m.alpha_t_stat_vs_spy)}</td></tr>
            <tr><td>Sig Weight</td><td>${fmt2(m.significance_weight)}</td></tr>
          </tbody>
        </table>
      `;
      statsEl.appendChild(div);
    };
    renderStats(dataA, idA, "#58a6ff");
    renderStats(dataB, idB, "#3fb950");

    // Daily differential chart (A - B)
    const retsA = dailyRets(mapA, allDates);
    const retsB = dailyRets(mapB, allDates);
    const diff = retsA.map((a, i) =>
      a != null && retsB[i] != null ? a - retsB[i] : null
    );
    const diffLabels = allDates.slice(1);

    if (diffChart) diffChart.destroy();
    const ctxDiff = document.getElementById("diff-chart").getContext("2d");
    diffChart = new Chart(ctxDiff, {
      type: "bar",
      data: {
        labels: diffLabels,
        datasets: [{
          label: `${idA} − ${idB} daily return`,
          data: diff,
          backgroundColor: diff.map(v =>
            v == null ? "transparent" : v >= 0 ? "rgba(63,185,80,0.6)" : "rgba(248,81,73,0.6)"
          ),
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e6edf3" } } },
        scales: {
          x: { ticks: { color: "#8b949e", maxTicksLimit: 12 }, grid: { color: "#30363d" } },
          y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
        },
      },
    });
  }

  document.getElementById("run-compare").addEventListener("click", runCompare);

  // Auto-run on load
  await runCompare();
})();
