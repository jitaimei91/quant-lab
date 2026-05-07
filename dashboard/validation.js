(async () => {
  const fmt2 = (v) => v == null ? "—" : v.toFixed(2);

  let data;
  try {
    const res = await fetch("data/validation.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch {
    // Fall back to backtest_results.json directly for compatibility
    try {
      const res2 = await fetch("data/backtest/backtest_results.json");
      if (!res2.ok) throw new Error("no backtest data");
      const raw = await res2.json();
      // Build validation.json-compatible shape client-side
      data = {
        strategies: (raw.strategies || []).map(s => {
          const agg = s.aggregate || {};
          const sw = agg.significance_weight ?? 0;
          return {
            bot_id: s.bot_id,
            aggregate: agg,
            per_window: s.per_window || [],
            significance_badge: sw >= 0.7 ? "green" : sw >= 0.3 ? "yellow" : "gray",
            failed_validation: sw < 0.3,
          };
        }),
      };
    } catch {
      document.getElementById("val-grid").textContent = "No validation data found. Run the backtest command first.";
      return;
    }
  }

  const grid = document.getElementById("val-grid");

  for (const s of data.strategies || []) {
    const agg = s.aggregate || {};
    const badge = s.significance_badge || "gray";
    const failed = s.failed_validation;

    const card = document.createElement("div");
    card.className = "val-card" + (failed ? " val-failed" : "");

    const badgeHtml = {
      green: `<span class="sig-badge sig-green">Significant</span>`,
      yellow: `<span class="sig-badge sig-yellow">Marginal</span>`,
      gray: `<span class="sig-badge sig-gray">Not significant</span>`,
    }[badge] || "";

    card.innerHTML = `
      <h3>
        ${s.bot_id}
        ${badgeHtml}
        ${failed ? `<span class="sig-badge sig-gray" style="background:rgba(248,81,73,0.1);color:var(--bad);border-color:var(--bad)">Failed validation</span>` : ""}
      </h3>
      <div class="val-stats">
        <span>Sharpe: <strong class="${agg.sharpe >= 0 ? 'pos' : 'neg'}">${fmt2(agg.sharpe)}</strong></span>
        <span>95% CI: <strong>[${fmt2(agg.sharpe_ci_lo)}, ${fmt2(agg.sharpe_ci_hi)}]</strong></span>
        <span>Median α t-stat: <strong class="${(agg.median_alpha_t ?? 0) >= 0 ? 'pos' : 'neg'}">${fmt2(agg.median_alpha_t)}</strong></span>
        <span>Sig weight: <strong>${fmt2(agg.significance_weight)}</strong></span>
        <span>Windows: <strong>${agg.windows_evaluated ?? "—"}</strong></span>
        <span>Test days: <strong>${agg.total_test_days ?? "—"}</strong></span>
      </div>
      <canvas class="val-mini" height="60"></canvas>
    `;

    grid.appendChild(card);

    // Per-window mini Sharpe bar chart
    const windows = s.per_window || [];
    if (windows.length > 0) {
      const canvas = card.querySelector(".val-mini");
      const ctx = canvas.getContext("2d");
      new Chart(ctx, {
        type: "bar",
        data: {
          labels: windows.map(w => w.window || ""),
          datasets: [{
            label: "Window Sharpe",
            data: windows.map(w => w.sharpe ?? null),
            backgroundColor: windows.map(w =>
              (w.sharpe ?? 0) >= 1.0 ? "rgba(63,185,80,0.7)" :
              (w.sharpe ?? 0) >= 0.0 ? "rgba(88,166,255,0.7)" :
              "rgba(248,81,73,0.7)"
            ),
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false },
          },
          scales: {
            x: { ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: "#30363d" } },
            y: { ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: "#30363d" } },
          },
        },
      });
    }
  }
})();
