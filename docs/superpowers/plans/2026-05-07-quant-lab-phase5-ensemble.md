# Quant Lab Phase 5 — Meta-Ensemble Synthesis (the "usable model")

> **For agentic workers:** Use superpowers:subagent-driven-development. Compact plan: signatures + behaviors + tests, not byte-for-byte code.

**Goal:** Build a meta-ensemble bot that synthesizes calibrated evidence from every strategy in the tournament into a single weighted signal. Each strategy's weight = `calibrated_sharpe × significance_factor × regime_stability`. The ensemble bot competes in the live tournament as bot #N+1 alongside individual strategies. This is the **"usable model"** deliverable.

**Architecture:**

1. The walk-forward backtest harness already produces per-strategy calibration metrics (`dashboard/data/backtest/backtest_results.json`) — Sharpe + bootstrap CIs + alpha t-stat + significance weight.
2. Phase 5 adds a `MetaEnsemble` strategy class that:
   - Loads the calibration report at strategy-init time
   - Computes a weight per other registered strategy from the calibration data
   - On each morning step, polls each component strategy's `target_weights(...)`, takes a weighted blend, returns the result
3. Online learning: as live tournament data accumulates, an optional update step recomputes the calibrated weights from the live `nav_history.json` (so the ensemble adapts).

**Tech stack:** No new deps. Pure Python.

---

## File Structure

```
src/quant_lab/
├── strategies/
│   └── ensemble.py              # NEW — MetaEnsemble strategy class
├── ensemble/                    # NEW PACKAGE
│   ├── __init__.py
│   ├── weights.py               # weight = sharpe * sig * regime_stability
│   └── live_calibration.py      # online update from live NAV history

tests/
├── test_ensemble_weights.py
├── test_meta_ensemble.py
└── test_live_calibration.py
```

---

## Task 1: Weight computation from calibration report

**Files:**
- Create: `src/quant_lab/ensemble/__init__.py`
- Create: `src/quant_lab/ensemble/weights.py`
- Create: `tests/test_ensemble_weights.py`

**API:**

```python
def compute_strategy_weights(
    calibration_results: dict,  # parsed backtest_results.json["strategies"]
    floor: float = 0.0,
    cap: float = 0.30,
) -> dict[str, float]:
    """For each strategy in the calibration report, compute:
        raw_weight = max(0, sharpe_ci_lo) * significance_weight
    Then normalize so sum = 1.0, clip per-strategy at `cap`, fall back to
    equal-weight across positive-Sharpe strategies if everything is filtered out.
    Returns {bot_id: weight}.
    """

def regime_stability_factor(per_window_results: list[dict]) -> float:
    """Reward strategies that work consistently across windows.
    1.0 if all per-window Sharpes have the same sign as aggregate.
    Lower (0 - 1.0) when sign-flipping is frequent.
    """
```

**Acceptance tests:**

- Strategy with Sharpe=1.0, sig_weight=0.8, all positive windows → high weight
- Strategy with Sharpe=0.0, sig_weight=0.0 → weight=0 (zeroed out)
- All strategies negative → equal weight across positive ones (or all-cash if none)
- Per-strategy weight cap at 0.30 enforced
- Sum of returned weights = 1.0 (or 0.0 if all-cash)

**Steps:**
- [ ] Write 5-6 failing tests
- [ ] Implement `weights.py` with both functions
- [ ] Verify tests pass
- [ ] Commit: `feat(ensemble): calibrated-Sharpe-weighted strategy weights`

---

## Task 2: MetaEnsemble strategy

**Files:**
- Create: `src/quant_lab/strategies/ensemble.py`
- Create: `tests/test_meta_ensemble.py`

**API:**

```python
@register
class MetaEnsemble(Strategy):
    bot_id = "meta-ensemble"
    description = "Calibrated-evidence-weighted blend of all other strategies"

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        weights_override: dict[str, float] | None = None,
    ):
        """Loads calibration weights from `weights_path` (defaults to
        dashboard/data/backtest/backtest_results.json).
        `weights_override` is for testing.
        Caches the loaded weights on the instance.
        """

    def target_weights(self, histories, as_of) -> dict[str, float]:
        """For each component strategy with weight > 0:
          1. Get its target_weights(histories, as_of)
          2. Multiply by ensemble_weight
        Aggregate by ticker (sum the weighted contributions).
        Final cap: per-ticker weight at 0.10, total at 0.95.
        Excludes itself (bot_id="meta-ensemble") to avoid recursion.
        """
```

**Acceptance tests:**
- Override weights `{"momo": 0.5, "spy-vol": 0.5}` with synthetic data → ensemble's weights are a 50/50 blend of Momo's and SPY-Vol's
- If only `spy-vol` has weight > 0 → ensemble's weights ≈ SPY-Vol's
- Self-reference (`meta-ensemble` weight) is ignored
- Per-ticker cap of 0.10 enforced

**Steps:**
- [ ] Write 4-5 failing tests
- [ ] Implement `ensemble.py`
- [ ] Add `from . import ensemble` to `strategies/__init__.py`
- [ ] Run full suite — ~115+ tests passing (109 + ~5 weights + ~5 ensemble)
- [ ] Commit: `feat(strategies): MetaEnsemble bot synthesizing calibrated evidence`

---

## Task 3: Live calibration update

**Files:**
- Create: `src/quant_lab/ensemble/live_calibration.py`
- Create: `tests/test_live_calibration.py`

**API:**

```python
def update_weights_from_live(
    nav_history: dict[str, list[tuple[date, float]]],
    benchmark_returns: dict[str, list[float]],  # SPY returns aligned per bot
    min_days: int = 60,
    weights_path: Path,
) -> dict[str, float]:
    """For each bot with >= min_days of live NAV:
        1. Compute live Sharpe with bootstrap CI
        2. Compute alpha t-stat vs SPY
        3. Compute significance weight
    Combine into refreshed strategy weights via compute_strategy_weights().
    Write to weights_path. Returns the new weights.

    Bots with < min_days of history fall back to their backtest-calibrated weight.
    """
```

**Wire into `main.py morning_command`:**
After saving NAV history, call `update_weights_from_live(...)` to refresh ensemble weights using the *live* paper-trading data accumulated so far. Write new weights to `dashboard/data/backtest/live_weights.json`.

The MetaEnsemble class should prefer `live_weights.json` over `backtest_results.json` when present (live data is more current).

**Acceptance tests:**
- With < min_days of data → returns backtest-calibrated weights
- With >= min_days → returns live-Sharpe-weighted weights
- File is written at the expected path

**Steps:**
- [ ] Write 3-4 failing tests
- [ ] Implement `live_calibration.py`
- [ ] Wire into `morning_command` (after persist_state, before write_dashboard_data)
- [ ] Update MetaEnsemble to prefer live_weights.json when present
- [ ] Run full suite
- [ ] Commit: `feat(ensemble): online weight updates from live tournament evidence`

---

## Task 4: E2E integration test for ensemble

**Files:**
- Create: `tests/test_phase5_e2e.py`

**Test:**
- Set up synthetic data (SPY, QQQ, IWM, VTV, VUG, AAPL, NVDA, ^VIX)
- Run `morning_command` for 70 simulated days (so live calibration has ≥ 60 days of data)
- Assert `meta-ensemble` is in the leaderboard
- Assert `live_weights.json` exists after the run with non-trivial weights
- Verify the ensemble's NAV is sensible (not zero, not infinite)

**Steps:**
- [ ] Implement test with a 70-day synthetic loop
- [ ] Run → passes
- [ ] Commit: `test: Phase 5 e2e integration test for meta-ensemble`

---

## Self-review

**Spec coverage:**
- §5 spec deliverable "synthesis": weight = `calibrated_sharpe × significance_factor × regime_stability` — Task 1 ✓
- Live evidence updates — Task 3 ✓
- Ensemble competes in tournament alongside other bots — Task 2 ✓
- The "usable model" — emerges from Tasks 1-2-3 ✓

After Phase 5: live tournament includes `meta-ensemble` as bot #12, weights initialized from backtest calibration, refreshed online from live NAV. Ready for Phase 4 (dashboard Bot-vs-Bot view).
