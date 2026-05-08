# Quant Lab Overnight Mega-Plan — Phase 3 + Phase 7 + Phase 12 + Full Calibration

> **For agentic workers:** Use superpowers:subagent-driven-development. Compact plan: function signatures + behaviors + acceptance tests. Built for overnight unattended execution.

**Goal:** Ship the three most impactful structural additions plus a fresh real-data calibration before the user wakes:

1. **Phase 12** — Multi-asset universe (bonds, commodities, international, real estate)
2. **Phase 7** — Hidden Markov regime detector + per-regime strategy gating
3. **Phase 3** — XGBoost + LightGBM + Ensemble ML bots with strict validation gates
4. **Full recalibration** — Real 10-year walk-forward backtest including all new strategies
5. **Push live** — Force-push to public clone, verify GH Pages updates

**Why this combination:** Phase 12 expands the addressable opportunity surface (trend works on bonds even when stocks chop). Phase 7 routes capital to whichever strategies fit the current macro regime. Phase 3 adds learned signals beyond hand-coded rules. Together they substantially increase the meta-ensemble's information content.

**Architecture:** All new strategies plug into existing `Strategy` ABC. HMM regime is a new module under `engine/`. ML pipeline reuses the existing backtest harness's `fit_callback` hook. Calibration uses existing `quant-lab backtest` CLI with extended universe.

**Honest scoping:** Each phase is independent. Whatever subset finishes overnight ships. The meta-ensemble auto-picks up new strategies' calibrated weights at next morning run.

---

## Phase 12: Multi-Asset Universe (1.5 hours)

### Files
- Modify: `config/universe_r1000.txt` — add ~15 multi-asset tickers
- Create: `config/universe_multiasset.txt` — separate clean list of multi-asset ETFs
- Modify: `src/quant_lab/main.py` `SYMBOLS_FOR_PHASE_1` — extend to include the multi-asset symbols

### Tickers to add
- **Bonds:** TLT (long-term Treasuries), IEF (10y Treasuries), SHY (short-term), HYG (high-yield), LQD (investment-grade)
- **Commodities:** GLD (gold), SLV (silver), USO (oil), DBA (agriculture)
- **International:** EFA (developed), EEM (emerging), VWO (emerging)
- **Real estate:** VNQ (US REITs), VNQI (intl REITs)
- **Volatility proxies:** VXX (VIX short-term), SVXY (inverse VIX) — proxies for vol carry

### Acceptance
- These tickers appear in the universe loader output
- Existing strategies (Momo, MeanRev, Breakout) operate on these symbols when they're in `histories`
- `quant-lab backtest` pulls them via yfinance

### Tasks
1. Append the 15 tickers to `config/universe_r1000.txt`
2. Create `config/universe_multiasset.txt` (just the multi-asset subset)
3. Extend `SYMBOLS_FOR_PHASE_1` in `main.py` (+ the live morning fetch)
4. Run `pytest -q` — confirm no regressions
5. Commit: `feat(universe): add multi-asset ETFs (bonds, commodities, intl, REITs, vol)`

---

## Phase 7: HMM Regime Detector + Per-Regime Gating (2 hours)

### Files
- Create: `src/quant_lab/engine/hmm_regime.py` — Gaussian HMM with EM training
- Modify: `src/quant_lab/engine/regime.py` — extend `regime_state()` to optionally use HMM output
- Create: `src/quant_lab/strategies/regime_aware.py` — wrapper that gates a base strategy on regime
- Create: `tests/test_hmm_regime.py`

### HMM design
- **States:** 4 macro regimes — Risk-On (low VIX, positive momentum), Mean-Reverting Chop (mid VIX), Risk-Off (high VIX, negative momentum), Crisis (extreme VIX, sharp negative returns)
- **Observations:** daily features — VIX level, VIX change, SPY 20-day return, SPY 20-day vol, term spread proxy (TLT/SHY return differential)
- **Training:** EM on rolling 5-year window, retrained weekly via `recalibrate.yml`
- **Inference:** posterior probability of each regime given recent observations
- **Output:** dict with `regime_probs`, `dominant_regime`, `regime_confidence`

### `regime_aware.py` wrapper
```python
@register
class RegimeAware<Base>(Strategy):
    """Wrap a base strategy to fire only in compatible regimes."""
    base_bot_id: str
    allowed_regimes: list[str]  # e.g. ["risk-on", "chop"]

    def target_weights(self, histories, as_of) -> dict[str, float]:
        regime = current_dominant_regime(histories, as_of)
        if regime not in self.allowed_regimes:
            return {}  # all cash
        return get_strategy(self.base_bot_id).target_weights(histories, as_of)
```

### Regime → strategy mapping (for v1)
- **Risk-On:** Momo, Breakout, MA-Cross active; MeanRev, RSI-Rev quiet
- **Chop:** MeanRev, RSI-Rev active; Momo, Breakout quiet
- **Risk-Off:** Defensive — only SPY-Vol at half allocation, increase TLT exposure
- **Crisis:** All strategies → cash; only QQQ-Vol at minimum

### Implementation
- Use scikit-learn's `hmmlearn` if available (free dep, minimal); else implement Gaussian HMM with EM in pure NumPy (~150 LOC, doable)
- HMM model is small (4 states × 5 features = ~25 params). Trains in seconds.
- Persist trained model state to `state/hmm_state.json` (means, covariances, transition matrix)

### Acceptance
- `hmm_regime.py` produces deterministic regime classification on synthetic data
- Test: regime classifier correctly identifies "high-VIX" period as Risk-Off or Crisis
- `RegimeAware<X>` wrapper returns `{}` when regime is incompatible
- New "regime-momo" and "regime-meanrev" bots appear in tournament

### Tasks
1. Implement `hmm_regime.py` with `fit(observations)`, `predict(obs)`, `posteriors(obs)`
2. Add `state/hmm_state.json` persistence
3. Wire into `regime_state()` for an `hmm_regime` field alongside `vix`
4. Implement `regime_aware.py` wrapper class
5. Register 2-3 regime-aware variants of existing strategies
6. Tests + commit per task
7. Final commit: `feat(regime): HMM regime detector + per-regime strategy gating`

---

## Phase 3: ML Strategies — XGBoost + LightGBM + Ensemble (2 hours)

### Files
- Create: `src/quant_lab/ml/__init__.py`
- Create: `src/quant_lab/ml/features.py` — point-in-time-safe feature engineering
- Create: `src/quant_lab/ml/train.py` — walk-forward training driver
- Create: `src/quant_lab/ml/validate.py` — label-shuffle + OOS-stability tests
- Create: `src/quant_lab/strategies/gradboost.py` — XGBoost-based bot
- Create: `src/quant_lab/strategies/lightforest.py` — LightGBM-based bot
- Create: `src/quant_lab/strategies/ml_ensemble.py` — average of GB + LF
- Modify: `pyproject.toml` — add `xgboost`, `lightgbm` dependencies
- Create: `models/.gitkeep`
- Tests across the above

### Feature engineering (`features.py`)
~30 features per (ticker, date) — strict point-in-time:

- **Returns:** 1d, 5d, 20d, 60d, 120d, 252d rolling returns
- **Vol:** 20d, 60d realized vol; vol-of-vol
- **Momentum:** trailing-window z-scores
- **Mean reversion:** distance to 20d/60d/200d SMA in vol units
- **Volume:** rel volume vs 20d avg, on-balance volume change
- **Technical:** RSI(14), MACD, BB-z, ATR
- **Cross-sectional:** sector-relative versions of returns + vol (vs SPY)
- **Liquidity:** ADV in dollars, 20d-avg spread proxy

### Training (`train.py`)
- Walk-forward: 5-year train, 1-year test, monthly step
- Target: 5-day forward return, ranked into deciles
- Loss: rank correlation (Spearman) — not MSE
- XGBoost params: 200 trees, depth=4, learning_rate=0.05
- LightGBM params: similar but with histogram-based splitting

### Validation gates (`validate.py`)
A bot is **excluded from live tournament** unless ALL three pass:
1. **Walk-forward Sharpe** > SPY-Vol's median walk-forward Sharpe (over same windows)
2. **Label-shuffle test:** train on shuffled forward-return labels; resulting Sharpe must be near zero (mean ±0.1 across 10 shuffles). Failure = lookahead leakage somewhere.
3. **OOS stability:** live-fold Sharpe within ±30% of backtest fold Sharpe

If a bot fails, surface on the validation page with red badge + reason. Do NOT silently include in tournament.

### Strategy classes
```python
@register
class GradBoost(Strategy):
    bot_id = "gradboost"
    description = "XGBoost ranking ~30 technical features → top decile = buy"
    def __init__(self): self._model = load_or_train()
    def target_weights(self, histories, as_of): ...

@register
class LightForest(Strategy):
    bot_id = "lightforest"
    # similar
    
@register
class MLEnsemble(Strategy):
    bot_id = "ml-ensemble"
    description = "Average of GradBoost + LightForest signals"
    # blends their weight outputs
```

### Acceptance
- Models train without errors on 5+ years of synthetic data
- Validation gates correctly fail a model trained on shuffled labels
- Live `target_weights` produces sensible weights summing ≤ 0.95
- Models persist to `models/<bot>/<date>.joblib` (or .pkl) — committed to git but small (<5MB each typical)

### Tasks
1. Add `xgboost`, `lightgbm` to `pyproject.toml`
2. Implement `features.py` with point-in-time-safe feature computation
3. Implement `train.py` walk-forward driver
4. Implement `validate.py` with label-shuffle test
5. Implement `GradBoost`, `LightForest`, `MLEnsemble` classes
6. Run validation gates on all three; only register the ones that pass
7. Tests + commits per task
8. Final commit: `feat(ml): XGBoost + LightGBM + Ensemble bots with strict validation gates`

---

## Final Step: Real-Data Calibration (30 min compute)

After all code lands:

```bash
cd "<project-dir>/quant-lab"
source .venv/bin/activate
quant-lab backtest --start 2015-01-01 --end 2026-04-01 --train-years 5 --step-months 12 --no-regime-stress
```

This produces fresh `dashboard/data/backtest/{backtest_results.json, backtest_curves.json, calibration_report.md}` with all bots' walk-forward evidence, including the new ML and regime-aware variants. The meta-ensemble auto-picks up new weights.

Commit the artifacts:
```
data: full recalibration with multi-asset universe + ML + regime bots (2015-2026)
```

---

## Push to Public Clone

After everything lands in `quant-lab/`:

```bash
cd "<project-dir>"
# Sync new commits to public clone (privacy-preserving)
cd quant-lab-public
git remote add backup ../quant-lab 2>/dev/null
git fetch backup main
# Cherry-pick all new commits with attribution rewrite
git rev-list HEAD..backup/main | tac | while read sha; do
  git -c user.email="255495556+jitaimei91@users.noreply.github.com" -c user.name="jitaimei91" cherry-pick "$sha"
done
git -c http.postBuffer=524288000 -c http.version=HTTP/1.1 push origin main
```

Or simpler if cherry-picking is fragile:
```bash
# Re-clone + re-rewrite (clean every time)
cd "<project-dir>"
rm -rf quant-lab-public
git clone quant-lab quant-lab-public
cd quant-lab-public
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter '
  export GIT_AUTHOR_EMAIL="255495556+jitaimei91@users.noreply.github.com"
  export GIT_AUTHOR_NAME="jitaimei91"
  export GIT_COMMITTER_EMAIL="255495556+jitaimei91@users.noreply.github.com"
  export GIT_COMMITTER_NAME="jitaimei91"
' --tag-name-filter cat -- --branches --tags
git remote remove origin 2>/dev/null
git remote add origin https://github.com/jitaimei91/quant-lab.git
git update-ref -d refs/original/refs/heads/main
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git fetch origin
git -c http.postBuffer=524288000 -c http.version=HTTP/1.1 push origin main --force
```

---

## Self-review

Each phase is independent — if one fails, the others still ship. Push happens after each phase completes so the user wakes up to whatever subset succeeded.

Validation gates are non-negotiable: if an ML bot fails the label-shuffle test, it does NOT get registered. Better to ship 0 ML bots than 3 spurious ones.

The meta-ensemble auto-picks up new strategies' calibrated weights at the next morning run — no manual reconciliation needed.

Goodnight.
