"""Tests for the daily journal + summary writer."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from quant_lab.journal import (
    journal_entry,
    append_journal,
    write_journal_summary,
    _classify_day,
    _annualised_sharpe,
)


# ---------------------------------------------------------------------------
# _classify_day
# ---------------------------------------------------------------------------


def test_classify_good_day_when_positive_and_beats_spy():
    assert _classify_day(daily_return=0.01, spy_return=0.005) == "good_day"


def test_classify_bad_day_when_loses_more_than_50bps_to_spy():
    assert _classify_day(daily_return=-0.012, spy_return=-0.003) == "bad_day"


def test_classify_neutral_when_positive_but_lags_spy():
    assert _classify_day(daily_return=0.002, spy_return=0.005) == "neutral"


def test_classify_neutral_when_lost_to_spy_within_50bps():
    assert _classify_day(daily_return=-0.001, spy_return=0.001) == "neutral"


# ---------------------------------------------------------------------------
# _annualised_sharpe
# ---------------------------------------------------------------------------


def test_annualised_sharpe_returns_none_for_insufficient_data():
    assert _annualised_sharpe([]) is None
    assert _annualised_sharpe([0.01]) is None


def test_annualised_sharpe_zero_when_constant_returns():
    """std=0 → undefined Sharpe → return None."""
    assert _annualised_sharpe([0.01, 0.01, 0.01]) is None


def test_annualised_sharpe_positive_for_profitable_low_vol_series():
    rets = [0.001] * 30 + [0.0015] * 30
    s = _annualised_sharpe(rets)
    assert s is not None
    assert s > 1.0


# ---------------------------------------------------------------------------
# journal_entry — pure builder
# ---------------------------------------------------------------------------


def _nav_series(start: date, n: int, daily_return: float = 0.001, vol: float = 0.005, seed: int = 1) -> list[tuple[date, float]]:
    """Synthetic NAV with both drift and noise, so std > 0 and Sharpe is defined."""
    import random
    rng = random.Random(seed)
    out = []
    nav = 100_000.0
    for i in range(n):
        nav *= 1 + daily_return + rng.gauss(0.0, vol)
        out.append((start + timedelta(days=i), nav))
    return out


def test_journal_entry_records_all_fields():
    today = date(2026, 5, 9)
    # 100 days of returns — enough for 30/60/90d rolling Sharpes.
    # Strong drift, low vol → positive daily_return on the last day, beats SPY.
    navs = _nav_series(today - timedelta(days=99), 100, daily_return=0.003, vol=0.001, seed=42)
    # SPY returns negative each day so the bot reliably beats SPY → good_day
    spy_rets = [-0.001] * 99
    entry = journal_entry(
        bot_id="apex",
        today=today,
        weights={"SSO": 0.4, "TMF": 0.3, "UGL": 0.2},
        nav_series=navs,
        spy_returns=spy_rets,
    )
    assert entry["bot_id"] == "apex"
    assert entry["date"] == "2026-05-09"
    assert entry["weights"] == {"SSO": 0.4, "TMF": 0.3, "UGL": 0.2}
    assert entry["beats_spy"] is True
    # All rolling Sharpes should be defined and positive (we have 99 returns)
    assert entry["rolling_sharpe_30d"] is not None and entry["rolling_sharpe_30d"] > 0
    assert entry["rolling_sharpe_60d"] is not None and entry["rolling_sharpe_60d"] > 0
    assert entry["rolling_sharpe_90d"] is not None and entry["rolling_sharpe_90d"] > 0
    assert entry["tag"] == "good_day"


def test_journal_entry_with_short_history():
    """Fewer than 30 returns → all rolling Sharpes None (we'd rather report
    None than a misleading '90d Sharpe' computed from 4 days)."""
    today = date(2026, 5, 9)
    navs = _nav_series(today - timedelta(days=4), 5)
    entry = journal_entry(
        bot_id="new-bot", today=today, weights={"SPY": 0.5},
        nav_series=navs, spy_returns=[0.0],
    )
    assert entry["bot_id"] == "new-bot"
    assert entry["rolling_sharpe_30d"] is None
    assert entry["rolling_sharpe_60d"] is None
    assert entry["rolling_sharpe_90d"] is None
    # daily_return should still be reported (we have 2+ NAVs)
    assert isinstance(entry["daily_return"], float)


# ---------------------------------------------------------------------------
# append_journal — idempotency
# ---------------------------------------------------------------------------


def test_append_journal_creates_file(tmp_path):
    path = tmp_path / "bot_journal.jsonl"
    entries = [
        {"date": "2026-05-09", "bot_id": "apex", "nav": 100_000, "tag": "neutral"},
        {"date": "2026-05-09", "bot_id": "calendar", "nav": 100_000, "tag": "neutral"},
    ]
    append_journal(path, entries)
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["bot_id"] == "apex"


def test_append_journal_skips_duplicates(tmp_path):
    path = tmp_path / "bot_journal.jsonl"
    entries = [
        {"date": "2026-05-09", "bot_id": "apex", "nav": 100_000, "tag": "neutral"},
    ]
    append_journal(path, entries)
    append_journal(path, entries)  # second call should skip
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1


def test_append_journal_appends_new_rows(tmp_path):
    path = tmp_path / "bot_journal.jsonl"
    append_journal(path, [{"date": "2026-05-09", "bot_id": "apex", "nav": 100, "tag": "neutral"}])
    append_journal(path, [{"date": "2026-05-10", "bot_id": "apex", "nav": 101, "tag": "good_day"}])
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[1]["date"] == "2026-05-10"


# ---------------------------------------------------------------------------
# write_journal_summary
# ---------------------------------------------------------------------------


def test_summary_aggregates_per_bot_counts(tmp_path):
    journal_path = tmp_path / "bot_journal.jsonl"
    summary_path = tmp_path / "journal_summary.json"

    rows = []
    for i in range(10):
        d = (date(2026, 5, 1) + timedelta(days=i)).isoformat()
        tag = "good_day" if i % 2 == 0 else "bad_day"
        rows.append({
            "date": d, "bot_id": "apex", "nav": 100_000 + i * 100,
            "tag": tag, "vs_spy": 0.001 if i % 2 == 0 else -0.002,
            "rolling_sharpe_30d": 0.5, "rolling_sharpe_60d": None,
            "rolling_sharpe_90d": None,
        })
    journal_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    write_journal_summary(journal_path, summary_path, lookback_days=10)
    summary = json.loads(summary_path.read_text())
    apex = summary["bots"]["apex"]
    assert apex["good_days"] == 5
    assert apex["bad_days"] == 5
    assert apex["win_rate"] == 0.5
    assert apex["latest_nav"] == 100_900
    assert apex["latest_date"] == "2026-05-10"


def test_summary_handles_empty_journal(tmp_path):
    journal_path = tmp_path / "missing.jsonl"
    summary_path = tmp_path / "summary.json"
    write_journal_summary(journal_path, summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary == {"bots": {}}
