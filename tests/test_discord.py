from datetime import date
from unittest.mock import patch

from quant_lab.reporting.discord import build_message, post_to_discord
from quant_lab.tournament.stats import Metrics


def test_build_message_includes_market_snapshot():
    leaderboard = [
        ("spy-vol", Metrics(0.05, 0.10, 0.6, 0.15, -0.05, 100), {"SPY": 1.0}),
        ("qqq-vol", Metrics(0.08, 0.16, 0.7, 0.18, -0.07, 100), {"QQQ": 1.0}),
    ]
    market = {"SPY": {"change_pct": 0.31, "ytd_pct": 5.1},
              "QQQ": {"change_pct": 0.54, "ytd_pct": 8.7}}
    msg = build_message(date(2026, 5, 7), leaderboard, market)
    assert "SPY" in msg
    assert "QQQ" in msg
    assert "+0.31%" in msg or "0.31%" in msg
    assert "spy-vol" in msg
    assert "Not financial advice" in msg


def test_post_to_discord_calls_webhook():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        post_to_discord("https://discord.test/webhook", "hello")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://discord.test/webhook"
        assert kwargs["json"]["content"].startswith("hello")


def test_post_to_discord_truncates_long_messages():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        long = "x" * 5000
        post_to_discord("https://discord.test/webhook", long)
        body = mock_post.call_args.kwargs["json"]["content"]
        assert len(body) <= 2000
