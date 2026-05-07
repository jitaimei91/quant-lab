from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bot import export_site, run_evolution_only, run_morning, run_simple_backtest
from .config import (
    DEFAULT_STARTING_CASH,
    ensure_runtime_dirs,
    load_env_file,
    parse_positions,
    save_account,
)
from .models import Account
from .paths import (
    CACHE_DIR,
    CONFIG_DIR,
    DEFAULT_ACCOUNT_PATH,
    DEFAULT_UNIVERSE_PATH,
    PAPER_ACCOUNT_PATH,
    REPORTS_DIR,
    STATE_DIR,
    TRADE_LOG_PATH,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="morning-quant-bot",
        description="Free local paper-trading quant assistant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-account", help="Create config/account.json")
    init_parser.add_argument("--cash", type=float, default=DEFAULT_STARTING_CASH)
    init_parser.add_argument(
        "--positions",
        default="",
        help='Comma list like "SPY:2:500,QQQ:1:430"',
    )

    morning_parser = sub.add_parser("morning", help="Generate morning action report")
    morning_parser.add_argument("--account", type=Path, default=DEFAULT_ACCOUNT_PATH)
    morning_parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    morning_parser.add_argument("--years", type=int, default=8)
    morning_parser.add_argument("--generations", type=int, default=4)
    morning_parser.add_argument("--population", type=int, default=32)
    morning_parser.add_argument("--paper-fill", action="store_true")
    morning_parser.add_argument("--bot-id", default="morning-quant-v1")
    morning_parser.add_argument(
        "--redact-public",
        action="store_true",
        help="Hide cash, equity, and share counts in public JSON.",
    )

    evolve_parser = sub.add_parser("evolve", help="Run strategy evolution only")
    evolve_parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    evolve_parser.add_argument("--years", type=int, default=8)
    evolve_parser.add_argument("--generations", type=int, default=6)
    evolve_parser.add_argument("--population", type=int, default=48)

    backtest_parser = sub.add_parser("backtest", help="Backtest on a symbol list")
    backtest_parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    backtest_parser.add_argument("--years", type=int, default=8)
    backtest_parser.add_argument("--generations", type=int, default=3)
    backtest_parser.add_argument("--population", type=int, default=24)

    sub.add_parser("schedule", help="Print a weekday cron command")
    sub.add_parser("export-site", help="Rebuild public dashboard files")

    args = parser.parse_args(argv)
    ensure_runtime_dirs([CONFIG_DIR, CACHE_DIR, STATE_DIR, REPORTS_DIR])
    load_env_file(CONFIG_DIR / ".env")

    if args.command == "init-account":
        account = Account.from_dict(
            {
                "cash": args.cash,
                "positions": parse_positions(args.positions),
                "notes": "Simulation account only.",
            }
        )
        save_account(DEFAULT_ACCOUNT_PATH, account)
        save_account(PAPER_ACCOUNT_PATH, account)
        print(f"Wrote {DEFAULT_ACCOUNT_PATH}")
        print(f"Wrote {PAPER_ACCOUNT_PATH}")
        return 0

    if args.command == "morning":
        report_path = run_morning(
            account_path=args.account,
            universe_path=args.universe,
            years=args.years,
            generations=args.generations,
            population=args.population,
            paper_fill=args.paper_fill,
            bot_id=args.bot_id,
            redact_public=args.redact_public,
        )
        print(f"Wrote {report_path}")
        if args.paper_fill:
            print(f"Updated {PAPER_ACCOUNT_PATH}")
            print(f"Appended fills to {TRADE_LOG_PATH}")
        return 0

    if args.command == "evolve":
        count = run_evolution_only(
            universe_path=args.universe,
            years=args.years,
            generations=args.generations,
            population=args.population,
        )
        print(f"Evaluated strategy population. Leaderboard size: {count}")
        return 0

    if args.command == "backtest":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        path, sharpe = run_simple_backtest(
            symbols=symbols,
            years=args.years,
            generations=args.generations,
            population=args.population,
        )
        print(f"Wrote {path}")
        print(f"Best strategy Sharpe: {sharpe:.2f}")
        return 0

    if args.command == "schedule":
        root = Path(__file__).resolve().parents[2]
        command = f"cd {json.dumps(str(root))} && /usr/bin/python3 run_bot.py morning --paper-fill"
        print("# Weekdays at 8:00 AM local time:")
        print(f"0 8 * * 1-5 {command}")
        print("")
        print("Install with: crontab -e")
        return 0

    if args.command == "export-site":
        path = export_site()
        print(f"Wrote {path}")
        return 0

    return 2
