from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STATE_DIR = PROJECT_ROOT / "state"
REPORTS_DIR = PROJECT_ROOT / "reports"
PUBLIC_DIR = PROJECT_ROOT / "public"

DEFAULT_ACCOUNT_PATH = CONFIG_DIR / "account.json"
DEFAULT_UNIVERSE_PATH = CONFIG_DIR / "universe.txt"
LEADERBOARD_PATH = STATE_DIR / "leaderboard.json"
PAPER_ACCOUNT_PATH = STATE_DIR / "paper_account.json"
TRADE_LOG_PATH = STATE_DIR / "paper_trades.jsonl"
RUN_LOG_PATH = STATE_DIR / "bot_runs.jsonl"
