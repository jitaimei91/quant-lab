from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Account


DEFAULT_STARTING_CASH = 10_000.0


def ensure_runtime_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_universe(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        symbols.append(clean.upper())
    if not symbols:
        raise ValueError(f"No symbols found in universe file: {path}")
    return sorted(set(symbols))


def load_account(path: Path) -> Account:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing account file: {path}. Run init-account first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Account.from_dict(raw)


def save_account(path: Path, account: Account) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(account.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_positions(raw: str) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    if not raw.strip():
        return positions
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(
                "Positions must look like SYMBOL:SHARES or SYMBOL:SHARES:AVG_COST"
            )
        symbol = parts[0].upper()
        shares = float(parts[1])
        avg_cost = float(parts[2]) if len(parts) == 3 else 0.0
        positions[symbol] = {"shares": shares, "avg_cost": avg_cost}
    return positions
