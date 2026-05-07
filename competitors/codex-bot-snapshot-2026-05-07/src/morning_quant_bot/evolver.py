from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backtest import common_dates, run_backtest, split_periods
from .models import BacktestMetrics, Bar, StrategyParams
from .strategy import DEFAULT_STRATEGY, mutate_params, random_params


@dataclass
class StrategyRecord:
    params: StrategyParams
    fitness: float
    train: BacktestMetrics
    validation: BacktestMetrics
    test: BacktestMetrics
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.to_dict(),
            "fitness": self.fitness,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyRecord":
        return cls(
            params=StrategyParams.from_dict(raw["params"]),
            fitness=float(raw["fitness"]),
            train=BacktestMetrics(**raw["train"]),
            validation=BacktestMetrics(**raw["validation"]),
            test=BacktestMetrics(**raw["test"]),
            evaluated_at=str(raw["evaluated_at"]),
        )


class StrategyEvolver:
    def __init__(self, state_path: Path, seed: int = 11) -> None:
        self.state_path = state_path
        self.rng = random.Random(seed)

    def evolve(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        generations: int = 4,
        population_size: int = 32,
        elite_count: int = 8,
    ) -> list[StrategyRecord]:
        candidates = self._initial_population(population_size)
        seen = {candidate.key() for candidate in candidates}

        all_records: dict[str, StrategyRecord] = {}
        for generation in range(generations):
            for candidate in candidates:
                if candidate.key() in all_records:
                    continue
                all_records[candidate.key()] = self.evaluate(bars_by_symbol, candidate)

            elites = sorted(
                all_records.values(),
                key=lambda record: record.fitness,
                reverse=True,
            )[:elite_count]

            next_candidates = [record.params for record in elites]
            while len(next_candidates) < population_size:
                parent = self.rng.choice(elites).params if elites else DEFAULT_STRATEGY
                child = mutate_params(parent, self.rng)
                if child.key() in seen:
                    child = random_params(self.rng)
                seen.add(child.key())
                next_candidates.append(child)
            candidates = next_candidates

        records = sorted(
            all_records.values(),
            key=lambda record: record.fitness,
            reverse=True,
        )
        self.save(records[:50])
        return records

    def evaluate(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        params: StrategyParams,
    ) -> StrategyRecord:
        dates = common_dates(bars_by_symbol)
        periods = split_periods(dates)
        if periods is None:
            full = run_backtest(bars_by_symbol, params).metrics
            fitness = _fitness(full, full, full)
            return StrategyRecord(
                params=params,
                fitness=fitness,
                train=full,
                validation=full,
                test=full,
                evaluated_at=_now(),
            )

        start, train_end, validation_end, end = periods
        train = run_backtest(
            bars_by_symbol,
            params,
            start_date=start,
            end_date=train_end,
        ).metrics
        validation = run_backtest(
            bars_by_symbol,
            params,
            start_date=train_end,
            end_date=validation_end,
        ).metrics
        test = run_backtest(
            bars_by_symbol,
            params,
            start_date=validation_end,
            end_date=end,
        ).metrics
        return StrategyRecord(
            params=params,
            fitness=_fitness(train, validation, test),
            train=train,
            validation=validation,
            test=test,
            evaluated_at=_now(),
        )

    def load(self) -> list[StrategyRecord]:
        if not self.state_path.exists():
            return []
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return [StrategyRecord.from_dict(item) for item in raw.get("leaderboard", [])]

    def save(self, records: list[StrategyRecord]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _now(),
            "leaderboard": [record.to_dict() for record in records],
        }
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _initial_population(self, population_size: int) -> list[StrategyParams]:
        loaded = [record.params for record in self.load()]
        candidates = [DEFAULT_STRATEGY] + loaded[: max(0, population_size // 4)]
        while len(candidates) < population_size:
            if loaded and self.rng.random() < 0.40:
                candidates.append(mutate_params(self.rng.choice(loaded), self.rng))
            else:
                candidates.append(random_params(self.rng))
        return candidates[:population_size]


def _fitness(
    train: BacktestMetrics,
    validation: BacktestMetrics,
    test: BacktestMetrics,
) -> float:
    # Validation and test matter more than train to reduce curve-fitting.
    score = (
        0.15 * train.sharpe
        + 0.45 * validation.sharpe
        + 0.40 * test.sharpe
        + 0.50 * validation.cagr
        + 0.30 * test.cagr
    )
    drawdown_penalty = abs(min(train.max_drawdown, validation.max_drawdown, test.max_drawdown, 0.0))
    turnover_penalty = 0.002 * (train.turnover + validation.turnover + test.turnover)
    fragility_penalty = 0.0
    if validation.total_return < 0:
        fragility_penalty += 0.50
    if test.total_return < 0:
        fragility_penalty += 0.75
    if train.sharpe > 1.5 and validation.sharpe < 0.4:
        fragility_penalty += 0.40
    return score - drawdown_penalty - turnover_penalty - fragility_penalty


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

