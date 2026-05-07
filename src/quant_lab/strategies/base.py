from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Type

from ..types import Bar


_REGISTRY: dict[str, "Strategy"] = {}


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must define `bot_id` and `description` as class attributes,
    and implement `target_weights`.
    """

    bot_id: str = ""
    description: str = ""

    @abstractmethod
    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        """Return target portfolio weights as {symbol: weight}.

        Weights should sum to <= 1.0; the remainder is held in cash.
        Implementations must use only data with date <= as_of.
        """


def register(strategy_cls: Type[Strategy]) -> Type[Strategy]:
    """Register a Strategy subclass. Decorator-friendly."""
    instance = strategy_cls()
    if not instance.bot_id:
        raise ValueError(f"{strategy_cls.__name__} missing bot_id")
    _REGISTRY[instance.bot_id] = instance
    return strategy_cls


def get_all() -> list[Strategy]:
    """Return all registered strategy instances."""
    return list(_REGISTRY.values())


def get(bot_id: str) -> Strategy:
    return _REGISTRY[bot_id]
