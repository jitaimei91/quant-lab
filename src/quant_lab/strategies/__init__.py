"""Strategy package.

Strategies are auto-registered when their module is imported. The morning
runner imports all strategy modules to populate the registry.
"""
from .base import Strategy, register, get_all  # noqa: F401
from . import spy_vol, qqq_vol, momo, meanrev  # noqa: F401  (triggers @register decorators)
