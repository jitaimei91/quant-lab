"""Strategy package.

Strategies are auto-registered when their module is imported. The morning
runner imports all strategy modules to populate the registry.
"""
from .base import Strategy, register, get_all  # noqa: F401
