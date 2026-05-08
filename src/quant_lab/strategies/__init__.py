"""Strategy package.

Strategies are auto-registered when their module is imported. The morning
runner imports all strategy modules to populate the registry.
"""
from .base import Strategy, register, get_all  # noqa: F401
from . import spy_vol, qqq_vol, momo, meanrev, breakout, ma_cross, rsi_rev, codex_bot  # noqa: F401  (triggers @register decorators)
from . import risk_parity  # noqa: F401  (registers RiskParity — orthogonal SPY/TLT/GLD sleeve)
from . import apex  # noqa: F401  (registers Apex — dual-momentum + vol-targeting + master switches)
from . import calendar  # noqa: F401  (registers Calendar — turn-of-month SPY sleeve)
from . import regime_aware  # noqa: F401  (registers RegimeMomo, RegimeMeanRev, RegimeBreakout)
from . import gradboost, lightforest, ml_ensemble  # noqa: F401  (ML bots — always register, SPY fallback when gates fail)
from . import ensemble  # noqa: F401  (MetaEnsemble — must import LAST so it sees all components)
