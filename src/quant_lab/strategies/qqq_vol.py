from __future__ import annotations

from .spy_vol import _VolTargetedIndex
from .base import register


@register
class QQQVol(_VolTargetedIndex):
    bot_id = "qqq-vol"
    description = "Vol-targeted long Nasdaq-100 (QQQ). Honest Nasdaq benchmark."
    symbol = "QQQ"
