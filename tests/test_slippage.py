import math
import pytest
from quant_lab.slippage import spread_bps, apply_slippage


def test_spread_bps_floors_at_one():
    # Hypothetical infinite-liquidity name
    assert spread_bps(adv_dollars=1e15) >= 1.0


def test_spread_bps_grows_for_low_liquidity():
    high = spread_bps(adv_dollars=10e6)   # $10M ADV (low)
    low = spread_bps(adv_dollars=10e9)    # $10B ADV (high)
    assert high > low


def test_apply_slippage_buy_increases_price():
    fill = apply_slippage(price=100.0, side="BUY", spread_bps_value=10.0)
    assert math.isclose(fill, 100.0 * (1 + 10.0 / 10_000))


def test_apply_slippage_sell_decreases_price():
    fill = apply_slippage(price=100.0, side="SELL", spread_bps_value=10.0)
    assert math.isclose(fill, 100.0 * (1 - 10.0 / 10_000))


def test_apply_slippage_invalid_side_raises():
    with pytest.raises(ValueError):
        apply_slippage(price=100.0, side="HOLD", spread_bps_value=10.0)
