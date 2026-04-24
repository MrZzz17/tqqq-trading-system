"""B.5 #2: trade_count_breakdown closed + open is the canonical total for UI copy."""
from core.dashboard_metrics import trade_count_breakdown


def test_synthetic_year_rows_plus_open_leg():
    class YR:
        def __init__(self, n, open_leg=None):
            self.num_trades = n
            self.open_leg = open_leg

    live = type("L", (), {"in_position": True, "shares": 10.0})()
    closed, op = trade_count_breakdown([YR(5), YR(6), YR(0, open_leg={"x": 1})], live)
    assert closed == 11
    assert op == 1
    assert closed + op == 12
