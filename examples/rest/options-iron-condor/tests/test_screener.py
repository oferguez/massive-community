import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screener import IronCondorScreener  # noqa: E402


class _TestClient:
    """Minimal placeholder client for unit tests."""
    pass


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.screener = IronCondorScreener(client=_TestClient())

    def test_probability_monotonic_with_strike(self):
        higher_strike_prob = self.screener.calculate_black_scholes_probability(
            spot=100, strike=120, days_to_exp=30, volatility=0.25
        )
        lower_strike_prob = self.screener.calculate_black_scholes_probability(
            spot=100, strike=90, days_to_exp=30, volatility=0.25
        )

        self.assertGreater(higher_strike_prob, lower_strike_prob)
        self.assertTrue(0.0 <= higher_strike_prob <= 1.0)
        self.assertTrue(0.0 <= lower_strike_prob <= 1.0)


class RiskMetricTests(unittest.TestCase):
    def setUp(self):
        self.screener = IronCondorScreener(client=_TestClient())

    def test_widest_spread_sets_max_loss(self):
        calls = [
            {'strike': 105, 'bid': 2.5, 'ask': 2.7, 'volume': 100, 'open_interest': 200, 'implied_volatility': 0.3},
            {'strike': 110, 'bid': 1.1, 'ask': 1.2, 'volume': 100, 'open_interest': 200, 'implied_volatility': 0.3},
        ]
        puts = [
            {'strike': 95, 'bid': 2.4, 'ask': 2.6, 'volume': 100, 'open_interest': 200, 'implied_volatility': 0.32},
            {'strike': 88, 'bid': 1.2, 'ask': 1.3, 'volume': 100, 'open_interest': 200, 'implied_volatility': 0.32},
        ]

        condors = self.screener.construct_iron_condors(
            symbol="TEST",
            spot_price=100,
            options_chain={'calls': calls, 'puts': puts},
            expiration="2025-12-19",
            min_vol=1,
            min_oi=1
        )

        self.assertTrue(condors, "Expected at least one iron condor to be constructed")
        target = next(
            (ic for ic in condors if ic.call_spread == (105, 110) and ic.put_spread == (95, 88)),
            None
        )
        self.assertIsNotNone(target, "Could not find expected spread combination in test output")

        expected_max_loss = round(7 - target.net_credit, 2)
        self.assertEqual(target.max_loss, expected_max_loss)
        self.assertGreater(target.probability_of_profit, 0)
        self.assertLessEqual(target.probability_of_profit, 100)


class PnLTests(unittest.TestCase):
    class _ScreenerWithFixedCloses(IronCondorScreener):
        def __init__(self, close_prices):
            super().__init__(client=_TestClient())
            self._close_prices = close_prices

        def _get_expiration_close_price(self, symbol, expiration):
            return self._close_prices.get((symbol, expiration))

    def _write_csv(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
        writer = csv.DictWriter(tmp, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.flush()
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_calculate_pnl_uses_fetched_closes(self):
        rows = [
            {
                'symbol': 'AAPL',
                'expiration': '2025-01-17',
                'call_sell_strike': 120,
                'call_buy_strike': 125,
                'put_sell_strike': 110,
                'put_buy_strike': 105,
                'net_credit': 2.0,
                'max_profit': 2.0,
                'max_loss': 3.0,
                'profit_zone_lower': 110,
                'profit_zone_upper': 120,
                'probability_of_profit': 60,
                'risk_reward_ratio': 0.4,
                'days_to_expiration': 5,
                'spot_price': 118,
                'has_upcoming_earnings': False,
                'timestamp': '2025-01-10T00:00:00',
            },
            {
                'symbol': 'AAPL',
                'expiration': '2025-01-24',
                'call_sell_strike': 130,
                'call_buy_strike': 135,
                'put_sell_strike': 120,
                'put_buy_strike': 115,
                'net_credit': 1.0,
                'max_profit': 1.0,
                'max_loss': 4.0,
                'profit_zone_lower': 120,
                'profit_zone_upper': 130,
                'probability_of_profit': 55,
                'risk_reward_ratio': 0.25,
                'days_to_expiration': 12,
                'spot_price': 125,
                'has_upcoming_earnings': False,
                'timestamp': '2025-01-10T00:00:00',
            },
        ]
        csv_path = self._write_csv(rows)
        screener = self._ScreenerWithFixedCloses({
            ('AAPL', '2025-01-17'): 118,
            ('AAPL', '2025-01-24'): 140,
        })

        summary = screener.calculate_pnl(csv_path)

        self.assertEqual(summary['evaluated_trades'], 2)
        self.assertEqual(summary['profitable_trades'], 1)
        self.assertAlmostEqual(summary['total_pnl'], -2.0, places=2)
        self.assertGreater(summary['win_rate'], 0)
        self.assertLess(summary['win_rate'], 100)


if __name__ == "__main__":
    unittest.main()

