 To write unit tests for the `accounts.py` module, we need to first create a separate test file named `test_accounts.py`. We will then use Python's built-in `unittest` framework along with the `datetime` fixture provided by the `datetime` module from the standard library.

```python
from datetime import datetime
import unittest
from accounts import Account, InsufficientFundsError, InsufficientSharesError, NegativeBalanceError

class TestAccount(unittest.TestCase):
    def setUp(self):
        self.account = Account("user123")

    def test_initial_balance(self):
        self.assertEqual(self.account._balance, 0.0)
        self.assertEqual(self.account._holdings, {})

    def test_deposit_positive_amount(self):
        self.account.deposit(100.0)
        self.assertEqual(self.account._balance, 100.0)

    def test_deposit_zero_amount(self):
        with self.assertRaises(ValueError):
            self.account.deposit(0.0)

    def test_withdraw_sufficient_funds(self):
        self.account.deposit(100.0)
        self.account.withdraw(50.0)
        self.assertEqual(self.account._balance, 50.0)

    def test_withdraw_insufficient_funds(self):
        with self.assertRaises(NegativeBalanceError):
            self.account.withdraw(10.0)

    def test_buy_sufficient_funds(self):
        self.account.deposit(500.0)
        self.account.buy("AAPL", 2)
        self.assertEqual(self.account._balance, 3400.0)
        self.assertIn("AAPL", self.account._holdings)
        self.assertEqual(self.account._holdings["AAPL"], 2)

    def test_buy_insufficient_funds(self):
        with self.assertRaises(InsufficientFundsError):
            self.account.buy("AAPL", 10)

    def test_sell_sufficient_shares(self):
        self.account.deposit(500.0)
        self.account.buy("AAPL", 2)
        self.account.sell("AAPL", 1)
        self.assertEqual(self.account._balance, 4300.0)
        self.assertIn("AAPL", self.account._holdings)
        self.assertEqual(self.account._holdings["AAPL"], 1)

    def test_sell_insufficient_shares(self):
        with self.assertRaises(InsufficientSharesError):
            self.account.sell("AAPL", 5)

    def test_get_portfolio_value(self):
        self.account.deposit(1000.0)
        self.account.buy("TSLA", 3)
        value = self.account.get_portfolio_value()
        self.assertAlmostEqual(value, 2800 * 3 + (150.0 if "AAPL" in self.account._holdings else 0), places=2)

    def test_get_profit_loss(self):
        self.account.deposit(1000.0)
        initial_value = self.account._initial_deposit
        current_balance = self.account._balance + self.account.get_portfolio_value()
        profit_loss = self.account.get_profit_loss()
        self.assertEqual(profit_loss, current_balance - initial_value)

if __name__ == "__main__":
    unittest.main()
```