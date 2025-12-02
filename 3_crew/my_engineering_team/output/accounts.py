from datetime import datetime

class InsufficientFundsError(Exception):
    pass

class InsufficientSharesError(Exception):
    pass

class NegativeBalanceError(Exception):
    pass

def get_share_price(symbol):
    prices = {"AAPL": 150.0, "TSLA": 700.0, "GOOGL": 2800.0}
    if symbol not in prices:
        raise ValueError("Invalid symbol")
    return prices[symbol]

class Account:
    def __init__(self, user_id):
        self._user_id = user_id
        self._balance = 0.0
        self._holdings = {}
        self._transactions = []
        self._initial_deposit = 0.0

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        if self._initial_deposit == 0:
            self._initial_deposit = amount
        self._transactions.append({
            'timestamp': datetime.now(),
            'type': 'deposit',
            'amount': amount
        })

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if self._balance < amount:
            raise NegativeBalanceError("Insufficient funds to withdraw")
        self._balance -= amount
        self._transactions.append({
            'timestamp': datetime.now(),
            'type': 'withdraw',
            'amount': amount
        })

    def buy(self, symbol, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        price = get_share_price(symbol)
        cost = price * quantity
        if self._balance < cost:
            raise InsufficientFundsError("Insufficient funds to buy shares")
        self._balance -= cost
        if symbol in self._holdings:
            self._holdings[symbol] += quantity
        else:
            self._holdings[symbol] = quantity
        self._transactions.append({
            'timestamp': datetime.now(),
            'type': 'buy',
            'symbol': symbol,
            'quantity': quantity,
            'price': price
        })

    def sell(self, symbol, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        price = get_share_price(symbol)
        owned = self._holdings.get(symbol, 0)
        if quantity > owned:
            raise InsufficientSharesError("Insufficient shares to sell")
        proceeds = price * quantity
        self._balance += proceeds
        if owned == quantity:
            del self._holdings[symbol]
        else:
            self._holdings[symbol] -= quantity
        self._transactions.append({
            'timestamp': datetime.now(),
            'type': 'sell',
            'symbol': symbol,
            'quantity': quantity,
            'price': price
        })

    def get_portfolio_value(self):
        value = 0.0
        for symbol, quantity in self._holdings.items():
            value += quantity * get_share_price(symbol)
        return value

    def get_profit_loss(self):
        total_value = self._balance + self.get_portfolio_value()
        return total_value - self._initial_deposit

    def get_holdings(self):
        return dict(self._holdings)

    def get_transactions(self):
        return list(self._transactions)