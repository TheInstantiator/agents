# Detailed Design for accounts.py Module

## Overview
This document provides a detailed design for a single Python module named `accounts.py` that implements a simple account management system for a trading simulation platform. The module is self-contained, meaning it includes all necessary classes, methods, and a test implementation for the `get_share_price` function. It can be directly used for testing or integrated with a simple UI.

The core class is `Account`, which manages user accounts, including fund deposits/withdrawals, buying/selling shares, calculating portfolio values, profits/losses, reporting holdings, and listing transactions. All operations include validation to prevent invalid actions (e.g., negative balances, insufficient funds for buys, or selling unowned shares).

The module assumes a simulation environment where share prices are fetched via `get_share_price(symbol)`, with a test implementation returning fixed prices:
- AAPL: 150.0
- TSLA: 700.0
- GOOGL: 2800.0

For any other symbol, it raises a `ValueError`.

Transactions are recorded with timestamps for historical reporting. The system uses Python's standard library (e.g., `datetime` for timestamps) to keep it self-contained.

## Module Structure
- **File Name**: accounts.py
- **Imports**: Only standard library imports (e.g., `from datetime import datetime`).
- **Global Functions**: 
  - `get_share_price(symbol)`: A test function to simulate share price retrieval.
- **Classes**:
  - `Account`: The main class for managing a user's account.
- **Exceptions**: Custom exceptions for error handling (e.g., `InsufficientFundsError`, `InsufficientSharesError`, `NegativeBalanceError`).

The module is designed to be instantiated and used as follows:
```python
# Example usage
from accounts import Account

account = Account("user123")
account.deposit(1000.0)
account.buy("AAPL", 5)
print(account.get_portfolio_value())
print(account.get_profit_loss())
```

## Global Functions

### get_share_price(symbol: str) -> float
- **Description**: Retrieves the current price of a share for the given symbol. This is a test implementation returning fixed prices for simulation purposes. In a real system, this could be replaced with an API call.
- **Parameters**:
  - `symbol`: The stock symbol (e.g., "AAPL").
- **Returns**: The current price as a float.
- **Raises**: `ValueError` if the symbol is not recognized.
- **Functionality**:
  - Uses a dictionary of fixed prices: {"AAPL": 150.0, "TSLA": 700.0, "GOOGL": 2800.0}.
  - Example: `get_share_price("AAPL")` returns 150.0.

## Classes

### class Account
- **Description**: Represents a user's trading account. Manages balance, holdings (dictionary of symbol to quantity), transaction history, and initial deposit for profit/loss calculations. All methods ensure thread-safety is not required for this simulation, but could be added if needed.
- **Attributes** (Private):
  - `_user_id`: str - Unique identifier for the user.
  - `_balance`: float - Current cash balance (initially 0.0).
  - `_holdings`: dict[str, float] - Dictionary of stock symbols to quantities owned (e.g., {"AAPL": 5.0}).
  - `_transactions`: list[dict] - List of transaction records, each a dict with keys: 'timestamp' (datetime), 'type' (str: 'deposit', 'withdraw', 'buy', 'sell'), 'amount' (float, for funds), 'symbol' (str, optional), 'quantity' (float, optional), 'price' (float, optional for trades).
  - `_initial_deposit`: float - Tracks the very first deposit for profit/loss calculation (updated only on first deposit).
- **Initializer**:
  - `__init__(self, user_id: str)`: Initializes the account with a user ID, zero balance, empty holdings, empty transactions, and initial_deposit=0.0.

#### Methods

##### deposit(self, amount: float) -> None
- **Description**: Adds funds to the account balance and records the transaction. If this is the first deposit, sets the initial_deposit.
- **Parameters**:
  - `amount`: Positive float amount to deposit.
- **Raises**: `ValueError` if amount <= 0.
- **Functionality**:
  - Updates `_balance += amount`.
  - If `_initial_deposit == 0`, sets `_initial_deposit = amount`.
  - Appends a transaction: {'timestamp': datetime.now(), 'type': 'deposit', 'amount': amount}.

##### withdraw(self, amount: float) -> None
- **Description**: Removes funds from the account balance, preventing negative balance.
- **Parameters**:
  - `amount`: Positive float amount to withdraw.
- **Raises**:
  - `ValueError` if amount <= 0.
  - `NegativeBalanceError` if withdrawal would result in `_balance < 0`.
- **Functionality**:
  - Checks if `_balance >= amount`.
  - Updates `_balance -= amount`.
  - Appends a transaction: {'timestamp': datetime.now(), 'type': 'withdraw', 'amount': amount}.

##### buy(self, symbol: str, quantity: float) -> None
- **Description**: Records a buy transaction for shares, deducting cost from balance if affordable.
- **Parameters**:
  - `symbol`: Stock symbol.
  - `quantity`: Positive float quantity to buy.
- **Raises**:
  - `ValueError` if quantity <= 0 or symbol invalid (via get_share_price).
  - `InsufficientFundsError` if cost > current balance.
- **Functionality**:
  - price = get_share_price(symbol)
  - cost = price * quantity
  - If `_balance < cost`, raise error.
  - Update `_balance -= cost`
  - Update `_holdings[symbol] = _holdings.get(symbol, 0) + quantity`
  - Append transaction: {'timestamp': datetime.now(), 'type': 'buy', 'symbol': symbol, 'quantity': quantity, 'price': price}

##### sell(self, symbol: str, quantity: float) -> None
- **Description**: Records a sell transaction for shares, adding proceeds to balance if shares are owned.
- **Parameters**:
  - `symbol`: Stock symbol.
  - `quantity`: Positive float quantity to sell.
- **Raises**:
  - `ValueError` if quantity <= 0 or symbol invalid.
  - `InsufficientSharesError` if quantity > owned quantity.
- **Functionality**:
  - price = get_share_price(symbol)
  - owned = _holdings.get(symbol, 0)
  - If quantity > owned, raise error.
  - proceeds = price * quantity
  - Update `_balance += proceeds`
  - Update `_holdings[symbol] = owned - quantity` (remove if == 0)
  - Append transaction: {'timestamp': datetime.now(), 'type': 'sell', 'symbol': symbol, 'quantity': quantity, 'price': price}

##### get_portfolio_value(self) -> float
- **Description**: Calculates the current total value of holdings based on current prices.
- **Returns**: Float representing sum of (quantity * current_price) for all holdings.
- **Functionality**:
  - Initialize value = 0.0
  - For each symbol, quantity in _holdings: value += quantity * get_share_price(symbol)
  - Return value

##### get_profit_loss(self) -> float
- **Description**: Calculates profit/loss as (current_balance + portfolio_value) - initial_deposit.
- **Returns**: Float (positive for profit, negative for loss).
- **Functionality**:
  - total_value = self._balance + self.get_portfolio_value()
  - return total_value - self._initial_deposit

##### get_holdings(self) -> dict[str, float]
- **Description**: Reports current holdings.
- **Returns**: Copy of _holdings dictionary.
- **Functionality**: return dict(self._holdings)

##### get_transactions(self) -> list[dict]
- **Description**: Lists all transactions in chronological order.
- **Returns**: Copy of _transactions list.
- **Functionality**: return list(self._transactions)

## Custom Exceptions
- **InsufficientFundsError(Exception)**: Raised when buying more than affordable.
- **InsufficientSharesError(Exception)**: Raised when selling more than owned.
- **NegativeBalanceError(Exception)**: Raised when withdrawal would cause negative balance.

## Additional Notes
- All methods are designed for simplicity and self-containment. No external dependencies beyond Python standard library.
- For testing: The module can be imported and an Account instance created. Methods can be called sequentially to simulate trading, with prints or assertions for verification.
- Edge Cases Handled: Zero quantities, unknown symbols, multiple buys/sells of same symbol, no initial deposit (profit/loss=0), empty holdings/transactions.
- Extensibility: In a real system, persist data to a database, but here it's in-memory for self-containment.