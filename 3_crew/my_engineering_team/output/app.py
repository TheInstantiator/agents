import gradio as gr
from accounts import Account, InsufficientFundsError, InsufficientSharesError, NegativeBalanceError

def create_account(user_id):
    return Account(user_id)

def deposit(account, amount):
    account.deposit(amount)
    return f"Deposited {amount} successfully."

def withdraw(account, amount):
    try:
        account.withdraw(amount)
        return f"Withdrew {amount} successfully."
    except NegativeBalanceError as e:
        return str(e)

def buy_shares(account, symbol, quantity):
    try:
        account.buy(symbol, quantity)
        return f"Bought {quantity} shares of {symbol} successfully."
    except InsufficientFundsError as e:
        return str(e)
    except ValueError as e:
        return str(e)

def sell_shares(account, symbol, quantity):
    try:
        account.sell(symbol, quantity)
        return f"Sold {quantity} shares of {symbol} successfully."
    except InsufficientSharesError as e:
        return str(e)
    except ValueError as e:
        return str(e)

def get_portfolio_value(account):
    return f"Portfolio value is ${account.get_portfolio_value():,.2f}."

def get_profit_loss(account):
    profit_loss = account.get_profit_loss()
    if profit_loss > 0:
        return f"Profit/Loss: +${abs(profit_loss):,.2f}"
    else:
        return f"Profit/Loss: -${abs(profit_loss):,.2f}"

def get_holdings(account):
    holdings = account.get_holdings()
    return ", ".join([f"{symbol}: {quantity}" for symbol, quantity in holdings.items()])

def get_transactions(account):
    transactions = account.get_transactions()
    trans_str = "\n".join([f"{t['timestamp']} - {t['type'].capitalize()}: {t['amount'] if t['type'] == 'deposit' or t['type'] == 'withdraw' else f'{t['quantity']} shares of {t['symbol']}'}" for t in transactions])
    return trans_str

def main(user_id, action, amount=None, symbol=None, quantity=None):
    account = create_account(user_id)
    if action == "Deposit":
        return deposit(account, float(amount))
    elif action == "Withdraw":
        return withdraw(account, float(amount))
    elif action == "Buy Shares":
        return buy_shares(account, symbol, float(quantity))
    elif action == "Sell Shares":
        return sell_shares(account, symbol, float(quantity))
    elif action == "Portfolio Value":
        return get_portfolio_value(account)
    elif action == "Profit/Loss":
        return get_profit_loss(account)
    elif action == "Holdings":
        return get_holdings(account)
    elif action == "Transactions":
        return get_transactions(account)

with gr.Blocks() as demo:
    with gr.Row():
        user_id = gr.Textbox(label="User ID")
        account = gr.State()
    
    with gr.Row():
        action = gr.Dropdown(["Deposit", "Withdraw", "Buy Shares", "Sell Shares", "Portfolio Value", "Profit/Loss", "Holdings", "Transactions"], label="Action")
        amount = gr.Number(label="Amount (USD)")
        symbol = gr.Textbox(label="Symbol")
        quantity = gr.Number(label="Quantity")
    
    with gr.Row():
        output = gr.Textbox(label="Output")
    
    user_id.submit(create_account, inputs=user_id, outputs=account)
    action.select(main, inputs=[user_id, account, action, amount, symbol, quantity], outputs=output)

demo.launch()