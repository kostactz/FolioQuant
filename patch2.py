import re

with open("src/services/metrics_service.py", "r") as f:
    text = f.read()

old_code = """        current_return = (current_price - entry_price) / entry_price
        
        # Condition 1: Stop-loss at -5% drawdown
        # This bounds maximum loss per trade and prevents spiral into deeper loss
        STOP_LOSS_THRESHOLD = Decimal('-0.05')
        if current_return < STOP_LOSS_THRESHOLD:
            return True, "stop_loss"
        
        # Condition 2: Profit-taking at +2% gain
        # Lock in wins to avoid giving back edge
        PROFIT_TARGET = Decimal('0.02')
        if current_return > PROFIT_TARGET:
            return True, "profit_taking\""""

new_code = """        price_return = (current_price - entry_price) / entry_price
        direction = Decimal('1.0') if entry_signal > 0 else Decimal('-1.0')
        position_return = price_return * direction
        
        # Condition 1: Stop-loss at -5% drawdown
        # This bounds maximum loss per trade and prevents spiral into deeper loss
        STOP_LOSS_THRESHOLD = Decimal('-0.05')
        if position_return < STOP_LOSS_THRESHOLD:
            return True, "stop_loss"
        
        # Condition 2: Profit-taking at +2% gain
        # Lock in wins to avoid giving back edge
        PROFIT_TARGET = Decimal('0.02')
        if position_return > PROFIT_TARGET:
            return True, "profit_taking\""""

if old_code in text:
    text = text.replace(old_code, new_code)
    with open("src/services/metrics_service.py", "w") as f:
        f.write(text)
    print("Patched return logic in exit position.")
else:
    print("Could not find the old code to replace!")
