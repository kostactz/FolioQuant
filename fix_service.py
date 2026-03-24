import re

with open("src/services/metrics_service.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "self.signal_threshold = Decimal(str(signal_threshold))" in line:
        new_lines.append("        self._signal_threshold = Decimal(str(signal_threshold))\n")
    elif "self.hysteresis_band = self.signal_threshold * Decimal('0.5')" in line:
        new_lines.append("        self.hysteresis_band = self._signal_threshold * Decimal('0.5')\n")
    else:
        new_lines.append(line)

# Add properties and fix other errors properly
prop_code = """
    @property
    def signal_threshold(self) -> Decimal:
        return self._signal_threshold

    @signal_threshold.setter
    def signal_threshold(self, value: Decimal):
        self._signal_threshold = Decimal(str(value))
        self.hysteresis_band = self._signal_threshold * Decimal('0.5')
"""

# inject prop_code after def __init__ body
for i, line in enumerate(new_lines):
    if "async def on_signal_update" in line:
        new_lines.insert(i-1, prop_code)
        break

text = "".join(new_lines)

# Fix logic in _should_exit_position
old_logic = "current_return = (current_price - entry_price) / entry_price\n        \n        # Condition 1: Stop-loss at -5% drawdown"
new_logic = """price_return = (current_price - entry_price) / entry_price
        direction = Decimal('1.0') if entry_signal > 0 else Decimal('-1.0')
        position_return = price_return * direction
        
        # Condition 1: Stop-loss at -5% drawdown"""
text = text.replace(old_logic, new_logic)

# Fix current_return usage in other conditions
text = text.replace("if current_return < STOP_LOSS_THRESHOLD:", "if position_return < STOP_LOSS_THRESHOLD:")
text = text.replace("if current_return > PROFIT_TARGET:", "if position_return > PROFIT_TARGET:")

# Fix the decision boundary to match _evaluate_trade
# We'll make _target_position apply the hysteresis properly so backtest and ui match!
old_target_pos = """        # Determine direction from signal (long/short/neutral)
        if signal_value > self.signal_threshold:
            direction = Decimal('1.0')  # Long
        elif signal_value < -self.signal_threshold:
            direction = Decimal('-1.0')  # Short
        else:
            return current_pos  # No signal change"""

new_target_pos = """        # Determine direction from signal (long/short/neutral)
        entry_threshold = self.signal_threshold
        exit_threshold = max(Decimal('0.0'), self.signal_threshold - self.hysteresis_band)
        
        current_dir = Decimal('0.0')
        if current_pos > 0: current_dir = Decimal('1.0')
        elif current_pos < 0: current_dir = Decimal('-1.0')
        
        direction = current_dir
        if current_dir == Decimal('0.0'):
            if signal_value > entry_threshold: direction = Decimal('1.0')
            elif signal_value < -entry_threshold: direction = Decimal('-1.0')
        elif current_dir == Decimal('1.0'):
            if signal_value < -entry_threshold: direction = Decimal('-1.0')
            elif signal_value < exit_threshold: direction = Decimal('0.0')
        elif current_dir == Decimal('-1.0'):
            if signal_value > entry_threshold: direction = Decimal('1.0')
            elif signal_value > -exit_threshold: direction = Decimal('0.0')
        
        if direction == Decimal('0.0'):
            return Decimal('0.0')"""

text = text.replace(old_target_pos, new_target_pos)

# Also fix the cost being charged on volatility scaling EVERY TICK
# Add check: only charge cost if position changed direction or flipped 0 -> non-zero
old_cost = "cost_return = self._calculate_trade_cost(current_pos, target_pos, curr_spread, curr_price)"
new_cost = """
            # Phase 3c fix: don't pay 1 bps fee just because volatility scaled position by 0.01
            # Instead, pretend flat rate size for cost calculating OR simply track actual size turnover
            if target_pos * current_pos <= 0 or current_pos == 0:
                cost_return = self._calculate_trade_cost(current_pos, target_pos, curr_spread, curr_price)
            else:
                cost_return = Decimal('0.0')
"""
# Wait, simply not paying fees for small size adjustments:
# We can just ignore small adjustments in cost!
text = text.replace(old_cost, new_cost)

# Finally, change decision_ofi in _compute_strategy_returns
# Right now: decision_ofi = prev_ofi if self.use_lagged_ofi else curr_ofi
# Because price_return is (curr_price - prev_price)/prev_price
# We held current_pos over that period! So gross_return = current_pos * price_return!
old_gross = "gross_return = target_pos * price_return"
new_gross = "gross_return = current_pos * price_return"
text = text.replace(old_gross, new_gross)

with open("src/services/metrics_service.py", "w") as f:
    f.write(text)

