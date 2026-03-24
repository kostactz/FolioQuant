import re

with open("src/services/metrics_service.py", "r") as f:
    text = f.read()

# 1. Update _evaluate_trade to match the advanced logic
# Replace the whole _evaluate_trade body
new_eval_trade = """    def _evaluate_trade(self, signal: OFISignal) -> None:
        if len(self.signal_history) < 2:
            return
            
        prev_data = self.signal_history[-2]
        curr_data = self.signal_history[-1]
        
        prev_timestamp, prev_ofi, prev_price, prev_spread = prev_data
        curr_timestamp, curr_ofi, curr_price, curr_spread = curr_data
        
        # Phase 3b: Latency Compensation
        decision_ofi = prev_ofi if self.use_lagged_ofi else curr_ofi
        
        target_pos = self._target_position(decision_ofi, self.current_position)
        
        # Phase 3a: Exit Logic Integration
        if self.current_position != Decimal('0.0') and getattr(self, 'position_entry_info', None) is not None:
            ticks_held = len(self.signal_history) - self.position_entry_info['entry_tick']
            should_exit, exit_reason = self._should_exit_position(
                entry_price=self.position_entry_info['entry_price'],
                current_price=curr_price,
                entry_signal=self.position_entry_info['entry_signal'],
                current_signal=decision_ofi,
                ticks_held=ticks_held
            )
            if should_exit:
                target_pos = Decimal('0.0')
                logger.debug(f"[EXIT] Triggered: {exit_reason}, held {ticks_held} ticks")

        # 3. Execute trade only if the target position actually changed direction/size
        # We ignore micro-changes in size for logging to avoid noise
        direction_changed = (target_pos * self.current_position <= 0 and target_pos != self.current_position)
        size_changed = False
        if target_pos != 0 and self.current_position != 0:
             size_changed = abs(target_pos - self.current_position) / abs(self.current_position) > Decimal('0.1')
             
        if direction_changed or size_changed or (target_pos == 0 and self.current_position != 0):
            trade_size = abs(target_pos - self.current_position)
            side = 'buy' if target_pos > self.current_position else 'sell'
            
            trade = {
                'timestamp': signal.timestamp,
                'side': side,
                'price': float(curr_price),
                'size': float(trade_size),
                'ofi': float(curr_ofi)
            }
            
            trade_logger.info(
                "trade side=%s size=%.4f price=%.2f ofi=%.4f from_pos=%.2f to_pos=%.2f",
                side,
                float(trade_size),
                float(curr_price),
                float(curr_ofi),
                float(self.current_position),
                float(target_pos),
            )
            
            self.recent_trades.append(trade)
            
            # Track new position entry
            if target_pos != Decimal('0.0') and self.current_position == Decimal('0.0'):
                self.position_entry_info = {
                    'entry_price': curr_price,
                    'entry_signal': decision_ofi,
                    'entry_tick': len(self.signal_history)
                }
            # Clean up exited positions
            elif target_pos == Decimal('0.0'):
                self.position_entry_info = None
                
            self.current_position = target_pos"""

# Find def _evaluate_trade and replace up to def _update_returns
text = re.sub(r'    def _evaluate_trade\(self, signal: OFISignal\) -> None:.*?    def _update_returns\(self\) -> None:', new_eval_trade + '\n\n    def _update_returns(self) -> None:', text, flags=re.DOTALL)

# 2. Fix _target_position to not flatten on exit_threshold since Phase 3a handles it
new_target_pos = """        direction = current_dir
        if current_dir == Decimal('0.0'):
            if signal_value > entry_threshold: direction = Decimal('1.0')
            elif signal_value < -entry_threshold: direction = Decimal('-1.0')
        elif current_dir == Decimal('1.0'):
            if signal_value < -entry_threshold: direction = Decimal('-1.0')
            # Let Phase 3a handle the flattening:
            # elif signal_value < exit_threshold: direction = Decimal('0.0')
        elif current_dir == Decimal('-1.0'):
            if signal_value > entry_threshold: direction = Decimal('1.0')
            # Let Phase 3a handle the flattening:
            # elif signal_value > -exit_threshold: direction = Decimal('0.0')"""
text = re.sub(r'        direction = current_dir\n        if current_dir == Decimal\(\'0\.0\'\):.*?elif signal_value > -exit_threshold: direction = Decimal\(\'0\.0\'\)', new_target_pos, text, flags=re.DOTALL)

with open("src/services/metrics_service.py", "w") as f:
    f.write(text)

