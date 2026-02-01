
import unittest
from decimal import Decimal
from datetime import datetime, timedelta
from src.services.metrics_service import MetricsService
from src.models.signals import OFISignal

class TestMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_sharpe_explosion(self):
        """Test how Sharpe Ratio behaves with high frequency updates and flat periods."""
        # Use larger window for bucketing
        service = MetricsService(window_size=1000, use_dynamic_scaling=False)
        
        base_time = datetime.now()
        base_price = Decimal("50000.00")
        spread = Decimal("5.00") # 1 bp spread
        
        # Simulate 200 updates over 10 seconds (50ms interval)
        for i in range(200):
            timestamp = base_time + timedelta(milliseconds=50 * i) 
            
            # Create a signal
            # Flip OFI every 10 ticks to incur costs
            if (i // 10) % 2 == 0:
                ofi = Decimal("10.0")
            else:
                ofi = Decimal("-10.0") 
            
            # Pattern: move every 10 ticks
            if i == 0:
                price = base_price
            else:
                if i % 10 < 6: # Flat
                    price = base_price
                elif i % 10 < 9: # Win
                    price = base_price + Decimal("5.00")
                else: # Loss
                    price = base_price - Decimal("5.00")
            
            base_price = price
            
            signal = OFISignal(
                timestamp=timestamp,
                ofi_value=ofi,
                window_size=1,
                max_window_size=100,
                event_count=i,
                mid_price=price,
                spread=spread,
                micro_price=price, 
                mean_event_value=ofi,
                std_event_value=Decimal(0),
                min_event_value=ofi,
                max_event_value=ofi
            )
            
            await service.on_signal_update(signal)
            
        snapshot = service.get_metrics_snapshot()
        print(f"\nSimulation Results:")
        if snapshot.sharpe_ratio:
            print(f"Sharpe: {snapshot.sharpe_ratio}")
        else:
            print("Sharpe: None (Insufficient Data)")
        print(f"Hit Rate: {snapshot.hit_rate}%")
        print(f"Win/Loss: {snapshot.win_loss_ratio}")
        print(f"Max Drawdown: {snapshot.max_drawdown}%")
        
        # Expect Sharpe to be much lower, potentially negative due to spread costs
        # We trade often (every flat->win switch).
        
if __name__ == '__main__':
    unittest.main()
