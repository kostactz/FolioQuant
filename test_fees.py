import asyncio
from decimal import Decimal
from datetime import datetime
from src.services.metrics_service import MetricsService
from src.models.signals import OFISignal

async def test():
    service = MetricsService(window_size=100, use_dynamic_scaling=True)
    service.signal_threshold = Decimal('0.5')
    
    # Fake history
    base_time = datetime.now()
    price = 60000.0
    
    for i in range(100):
        # Good signal that stays active
        sig = OFISignal(
            timestamp=base_time,
            ofi_value=Decimal('1.0'),
            window_size=1, max_window_size=100, event_count=1,
            mid_price=Decimal(str(price)), spread=Decimal('1.0'),
            micro_price=Decimal(str(price)), mean_event_value=Decimal('1.0'),
            std_event_value=Decimal('0.1'), min_event_value=Decimal('1.0'), max_event_value=Decimal('1.0')
        )
        await service.on_signal_update(sig)
        price += 1.0
        
    returns = service._compute_strategy_returns()
    costs = [item['return'] for item in returns if item['return'] < 0]
    print(f"Num negative returns: {len(costs)}")

asyncio.run(test())
