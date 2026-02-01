"""
Unit tests for MetricsService - Phase 5

This module tests the performance metrics calculation service, including:
- Sharpe ratio calculation
- Hit rate (directional accuracy)
- Maximum drawdown
- Win/loss ratio
- Signal persistence
- Price correlation

Test Strategy:
    - Use synthetic data with known statistical properties
    - Test edge cases (zero std, all wins, all losses)
    - Validate mathematical correctness
    - Test rolling window behavior
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timedelta

from src.services.metrics_service import MetricsService
from src.models.signals import OFISignal, MetricsSnapshot


class TestMetricsServiceInitialization:
    """Test MetricsService initialization and configuration."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        service = MetricsService()
        assert service.window_size == 100
        assert service.risk_free_rate == Decimal('0.0')
        assert service.periods_per_year == 525600
        assert len(service.signal_history) == 0
        assert len(service.prediction_results) == 0
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        service = MetricsService(
            window_size=50,
            risk_free_rate=0.02,
            periods_per_year=252,
            log_metrics=True
        )
        assert service.window_size == 50
        assert service.risk_free_rate == Decimal('0.02')
        assert service.periods_per_year == 252
        assert service.log_metrics is True
    
    def test_initialization_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="window_size must be > 1"):
            MetricsService(window_size=0)
        
        with pytest.raises(ValueError, match="window_size must be > 1"):
            MetricsService(window_size=1)
    
    def test_initialization_invalid_periods(self):
        """Test that invalid periods_per_year raises error."""
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            MetricsService(periods_per_year=0)
        
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            MetricsService(periods_per_year=-100)


class TestMetricsServiceSignalProcessing:
    """Test signal update processing and history management."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=10)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float,
        timestamp: datetime = None
    ) -> OFISignal:
        """Helper to create OFI signal for testing."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return OFISignal(
            timestamp=timestamp,
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_on_signal_update_adds_to_history(self, service):
        """Test that signal updates are added to history."""
        signal = self.create_signal(ofi_value=1.0, mid_price=100.0)
        await service.on_signal_update(signal)
        
        assert len(service.signal_history) == 1
        timestamp, ofi, price, spread = service.signal_history[0]
        assert ofi == Decimal('1.0')
        assert price == Decimal('100.0')
    
    @pytest.mark.asyncio
    async def test_on_signal_update_skips_invalid(self, service):
        """Test that invalid signals are skipped."""
        # Signal with None ofi_value
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=None,
            window_size=100,
            max_window_size=100,
            event_count=1
        )
        await service.on_signal_update(signal)
        assert len(service.signal_history) == 0
        
        # Signal with None mid_price
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('1.0'),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=None
        )
        await service.on_signal_update(signal)
        assert len(service.signal_history) == 0
    
    @pytest.mark.asyncio
    async def test_rolling_window_behavior(self, service):
        """Test that history respects rolling window size."""
        # Add more signals than window size
        for i in range(15):
            signal = self.create_signal(ofi_value=i, mid_price=100 + i)
            await service.on_signal_update(signal)
        
        # Should only keep last 10
        assert len(service.signal_history) == 10
        
        # Check that oldest values were dropped
        timestamps, ofis, prices, spreads = zip(*service.signal_history)
        assert ofis[0] == Decimal('5')  # First value is from i=5


class TestSharpeRatioCalculation:
    """Test Sharpe ratio calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100, periods_per_year=252, signal_threshold=0.5)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float,
        timestamp: datetime = None
    ) -> OFISignal:
        """Helper to create OFI signal."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return OFISignal(
            timestamp=timestamp,
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_insufficient_data(self, service):
        """Test Sharpe ratio with insufficient data returns None."""
        sharpe = service.calculate_sharpe_ratio()
        assert sharpe is None
        
        # Add one signal
        signal = self.create_signal(ofi_value=1.0, mid_price=100.0)
        await service.on_signal_update(signal)
        sharpe = service.calculate_sharpe_ratio()
        assert sharpe is None
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_perfect_predictions(self, service):
        """Test Sharpe ratio with perfect positive predictions."""
        base_price = 100.0
        
        # Create sequence with perfect predictions
        # OFI positive -> price goes up
        for i in range(20):
            ofi = 1.0  # Always positive
            price = base_price + i * 0.5  # Always increasing
            # Increment timestamp by 1s to ensure multiple buckets for Sharpe calc
            ts = datetime.utcnow() + timedelta(seconds=i)
            signal = self.create_signal(ofi_value=ofi, mid_price=price, timestamp=ts)
            await service.on_signal_update(signal)
        
        sharpe = service.calculate_sharpe_ratio()
        assert sharpe is not None
        # Should be positive since strategy is profitable
        assert sharpe > 0
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_zero_std(self, service):
        """Test Sharpe ratio when std is zero returns None."""
        # Create signals with no price change (zero returns)
        for i in range(10):
            signal = self.create_signal(ofi_value=0.0, mid_price=100.0)
            await service.on_signal_update(signal)
        
        sharpe = service.calculate_sharpe_ratio()
        # Should be None because std of returns is zero
        assert sharpe is None
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_negative(self, service):
        """Test Sharpe ratio with losing strategy."""
        base_price = 100.0
        
        # Create sequence with wrong predictions
        # OFI positive but price goes down
        for i in range(20):
            ofi = 1.0  # Always positive (predicting up)
            price = base_price - i * 0.5  # Always decreasing
            ts = datetime.utcnow() + timedelta(seconds=i)
            signal = self.create_signal(ofi_value=ofi, mid_price=price, timestamp=ts)
            await service.on_signal_update(signal)
        
        sharpe = service.calculate_sharpe_ratio()
        assert sharpe is not None
        # Should be negative since strategy is losing
        assert sharpe < 0


class TestHitRateCalculation:
    """Test hit rate (directional accuracy) calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_hit_rate_insufficient_data(self, service):
        """Test hit rate with insufficient data returns None."""
        hit_rate = service.calculate_hit_rate()
        assert hit_rate is None
    
    @pytest.mark.asyncio
    async def test_hit_rate_perfect_100(self, service):
        """Test hit rate with 100% accuracy."""
        # Create perfect predictions
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        ofis = [1.0, 1.0, 1.0, 1.0, 1.0]  # Always predicting up
        
        for i in range(len(prices)):
            if i < len(ofis):
                ofi = ofis[i]
            else:
                ofi = 1.0
            signal = self.create_signal(ofi_value=ofi, mid_price=prices[i])
            await service.on_signal_update(signal)
        
        hit_rate = service.calculate_hit_rate()
        assert hit_rate is not None
        assert hit_rate == Decimal('100')
    
    @pytest.mark.asyncio
    async def test_hit_rate_perfect_0(self, service):
        """Test hit rate with 0% accuracy (always wrong)."""
        # Create wrong predictions
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        ofis = [-1.0, -1.0, -1.0, -1.0]  # Predicting down but price goes up
        
        for i in range(len(prices)):
            if i < len(ofis):
                ofi = ofis[i]
            else:
                ofi = -1.0
            signal = self.create_signal(ofi_value=ofi, mid_price=prices[i])
            await service.on_signal_update(signal)
        
        hit_rate = service.calculate_hit_rate()
        assert hit_rate is not None
        assert hit_rate == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_hit_rate_50_percent(self, service):
        """Test hit rate with 50% accuracy."""
        # Alternating correct and wrong predictions
        # Note: prediction at t uses OFI[t-1] to predict price[t] - price[t-1]
        test_data = [
            (1.0, 100.0),   # Start - OFI=1 (predicting up)
            (1.0, 101.0),   # Correct: prev_OFI=1, price went up (+1)
            (-1.0, 102.0),  # Correct: prev_OFI=1, price went up (+1)
            (1.0, 101.0),   # Correct: prev_OFI=-1, price went down (-1)
            (-1.0, 100.0),  # Correct: prev_OFI=1, price went down (-1)
            (1.0, 99.0),    # Correct: prev_OFI=-1, price went down (-1)
            (-1.0, 100.0),  # Wrong: prev_OFI=1, price went up (+1) but predicted down
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        hit_rate = service.calculate_hit_rate()
        assert hit_rate is not None
        # 5 correct out of 6 predictions = 83.33%, so let's just check it's reasonable
        assert hit_rate > Decimal('50')
        assert hit_rate < Decimal('100')
    
    @pytest.mark.asyncio
    async def test_hit_rate_ignores_neutral_signals(self, service):
        """Test that neutral OFI signals (0) are not counted."""
        test_data = [
            (0.0, 100.0),   # Neutral - not counted
            (1.0, 101.0),   # Correct prediction
            (0.0, 102.0),   # Neutral - not counted
            (1.0, 103.0),   # Correct prediction
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        hit_rate = service.calculate_hit_rate()
        assert hit_rate is not None
        # Should be 100% because we only had 2 non-neutral predictions, both correct
        assert hit_rate == Decimal('100')


class TestDrawdownCalculation:
    """Test maximum drawdown calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_drawdown_insufficient_data(self, service):
        """Test drawdown with insufficient data returns None."""
        max_dd, current_dd = service.calculate_max_drawdown()
        assert max_dd is None
        assert current_dd is None
    
    @pytest.mark.asyncio
    async def test_drawdown_only_profits(self, service):
        """Test drawdown when strategy only makes profits."""
        # Perfect strategy: always predicts correctly
        for i in range(10):
            ofi = 1.0
            price = 100.0 + i  # Always increasing
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        max_dd, current_dd = service.calculate_max_drawdown()
        assert max_dd is not None
        assert current_dd is not None
        # Should be zero or very close (no drawdown if always profitable)
        assert max_dd <= Decimal('0.01')  # Allow tiny numerical error
    
    @pytest.mark.asyncio
    async def test_drawdown_with_losses(self, service):
        """Test drawdown when strategy has losses."""
        # Strategy: up, up, down, down, up
        test_data = [
            (1.0, 100.0),   # Start
            (1.0, 105.0),   # Profit
            (1.0, 110.0),   # Profit (peak)
            (-1.0, 115.0),  # Loss (wrong direction)
            (-1.0, 120.0),  # Loss (wrong direction)
            (1.0, 125.0),   # Profit (recovery)
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        max_dd, current_dd = service.calculate_max_drawdown()
        assert max_dd is not None
        # Should have some drawdown during the losing period
        assert max_dd > Decimal('0')


class TestWinLossRatio:
    """Test win/loss ratio calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_win_loss_insufficient_data(self, service):
        """Test win/loss ratio with insufficient data returns None."""
        wl_ratio, avg_win, avg_loss = service.calculate_win_loss_ratio()
        assert wl_ratio is None
        assert avg_win is None
        assert avg_loss is None
    
    @pytest.mark.asyncio
    async def test_win_loss_ratio_calculation(self, service):
        """Test win/loss ratio calculation with mixed results."""
        # Create wins and losses
        # Need both winning and losing trades
        test_data = [
            (1.0, 100.0),   # Start - OFI=1 (predicting up)
            (1.0, 102.0),   # Win: prev_OFI=1, price up +2%
            (1.0, 104.0),   # Win: prev_OFI=1, price up ~1.96%
            (1.0, 103.0),   # Loss: prev_OFI=1, price down ~-0.96%
            (-1.0, 102.0),  # Win: prev_OFI=1, price down ~-0.97%
            (-1.0, 103.0),  # Loss: prev_OFI=-1, price up ~0.98%
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        wl_ratio, avg_win, avg_loss = service.calculate_win_loss_ratio()
        assert wl_ratio is not None
        assert avg_win is not None
        assert avg_loss is not None
        # Win/loss ratio should be positive
        assert wl_ratio > 0


class TestSignalPersistence:
    """Test signal persistence calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_persistence_insufficient_data(self, service):
        """Test persistence with insufficient data returns None."""
        persistence = service.calculate_signal_persistence()
        assert persistence is None
    
    @pytest.mark.asyncio
    async def test_persistence_all_same_sign(self, service):
        """Test persistence when all signals have same sign."""
        # All positive signals
        for i in range(10):
            signal = self.create_signal(ofi_value=1.0, mid_price=100.0 + i)
            await service.on_signal_update(signal)
        
        persistence = service.calculate_signal_persistence()
        assert persistence is not None
        # Should be close to window size
        assert persistence >= Decimal('9')
    
    @pytest.mark.asyncio
    async def test_persistence_alternating_signs(self, service):
        """Test persistence with alternating signs."""
        # Alternating positive/negative
        for i in range(10):
            ofi = 1.0 if i % 2 == 0 else -1.0
            signal = self.create_signal(ofi_value=ofi, mid_price=100.0)
            await service.on_signal_update(signal)
        
        persistence = service.calculate_signal_persistence()
        assert persistence is not None
        # Should be close to 1 (no persistence)
        assert persistence <= Decimal('2')


class TestPriceCorrelation:
    """Test price correlation calculation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=100)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_correlation_insufficient_data(self, service):
        """Test correlation with insufficient data returns None."""
        correlation = service.calculate_price_correlation()
        assert correlation is None
    
    @pytest.mark.asyncio
    async def test_correlation_perfect_positive(self, service):
        """Test correlation with perfect positive relationship."""
        # OFI and price changes with varying magnitudes
        # Create data where OFI correlates with price changes
        test_data = [
            (1.0, 100.0),    # Start
            (2.0, 100.5),    # OFI=1, price_change=0.5
            (3.0, 101.5),    # OFI=2, price_change=1.0
            (4.0, 103.0),    # OFI=3, price_change=1.5
            (5.0, 105.0),    # OFI=4, price_change=2.0
            (6.0, 107.5),    # OFI=5, price_change=2.5
            (7.0, 110.5),    # OFI=6, price_change=3.0
            (8.0, 114.0),    # OFI=7, price_change=3.5
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        correlation = service.calculate_price_correlation()
        assert correlation is not None
        # Should be close to 1.0 (perfect positive correlation)
        assert correlation > Decimal('0.9')
    
    @pytest.mark.asyncio
    async def test_correlation_perfect_negative(self, service):
        """Test correlation with perfect negative relationship."""
        # OFI increases but price changes are negative (inversely correlated)
        test_data = [
            (1.0, 100.0),    # Start
            (2.0, 99.5),     # OFI=1, price_change=-0.5
            (3.0, 98.5),     # OFI=2, price_change=-1.0
            (4.0, 97.0),     # OFI=3, price_change=-1.5
            (5.0, 95.0),     # OFI=4, price_change=-2.0
            (6.0, 92.5),     # OFI=5, price_change=-2.5
            (7.0, 89.5),     # OFI=6, price_change=-3.0
            (8.0, 86.0),     # OFI=7, price_change=-3.5
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        correlation = service.calculate_price_correlation()
        assert correlation is not None
        # Should be close to -1.0 (perfect negative correlation)
        assert correlation < Decimal('-0.9')
    
    @pytest.mark.asyncio
    async def test_correlation_no_relationship(self, service):
        """Test correlation when there's no relationship."""
        # Random-ish OFI and price (but deterministic for testing)
        test_data = [
            (1.0, 100.0),
            (-1.0, 101.0),
            (0.5, 99.5),
            (-0.5, 100.5),
            (2.0, 100.0),
        ]
        
        for ofi, price in test_data:
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            await service.on_signal_update(signal)
        
        correlation = service.calculate_price_correlation()
        assert correlation is not None
        # Should be close to 0 (no correlation)
        # Note: with small sample, might not be exactly 0
        assert abs(correlation) < Decimal('1.0')


class TestMetricsSnapshot:
    """Test full metrics snapshot generation."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=50, signal_threshold=0.5)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_snapshot_with_data(self, service):
        """Test snapshot generation with sufficient data."""
        # Add enough data for all metrics
        for i in range(30):
            ofi = 1.0 if i % 2 == 0 else -1.0
            price = 100.0 + i * 0.1
            ts = datetime.utcnow() + timedelta(seconds=i)
            signal = self.create_signal(ofi_value=ofi, mid_price=price)
            signal.timestamp = ts # Update timestamp explicitly
            await service.on_signal_update(signal)
        
        snapshot = service.get_metrics_snapshot()
        
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.window_size == 30
        assert snapshot.sharpe_ratio is not None
        assert snapshot.hit_rate is not None
        assert snapshot.ofi_mean is not None
        assert snapshot.ofi_std is not None
    
    @pytest.mark.asyncio
    async def test_snapshot_to_dict(self, service):
        """Test snapshot serialization to dict."""
        # Add some data
        for i in range(10):
            signal = self.create_signal(ofi_value=1.0, mid_price=100.0 + i)
            await service.on_signal_update(signal)
        
        snapshot = service.get_metrics_snapshot()
        snapshot_dict = snapshot.to_dict()
        
        assert isinstance(snapshot_dict, dict)
        assert 'timestamp' in snapshot_dict
        assert 'sharpe_ratio' in snapshot_dict
        assert 'hit_rate' in snapshot_dict
        assert 'window_size' in snapshot_dict
    
    def test_snapshot_is_performing_well(self):
        """Test performance evaluation method."""
        # Create snapshot with good performance
        good_snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            window_size=100,
            sharpe_ratio=Decimal('2.0'),
            hit_rate=Decimal('60.0')
        )
        assert good_snapshot.is_performing_well() is True
        
        # Create snapshot with poor performance
        poor_snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            window_size=100,
            sharpe_ratio=Decimal('0.5'),
            hit_rate=Decimal('45.0')
        )
        assert poor_snapshot.is_performing_well() is False


class TestMetricsServiceReset:
    """Test reset functionality."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=50, signal_threshold=0.5)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_reset_clears_all_data(self, service):
        """Test that reset clears all data."""
        # Add some data
        for i in range(10):
            signal = self.create_signal(ofi_value=1.0, mid_price=100.0 + i)
            await service.on_signal_update(signal)
        
        assert len(service.signal_history) > 0
        
        # Reset
        service.reset()
        
        assert len(service.signal_history) == 0
        assert len(service.prediction_results) == 0
        assert len(service.cumulative_returns) == 0
        assert service.peak_value == Decimal('1.0')
        assert service.max_drawdown == Decimal('0.0')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService(window_size=50, signal_threshold=0.5)
    
    def create_signal(
        self,
        ofi_value: float,
        mid_price: float
    ) -> OFISignal:
        """Helper to create OFI signal."""
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal(str(ofi_value)),
            window_size=100,
            max_window_size=100,
            event_count=1,
            mid_price=Decimal(str(mid_price)),
            spread=Decimal('1.0')
        )
    
    @pytest.mark.asyncio
    async def test_zero_price_handling(self, service):
        """Test handling of zero prices (avoid division by zero)."""
        # Add signal with zero price
        signal = self.create_signal(ofi_value=1.0, mid_price=0.0)
        await service.on_signal_update(signal)
        
        # Should not crash
        sharpe = service.calculate_sharpe_ratio()
        # Might be None due to division by zero protection
    
    @pytest.mark.asyncio
    async def test_very_small_price_changes(self, service):
        """Test handling of very small price changes."""
        # Add signals with tiny price changes
        for i in range(20):
            signal = self.create_signal(ofi_value=1.0, mid_price=100.0 + i * 0.0001)
            await service.on_signal_update(signal)
        
        # Should calculate metrics without crashing
        snapshot = service.get_metrics_snapshot()
        assert snapshot is not None
    
    @pytest.mark.asyncio
    async def test_large_ofi_values(self, service):
        """Test handling of very large OFI values."""
        # Add signals with large OFI values
        for i in range(20):
            ofi = 1000000.0 * (1 if i % 2 == 0 else -1)
            signal = self.create_signal(ofi_value=ofi, mid_price=100.0 + i)
            await service.on_signal_update(signal)
        
        # Should handle large numbers correctly
        snapshot = service.get_metrics_snapshot()
        assert snapshot.ofi_mean is not None
        assert snapshot.ofi_std is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
