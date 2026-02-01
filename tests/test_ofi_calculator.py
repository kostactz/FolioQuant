"""
Unit Tests for OFI Calculator - Phase 4

This module contains comprehensive tests for the OFICalculator service,
validating the mathematical correctness of the OFI formula and the
robustness of the implementation.

Test Coverage:
1. OFI event models (OFIEvent, OFISignal, OFIStatistics)
2. OFI calculator initialization and configuration
3. Bid/ask delta calculation (core OFI logic)
4. Event aggregation and rolling window
5. Observer pattern integration
6. Statistical calculations
7. Edge cases and error handling
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from collections import deque

from src.models.signals import OFIEvent, OFISignal, OFIStatistics
from src.services.ofi_calculator import OFICalculator
from src.services.book_manager import BBOState


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_bbo_state():
    """Create a sample BBOState for testing."""
    return BBOState(
        best_bid_price=Decimal('50000.00'),
        best_bid_size=Decimal('1.5'),
        best_ask_price=Decimal('50001.00'),
        best_ask_size=Decimal('2.0'),
        timestamp=datetime.utcnow(),
        sequence=100
    )


@pytest.fixture
def calculator():
    """Create an OFI calculator with small window for testing."""
    return OFICalculator(window_size=5, log_events=False)


# ============================================================================
# OFIEvent Model Tests
# ============================================================================

class TestOFIEvent:
    """Test OFIEvent data model."""
    
    def test_ofi_event_creation(self):
        """Test creating an OFI event."""
        event = OFIEvent(
            timestamp=datetime.utcnow(),
            sequence=100,
            delta_bid=Decimal('1.0'),
            delta_ask=Decimal('0.5'),
            event_value=Decimal('0.5'),  # delta_bid - delta_ask
            prev_bid_price=Decimal('50000'),
            prev_bid_size=Decimal('1.0'),
            curr_bid_price=Decimal('50000'),
            curr_bid_size=Decimal('2.0'),
            mid_price=Decimal('50000.5'),
            spread=Decimal('1.0')
        )
        
        assert event.event_value == Decimal('0.5')
        assert event.delta_bid == Decimal('1.0')
        assert event.delta_ask == Decimal('0.5')
    
    def test_ofi_event_to_dict(self):
        """Test OFI event serialization."""
        event = OFIEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            sequence=100,
            delta_bid=Decimal('1.0'),
            delta_ask=Decimal('0.5'),
            event_value=Decimal('0.5')
        )
        
        d = event.to_dict()
        assert d['sequence'] == 100
        assert d['delta_bid'] == 1.0
        assert d['delta_ask'] == 0.5
        assert d['event_value'] == 0.5


# ============================================================================
# OFISignal Model Tests
# ============================================================================

class TestOFISignal:
    """Test OFISignal data model."""
    
    def test_ofi_signal_creation(self):
        """Test creating an OFI signal."""
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('10.5'),
            window_size=50,
            max_window_size=100,
            event_count=150,
            mid_price=Decimal('50000'),
            spread=Decimal('1.0')
        )
        
        assert signal.ofi_value == Decimal('10.5')
        assert signal.window_size == 50
        assert signal.max_window_size == 100
    
    def test_ofi_signal_bullish(self):
        """Test bullish signal detection."""
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('5.0'),
            window_size=10,
            max_window_size=100,
            event_count=10
        )
        
        assert signal.is_bullish() is True
        assert signal.is_bearish() is False
        assert signal.is_neutral() is False
    
    def test_ofi_signal_bearish(self):
        """Test bearish signal detection."""
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('-5.0'),
            window_size=10,
            max_window_size=100,
            event_count=10
        )
        
        assert signal.is_bullish() is False
        assert signal.is_bearish() is True
        assert signal.is_neutral() is False
    
    def test_ofi_signal_neutral(self):
        """Test neutral signal detection."""
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('0'),
            window_size=10,
            max_window_size=100,
            event_count=10
        )
        
        assert signal.is_neutral() is True
        assert signal.is_bullish() is False
        assert signal.is_bearish() is False
    
    def test_ofi_signal_strength(self):
        """Test signal strength classification."""
        # Neutral
        signal = OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=Decimal('0'),
            window_size=10,
            max_window_size=100,
            event_count=10
        )
        assert signal.signal_strength() == 'neutral'
        
        # Weak
        signal.ofi_value = Decimal('0.5')
        assert signal.signal_strength() == 'weak'
        
        # Moderate
        signal.ofi_value = Decimal('5.0')
        assert signal.signal_strength() == 'moderate'
        
        # Strong
        signal.ofi_value = Decimal('15.0')
        assert signal.signal_strength() == 'strong'


# ============================================================================
# OFIStatistics Model Tests
# ============================================================================

class TestOFIStatistics:
    """Test OFIStatistics data model."""
    
    def test_statistics_creation(self):
        """Test creating statistics object."""
        stats = OFIStatistics(
            total_events_processed=100,
            total_updates_received=110,
            events_skipped=10,
            max_window_size=50
        )
        
        assert stats.total_events_processed == 100
        assert stats.events_skipped == 10
    
    def test_processing_rate(self):
        """Test processing rate calculation."""
        stats = OFIStatistics(
            total_events_processed=100,
            uptime_seconds=10.0
        )
        
        assert stats.processing_rate() == 10.0  # 100 events / 10 seconds
    
    def test_skip_rate(self):
        """Test skip rate calculation."""
        stats = OFIStatistics(
            total_updates_received=100,
            events_skipped=10
        )
        
        assert stats.skip_rate() == 10.0  # 10%


# ============================================================================
# OFICalculator Initialization Tests
# ============================================================================

class TestOFICalculatorInitialization:
    """Test OFI calculator initialization."""
    
    def test_calculator_creation(self):
        """Test creating a calculator."""
        calc = OFICalculator(window_size=100)
        
        assert calc.window_size == 100
        assert len(calc.events) == 0
        assert calc.is_initialized() is False
        assert calc.stats.max_window_size == 100
    
    def test_calculator_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            OFICalculator(window_size=0)
        
        with pytest.raises(ValueError, match="window_size must be positive"):
            OFICalculator(window_size=-10)
    
    def test_calculator_repr(self):
        """Test string representation."""
        calc = OFICalculator(window_size=10)
        repr_str = repr(calc)
        
        assert "OFICalculator" in repr_str
        assert "window_size=10" in repr_str


# ============================================================================
# BBO Validation Tests
# ============================================================================

class TestBBOValidation:
    """Test BBO state validation logic."""
    
    @pytest.mark.asyncio
    async def test_skip_first_update_no_previous(self, calculator):
        """Test that first update is skipped (no previous BBO)."""
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        await calculator.on_book_update({
            'previous_bbo': None,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 1
        })
        
        # Should be skipped
        assert len(calculator.events) == 0
        assert calculator.stats.events_skipped == 1
        assert calculator.is_initialized() is False
    
    @pytest.mark.asyncio
    async def test_skip_invalid_previous_bbo(self, calculator):
        """Test skipping when previous BBO is invalid."""
        # Previous BBO missing ask
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=None,  # Invalid
            best_ask_size=None,
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        # Should be skipped
        assert len(calculator.events) == 0
        assert calculator.stats.events_skipped == 1


# ============================================================================
# Bid Delta Calculation Tests (Core OFI Logic)
# ============================================================================

class TestBidDeltaCalculation:
    """Test bid-side delta calculation logic."""
    
    def test_bid_price_improvement(self, calculator):
        """Test bid delta when price improves (moves up)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50001'),  # Price up
            best_bid_size=Decimal('2.0'),
            best_ask_price=Decimal('50002'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_bid_delta(prev_bbo, curr_bbo)
        
        # Price improved: delta = new size
        assert delta == Decimal('2.0')
    
    def test_bid_size_increase(self, calculator):
        """Test bid delta when size increases at same price."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),  # Same price
            best_bid_size=Decimal('3.0'),  # Size up
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_bid_delta(prev_bbo, curr_bbo)
        
        # Same price: delta = size change
        assert delta == Decimal('2.0')  # 3.0 - 1.0
    
    def test_bid_size_decrease(self, calculator):
        """Test bid delta when size decreases at same price."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('3.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),  # Same price
            best_bid_size=Decimal('1.0'),  # Size down
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_bid_delta(prev_bbo, curr_bbo)
        
        # Same price: delta = size change (negative)
        assert delta == Decimal('-2.0')  # 1.0 - 3.0
    
    def test_bid_price_degradation(self, calculator):
        """Test bid delta when price degrades (moves down)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.5'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('49999'),  # Price down
            best_bid_size=Decimal('2.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_bid_delta(prev_bbo, curr_bbo)
        
        # Price degraded: delta = -old size
        assert delta == Decimal('-1.5')


# ============================================================================
# Ask Delta Calculation Tests (Core OFI Logic)
# ============================================================================

class TestAskDeltaCalculation:
    """Test ask-side delta calculation logic."""
    
    def test_ask_price_improvement(self, calculator):
        """Test ask delta when price improves (moves down)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50002'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),  # Price down (improvement)
            best_ask_size=Decimal('2.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_ask_delta(prev_bbo, curr_bbo)
        
        # Price improved: delta = new size
        assert delta == Decimal('2.0')
    
    def test_ask_size_increase(self, calculator):
        """Test ask delta when size increases at same price."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),  # Same price
            best_ask_size=Decimal('3.0'),  # Size up
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_ask_delta(prev_bbo, curr_bbo)
        
        # Same price: delta = size change
        assert delta == Decimal('2.0')  # 3.0 - 1.0
    
    def test_ask_size_decrease(self, calculator):
        """Test ask delta when size decreases at same price."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('3.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),  # Same price
            best_ask_size=Decimal('1.0'),  # Size down
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_ask_delta(prev_bbo, curr_bbo)
        
        # Same price: delta = size change (negative)
        assert delta == Decimal('-2.0')  # 1.0 - 3.0
    
    def test_ask_price_degradation(self, calculator):
        """Test ask delta when price degrades (moves up)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.5'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50002'),  # Price up (degradation)
            best_ask_size=Decimal('2.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        delta = calculator._calculate_ask_delta(prev_bbo, curr_bbo)
        
        # Price degraded: delta = -old size
        assert delta == Decimal('-1.5')


# ============================================================================
# OFI Event Calculation Tests
# ============================================================================

class TestOFIEventCalculation:
    """Test complete OFI event calculation (e_n = delta_bid - delta_ask)."""
    
    @pytest.mark.asyncio
    async def test_bullish_event_bid_improvement(self, calculator):
        """Test bullish OFI event (bid price improves)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50001'),  # Bid up
            best_bid_size=Decimal('2.0'),
            best_ask_price=Decimal('50001'),  # Ask same
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        assert len(calculator.events) == 1
        event = calculator.events[0]
        
        # delta_bid = 2.0 (price improved)
        # delta_ask = 0.0 (same price, same size)
        # e_n = 2.0 - 0.0 = 2.0 (bullish)
        assert event.delta_bid == Decimal('2.0')
        assert event.delta_ask == Decimal('0.0')
        assert event.event_value == Decimal('2.0')
    
    @pytest.mark.asyncio
    async def test_bearish_event_ask_improvement(self, calculator):
        """Test bearish OFI event (ask price improves)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50002'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),  # Bid same
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),  # Ask down (improvement)
            best_ask_size=Decimal('2.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        assert len(calculator.events) == 1
        event = calculator.events[0]
        
        # delta_bid = 0.0 (same price, same size)
        # delta_ask = 2.0 (price improved)
        # e_n = 0.0 - 2.0 = -2.0 (bearish)
        assert event.delta_bid == Decimal('0.0')
        assert event.delta_ask == Decimal('2.0')
        assert event.event_value == Decimal('-2.0')
    
    @pytest.mark.asyncio
    async def test_neutral_event_symmetric_changes(self, calculator):
        """Test neutral OFI event (symmetric bid/ask changes)."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('2.0'),  # Bid size up
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('2.0'),  # Ask size up
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        assert len(calculator.events) == 1
        event = calculator.events[0]
        
        # delta_bid = 1.0 (size increased)
        # delta_ask = 1.0 (size increased)
        # e_n = 1.0 - 1.0 = 0.0 (neutral)
        assert event.delta_bid == Decimal('1.0')
        assert event.delta_ask == Decimal('1.0')
        assert event.event_value == Decimal('0.0')


# ============================================================================
# Rolling Window Tests
# ============================================================================

class TestRollingWindow:
    """Test rolling window behavior."""
    
    @pytest.mark.asyncio
    async def test_window_fills_up(self):
        """Test that window fills to capacity."""
        calc = OFICalculator(window_size=3)
        
        # Create 5 updates (window size is 3)
        for i in range(5):
            prev_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('1.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i
            )
            
            curr_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('2.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i + 1
            )
            
            await calc.on_book_update({
                'previous_bbo': prev_bbo,
                'current_bbo': curr_bbo,
                'timestamp': datetime.utcnow(),
                'sequence': i + 1
            })
        
        # Window should be at capacity (3)
        assert len(calc.events) == 3
        assert calc.stats.total_events_processed == 5
    
    @pytest.mark.asyncio
    async def test_window_oldest_evicted(self):
        """Test that oldest events are evicted when window is full."""
        calc = OFICalculator(window_size=2)
        
        # Add 3 events
        for seq in [1, 2, 3]:
            prev_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('1.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=seq
            )
            
            curr_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('2.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=seq + 1
            )
            
            await calc.on_book_update({
                'previous_bbo': prev_bbo,
                'current_bbo': curr_bbo,
                'timestamp': datetime.utcnow(),
                'sequence': seq + 1
            })
        
        # Only last 2 should remain
        assert len(calc.events) == 2
        assert calc.events[0].sequence == 3  # Sequence 2 was evicted
        assert calc.events[1].sequence == 4


# ============================================================================
# OFI Signal Aggregation Tests
# ============================================================================

class TestOFISignalAggregation:
    """Test OFI signal calculation from events."""
    
    @pytest.mark.asyncio
    async def test_ofi_signal_aggregation(self):
        """Test that OFI is sum of event values."""
        calc = OFICalculator(window_size=10)
        
        # Create events with known values
        event_values = [Decimal('1.0'), Decimal('2.0'), Decimal('-0.5'), Decimal('1.5')]
        
        for i, value in enumerate(event_values):
            # Create update that produces this event value
            # e_n = delta_bid - delta_ask
            # Let's make delta_bid = value, delta_ask = 0
            prev_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('1.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i
            )
            
            # Make bid size increase by 'value' amount
            curr_bbo = BBOState(
                best_bid_price=Decimal('50000'),  # Same price
                best_bid_size=Decimal('1.0') + value,  # Size change
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),  # No change
                timestamp=datetime.utcnow(),
                sequence=i + 1
            )
            
            await calc.on_book_update({
                'previous_bbo': prev_bbo,
                'current_bbo': curr_bbo,
                'timestamp': datetime.utcnow(),
                'sequence': i + 1
            })
        
        # Get signal
        signal = calc.get_current_signal()
        
        assert signal is not None
        # OFI should be sum of event values
        expected_ofi = sum(event_values)
        assert signal.ofi_value == expected_ofi
        assert signal.window_size == 4
    
    @pytest.mark.asyncio
    async def test_signal_before_any_events(self, calculator):
        """Test that signal is None before any events."""
        signal = calculator.get_current_signal()
        assert signal is None


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Test statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_statistics_update(self):
        """Test that statistics are updated correctly."""
        calc = OFICalculator(window_size=5)
        
        # Process 3 valid updates
        for i in range(3):
            prev_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('1.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i
            )
            
            curr_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('2.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i + 1
            )
            
            await calc.on_book_update({
                'previous_bbo': prev_bbo,
                'current_bbo': curr_bbo,
                'timestamp': datetime.utcnow(),
                'sequence': i + 1
            })
        
        stats = calc.get_statistics()
        assert stats.total_events_processed == 3
        assert stats.total_updates_received == 3
        assert stats.current_window_size == 3
        assert stats.events_skipped == 0


# ============================================================================
# Utility Method Tests
# ============================================================================

class TestUtilityMethods:
    """Test utility methods."""
    
    @pytest.mark.asyncio
    async def test_get_recent_events(self):
        """Test getting recent events."""
        calc = OFICalculator(window_size=10)
        
        # Add 5 events
        for i in range(5):
            prev_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('1.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i
            )
            
            curr_bbo = BBOState(
                best_bid_price=Decimal('50000'),
                best_bid_size=Decimal('2.0'),
                best_ask_price=Decimal('50001'),
                best_ask_size=Decimal('1.0'),
                timestamp=datetime.utcnow(),
                sequence=i + 1
            )
            
            await calc.on_book_update({
                'previous_bbo': prev_bbo,
                'current_bbo': curr_bbo,
                'timestamp': datetime.utcnow(),
                'sequence': i + 1
            })
        
        # Get last 3 events
        recent = calc.get_recent_events(n=3)
        assert len(recent) == 3
        assert recent[0].sequence == 3
        assert recent[2].sequence == 5
    
    def test_reset(self, calculator):
        """Test resetting calculator."""
        # Manually add some state
        calculator.stats.total_events_processed = 10
        calculator._initialized = True
        
        # Reset
        calculator.reset()
        
        assert len(calculator.events) == 0
        assert calculator.stats.total_events_processed == 0
        assert calculator.is_initialized() is False


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_missing_current_bbo(self, calculator):
        """Test handling when current BBO is missing."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': None,  # Missing
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        # Should be skipped
        assert len(calculator.events) == 0
        assert calculator.stats.events_skipped == 1
    
    @pytest.mark.asyncio
    async def test_micro_price_calculation(self, calculator):
        """Test micro-price calculation in signal."""
        prev_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('1.0'),
            best_ask_price=Decimal('50001'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=1
        )
        
        curr_bbo = BBOState(
            best_bid_price=Decimal('50000'),
            best_bid_size=Decimal('2.0'),  # More bid size
            best_ask_price=Decimal('50002'),
            best_ask_size=Decimal('1.0'),
            timestamp=datetime.utcnow(),
            sequence=2
        )
        
        await calculator.on_book_update({
            'previous_bbo': prev_bbo,
            'current_bbo': curr_bbo,
            'timestamp': datetime.utcnow(),
            'sequence': 2
        })
        
        signal = calculator.get_current_signal()
        assert signal is not None
        
        # Micro-price = (q_bid * P_ask + q_ask * P_bid) / (q_bid + q_ask)
        # = (2.0 * 50002 + 1.0 * 50000) / (2.0 + 1.0)
        # = (100004 + 50000) / 3
        # = 150004 / 3
        # = 50001.333...
        expected_micro = (Decimal('2.0') * Decimal('50002') + Decimal('1.0') * Decimal('50000')) / Decimal('3.0')
        assert signal.micro_price == expected_micro
