"""
OFI Calculator Service - Application Layer

This service calculates the Order Flow Imbalance (OFI) metric by subscribing to
BookManager updates and computing signed volume deltas based on BBO state changes.

OFI measures the net signed volume pushing the market by tracking how liquidity
at the best bid and ask levels changes over time. The formula is:

    e_n = delta_bid - delta_ask
    OFI = sum(e_n over window)

Where:
    - delta_bid: Signed change in bid-side liquidity
    - delta_ask: Signed change in ask-side liquidity
    - window: Rolling time window (implemented as deque with maxlen)

The delta calculation follows these rules:

BID SIDE:
    - If P_bid_new > P_bid_old: delta = q_bid_new (price improvement)
    - If P_bid_new == P_bid_old: delta = q_bid_new - q_bid_old (size change)
    - If P_bid_new < P_bid_old: delta = -q_bid_old (level removed)

ASK SIDE:
    - If P_ask_new < P_ask_old: delta = q_ask_new (price improvement)
    - If P_ask_new == P_ask_old: delta = q_ask_new - q_ask_old (size change)
    - If P_ask_new > P_ask_old: delta = -q_ask_old (level removed)

Key Insights:
    - Positive OFI: Net buying pressure (bids added/asks removed)
    - Negative OFI: Net selling pressure (asks added/bids removed)
    - OFI often leads price changes in high-frequency markets
"""

import logging
from collections import deque
from decimal import Decimal
from typing import Optional, Callable, Awaitable, List
from datetime import datetime
import statistics

from ..models.signals import OFIEvent, OFISignal, OFIStatistics
from .book_manager import BBOState


# Configure logging
logger = logging.getLogger(__name__)


class OFICalculator:
    """
    Order Flow Imbalance calculator.
    
    This service subscribes to BookManager updates via the observer pattern and
    calculates OFI events and aggregated signals.
    
    Architecture:
        - Subscribes to BookManager via manager.subscribe(calculator.on_book_update)
        - Receives BBOState before/after each update
        - Calculates delta_bid and delta_ask using OFI formula
        - Maintains rolling window of events using deque
        - Computes aggregated OFI signal and statistics
    
    Attributes:
        window_size: Maximum number of events to retain in rolling window
        events: Deque of OFI events (bounded by window_size)
        stats: Statistical summary of calculator performance
        _initialized: Whether the calculator has received its first valid update
        _start_time: When the calculator was initialized (for uptime tracking)
    """
    
    def __init__(
        self,
        window_size: int = 100,
        log_events: bool = False
    ):
        """
        Initialize OFI calculator.
        
        Args:
            window_size: Number of events to retain in rolling window (default: 100)
            log_events: Whether to log each OFI event (verbose, default: False)
        
        Raises:
            ValueError: If window_size <= 0
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        
        self.window_size = window_size
        self.log_events = log_events
        
        # Rolling window of OFI events (bounded FIFO queue)
        self.events: deque[OFIEvent] = deque(maxlen=window_size)
        
        # Statistics tracking
        self.stats = OFIStatistics(max_window_size=window_size)
        
        # Internal state
        self._initialized = False
        self._start_time = datetime.utcnow()
        
        logger.debug(f"OFICalculator initialized with window_size={window_size}")
    
    async def on_book_update(self, event: dict) -> None:
        """
        Callback for BookManager updates (observer pattern).
        
        This method is registered as a subscriber to BookManager and receives
        notifications whenever the order book is updated.
        
        Args:
            event: Dictionary containing:
                - 'previous_bbo': BBOState dict or object before the update
                - 'current_bbo': BBOState dict or object after the update
                - 'timestamp': When the update occurred
                - 'sequence': Sequence number (optional)
        """
        self.stats.total_updates_received += 1
        
        # Extract BBO states (they come as dicts from BookManager, or objects from tests)
        prev_bbo_raw = event.get('previous_bbo')
        curr_bbo_raw = event.get('current_bbo')
        timestamp: datetime = event.get('timestamp', datetime.utcnow())
        sequence: Optional[int] = event.get('sequence')
        
        # Convert to BBOState objects if needed
        prev_bbo = self._ensure_bbo_state(prev_bbo_raw) if prev_bbo_raw else None
        curr_bbo = self._ensure_bbo_state(curr_bbo_raw) if curr_bbo_raw else None
        
        # Validate inputs
        if not self._validate_bbo_states(prev_bbo, curr_bbo):
            self.stats.events_skipped += 1
            return
        
        # Calculate OFI event
        ofi_event = self._calculate_ofi_event(prev_bbo, curr_bbo, timestamp, sequence)
        
        # Add to rolling window
        self.events.append(ofi_event)
        
        # Update statistics
        self.stats.total_events_processed += 1
        self.stats.current_window_size = len(self.events)
        self.stats.last_update_timestamp = timestamp
        
        if self.stats.first_event_timestamp is None:
            self.stats.first_event_timestamp = timestamp
        
        # Calculate uptime
        self.stats.uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
        
        # Mark as initialized after first successful event
        if not self._initialized:
            self._initialized = True
            logger.debug("OFICalculator received first valid update")
        
        # Optional logging
        if self.log_events:
            logger.debug(
                f"OFI Event: e_n={float(ofi_event.event_value):.4f}, "
                f"delta_bid={float(ofi_event.delta_bid):.4f}, "
                f"delta_ask={float(ofi_event.delta_ask):.4f}"
            )
    
    def _ensure_bbo_state(self, data) -> BBOState:
        """
        Ensure data is a BBOState object.
        
        Args:
            data: Either a BBOState object or a dictionary
        
        Returns:
            BBOState object
        """
        # If already a BBOState, return as-is
        if isinstance(data, BBOState):
            return data
        
        # Otherwise, convert from dict
        return self._bbo_from_dict(data)
    
    def _bbo_from_dict(self, data: dict) -> BBOState:
        """
        Create BBOState from dictionary.
        
        Args:
            data: Dictionary representation of BBOState
        
        Returns:
            BBOState object
        """
        return BBOState(
            best_bid_price=Decimal(str(data['best_bid_price'])) if data.get('best_bid_price') else None,
            best_bid_size=Decimal(str(data['best_bid_size'])) if data.get('best_bid_size') else None,
            best_ask_price=Decimal(str(data['best_ask_price'])) if data.get('best_ask_price') else None,
            best_ask_size=Decimal(str(data['best_ask_size'])) if data.get('best_ask_size') else None,
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data.get('timestamp'), str) else data.get('timestamp'),
            sequence=data.get('sequence')
        )
    
    def _validate_bbo_states(
        self,
        prev_bbo: Optional[BBOState],
        curr_bbo: Optional[BBOState]
    ) -> bool:
        """
        Validate that BBO states are suitable for OFI calculation.
        
        Args:
            prev_bbo: Previous BBO state
            curr_bbo: Current BBO state
        
        Returns:
            True if both states are valid and can be compared, False otherwise
        """
        # First update has no previous state - skip it
        if prev_bbo is None:
            logger.debug("Skipping first update (no previous BBO)")
            return False
        
        # Current state must exist
        if curr_bbo is None:
            logger.warning("Current BBO is None, skipping")
            return False
        
        # Both states must be valid (have bid and ask)
        if not prev_bbo.is_valid():
            logger.debug("Previous BBO invalid (missing bid or ask)")
            return False
        
        if not curr_bbo.is_valid():
            logger.debug("Current BBO invalid (missing bid or ask)")
            return False
        
        return True
    
    def _calculate_ofi_event(
        self,
        prev_bbo: BBOState,
        curr_bbo: BBOState,
        timestamp: datetime,
        sequence: Optional[int]
    ) -> OFIEvent:
        """
        Calculate OFI event from BBO state transition.
        
        This implements the core OFI formula:
            e_n = delta_bid - delta_ask
        
        Args:
            prev_bbo: Previous BBO state (before update)
            curr_bbo: Current BBO state (after update)
            timestamp: When the update occurred
            sequence: Sequence number (optional)
        
        Returns:
            OFIEvent with calculated deltas and event value
        """
        # Calculate bid-side delta
        delta_bid = self._calculate_bid_delta(prev_bbo, curr_bbo)
        
        # Calculate ask-side delta
        delta_ask = self._calculate_ask_delta(prev_bbo, curr_bbo)
        
        # Event contribution: e_n = delta_bid - delta_ask
        event_value = delta_bid - delta_ask
        
        # Calculate current market state
        mid_price = (curr_bbo.best_bid_price + curr_bbo.best_ask_price) / Decimal('2')
        spread = curr_bbo.best_ask_price - curr_bbo.best_bid_price
        
        return OFIEvent(
            timestamp=timestamp,
            sequence=sequence,
            delta_bid=delta_bid,
            delta_ask=delta_ask,
            event_value=event_value,
            prev_bid_price=prev_bbo.best_bid_price,
            prev_bid_size=prev_bbo.best_bid_size,
            prev_ask_price=prev_bbo.best_ask_price,
            prev_ask_size=prev_bbo.best_ask_size,
            curr_bid_price=curr_bbo.best_bid_price,
            curr_bid_size=curr_bbo.best_bid_size,
            curr_ask_price=curr_bbo.best_ask_price,
            curr_ask_size=curr_bbo.best_ask_size,
            mid_price=mid_price,
            spread=spread
        )
    
    def _calculate_bid_delta(self, prev_bbo: BBOState, curr_bbo: BBOState) -> Decimal:
        """
        Calculate signed change in bid-side liquidity.
        
        Formula:
            - If P_bid_new > P_bid_old: delta = q_bid_new (price improvement)
            - If P_bid_new == P_bid_old: delta = q_bid_new - q_bid_old (size change)
            - If P_bid_new < P_bid_old: delta = -q_bid_old (level removed/downgrade)
        
        Interpretation:
            - Positive delta: Buying pressure increased (higher price or more size)
            - Negative delta: Buying pressure decreased (lower price or less size)
        
        Args:
            prev_bbo: Previous BBO state
            curr_bbo: Current BBO state
        
        Returns:
            Signed bid-side delta
        """
        prev_price = prev_bbo.best_bid_price
        prev_size = prev_bbo.best_bid_size
        curr_price = curr_bbo.best_bid_price
        curr_size = curr_bbo.best_bid_size
        
        if curr_price > prev_price:
            # Price improved (moved up) - new aggressive buying
            # Full new size counts as positive pressure
            delta = curr_size
        elif curr_price == prev_price:
            # Price stayed same - size changed
            # Delta is the change in size
            delta = curr_size - prev_size
        else:  # curr_price < prev_price
            # Price degraded (moved down) - liquidity withdrawn
            # Old size counts as negative pressure
            delta = -prev_size
        
        return delta
    
    def _calculate_ask_delta(self, prev_bbo: BBOState, curr_bbo: BBOState) -> Decimal:
        """
        Calculate signed change in ask-side liquidity.
        
        Formula:
            - If P_ask_new < P_ask_old: delta = q_ask_new (price improvement)
            - If P_ask_new == P_ask_old: delta = q_ask_new - q_ask_old (size change)
            - If P_ask_new > P_ask_old: delta = -q_ask_old (level removed/downgrade)
        
        Interpretation:
            - Positive delta: Selling pressure increased (lower price or more size)
            - Negative delta: Selling pressure decreased (higher price or less size)
        
        Note: For asks, price improvement means price going DOWN (cheaper to buy).
        
        Args:
            prev_bbo: Previous BBO state
            curr_bbo: Current BBO state
        
        Returns:
            Signed ask-side delta
        """
        prev_price = prev_bbo.best_ask_price
        prev_size = prev_bbo.best_ask_size
        curr_price = curr_bbo.best_ask_price
        curr_size = curr_bbo.best_ask_size
        
        if curr_price < prev_price:
            # Price improved (moved down) - new aggressive selling
            # Full new size counts as positive pressure
            delta = curr_size
        elif curr_price == prev_price:
            # Price stayed same - size changed
            # Delta is the change in size
            delta = curr_size - prev_size
        else:  # curr_price > prev_price
            # Price degraded (moved up) - liquidity withdrawn
            # Old size counts as negative pressure
            delta = -prev_size
        
        return delta
    
    def get_current_signal(self) -> Optional[OFISignal]:
        """
        Get the current OFI signal aggregated over the window.
        
        Returns:
            OFISignal with aggregated metrics, or None if no events processed
        """
        if not self.events:
            return None
        
        # Aggregate OFI: sum of all e_n in window
        ofi_value = sum(event.event_value for event in self.events)
        
        # Get current market state from most recent event
        latest = self.events[-1]
        
        # Calculate statistics over the window
        event_values = [float(event.event_value) for event in self.events]
        mean_event = Decimal(str(statistics.mean(event_values)))
        std_event = Decimal(str(statistics.stdev(event_values))) if len(event_values) > 1 else Decimal('0')
        min_event = Decimal(str(min(event_values)))
        max_event = Decimal(str(max(event_values)))
        
        return OFISignal(
            timestamp=datetime.utcnow(),
            ofi_value=ofi_value,
            window_size=len(self.events),
            max_window_size=self.window_size,
            event_count=self.stats.total_events_processed,
            mid_price=latest.mid_price,
            spread=latest.spread,
            micro_price=self._calculate_micro_price(latest),
            mean_event_value=mean_event,
            std_event_value=std_event,
            min_event_value=min_event,
            max_event_value=max_event
        )
    
    def _calculate_micro_price(self, event: OFIEvent) -> Optional[Decimal]:
        """
        Calculate volume-weighted micro-price.
        
        Formula:
            P_micro = (q_bid * P_ask + q_ask * P_bid) / (q_bid + q_ask)
        
        The micro-price is a more accurate representation of the "true" price
        than the simple mid-price, as it weights by available liquidity.
        
        Args:
            event: OFI event with current BBO state
        
        Returns:
            Micro-price, or None if cannot be calculated
        """
        if event.curr_bid_price is None or event.curr_ask_price is None:
            return None
        if event.curr_bid_size is None or event.curr_ask_size is None:
            return None
        
        bid_size = event.curr_bid_size
        ask_size = event.curr_ask_size
        bid_price = event.curr_bid_price
        ask_price = event.curr_ask_price
        
        total_size = bid_size + ask_size
        if total_size == 0:
            return None
        
        # Micro-price formula
        micro_price = (bid_size * ask_price + ask_size * bid_price) / total_size
        
        return micro_price
    
    def get_statistics(self) -> OFIStatistics:
        """
        Get current statistics for monitoring.
        
        Returns:
            OFIStatistics with current operational metrics
        """
        return self.stats
    
    def get_recent_events(self, n: int = 10) -> List[OFIEvent]:
        """
        Get the N most recent OFI events.
        
        Args:
            n: Number of recent events to return (default: 10)
        
        Returns:
            List of recent OFI events (most recent last)
        """
        # Convert deque to list and get last n
        all_events = list(self.events)
        return all_events[-n:] if len(all_events) >= n else all_events
    
    def reset(self) -> None:
        """
        Reset the calculator state.
        
        This clears all events and statistics, useful for restarting or testing.
        """
        self.events.clear()
        self.stats = OFIStatistics(max_window_size=self.window_size)
        self._initialized = False
        self._start_time = datetime.utcnow()
        logger.debug("OFICalculator reset")
    
    def is_initialized(self) -> bool:
        """
        Check if calculator has received at least one valid update.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        signal = self.get_current_signal()
        if signal:
            return (
                f"OFICalculator(window_size={self.window_size}, "
                f"current_ofi={float(signal.ofi_value):.2f}, "
                f"events={len(self.events)}/{self.window_size})"
            )
        return f"OFICalculator(window_size={self.window_size}, events=0/{self.window_size})"
