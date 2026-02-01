"""
Signal Models - Domain Layer

This module defines data structures for trading signals, specifically the Order Flow Imbalance (OFI)
metric. These models are part of the domain layer and represent the core quantitative concepts.

OFI measures the net signed volume pushing the market in a particular direction by tracking
changes in the best bid and ask prices and their associated quantities.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, List, Tuple
from datetime import datetime


@dataclass
class OFIEvent:
    """
    Represents a single Order Flow Imbalance event.
    
    An OFI event captures the contribution to the overall imbalance from a single
    order book update. The event value (e_n) is calculated as:
    
        e_n = delta_bid - delta_ask
    
    Where delta_bid and delta_ask are computed based on price and size changes
    at the best bid and ask levels.
    
    Attributes:
        timestamp: When this event occurred (UTC)
        sequence: Sequence number of the book update that generated this event
        delta_bid: Signed volume change on the bid side
        delta_ask: Signed volume change on the ask side
        event_value: e_n = delta_bid - delta_ask (contribution to OFI)
        prev_bid_price: Previous best bid price (for debugging)
        prev_bid_size: Previous best bid size
        prev_ask_price: Previous best ask price
        prev_ask_size: Previous best ask size
        curr_bid_price: Current best bid price
        curr_bid_size: Current best bid size
        curr_ask_price: Current best ask price
        curr_ask_size: Current best ask size
        mid_price: Mid-price at the time of this event
        spread: Bid-ask spread at the time of this event
    """
    timestamp: datetime
    sequence: Optional[int]
    delta_bid: Decimal
    delta_ask: Decimal
    event_value: Decimal  # e_n = delta_bid - delta_ask
    
    # Previous state (for transparency and debugging)
    prev_bid_price: Optional[Decimal] = None
    prev_bid_size: Optional[Decimal] = None
    prev_ask_price: Optional[Decimal] = None
    prev_ask_size: Optional[Decimal] = None
    
    # Current state
    curr_bid_price: Optional[Decimal] = None
    curr_bid_size: Optional[Decimal] = None
    curr_ask_price: Optional[Decimal] = None
    curr_ask_size: Optional[Decimal] = None
    
    # Market state
    mid_price: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'sequence': self.sequence,
            'delta_bid': float(self.delta_bid) if self.delta_bid else None,
            'delta_ask': float(self.delta_ask) if self.delta_ask else None,
            'event_value': float(self.event_value),
            'prev_bid_price': float(self.prev_bid_price) if self.prev_bid_price else None,
            'prev_bid_size': float(self.prev_bid_size) if self.prev_bid_size else None,
            'prev_ask_price': float(self.prev_ask_price) if self.prev_ask_price else None,
            'prev_ask_size': float(self.prev_ask_size) if self.prev_ask_size else None,
            'curr_bid_price': float(self.curr_bid_price) if self.curr_bid_price else None,
            'curr_bid_size': float(self.curr_bid_size) if self.curr_bid_size else None,
            'curr_ask_price': float(self.curr_ask_price) if self.curr_ask_price else None,
            'curr_ask_size': float(self.curr_ask_size) if self.curr_ask_size else None,
            'mid_price': float(self.mid_price) if self.mid_price else None,
            'spread': float(self.spread) if self.spread else None,
        }


@dataclass
class OFISignal:
    """
    Aggregated Order Flow Imbalance signal over a time window.
    
    The OFI signal is the sum of OFI events (e_n) over a rolling window:
    
        OFI = sum(e_n for n in window)
    
    A positive OFI suggests net buying pressure (aggressive buys or passive bid additions),
    while a negative OFI suggests net selling pressure (aggressive sells or passive ask additions).
    
    Attributes:
        timestamp: When this signal was computed (UTC)
        ofi_value: Aggregated OFI over the window
        window_size: Number of events in the current window
        max_window_size: Maximum window size (capacity)
        event_count: Total number of events processed since initialization
        mid_price: Current mid-price
        spread: Current bid-ask spread
        micro_price: Volume-weighted micro-price
        mean_event_value: Average e_n over the window
        std_event_value: Standard deviation of e_n over the window
    """
    timestamp: datetime
    ofi_value: Decimal
    window_size: int  # Current number of events in window
    max_window_size: int  # Maximum capacity of window
    event_count: int  # Total events processed
    
    # Market state
    mid_price: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    micro_price: Optional[Decimal] = None
    
    # Statistical properties
    mean_event_value: Optional[Decimal] = None
    std_event_value: Optional[Decimal] = None
    min_event_value: Optional[Decimal] = None
    max_event_value: Optional[Decimal] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ofi_value': float(self.ofi_value),
            'window_size': self.window_size,
            'max_window_size': self.max_window_size,
            'event_count': self.event_count,
            'mid_price': float(self.mid_price) if self.mid_price else None,
            'spread': float(self.spread) if self.spread else None,
            'micro_price': float(self.micro_price) if self.micro_price else None,
            'mean_event_value': float(self.mean_event_value) if self.mean_event_value else None,
            'std_event_value': float(self.std_event_value) if self.std_event_value else None,
            'min_event_value': float(self.min_event_value) if self.min_event_value else None,
            'max_event_value': float(self.max_event_value) if self.max_event_value else None,
        }
    
    def is_bullish(self) -> bool:
        """Check if signal indicates bullish pressure (positive OFI)."""
        return self.ofi_value > 0
    
    def is_bearish(self) -> bool:
        """Check if signal indicates bearish pressure (negative OFI)."""
        return self.ofi_value < 0
    
    def is_neutral(self) -> bool:
        """Check if signal is neutral (zero OFI)."""
        return self.ofi_value == 0
    
    def signal_strength(self) -> str:
        """
        Classify signal strength based on magnitude.
        
        This is a simple heuristic classification. In production, thresholds
        would be calibrated based on historical data and asset characteristics.
        
        Returns:
            'strong', 'moderate', 'weak', or 'neutral'
        """
        abs_ofi = abs(self.ofi_value)
        
        if abs_ofi == 0:
            return 'neutral'
        elif abs_ofi < 1:
            return 'weak'
        elif abs_ofi < 10:
            return 'moderate'
        else:
            return 'strong'


@dataclass
class OFIStatistics:
    """
    Statistical summary of OFI calculator performance.
    
    This tracks operational metrics for monitoring and debugging the OFI calculator.
    
    Attributes:
        total_events_processed: Total number of OFI events calculated
        total_updates_received: Total number of book updates received
        events_skipped: Number of updates skipped (e.g., invalid BBO)
        current_window_size: Current number of events in the rolling window
        max_window_size: Maximum capacity of the window
        last_update_timestamp: When the last update was processed
        first_event_timestamp: When the first event was processed
        uptime_seconds: How long the calculator has been running
    """
    total_events_processed: int = 0
    total_updates_received: int = 0
    events_skipped: int = 0
    current_window_size: int = 0
    max_window_size: int = 0
    last_update_timestamp: Optional[datetime] = None
    first_event_timestamp: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_events_processed': self.total_events_processed,
            'total_updates_received': self.total_updates_received,
            'events_skipped': self.events_skipped,
            'current_window_size': self.current_window_size,
            'max_window_size': self.max_window_size,
            'last_update_timestamp': self.last_update_timestamp.isoformat() if self.last_update_timestamp else None,
            'first_event_timestamp': self.first_event_timestamp.isoformat() if self.first_event_timestamp else None,
            'uptime_seconds': self.uptime_seconds,
        }
    
    def processing_rate(self) -> float:
        """
        Calculate events processed per second.
        
        Returns:
            Events per second, or 0.0 if no time has elapsed
        """
        if self.uptime_seconds > 0:
            return self.total_events_processed / self.uptime_seconds
        return 0.0
    
    def skip_rate(self) -> float:
        """
        Calculate percentage of updates that were skipped.
        
        Returns:
            Percentage (0-100) of updates skipped
        """
        if self.total_updates_received > 0:
            return (self.events_skipped / self.total_updates_received) * 100
        return 0.0


@dataclass
class MetricsSnapshot:
    """
    Performance metrics snapshot for OFI signal evaluation.
    
    This captures risk-adjusted performance metrics that quantify the quality
    and predictive power of the OFI signal over a given time period.
    
    Key Metrics:
        - Sharpe Ratio: Risk-adjusted return metric
        - Hit Rate: Directional accuracy (% of correct predictions)
        - Max Drawdown: Largest peak-to-trough decline
        - Win/Loss Ratio: Average winning prediction vs losing prediction
        - Signal Correlation: Linear relationship with price changes
    
    Attributes:
        timestamp: When this snapshot was created
        window_size: Number of observations in the window
        sharpe_ratio: Rolling Sharpe ratio (annualized)
        hit_rate: Percentage of correct directional predictions (0-100)
        total_predictions: Total number of predictions evaluated
        correct_predictions: Number of correct predictions
        incorrect_predictions: Number of incorrect predictions
        max_drawdown: Maximum drawdown percentage
        current_drawdown: Current drawdown percentage
        win_loss_ratio: Average winning prediction / average losing prediction
        avg_winning_prediction: Average magnitude of winning predictions
        avg_losing_prediction: Average magnitude of losing predictions
        signal_persistence: Average consecutive periods in same direction
        price_correlation: Correlation coefficient with price changes
        ofi_mean: Mean OFI value over window
        ofi_std: Standard deviation of OFI values
        ofi_min: Minimum OFI value
        ofi_max: Maximum OFI value
    """
    timestamp: datetime
    window_size: int
    
    # Risk-adjusted performance
    sharpe_ratio: Optional[Decimal] = None
    
    # Directional accuracy
    hit_rate: Optional[Decimal] = None  # Percentage (0-100)
    total_predictions: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    
    # Drawdown metrics
    max_drawdown: Optional[Decimal] = None  # Percentage
    current_drawdown: Optional[Decimal] = None  # Percentage
    
    # Win/Loss analysis
    win_loss_ratio: Optional[Decimal] = None
    avg_winning_prediction: Optional[Decimal] = None
    avg_losing_prediction: Optional[Decimal] = None
    
    # Signal characteristics
    signal_persistence: Optional[Decimal] = None  # Average consecutive periods
    price_correlation: Optional[Decimal] = None  # Correlation with price changes
    information_coefficient: Optional[Decimal] = None  # Predictive correlation (IC)
    
    # Advanced metrics
    alpha_decay: Optional[dict] = field(default_factory=dict)  # IC at different horizons
    slippage_bps: Optional[Decimal] = None  # Average slippage in basis points
    market_impact_bps: Optional[Decimal] = None  # Price reversion after trade
    
    # OFI signal statistics
    ofi_mean: Optional[Decimal] = None
    ofi_std: Optional[Decimal] = None
    ofi_min: Optional[Decimal] = None
    ofi_min: Optional[Decimal] = None
    ofi_max: Optional[Decimal] = None

    # Diagnostic Data
    scatter_data: Optional[List[Tuple[float, float]]] = field(default_factory=list)
    rolling_volatility: Optional[float] = None
    
    # Execution Tracking
    recent_trades: Optional[List[dict]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'window_size': self.window_size,
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'hit_rate': float(self.hit_rate) if self.hit_rate else None,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'incorrect_predictions': self.incorrect_predictions,
            'max_drawdown': float(self.max_drawdown) if self.max_drawdown else None,
            'current_drawdown': float(self.current_drawdown) if self.current_drawdown else None,
            'win_loss_ratio': float(self.win_loss_ratio) if self.win_loss_ratio else None,
            'avg_winning_prediction': float(self.avg_winning_prediction) if self.avg_winning_prediction else None,
            'avg_losing_prediction': float(self.avg_losing_prediction) if self.avg_losing_prediction else None,
            'signal_persistence': float(self.signal_persistence) if self.signal_persistence else None,
            'price_correlation': float(self.price_correlation) if self.price_correlation else None,
            'information_coefficient': float(self.information_coefficient) if self.information_coefficient else None,
            'alpha_decay': self.alpha_decay,
            'slippage_bps': float(self.slippage_bps) if self.slippage_bps else None,
            'market_impact_bps': float(self.market_impact_bps) if self.market_impact_bps else None,
            'ofi_mean': float(self.ofi_mean) if self.ofi_mean else None,
            'ofi_std': float(self.ofi_std) if self.ofi_std else None,
            'ofi_min': float(self.ofi_min) if self.ofi_min else None,
            'ofi_max': float(self.ofi_max) if self.ofi_max else None,
        }
    
    def is_performing_well(self, sharpe_threshold: float = 1.0, hit_rate_threshold: float = 55.0) -> bool:
        """
        Check if metrics indicate good performance.
        
        Args:
            sharpe_threshold: Minimum acceptable Sharpe ratio (default: 1.0)
            hit_rate_threshold: Minimum acceptable hit rate % (default: 55%)
        
        Returns:
            True if both Sharpe and hit rate exceed thresholds
        """
        sharpe_ok = self.sharpe_ratio is not None and float(self.sharpe_ratio) >= sharpe_threshold
        hit_rate_ok = self.hit_rate is not None and float(self.hit_rate) >= hit_rate_threshold
        return sharpe_ok and hit_rate_ok
