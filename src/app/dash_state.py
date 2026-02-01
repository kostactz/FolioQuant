"""
Global state for Dash application.

Unlike Streamlit's session_state, Dash requires global state for communication
between background threads and callbacks. This module provides a shared state
object that is thread-safe for basic read/write operations (Python GIL).

Architecture:
    - Single global DashboardState instance
    - Background WebSocket thread writes to state
    - Dash callbacks read from state
    - Python's GIL ensures thread safety for basic operations
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, List, Tuple
from datetime import datetime
from decimal import Decimal


@dataclass
class DashboardState:
    """
    Global state shared between WebSocket thread and Dash callbacks.
    
    This replaces Streamlit's session_state and is thread-safe for
    basic read/write operations thanks to Python's Global Interpreter Lock (GIL).
    
    Attributes:
        connected: WebSocket connection status
        last_heartbeat: Timestamp of last heartbeat message
        message_count: Total messages received
        error_message: Latest error message (if any)
        
        book_initialized: Order book has received snapshot
        book: Reference to OrderBook instance (for deep queries)
        best_bid: (price, size) tuple for best bid
        best_ask: (price, size) tuple for best ask
        mid_price: Mid-price (bid + ask) / 2
        spread: Bid-ask spread
        micro_price: Volume-weighted micro-price
        
        ofi_history: Rolling window of OFI signals
        bid_depth: Order book bid side depth [(price, cumulative_size), ...]
        ask_depth: Order book ask side depth [(price, cumulative_size), ...]
        
        sharpe_ratio: Sharpe ratio performance metric
        hit_rate: Hit rate (% of correct predictions)
        max_drawdown: Maximum drawdown
        win_loss_ratio: Win/loss ratio
        total_predictions: Total predictions made
        price_correlation: OFI-price correlation
    """
    
    # Connection state
    connected: bool = False
    streaming_enabled: bool = True  # Control for pause/resume streaming
    last_heartbeat: Optional[datetime] = None
    message_count: int = 0
    error_message: Optional[str] = None
    
    # Configuration (dynamic settings)
    product_id: str = "BTC-USD"
    ofi_window: int = 100
    chart_history: int = 500
    book_depth: int = 10
    signal_threshold: float = 5.0  # Adjustable signal threshold
    
    # Service references
    metrics_service: Optional[object] = None
    
    # Order book state
    book_initialized: bool = False
    book: Optional[object] = None  # OrderBook instance
    best_bid: Optional[Tuple[float, float]] = None
    best_ask: Optional[Tuple[float, float]] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    micro_price: Optional[float] = None
    
    # OFI state (rolling windows)
    ofi_history: Deque = field(default_factory=lambda: deque(maxlen=500))
    metrics_history: Deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Depth chart data
    bid_depth: List[Tuple[float, float]] = field(default_factory=list)
    ask_depth: List[Tuple[float, float]] = field(default_factory=list)
    
    # Performance metrics
    sharpe_ratio: Optional[float] = None
    hit_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_loss_ratio: Optional[float] = None
    total_predictions: int = 0
    price_correlation: Optional[float] = None
    
    # Diagnostics
    scatter_data: List[Tuple[float, float]] = field(default_factory=list)
    volatility_history: Deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Latency & Throughput metrics
    message_latency_ms: Optional[float] = None  # Latest message latency (Total)
    avg_latency_ms: Optional[float] = None  # Rolling average (Total)
    avg_network_latency: Optional[float] = None  # Exchange -> Ingest
    avg_system_latency: Optional[float] = None   # Ingest -> Process
    messages_per_second: float = 0.0  # Message throughput
    last_message_time: Optional[datetime] = None  # For throughput calculation
    
    # Analyst Metrics
    slippage_buy: Optional[float] = None  # bps for 1.0 BTC
    slippage_sell: Optional[float] = None  # bps for 1.0 BTC
    information_coefficient: Optional[float] = None
    information_coefficient: Optional[float] = None
    alpha_decay: dict = field(default_factory=dict)  # lag -> ic mapping
    recent_trades: List[dict] = field(default_factory=list)
    
    def reset(self):
        """Reset state (useful for reconnection)."""
        self.connected = False
        self.book = None
        self.book_initialized = False
        self.ofi_history.clear()
        self.metrics_history.clear()
        self.bid_depth.clear()
        self.ask_depth.clear()
        self.message_count = 0
        self.error_message = None
        self.sharpe_ratio = None
        self.hit_rate = None
        self.max_drawdown = None
        self.win_loss_ratio = None
        self.total_predictions = 0
        self.total_predictions = 0
        self.price_correlation = None
        self.scatter_data = []
        self.volatility_history.clear()
        # Keep configuration settings (don't reset on reconnect)


# CRITICAL: Single global instance
# This is shared between:
# - Background WebSocket thread (writes)
# - Dash callbacks (reads)
# Python's GIL ensures basic thread safety for reads/writes
state = DashboardState()
