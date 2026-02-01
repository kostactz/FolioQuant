"""
BookManager Service - Application Layer

This service manages the order book state, consuming messages from the infrastructure
layer and maintaining synchronized book state. It tracks sequence numbers for gap
detection and stores previous BBO state for OFI calculation in Phase 4.

Key Responsibilities:
1. Initialize order book from snapshot messages
2. Apply incremental l2update messages
3. Track sequence numbers to detect dropped messages
4. Store previous BBO (best bid/offer) state for OFI calculation
5. Provide clean API for state queries and subscribers

Architecture:
- Consumes from AsyncMessageQueue (Phase 1)
- Uses OrderBook data structure (Phase 2)
- Prepares state for OFICalculator (Phase 4)
"""

import asyncio
import logging
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..models.order_book import OrderBook
from ..models.market_data import SnapshotMessage, L2UpdateMessage, HeartbeatMessage, parse_message


@dataclass
class BBOState:
    """
    Best Bid/Offer state snapshot.
    
    This is stored before and after each update to calculate OFI deltas in Phase 4.
    The OFI formula requires comparing the previous BBO with the current BBO to
    determine whether a price level improved, stayed the same, or degraded.
    
    Attributes:
        best_bid_price: Price of the best bid (highest buy order)
        best_bid_size: Size at the best bid
        best_ask_price: Price of the best ask (lowest sell order)
        best_ask_size: Size at the best ask
        timestamp: When this state was captured
        sequence: Sequence number when this state was captured
    """
    best_bid_price: Optional[Decimal] = None
    best_bid_size: Optional[Decimal] = None
    best_ask_price: Optional[Decimal] = None
    best_ask_size: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    sequence: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Check if this BBO state is valid (both bid and ask present)."""
        return (
            self.best_bid_price is not None and
            self.best_bid_size is not None and
            self.best_ask_price is not None and
            self.best_ask_size is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "best_bid_price": float(self.best_bid_price) if self.best_bid_price else None,
            "best_bid_size": float(self.best_bid_size) if self.best_bid_size else None,
            "best_ask_price": float(self.best_ask_price) if self.best_ask_price else None,
            "best_ask_size": float(self.best_ask_size) if self.best_ask_size else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sequence": self.sequence,
        }


@dataclass
class BookManagerStats:
    """Statistics for BookManager performance monitoring."""
    total_messages_processed: int = 0
    snapshots_applied: int = 0
    updates_applied: int = 0
    heartbeats_received: int = 0
    sequence_gaps_detected: int = 0
    errors_encountered: int = 0
    last_update_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "total_messages_processed": self.total_messages_processed,
            "snapshots_applied": self.snapshots_applied,
            "updates_applied": self.updates_applied,
            "heartbeats_received": self.heartbeats_received,
            "sequence_gaps_detected": self.sequence_gaps_detected,
            "errors_encountered": self.errors_encountered,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
        }


class BookManager:
    """
    Order book state manager service.
    
    This service sits between the infrastructure layer (WebSocket client + message queue)
    and the application layer (OFI calculator). It maintains the order book state and
    provides a clean API for querying current state and subscribing to updates.
    
    Usage:
        manager = BookManager(product_id="BTC-USD")
        
        # Process a message from the queue
        await manager.process_message(message_dict)
        
        # Query current state
        bbo = manager.get_current_bbo()
        book_state = manager.get_book_state()
        
        # Subscribe to updates (Phase 4)
        manager.subscribe(on_book_update)
    
    Attributes:
        product_id: Trading pair identifier (e.g., "BTC-USD")
        book: OrderBook instance maintaining current state
        initialized: Whether we've received and applied a snapshot
        last_sequence: Last sequence number processed
        previous_bbo: BBO state before the most recent update (for OFI)
        current_bbo: BBO state after the most recent update (for OFI)
        stats: Performance and diagnostic statistics
    """
    
    def __init__(
        self,
        product_id: str,
        enable_sequence_tracking: bool = True,
        log_sequence_gaps: bool = True
    ):
        """
        Initialize BookManager.
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD", "ETH-USD")
            enable_sequence_tracking: If True, track sequence numbers for gap detection
            log_sequence_gaps: If True, log when sequence gaps are detected
        """
        self.product_id = product_id
        self.book = OrderBook(product_id)
        self.initialized = False
        
        # Sequence tracking for gap detection
        self.enable_sequence_tracking = enable_sequence_tracking
        self.log_sequence_gaps = log_sequence_gaps
        self.last_sequence: Optional[int] = None
        
        # BBO tracking for OFI calculation (Phase 4)
        self.previous_bbo = BBOState()
        self.current_bbo = BBOState()
        
        # Statistics
        self.stats = BookManagerStats()
        
        # Subscriber callbacks (observer pattern for Phase 4)
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{product_id}")
        self.logger.debug(f"BookManager initialized for {product_id}")
    
    async def process_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a message from the WebSocket feed.
        
        This is the main entry point for consuming messages from AsyncMessageQueue.
        It routes messages to the appropriate handler based on type.
        
        Args:
            message: Raw message dictionary from Coinbase WebSocket
            
        Returns:
            True if message was processed successfully, False otherwise
            
        Note:
            This method updates self.previous_bbo and self.current_bbo for OFI calculation.
        """
        try:
            self.stats.total_messages_processed += 1
            self.stats.last_update_time = datetime.utcnow()
            
            msg_type = message.get("type")
            
            if msg_type == "snapshot":
                return await self._process_snapshot(message)
            elif msg_type == "l2update":
                return await self._process_update(message)
            elif msg_type == "heartbeat":
                return await self._process_heartbeat(message)
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                return False
                
        except Exception as e:
            self.stats.errors_encountered += 1
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            return False
    
    async def _process_snapshot(self, message: Dict[str, Any]) -> bool:
        """
        Process a snapshot message to initialize the order book.
        
        Args:
            message: Snapshot message dictionary
            
        Returns:
            True if snapshot was applied successfully
        """
        try:
            snapshot = SnapshotMessage.from_dict(message)
            
            # Apply snapshot to order book
            self.book.apply_snapshot(snapshot.to_dict())
            
            # Update sequence tracking
            if self.enable_sequence_tracking and hasattr(snapshot, 'sequence'):
                self.last_sequence = snapshot.sequence
            
            # Capture initial BBO state
            self._capture_bbo_state()
            
            # Mark as initialized
            self.initialized = True
            self.stats.snapshots_applied += 1
            
            self.logger.debug(
                f"Snapshot applied: {len(self.book.bids)} bids, {len(self.book.asks)} asks"
            )
            
            # Notify subscribers
            await self._notify_subscribers("snapshot")
            
            return True
            
        except Exception as e:
            self.stats.errors_encountered += 1
            self.logger.error(f"Error processing snapshot: {e}", exc_info=True)
            return False
    
    async def _process_update(self, message: Dict[str, Any]) -> bool:
        """
        Process an l2update message to update the order book.
        
        Optimized for high-frequency updates:
        - Direct dict access (no L2UpdateMessage overhead)
        - Throttled logging for sequence gaps
        """
        # Can't process updates until initialized with snapshot
        if not self.initialized:
            # Throttle this warning too
            current_time = time.time()
            if getattr(self, '_last_init_warn_time', 0) + 5.0 < current_time:
                self.logger.warning("Received update before snapshot, skipping")
                self._last_init_warn_time = current_time
            return False
        
        try:
            # OPTIMIZATION: Access raw dict directly, skip L2UpdateMessage.from_dict
            changes = message.get("changes", [])
            product_id = message.get("product_id")
            time_str = message.get("time")
            
            # Check for sequence gaps
            if self.enable_sequence_tracking and "sequence" in message:
                sequence = int(message["sequence"])
                
                # Inline check_sequence to avoid method call overhead
                if self.last_sequence is not None and sequence != self.last_sequence + 1:
                    self.stats.sequence_gaps_detected += 1
                    
                    # LOGGING THROTTLE: Only log 1 gap per second
                    current_time = time.time()
                    if self.log_sequence_gaps and (getattr(self, '_last_gap_log_time', 0) + 1.0 < current_time):
                        self.logger.warning(
                            f"Sequence gap detected: expected {self.last_sequence + 1}, "
                            f"got {sequence} (gap of {sequence - self.last_sequence - 1}). "
                            f"Suppressed for 1s."
                        )
                        self._last_gap_log_time = current_time
                
                self.last_sequence = sequence
            
            # CRITICAL: Store previous BBO state BEFORE applying update
            # OPTIMIZATION: reused existing BBOState class but shallow copy could be faster?
            # For now, keeping explicit copy as it's safe.
            self.previous_bbo = BBOState(
                best_bid_price=self.current_bbo.best_bid_price,
                best_bid_size=self.current_bbo.best_bid_size,
                best_ask_price=self.current_bbo.best_ask_price,
                best_ask_size=self.current_bbo.best_ask_size,
                timestamp=self.current_bbo.timestamp,
                sequence=self.current_bbo.sequence,
            )
            
            # Apply update to order book
            # OPTIMIZATION: Pass changes directly to book
            # book.apply_update expects a dict with 'changes'
            self.book.apply_update({
                'type': 'l2update',
                'changes': changes,
                'product_id': product_id,
                'time': time_str
            })
            
            # Capture new BBO state
            sequence_for_bbo = message.get("sequence") if self.enable_sequence_tracking else None
            self._capture_bbo_state(sequence=sequence_for_bbo)
            
            self.stats.updates_applied += 1
            
            # Notify subscribers
            await self._notify_subscribers("l2update")
            
            return True
            
        except Exception as e:
            self.stats.errors_encountered += 1
            # Throttle error logs too
            current_time = time.time()
            if getattr(self, '_last_error_log_time', 0) + 1.0 < current_time:
                self.logger.error(f"Error processing update: {e}", exc_info=True)
                self._last_error_log_time = current_time
            return False
    
    async def _process_heartbeat(self, message: Dict[str, Any]) -> bool:
        """
        Process a heartbeat message for connection health monitoring.
        
        Optimized:
        - Direct dict access
        - Throttled logging
        """
        try:
            # OPTIMIZATION: raw dict access
            sequence = int(message.get("sequence", 0))
            time_str = message.get("time")
            
            # Update sequence tracking from heartbeat
            if self.enable_sequence_tracking:
                # Check for sequence gap
                if self.last_sequence is not None and sequence != self.last_sequence + 1:
                    self.stats.sequence_gaps_detected += 1
                    
                    # LOGGING THROTTLE: Only log 1 gap per second
                    current_time = time.time()
                    if self.log_sequence_gaps and (getattr(self, '_last_gap_log_time', 0) + 1.0 < current_time):
                        self.logger.warning(
                            f"Sequence gap in heartbeat: expected {self.last_sequence + 1}, "
                            f"got {sequence}. Suppressed for 1s."
                        )
                        self._last_gap_log_time = current_time
                        
                self.last_sequence = sequence
            
            self.stats.heartbeats_received += 1
            
            return True
            
        except Exception as e:
            self.stats.errors_encountered += 1
            current_time = time.time()
            if getattr(self, '_last_error_log_time', 0) + 1.0 < current_time:
                self.logger.error(f"Error processing heartbeat: {e}", exc_info=True)
                self._last_error_log_time = current_time
            return False
    
    def _check_sequence(self, sequence: int) -> bool:
        """
        Check if sequence number is consecutive.
        
        Args:
            sequence: Current sequence number
            
        Returns:
            True if sequence is consecutive (or first sequence), False if gap detected
        """
        if self.last_sequence is None:
            return True
        
        expected = self.last_sequence + 1
        return sequence == expected
    
    def _capture_bbo_state(self, sequence: Optional[int] = None) -> None:
        """
        Capture current BBO state for OFI calculation.
        
        This method updates self.current_bbo with the current best bid/ask
        from the order book. It's called after each snapshot or update.
        
        Args:
            sequence: Optional sequence number to store with the BBO state
        """
        # OrderBook.best_bid and best_ask return (price, size) tuples or None
        best_bid = self.book.best_bid
        best_ask = self.book.best_ask
        
        self.current_bbo = BBOState(
            best_bid_price=best_bid[0] if best_bid else None,
            best_bid_size=best_bid[1] if best_bid else None,
            best_ask_price=best_ask[0] if best_ask else None,
            best_ask_size=best_ask[1] if best_ask else None,
            timestamp=datetime.utcnow(),
            sequence=sequence,
        )
    
    def get_current_bbo(self) -> BBOState:
        """
        Get current best bid/offer state.
        
        Returns:
            Current BBOState with best bid/ask prices and sizes
        """
        return self.current_bbo
    
    def get_previous_bbo(self) -> BBOState:
        """
        Get previous best bid/offer state (before last update).
        
        This is used for OFI calculation to compare state changes.
        
        Returns:
            Previous BBOState
        """
        return self.previous_bbo
    
    def get_book_state(self) -> Dict[str, Any]:
        """
        Get complete order book state.
        
        Returns:
            Dictionary with current book state including:
            - product_id
            - initialized
            - best_bid_price, best_bid_size
            - best_ask_price, best_ask_size
            - mid_price
            - spread
            - micro_price
            - num_bid_levels, num_ask_levels
            - last_sequence
        """
        # OrderBook.best_bid and best_ask return (price, size) tuples or None
        best_bid = self.book.best_bid
        best_ask = self.book.best_ask
        
        return {
            "product_id": self.product_id,
            "initialized": self.initialized,
            "best_bid_price": float(best_bid[0]) if best_bid else None,
            "best_bid_size": float(best_bid[1]) if best_bid else None,
            "best_ask_price": float(best_ask[0]) if best_ask else None,
            "best_ask_size": float(best_ask[1]) if best_ask else None,
            "mid_price": float(self.book.mid_price) if self.book.mid_price else None,
            "spread": float(self.book.spread) if self.book.spread else None,
            "micro_price": float(self.book.micro_price) if self.book.micro_price else None,
            "num_bid_levels": len(self.book.bids),
            "num_ask_levels": len(self.book.asks),
            "last_sequence": self.last_sequence,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get BookManager statistics.
        
        Returns:
            Statistics dictionary with processing counts and diagnostics
        """
        return self.stats.to_dict()
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to book update events.
        
        This implements the observer pattern for Phase 4 integration with OFI calculator.
        Subscribers are notified after each successful snapshot or update.
        
        Args:
            callback: Async function to call on book updates.
                      Receives a dict with event info: {"type": "snapshot"|"l2update", "state": {...}}
        
        Example:
            async def on_update(event):
                print(f"Book updated: {event['type']}")
                
            manager.subscribe(on_update)
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from book update events.
        
        Args:
            callback: Previously registered callback to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            self.logger.debug(f"Subscriber removed (total: {len(self._subscribers)})")
    
    async def _notify_subscribers(self, event_type: str) -> None:
        """
        Notify all subscribers of a book update event.
        
        Args:
            event_type: Type of event ("snapshot" or "l2update")
        """
        if not self._subscribers:
            return
        
        event = {
            "type": event_type,
            "product_id": self.product_id,
            "state": self.get_book_state(),
            "previous_bbo": self.previous_bbo,  # Pass BBOState object directly
            "current_bbo": self.current_bbo,    # Pass BBOState object directly
            "timestamp": self.current_bbo.timestamp or datetime.utcnow(),
            "sequence": self.current_bbo.sequence,
        }
        
        # Call all subscribers (they should be async)
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    # If callback is not async, call it directly (less preferred)
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}", exc_info=True)
    
    def reset(self) -> None:
        """
        Reset the book manager state.
        
        This clears all state including the order book, sequence tracking,
        and BBO history. Useful for reconnection scenarios.
        """
        self.book = OrderBook(self.product_id)
        self.initialized = False
        self.last_sequence = None
        self.previous_bbo = BBOState()
        self.current_bbo = BBOState()
        # Don't reset stats - keep cumulative counts
        self.logger.debug("BookManager state reset")
    
    
    def calculate_slippage(self, size: float, side: str) -> Optional[float]:
        """
        Calculate expected slippage for a market order of given size.
        
        Slippage is the difference between the VWAP fill price and the current mid price,
        expressed in basis points (bps).
        
        Args:
            size: Order size to simulate (e.g., 1.0 BTC)
            side: 'buy' (consumes asks) or 'sell' (consumes bids)
            
        Returns:
            Slippage in basis points, or None if book is empty/insufficient
        """
        if not self.book or not self.book.mid_price:
            return None
        
        target_size = Decimal(str(size))
        mid_price = self.book.mid_price
        
        # Select side to consume
        if side.lower() == 'buy':
            # Buy consumes asks (sorted ascending - lowest price first)
            levels = self.book.asks
        else:
            # Sell consumes bids (sorted descending - highest price first)
            levels = self.book.bids
            
        if not levels:
            return None
            
        remaining_size = target_size
        total_cost = Decimal('0')
        filled_size = Decimal('0')
        
        # Walk the book
        for price, level_size in levels.items():
            # How much can we fill at this level?
            fill_at_level = min(remaining_size, level_size)
            
            total_cost += fill_at_level * price
            filled_size += fill_at_level
            remaining_size -= fill_at_level
            
            if remaining_size <= 0:
                break
        
        if filled_size == 0:
            return None
            
        # Calculate VWAP
        vwap = total_cost / filled_size
        
        # Calculate slippage
        # Buy: VWAP > Mid (Slippage = VWAP - Mid)
        # Sell: VWAP < Mid (Slippage = Mid - VWAP)
        # In both cases: |VWAP - Mid|
        diff = abs(vwap - mid_price)
        
        # Convert to bps
        if mid_price == 0:
            return None
            
        slippage_bps = (diff / mid_price) * 10000
        
        return float(slippage_bps)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BookManager(product_id='{self.product_id}', "
            f"initialized={self.initialized}, "
            f"sequence={self.last_sequence}, "
            f"bids={len(self.book.bids)}, "
            f"asks={len(self.book.asks)})"
        )
