"""
Tests for BookManager service.

This module tests the application layer service that manages order book state,
including sequence tracking, BBO state management, and message processing.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime

from src.services.book_manager import BookManager, BBOState, BookManagerStats


class TestBBOState:
    """Tests for BBOState data class."""
    
    def test_bbo_state_initialization(self):
        """Test BBOState can be initialized with default values."""
        bbo = BBOState()
        
        assert bbo.best_bid_price is None
        assert bbo.best_bid_size is None
        assert bbo.best_ask_price is None
        assert bbo.best_ask_size is None
        assert bbo.timestamp is None
        assert bbo.sequence is None
    
    def test_bbo_state_with_values(self):
        """Test BBOState with actual values."""
        now = datetime.utcnow()
        bbo = BBOState(
            best_bid_price=Decimal("50000.0"),
            best_bid_size=Decimal("1.5"),
            best_ask_price=Decimal("50001.0"),
            best_ask_size=Decimal("2.0"),
            timestamp=now,
            sequence=12345,
        )
        
        assert bbo.best_bid_price == Decimal("50000.0")
        assert bbo.best_bid_size == Decimal("1.5")
        assert bbo.best_ask_price == Decimal("50001.0")
        assert bbo.best_ask_size == Decimal("2.0")
        assert bbo.timestamp == now
        assert bbo.sequence == 12345
    
    def test_bbo_state_is_valid(self):
        """Test BBOState validity checking."""
        # Invalid: empty
        bbo = BBOState()
        assert not bbo.is_valid()
        
        # Invalid: only bid
        bbo = BBOState(
            best_bid_price=Decimal("50000"),
            best_bid_size=Decimal("1.0"),
        )
        assert not bbo.is_valid()
        
        # Valid: both bid and ask
        bbo = BBOState(
            best_bid_price=Decimal("50000"),
            best_bid_size=Decimal("1.0"),
            best_ask_price=Decimal("50001"),
            best_ask_size=Decimal("2.0"),
        )
        assert bbo.is_valid()
    
    def test_bbo_state_to_dict(self):
        """Test BBOState serialization to dictionary."""
        now = datetime.utcnow()
        bbo = BBOState(
            best_bid_price=Decimal("50000.0"),
            best_bid_size=Decimal("1.5"),
            best_ask_price=Decimal("50001.0"),
            best_ask_size=Decimal("2.0"),
            timestamp=now,
            sequence=12345,
        )
        
        result = bbo.to_dict()
        
        assert result["best_bid_price"] == 50000.0
        assert result["best_bid_size"] == 1.5
        assert result["best_ask_price"] == 50001.0
        assert result["best_ask_size"] == 2.0
        assert result["timestamp"] == now.isoformat()
        assert result["sequence"] == 12345


class TestBookManagerStats:
    """Tests for BookManagerStats data class."""
    
    def test_stats_initialization(self):
        """Test stats initialization with default values."""
        stats = BookManagerStats()
        
        assert stats.total_messages_processed == 0
        assert stats.snapshots_applied == 0
        assert stats.updates_applied == 0
        assert stats.heartbeats_received == 0
        assert stats.sequence_gaps_detected == 0
        assert stats.errors_encountered == 0
        assert stats.last_update_time is None
    
    def test_stats_to_dict(self):
        """Test stats serialization."""
        now = datetime.utcnow()
        stats = BookManagerStats(
            total_messages_processed=100,
            snapshots_applied=1,
            updates_applied=95,
            heartbeats_received=4,
            sequence_gaps_detected=2,
            errors_encountered=0,
            last_update_time=now,
        )
        
        result = stats.to_dict()
        
        assert result["total_messages_processed"] == 100
        assert result["snapshots_applied"] == 1
        assert result["updates_applied"] == 95
        assert result["heartbeats_received"] == 4
        assert result["sequence_gaps_detected"] == 2
        assert result["errors_encountered"] == 0
        assert result["last_update_time"] == now.isoformat()


class TestBookManagerInitialization:
    """Tests for BookManager initialization."""
    
    def test_initialization(self):
        """Test BookManager can be initialized."""
        manager = BookManager(product_id="BTC-USD")
        
        assert manager.product_id == "BTC-USD"
        assert manager.initialized is False
        assert manager.last_sequence is None
        assert manager.enable_sequence_tracking is True
        assert manager.log_sequence_gaps is True
        assert isinstance(manager.previous_bbo, BBOState)
        assert isinstance(manager.current_bbo, BBOState)
        assert isinstance(manager.stats, BookManagerStats)
    
    def test_initialization_with_options(self):
        """Test BookManager initialization with custom options."""
        manager = BookManager(
            product_id="ETH-USD",
            enable_sequence_tracking=False,
            log_sequence_gaps=False,
        )
        
        assert manager.product_id == "ETH-USD"
        assert manager.enable_sequence_tracking is False
        assert manager.log_sequence_gaps is False
    
    def test_repr(self):
        """Test string representation."""
        manager = BookManager(product_id="BTC-USD")
        
        repr_str = repr(manager)
        
        assert "BTC-USD" in repr_str
        assert "initialized=False" in repr_str


class TestBookManagerSnapshotProcessing:
    """Tests for snapshot message processing."""
    
    @pytest.mark.asyncio
    async def test_process_snapshot(self):
        """Test processing a snapshot message."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"],
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "1.5"],
            ],
        }
        
        result = await manager.process_message(snapshot)
        
        assert result is True
        assert manager.initialized is True
        assert manager.stats.snapshots_applied == 1
        assert manager.stats.total_messages_processed == 1
        
        # Check order book state (best_bid returns tuple)
        assert manager.book.best_bid[0] == Decimal("50000.00")
        assert manager.book.best_ask[0] == Decimal("50001.00")
        
        # Check BBO state was captured
        assert manager.current_bbo.best_bid_price == Decimal("50000.00")
        assert manager.current_bbo.best_bid_size == Decimal("1.5")
        assert manager.current_bbo.best_ask_price == Decimal("50001.00")
        assert manager.current_bbo.best_ask_size == Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_process_snapshot_with_sequence(self):
        """Test snapshot processing updates sequence tracking."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        
        await manager.process_message(snapshot)
        
        # Snapshot doesn't have sequence in Coinbase protocol
        # But our implementation should handle it gracefully
        assert manager.initialized is True


class TestBookManagerUpdateProcessing:
    """Tests for l2update message processing."""
    
    @pytest.mark.asyncio
    async def test_process_update_before_snapshot(self):
        """Test that updates are rejected before snapshot initialization."""
        manager = BookManager(product_id="BTC-USD")
        
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.00", "1.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
        }
        
        result = await manager.process_message(update)
        
        assert result is False
        assert manager.stats.updates_applied == 0
    
    @pytest.mark.asyncio
    async def test_process_update_after_snapshot(self):
        """Test processing updates after snapshot initialization."""
        manager = BookManager(product_id="BTC-USD")
        
        # First apply snapshot
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # Now apply update
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "2.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
        }
        
        result = await manager.process_message(update)
        
        assert result is True
        assert manager.stats.updates_applied == 1
        assert manager.book.best_bid[0] == Decimal("50000.50")
    
    @pytest.mark.asyncio
    async def test_bbo_tracking_on_update(self):
        """Test that previous and current BBO are tracked correctly."""
        manager = BookManager(product_id="BTC-USD")
        
        # Apply snapshot
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # Capture initial state
        initial_bid = manager.current_bbo.best_bid_price
        
        # Apply update that improves bid
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "2.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
        }
        await manager.process_message(update)
        
        # Previous BBO should have old bid
        assert manager.previous_bbo.best_bid_price == Decimal("50000.00")
        assert manager.previous_bbo.best_bid_size == Decimal("1.0")
        
        # Current BBO should have new bid
        assert manager.current_bbo.best_bid_price == Decimal("50000.50")
        assert manager.current_bbo.best_bid_size == Decimal("2.0")
    
    @pytest.mark.asyncio
    async def test_multiple_updates(self):
        """Test processing multiple sequential updates."""
        manager = BookManager(product_id="BTC-USD")
        
        # Apply snapshot
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # Apply multiple updates
        updates = [
            {
                "type": "l2update",
                "product_id": "BTC-USD",
                "changes": [["buy", "50000.50", "2.0"]],
                "time": "2024-01-01T00:00:00.000000Z",
            },
            {
                "type": "l2update",
                "product_id": "BTC-USD",
                "changes": [["sell", "50000.75", "1.5"]],
                "time": "2024-01-01T00:00:01.000000Z",
            },
            {
                "type": "l2update",
                "product_id": "BTC-USD",
                "changes": [["buy", "50000.00", "0"]],  # Delete
                "time": "2024-01-01T00:00:02.000000Z",
            },
        ]
        
        for update in updates:
            await manager.process_message(update)
        
        assert manager.stats.updates_applied == 3
        assert manager.book.best_bid[0] == Decimal("50000.50")


class TestBookManagerSequenceTracking:
    """Tests for sequence number tracking and gap detection."""
    
    @pytest.mark.asyncio
    async def test_sequence_tracking_disabled(self):
        """Test that sequence tracking can be disabled."""
        manager = BookManager(product_id="BTC-USD", enable_sequence_tracking=False)
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "2.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
            "sequence": 12345,
        }
        await manager.process_message(update)
        
        # Sequence should not be tracked
        assert manager.last_sequence is None
    
    @pytest.mark.asyncio
    async def test_sequence_tracking_consecutive(self):
        """Test consecutive sequence numbers are accepted."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # Apply updates with consecutive sequences
        for seq in [100, 101, 102]:
            update = {
                "type": "l2update",
                "product_id": "BTC-USD",
                "changes": [["buy", f"{50000 + seq}.00", "1.0"]],
                "time": "2024-01-01T00:00:00.000000Z",
                "sequence": seq,
            }
            await manager.process_message(update)
        
        assert manager.last_sequence == 102
        assert manager.stats.sequence_gaps_detected == 0
    
    @pytest.mark.asyncio
    async def test_sequence_gap_detection(self):
        """Test that sequence gaps are detected and logged."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # First update
        update1 = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "1.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
            "sequence": 100,
        }
        await manager.process_message(update1)
        
        # Second update with gap (skips 101, 102)
        update2 = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.75", "1.0"]],
            "time": "2024-01-01T00:00:01.000000Z",
            "sequence": 103,
        }
        await manager.process_message(update2)
        
        assert manager.stats.sequence_gaps_detected == 1
        assert manager.last_sequence == 103
    
    @pytest.mark.asyncio
    async def test_heartbeat_sequence_tracking(self):
        """Test that heartbeat messages update sequence tracking."""
        manager = BookManager(product_id="BTC-USD")
        
        heartbeat1 = {
            "type": "heartbeat",
            "sequence": 100,
            "last_trade_id": 12345,
            "product_id": "BTC-USD",
            "time": "2024-01-01T00:00:00.000000Z",
        }
        await manager.process_message(heartbeat1)
        
        assert manager.last_sequence == 100
        assert manager.stats.heartbeats_received == 1
        
        # Heartbeat with gap
        heartbeat2 = {
            "type": "heartbeat",
            "sequence": 105,  # Gap of 4
            "last_trade_id": 12346,
            "product_id": "BTC-USD",
            "time": "2024-01-01T00:00:01.000000Z",
        }
        await manager.process_message(heartbeat2)
        
        assert manager.stats.sequence_gaps_detected == 1


class TestBookManagerStateQueries:
    """Tests for state query methods."""
    
    @pytest.mark.asyncio
    async def test_get_current_bbo(self):
        """Test getting current BBO state."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"]],
            "asks": [["50001.00", "2.0"]],
        }
        await manager.process_message(snapshot)
        
        bbo = manager.get_current_bbo()
        
        assert bbo.best_bid_price == Decimal("50000.00")
        assert bbo.best_bid_size == Decimal("1.5")
        assert bbo.best_ask_price == Decimal("50001.00")
        assert bbo.best_ask_size == Decimal("2.0")
    
    @pytest.mark.asyncio
    async def test_get_previous_bbo(self):
        """Test getting previous BBO state."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "2.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
        }
        await manager.process_message(update)
        
        prev_bbo = manager.get_previous_bbo()
        
        assert prev_bbo.best_bid_price == Decimal("50000.00")
        assert prev_bbo.best_bid_size == Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_get_book_state(self):
        """Test getting complete book state."""
        manager = BookManager(product_id="BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.0"],
                ["49999.00", "2.0"],
            ],
            "asks": [
                ["50001.00", "1.5"],
                ["50002.00", "2.5"],
            ],
        }
        await manager.process_message(snapshot)
        
        state = manager.get_book_state()
        
        assert state["product_id"] == "BTC-USD"
        assert state["initialized"] is True
        assert state["best_bid_price"] == 50000.00
        assert state["best_ask_price"] == 50001.00
        assert state["mid_price"] == 50000.50
        assert state["spread"] == 1.00
        assert state["num_bid_levels"] == 2
        assert state["num_ask_levels"] == 2
    
    def test_get_stats(self):
        """Test getting statistics."""
        manager = BookManager(product_id="BTC-USD")
        
        stats = manager.get_stats()
        
        assert stats["total_messages_processed"] == 0
        assert stats["snapshots_applied"] == 0
        assert "last_update_time" in stats


class TestBookManagerSubscribers:
    """Tests for observer pattern / subscriber functionality."""
    
    @pytest.mark.asyncio
    async def test_subscribe_and_notify(self):
        """Test subscribing to updates and receiving notifications."""
        manager = BookManager(product_id="BTC-USD")
        
        events_received = []
        
        async def on_update(event):
            events_received.append(event)
        
        manager.subscribe(on_update)
        
        # Apply snapshot
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        assert len(events_received) == 1
        assert events_received[0]["type"] == "snapshot"
        assert events_received[0]["product_id"] == "BTC-USD"
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers receive events."""
        manager = BookManager(product_id="BTC-USD")
        
        events1 = []
        events2 = []
        
        async def subscriber1(event):
            events1.append(event)
        
        async def subscriber2(event):
            events2.append(event)
        
        manager.subscribe(subscriber1)
        manager.subscribe(subscriber2)
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        assert len(events1) == 1
        assert len(events2) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from updates."""
        manager = BookManager(product_id="BTC-USD")
        
        events = []
        
        async def on_update(event):
            events.append(event)
        
        manager.subscribe(on_update)
        manager.unsubscribe(on_update)
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        # Should not receive event after unsubscribe
        assert len(events) == 0


class TestBookManagerReset:
    """Tests for reset functionality."""
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting manager state."""
        manager = BookManager(product_id="BTC-USD")
        
        # Apply snapshot and update
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]],
        }
        await manager.process_message(snapshot)
        
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "50000.50", "2.0"]],
            "time": "2024-01-01T00:00:00.000000Z",
            "sequence": 100,
        }
        await manager.process_message(update)
        
        # Reset
        manager.reset()
        
        assert manager.initialized is False
        assert manager.last_sequence is None
        assert len(manager.book.bids) == 0
        assert len(manager.book.asks) == 0
        assert not manager.current_bbo.is_valid()
        assert not manager.previous_bbo.is_valid()
        # Stats should be preserved
        assert manager.stats.total_messages_processed == 2


class TestBookManagerErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_message_type(self):
        """Test handling of unknown message types."""
        manager = BookManager(product_id="BTC-USD")
        
        invalid = {
            "type": "unknown_type",
            "data": "test",
        }
        
        result = await manager.process_message(invalid)
        
        assert result is False
        assert manager.stats.total_messages_processed == 1
    
    @pytest.mark.asyncio
    async def test_malformed_message(self):
        """Test handling of malformed messages."""
        manager = BookManager(product_id="BTC-USD")
        
        malformed = {
            "type": "snapshot",
            # Missing required fields
        }
        
        result = await manager.process_message(malformed)
        
        assert result is False
        assert manager.stats.errors_encountered == 1
