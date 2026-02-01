"""
Simplified test suite for CoinbaseWebSocketClient focusing on testable logic.

These tests validate the client's core logic without requiring complex
async WebSocket mocking.
"""

import asyncio
import pytest

from src.clients.coinbase_client import CoinbaseWebSocketClient, ConnectionState
from src.clients.message_queue import AsyncMessageQueue


class TestCoinbaseWebSocketClientBasic:
    """Basic tests for CoinbaseWebSocketClient that don't require mocking."""
    
    def test_initialization_defaults(self):
        """Test client initialization with default parameters."""
        client = CoinbaseWebSocketClient()
        
        assert client.ws_url == "wss://ws-feed.exchange.coinbase.com"
        assert client.product_ids == ["BTC-USD"]
        assert client.channels == ["level2_batch", "heartbeat"]
        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_connected
    
    def test_initialization_custom(self):
        """Test client initialization with custom parameters."""
        client = CoinbaseWebSocketClient(
            ws_url="wss://custom.url",
            product_ids=["ETH-USD", "BTC-USD"],
            channels=["level2"],
            subscription_timeout=3.0,
            reconnect_delay=2.0,
            max_reconnect_delay=120.0,
            ping_interval=60.0,
        )
        
        assert client.ws_url == "wss://custom.url"
        assert client.product_ids == ["ETH-USD", "BTC-USD"]
        assert client.channels == ["level2"]
        assert client.subscription_timeout == 3.0
        assert client.initial_reconnect_delay == 2.0
        assert client.max_reconnect_delay == 120.0
        assert client.ping_interval == 60.0
    
    def test_callbacks(self):
        """Test adding message and error callbacks."""
        client = CoinbaseWebSocketClient()
        
        async def message_callback(msg_type, msg_data):
            pass
        
        async def error_callback(error_type, error_msg):
            pass
        
        client.add_message_callback(message_callback)
        client.add_error_callback(error_callback)
        
        assert len(client._on_message_callbacks) == 1
        assert len(client._on_error_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_reconnection_backoff(self):
        """Test exponential backoff on reconnection."""
        client = CoinbaseWebSocketClient(
            reconnect_delay=1.0,
            max_reconnect_delay=10.0
        )
        
        # Initial delay
        assert client._reconnect_delay == 1.0
        
        # Simulate reconnections
        await client._reconnect()
        assert client._reconnect_delay == 2.0
        assert client._reconnect_attempts == 1
        
        await client._reconnect()
        assert client._reconnect_delay == 4.0
        assert client._reconnect_attempts == 2
        
        await client._reconnect()
        assert client._reconnect_delay == 8.0
        
        await client._reconnect()
        assert client._reconnect_delay == 10.0  # Capped at max
    
    @pytest.mark.asyncio
    async def test_reconnection_resets_sequence(self):
        """Test that reconnection resets sequence tracking."""
        client = CoinbaseWebSocketClient()
        
        # Set a sequence number
        client._last_sequence = 12345
        
        # Reconnect
        await client._reconnect()
        
        # Sequence should be reset
        assert client._last_sequence is None
        assert client._reconnect_attempts == 1
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        client = CoinbaseWebSocketClient()
        
        stats = client.get_stats()
        assert stats["state"] == ConnectionState.DISCONNECTED.value
        assert stats["total_messages"] == 0
        assert stats["sequence_gaps"] == 0
        assert stats["last_sequence"] is None
        assert stats["reconnect_attempts"] == 0
    
    def test_repr(self):
        """Test string representation."""
        client = CoinbaseWebSocketClient()
        repr_str = repr(client)
        
        assert "CoinbaseWebSocketClient" in repr_str
        assert "state=disconnected" in repr_str
        assert "messages=0" in repr_str
        assert "gaps=0" in repr_str
    
    def test_state_property(self):
        """Test state property."""
        client = CoinbaseWebSocketClient()
        
        assert client.state == ConnectionState.DISCONNECTED
        
        client._state = ConnectionState.CONNECTING
        assert client.state == ConnectionState.CONNECTING
        assert not client.is_connected
        
        client._state = ConnectionState.SUBSCRIBED
        assert client.state == ConnectionState.SUBSCRIBED
        assert client.is_connected
    
    @pytest.mark.asyncio
    async def test_message_callbacks_invocation(self):
        """Test that callbacks can be invoked."""
        client = CoinbaseWebSocketClient()
        
        callback_invoked = []
        
        async def test_callback(msg_type, msg_data):
            callback_invoked.append((msg_type, msg_data))
        
        client.add_message_callback(test_callback)
        
        # Manually trigger callback
        await client._trigger_message_callbacks("test_type", {"data": "test"})
        
        assert len(callback_invoked) == 1
        assert callback_invoked[0][0] == "test_type"
        assert callback_invoked[0][1]["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_error_callbacks_invocation(self):
        """Test that error callbacks can be invoked."""
        client = CoinbaseWebSocketClient()
        
        error_invoked = []
        
        async def test_error_callback(error_type, error_msg):
            error_invoked.append((error_type, error_msg))
        
        client.add_error_callback(test_error_callback)
        
        # Manually trigger error callback
        await client._trigger_error_callbacks("test_error", "test message")
        
        assert len(error_invoked) == 1
        assert error_invoked[0][0] == "test_error"
        assert error_invoked[0][1] == "test message"
    
    @pytest.mark.asyncio
    async def test_sequence_gap_tracking(self):
        """Test sequence gap detection logic."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        # Set initial sequence
        client._last_sequence = 1000
        
        # Simulate gap detection in handle_message
        message_with_gap = {
            "type": "l2update",
            "sequence": 1005,  # Gap of 4
            "changes": []
        }
        
        # Process message
        await client._handle_message(message_with_gap, queue)
        
        # Check gap was detected
        assert client._sequence_gaps == 1
        assert client._last_sequence == 1005
    
    @pytest.mark.asyncio
    async def test_heartbeat_tracking(self):
        """Test heartbeat timestamp tracking."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        heartbeat_msg = {
            "type": "heartbeat",
            "sequence": 100
        }
        
        # Process heartbeat
        await client._handle_message(heartbeat_msg, queue)
        
        # Check heartbeat time was updated
        assert client._last_heartbeat_time is not None
        
        # Check stats include heartbeat
        stats = client.get_stats()
        assert "last_heartbeat" in stats
    
    @pytest.mark.asyncio
    async def test_message_enqueueing(self):
        """Test that messages are properly enqueued."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        test_msg = {
            "type": "snapshot",
            "sequence": 100,
            "bids": [],
            "asks": []
        }
        
        # Process message
        await client._handle_message(test_msg, queue)
        
        # Message should be in queue
        assert queue.qsize() == 1
        retrieved = await queue.get()
        assert retrieved["type"] == "snapshot"
        assert retrieved["sequence"] == 100
    
    @pytest.mark.asyncio
    async def test_subscription_message_not_enqueued(self):
        """Test that subscription confirmations are not enqueued."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        subscription_msg = {
            "type": "subscriptions",
            "channels": []
        }
        
        # Process subscription message
        await client._handle_message(subscription_msg, queue)
        
        # Should not be enqueued
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_error_message_not_enqueued(self):
        """Test that error messages are not enqueued."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        error_msg = {
            "type": "error",
            "message": "test error"
        }
        
        # Process error message
        await client._handle_message(error_msg, queue)
        
        # Should not be enqueued
        assert queue.empty()


# Integration test (optional - requires network)
class TestCoinbaseWebSocketClientIntegration:
    """
    Integration tests that connect to real Coinbase feed.
    
    These are marked as integration tests and can be run separately.
    They require network connectivity.
    """
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_connection(self):
        """Test real connection to Coinbase (requires network)."""
        client = CoinbaseWebSocketClient(
            product_ids=["BTC-USD"],
            channels=["heartbeat"]  # Minimal data
        )
        queue = AsyncMessageQueue()
        
        # Run for a short time
        task = asyncio.create_task(client.run(queue, auto_reconnect=False))
        
        # Wait for connection
        await asyncio.sleep(2.0)
        
        # Should have connected
        assert client.is_connected or client.state == ConnectionState.RECONNECTING
        
        # Cancel and cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        await client.disconnect()
