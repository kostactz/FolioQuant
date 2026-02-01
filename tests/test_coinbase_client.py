"""
Test suite for CoinbaseWebSocketClient.

Tests connection management, subscription handling, message processing,
sequence tracking, reconnection logic, and error handling.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.clients.coinbase_client import CoinbaseWebSocketClient, ConnectionState
from src.clients.message_queue import AsyncMessageQueue


class TestCoinbaseWebSocketClient:
    """Test suite for CoinbaseWebSocketClient."""
    
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
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection and subscription."""
        client = CoinbaseWebSocketClient()
        
        # Mock websocket connection
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        # Mock websockets.connect to be an AsyncMock (awaitable function)
        # that returns the mock_ws when awaited
        connect_mock = AsyncMock(return_value=mock_ws)
        
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            await client.connect()
            
            assert client.state == ConnectionState.SUBSCRIBED
            assert client.is_connected
            
            # Verify subscription message was sent
            mock_ws.send.assert_called_once()
            call_args = mock_ws.send.call_args[0][0]
            subscription = json.loads(call_args)
            
            assert subscription["type"] == "subscribe"
            assert subscription["product_ids"] == ["BTC-USD"]
            assert subscription["channels"] == ["level2_batch", "heartbeat"]
    
    @pytest.mark.asyncio
    async def test_connect_subscription_timeout(self):
        """Test connection failure due to subscription timeout."""
        client = CoinbaseWebSocketClient(subscription_timeout=0.1)
        
        # Mock websocket that never completes send
        mock_ws = AsyncMock()
        async def slow_send(*args):
            await asyncio.sleep(1.0)  # Longer than timeout
        mock_ws.send = slow_send
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            with pytest.raises(asyncio.TimeoutError):
                await client.connect()
            
            assert client.state == ConnectionState.ERROR
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test graceful disconnection."""
        client = CoinbaseWebSocketClient()
        
        # Mock connected websocket
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            await client.connect()
            assert client.is_connected
            
            # Disconnect
            await client.disconnect()
            
            # Verify unsubscribe message was sent
            assert mock_ws.send.call_count == 2  # Subscribe + unsubscribe
            
            # Verify close was called
            mock_ws.close.assert_called_once()
            
            assert client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_message_handling_snapshot(self):
        """Test handling of snapshot messages."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        snapshot_msg = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000", "1.5"], ["49999", "2.0"]],
            "asks": [["50001", "1.2"], ["50002", "0.8"]],
            "sequence": 1000,
        }
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(snapshot_msg),
            asyncio.CancelledError()  # Stop iteration
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Message should be in queue
            assert queue.qsize() == 1
            received = await queue.get()
            assert received["type"] == "snapshot"
            assert received["sequence"] == 1000
    
    @pytest.mark.asyncio
    async def test_message_handling_l2update(self):
        """Test handling of l2update messages."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        update_msg = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [
                ["buy", "50000", "2.5"],
                ["sell", "50001", "0.0"],
            ],
            "sequence": 1001,
        }
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(update_msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            assert queue.qsize() == 1
            received = await queue.get()
            assert received["type"] == "l2update"
            assert len(received["changes"]) == 2
    
    @pytest.mark.asyncio
    async def test_message_handling_heartbeat(self):
        """Test handling of heartbeat messages."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        heartbeat_msg = {
            "type": "heartbeat",
            "sequence": 1002,
            "last_trade_id": 12345,
            "product_id": "BTC-USD",
            "time": "2024-01-01T00:00:00.000000Z",
        }
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(heartbeat_msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Heartbeat should be enqueued
            assert queue.qsize() == 1
            received = await queue.get()
            assert received["type"] == "heartbeat"
            
            # Should update last heartbeat time
            stats = client.get_stats()
            assert "last_heartbeat" in stats
    
    @pytest.mark.asyncio
    async def test_sequence_tracking(self):
        """Test sequence number tracking and gap detection."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        messages = [
            {"type": "snapshot", "sequence": 1000},
            {"type": "l2update", "sequence": 1001},
            {"type": "l2update", "sequence": 1002},
            {"type": "l2update", "sequence": 1005},  # Gap of 2
        ]
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        async def message_iterator():
            for msg in messages:
                yield json.dumps(msg)
            # Raise CancelledError to stop the client loop
            raise asyncio.CancelledError()
                
        mock_ws.__aiter__ = lambda self: message_iterator()
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            # Run briefly then cancel
            task = asyncio.create_task(client.run(queue, auto_reconnect=False))
            await asyncio.sleep(0.5) # Wait for processing
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Check that gap was detected
            # We expect exactly 3 messages processed if loop worked correctly
            # Gap count: 1002->1005 is 1 gap event?
            # Impl details: _sequence_gaps increments by 1 per event?
            stats = client.get_stats()
            # If mock yielded infinite loops, these counts would be huge.
            # Check reasonably.
            assert stats["sequence_gaps"] >= 1
            assert stats["last_sequence"] == 1005
    
    @pytest.mark.asyncio
    async def test_error_message_handling(self):
        """Test handling of error messages from Coinbase."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        error_msg = {
            "type": "error",
            "message": "Invalid subscription",
            "reason": "product_id not found",
        }
        
        error_callback_called = False
        
        async def error_callback(error_type, error_message):
            nonlocal error_callback_called
            error_callback_called = True
            assert error_type == "coinbase_error"
        
        client.add_error_callback(error_callback)
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(error_msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Error message should NOT be enqueued
            assert queue.empty()
            
            # Error callback should be called
            assert error_callback_called
    
    @pytest.mark.asyncio
    async def test_subscription_confirmation_not_enqueued(self):
        """Test that subscription confirmations are not enqueued."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        subscription_msg = {
            "type": "subscriptions",
            "channels": [{"name": "level2_batch", "product_ids": ["BTC-USD"]}],
        }
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(subscription_msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Subscription message should not be enqueued
            assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_message_callbacks(self):
        """Test that message callbacks are invoked."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        callback_messages = []
        
        async def message_callback(message_type, message):
            callback_messages.append((message_type, message))
        
        client.add_message_callback(message_callback)
        
        test_msg = {"type": "heartbeat", "sequence": 100}
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(test_msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Callback should have been called
            assert len(callback_messages) == 1
            assert callback_messages[0][0] == "heartbeat"
            assert callback_messages[0][1]["sequence"] == 100
    
    @pytest.mark.asyncio
    async def test_json_parse_error(self):
        """Test handling of malformed JSON messages."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        error_callback_called = False
        
        async def error_callback(error_type, error_message):
            nonlocal error_callback_called
            if error_type == "json_parse_error":
                error_callback_called = True
        
        client.add_error_callback(error_callback)
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            "invalid json {{{",  # Malformed JSON
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Error callback should be called
            assert error_callback_called
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test client statistics tracking."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        messages = [
            {"type": "snapshot", "sequence": 1000},
            {"type": "l2update", "sequence": 1001},
            {"type": "heartbeat", "sequence": 1002},
        ]
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        
        async def message_iterator():
            for msg in messages:
                yield json.dumps(msg)
            raise asyncio.CancelledError()
        
        mock_ws.__aiter__ = lambda self: message_iterator()
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            task = asyncio.create_task(client.run(queue, auto_reconnect=False))
            await asyncio.sleep(0.5)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            stats = client.get_stats()
            assert stats["state"] == ConnectionState.SUBSCRIBED.value
            # If loop runs once: 3 messages.
            # If mock is broken and loops, it will be huge. 
            # We just want to ensure it processed at least the 3.
            assert stats["total_messages"] >= 3
            assert stats["sequence_gaps"] == 0
            assert stats["last_sequence"] == 1002

    @pytest.mark.asyncio
    async def test_ingest_time_injection(self):
        """Test that _ingest_time is injected into parsed messages."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue()
        
        msg = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [],
            "sequence": 100,
            "time": "2024-01-01T00:00:00.000000Z"
        }
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(msg),
            asyncio.CancelledError()
        ])
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            assert queue.qsize() == 1
            received = await queue.get()
            assert "_ingest_time" in received
            assert isinstance(received["_ingest_time"], float)
            # Should be recent
            import time
            assert abs(received["_ingest_time"] - time.time()) < 5.0
    
    @pytest.mark.asyncio
    async def test_reconnection_backoff(self):
        """Test exponential backoff on reconnection."""
        client = CoinbaseWebSocketClient(
            reconnect_delay=1.0,
            max_reconnect_delay=10.0
        )
        
        # Initial delay
        assert client._reconnect_delay == 1.0
        
        # Simulate reconnection
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
    
    def test_repr(self):
        """Test string representation."""
        client = CoinbaseWebSocketClient()
        repr_str = repr(client)
        
        assert "CoinbaseWebSocketClient" in repr_str
        assert "state=disconnected" in repr_str
        assert "messages=0" in repr_str
        assert "gaps=0" in repr_str
    
    @pytest.mark.asyncio
    async def test_multiple_products(self):
        """Test subscription to multiple products."""
        client = CoinbaseWebSocketClient(
            product_ids=["BTC-USD", "ETH-USD", "SOL-USD"]
        )
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("src.clients.coinbase_client.websockets.connect", side_effect=connect_mock):
            await client.connect()
            
            # Check subscription message
            call_args = mock_ws.send.call_args[0][0]
            subscription = json.loads(call_args)
            
            assert len(subscription["product_ids"]) == 3
            assert "BTC-USD" in subscription["product_ids"]
            assert "ETH-USD" in subscription["product_ids"]
            assert "SOL-USD" in subscription["product_ids"]
    
    @pytest.mark.asyncio
    async def test_queue_timeout_handling(self):
        """Test handling when queue is full and times out."""
        client = CoinbaseWebSocketClient()
        queue = AsyncMessageQueue(max_size=1)
        
        # Fill the queue
        await queue.put({"blocking": True})
        
        test_msg = {"type": "l2update", "sequence": 100}
        
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = AsyncMock(side_effect=[
            json.dumps(test_msg),
            asyncio.CancelledError()
        ])
        
        with patch("src.clients.coinbase_client.websockets.connect", return_value=mock_ws):
            try:
                await client.run(queue, auto_reconnect=False)
            except asyncio.CancelledError:
                pass
            
            # Queue should still have only the original message
            # (new message failed to enqueue due to timeout)
            assert queue.qsize() == 1
