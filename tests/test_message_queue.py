"""
Test suite for AsyncMessageQueue.

Tests the producer-consumer pattern implementation, queue operations,
backpressure handling, and statistics tracking.
"""

import asyncio
import pytest

from src.clients.message_queue import AsyncMessageQueue


class TestAsyncMessageQueue:
    """Test suite for AsyncMessageQueue."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test queue initialization with default and custom sizes."""
        # Default size
        queue = AsyncMessageQueue()
        assert queue.max_size == 10000
        assert queue.qsize() == 0
        assert queue.empty()
        assert not queue.full()
        
        # Custom size
        queue = AsyncMessageQueue(max_size=100)
        assert queue.max_size == 100
    
    @pytest.mark.asyncio
    async def test_put_get_basic(self):
        """Test basic put and get operations."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Put a message
        message = {"type": "test", "data": "hello"}
        result = await queue.put(message)
        assert result is True
        assert queue.qsize() == 1
        assert not queue.empty()
        
        # Get the message
        retrieved = await queue.get()
        assert retrieved == message
        assert queue.qsize() == 0
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_put_get_multiple(self):
        """Test multiple messages maintain FIFO order."""
        queue = AsyncMessageQueue(max_size=10)
        
        messages = [
            {"id": 1, "data": "first"},
            {"id": 2, "data": "second"},
            {"id": 3, "data": "third"},
        ]
        
        # Put all messages
        for msg in messages:
            await queue.put(msg)
        
        assert queue.qsize() == 3
        
        # Get all messages in order
        for expected_msg in messages:
            retrieved = await queue.get()
            assert retrieved == expected_msg
        
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_put_nowait_get_nowait(self):
        """Test non-blocking put and get operations."""
        queue = AsyncMessageQueue(max_size=5)
        
        # Put without blocking
        message = {"type": "test"}
        queue.put_nowait(message)
        assert queue.qsize() == 1
        
        # Get without blocking
        retrieved = queue.get_nowait()
        assert retrieved == message
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_put_nowait_full_queue(self):
        """Test that put_nowait raises QueueFull when queue is full."""
        queue = AsyncMessageQueue(max_size=2)
        
        # Fill the queue
        queue.put_nowait({"id": 1})
        queue.put_nowait({"id": 2})
        
        assert queue.full()
        
        # Try to add one more
        with pytest.raises(asyncio.QueueFull):
            queue.put_nowait({"id": 3})
    
    @pytest.mark.asyncio
    async def test_get_nowait_empty_queue(self):
        """Test that get_nowait raises QueueEmpty when queue is empty."""
        queue = AsyncMessageQueue(max_size=10)
        
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()
    
    @pytest.mark.asyncio
    async def test_put_with_timeout(self):
        """Test put operation with timeout on full queue."""
        queue = AsyncMessageQueue(max_size=1)
        
        # Fill the queue
        await queue.put({"id": 1})
        assert queue.full()
        
        # Try to put with short timeout (should fail)
        result = await queue.put({"id": 2}, timeout=0.1)
        assert result is False
        assert queue.qsize() == 1  # Original message still there
    
    @pytest.mark.asyncio
    async def test_get_with_timeout(self):
        """Test get operation with timeout on empty queue."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Try to get from empty queue with timeout
        with pytest.raises(asyncio.TimeoutError):
            await queue.get(timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test concurrent producer and consumer operations."""
        queue = AsyncMessageQueue(max_size=100)
        produced = []
        consumed = []
        
        async def producer():
            """Produce messages."""
            for i in range(10):
                message = {"id": i, "data": f"message_{i}"}
                await queue.put(message)
                produced.append(message)
                await asyncio.sleep(0.01)  # Simulate work
        
        async def consumer():
            """Consume messages."""
            for _ in range(10):
                message = await queue.get()
                consumed.append(message)
                await asyncio.sleep(0.01)  # Simulate processing
        
        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())
        
        # Verify all messages were consumed in order
        assert len(consumed) == 10
        assert consumed == produced
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_multiple_producers_single_consumer(self):
        """Test multiple producers with single consumer."""
        queue = AsyncMessageQueue(max_size=100)
        consumed = []
        
        async def producer(producer_id: int, count: int):
            """Produce messages with producer ID."""
            for i in range(count):
                await queue.put({"producer": producer_id, "seq": i})
                await asyncio.sleep(0.01)
        
        async def consumer(expected_count: int):
            """Consume all messages."""
            for _ in range(expected_count):
                message = await queue.get()
                consumed.append(message)
        
        # Run 3 producers and 1 consumer
        await asyncio.gather(
            producer(1, 5),
            producer(2, 5),
            producer(3, 5),
            consumer(15),
        )
        
        assert len(consumed) == 15
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the queue."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Add some messages
        for i in range(5):
            await queue.put({"id": i})
        
        assert queue.qsize() == 5
        
        # Clear the queue
        cleared_count = queue.clear()
        assert cleared_count == 5
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_drain(self):
        """Test draining all messages from queue."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Add messages
        messages = [{"id": i} for i in range(5)]
        for msg in messages:
            await queue.put(msg)
        
        # Drain the queue
        drained = await queue.drain()
        assert len(drained) == 5
        assert drained == messages
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test queue statistics tracking."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Initial stats
        stats = queue.get_stats()
        assert stats["current_size"] == 0
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.0
        assert stats["total_messages"] == 0
        assert stats["dropped_messages"] == 0
        
        # Add some messages
        for i in range(5):
            await queue.put({"id": i})
        
        stats = queue.get_stats()
        assert stats["current_size"] == 5
        assert stats["utilization"] == 0.5
        assert stats["total_messages"] == 5
        
        # Consume some messages
        await queue.get()
        await queue.get()
        
        stats = queue.get_stats()
        assert stats["current_size"] == 3
        assert stats["utilization"] == 0.3
        assert stats["total_messages"] == 5  # Total doesn't decrease
    
    @pytest.mark.asyncio
    async def test_dropped_messages_tracking(self):
        """Test that dropped messages are tracked."""
        queue = AsyncMessageQueue(max_size=1)
        
        # Fill the queue
        await queue.put({"id": 1})
        
        # Try to add with timeout (will fail)
        result = await queue.put({"id": 2}, timeout=0.1)
        assert result is False
        
        stats = queue.get_stats()
        assert stats["dropped_messages"] == 1
    
    @pytest.mark.asyncio
    async def test_queue_states(self):
        """Test empty and full state detection."""
        queue = AsyncMessageQueue(max_size=2)
        
        # Initially empty
        assert queue.empty()
        assert not queue.full()
        
        # Partially filled
        await queue.put({"id": 1})
        assert not queue.empty()
        assert not queue.full()
        
        # Full
        await queue.put({"id": 2})
        assert not queue.empty()
        assert queue.full()
        
        # Drain
        await queue.get()
        await queue.get()
        assert queue.empty()
        assert not queue.full()
    
    @pytest.mark.asyncio
    async def test_repr(self):
        """Test string representation."""
        queue = AsyncMessageQueue(max_size=10)
        await queue.put({"id": 1})
        
        repr_str = repr(queue)
        assert "AsyncMessageQueue" in repr_str
        assert "1/10" in repr_str
        assert "total=1" in repr_str
    
    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test that queue handles backpressure correctly."""
        queue = AsyncMessageQueue(max_size=5)
        
        # Fast producer, slow consumer
        produced_count = 0
        consumed_count = 0
        
        async def fast_producer():
            """Produce messages quickly."""
            nonlocal produced_count
            for i in range(10):
                await queue.put({"id": i})
                produced_count += 1
        
        async def slow_consumer():
            """Consume messages slowly."""
            nonlocal consumed_count
            for _ in range(10):
                await queue.get()
                consumed_count += 1
                await asyncio.sleep(0.05)  # Slow processing
        
        # Run concurrently
        await asyncio.gather(fast_producer(), slow_consumer())
        
        # All messages should be produced and consumed
        assert produced_count == 10
        assert consumed_count == 10
        assert queue.empty()
    
    @pytest.mark.asyncio
    async def test_message_types(self):
        """Test queue handles different message types."""
        queue = AsyncMessageQueue(max_size=10)
        
        # Test different types
        messages = [
            {"type": "dict", "value": 123},
            "string message",
            42,
            [1, 2, 3],
            ("tuple", "message"),
            None,
        ]
        
        for msg in messages:
            await queue.put(msg)
        
        for expected_msg in messages:
            retrieved = await queue.get()
            assert retrieved == expected_msg
    
    @pytest.mark.asyncio
    async def test_concurrent_put_get(self):
        """Test concurrent put and get operations don't corrupt queue."""
        queue = AsyncMessageQueue(max_size=100)
        
        async def putter(start: int, count: int):
            """Put numbered messages."""
            for i in range(start, start + count):
                await queue.put(i)
        
        async def getter(count: int) -> list:
            """Get messages."""
            results = []
            for _ in range(count):
                results.append(await queue.get())
            return results
        
        # Run multiple putters and getters
        putters = [putter(i * 10, 10) for i in range(5)]
        getters = [getter(10) for _ in range(5)]
        
        results = await asyncio.gather(*putters, *getters)
        
        # Extract getter results
        all_retrieved = []
        for result in results[5:]:  # Getters are after putters
            all_retrieved.extend(result)
        
        # Should have gotten all 50 messages
        assert len(all_retrieved) == 50
        assert queue.empty()
