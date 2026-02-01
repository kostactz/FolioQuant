"""
AsyncMessageQueue - Thread-safe asynchronous queue for producer-consumer pattern.

This module implements a bounded queue to decouple WebSocket message ingestion
(producer) from order book processing (consumer), ensuring that the WebSocket
connection remains responsive even when processing lags.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AsyncMessageQueue:
    """
    Asynchronous message queue implementing the producer-consumer pattern.
    
    The queue acts as a buffer between the WebSocket client (producer) and
    the order book processor (consumer), preventing backpressure on the
    network connection when processing is slower than message arrival.
    
    Attributes:
        max_size: Maximum number of messages the queue can hold.
        queue: The underlying asyncio.Queue instance.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the message queue.
        
        Args:
            max_size: Maximum queue size. When full, put() will block.
                     Default is 10,000 messages (~1-2 minutes at high frequency).
        """
        self.max_size = max_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._total_messages = 0
        self._dropped_messages = 0
        
        logger.debug(f"AsyncMessageQueue initialized with max_size={max_size}")
    
    async def put(self, message: Any, timeout: Optional[float] = None) -> bool:
        """
        Add a message to the queue (producer operation).
        
        This is called by the WebSocket client when a new message arrives.
        If the queue is full, this will wait (block) until space is available,
        preventing memory overflow but potentially slowing message ingestion.
        
        Args:
            message: The message to enqueue (typically a dict from JSON).
            timeout: Optional timeout in seconds. If specified and exceeded,
                    returns False without adding the message.
        
        Returns:
            True if message was successfully enqueued, False on timeout.
        
        Raises:
            asyncio.TimeoutError: If timeout is specified and exceeded.
        """
        try:
            if timeout:
                await asyncio.wait_for(self._queue.put(message), timeout=timeout)
            else:
                await self._queue.put(message)
            
            self._total_messages += 1
            
            # Log warnings if queue is filling up (potential backpressure)
            current_size = self._queue.qsize()
            if current_size > self.max_size * 0.8:
                logger.warning(
                    f"Queue is {current_size / self.max_size * 100:.1f}% full "
                    f"({current_size}/{self.max_size}). Consumer may be slow."
                )
            
            return True
            
        except asyncio.TimeoutError:
            self._dropped_messages += 1
            logger.error(
                f"Failed to enqueue message within {timeout}s timeout. "
                f"Queue size: {self._queue.qsize()}/{self.max_size}"
            )
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Any:
        """
        Retrieve a message from the queue (consumer operation).
        
        This is called by the order book processor. If the queue is empty,
        this will wait (block) until a message arrives.
        
        Args:
            timeout: Optional timeout in seconds. If specified and exceeded,
                    raises asyncio.TimeoutError.
        
        Returns:
            The next message from the queue.
        
        Raises:
            asyncio.TimeoutError: If timeout is specified and exceeded.
        """
        if timeout:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            return await self._queue.get()
    
    def get_nowait(self) -> Any:
        """
        Retrieve a message without blocking.
        
        Returns:
            The next message from the queue.
        
        Raises:
            asyncio.QueueEmpty: If the queue is empty.
        """
        return self._queue.get_nowait()
    
    def put_nowait(self, message: Any) -> None:
        """
        Add a message without blocking.
        
        Args:
            message: The message to enqueue.
        
        Raises:
            asyncio.QueueFull: If the queue is full.
        """
        self._queue.put_nowait(message)
        self._total_messages += 1
    
    def qsize(self) -> int:
        """
        Return the current number of messages in the queue.
        
        Note: This is an approximate size due to concurrent operations.
        """
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """
        Return True if the queue is empty.
        
        Note: Result may be stale immediately due to concurrent operations.
        """
        return self._queue.empty()
    
    def full(self) -> bool:
        """
        Return True if the queue is full.
        
        Note: Result may be stale immediately due to concurrent operations.
        """
        return self._queue.full()
    
    def clear(self) -> int:
        """
        Remove all messages from the queue.
        
        This is useful when resyncing the order book after a sequence gap,
        as old messages in the queue would be stale.
        
        Returns:
            Number of messages that were removed.
        """
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        
        if cleared > 0:
            logger.debug(f"Cleared {cleared} messages from queue")
        
        return cleared
    
    def get_stats(self) -> dict:
        """
        Get queue statistics for monitoring and debugging.
        
        Returns:
            Dict containing:
                - current_size: Number of messages currently in queue
                - max_size: Maximum queue capacity
                - utilization: Current fill percentage
                - total_messages: Total messages ever enqueued
                - dropped_messages: Messages dropped due to timeout
        """
        current_size = self._queue.qsize()
        return {
            "current_size": current_size,
            "max_size": self.max_size,
            "utilization": current_size / self.max_size if self.max_size > 0 else 0,
            "total_messages": self._total_messages,
            "dropped_messages": self._dropped_messages,
        }
    
    async def drain(self) -> list[Any]:
        """
        Retrieve all messages currently in the queue without blocking.
        
        This is useful for batch processing or graceful shutdown.
        
        Returns:
            List of all messages that were in the queue.
        """
        messages = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        logger.debug(f"Drained {len(messages)} messages from queue")
        return messages
    
    def task_done(self) -> None:
        """
        Indicate that a formerly enqueued task is complete.
        
        Used by queue consumers to signal that a message has been processed.
        This is a passthrough to the underlying asyncio.Queue method.
        """
        self._queue.task_done()
    
    async def join(self) -> None:
        """
        Block until all items in the queue have been processed.
        
        Used to wait for all enqueued messages to be consumed and processed.
        This is a passthrough to the underlying asyncio.Queue method.
        """
        await self._queue.join()
    
    def __repr__(self) -> str:
        """String representation showing queue state."""
        stats = self.get_stats()
        return (
            f"AsyncMessageQueue(size={stats['current_size']}/{stats['max_size']}, "
            f"utilization={stats['utilization']:.1%}, "
            f"total={stats['total_messages']}, "
            f"dropped={stats['dropped_messages']})"
        )
