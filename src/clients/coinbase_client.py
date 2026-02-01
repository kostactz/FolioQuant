"""
CoinbaseWebSocketClient - WebSocket connection manager for Coinbase Exchange.

This module handles the WebSocket lifecycle, subscription management, heartbeat
monitoring, and automatic reconnection with exponential backoff. It implements
the producer side of the producer-consumer pattern by feeding messages into
an AsyncMessageQueue.
"""

import asyncio
import json
import logging
import time
from typing import Optional, List, Callable
from enum import Enum

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from .message_queue import AsyncMessageQueue

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SUBSCRIBING = "subscribing"
    SUBSCRIBED = "subscribed"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class CoinbaseWebSocketClient:
    """
    Asynchronous WebSocket client for Coinbase Exchange market data.
    
    This client manages the connection to the Coinbase WebSocket feed,
    handles subscription to channels, tracks sequence numbers for gap
    detection, and implements automatic reconnection with exponential backoff.
    
    Key Features:
    - Automatic subscription within 5-second requirement
    - Heartbeat monitoring to detect stale connections
    - Sequence number tracking for data integrity
    - Exponential backoff reconnection strategy
    - Producer-consumer pattern via AsyncMessageQueue
    
    Attributes:
        ws_url: WebSocket endpoint URL
        product_ids: List of trading pairs to subscribe to
        channels: List of channels to subscribe to
        state: Current connection state
    """
    
    def __init__(
        self,
        ws_url: str = "wss://ws-feed.exchange.coinbase.com",
        product_ids: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        subscription_timeout: float = 5.0,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ping_interval: float = 30.0,
    ):
        """
        Initialize the Coinbase WebSocket client.
        
        Args:
            ws_url: WebSocket endpoint URL. Default is production feed.
            product_ids: List of product IDs (e.g., ["BTC-USD", "ETH-USD"]).
                        Defaults to ["BTC-USD"].
            channels: List of channels to subscribe to.
                     Defaults to ["level2_batch", "heartbeat"].
            subscription_timeout: Max seconds to send subscription after connect.
                                 Coinbase requires subscription within 5 seconds.
            reconnect_delay: Initial reconnect delay in seconds.
            max_reconnect_delay: Maximum reconnect delay (exponential backoff cap).
            ping_interval: Interval for connection health checks.
        """
        self.ws_url = ws_url
        self.product_ids = product_ids or ["BTC-USD"]
        self.channels = channels or ["level2_batch", "heartbeat"]
        self.subscription_timeout = subscription_timeout
        self.initial_reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_interval = ping_interval
        
        # Connection state
        self._ws: Optional[WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_delay = reconnect_delay
        self._reconnect_attempts = 0
        
        # Sequence tracking for gap detection
        self._last_sequence: Optional[int] = None
        self._sequence_gaps = 0
        
        # Message statistics
        self._total_messages = 0
        self._last_heartbeat_time: Optional[float] = None
        
        # Callbacks for application-level hooks
        self._on_message_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        
        logger.debug(
            f"CoinbaseWebSocketClient initialized: "
            f"url={ws_url}, products={product_ids}, channels={channels}"
        )
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """True if connected and subscribed."""
        return self._state == ConnectionState.SUBSCRIBED
    
    def add_message_callback(self, callback: Callable) -> None:
        """
        Add a callback to be invoked for each message.
        
        Args:
            callback: Async function called with (message_type, message_data).
        """
        self._on_message_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable) -> None:
        """
        Add a callback to be invoked on errors.
        
        Args:
            callback: Async function called with (error_type, error_message).
        """
        self._on_error_callbacks.append(callback)
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection and subscribe to channels.
        
        This method:
        1. Connects to the WebSocket endpoint
        2. Sends subscription message within the timeout
        3. Waits for subscription confirmation
        
        Raises:
            websockets.exceptions.WebSocketException: On connection failure
            asyncio.TimeoutError: If subscription times out
        """
        self._state = ConnectionState.CONNECTING
        logger.debug(f"Connecting to {self.ws_url}...")
        
        try:
            # Establish connection with ping to keep connection alive
            # Set max_size to None to allow large messages (snapshot can be very large)
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_interval * 2,
                max_size=None,  # Allow unlimited message size for snapshot messages
            )
            
            self._state = ConnectionState.CONNECTED
            logger.debug("WebSocket connection established")
            
            # Must subscribe within 5 seconds per Coinbase requirement
            await self._subscribe()
            
            self._state = ConnectionState.SUBSCRIBED
            self._reconnect_delay = self.initial_reconnect_delay
            self._reconnect_attempts = 0
            
            logger.info(f"Connected to Coinbase feed: {self.product_ids}")
            
        except Exception as e:
            self._state = ConnectionState.ERROR
            logger.error(f"Connection failed: {e}")
            raise
    
    async def _subscribe(self) -> None:
        """
        Send subscription message to Coinbase.
        
        The subscription must be sent within 5 seconds of connection
        establishment, otherwise Coinbase will close the connection.
        
        Raises:
            asyncio.TimeoutError: If subscription times out
        """
        self._state = ConnectionState.SUBSCRIBING
        
        subscription_message = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channels": self.channels,
        }
        
        logger.debug(f"Sending subscription: {subscription_message}")
        
        try:
            await asyncio.wait_for(
                self._ws.send(json.dumps(subscription_message)),
                timeout=self.subscription_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Failed to send subscription within {self.subscription_timeout}s"
            )
            raise
    
    async def disconnect(self) -> None:
        """
        Gracefully close the WebSocket connection.
        
        This sends an unsubscribe message before closing the connection.
        """
        if self._ws:
            try:
                # Send unsubscribe message
                unsubscribe_message = {
                    "type": "unsubscribe",
                    "product_ids": self.product_ids,
                    "channels": self.channels,
                }
                await self._ws.send(json.dumps(unsubscribe_message))
                
                # Close connection
                await self._ws.close()
                logger.info("WebSocket connection closed gracefully")
                
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            
            finally:
                self._ws = None
                self._state = ConnectionState.DISCONNECTED
    
    async def run(
        self,
        message_queue: AsyncMessageQueue,
        auto_reconnect: bool = True,
    ) -> None:
        """
        Main event loop: receive messages and enqueue them.
        
        This is the producer in the producer-consumer pattern. It continuously
        receives messages from the WebSocket and puts them into the queue for
        processing by the consumer.
        
        Args:
            message_queue: Queue to send received messages to.
            auto_reconnect: If True, automatically reconnect on disconnect.
        
        The loop runs indefinitely until explicitly stopped. It handles:
        - Message reception and enqueueing
        - Sequence number tracking
        - Automatic reconnection on failure
        """
        while True:
            try:
                # Connect if not connected
                if not self.is_connected:
                    await self.connect()
                
                # Receive and process messages
                async for message_str in self._ws:
                    try:
                        message = json.loads(message_str)
                        await self._handle_message(message, message_queue)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}")
                        await self._trigger_error_callbacks("json_parse_error", str(e))
                    
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        await self._trigger_error_callbacks("message_handler_error", str(e))
            
            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                
                if not auto_reconnect:
                    logger.info("Auto-reconnect disabled, stopping")
                    break
                
                # Reconnect with exponential backoff
                await self._reconnect()
            
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                
                if not auto_reconnect:
                    break
                
                await self._reconnect()
            
            except Exception as e:
                logger.error(f"Unexpected error in event loop: {e}")
                
                if not auto_reconnect:
                    break
                
                await self._reconnect()
    
    async def _handle_message(
        self,
        message: dict,
        message_queue: AsyncMessageQueue,
    ) -> None:
        """
        Process a received message.
        
        This method:
        1. Tracks message statistics
        2. Checks sequence numbers for gaps
        3. Enqueues the message for processing
        4. Triggers application callbacks
        
        Args:
            message: Parsed JSON message from Coinbase
            message_queue: Queue to send message to
        """
        # Inject ingestion timestamp immediately
        message["_ingest_time"] = time.time()
        
        message_type = message.get("type", "unknown")
        
        # Track statistics
        self._total_messages += 1
        
        # Handle different message types
        if message_type == "heartbeat":
            self._last_heartbeat_time = time.time()
            logger.debug(f"Heartbeat received: sequence={message.get('sequence')}")
        
        elif message_type == "subscriptions":
            # Subscription confirmed - don't log or enqueue
            return  # Don't enqueue subscription confirmations
        
        elif message_type == "error":
            logger.error(f"Received error message: {message}")
            await self._trigger_error_callbacks("coinbase_error", message.get("message", ""))
            return  # Don't enqueue error messages
        
        # Sequence number tracking (for snapshot and l2update)
        if "sequence" in message:
            sequence = message["sequence"]
            
            if self._last_sequence is not None:
                expected = self._last_sequence + 1
                
                if sequence != expected:
                    gap = sequence - expected
                    self._sequence_gaps += 1
                    logger.debug(
                        f"Sequence gap detected: expected {expected}, got {sequence} "
                        f"(gap of {gap} messages). Total gaps: {self._sequence_gaps}"
                    )
                    await self._trigger_error_callbacks("sequence_gap", f"Gap: {gap}")
            
            self._last_sequence = sequence
        
        # Enqueue message for processing
        try:
            await message_queue.put(message, timeout=1.0)
        except asyncio.TimeoutError:
            logger.error(f"Failed to enqueue {message_type} message (queue full)")
        
        # Trigger application callbacks
        await self._trigger_message_callbacks(message_type, message)
    
    async def _reconnect(self) -> None:
        """
        Reconnect with exponential backoff.
        
        The delay doubles on each attempt until max_reconnect_delay is reached.
        """
        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        
        logger.info(
            f"Reconnecting in {self._reconnect_delay:.1f}s "
            f"(attempt {self._reconnect_attempts})..."
        )
        
        await asyncio.sleep(self._reconnect_delay)
        
        # Exponential backoff
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self.max_reconnect_delay
        )
        
        # Reset sequence tracking on reconnect
        # (we'll get a new snapshot)
        self._last_sequence = None
    
    async def _trigger_message_callbacks(
        self,
        message_type: str,
        message: dict,
    ) -> None:
        """Invoke all registered message callbacks."""
        for callback in self._on_message_callbacks:
            try:
                await callback(message_type, message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
    
    async def _trigger_error_callbacks(
        self,
        error_type: str,
        error_message: str,
    ) -> None:
        """Invoke all registered error callbacks."""
        for callback in self._on_error_callbacks:
            try:
                await callback(error_type, error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def get_stats(self) -> dict:
        """
        Get connection and message statistics.
        
        Returns:
            Dict containing:
                - state: Current connection state
                - total_messages: Total messages received
                - sequence_gaps: Number of sequence gaps detected
                - last_sequence: Last sequence number received
                - reconnect_attempts: Number of reconnection attempts
                - last_heartbeat: Seconds since last heartbeat
        """
        stats = {
            "state": self._state.value,
            "total_messages": self._total_messages,
            "sequence_gaps": self._sequence_gaps,
            "last_sequence": self._last_sequence,
            "reconnect_attempts": self._reconnect_attempts,
        }
        
        if self._last_heartbeat_time:
            stats["last_heartbeat"] = time.time() - self._last_heartbeat_time
        
        return stats
    
    def __repr__(self) -> str:
        """String representation showing connection state."""
        stats = self.get_stats()
        return (
            f"CoinbaseWebSocketClient(state={stats['state']}, "
            f"messages={stats['total_messages']}, "
            f"gaps={stats['sequence_gaps']})"
        )
