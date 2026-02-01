"""
Main Dash application entry point.

This module creates and runs the Plotly Dash application for FolioQuant.
It integrates:
    - Dash web server (Flask-based)
    - Background WebSocket thread (asyncio)
    - Global state management
    - Real-time callbacks

Architecture:
    1. Background Thread: Runs asyncio event loop for WebSocket
    2. Main Thread: Runs Flask/Dash server
    3. Shared State: Global DashboardState object
    4. Callbacks: Read from state, update UI components

Usage:
    python src/app/dash_app.py --product BTC-USD --ofi-window 100 --port 8501
"""

import numpy
import time
import asyncio
import threading
import argparse
import logging
from datetime import datetime
import subprocess
import webbrowser
import socket

def find_free_port(start_port, max_attempts=100):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise OSError(f"No free ports found starting from {start_port}")

def open_browser(host, port):
    """
    Open the dashboard in a browser popup.
    """
    # Wait for server to start
    time.sleep(1.5)
    
    # Convert 0.0.0.0 to localhost for browser
    if host == "0.0.0.0":
        host = "127.0.0.1"
        
    url = f"http://{host}:{port}/"
    
    try:
        # Try to open in Chrome in app mode (popup window)
        subprocess.Popen(['google-chrome', f'--app={url}'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # Fallback to default browser
        webbrowser.open(url)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to open browser: {e}")

from dash import Dash
import dash_bootstrap_components as dbc

from .dash_layout import create_layout
from .dash_state import state
from .dash_callbacks import register_callbacks

from ..clients.coinbase_client import CoinbaseWebSocketClient
from ..clients.message_queue import AsyncMessageQueue
from ..services.book_manager import BookManager
from ..services.ofi_calculator import OFICalculator
from ..services.metrics_service import MetricsService
from ..utils.time_utils import calculate_latency


# Configure logging
# Set root logger to WARNING to reduce noise
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure specific loggers BEFORE importing Dash
# Suppress werkzeug HTTP request logs (Flask development server)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Suppress dash request logs and startup messages
logging.getLogger('dash').setLevel(logging.ERROR)
logging.getLogger('dash.dash').setLevel(logging.ERROR)

# Configure specific loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep our app logs at INFO

# Keep websocket client at INFO for connection issues
logging.getLogger('src.clients.coinbase_client').setLevel(logging.INFO)

# Keep book manager at WARNING (only show issues)
logging.getLogger('src.services.book_manager').setLevel(logging.WARNING)

# Keep message queue quiet
logging.getLogger('src.clients.message_queue').setLevel(logging.WARNING)

# Keep OFI and metrics services quiet unless there's an issue
logging.getLogger('src.services.ofi_calculator').setLevel(logging.WARNING)
logging.getLogger('src.services.metrics_service').setLevel(logging.WARNING)


# Global connection management
_event_loop = None
_connection_task = None
_event_loop_lock = threading.Lock()


# ============================================================================
# WebSocket Background Thread
# ============================================================================

def run_async_loop(loop: asyncio.AbstractEventLoop):
    """
    Run asyncio event loop in background thread.
    
    This thread runs independently of the Flask server and handles
    all WebSocket communication and message processing.
    
    Args:
        loop: Asyncio event loop to run
    """
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def process_websocket_messages(
    book_manager: BookManager,
    ofi_calculator: OFICalculator,
    metrics_service: MetricsService,
    message_queue: AsyncMessageQueue
):
    """
    Process WebSocket messages and update global state.
    
    This is the core data processing loop that:
        1. Receives messages from the queue
        2. Processes them through BookManager
        3. Calculates OFI signals
        4. Updates performance metrics
        5. Updates global state for callbacks to read
    
    Args:
        book_manager: Order book reconstruction service
        ofi_calculator: OFI signal calculator
        metrics_service: Performance metrics calculator
        message_queue: WebSocket message queue
    """
    
    # Subscribe to book manager events to trigger OFI calculation
    async def on_book_update(event: dict):
        """Callback for book updates - triggers OFI calculation."""
        event_type = event.get("type")
        
        if event_type == "l2update":
            # Calculate OFI for this update
            await ofi_calculator.on_book_update(event)
            signal = ofi_calculator.get_current_signal()
            
            if signal:
                # Update global state with new OFI signal
                state.ofi_history.append({
                    'timestamp': signal.timestamp,
                    'ofi': float(signal.ofi_value),
                    'mid_price': float(signal.mid_price)
                })
                
                # Calculate performance metrics
                # Always update history
                await metrics_service.on_signal_update(signal)
                
                # Throttle expensive snapshot calculation and state updates
                nonlocal last_metrics_calc
                current_time = time.time()
                if current_time - last_metrics_calc > METRICS_CALC_INTERVAL:
                    snapshot = metrics_service.get_metrics_snapshot()
                    last_metrics_calc = current_time
                else:
                    snapshot = None
                
                if snapshot:
                    state.sharpe_ratio = float(snapshot.sharpe_ratio) if snapshot.sharpe_ratio else None
                    state.hit_rate = float(snapshot.hit_rate) if snapshot.hit_rate else None
                    state.max_drawdown = float(snapshot.max_drawdown) if snapshot.max_drawdown else None
                    state.win_loss_ratio = float(snapshot.win_loss_ratio) if snapshot.win_loss_ratio else None
                    state.total_predictions = snapshot.total_predictions
                    state.price_correlation = float(snapshot.price_correlation) if snapshot.price_correlation else None
                    state.information_coefficient = float(snapshot.information_coefficient) if snapshot.information_coefficient else None
                    state.information_coefficient = float(snapshot.information_coefficient) if snapshot.information_coefficient else None
                    state.alpha_decay = snapshot.alpha_decay or {}

                    # Append to metrics history for charting
                    state.metrics_history.append({
                        'timestamp': signal.timestamp,
                        'sharpe_ratio': state.sharpe_ratio,
                        'sharpe_ratio': state.sharpe_ratio,
                        'hit_rate': state.hit_rate
                    })
                    
                    # Update diagnostics
                    state.scatter_data = snapshot.scatter_data or []
                    if snapshot.rolling_volatility is not None:
                        state.volatility_history.append({
                            'timestamp': signal.timestamp,
                            'volatility': snapshot.rolling_volatility
                        })
                    
                    # Update trades
                    state.recent_trades = snapshot.recent_trades or []

                # Calculate Execution Metrics (Slippage for 1.0 BTC)
                if state.book_initialized:
                    size = 1.0
                    state.slippage_buy = book_manager.calculate_slippage(size, 'buy')
                    state.slippage_sell = book_manager.calculate_slippage(size, 'sell')
        
        elif event_type == "snapshot":
            logger.info("[PROCESSOR] Order book initialized from snapshot")
    
    # Register callback
    book_manager.subscribe(on_book_update)
    
    # Performance throttling state
    last_metrics_calc = 0.0
    METRICS_CALC_INTERVAL = 0.5  # Calculate expensive metrics at most every 500ms
    
    # Logging state
    last_queue_log = 0.0
    QUEUE_LOG_INTERVAL = 5.0
    
    last_latency_log = 0.0
    LATENCY_LOG_INTERVAL = 5.0

    # Main processing loop
    message_times = []  # Track last 100 message timestamps for throughput
    latency_samples = []  # Track last 100 latencies for average
    high_latency_threshold = 100  # Warn if latency exceeds 100ms
    
    while True:
        try:
            # Monitor queue processing lag
            current_time = time.time()
            if current_time - last_queue_log > QUEUE_LOG_INTERVAL:
                qsize = message_queue.qsize()
                if qsize > 100:
                    logger.warning(f"[PERFORMANCE] Queue backlog: {qsize} messages")
                else:
                    logger.debug(f"[PERFORMANCE] Queue size: {qsize}")
                last_queue_log = current_time

            # Wait for next message (with timeout to allow graceful shutdown)
            try:
                start_time = datetime.now()
                message = await asyncio.wait_for(
                    message_queue.get(),
                    timeout=5.0
                )
                receive_time = datetime.now()
            except asyncio.TimeoutError:
                # Don't log timeout - this is normal when idle
                continue
            
            state.message_count += 1
            msg_type = message.get("type")
            
            # Calculate message latency using shared logic
            process_ts = start_time.timestamp()
            total_latency, network_latency, system_latency = calculate_latency(message, process_ts)

            state.message_latency_ms = total_latency
            
            # Update rolling averages
            latency_samples.append((total_latency, network_latency, system_latency))
            if len(latency_samples) > 100:
                latency_samples.pop(0)
            
            # Calculate averages
            avgs = [sum(x) / len(latency_samples) for x in zip(*latency_samples)]
            state.avg_latency_ms = avgs[0]
            state.avg_network_latency = avgs[1]
            state.avg_system_latency = avgs[2]
            
            # Warn on high latency
            # Warn on high latency (rate limited)
            current_loop_time = time.time()
            if total_latency > high_latency_threshold:
                if current_loop_time - last_latency_log > LATENCY_LOG_INTERVAL:
                    logger.warning(f"[PROCESSOR] High latency: Total={total_latency:.1f}ms (Net={network_latency:.1f}ms, Sys={system_latency:.1f}ms)")
                    last_latency_log = current_loop_time
            
            # Calculate throughput (messages per second)
            message_times.append(receive_time)
            if len(message_times) > 100:
                message_times.pop(0)
            
            if len(message_times) >= 2:
                time_span = (message_times[-1] - message_times[0]).total_seconds()
                if time_span > 0:
                    state.messages_per_second = (len(message_times) - 1) / time_span
            
            state.last_message_time = receive_time
            
            # Process message through BookManager
            success = await book_manager.process_message(message)
            
            if success:
                # Update global state from order book
                state.book = book_manager.book
                
                if msg_type == "snapshot":
                    state.book_initialized = True
                    state.connected = True
                    _update_book_state()
                    logger.info(f"[PROCESSOR] Book initialized: {len(state.book.bids)} bids, {len(state.book.asks)} asks")
                
                elif msg_type == "l2update" and state.book_initialized:
                    _update_book_state()
                
                elif msg_type == "heartbeat":
                    from ..models.market_data import HeartbeatMessage
                    heartbeat = HeartbeatMessage.from_dict(message)
                    state.last_heartbeat = heartbeat.time
            
            else:
                # Only warn once when receiving updates before snapshot
                if msg_type == "l2update" and not state.book_initialized:
                    if state.message_count <= 5:  # Only warn for first few messages
                        logger.warning("[PROCESSOR] Waiting for snapshot before processing updates")
            
            # Broadcast updates (rate limited to ~10Hz to avoid CPU spike)
            # We use a simple time check. 
            # Note: broadcast_metrics is defined below, we need to make sure it's available.
            # Since Python functions are objects, if we define broadcast_metrics before this function runs (it is async),
            # it should be fine if it's in the global scope. 
            # However, process_websocket_messages is defined ABOVE broadcast_metrics in the file (currently).
            # We need to move broadcast_metrics UP or define it before use.
            # Actually, I added broadcast_metrics at the END of the file in the previous step (around line 456).
            # This works if process_websocket_messages uses it at RUNTIME.
            current_time = time.time()
            if getattr(state, 'last_broadcast_time', 0) + 0.1 < current_time:
                 await broadcast_metrics()
                 state.last_broadcast_time = current_time
        
        except Exception as e:
            logger.error(f"[PROCESSOR] Error processing message: {e}", exc_info=True)


def _update_book_state():
    """
    Update global state from current order book.
    
    Extracts relevant data from OrderBook instance and updates
    the global state dict for callbacks to read.
    """
    if not state.book:
        return
    
    # Update BBO
    best_bid_tuple = state.book.best_bid
    best_ask_tuple = state.book.best_ask
    
    if best_bid_tuple:
        state.best_bid = (float(best_bid_tuple[0]), float(best_bid_tuple[1]))
    
    if best_ask_tuple:
        state.best_ask = (float(best_ask_tuple[0]), float(best_ask_tuple[1]))
    
    # Update derived metrics
    state.mid_price = float(state.book.mid_price) if state.book.mid_price else None
    state.spread = float(state.book.spread) if state.book.spread else None
    state.micro_price = float(state.book.micro_price) if state.book.micro_price else None
    
    # Update depth for visualization (use dynamic book_depth setting)
    # Bids: descending price, cumulative from best bid
    bid_cumulative = []
    cumsum = 0
    for price, size in list(state.book.bids.items())[:state.book_depth]:
        cumsum += float(size)
        bid_cumulative.append((float(price), cumsum))
    state.bid_depth = bid_cumulative
    
    # Asks: ascending price, cumulative from best ask
    ask_cumulative = []
    cumsum = 0
    for price, size in list(state.book.asks.items())[:state.book_depth]:
        cumsum += float(size)
        ask_cumulative.append((float(price), cumsum))
    state.ask_depth = ask_cumulative


async def start_websocket_connection(product_id: str, ofi_window: int, port: int = 5000):
    """
    Start WebSocket connection and processing pipeline.
    
    This coroutine:
        1. Creates services (BookManager, OFI, Metrics)
        2. Creates WebSocket client
        3. Starts message processing
        4. Runs until connection is closed
    
    Args:
    Args:
        product_id: Trading pair (e.g., "BTC-USD")
        ofi_window: OFI calculation window size
        port: WebSocket server port
    """
    
    # Create message queue
    message_queue = AsyncMessageQueue(max_size=1000)
    
    # Create services
    book_manager = BookManager(product_id=product_id)
    ofi_calculator = OFICalculator(window_size=ofi_window)
    metrics_service = MetricsService(
        window_size=1000,
        use_dynamic_scaling=False,  # Fixed annualization for sanity
        trading_fee_bps=1.0,  # 1 bps trading fee to enforce realism
        signal_threshold=state.signal_threshold  # Use dynamic threshold
    )
    state.metrics_service = metrics_service
    
    # Create WebSocket client
    ws_client = CoinbaseWebSocketClient(
        product_ids=[product_id],
        channels=["level2_batch", "heartbeat"]
    )
    
    ws_server = None
    try:
        # Mark as connected (UI will show "connecting")
        state.connected = True
        
        # Create tasks
        ws_task = asyncio.create_task(
            ws_client.run(message_queue, auto_reconnect=True)
        )
        
        process_task = asyncio.create_task(
            process_websocket_messages(
                book_manager, ofi_calculator, metrics_service, message_queue
            )
        )
        
        logger.info(f"[CONNECTION] Connected to {product_id}")
        
        # Start WebSocket Server
        # Start WebSocket Server
        # Enable address and port reuse to handle restarts gracefully
        serve_kwargs = {"reuse_address": True}
        if hasattr(socket, "SO_REUSEPORT"):
            serve_kwargs["reuse_port"] = True
            
        ws_server = await websockets.serve(ws_handler, "0.0.0.0", port, **serve_kwargs)
        logger.info(f"[WS] Server started on port {port}")
        
        # Wait for tasks (runs indefinitely)
        await asyncio.gather(ws_task, process_task)
    
    except asyncio.CancelledError:
        logger.info("[CONNECTION] Connection cancelled")
        raise
    
    except Exception as e:
        logger.error(f"[CONNECTION] Error: {e}", exc_info=True)
        state.error_message = str(e)
        state.connected = False
    
    finally:
        if ws_server:
            ws_server.close()
            await ws_server.wait_closed()
            logger.info("[WS] Server closed")
            
        state.connected = False
        logger.info("[CONNECTION] Cleanup complete")


def start_background_websocket(product_id: str = "BTC-USD", ofi_window: int = 100, port: int = 5000):
    """
    Start WebSocket connection in background thread.
    
    This creates a new asyncio event loop in a daemon thread and
    schedules the WebSocket connection coroutine.
    
    Args:
        product_id: Trading pair to subscribe to
        ofi_window: OFI calculation window size
        port: WebSocket server port
    """
    global _event_loop, _connection_task
    
    with _event_loop_lock:
        # Create event loop if it doesn't exist
        if _event_loop is None:
            _event_loop = asyncio.new_event_loop()
            
            # Start thread to run the loop
            thread = threading.Thread(
                target=run_async_loop,
                args=(_event_loop,),
                daemon=True,
                name="WebSocketThread"
            )
            thread.start()
        
        # Cancel existing connection if any
        if _connection_task is not None:
            try:
                _connection_task.cancel()
                logger.info("[MAIN] Cancelled previous connection")
            except Exception as e:
                logger.warning(f"[MAIN] Error cancelling previous connection: {e}")
        
        # Schedule new WebSocket connection
        _connection_task = asyncio.run_coroutine_threadsafe(
            start_websocket_connection(product_id, ofi_window, port),
            _event_loop
        )


def stop_background_websocket():
    """
    Stop WebSocket connection.
    
    Cancels the current connection task and resets state.
    """
    global _connection_task
    
    with _event_loop_lock:
        if _connection_task is not None:
            try:
                _connection_task.cancel()
                logger.info("[MAIN] Connection stopped")
                state.connected = False
                state.error_message = None
            except Exception as e:
                logger.error(f"[MAIN] Error stopping connection: {e}")
                state.error_message = f"Stop error: {str(e)}"
        else:
            logger.warning("[MAIN] No active connection to stop")


# ============================================================================
# Dash App Creation
# ============================================================================

def create_app(ws_port=5000):
    """
    Create and configure Dash application.
    
    Returns:
        Dash app instance
    """
    
    # Suppress Dash's logging before creating the app
    logging.getLogger('dash').setLevel(logging.ERROR)
    logging.getLogger('dash.dash').setLevel(logging.ERROR)
    
    # Create Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.FLATLY,  # Modern light theme
            'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css'
        ],
        suppress_callback_exceptions=True,
        title="FolioQuant - OFI Analytics",
        # Add cache-busting to force browser to reload JavaScript
        update_title=None,  # Prevent title updates
        serve_locally=True  # Ensure assets are served fresh
    )
    
    # Disable client-side caching
    app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
    # Set layout FIRST (from dash_layout.py)
    # Set layout FIRST (from dash_layout.py)
    # Use wrapper function to ensure visibility of layout generation
    def layout_wrapper():
        print("DEBUG: [APP] layout_wrapper called. Regenerating layout...")
        return create_layout(ws_port=ws_port, state=state)
        
    app.layout = layout_wrapper
    
    # Register callbacks AFTER layout to ensure components exist
    register_callbacks(app)
    
    return app


# ============================================================================
# WebSocket Server & Broadcast Logic
# ============================================================================

import json
import websockets
from .dash_callbacks import update_fast_metrics, update_slow_metrics, update_ofi_chart_data, update_metrics_chart_data

connected_clients = set()

async def ws_handler(websocket):
    """Handle new WebSocket connections."""
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)

async def broadcast_metrics():
    """Broadcast metrics to all connected clients."""
    if not connected_clients:
        return

    # Check if streaming is enabled
    if not state.streaming_enabled:
        return
        
    # Generate metrics using existing callback logic
    # We strip the Component objects to JSON-compatible dictionaries/strings
    # using Dash's internal serialization if needed, but for simple components 
    # (strings, numbers, basic html), Python dicts/strings usually work.
    # However, update_fast_metrics returns Dash Components.
    # We need to serialize them.
    from plotly.utils import PlotlyJSONEncoder
    
    try:
        # Get latest metrics
        fast_metrics = update_fast_metrics(0)
        slow_metrics = update_slow_metrics(0)
        ofi_metrics = update_ofi_chart_data(0)
        metrics_chart = update_metrics_chart_data(0)
        
        # New: Trades chart
        from .dash_callbacks import update_execution_chart_data
        trades_chart = update_execution_chart_data(0)
        
        payload = {
            "type": "metrics_update",
            "fast": fast_metrics,
            "slow": slow_metrics,
            "ofi": ofi_metrics,
            "metrics": metrics_chart,
            "trades": trades_chart
        }
        
        # Serialize with Plotly encoder to handle Dash components/Figures
        message = json.dumps(payload, cls=PlotlyJSONEncoder)
        
        # Broadcast
        websockets_to_remove = set()
        for ws in connected_clients.copy():
            try:
                await ws.send(message)
            except Exception:
                websockets_to_remove.add(ws)
        
        for ws in websockets_to_remove:
            connected_clients.discard(ws)
            
    except Exception as e:
        logger.error(f"[WS] Broadcast error: {e}")

# Modify process_websocket_messages to broadcast updates
# We'll patch this into the existing function or simpler: 
# We'll run a separate broadcast loop or trigger it on data update.
# Triggering on data update is better for latency.



# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FolioQuant - Real-Time OFI Analytics Dashboard"
    )
    
    parser.add_argument(
        '--product',
        type=str,
        default='BTC-USD',
        help='Product ID (default: BTC-USD)'
    )
    
    parser.add_argument(
        '--ofi-window',
        type=int,
        default=100,
        help='OFI calculation window size (default: 100)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Server port (default: 8501)'
    )
    
    parser.add_argument(
        '--ws-port',
        type=int,
        default=5000,
        help='WebSocket port (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    # Parse arguments
    args = parse_args()
    
    # Initialize state configuration from command-line args
    state.product_id = args.product
    state.ofi_window = args.ofi_window
    
    
    # Find free WebSocket port
    ws_port = find_free_port(args.ws_port)
    
    # Print banner
    print("=" * 80)
    print("FolioQuant - Real-Time Order Flow Imbalance Analytics")
    print("Powered by Plotly Dash")
    print("=" * 80)
    print(f"Product: {args.product}")
    print(f"OFI Window: {args.ofi_window}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"WebSocket: ws://{args.host}:{ws_port}")
    print("=" * 80)
    
    # Start WebSocket background thread
    # Guard against double-execution when using Dash/Flask reloader (debug=True)
    # The reloader spawns a child process. We want to run ONLY in the child process
    # (where WERKZEUG_RUN_MAIN is set) or if debug is disabled (single process).
    import os
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_background_websocket(
            product_id=state.product_id,
            ofi_window=state.ofi_window,
            port=ws_port
        )
    
    # Create Dash app
    app = create_app(ws_port=ws_port)
    
    print(f"\n🚀 Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    # Suppress Flask's startup banner
    import sys
    cli = sys.modules.get('flask.cli')
    if cli:
        cli.show_server_banner = lambda *x: None
    
    # Auto-open browser in a separate thread
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(
            target=open_browser,
            args=(args.host, args.port),
            daemon=True
        ).start()

    app.run_server(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
