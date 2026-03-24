"""
Dash layout components.

This module defines the HTML/CSS structure of the dashboard using Dash's
declarative layout system. Unlike Streamlit's procedural API, Dash uses
a component-based architecture similar to React.

Key Components:
    - Header: Title, connection status, current price
    - Metrics Panel: Best bid/ask, spread, micro-price, book imbalance
    - Depth Chart: Order book visualization
    - OFI Chart: Streaming OFI signal with dual-axis
    - Performance Metrics: Sharpe, hit rate, drawdown
    - Control Panel: Configuration and connection controls
"""

from dash import html, dcc
from dash_extensions import WebSocket
import dash_bootstrap_components as dbc
from dash import dash_table
import plotly.graph_objects as go


def _help_icon(tooltip_text):
    """Create a consistent hoverable question-mark help icon."""
    return html.I(
        className="bi bi-question-circle tooltip-icon ms-1",
        title=tooltip_text,
        **{"aria-label": "Help"}
    )

def create_header():
    """
    Create dashboard header with title and connection status.
    
    Returns:
        dbc.Row component with header content
    """
    return html.Div([
        # Error message alert (dismissible)
        dbc.Alert(
            id='error-alert',
            color='danger',
            is_open=False,
            dismissable=True,
            className='mb-3'
        ),
        
        dbc.Row([
            dbc.Col([
                html.H1("FolioQuant", className="text-primary mb-1"),
                html.P("Real-Time Order Flow Imbalance Analytics", 
                       className="text-muted")
            ], width=6),
            
            dbc.Col([
                html.Div(id='connection-status', className="text-end mb-2"),
                html.Div(id='current-price', className="text-end h3 text-info")
            ], width=6)
        ], className="mb-4")
    ])


def create_metrics_panel():
    """
    Create market metrics panel with real-time BBO data.
    
    Displays:
        - Best bid price
        - Best ask price
        - Bid-ask spread
        - Micro-price (volume-weighted)
        - Book imbalance progress bar
    
    Returns:
        dbc.Card component with metrics
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "Market Metrics",
            _help_icon("Live top-of-book values. Compare these to gauge liquidity and short-term pressure.")
        ], className="mb-0")),
        dbc.CardBody([
            # Stack metrics vertically for sidebar
            html.Div([
                html.H6([
                    "Best Bid",
                    _help_icon("Highest current buy price. Higher bid often signals stronger demand.")
                ], className="text-muted mb-2"),
                html.Div(id='metric-bid', className="h4 text-success mb-3 font-monospace")
            ], className="metric-card mb-3"),
            
            html.Div([
                html.H6([
                    "Best Ask",
                    _help_icon("Lowest current sell price. Lower ask can indicate stronger selling pressure.")
                ], className="text-muted mb-2"),
                html.Div(id='metric-ask', className="h4 text-danger mb-3 font-monospace")
            ], className="metric-card mb-3"),
            
            html.Div([
                html.H6([
                    "Spread",
                    _help_icon("Ask minus bid. Smaller spread means tighter market and usually better liquidity.")
                ], className="text-muted mb-2"),
                html.Div(id='metric-spread', className="h4 text-warning mb-3 font-monospace")
            ], className="metric-card mb-3"),
            
            html.Div([
                html.H6([
                    "Micro-Price",
                    _help_icon("Fair-value estimate weighted by top bid/ask sizes. Above mid can hint buy pressure.")
                ], className="text-muted mb-2"),
                html.Div(id='metric-micro', className="h4 text-info mb-3 font-monospace")
            ], className="metric-card mb-3"),
            
            # Book imbalance progress bar
            html.Hr(className="my-3", style={'border': 'none', 'height': '1px', 'backgroundColor': 'rgba(0, 0, 0, 0.04)'}),
            html.H6([
                "Book Imbalance",
                _help_icon("Compares top bid vs ask size. Positive means buy-side depth is larger; negative means sell-side depth is larger.")
            ], className="text-muted mb-2"),
            html.Div(id='book-imbalance')
        ])
    ], className="mb-4 shadow-sm", style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})


def create_depth_chart():
    """
    Create order book depth visualization with table.
    
    Shows order book table (left) and cumulative depth chart (right).
    Updates at 0.5 Hz (2000ms interval).
    
    Returns:
        dbc.Card component with order book table and depth chart
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "Order Book & Depth",
            _help_icon("Shows visible liquidity by price level. Steeper depth means more size available to trade.")
        ], className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Left: Order Book Table
                dbc.Col([
                    html.H6([
                        "Order Book",
                        _help_icon("Top bid/ask levels with cumulative size. Watch how size stacks or disappears.")
                    ], className="text-center text-muted mb-3"),
                    
                    # Asks table (reversed, worst at top)
                    dash_table.DataTable(
                        id='table-asks',
                        columns=[
                            {'name': 'Cumulative', 'id': 'cumulative'},
                            {'name': 'Size', 'id': 'size'},
                            {'name': 'Ask', 'id': 'price'}
                        ],
                        data=[],
                        style_table={
                            'height': '200px',
                            'overflowY': 'auto'
                        },
                        style_header={
                            'backgroundColor': '#F8F9FA',
                            'color': '#495057',
                            'fontWeight': 'bold',
                            'textTransform': 'uppercase',
                            'fontSize': '11px',
                            'textAlign': 'right',
                            'padding': '8px',
                            'borderBottom': '2px solid #DEE2E6'
                        },
                        style_cell={
                            'textAlign': 'right',
                            'padding': '6px 10px',
                            'fontFamily': 'JetBrains Mono, Consolas, monospace',
                            'fontSize': '12px',
                            'backgroundColor': '#FFFFFF',
                            'color': '#212529',
                            'borderBottom': '1px solid #F1F3F5'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'price'},
                                'fontWeight': '600',
                                'color': '#DC3545'
                            }
                        ]
                    ),
                    
                    # Spread indicator
                    dbc.Alert(
                        id='spread-indicator',
                        color='warning',
                        className='text-center py-2 my-2 small'
                    ),
                    
                    # Bids table
                    dash_table.DataTable(
                        id='table-bids',
                        columns=[
                            {'name': 'Bid', 'id': 'price'},
                            {'name': 'Size', 'id': 'size'},
                            {'name': 'Cumulative', 'id': 'cumulative'}
                        ],
                        data=[],
                        style_table={
                            'height': '200px',
                            'overflowY': 'auto'
                        },
                        style_header={
                            'backgroundColor': '#F8F9FA',
                            'color': '#495057',
                            'fontWeight': 'bold',
                            'textTransform': 'uppercase',
                            'fontSize': '11px',
                            'textAlign': 'right',
                            'padding': '8px',
                            'borderBottom': '2px solid #DEE2E6'
                        },
                        style_cell={
                            'textAlign': 'right',
                            'padding': '6px 10px',
                            'fontFamily': 'JetBrains Mono, Consolas, monospace',
                            'fontSize': '12px',
                            'backgroundColor': '#FFFFFF',
                            'color': '#212529',
                            'borderBottom': '1px solid #F1F3F5'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'price'},
                                'fontWeight': '600',
                                'color': '#28A745'
                            }
                        ]
                    )
                ], width=12, md=6, className="mb-3 mb-md-0"),
                
                # Right: Depth Chart
                dbc.Col([
                    html.H6([
                        "Depth Chart",
                        _help_icon("Cumulative bid/ask size across prices. Larger one-sided depth can act as short-term support/resistance.")
                    ], className="text-center text-muted mb-3"),
                    dcc.Graph(
                        id='depth-chart',
                        config={'displayModeBar': False},
                        style={'height': '460px'}
                    )
                ], width=12, md=6)
            ])
        ])
    ], className="shadow-sm")


def create_ofi_chart():
    """
    Create OFI signal chart with streaming support.
    
    THIS IS THE KEY COMPONENT that eliminates flickering!
    
    Uses:
        - extendData for incremental updates (no chart recreation)
        - Dual-axis subplot (OFI bars + mid-price line)
        - 0.5 Hz update rate (2000ms interval)
    
    Returns:
        dbc.Card component with OFI chart
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "OFI Signal (Streaming)",
            _help_icon("Order Flow Imbalance over time. Positive values suggest buy pressure; negative values suggest sell pressure.")
        ], className="mb-0")),
        dbc.CardBody([
            dcc.Graph(
                id='ofi-chart',
                config={'displayModeBar': False},
                style={'height': '500px'}
            ),
            
            # Current OFI value
            html.Hr(className="my-3"),
            html.Div([
                html.H6([
                    "Current OFI",
                    _help_icon("Latest OFI reading. Magnitude shows strength; sign shows direction.")
                ], className="text-center text-muted mb-2"),
                html.Div(id='current-ofi', className="h3 text-center")
            ])
        ])
    ], className="mb-4 shadow-sm")


def create_execution_chart(trades=None):
    """
    Create Trade Execution Chart (Price + Trades).
    
    Displays mid-price evolution with markers for trade executions.
    - Green Triangle Up: Buy
    - Red Triangle Down: Sell
    
    Args:
        trades: List of trade dictionaries to initialize the chart with
    
    Returns:
        dbc.Card component
    """
    # Create initial figure
    fig = go.Figure()
    
    # Trace 0: Mid Price (Empty initially, or could pass price history too)
    # We rely on extendData for price updates, so start empty.
    fig.add_trace(go.Scatter(
        x=[], y=[],
        name='Mid Price',
        mode='lines',
        line=dict(color='#3498db', width=2),
        hovertemplate='Price: $%{y:,.2f}<extra></extra>'
    ))
    
    # Trace 1: Buys (Green Triangles)
    buys_x = [t['timestamp'] for t in trades if t['side'] == 'buy'] if trades else []
    buys_y = [t['price'] for t in trades if t['side'] == 'buy'] if trades else []
    
    fig.add_trace(go.Scatter(
        x=buys_x, y=buys_y,
        name='Buy',
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='#2ecc71', line=dict(width=1, color='black')),
        hovertemplate='<b>BUY</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
    ))
    
    # Trace 2: Sells (Red Triangles)
    sells_x = [t['timestamp'] for t in trades if t['side'] == 'sell'] if trades else []
    sells_y = [t['price'] for t in trades if t['side'] == 'sell'] if trades else []
    
    fig.add_trace(go.Scatter(
        x=sells_x, y=sells_y,
        name='Sell',
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='#e74c3c', line=dict(width=1, color='black')),
        hovertemplate='<b>SELL</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                "Live Execution Feed",
                _help_icon("Price line with simulated buys and sells. Trade markers help inspect entry/exit timing.")
            ], className="mb-0"),
            html.Small("Real-time price feed with simulated strategy trades", className="text-muted")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id='execution-chart',
                figure=fig,
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ], className="mb-4 shadow-sm")


def create_metrics_chart():
    """
    Create Historical Metrics Chart (Sharpe & Hit Rate).
    
    Displays rolling Sharpe Ratio and Hit Rate over time.
    Uses dual-axis chart:
    - Left Axis: Sharpe Ratio (Line)
    - Right Axis: Hit Rate (Line/Area)
    
    Returns:
        dbc.Card component with metrics chart
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "Strategy Performance History",
            _help_icon("Rolling history of key strategy quality metrics. Look for stable trends instead of single spikes.")
        ], className="mb-0")),
        dbc.CardBody([
            dcc.Graph(
                id='metrics-chart',
                config={'displayModeBar': False},
                style={'height': '350px'}
            )
        ])
    ], className="mb-4 shadow-sm")


def create_performance_metrics():
    """
    Create performance metrics panel.
    
    Displays key trading strategy metrics:
        - Sharpe Ratio: Risk-adjusted returns
        - Hit Rate: Directional accuracy
        - Max Drawdown: Largest decline
        - Win/Loss Ratio: Average win vs loss size
        - Price Correlation: OFI-price relationship
        - Latency: Message processing latency
        - Throughput: Messages per second
    
    Returns:
        dbc.Card component with performance metrics
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "Performance Metrics",
            _help_icon("Real-time strategy quality and system health indicators.")
        ], className="mb-0")),
        dbc.CardBody([
            # Row 1: Latency & Throughput
            dbc.Row([
                dbc.Col([
                    html.H6([
                        "Total Latency",
                        _help_icon("Total delay from exchange event to dashboard update. Lower is better.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-latency', className="h4 mb-0 font-monospace")
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        "Network vs System",
                        _help_icon("Latency split: network transit vs local processing. Helps identify bottlenecks.")
                    ], className="text-muted mb-2"),
                     html.Div(id='perf-latency-breakdown', className="mb-0 font-monospace")
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        "Throughput",
                        _help_icon("How many messages are processed per second. Higher is better if latency remains stable.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-throughput', className="h4 mb-0 font-monospace")
                ], width=4)
            ], className="mb-3"),
            
            html.Hr(className="my-3", style={'border': 'none', 'height': '1px', 'backgroundColor': 'rgba(0, 0, 0, 0.04)'}),
            
            # Row 2: Trading Performance
            dbc.Row([
                dbc.Col([
                    html.H6([
                        "Sharpe Ratio",
                        _help_icon("Risk-adjusted return. Higher is better; near zero means weak edge.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-sharpe', className="h4 mb-0 font-monospace")
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        "Hit Rate",
                        _help_icon("Percent of correct directional signals. Above 50% is generally favorable.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-hitrate', className="h4 mb-0 font-monospace")
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        "Max Drawdown",
                        _help_icon("Largest peak-to-trough decline. Smaller absolute drawdown is better.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-drawdown', className="h4 mb-0 font-monospace")
                ], width=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6([
                        "Win/Loss Ratio",
                        _help_icon("Average win size divided by average loss size. Above 1 means wins outweigh losses.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-winloss', className="h4 mb-0 font-monospace")
                ], width=6),
                
                dbc.Col([
                    html.H6([
                        "Price Correlation",
                        _help_icon("How strongly OFI and price changes move together. Positive values suggest alignment.")
                    ], className="text-muted mb-2"),
                    html.Div(id='perf-correlation', className="h4 mb-0 font-monospace")
                ], width=6)
            ])
        ])
    ], className="shadow-sm")


def create_sidebar():
    """
    Create sidebar with configuration and controls.
    
    Features:
        - Product selection dropdown
        - OFI window size slider
        - Start/Stop connection buttons
        - Info panel with instructions
    
    Returns:
        dbc.Card component for sidebar
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Configuration", className="mb-0")),
        dbc.CardBody([
            # Product selection
            html.Div([
                html.Label([
                    "Product ",
                    html.I(className="bi bi-question-circle tooltip-icon", 
                           title="Select trading pair to analyze")
                ], className="fw-bold"),
                dcc.Dropdown(
                    id='product-selector',
                    options=[
                        {'label': 'BTC-USD', 'value': 'BTC-USD'},
                        {'label': 'ETH-USD', 'value': 'ETH-USD'},
                        {'label': 'SOL-USD', 'value': 'SOL-USD'},
                        {'label': 'AVAX-USD', 'value': 'AVAX-USD'},
                    ],
                    value='BTC-USD',
                    clearable=False,
                    className="mb-3"
                )
            ]),
            
            # OFI window slider
            html.Div([
                html.Label([
                    "OFI Window Size ",
                    html.I(className="bi bi-question-circle tooltip-icon",
                           title="Number of events to aggregate for OFI calculation. Higher = smoother signals")
                ], className="fw-bold"),
                dcc.Slider(
                    id='ofi-window-slider',
                    min=10, max=500, step=10, value=100,
                    marks={10: '10', 100: '100', 500: '500'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className="mb-3"
                ),
                html.Div(id='ofi-window-display', className='text-center small text-muted mb-3')
            ]),

            # Signal Threshold slider
            html.Div([
                html.Label([
                    "Signal Threshold ",
                    html.I(className="bi bi-question-circle tooltip-icon",
                           title="Minimum absolute OFI value to trigger a trade signal")
                ], className="fw-bold"),
                dcc.Slider(
                    id='signal-threshold-slider',
                    min=0.1, max=20.0, step=0.1, value=5.0,
                    marks={1: '1', 5: '5', 10: '10', 20: '20'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className="mb-3"
                ),
                html.Div(id='signal-threshold-display', className='text-center small text-muted mb-3')
            ]),
            
            # Chart history slider
            html.Div([
                html.Label([
                    "Chart History ",
                    html.I(className="bi bi-question-circle tooltip-icon",
                           title="Number of data points to display in charts. Higher = longer history")
                ], className="fw-bold"),
                dcc.Slider(
                    id='chart-history-slider',
                    min=100, max=1000, step=100, value=500,
                    marks={100: '100', 500: '500', 1000: '1000'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className="mb-3"
                ),
                html.Div(id='chart-history-display', className='text-center small text-muted mb-3')
            ]),
            
            # Book depth slider
            html.Div([
                html.Label([
                    "Book Depth Levels ",
                    html.I(className="bi bi-question-circle tooltip-icon",
                           title="Number of price levels to show in order book. Higher = deeper market view")
                ], className="fw-bold"),
                dcc.Slider(
                    id='book-depth-slider',
                    min=5, max=50, step=5, value=10,
                    marks={5: '5', 10: '10', 25: '25', 50: '50'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className="mb-4"
                ),
                html.Div(id='book-depth-display', className='text-center small text-muted mb-3')
            ]),
            
            # Control buttons
            html.Label("Connection", className="fw-bold"),
            html.Div([
                dbc.Button(
                    "Start", 
                    id='btn-start', 
                    color="success", 
                    className="w-100 mb-2"
                ),
                dbc.Button(
                    "Stop", 
                    id='btn-stop', 
                    color="danger", 
                    className="w-100 mb-3"
                )
            ]),
            
            # Streaming controls
            html.Label("Streaming", className="fw-bold"),
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="me-1"), "Stream"],
                    id='btn-stream-play',
                    color='success',
                    outline=True,
                    size='sm'
                ),
                dbc.Button(
                    [html.I(className="me-1"), "Pause"],
                    id='btn-stream-pause',
                    color='warning',
                    outline=True,
                    size='sm'
                )
            ], className="w-100 mb-3"),
            
            html.Div(id='streaming-status', className='text-center small text-muted mb-3'),
            
            html.Hr(className="my-3"),
            
            # Info panel
            dbc.Alert([
                html.H6([
                    "How to use:"
                ], className="alert-heading mb-3"),
                html.Ul([
                    html.Li("Select product and adjust parameters above"),
                    html.Li("Click 'Start' to connect to live data"),
                    html.Li("Use Stream/Pause buttons to control streaming"),
                    html.Li("Monitor real-time OFI signals and metrics")
                ], className="mb-3"),
                html.Hr(className="my-2"),
                html.Div([
                    html.Strong([
                        "OFI Interpretation:"
                    ]),
                    html.Br(),
                    html.Div([
                        html.Strong("Positive", className="text-success"),
                        " = Buying pressure"
                    ], className="mb-1"),
                    html.Div([
                        html.Strong("Negative", className="text-danger"),
                        " = Selling pressure"
                    ], className="mb-1"),
                    html.Div([
                        html.Strong("Zero", className="text-muted"),
                        " = Balanced market"
                    ])
                ], className="mb-0")
            ], color="light")
        ])
    ], className="shadow-sm sticky-top", style={'top': '20px'})


def create_footer():
    """
    Create footer with streaming status and message count.
    
    Updates at 1 Hz (1000ms) for real-time feedback.
    
    Returns:
        html.Div component with footer content
    """
    return html.Div([
        html.Hr(className="mt-4 mb-3"),
        html.Div(
            id='footer-status', 
            className="text-center text-muted"
        )
    ])


def create_analyst_view():
    """
    Create Institutional Analyst View panel.
    
    Features:
        - Execution Quality (Slippage)
        - Signal Quality (Alpha Decay Chart)
        - Mathematical Basis (LaTeX)
        - Inventory Risk (Simulated)
    
    Returns:
        dbc.Card component
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Institutional Analyst View", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                # Left Column: Execution & Math
                dbc.Col([
                    html.H6("1. Execution Quality (Slippage 1.0 BTC)", className="text-primary mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id='analyst-slippage-buy', className="text-danger mb-1"),
                                    html.Small([
                                        "Buy Slippage (bps)",
                                        _help_icon("Extra cost versus mid-price when buying. Lower is better.")
                                    ], className="text-muted")
                                ])
                            ], className="text-center mb-3 bg-light border-0")
                        ]),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id='analyst-slippage-sell', className="text-success mb-1"),
                                    html.Small([
                                        "Sell Slippage (bps)",
                                        _help_icon("Price concession versus mid-price when selling. Lower is better.")
                                    ], className="text-muted")
                                ])
                            ], className="text-center mb-3 bg-light border-0")
                        ])
                    ]),
                    
                ], width=12, lg=5),
                
                # Right Column: Signal Quality & Inventory
                dbc.Col([
                    html.H6([
                        "2. Signal Quality (Alpha Decay)",
                        _help_icon("Shows how quickly signal edge fades after entry. Slower decay means signal lasts longer.")
                    ], className="text-primary mb-3"),
                    dcc.Graph(
                        id='alpha-decay-chart',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    ),
                    
                    html.Hr(),
                    
                    html.H6([
                        "4. Inventory Risk (Simulated)",
                        _help_icon("Open inventory exposure. Higher skew increases risk and can require stricter signals.")
                    ], className="text-primary mb-3"),
                    html.Div([
                        html.Div([
                            html.Span("Inventory Usage: ", className="fw-bold"),
                            html.Span("70% (Long Skew)", className="text-warning")
                        ], className="mb-2"),
                        dbc.Progress(value=70, color="warning", className="mb-2"),
                        html.Small("Warning: High inventory skew reduces signal threshold.", className="text-danger")
                    ])
                    
                ], width=12, lg=7)
            ])
        ])
    ], className="shadow-sm border-primary")


def create_scatter_chart():
    """
    Create OFI vs Price Change Scatter Plot.
    
    Diagnostics tool to verify signal predictive power.
    - X: OFI Value
    - Y: Future Price Change (bps)
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                "Signal Diagnostics: OFI vs Price Change",
                _help_icon("Each point compares OFI to future price move. A clearer upward slope suggests better predictive power.")
            ], className="mb-0"),
            html.Small("Correlation (Slope) indicates predictive power", className="text-muted")
        ]),
        dbc.CardBody([
            dcc.Graph(
                id='scatter-chart',
                config={'displayModeBar': False},
                style={'height': '300px'}
            )
        ])
    ], className="mb-4 shadow-sm")


def create_volatility_chart():
    """
    Create Rolling Volatility Chart.
    
    Diagnostics tool to explain Sharpe Ratio jitter.
    - Y: Annualized Volatility
    """
    return dbc.Card([
        dbc.CardHeader(html.H5([
            "Rolling Volatility (Risk)",
            _help_icon("Recent variability of returns. Higher volatility means higher risk and noisier PnL.")
        ], className="mb-0")),
        dbc.CardBody([
            dcc.Graph(
                id='volatility-chart',
                config={'displayModeBar': False},
                style={'height': '200px'}
            )
        ])
    ], className="mb-4 shadow-sm")


def create_layout(ws_port=5000, state=None):
    """
    Create complete dashboard layout.
    
    This assembles all components into a cohesive dashboard using
    Bootstrap's responsive grid system.
    
    Layout Structure:
        - Fixed header
        - 2-column layout (4:8 ratio)
            - Left: Sidebar (controls)
            - Right: Main content (Tabs)
    
    Args:
        ws_port: Port number for WebSocket connection
        
    Returns:
        dbc.Container with complete layout
    """
    return dbc.Container([
        # Header
        create_header(),
        
        # Main content
        dbc.Row([
            # Left column: Sidebar (controls and config)
            dbc.Col([
                create_sidebar()
            ], width=12, lg=3, className="mb-4"),
            
            # Right column: Dashboard content
            dbc.Col([
                dbc.Tabs([
                    # Tab 1: Overview
                    dbc.Tab([
                        html.Br(),
                        dbc.Row([
                            # Left: Main content (Order Book, OFI Chart)
                            dbc.Col([
                                # Order Book & Depth (wider)
                                dbc.Row([
                                    dbc.Col([
                                        create_depth_chart()
                                    ], width=12, className="mb-4")
                                ]),
                                
                                # OFI Signal (wider, below order book)
                                dbc.Row([
                                    dbc.Col([
                                        create_ofi_chart()
                                    ], width=12, className="mb-4")
                                ]),

                                # Execution Chart (Live Feed)
                                dbc.Row([
                                    dbc.Col([
                                        create_execution_chart(trades=list(state.recent_trades) if state else None)
                                    ], width=12, className="mb-4")
                                ]),

                                # Metrics History (wider, below OFI chart)
                                dbc.Row([
                                    dbc.Col([
                                        create_metrics_chart()
                                    ], width=12, className="mb-4")
                                ]),
                                
                                # Performance Metrics (full width below)
                                dbc.Row([
                                    dbc.Col([
                                        create_performance_metrics()
                                    ], width=12)
                                ])
                            ], width=12, lg=9),
                            
                            # Right: Market Metrics (narrow sidebar)
                            dbc.Col([
                                create_metrics_panel()
                            ], width=12, lg=3, className="mb-4")
                        ])
                    ], label="Overview", tab_id="tab-overview"),
                    
                    # Tab 2: Analyst View
                    dbc.Tab([
                        html.Br(),
                        create_analyst_view(),
                        
                        # Diagnostics
                        dbc.Row([
                            dbc.Col([
                                create_scatter_chart()
                            ], width=12, md=6),
                            dbc.Col([
                                create_volatility_chart()
                            ], width=12, md=6)
                        ])
                    ], label="Analyst View", tab_id="tab-analyst", label_style={"fontWeight": "bold", "color": "#2c3e50"}),
                    
                ], id="tabs", active_tab="tab-overview")
                
            ], width=12, lg=9)
        ]),
        
        # Footer
        create_footer(),
        
        # WebSocket component for real-time updates
        WebSocket(id="ws", url=f"ws://127.0.0.1:{ws_port}"),
        
        # Store component for initial data
        dcc.Store(id='chart-initialized', data=True)
        
    ], fluid=True, className="p-4", style={'backgroundColor': '#FFFFFF'})
