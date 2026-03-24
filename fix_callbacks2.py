import re
with open('src/app/dash_callbacks.py', 'r') as f:
    text = f.read()

# Replace _get_market_metrics_components
new_text = text.replace("""def _get_market_metrics_components():
    \"\"\"
    Generate all market metrics components.
    
    Returns:
        Tuple of (bid_text, ask_text, spread_text, micro_text, imbalance_component)
    \"\"\"
    if not state.book_initialized or not state.best_bid or not state.best_ask:
        return "-", "-", "-", "-", html.Small("Waiting for data...", className="text-muted")
    
    # Format metrics
    bid_price, bid_size = state.best_bid
    ask_price, ask_size = state.best_ask
    
    bid_text = f"${bid_price:,.2f}"
    ask_text = f"${ask_price:,.2f}"
    
    # Spread with BPS
    if state.spread and state.mid_price:
        spread_bps = (state.spread / state.mid_price) * 10000
        spread_text = [
            f"${state.spread:.2f} ",
            html.Small(f"({spread_bps:.1f} bps)", className="text-muted ms-1")
        ]
    elif state.spread:
        spread_text = f"${state.spread:.2f}"
    else:
        spread_text = "-"
    
    micro_text = f"${state.micro_price:,.2f}" if state.micro_price else "-"
    
    # Book imbalance progress bar
    total = bid_size + ask_size
    imbalance = ((bid_size - ask_size) / total * 100) if total > 0 else 0
    
    progress_value = (imbalance + 100) / 2
    color = "success" if imbalance > 0 else "danger"
    label_text = 'BUY' if imbalance > 0 else 'SELL' if imbalance < 0 else 'BALANCED'
    
    imbalance_component = html.Div([
        html.Small(
            f"Book Imbalance: {imbalance:+.1f}% ({label_text} pressure)",
            className="text-muted"
        ),
        dbc.Progress(
            value=progress_value,
            color=color,
            className="mt-2",
            style={'height': '10px'}
        )
    ])
    
    return bid_text, ask_text, spread_text, micro_text, imbalance_component""", 
"""def _get_market_metrics_components():
    if not getattr(state, 'book_initialized', False) or getattr(state, 'best_bid', None) is None or getattr(state, 'best_ask', None) is None:
        return "-", "-", "-", "-", "Waiting for data..."
    
    bid_price, bid_size = state.best_bid
    ask_price, ask_size = state.best_ask
    
    bid_text = f"${bid_price:,.2f}"
    ask_text = f"${ask_price:,.2f}"
    
    if state.spread and state.mid_price:
        spread_bps = (state.spread / state.mid_price) * 10000
        spread_text = f"${state.spread:.2f} ({spread_bps:.1f} bps)"
    elif state.spread:
        spread_text = f"${state.spread:.2f}"
    else:
        spread_text = "-"
    
    micro_text = f"${state.micro_price:,.2f}" if getattr(state, 'micro_price', None) else "-"
    
    total = bid_size + ask_size
    imbalance = ((bid_size - ask_size) / total * 100) if total > 0 else 0
    label_text = 'BUY' if imbalance > 0 else 'SELL' if imbalance < 0 else 'BALANCED'
    imbalance_component = f"Book Imbalance: {imbalance:+.1f}% ({label_text} pressure)"
    
    return bid_text, ask_text, spread_text, micro_text, imbalance_component""")

with open('src/app/dash_callbacks.py', 'w') as f:
    f.write(new_text)

