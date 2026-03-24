import re
with open('src/app/dash_callbacks.py', 'r') as f:
    text = f.read()

new_text = text.replace("""    # Spread indicator
    if state.spread and state.mid_price:
        spread_bps = (state.spread / state.mid_price) * 10000
        spread_text = html.Div([
            html.I(className="bi bi-arrow-left-right me-2"),
            html.Span(f"Spread: ${state.spread:.2f} "),
            html.Span(f"({spread_bps:.1f} bps)", className="small")
        ], className="spread-indicator")
    else:
        spread_text = html.Div("—", className="spread-indicator")""",
"""    # Spread indicator
    if state.spread and state.mid_price:
        spread_bps = (state.spread / state.mid_price) * 10000
        spread_text = f"Spread: ${state.spread:.2f} ({spread_bps:.1f} bps)"
    else:
        spread_text = "—" """)

with open('src/app/dash_callbacks.py', 'w') as f:
    f.write(new_text)
