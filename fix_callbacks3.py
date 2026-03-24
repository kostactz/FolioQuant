import re
with open('src/app/dash_callbacks.py', 'r') as f:
    text = f.read()

# Replace _get_ofi_component
new_text = text.replace("""def _get_ofi_component():
    \"\"\"Generate the OFI display component.\"\"\"
    if not state.ofi_history or len(state.ofi_history) == 0:
        return "—"
    
    latest_ofi = state.ofi_history[-1]['ofi']
    
    if latest_ofi > 0:
        direction = "BUY"
        color_class = "text-success"
    elif latest_ofi < 0:
        direction = "SELL"
        color_class = "text-danger"
    else:
        direction = "NEUTRAL"
        color_class = "text-muted"
    
    return html.Div([
        html.Span(f"{latest_ofi:+,.2f}", className=f"{color_class} me-2"),
        html.Small(direction, className="text-muted")
    ])""", 
"""def _get_ofi_component():
    if not getattr(state, 'ofi_history', None) or len(state.ofi_history) == 0:
        return "—"
    
    latest_ofi = state.ofi_history[-1]['ofi']
    direction = "BUY" if latest_ofi > 0 else "SELL" if latest_ofi < 0 else "NEUTRAL"
    return f"{latest_ofi:+,.2f} {direction}" """)

# Replace _get_error_component 
new_text = new_text.replace("""def _get_error_component():
    \"\"\"
    Generate the error alert component.
    
    Returns:
        Tuple of (error_text, error_open_boolean)
    \"\"\"
    if state.error_message:
        error_text = html.Div([
            html.Strong("Error: ", className="me-2"),
            html.Span(state.error_message)
        ])
        return error_text, True
    
    return "", False""", 
"""def _get_error_component():
    if getattr(state, 'error_message', None):
        return f"Error: {state.error_message}", True
    return "", False""")

with open('src/app/dash_callbacks.py', 'w') as f:
    f.write(new_text)
