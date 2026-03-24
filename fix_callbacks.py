import re
with open('src/app/dash_callbacks.py', 'r') as f:
    text = f.read()

# Replace _get_connection_status_component to just return strings
new_text = text.replace("""def _get_connection_status_component():
    \"\"\"Generate the connection status badge component.\"\"\"
    if state.connected and state.book_initialized:
        return html.Span("Connected", className="badge bg-success")
    elif state.connected:
        return html.Span("Initializing", className="badge bg-warning text-dark")
    else:
        return html.Span("Disconnected", className="badge bg-danger")""",
"""def _get_connection_status_component():
    if getattr(state, 'connected', False) and getattr(state, 'book_initialized', False):
        return "Connected"
    elif getattr(state, 'connected', False):
        return "Initializing"
    else:
        return "Disconnected" """)

# Let's fix _get_footer_component
new_text = new_text.replace("""def _get_footer_component():
    \"\"\"Generate the footer status component.\"\"\"
    heartbeat_str = ""
    if state.last_heartbeat:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(state.last_heartbeat.replace('Z', '+00:00'))
            heartbeat_str = f" | Heartbeat: {dt.strftime('%H:%M:%S')}"
        except Exception:
            # Fallback if parsing fails
            heartbeat_str = f" | Heartbeat: {state.last_heartbeat[:19]}"
    
    return html.Small([
        f"Messages: {state.message_count:,}",
        heartbeat_str,
        f" | {'Streaming' if state.connected else 'Offline'}"
    ])""",
"""def _get_footer_component():
    heartbeat_str = ""
    if state.last_heartbeat:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(state.last_heartbeat.replace('Z', '+00:00'))
            heartbeat_str = f" | Heartbeat: {dt.strftime('%H:%M:%S')}"
        except Exception:
            heartbeat_str = f" | Heartbeat: {state.last_heartbeat[:19]}"
    
    return f"Messages: {state.message_count:,}{heartbeat_str} | {'Streaming' if getattr(state, 'connected', False) else 'Offline'}"
""")

with open('src/app/dash_callbacks.py', 'w') as f:
    f.write(new_text)

