import re
with open('src/app/dash_callbacks.py', 'r') as f:
    text = f.read()

new_text = text.replace("""        if state.avg_latency_ms < 5:
            latency_text = html.Span(latency_text, className="text-success")
        elif state.avg_latency_ms < 20:
            latency_text = html.Span(latency_text, className="text-info")
        else:
            latency_text = html.Span(latency_text, className="text-warning")""", "")

new_text = new_text.replace("""        breakdown_text = html.Div([
            html.Span(f"Net: {net_lat:.0f}ms", className=f"me-2 {net_color}"),
            html.Span(f"|", className="text-muted me-2"),
            html.Span(f"Sys: {sys_lat:.1f}ms", className=sys_color)
        ])""", """        breakdown_text = f"Net: {net_lat:.0f}ms | Sys: {sys_lat:.1f}ms" """)

new_text = new_text.replace("""        if state.messages_per_second > 50:
            throughput_text = html.Span(throughput_text, className="text-success")
        elif state.messages_per_second > 20:
            throughput_text = html.Span(throughput_text, className="text-info")
        else:
            throughput_text = html.Span(throughput_text, className="text-muted")""", "")

new_text = new_text.replace("""        if state.sharpe_ratio > 2.0:
            sharpe_text = html.Span(sharpe_text, className="text-success fw-bold")
        elif state.sharpe_ratio > 1.0:
            sharpe_text = html.Span(sharpe_text, className="text-info")
        elif state.sharpe_ratio > 0:
            sharpe_text = html.Span(sharpe_text, className="text-warning")
        else:
            sharpe_text = html.Span(sharpe_text, className="text-danger")""", "")

new_text = new_text.replace("""        if state.hit_rate > 60:
            hitrate_text = html.Span(hitrate_text, className="text-success fw-bold")
        elif state.hit_rate > 55:
            hitrate_text = html.Span(hitrate_text, className="text-info")
        elif state.hit_rate > 50:
            hitrate_text = html.Span(hitrate_text, className="text-warning")
        else:
            hitrate_text = html.Span(hitrate_text, className="text-danger")""", "")

new_text = new_text.replace("""        if abs(state.max_drawdown) < 5:
            drawdown_text = html.Span(drawdown_text, className="text-success")
        elif abs(state.max_drawdown) < 10:
            drawdown_text = html.Span(drawdown_text, className="text-warning")
        else:
            drawdown_text = html.Span(drawdown_text, className="text-danger")""", "")

new_text = new_text.replace("""        if state.win_loss_ratio > 1.5:
            winloss_text = html.Span(winloss_text, className="text-success fw-bold")
        elif state.win_loss_ratio > 1.0:
            winloss_text = html.Span(winloss_text, className="text-info")
        else:
            winloss_text = html.Span(winloss_text, className="text-warning")""", "")

new_text = new_text.replace("""        if abs(state.price_correlation) > 0.7:
            correlation_text = html.Span(correlation_text, className="text-success fw-bold")
        elif abs(state.price_correlation) > 0.3:
            correlation_text = html.Span(correlation_text, className="text-info")
        else:
            correlation_text = html.Span(correlation_text, className="text-warning")""", "")

with open('src/app/dash_callbacks.py', 'w') as f:
    f.write(new_text)

