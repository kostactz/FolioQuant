if (!window.dash_clientside) {
    window.dash_clientside = {};
}
window.__folioquant_mount_time = Date.now();

window.__folioquant_stream_state = window.__folioquant_stream_state || {
    lastOfiTs: null,
    lastMetricsTs: null,
    lastExecTs: null,
    firstMessageLogged: false
};

window.dash_clientside.ws_clientside = {

    update_connection_status: function (msg) {
        if (!msg || !msg.data) return Array(2).fill(window.dash_clientside.no_update);
        
        if (!window.__folioquant_stream_state.firstMessageLogged) {
            console.log("🟢 First WebSocket message received successfully!", msg.data.substring(0, 100) + "...");
            window.__folioquant_stream_state.firstMessageLogged = true;
        }

        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                return data.fast.slice(1, 3); // current-price, footer-status
            }
        } catch (e) {
            console.error("Error parsing metrics_update:", e);
        }
        return Array(2).fill(window.dash_clientside.no_update);
    },

    monitor_ws_state: function (ws_state, ws_error) {
        // ws_state corresponds to the state of the websocket component
        // readyState: 0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED
        if (ws_error) {
            console.error("WebSocket Error:", ws_error);
        }
        
        if (!ws_state) {
            return "WS Status: Unknown";
        }

        console.log("WebSocket State Change:", ws_state);
        
        // Debugging information for closed websockets
        if (ws_state.readyState === 3 || ws_state.readyState === 2) {
            console.warn(`WebSocket is closing or closed. readyState: ${ws_state.readyState}`);
            if (ws_state.code) console.warn(`Close code: ${ws_state.code} (1000=Normal, 1001=Going Away, 1006=Abnormal)`);
            if (ws_state.reason) console.warn(`Close reason: "${ws_state.reason}"`);
            if (ws_state.wasClean !== undefined) console.warn(`Was clean closure: ${ws_state.wasClean}`);
        }

        switch (ws_state.readyState) {
            case 0:
                return "WS: Connecting...";
            case 1:
                return "🟢 WS: Connected";
            case 2:
                return "WS: Closing...";
            case 3:
                return "🔴 WS: Closed";
            default:
                return "WS: Unknown (" + ws_state.readyState + ")";
        }
    },

    update_market_metrics: function (msg) {
        if (!msg || !msg.data) return Array(5).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                return data.fast.slice(3, 8);
            }
        } catch (e) { console.error(e); }
        return Array(5).fill(window.dash_clientside.no_update);
    },

    update_ofi_display: function (msg) {
        if (!msg || !msg.data) return window.dash_clientside.no_update;
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                return data.fast[8];
            }
        } catch (e) { console.error(e); }
        return window.dash_clientside.no_update;
    },

    update_error_alert: function (msg) {
        if (!msg || !msg.data) return Array(2).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                return data.fast.slice(9, 11);
            }
        } catch (e) { console.error(e); }
        return Array(2).fill(window.dash_clientside.no_update);
    },

    update_perf_metrics: function (msg) {
        if (!msg || !msg.data) return Array(8).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update" && data.slow && Array.isArray(data.slow)) {
                return data.slow.slice(0, 8);
            }
        } catch (e) { console.error(e); }
        return Array(8).fill(window.dash_clientside.no_update);
    },

    update_depth_chart: function (msg) {
        if (Date.now() - window.__folioquant_mount_time < 1500) return window.dash_clientside.no_update;
        if (!msg || !msg.data) return window.dash_clientside.no_update;
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update" && data.slow && Array.isArray(data.slow)) {
                return data.slow[8];
            }
        } catch (e) { console.error(e); }
        return window.dash_clientside.no_update;
    },

    update_order_book: function (msg) {
        if (!msg || !msg.data) return Array(3).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update" && data.slow && Array.isArray(data.slow)) {
                return data.slow.slice(9, 12);
            }
        } catch (e) { console.error(e); }
        return Array(3).fill(window.dash_clientside.no_update);
    },

    update_analyst_metrics: function (msg) {
        if (Date.now() - window.__folioquant_mount_time < 1500) return Array(5).fill(window.dash_clientside.no_update);
        if (!msg || !msg.data) return Array(5).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update" && data.slow && Array.isArray(data.slow)) {
                return data.slow.slice(12, 17);
            }
        } catch (e) { console.error(e); }
        return Array(5).fill(window.dash_clientside.no_update);
    },

    update_ofi_chart: function (msg) {
        if (Date.now() - window.__folioquant_mount_time < 1500) return window.dash_clientside.no_update;
        if (!msg || !msg.data) return window.dash_clientside.no_update;
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                const payload = data.ofi;
                if (!payload || !Array.isArray(payload) || payload.length < 1) {
                    return window.dash_clientside.no_update;
                }
                const dataDict = payload[0] || {};
                const ts = dataDict.x && dataDict.x[0] && dataDict.x[0][0] ? dataDict.x[0][0] : null;
                if (!ts || ts === window.__folioquant_stream_state.lastOfiTs) {
                    return window.dash_clientside.no_update;
                }
                window.__folioquant_stream_state.lastOfiTs = ts;
                return payload;
            }
        } catch (e) { console.error(e); }
        return window.dash_clientside.no_update;
    },

    update_metrics_chart: function (msg) {
        if (Date.now() - window.__folioquant_mount_time < 1500) return window.dash_clientside.no_update;
        if (!msg || !msg.data) return window.dash_clientside.no_update;
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                const payload = data.metrics;
                if (!payload || !Array.isArray(payload) || payload.length < 1) {
                    return window.dash_clientside.no_update;
                }
                const dataDict = payload[0] || {};
                const ts = dataDict.x && dataDict.x[0] && dataDict.x[0][0] ? dataDict.x[0][0] : null;
                if (!ts || ts === window.__folioquant_stream_state.lastMetricsTs) {
                    return window.dash_clientside.no_update;
                }
                window.__folioquant_stream_state.lastMetricsTs = ts;
                return payload;
            }
        } catch (e) { console.error(e); }
        return window.dash_clientside.no_update;
    },

    update_execution_chart: function (msg) {
        if (Date.now() - window.__folioquant_mount_time < 1500) return window.dash_clientside.no_update;
        if (!msg || !msg.data) return window.dash_clientside.no_update;
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                const payload = data.trades;
                if (!payload || !Array.isArray(payload) || payload.length < 1) {
                    return window.dash_clientside.no_update;
                }
                const dataDict = payload[0] || {};
                const ts = dataDict.x && dataDict.x[0] && dataDict.x[0][0] ? dataDict.x[0][0] : null;
                if (!ts || ts === window.__folioquant_stream_state.lastExecTs) {
                    return window.dash_clientside.no_update;
                }
                window.__folioquant_stream_state.lastExecTs = ts;
                return payload;
            }
        } catch (e) { console.error(e); }
        return window.dash_clientside.no_update;
    }
};
