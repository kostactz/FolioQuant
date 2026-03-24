if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.__folioquant_stream_state = window.__folioquant_stream_state || {
    lastOfiTs: null,
    lastMetricsTs: null,
    lastExecTs: null,
};

window.dash_clientside.ws_clientside = {

    update_connection_status: function (msg) {
        if (!msg || !msg.data) return Array(3).fill(window.dash_clientside.no_update);
        try {
            const data = JSON.parse(msg.data);
            if (data.type === "metrics_update") {
                return data.fast.slice(0, 3);
            }
        } catch (e) { console.error(e); }
        return Array(3).fill(window.dash_clientside.no_update);
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
