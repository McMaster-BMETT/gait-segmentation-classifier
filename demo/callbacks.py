"""Dash callbacks — stage selector drives all chart/card updates."""

from dash import Input, Output, State, ALL, callback_context, no_update
from dash.exceptions import PreventUpdate

from .constants import STAGE_ORDER, METRIC_DEFS, METRIC_ORDER
from .charts import build_signal_figure, build_radar_chart


def register_callbacks(app, demo_data):
    """Register all Dash callbacks.

    Args:
        app: the Dash app instance
        demo_data: loaded demo_data.json dict
    """
    phase_names = demo_data.get("phase_names", [])

    @app.callback(
        Output("selected-stage", "data"),
        Output({"type": "stage-btn", "index": ALL}, "className"),
        Input({"type": "stage-btn", "index": ALL}, "n_clicks"),
        State("selected-stage", "data"),
    )
    def update_selected_stage(n_clicks_list, current_stage):
        ctx = callback_context
        if not ctx.triggered or all(n == 0 for n in n_clicks_list):
            # Initial load — set first stage active
            classes = ["stage-btn active"] + ["stage-btn"] * (len(STAGE_ORDER) - 1)
            return STAGE_ORDER[0], classes

        triggered_id = ctx.triggered[0]["prop_id"]
        # Extract stage key from pattern-matched id
        for s in STAGE_ORDER:
            if s in triggered_id:
                selected = s
                break
        else:
            raise PreventUpdate

        classes = [
            "stage-btn active" if s == selected else "stage-btn"
            for s in STAGE_ORDER
        ]
        return selected, classes

    @app.callback(
        Output("signal-chart", "figure"),
        Output("radar-chart", "figure"),
        Output({"type": "metric-value", "index": ALL}, "children"),
        Output({"type": "metric-dot", "index": ALL}, "className"),
        Input("selected-stage", "data"),
    )
    def update_dashboard(selected_stage):
        stage_data = demo_data["stages"].get(selected_stage)
        if not stage_data:
            raise PreventUpdate

        # Signal figure
        signal_fig = build_signal_figure(stage_data, phase_names)

        # Radar chart
        radar_fig = build_radar_chart(stage_data["metrics"], selected_stage)

        # Metric card values and status dots
        values = []
        dot_classes = []
        for key in METRIC_ORDER:
            mdef = METRIC_DEFS[key]
            val = stage_data["metrics"].get(key)
            if val is None:
                values.append("—")
                dot_classes.append("metric-dot")
                continue

            fmt = mdef["format"]
            unit = mdef["unit"]
            values.append(f"{val:{fmt}}{unit}")

            # Determine status color
            status = _metric_status(val, mdef)
            dot_classes.append(f"metric-dot {status}")

        return signal_fig, radar_fig, values, dot_classes


def _metric_status(val, mdef):
    """Return 'green', 'amber', or 'red' based on metric thresholds."""
    lo, hi = mdef["good_range"]
    if lo is not None and hi is not None:
        if lo <= val <= hi:
            return "green"
    elif lo is not None:
        if val >= lo:
            return "green"
    elif hi is not None:
        if val <= hi:
            return "green"

    wlo, whi = mdef["warn_range"]
    if wlo is not None and whi is not None:
        if wlo <= val <= whi:
            return "amber"
    elif wlo is not None:
        if val >= wlo:
            return "amber"
    elif whi is not None:
        if val <= whi:
            return "amber"

    return "red"
